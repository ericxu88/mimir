"""Tests for mimir.stats — paired bootstrap CI, sign-flip p-value, power (DESIGN.md §7).

Pins the two brief proofs: identical synthetic distributions => CI contains 0 and
p > 0.05; shifted distributions => CI excludes 0, brackets the true shift, p < 0.05.
Those assertions are probabilistic in general but deterministic here: all data seeds
and resampling seeds are pinned literals, pre-screened so every assertion holds with
wide margin. Never change a seed to make a test pass — that inverts the oracle.
"""

import re
import sqlite3
import zlib

import numpy as np
import pytest

from mimir.clients.base import CompletionResponse
from mimir.clients.mock import MockClient
from mimir.runner import run_experiment
from mimir.spec import ExperimentSpec
from mimir.stats import (
    DEFAULT_CORRECTION,
    ReplicateTable,
    analyze_run,
    benjamini_hochberg,
    bootstrap_ci,
    decompose_variance,
    holm_bonferroni,
    pairwise_item_scores,
    pairwise_replicate_scores,
    replicate_diffs,
    required_items_for_power,
    rubric_item_scores,
    rubric_replicate_scores,
    score_variance_shares,
    sign_flip_pvalue,
    studentized_ci,
)
from mimir.store import Store


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "mimir.db")
    yield s
    s.close()


def make_run(store, *, mode="pairwise", variants=("control", "treatment")):
    """Run + conditions in declared order; spec dict carries the judge block analyze_run reads."""
    spec = {"name": "greeting-tone", "judge": {"model": "judge-model", "mode": mode}}
    run_id = store.create_run("greeting-tone", spec)
    condition_ids = {
        name: store.add_condition(
            run_id,
            variant_name=name,
            system_prompt="",
            user_template="Answer: {input}",
            sampling={"model": "claude-haiku-4-5-20251001"},
        )
        for name in variants
    }
    return run_id, condition_ids


def add_ok_sample(store, run_id, condition_id, item_id, sample_index=0):
    return store.add_sample(
        run_id=run_id,
        condition_id=condition_id,
        item_id=item_id,
        sample_index=sample_index,
        cache_key="k" * 64,
        request_json="{}",
        raw_response="{}",
        response_text=f"response for {item_id}",
        latency_ms=1.0,
        input_tokens=1,
        output_tokens=1,
    )


def add_pair_judgment(
    store, run_id, item_id, *, order, verdict, control_sid, treatment_sid, error=None
):
    # Encodes the presentation rule ONCE: 'ab' presents the declared order
    # (control in position A), 'ba' presents the swap (runner.py pairwise_task).
    a_sid, b_sid = (control_sid, treatment_sid) if order == "ab" else (treatment_sid, control_sid)
    store.add_judgment(
        run_id=run_id,
        item_id=item_id,
        judge_model="judge-model",
        mode="pairwise",
        sample_a_id=a_sid,
        sample_b_id=b_sid,
        position_order=order,
        cache_key="j" * 64,
        verdict=None if error else verdict,
        error=error,
    )


def add_rubric_judgment(store, run_id, item_id, *, sample_id, score, error=None):
    store.add_judgment(
        run_id=run_id,
        item_id=item_id,
        judge_model="judge-model",
        mode="rubric",
        sample_a_id=sample_id,
        cache_key="j" * 64,
        score=None if error else float(score),
        error=error,
    )


def tables(store, run_id):
    return store.get_judgments(run_id), store.get_samples(run_id), store.get_conditions(run_id)


def insert_legacy_judgment(
    db_path,
    run_id,
    item_id,
    *,
    mode,
    sample_a_id,
    sample_b_id=None,
    position_order=None,
    verdict=None,
    score=None,
):
    """Simulate legacy/drifted data: a judgment referencing another run's sample.

    New databases enforce the run-scoped judgment FK (M9), so the row goes in
    through a separate raw connection (sqlite3 foreign_keys defaults OFF per
    connection) — exactly how such rows exist in the wild: written under the
    pre-M9 schema. The skip-and-count guards under test exist for those files.
    """
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "INSERT INTO judgments (run_id, item_id, judge_model, mode, sample_a_id,"
            " sample_b_id, position_order, cache_key, verdict, score, created_at)"
            " VALUES (?, ?, 'judge-model', ?, ?, ?, ?, ?, ?, ?,"
            " '2026-01-01T00:00:00+00:00')",
            (
                run_id,
                item_id,
                mode,
                sample_a_id,
                sample_b_id,
                position_order,
                "j" * 64,
                verdict,
                score,
            ),
        )
        conn.commit()
    finally:
        conn.close()


# --- pure math: bootstrap CI ------------------------------------------------------


def test_bootstrap_ci_same_seed_bit_identical():
    d = np.random.default_rng(5).normal(0.3, 1, 200)
    ci = bootstrap_ci(d, seed=7)
    assert bootstrap_ci(d, seed=7) == ci
    assert bootstrap_ci(list(d), seed=7) == ci  # extraction hands plain sequences to the math layer


def test_bootstrap_ci_different_seed_differs():
    d = np.random.default_rng(5).normal(0.3, 1, 200)
    assert bootstrap_ci(d, seed=7) != bootstrap_ci(d, seed=8)


def test_bootstrap_ci_constant_diffs_collapses_to_point():
    # Every resample of a constant vector has mean exactly 2.5 (binary-exact value).
    assert bootstrap_ci([2.5, 2.5, 2.5, 2.5], seed=0) == (2.5, 2.5)


def test_bootstrap_ci_single_diff_is_point():
    # n=1 is degenerate but defined: every resample is the single item.
    assert bootstrap_ci([3.0], seed=0) == (3.0, 3.0)


def test_bootstrap_ci_contains_and_centers_on_sample_mean():
    d = np.random.default_rng(5).normal(0.3, 1, 200)  # realized mean 0.23507
    lo, hi = bootstrap_ci(d, seed=7)
    assert lo < d.mean() < hi
    # Verified off-center/width ratio is 0.0012 — the 0.1 bound is a 80x margin,
    # robust to any correct percentile bootstrap.
    assert abs((lo + hi) / 2 - d.mean()) < 0.1 * (hi - lo)


def test_bootstrap_ci_width_shrinks_with_more_items():
    lo_small, hi_small = bootstrap_ci(np.random.default_rng(1).normal(0.3, 1, 50), seed=0)
    lo_large, hi_large = bootstrap_ci(np.random.default_rng(2).normal(0.3, 1, 500), seed=0)
    # Verified widths: 0.483 (n=50) vs 0.176 (n=500) — a 2.7x gap no implementation closes.
    assert hi_large - lo_large < hi_small - lo_small


def test_bootstrap_ci_tail_mass_pins_95_percent_level():
    # Pins the CONFIDENCE LEVEL itself (review finding: every other CI assertion is
    # level-invariant). Independently re-estimate the bootstrap distribution with a
    # fresh seeded resampler and check ~2.5% of resample means fall outside each CI
    # bound. Verified: tails are 0.0243/0.0245 here; a 90% CI would give ~0.05 and a
    # 99% CI ~0.003 — both outside the asserted band. Any correct 95% percentile
    # bootstrap passes; any other level fails.
    d = np.random.default_rng(5).normal(0.3, 1, 200)
    lo, hi = bootstrap_ci(d, seed=7)
    fresh_rng = np.random.default_rng(123)
    fresh_means = d[fresh_rng.integers(0, d.size, size=(4000, d.size))].mean(axis=1)
    assert 0.015 < np.mean(fresh_means < lo) < 0.035
    assert 0.015 < np.mean(fresh_means > hi) < 0.035


# --- pure math: sign-flip permutation p-value -------------------------------------


def test_sign_flip_exhaustive_n3_hand_enumerated():
    # d = [1, 2, 3], observed sum 6. All 8 sign patterns:
    #   +++ -> 6   ++- -> 0   +-+ -> 2   +-- -> -4
    #   -++ -> 4   -+- -> -2  --+ -> 0   --- -> -6
    # |sum| >= 6 for exactly {+++, ---} -> p = 2/8. Also pins >= (not >): strict
    # comparison would exclude the identity pattern and give p = 0.0.
    assert sign_flip_pvalue([1.0, 2.0, 3.0]) == 0.25


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        pytest.param(2, 0.5, id="n2"),
        pytest.param(3, 0.25, id="n3"),
        pytest.param(4, 0.125, id="n4"),
    ],
)
def test_sign_flip_exhaustive_constant_diffs(n, expected):
    # Only the two all-same-sign patterns tie the observed |sum| -> p = 2/2^n exactly.
    assert sign_flip_pvalue([1.0] * n) == expected


def test_sign_flip_all_zero_diffs_p_is_one():
    # Every pattern's sum is 0 == observed -> all count.
    assert sign_flip_pvalue([0.0] * 4) == 1.0


def test_sign_flip_single_diff_p_is_one():
    # Patterns {+7, -7} both tie on |sum| -> n=1 can never be significant.
    assert sign_flip_pvalue([7.0]) == 1.0


def test_sign_flip_mc_invariant_under_global_negation():
    d = np.random.default_rng(9).normal(0.2, 1, 50)  # n=50 -> Monte Carlo path
    assert sign_flip_pvalue(d, seed=0) == sign_flip_pvalue(-d, seed=0)


def test_sign_flip_mc_same_seed_reproducible():
    d = np.random.default_rng(9).normal(0.2, 1, 50)
    assert sign_flip_pvalue(d, seed=3) == sign_flip_pvalue(d, seed=3)


def test_sign_flip_mc_strong_shift_hits_floor_never_zero():
    d = np.random.default_rng(3).normal(1.0, 0.05, 50)
    p = sign_flip_pvalue(d, seed=0)
    # Verified p = 1/10001. The `0 <` half mandates add-one smoothing (an MC p of
    # exactly 0 is indefensible); the upper bound doesn't overfit the formula.
    assert 0 < p <= 0.001


def test_sign_flip_exhaustive_at_boundary_n16():
    # Pins the exhaustive side of the n <= 16 regime boundary (review finding: no
    # test between n=4 and n=50). Constant diffs: only the two all-same-sign
    # patterns tie -> exactly 2/2^16, only reachable via exhaustive enumeration.
    assert sign_flip_pvalue([1.0] * 16) == 2 / 2**16


def test_sign_flip_mc_just_past_boundary_n17():
    # n=17 must take the Monte Carlo path: verified with seed 0 that exactly one of
    # the 10,000 draws is all-same-sign, so p = (1+1)/10001 — a value the exhaustive
    # path (2/2^17) cannot produce.
    assert sign_flip_pvalue([1.0] * 17, seed=0) == 2 / 10001


# --- pure math: power / items needed ----------------------------------------------


def test_power_hand_computed_required_items():
    # d = [-1, 1, 3]: mean 1.0, sd(ddof=1) = 2.0.
    # required = ceil(((1.9599640 + 0.8416212) * 2 / 1)^2) = ceil(5.6031704^2)
    #          = ceil(31.3955) = 32.
    assert required_items_for_power([-1.0, 1.0, 3.0]) == 32


def test_power_more_items_tighten_the_variance_estimate():
    # Same values x20 (n=60): mean 1.0, var(ddof=1) = 160/59 = 2.71186
    # -> required = ceil(7.8488792 * 2.7118644) = ceil(21.285) = 22.
    assert required_items_for_power([-1.0, 1.0, 3.0] * 20) == 22


def test_power_zero_variance_nonzero_mean_is_min_floor():
    # sd = 0 with a real effect: formula gives 0, floored to the minimum paired n of 2.
    assert required_items_for_power([2.0, 2.0, 2.0]) == 2


def test_power_zero_mean_returns_none():
    # No observed effect -> infinite sample size; None says "not estimable".
    assert required_items_for_power([-1.0, 1.0]) is None


def test_power_single_item_returns_none():
    # ddof=1 sd is undefined at n=1; must return None, never numpy's NaN/warning.
    assert required_items_for_power([3.0]) is None


def test_power_cancellation_null_returns_none():
    # Review finding: a perfectly balanced pairwise null with non-dyadic per-item
    # scores (means quantized at 1/12) sums to ~-8.3e-17, not exactly 0.0. The
    # guard must treat a mean that far below the data scale as not estimable —
    # an exact == 0.0 check reported ~6e31 required items here.
    scores = [float(np.mean([1, 1, 0.5, 0, 0, 0])), float(np.mean([1, 0.5, 0.5, 1, 0.5, 0]))]
    diffs = [(1.0 - s) - s for s in scores]  # exactly balanced: 1/6 and -1/6 up to ulps
    assert float(np.mean(diffs)) != 0.0  # the trap is real: cancellation is inexact
    assert required_items_for_power(diffs) is None


def test_power_extreme_cancellation_never_overflows():
    # |mean| ~ 5e-201 with sd ~ 1: without the tolerance guard math.ceil overflows.
    assert required_items_for_power([1.0, -1.0, 1e-200]) is None


# --- brief proofs: synthetic distributions (DESIGN §7 validation) -----------------


def test_identical_distributions_ci_contains_zero_and_not_significant():
    # Two draws from the SAME distribution, paired -> the true difference is 0.
    # Probabilistic in general, deterministic here: data seed 42 is pinned and
    # pre-screened (realized mean -0.0505, CI (-0.240, 0.142), p 0.614 — wide margins).
    rng = np.random.default_rng(42)
    d = rng.normal(5.0, 1.0, 200) - rng.normal(5.0, 1.0, 200)
    lo, hi = bootstrap_ci(d, seed=0)
    assert lo < 0 < hi
    assert sign_flip_pvalue(d, seed=0) > 0.05


def test_shifted_distributions_detected_and_ci_brackets_true_shift():
    # True shift 0.5, noise sd 0.1, n=100 -> detection is ~48 standard errors out.
    # Data seed 8 is pre-screened: realized mean 0.49927, CI (0.478, 0.521); seed 7
    # verifiably misses the 0.5 bracket by 0.0003 — a bracket failure means pick a
    # screened seed, never loosen the assertion.
    d = np.random.default_rng(8).normal(0.5, 0.1, 100)
    lo, hi = bootstrap_ci(d, seed=0)
    assert lo > 0
    assert lo < 0.5 < hi
    assert sign_flip_pvalue(d, seed=0) <= 0.001


# --- extraction: stored rows -> per-item scores per variant -----------------------


@pytest.mark.parametrize(
    ("verdict", "order", "expected_control"),
    [
        pytest.param("A", "ab", 1.0, id="A-ab"),
        pytest.param("B", "ab", 0.0, id="B-ab"),
        pytest.param("TIE", "ab", 0.5, id="TIE-ab"),
        pytest.param("A", "ba", 0.0, id="A-ba"),
        pytest.param("B", "ba", 1.0, id="B-ba"),
        pytest.param("TIE", "ba", 0.5, id="TIE-ba"),
    ],
)
def test_pairwise_verdict_position_mapping_all_six(store, verdict, order, expected_control):
    # THE mapping test. The verdict names the PRESENTED positions: under 'ba' the
    # declared-second variant sits in position A, so verdict "A" means it won.
    run_id, cond = make_run(store)
    c_sid = add_ok_sample(store, run_id, cond["control"], "q1")
    t_sid = add_ok_sample(store, run_id, cond["treatment"], "q1")
    add_pair_judgment(
        store, run_id, "q1", order=order, verdict=verdict, control_sid=c_sid, treatment_sid=t_sid
    )
    table = pairwise_item_scores(*tables(store, run_id))
    assert table.scores["control"]["q1"] == expected_control
    assert table.scores["treatment"]["q1"] == 1.0 - expected_control


def test_always_a_judge_washes_out_to_half(store):
    # A judge that always answers "A" under both orders: control gets 1.0 (ab) and
    # 0.0 (ba) -> per-item 0.5 for BOTH variants, diff exactly 0. Position bias must
    # wash out when both orders are stored — why M2 defaulted position_swap on.
    run_id, cond = make_run(store)
    c_sid = add_ok_sample(store, run_id, cond["control"], "q1")
    t_sid = add_ok_sample(store, run_id, cond["treatment"], "q1")
    for order in ("ab", "ba"):
        add_pair_judgment(
            store, run_id, "q1", order=order, verdict="A", control_sid=c_sid, treatment_sid=t_sid
        )
    table = pairwise_item_scores(*tables(store, run_id))
    assert table.scores["control"]["q1"] == 0.5
    assert table.scores["treatment"]["q1"] == 0.5


def test_pairwise_averages_over_replicates_and_orders(store):
    # 2 items x 2 replicates x 2 orders. q1: control wins all four judgments.
    # q2: control [1.0, 0.0, 0.5, 0.5] -> 0.5.
    run_id, cond = make_run(store)
    sids = {
        (name, item, index): add_ok_sample(store, run_id, cond[name], item, index)
        for name in ("control", "treatment")
        for item in ("q1", "q2")
        for index in (0, 1)
    }

    def judge(item, index, order, verdict):
        add_pair_judgment(
            store,
            run_id,
            item,
            order=order,
            verdict=verdict,
            control_sid=sids[("control", item, index)],
            treatment_sid=sids[("treatment", item, index)],
        )

    judge("q1", 0, "ab", "A")
    judge("q1", 0, "ba", "B")
    judge("q1", 1, "ab", "A")
    judge("q1", 1, "ba", "B")
    judge("q2", 0, "ab", "A")
    judge("q2", 0, "ba", "A")
    judge("q2", 1, "ab", "TIE")
    judge("q2", 1, "ba", "TIE")
    table = pairwise_item_scores(*tables(store, run_id))
    assert table.scores["control"] == {"q1": 1.0, "q2": 0.5}
    assert table.scores["treatment"] == {"q1": 0.0, "q2": 0.5}
    assert table.n_judgments_used == 8


def test_pairwise_error_rows_skipped_item_retained(store):
    run_id, cond = make_run(store)
    c_sid = add_ok_sample(store, run_id, cond["control"], "q1")
    t_sid = add_ok_sample(store, run_id, cond["treatment"], "q1")
    add_pair_judgment(
        store, run_id, "q1", order="ab", verdict="A", control_sid=c_sid, treatment_sid=t_sid
    )
    add_pair_judgment(
        store,
        run_id,
        "q1",
        order="ba",
        verdict=None,
        control_sid=c_sid,
        treatment_sid=t_sid,
        error="judge exploded",
    )
    table = pairwise_item_scores(*tables(store, run_id))
    assert table.scores["control"]["q1"] == 1.0  # computed from the valid row only
    assert table.n_judgments_used == 1
    assert table.n_judgments_errored == 1


def test_pairwise_drifted_rows_without_error_counted_errored(store):
    # Drifted data the runner never writes (error IS NULL but verdict NULL or
    # unknown) must be counted as errored, not crash and not score. The factory
    # cannot produce these rows by construction, so write them directly.
    run_id, cond = make_run(store)
    c_sid = add_ok_sample(store, run_id, cond["control"], "q1")
    t_sid = add_ok_sample(store, run_id, cond["treatment"], "q1")
    add_pair_judgment(
        store, run_id, "q1", order="ab", verdict="A", control_sid=c_sid, treatment_sid=t_sid
    )
    for drifted_verdict in ("C", None):
        store.add_judgment(
            run_id=run_id,
            item_id="q1",
            judge_model="judge-model",
            mode="pairwise",
            sample_a_id=c_sid,
            sample_b_id=t_sid,
            position_order="ab",
            cache_key="j" * 64,
            verdict=drifted_verdict,
        )
    table = pairwise_item_scores(*tables(store, run_id))
    assert table.scores["control"]["q1"] == 1.0  # only the valid row scores
    assert table.n_judgments_used == 1
    assert table.n_judgments_errored == 2


def test_pairwise_item_with_only_error_judgments_dropped(store):
    run_id, cond = make_run(store)
    for item in ("q1", "q2"):
        c_sid = add_ok_sample(store, run_id, cond["control"], item)
        t_sid = add_ok_sample(store, run_id, cond["treatment"], item)
        error = "judge exploded" if item == "q2" else None
        add_pair_judgment(
            store,
            run_id,
            item,
            order="ab",
            verdict=None if error else "A",
            control_sid=c_sid,
            treatment_sid=t_sid,
            error=error,
        )
    table = pairwise_item_scores(*tables(store, run_id))
    assert "q2" not in table.scores["control"]
    assert table.n_items == 2  # denominator: distinct items over samples
    result = analyze_run(store, run_id)
    assert result.comparisons[0].item_ids == ("q1",)
    assert result.comparisons[0].n_items_dropped == 1


def test_pairwise_item_with_no_judgments_dropped(store):
    # M2 skips pairs with failed samples entirely (no judgment rows at all) —
    # a different code path from error rows.
    run_id, cond = make_run(store)
    for item in ("q1", "q2", "q3"):
        c_sid = add_ok_sample(store, run_id, cond["control"], item)
        t_sid = add_ok_sample(store, run_id, cond["treatment"], item)
        if item != "q3":
            add_pair_judgment(
                store,
                run_id,
                item,
                order="ab",
                verdict="TIE",
                control_sid=c_sid,
                treatment_sid=t_sid,
            )
    result = analyze_run(store, run_id)
    assert result.comparisons[0].item_ids == ("q1", "q2")
    assert result.comparisons[0].n_items_dropped == 1


def test_pairwise_requires_exactly_two_conditions(store):
    run_id, _ = make_run(store, variants=("control", "friendly", "formal"))
    with pytest.raises(ValueError, match="two"):
        pairwise_item_scores(*tables(store, run_id))


def test_rubric_scores_are_per_item_replicate_means(store):
    # q1: control [8, 6] -> 7.0, treatment [5, 5] -> 5.0; q2: control [4, 6] -> 5.0,
    # treatment [6, 6] -> 6.0.
    run_id, cond = make_run(store, mode="rubric")
    scores = {
        ("control", "q1"): [8, 6],
        ("treatment", "q1"): [5, 5],
        ("control", "q2"): [4, 6],
        ("treatment", "q2"): [6, 6],
    }
    for (name, item), values in scores.items():
        for index, value in enumerate(values):
            sid = add_ok_sample(store, run_id, cond[name], item, index)
            add_rubric_judgment(store, run_id, item, sample_id=sid, score=value)
    table = rubric_item_scores(*tables(store, run_id))
    assert table.scores["control"] == {"q1": 7.0, "q2": 5.0}
    assert table.scores["treatment"] == {"q1": 5.0, "q2": 6.0}


def test_rubric_drifted_score_row_without_error_counted_errored(store):
    # score IS NULL with error IS NULL never comes from the runner; must count as
    # errored, not crash the mean.
    run_id, cond = make_run(store, mode="rubric")
    sid = add_ok_sample(store, run_id, cond["control"], "q1")
    add_rubric_judgment(store, run_id, "q1", sample_id=sid, score=6)
    store.add_judgment(
        run_id=run_id,
        item_id="q1",
        judge_model="judge-model",
        mode="rubric",
        sample_a_id=sid,
        cache_key="j" * 64,
    )
    table = rubric_item_scores(*tables(store, run_id))
    assert table.scores["control"] == {"q1": 6.0}
    assert table.n_judgments_used == 1
    assert table.n_judgments_errored == 1


def test_rubric_item_missing_one_variant_dropped_from_comparison_only(store):
    run_id, cond = make_run(store, mode="rubric")
    for item in ("q1", "q2"):
        sid = add_ok_sample(store, run_id, cond["control"], item)
        add_rubric_judgment(store, run_id, item, sample_id=sid, score=5)
    sid = add_ok_sample(store, run_id, cond["treatment"], "q1")
    add_rubric_judgment(store, run_id, "q1", sample_id=sid, score=7)
    result = analyze_run(store, run_id)
    assert result.comparisons[0].item_ids == ("q1",)
    assert result.comparisons[0].n_items_dropped == 1
    assert result.scores["control"] == {"q1": 5.0, "q2": 5.0}  # histograms keep all data


def test_rubric_three_variants_yield_three_comparisons(store):
    run_id, cond = make_run(store, mode="rubric", variants=("control", "friendly", "formal"))
    for name, value in (("control", 5), ("friendly", 7), ("formal", 3)):
        sid = add_ok_sample(store, run_id, cond[name], "q1")
        add_rubric_judgment(store, run_id, "q1", sample_id=sid, score=value)
    result = analyze_run(store, run_id)
    pairs = [(c.variant_a, c.variant_b) for c in result.comparisons]
    # C(3,2) pairs in declared-pair order.
    assert pairs == [("control", "friendly"), ("control", "formal"), ("friendly", "formal")]
    assert result.comparisons[0].mean_diff == 2.0  # friendly 7 - control 5


# --- analyze_run orchestration ----------------------------------------------------


def make_scored_pairwise_run(store):
    """The test_pairwise_averages_over_replicates_and_orders data: diffs q1 -> -1.0, q2 -> 0.0."""
    run_id, cond = make_run(store)
    for item, orders in (
        ("q1", [("ab", "A"), ("ba", "B")]),
        ("q2", [("ab", "A"), ("ba", "A")]),
    ):
        c_sid = add_ok_sample(store, run_id, cond["control"], item)
        t_sid = add_ok_sample(store, run_id, cond["treatment"], item)
        for order, verdict in orders:
            add_pair_judgment(
                store,
                run_id,
                item,
                order=order,
                verdict=verdict,
                control_sid=c_sid,
                treatment_sid=t_sid,
            )
    return run_id


def test_analyze_run_matches_math_layer_on_known_pairwise_data(store):
    run_id = make_scored_pairwise_run(store)
    result = analyze_run(store, run_id)
    assert len(result.comparisons) == 1
    comparison = result.comparisons[0]
    assert (comparison.variant_a, comparison.variant_b) == ("control", "treatment")
    # diffs are treatment - control: q1 = 0 - 1 = -1.0, q2 = 0.5 - 0.5 = 0.0.
    assert comparison.item_ids == ("q1", "q2")
    assert comparison.diffs == (-1.0, 0.0)
    assert comparison.mean_diff == -0.5
    # Layer consistency under the same seed — no hardcoding of resampling values.
    assert (comparison.ci_low, comparison.ci_high) == studentized_ci([-1.0, 0.0], seed=0)
    # Exhaustive n=2 on [-1, 0]: all 4 sign patterns give |sum| = 1 (the 0 never
    # moves) -> p = 4/4 = 1.0.
    assert comparison.p_value == 1.0
    assert comparison.p_method == "exhaustive"
    assert comparison.n_permutations == 4
    # Report-labeling metadata (the level itself is behaviorally pinned by
    # test_bootstrap_ci_tail_mass_pins_95_percent_level).
    assert comparison.ci_level == 0.95
    assert comparison.alpha == 0.05
    assert comparison.target_power == 0.80
    assert comparison.n_resamples == 10_000


def test_analyze_run_default_seed_reproducible(store):
    run_id = make_scored_pairwise_run(store)
    assert analyze_run(store, run_id) == analyze_run(store, run_id)


def test_analyze_run_always_a_data_degenerates(store):
    # The always-"A" judge: all diffs exactly 0 -> CI (0, 0), p 1.0, power not estimable.
    run_id, cond = make_run(store)
    for item in ("q1", "q2"):
        c_sid = add_ok_sample(store, run_id, cond["control"], item)
        t_sid = add_ok_sample(store, run_id, cond["treatment"], item)
        for order in ("ab", "ba"):
            add_pair_judgment(
                store,
                run_id,
                item,
                order=order,
                verdict="A",
                control_sid=c_sid,
                treatment_sid=t_sid,
            )
    comparison = analyze_run(store, run_id).comparisons[0]
    assert comparison.mean_diff == 0.0
    # M8: every difference is identical, so the standard error is exactly zero and
    # the interval is withheld rather than reported as zero-width.
    assert (comparison.ci_low, comparison.ci_high) == (None, None)
    assert comparison.p_value == 1.0
    assert comparison.n_required_items is None
    assert comparison.n_additional_items is None


def test_analyze_run_rubric_hand_computed_stats(store):
    # Per-item diffs (treatment - control): q1 = 4-5 = -1, q2 = 5-4 = 1, q3 = 5-2 = 3.
    run_id, cond = make_run(store, mode="rubric")
    values = {
        "q1": ("control", 5, "treatment", 4),
        "q2": ("control", 4, "treatment", 5),
        "q3": ("control", 2, "treatment", 5),
    }
    for item, (name_a, value_a, name_b, value_b) in values.items():
        for name, value in ((name_a, value_a), (name_b, value_b)):
            sid = add_ok_sample(store, run_id, cond[name], item)
            add_rubric_judgment(store, run_id, item, sample_id=sid, score=value)
    comparison = analyze_run(store, run_id).comparisons[0]
    assert comparison.diffs == (-1.0, 1.0, 3.0)
    assert comparison.mean_diff == 1.0
    # Exhaustive n=3 on [-1, 1, 3], |observed sum| = 3; the 8 pattern sums are
    # (3, -3, 1, -5, 5, -1, 3, -3) -> six of them reach |sum| >= 3 -> p = 6/8.
    assert comparison.p_value == 0.75
    # Power hand-check (verified): mean 1.0, sd 2.0 -> required 32, additional 29.
    assert comparison.n_required_items == 32
    assert comparison.n_additional_items == 29
    assert (comparison.ci_low, comparison.ci_high) == studentized_ci([-1.0, 1.0, 3.0], seed=0)


def test_analyze_run_constant_diffs_additional_clamped_to_zero(store):
    # Every item's diff is exactly 2 -> required = 2 (floor), additional = max(0, 2-3) = 0.
    run_id, cond = make_run(store, mode="rubric")
    for item in ("q1", "q2", "q3"):
        for name, value in (("control", 3), ("treatment", 5)):
            sid = add_ok_sample(store, run_id, cond[name], item)
            add_rubric_judgment(store, run_id, item, sample_id=sid, score=value)
    comparison = analyze_run(store, run_id).comparisons[0]
    # M8: every difference is identical, so the standard error is exactly zero and
    # the interval is withheld rather than reported as zero-width.
    assert (comparison.ci_low, comparison.ci_high) == (None, None)
    assert comparison.p_value == 0.25  # 2/8: only the all-same-sign patterns tie
    assert comparison.n_required_items == 2
    assert comparison.n_additional_items == 0


def test_analyze_run_unknown_run_raises(store):
    with pytest.raises(ValueError, match="not found"):
        analyze_run(store, "20990101-000000-dead")


def test_analyze_run_without_judge_raises(store):
    run_id = store.create_run("no-judge", {"name": "no-judge"})
    with pytest.raises(ValueError, match="no judge configured"):
        analyze_run(store, run_id)


def test_analyze_run_with_judge_but_no_judgments_raises(store):
    run_id, cond = make_run(store)
    add_ok_sample(store, run_id, cond["control"], "q1")
    with pytest.raises(ValueError, match="no judgments"):
        analyze_run(store, run_id)


def test_analyze_run_pair_with_no_shared_items_raises(store):
    run_id, cond = make_run(store, mode="rubric")
    for name, item in (("control", "q1"), ("treatment", "q2")):
        sid = add_ok_sample(store, run_id, cond[name], item)
        add_rubric_judgment(store, run_id, item, sample_id=sid, score=5)
    with pytest.raises(ValueError, match="no paired items"):
        analyze_run(store, run_id)


def test_analyze_run_unknown_judge_mode_raises(store):
    # A drifted spec_json must fail loudly, never silently analyze as rubric.
    run_id, cond = make_run(store, mode="ranked")
    sid = add_ok_sample(store, run_id, cond["control"], "q1")
    store.add_judgment(
        run_id=run_id,
        item_id="q1",
        judge_model="judge-model",
        mode="ranked",
        sample_a_id=sid,
        cache_key="j" * 64,
        score=5.0,
    )
    with pytest.raises(ValueError, match="unknown judge mode"):
        analyze_run(store, run_id)


def test_analyze_run_item_ids_sorted_regardless_of_insertion_order(store):
    # Items inserted q2-first must still come out lexicographic — the sort feeds the
    # seeded bootstrap, so its absence would silently change seeded CIs.
    run_id, cond = make_run(store, mode="rubric")
    for item, control_score, treatment_score in (("q2", 2, 9), ("q1", 5, 6)):
        for name, value in (("control", control_score), ("treatment", treatment_score)):
            sid = add_ok_sample(store, run_id, cond[name], item)
            add_rubric_judgment(store, run_id, item, sample_id=sid, score=value)
    comparison = analyze_run(store, run_id).comparisons[0]
    assert comparison.item_ids == ("q1", "q2")
    assert comparison.diffs == (1.0, 7.0)  # distinct values: a wrong order flips them


@pytest.mark.anyio
async def test_analyze_run_end_to_end_with_runner(store):
    # Reads rows exactly as the real runner writes them (runner-assigned sample ids,
    # both position orders) on the degenerate wash-out path. Direction of the
    # verdict->variant mapping is grounded by the position-sensitive judge test below.
    spec = ExperimentSpec.model_validate(
        {
            "name": "greeting-tone",
            "variants": [
                {"name": "control", "system": "You are helpful.", "user_template": "A: {input}"},
                {"name": "friendly", "system": "You are warm.", "user_template": "A: {input}"},
            ],
            "dataset": {"items": [{"id": "q1", "input": "sky?"}, {"id": "q2", "input": "grass?"}]},
            "sampling": {"model": "claude-haiku-4-5-20251001"},
            "n_samples": 1,
            "limits": {"concurrency": 4, "requests_per_minute": 100_000},
            "judge": {"model": "judge-model", "mode": "pairwise"},
        }
    )
    client = MockClient()
    client.add_rule(lambda request: request.model == "judge-model", "A")
    run_id = await run_experiment(spec, store, client)
    result = analyze_run(store, run_id)
    assert result.mode == "pairwise"
    assert result.n_items == 2
    comparison = result.comparisons[0]
    assert (comparison.variant_a, comparison.variant_b) == ("control", "friendly")
    # Always-"A" judge + both orders stored -> every per-item score is 0.5, diffs 0.
    assert result.scores["control"] == {"q1": 0.5, "q2": 0.5}
    assert result.scores["friendly"] == {"q1": 0.5, "q2": 0.5}
    assert comparison.diffs == (0.0, 0.0)
    # M8: every difference is identical, so the standard error is exactly zero and
    # the interval is withheld rather than reported as zero-width.
    assert (comparison.ci_low, comparison.ci_high) == (None, None)
    assert comparison.p_value == 1.0


@pytest.mark.anyio
async def test_analyze_run_position_sensitive_judge_grounds_mapping_direction(store):
    # THE anti-circularity test (review finding: the always-"A" judge washes out to
    # 0.5 under any mapping, and the six-way table shares its rule with the test
    # factory). Here the judge prefers CONTROL BY CONTENT — it reads the rendered
    # prompt and votes for whichever position holds control's canned response — so
    # the outcome is asymmetric and the expected sign comes from the runner's own
    # writes, not from any test factory. A globally inverted verdict->variant
    # mapping would report mean_diff +1.0 instead of -1.0.
    spec = ExperimentSpec.model_validate(
        {
            "name": "greeting-tone",
            "variants": [
                {"name": "control", "system": "You are helpful.", "user_template": "A: {input}"},
                {"name": "friendly", "system": "You are warm.", "user_template": "A: {input}"},
            ],
            "dataset": {"items": [{"id": "q1", "input": "sky?"}, {"id": "q2", "input": "grass?"}]},
            "sampling": {"model": "claude-haiku-4-5-20251001"},
            "n_samples": 1,
            "limits": {"concurrency": 4, "requests_per_minute": 100_000},
            "judge": {"model": "judge-model", "mode": "pairwise"},
        }
    )
    client = MockClient()
    client.add_rule(
        lambda request: (
            request.model == "judge-model" and "Response A:\nCONTROL-RESPONSE" in request.user
        ),
        "A",
    )
    client.add_rule(lambda request: request.model == "judge-model", "B")
    client.add_rule(lambda request: request.system == "You are helpful.", "CONTROL-RESPONSE")
    client.add_rule(lambda request: request.system == "You are warm.", "FRIENDLY-RESPONSE")
    run_id = await run_experiment(spec, store, client)
    result = analyze_run(store, run_id)
    # Control wins every judgment in BOTH presentation orders.
    assert result.scores["control"] == {"q1": 1.0, "q2": 1.0}
    assert result.scores["friendly"] == {"q1": 0.0, "q2": 0.0}
    comparison = result.comparisons[0]
    assert comparison.diffs == (-1.0, -1.0)  # friendly - control
    assert comparison.mean_diff == -1.0
    # M8: every difference is identical, so the standard error is exactly zero and
    # the interval is withheld rather than reported as zero-width.
    assert (comparison.ci_low, comparison.ci_high) == (None, None)
    assert comparison.p_value == 0.5  # constant n=2: 2/4 patterns tie


# --- input validation -------------------------------------------------------------


@pytest.mark.parametrize(
    "func",
    [
        pytest.param(lambda d: bootstrap_ci(d, n_resamples=100, seed=0), id="bootstrap_ci"),
        pytest.param(
            lambda d: sign_flip_pvalue(d, n_permutations=100, seed=0), id="sign_flip_pvalue"
        ),
        pytest.param(required_items_for_power, id="required_items_for_power"),
    ],
)
class TestMathInputValidation:
    def test_rejects_empty_diffs(self, func):
        with pytest.raises(ValueError, match="empty"):
            func([])

    @pytest.mark.parametrize(
        "bad",
        [
            pytest.param(float("nan"), id="nan"),
            pytest.param(float("inf"), id="inf"),
            pytest.param(float("-inf"), id="-inf"),
        ],
    )
    def test_rejects_non_finite_diffs(self, func, bad):
        with pytest.raises(ValueError, match="finite"):
            func([1.0, bad, 2.0])


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda: bootstrap_ci([1.0, 2.0], n_resamples=0), id="bootstrap_ci"),
        pytest.param(lambda: sign_flip_pvalue([1.0, 2.0], n_permutations=0), id="sign_flip_pvalue"),
    ],
)
def test_resampling_counts_must_be_positive(call):
    with pytest.raises(ValueError, match=">= 1"):
        call()


# --- review-driven hardening (M5): foreign-sample drifted rows --------------------


def test_pairwise_foreign_sample_judgment_counted_errored(store, tmp_path):
    # A judgment referencing another run's sample is drifted data (pre-M9 files
    # lack the run-scoped FK) and must skip-and-count, never KeyError
    # (found by the M5 review: analyze crashed where audit-judge handled it).
    run_id, cond = make_run(store)
    control_sid = add_ok_sample(store, run_id, cond["control"], "q1")
    treatment_sid = add_ok_sample(store, run_id, cond["treatment"], "q1")
    add_pair_judgment(
        store,
        run_id,
        "q1",
        order="ab",
        verdict="A",
        control_sid=control_sid,
        treatment_sid=treatment_sid,
    )
    other_run, other_cond = make_run(store)
    foreign_sid = add_ok_sample(store, other_run, other_cond["control"], "q1")
    insert_legacy_judgment(
        tmp_path / "mimir.db",
        run_id,
        "q1",
        mode="pairwise",
        sample_a_id=foreign_sid,
        sample_b_id=treatment_sid,
        position_order="ab",
        verdict="A",
    )
    table = pairwise_item_scores(*tables(store, run_id))
    assert table.n_judgments_used == 1
    assert table.n_judgments_errored == 1
    assert table.scores["control"] == {"q1": 1.0}


def test_rubric_foreign_sample_judgment_counted_errored(store, tmp_path):
    run_id, cond = make_run(store, mode="rubric")
    sid = add_ok_sample(store, run_id, cond["control"], "q1")
    add_rubric_judgment(store, run_id, "q1", sample_id=sid, score=7)
    other_run, other_cond = make_run(store, mode="rubric")
    foreign_sid = add_ok_sample(store, other_run, other_cond["control"], "q1")
    insert_legacy_judgment(
        tmp_path / "mimir.db", run_id, "q2", mode="rubric", sample_a_id=foreign_sid, score=3.0
    )
    table = rubric_item_scores(*tables(store, run_id))
    assert table.n_judgments_used == 1
    assert table.n_judgments_errored == 1
    assert table.scores["control"] == {"q1": 7.0}


# --- M7: common random numbers — variance quantification --------------------------


def test_crn_shared_noise_cuts_paired_diff_variance_by_the_theoretical_factor():
    # Validation test (GREEN on arrival — no production code under test beyond
    # required_items_for_power): quantifies WHY the CRN + item-pairing design
    # reduces variance. Model: score(v, i) = mu_v + b_i + (s_i + u_{v,i}) with b
    # the item effect, s the seed-shared noise component, u idiosyncratic.
    #   Var(unpaired diff)          = 2(sb^2 + ss^2 + su^2) = 2.9
    #   Var(paired, indep noise)    = 2(ss^2 + su^2)        = 0.9   (b cancels)
    #   Var(paired + CRN)           = 2su^2                 = 0.18  (s cancels too)
    # Theoretical CRN reduction 1/(1 - rho) = 5.0 at rho = 0.8. Data seed 2 is
    # pre-screened: realized 2.9389 / 0.9677 / 0.1906, ratio 5.078, required
    # items 85 (independent) vs 15 (CRN). Never change the seed to make an
    # assertion pass.
    rng = np.random.default_rng(2)
    n, delta = 400, 0.3
    sigma_b, sigma_s, sigma_u = 1.0, 0.6, 0.3
    b, b_other = rng.normal(0.0, sigma_b, (2, n))
    s_a, s_b, s_shared = rng.normal(0.0, sigma_s, (3, n))
    u_a1, u_b1, u_a2, u_b2 = rng.normal(0.0, sigma_u, (4, n))
    d_unpaired = (b_other + delta + s_b + u_b1) - (b + s_a + u_a1)
    d_ind = (b + delta + s_b + u_b1) - (b + s_a + u_a1)
    d_crn = (b + delta + s_shared + u_b2) - (b + s_shared + u_a2)
    var_unpaired = float(np.var(d_unpaired, ddof=1))
    var_ind = float(np.var(d_ind, ddof=1))
    var_crn = float(np.var(d_crn, ddof=1))
    assert var_unpaired > var_ind > var_crn
    assert 2.5 < var_unpaired < 3.3
    assert 0.75 < var_ind < 1.15
    assert 0.14 < var_crn < 0.24
    assert 4.0 < var_ind / var_crn < 6.5
    assert required_items_for_power(d_ind) == 85
    assert required_items_for_power(d_crn) == 15


class _EchoJudgeClient:
    """Rubric judge that parrots the SCORE=<n> marker embedded in the prompt, so
    judged scores equal the sampled values and analyze_run reads sampling noise
    directly. Subclasses define how sample text derives from the request."""

    async def complete(self, request):
        if request.model == "judge-model":
            text = re.search(r"SCORE=(\d+)", request.user).group(1)
        else:
            text = f"SCORE={self._sample_value(request)}"
        return CompletionResponse(
            text=text,
            raw={},
            input_tokens=1,
            output_tokens=1,
            latency_ms=1.0,
            model=request.model,
        )


class SeedHonoringClient(_EchoJudgeClient):
    # Pure function of request.seed ONLY — the M7 CRN contract is that the seed
    # alone identifies the replicate's random state; prompt content is ignored.
    def _sample_value(self, request):
        return 1 + (request.seed * 7) % 10  # 1..10; distinct for consecutive seeds


class PromptSensitiveClient(_EchoJudgeClient):
    # A real LLM without wire-level seeding: noise depends on the whole request,
    # so shared seeds cannot make it cancel across variants.
    def _sample_value(self, request):
        blob = f"{request.seed}|{request.system}|{request.user}".encode()
        return 1 + zlib.crc32(blob) % 10


def _crn_spec(items):
    return ExperimentSpec.model_validate(
        {
            "name": "crn-null",
            "variants": [
                {"name": "control", "system": "You are helpful.", "user_template": "A: {input}"},
                {"name": "friendly", "system": "You are warm.", "user_template": "A: {input}"},
            ],
            "dataset": {"items": items},
            "sampling": {"model": "claude-haiku-4-5-20251001", "seed": 7},
            "n_samples": 3,
            "judge": {"model": "judge-model", "mode": "rubric"},
            "limits": {"concurrency": 4, "requests_per_minute": 100_000},
        }
    )


_CRN_ITEMS = [{"id": f"q{i}", "input": f"question {i}"} for i in range(1, 5)]


@pytest.mark.anyio
async def test_crn_seed_honoring_client_zero_variance_under_null(store):
    # With CRN seeds (sampling.seed + sample_index, shared across variants) a
    # client whose stochasticity depends on request.seed ONLY gives replicate r
    # of both variants identical output — under the null every paired diff is
    # EXACTLY zero. len(set(scores)) == 3 is the anti-degeneracy guard and the
    # RED mechanism: without the CRN derivation every replicate carries seed 7
    # and all 24 judged scores collapse to one value.
    run_id = await run_experiment(_crn_spec(_CRN_ITEMS), store, SeedHonoringClient())
    assert store.get_run(run_id)["status"] == "complete"
    scores = {j["score"] for j in store.get_judgments(run_id)}
    assert len(scores) == 3  # seeds 7, 8, 9 -> values 10.0, 7.0, 4.0
    comparison = analyze_run(store, run_id).comparisons[0]
    assert comparison.diffs == (0.0, 0.0, 0.0, 0.0)  # exactly zero, not approx
    assert comparison.mean_diff == 0.0
    # M8: every difference is identical, so the standard error is exactly zero and
    # the interval is withheld rather than reported as zero-width.
    assert (comparison.ci_low, comparison.ci_high) == (None, None)
    assert comparison.p_value == 1.0


@pytest.mark.anyio
async def test_prompt_sensitive_client_diffs_not_cancelled_by_crn(store):
    # Contrast fixture (GREEN before and after M7): when noise depends on the
    # prompt — as on a real API with no wire seed — shared seeds cannot cancel
    # it, so the zero in the null test above comes from seed-honoring, not from
    # the pipeline. crc32 is platform-stable; realized diffs pre-screened nonzero.
    run_id = await run_experiment(_crn_spec(_CRN_ITEMS), store, PromptSensitiveClient())
    assert store.get_run(run_id)["status"] == "complete"
    comparison = analyze_run(store, run_id).comparisons[0]
    assert any(diff != 0.0 for diff in comparison.diffs)


# --- M7: multiple-comparison corrections (pure math) ------------------------------


def test_holm_and_bh_step_shapes_hand_computed():
    # Sorted p = (0.015625, 0.25, 0.25, 0.25), m = 4.
    # Holm scales (4,3,2,1): (0.0625, 0.75, 0.50, 0.25) -> running max ->
    #   (0.0625, 0.75, 0.75, 0.75).
    # BH scales (4/1, 4/2, 4/3, 4/4): (0.0625, 0.5, 1/3, 0.25) -> reverse running
    #   min -> (0.0625, 0.25, 0.25, 0.25). All values dyadic: exact ==.
    p = (0.015625, 0.25, 0.25, 0.25)
    assert holm_bonferroni(p) == (0.0625, 0.75, 0.75, 0.75)
    assert benjamini_hochberg(p) == (0.0625, 0.25, 0.25, 0.25)


def test_corrections_preserve_input_order():
    p = (0.25, 0.015625, 0.25, 0.25)
    assert holm_bonferroni(p) == (0.75, 0.0625, 0.75, 0.75)
    assert benjamini_hochberg(p) == (0.25, 0.0625, 0.25, 0.25)


def test_holm_clips_at_one_bh_capped_by_largest_p():
    # Holm: (2*0.6, 1*0.7) -> max -> (1.2, 1.2) -> clipped (1.0, 1.0).
    # BH: (2*0.6/1, 2*0.7/2) = (1.2, 0.7) -> reverse min -> (0.7, 0.7): the
    # largest adjusted BH value is p_(m) itself, so the clip is a no-op here.
    assert holm_bonferroni((0.6, 0.7)) == (1.0, 1.0)
    assert benjamini_hochberg((0.6, 0.7)) == (0.7, 0.7)


def test_textbook_ladder():
    # Holm on (0.01..0.05): scales (5,4,3,2,1) -> (0.05, 0.08, 0.09, 0.08, 0.05)
    # -> running max -> (0.05, 0.08, 0.09, 0.09, 0.09). Exact in float.
    # BH: every m*p_(i)/i is 0.05 in real arithmetic, but 5*0.03/3 is
    # 0.049999999999999996 in float -> approx, never a loosened oracle.
    p = (0.01, 0.02, 0.03, 0.04, 0.05)
    assert holm_bonferroni(p) == (0.05, 0.08, 0.09, 0.09, 0.09)
    assert benjamini_hochberg(p) == pytest.approx([0.05] * 5, abs=1e-12)


def test_corrections_m1_identity():
    assert holm_bonferroni((0.37,)) == (0.37,)
    assert benjamini_hochberg((0.37,)) == (0.37,)


def test_corrections_all_equal_ties():
    # Ties get identical adjusted values through the accumulates, no special case:
    # Holm's running max propagates the largest scale, BH's reverse min the smallest.
    p = (0.25, 0.25, 0.25, 0.25)
    assert holm_bonferroni(p) == (1.0, 1.0, 1.0, 1.0)
    assert benjamini_hochberg(p) == (0.25, 0.25, 0.25, 0.25)


def test_corrections_mixed_unsorted_vector():
    p = (0.001, 0.9, 0.02, 0.6, 0.04)
    assert holm_bonferroni(p) == (0.005, 1.0, 0.08, 1.0, 0.12)
    # 5*0.04/3 = 0.2/3 is non-dyadic -> approx on the BH vector.
    assert benjamini_hochberg(p) == pytest.approx(
        [0.005, 0.9, 0.05, 0.75, 0.06666666666666667], abs=1e-12
    )


def test_corrections_monotone_and_bh_dominates_holm():
    # On any vector: adjusted values ordered by raw-p rank are non-decreasing,
    # and BH <= Holm elementwise (scale m/i <= m-i+1 for every rank i, and the
    # accumulates preserve the inequality). Deterministic, not probabilistic.
    p = np.random.default_rng(0).random(20)
    order = np.argsort(p)
    for correct in (holm_bonferroni, benjamini_hochberg):
        adjusted = np.asarray(correct(p))
        ranked = adjusted[order]
        assert (np.diff(ranked) >= 0).all()
    holm = holm_bonferroni(p)
    bh = benjamini_hochberg(p)
    assert all(b <= h for b, h in zip(bh, holm, strict=True))


@pytest.mark.parametrize("correct", [holm_bonferroni, benjamini_hochberg])
class TestPValueValidation:
    def test_empty_rejected(self, correct):
        with pytest.raises(ValueError, match="empty"):
            correct([])

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_rejected(self, correct, bad):
        with pytest.raises(ValueError, match="finite"):
            correct([0.5, bad])

    @pytest.mark.parametrize("bad", [-0.1, 1.5])
    def test_out_of_range_rejected(self, correct, bad):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            correct([0.5, bad])


def test_null_calibration_uncorrected_vs_corrected():
    # Known-answer FDR/FWER calibration on m = 10 INDEPENDENT Uniform(0,1)
    # p-vectors, where the theory is exact: uncorrected any-rejection is
    # 1 - 0.95^10 = 0.4013 ("~40% false-positive runs"); BH any-discovery under
    # the global null equals alpha (Simes); Holm is Bonferroni-like on the min.
    # Seed 0, N = 10,000 pre-screened realized rates: 0.3994 / 0.0489 / 0.0473.
    # NOTE the brief's "10 conditions -> ~40%": 10 CONDITIONS give C(10,2) = 45
    # dependent comparisons with an any-false-winner rate near 0.75; the 40%
    # figure is exactly m = 10 independent tests, which is what this simulates.
    # Sign-flip p-values are discrete and super-uniform under the null, so the
    # corrections are conservative there — uniforms are the exact contract.
    # Never change the seed to make an assertion pass.
    rows = np.random.default_rng(0).random((10_000, 10))
    alpha = 0.05
    uncorrected = float(np.mean(rows.min(axis=1) < alpha))
    bh_rate = float(np.mean([min(benjamini_hochberg(row)) < alpha for row in rows]))
    holm_rate = float(np.mean([min(holm_bonferroni(row)) < alpha for row in rows]))
    assert 0.38 < uncorrected < 0.42
    assert 0.04 < bh_rate < 0.06
    assert 0.04 < holm_rate < 0.06
    assert uncorrected > 5 * bh_rate
    assert holm_rate <= bh_rate


# --- M7: replicate-level extraction -----------------------------------------------


def test_rubric_replicate_scores_keyed_by_sample_index(store):
    run_id, cond = make_run(store, mode="rubric")
    for name, base in (("control", 3), ("treatment", 5)):
        for item in ("q1", "q2"):
            for index in (0, 1):
                sid = add_ok_sample(store, run_id, cond[name], item, sample_index=index)
                add_rubric_judgment(store, run_id, item, sample_id=sid, score=base + index)
    table = rubric_replicate_scores(*tables(store, run_id))
    assert table.scores == {
        "control": {"q1": {0: 3.0, 1: 4.0}, "q2": {0: 3.0, 1: 4.0}},
        "treatment": {"q1": {0: 5.0, 1: 6.0}, "q2": {0: 5.0, 1: 6.0}},
    }
    assert table.n_judgments_used == 8
    assert table.n_judgments_errored == 0


def test_pairwise_replicate_scores_average_both_orders_within_replicate(store):
    # Split verdict across orders: 'ab' A -> control wins (1.0), 'ba' A ->
    # treatment wins (control 0.0) -> replicate mean 0.5 for both directions.
    run_id, cond = make_run(store)
    c_sid = add_ok_sample(store, run_id, cond["control"], "q1")
    t_sid = add_ok_sample(store, run_id, cond["treatment"], "q1")
    add_pair_judgment(
        store, run_id, "q1", order="ab", verdict="A", control_sid=c_sid, treatment_sid=t_sid
    )
    add_pair_judgment(
        store, run_id, "q1", order="ba", verdict="A", control_sid=c_sid, treatment_sid=t_sid
    )
    table = pairwise_replicate_scores(*tables(store, run_id))
    assert table.scores == {"control": {"q1": {0: 0.5}}, "treatment": {"q1": {0: 0.5}}}
    assert table.n_judgments_used == 2


def test_pairwise_replicate_scores_consistent_winner(store):
    # 'ab' A and 'ba' B both mean control won -> 1.0, complementary 0.0.
    run_id, cond = make_run(store)
    c_sid = add_ok_sample(store, run_id, cond["control"], "q1")
    t_sid = add_ok_sample(store, run_id, cond["treatment"], "q1")
    add_pair_judgment(
        store, run_id, "q1", order="ab", verdict="A", control_sid=c_sid, treatment_sid=t_sid
    )
    add_pair_judgment(
        store, run_id, "q1", order="ba", verdict="B", control_sid=c_sid, treatment_sid=t_sid
    )
    table = pairwise_replicate_scores(*tables(store, run_id))
    assert table.scores == {"control": {"q1": {0: 1.0}}, "treatment": {"q1": {0: 0.0}}}


def test_pairwise_replicate_scores_requires_two_conditions(store):
    run_id, _ = make_run(store, variants=("only",))
    with pytest.raises(ValueError, match="exactly two variants"):
        pairwise_replicate_scores(*tables(store, run_id))


def test_pairwise_replicate_scores_skip_and_count_drifted_rows(store, tmp_path):
    run_id, cond = make_run(store)
    c_sid = add_ok_sample(store, run_id, cond["control"], "q1")
    t_sid = add_ok_sample(store, run_id, cond["treatment"], "q1")
    add_pair_judgment(
        store, run_id, "q1", order="ab", verdict="A", control_sid=c_sid, treatment_sid=t_sid
    )
    add_pair_judgment(
        store,
        run_id,
        "q1",
        order="ba",
        verdict="A",
        control_sid=c_sid,
        treatment_sid=t_sid,
        error="boom",
    )
    other_run, other_cond = make_run(store)
    foreign_sid = add_ok_sample(store, other_run, other_cond["control"], "q1")
    insert_legacy_judgment(
        tmp_path / "mimir.db",
        run_id,
        "q1",
        mode="pairwise",
        sample_a_id=foreign_sid,
        sample_b_id=t_sid,
        position_order="ab",
        verdict="A",
    )
    table = pairwise_replicate_scores(*tables(store, run_id))
    assert table.n_judgments_used == 1
    assert table.n_judgments_errored == 2
    assert table.scores["control"] == {"q1": {0: 1.0}}


def test_rubric_replicate_scores_skip_and_count_drifted_rows(store, tmp_path):
    run_id, cond = make_run(store, mode="rubric")
    sid = add_ok_sample(store, run_id, cond["control"], "q1")
    add_rubric_judgment(store, run_id, "q1", sample_id=sid, score=7)
    add_rubric_judgment(store, run_id, "q1", sample_id=sid, score=3, error="boom")
    other_run, other_cond = make_run(store, mode="rubric")
    foreign_sid = add_ok_sample(store, other_run, other_cond["control"], "q1")
    insert_legacy_judgment(
        tmp_path / "mimir.db", run_id, "q2", mode="rubric", sample_a_id=foreign_sid, score=5.0
    )
    table = rubric_replicate_scores(*tables(store, run_id))
    assert table.n_judgments_used == 1
    assert table.n_judgments_errored == 2
    assert table.scores["control"] == {"q1": {0: 7.0}}


def test_replicate_diffs_pairs_by_index_and_drops_unmatched(store):
    run_id, cond = make_run(store, mode="rubric")
    # control q1 replicates {0, 1}; treatment q1 replicates {1, 2}; q2 control-only.
    for index, score in ((0, 3), (1, 4)):
        sid = add_ok_sample(store, run_id, cond["control"], "q1", sample_index=index)
        add_rubric_judgment(store, run_id, "q1", sample_id=sid, score=score)
    for index, score in ((1, 9), (2, 8)):
        sid = add_ok_sample(store, run_id, cond["treatment"], "q1", sample_index=index)
        add_rubric_judgment(store, run_id, "q1", sample_id=sid, score=score)
    sid = add_ok_sample(store, run_id, cond["control"], "q2")
    add_rubric_judgment(store, run_id, "q2", sample_id=sid, score=1)
    table = rubric_replicate_scores(*tables(store, run_id))
    diffs = replicate_diffs(table, "control", "treatment")
    assert diffs == {"q1": (5.0,)}  # only index 1 shared: 9 - 4; q2 dropped


def test_replicate_means_match_item_scores(store):
    # Collapsing the replicate table by mean reproduces the ScoreTable values.
    run_id, cond = make_run(store, mode="rubric")
    for name, scores in (("control", (2, 4)), ("treatment", (5, 9))):
        for index, score in enumerate(scores):
            sid = add_ok_sample(store, run_id, cond[name], "q1", sample_index=index)
            add_rubric_judgment(store, run_id, "q1", sample_id=sid, score=score)
    replicate_table = rubric_replicate_scores(*tables(store, run_id))
    item_table = rubric_item_scores(*tables(store, run_id))
    for variant in ("control", "treatment"):
        collapsed = float(np.mean(list(replicate_table.scores[variant]["q1"].values())))
        assert collapsed == item_table.scores[variant]["q1"]


# --- M7: variance decomposition + power planning ----------------------------------


def test_decompose_item_dominated_recommends_more_items():
    # Item means -3, 1, 5 -> var_item_mean = 16; per-item ss = 2 each, pooled
    # var_within = 6/3 = 2; var_between = 16 - 2*0.5 = 15. All dyadic: exact ==.
    # Requirements: K = (z_a+z_p)^2 = 7.8488797...; ceil(K*16) = 126,
    # ceil(K*15.5) = 122, ceil(K*15) = 118.
    v = decompose_variance({"q1": [-4.0, -2.0], "q2": [0.0, 2.0], "q3": [4.0, 6.0]}, mean_diff=1.0)
    assert v.n_items == 3
    assert v.n_items_with_replicates == 3
    assert v.mean_replicates == 2.0
    assert v.var_item_mean == 16.0
    assert v.var_within == 2.0
    assert v.var_between == 15.0
    assert v.share_between == 0.9375
    assert v.n_required_items_current == 126
    assert v.n_required_items_double == 122
    assert v.n_required_items_limit == 118
    assert v.recommendation == "more_items"


def test_decompose_noise_dominated_recommends_more_samples():
    # Item means 0, 1, 2 -> var_item_mean = 1; per-item ss = 32 each, pooled
    # var_within = 96/3 = 32; var_between = max(0, 1 - 16) = 0 (clamped).
    # M8/M3 re-cut these numbers: the ladder used to plan off the CLIPPED scale
    # (sd = sqrt(var_within * inv_r) = 4 -> 126 items) while the headline power row
    # planned off var_item_mean (sd = 1 -> 8 items), so one report carried two
    # different answers to "how many items do I need?". The current rung is now the
    # headline's own quantity, and each later rung is floored by the one before it,
    # so the ladder can never rank "double the samples" above "current".
    v = decompose_variance({"q1": [-4.0, 4.0], "q2": [-3.0, 5.0], "q3": [-2.0, 6.0]}, mean_diff=1.0)
    assert v.var_item_mean == 1.0
    assert v.var_within == 32.0
    assert v.var_between == 0.0
    assert v.share_between == 0.0
    assert v.n_required_items_current == required_items_for_power([0.0, 1.0, 2.0]) == 8
    assert v.n_required_items_double == 8  # clamped: more samples cannot need more items
    assert v.n_required_items_limit == 2
    assert v.recommendation == "more_samples_per_item"


def test_decompose_zero_within_variance():
    v = decompose_variance({"q1": [0.0, 0.0], "q2": [1.0, 1.0], "q3": [2.0, 2.0]}, mean_diff=1.0)
    assert v.var_within == 0.0
    assert v.var_between == 1.0
    assert v.share_between == 1.0
    assert v.n_required_items_current == 8  # ceil(K*1)
    assert v.n_required_items_limit == 8
    assert v.recommendation == "more_items"


def test_decompose_exact_tie_recommends_more_items():
    # var_item_mean = 2, var_within = 2, var_between = 1 -> noise share equals
    # item share exactly; replicates can never shrink var_between, so items win.
    v = decompose_variance({"q1": [-1.0, 1.0], "q2": [1.0, 3.0]}, mean_diff=1.0)
    assert v.var_between == 1.0
    assert v.var_within == 2.0
    assert v.share_between == 0.5
    assert v.recommendation == "more_items"


def test_decompose_unbalanced_uses_pooled_ms_and_harmonic_mean():
    # Pooled SS/df = (2 + 2 + 8) / (1 + 1 + 2) = 3.0 exactly — the naive mean of
    # per-item variances would give 8/3. Harmonic mean_replicates = 1/mean(1/r_i)
    # = 1/((1/2 + 1/2 + 1/3)/3) = 2.25; var_between = 183/9 - 3*(4/9) = 19.
    v = decompose_variance(
        {"q1": [0.0, 2.0], "q2": [4.0, 6.0], "q3": [8.0, 10.0, 12.0]}, mean_diff=16 / 3
    )
    assert v.var_within == 3.0
    assert v.mean_replicates == pytest.approx(2.25, abs=1e-12)
    assert v.var_between == pytest.approx(19.0, abs=1e-9)
    assert v.n_items == 3
    assert v.n_items_with_replicates == 3


def test_decompose_none_when_not_separable():
    # No item with >= 2 replicates (n_samples=1) or fewer than 2 items.
    assert decompose_variance({"q1": [1.0], "q2": [2.0]}, mean_diff=1.0) is None
    assert decompose_variance({"q1": [1.0, 2.0, 3.0]}, mean_diff=2.0) is None


def test_decompose_invalid_diffs_rejected():
    with pytest.raises(ValueError, match="empty"):
        decompose_variance({"q1": [], "q2": [1.0, 2.0]}, mean_diff=1.0)
    with pytest.raises(ValueError, match="finite"):
        decompose_variance({"q1": [float("nan"), 1.0], "q2": [1.0, 2.0]}, mean_diff=1.0)


def test_decompose_recovers_known_variance_ratio():
    # d_ir = 0.4 + b_i + e_ir with sigma_b^2 = 0.36, sigma_e^2 = 1.0, n = 400,
    # r = 4. Data seed 0 pre-screened: realized var_between 0.3587, var_within
    # 0.9941, var_item_mean 0.6072. The parameters are chosen so that forgetting
    # to subtract var_within/r returns var_item_mean (0.6072), which misses the
    # var_between bound by 3x. Never change the seed to make an assertion pass.
    rng = np.random.default_rng(0)
    b = rng.normal(0.0, 0.6, 400)
    e = rng.normal(0.0, 1.0, (400, 4))
    d = 0.4 + b[:, None] + e
    v = decompose_variance(
        {f"q{i:03d}": tuple(row) for i, row in enumerate(d)}, mean_diff=float(d.mean())
    )
    assert abs(v.var_between - 0.36) < 0.08
    assert abs(v.var_within - 1.0) < 0.05
    assert v.var_item_mean > 0.55


def test_decompose_required_current_matches_m3_power_estimator():
    # At the observed harmonic mean replicates, the planner degenerates to the M3
    # estimator over per-item means: sd^2 = var_between + var_within/r exactly on
    # balanced data (126 here, well away from a ceil boundary at 125.58).
    v = decompose_variance({"q1": [-4.0, -2.0], "q2": [0.0, 2.0], "q3": [4.0, 6.0]}, mean_diff=1.0)
    assert v.n_required_items_current == required_items_for_power([-3.0, 1.0, 5.0]) == 126


def test_position_swap_is_not_a_replicate(store):
    # THE trap: pairwise n_samples=1 judged in both orders yields TWO judgment
    # rows but ONE replicate index -> decomposition must be None. A row-naive
    # implementation would report r = 2 and pass off position-flip disagreement
    # as sampling noise.
    run_id, cond = make_run(store)
    c_sid = add_ok_sample(store, run_id, cond["control"], "q1")
    t_sid = add_ok_sample(store, run_id, cond["treatment"], "q1")
    c2 = add_ok_sample(store, run_id, cond["control"], "q2")
    t2 = add_ok_sample(store, run_id, cond["treatment"], "q2")
    for item, c, t in (("q1", c_sid, t_sid), ("q2", c2, t2)):
        add_pair_judgment(
            store, run_id, item, order="ab", verdict="A", control_sid=c, treatment_sid=t
        )
        add_pair_judgment(
            store, run_id, item, order="ba", verdict="B", control_sid=c, treatment_sid=t
        )
    table = pairwise_replicate_scores(*tables(store, run_id))
    assert set(table.scores["control"]["q1"]) == {0}  # one replicate, two orders
    diffs = replicate_diffs(table, "control", "treatment")
    assert decompose_variance(diffs, mean_diff=-1.0) is None


# --- M7: run-level score variance shares (rubric) ---------------------------------


def _shares_table(scores):
    return ReplicateTable(scores=scores, n_judgments_used=0, n_judgments_errored=0)


def test_score_variance_shares_hand_computed_additive():
    # 2 conditions x 3 items x 2 replicates, additive cell means with constant
    # within-cell spread: MS_C = 6, MS_I = 8, MS_res = 0, pooled var_e = 2.
    # var_cond = 6/3 = 2, var_item = 8/2 = 4, var_noise = 0 + 2 = 2 -> shares
    # (0.25, 0.5, 0.25). All dyadic: exact ==.
    v = score_variance_shares(
        _shares_table(
            {
                "control": {"q1": {0: 1.0, 1: 3.0}, "q2": {0: 3.0, 1: 5.0}, "q3": {0: 5.0, 1: 7.0}},
                "treatment": {
                    "q1": {0: 3.0, 1: 5.0},
                    "q2": {0: 5.0, 1: 7.0},
                    "q3": {0: 7.0, 1: 9.0},
                },
            }
        )
    )
    assert v.n_conditions == 2
    assert v.n_items == 3
    assert v.mean_replicates == 2.0
    assert v.var_condition == 2.0
    assert v.var_item == 4.0
    assert v.var_noise == 2.0
    assert (v.share_condition, v.share_item, v.share_noise) == (0.25, 0.5, 0.25)


def test_score_variance_shares_single_replicate_noise_is_residual():
    # r = 1: replicate noise and interaction are confounded; noise = MS_res.
    # Cells [[0, 2], [2, 6]]: MS_C = MS_I = 9, MS_res = 1 -> cond = item = 4,
    # noise = 1 -> shares (4/9, 4/9, 1/9).
    v = score_variance_shares(
        _shares_table(
            {
                "control": {"q1": {0: 0.0}, "q2": {0: 2.0}},
                "treatment": {"q1": {0: 2.0}, "q2": {0: 6.0}},
            }
        )
    )
    assert v.mean_replicates == 1.0
    assert v.var_condition == 4.0
    assert v.var_item == 4.0
    assert v.var_noise == 1.0
    assert v.share_condition == pytest.approx(4 / 9, abs=1e-12)
    assert v.share_noise == pytest.approx(1 / 9, abs=1e-12)


def test_score_variance_shares_incomplete_items_dropped():
    # q3 exists only for control -> complete-case: identical result without it.
    base = {
        "control": {"q1": {0: 1.0, 1: 3.0}, "q2": {0: 3.0, 1: 5.0}},
        "treatment": {"q1": {0: 3.0, 1: 5.0}, "q2": {0: 5.0, 1: 7.0}},
    }
    with_extra = {
        "control": {**base["control"], "q3": {0: 9.0}},
        "treatment": dict(base["treatment"]),
    }
    v = score_variance_shares(_shares_table(with_extra))
    assert v == score_variance_shares(_shares_table(base))
    assert v.n_items == 2


def test_score_variance_shares_none_when_degenerate():
    assert score_variance_shares(_shares_table({"only": {"q1": {0: 1.0}, "q2": {0: 2.0}}})) is None
    assert (
        score_variance_shares(_shares_table({"a": {"q1": {0: 1.0}}, "b": {"q1": {0: 2.0}}})) is None
    )


def test_score_variance_shares_recovers_known_components():
    # s_cir = 5 + a_c + b_i + e_cir, gamma = 0; C = 5, I = 200, r = 3;
    # sigma_a = 0.8, sigma_b = 0.5, sigma_e = 1.0. The oracle is the REALIZED
    # component variances (ddof=1 over the drawn effects), not the population
    # values: data seed 1 pre-screened -> recovered vs realized gaps 0.012
    # (condition), 0.021 (item), 0.017 (noise vs 1.0). Never change the seed to
    # make an assertion pass.
    rng = np.random.default_rng(1)
    conditions, items, reps = 5, 200, 3
    a = rng.normal(0.0, 0.8, conditions)
    b = rng.normal(0.0, 0.5, items)
    e = rng.normal(0.0, 1.0, (conditions, items, reps))
    s = 5.0 + a[:, None, None] + b[None, :, None] + e
    scores = {
        f"c{ci}": {f"q{i:03d}": dict(enumerate(s[ci, i].tolist())) for i in range(items)}
        for ci in range(conditions)
    }
    v = score_variance_shares(_shares_table(scores))
    assert abs(v.var_condition - float(np.var(a, ddof=1))) < 0.08
    assert abs(v.var_item - float(np.var(b, ddof=1))) < 0.08
    assert abs(v.var_noise - 1.0) < 0.05


# --- M7: analyze_run wiring — correction family + decomposition -------------------


_MULTIARM_SCORES = {
    "a": {"q1": 2.0, "q2": 4.0, "q3": 3.0},
    "b": {"q1": 5.0, "q2": 7.0, "q3": 6.0},
    "c": {"q1": 9.0, "q2": 3.0, "q3": 5.0},
    "d": {"q1": 1.0, "q2": 8.0, "q3": 2.0},
    "e": {"q1": 6.0, "q2": 6.0, "q3": 9.0},
}


def seed_rubric_multiarm(store, variant_scores):
    run_id, cond = make_run(store, mode="rubric", variants=tuple(variant_scores))
    for name, per_item in variant_scores.items():
        for item, score in per_item.items():
            sid = add_ok_sample(store, run_id, cond[name], item)
            add_rubric_judgment(store, run_id, item, sample_id=sid, score=score)
    return run_id


def seed_rubric_replicated(store):
    """Control 5.0 on both replicates of every item; treatment q1 (1,3), q2 (5,7),
    q3 (9,11) — the replicate diffs are exactly the item-dominated hand case."""
    run_id, cond = make_run(store, mode="rubric")
    treatment_scores = {"q1": (1, 3), "q2": (5, 7), "q3": (9, 11)}
    for item, scores in treatment_scores.items():
        for index in (0, 1):
            sid = add_ok_sample(store, run_id, cond["control"], item, sample_index=index)
            add_rubric_judgment(store, run_id, item, sample_id=sid, score=5)
        for index, score in enumerate(scores):
            sid = add_ok_sample(store, run_id, cond["treatment"], item, sample_index=index)
            add_rubric_judgment(store, run_id, item, sample_id=sid, score=score)
    return run_id


def test_analyze_run_family_of_one_correction_is_identity(store):
    run_id = make_scored_pairwise_run(store)
    result = analyze_run(store, run_id)
    comparison = result.comparisons[0]
    assert comparison.n_comparisons == 1
    assert comparison.p_value_corrected == comparison.p_value
    assert comparison.correction_method == DEFAULT_CORRECTION == "holm"
    assert result.correction_method == "holm"


def test_analyze_run_multiarm_correction_matches_pure_function(store):
    run_id = seed_rubric_multiarm(store, _MULTIARM_SCORES)
    result = analyze_run(store, run_id)
    assert len(result.comparisons) == 10  # C(5,2)
    assert all(c.n_comparisons == 10 for c in result.comparisons)
    raw = tuple(c.p_value for c in result.comparisons)
    corrected = tuple(c.p_value_corrected for c in result.comparisons)
    assert corrected == holm_bonferroni(raw)
    assert all(q >= p for p, q in zip(raw, corrected, strict=True))
    assert result.correction_method == "holm"
    pairs = [(c.variant_a, c.variant_b) for c in result.comparisons]
    assert pairs[:3] == [("a", "b"), ("a", "c"), ("a", "d")]  # declared-pair order kept


def test_analyze_run_correction_bh_selectable(store):
    run_id = seed_rubric_multiarm(store, _MULTIARM_SCORES)
    result = analyze_run(store, run_id, correction="bh")
    raw = tuple(c.p_value for c in result.comparisons)
    assert tuple(c.p_value_corrected for c in result.comparisons) == benjamini_hochberg(raw)
    assert result.correction_method == "bh"


def test_analyze_run_unknown_correction_rejected(store):
    run_id = make_scored_pairwise_run(store)
    with pytest.raises(ValueError, match="correction"):
        analyze_run(store, run_id, correction="bonferroni")


def test_analyze_run_populates_variance_decomposition(store):
    run_id = seed_rubric_replicated(store)
    comparison = analyze_run(store, run_id).comparisons[0]
    assert comparison.mean_diff == 1.0
    v = comparison.variance
    assert v is not None
    assert v.var_between == 15.0
    assert v.var_within == 2.0
    assert v.mean_replicates == 2.0
    assert v.n_required_items_current == 126
    assert v.recommendation == "more_items"


def test_analyze_run_single_replicate_variance_none_shares_defined(store):
    # n_samples = 1: the diff-scale decomposition needs replicates (None), but the
    # r=1 shares table is still defined (interaction + noise confounded).
    run_id, cond = make_run(store, mode="rubric")
    for item, (c_score, t_score) in (("q1", (2, 5)), ("q2", (4, 7))):
        sid = add_ok_sample(store, run_id, cond["control"], item)
        add_rubric_judgment(store, run_id, item, sample_id=sid, score=c_score)
        sid = add_ok_sample(store, run_id, cond["treatment"], item)
        add_rubric_judgment(store, run_id, item, sample_id=sid, score=t_score)
    result = analyze_run(store, run_id)
    assert result.comparisons[0].variance is None
    assert result.score_variance is not None


def test_analyze_run_rubric_score_variance_hand_computed(store):
    # Cell means [[5,5,5],[2,6,10]]: ms_cond 1.5, ms_item 8, ms_res 8. Within-cell
    # ss: control cells are constant (0 each), treatment cells (1,3)/(5,7)/(9,11)
    # contribute 2 each -> pooled var_e = 6/6 = 1; var_gamma = 8 - 1*0.5 = 7.5,
    # noise = 8.5; condition and item components clamp to 0: this fixture is
    # interaction-dominated by construction.
    shares = analyze_run(store, seed_rubric_replicated(store)).score_variance
    assert shares is not None
    assert (shares.var_condition, shares.var_item, shares.var_noise) == (0.0, 0.0, 8.5)
    assert shares.share_noise == 1.0


def test_analyze_run_pairwise_score_variance_is_none(store):
    # Pairwise scores are complementary (B = 1 - A): the 3-way split degenerates.
    result = analyze_run(store, make_scored_pairwise_run(store))
    assert result.score_variance is None


def test_analyze_run_determinism_with_replicates_and_correction(store):
    # Extends the M3 determinism pin to the nested VarianceDecomposition /
    # ScoreVarianceShares dataclasses and the correction pass.
    run_id = seed_rubric_replicated(store)
    assert analyze_run(store, run_id) == analyze_run(store, run_id)


# --- M8: empty family, and the ulp tolerance that keeps the identity countable ----


def test_sign_flip_tolerance_keeps_the_identity_pattern_countable():
    # Rubric means over n_samples=3 land on thirds, which are not representable, and
    # `signs @ d` sums them in a different order than np.sum(d). Without the ulp
    # tolerance the identity pattern misses its OWN comparison and the exhaustive
    # test returns 0.0 — impossible for a test that always counts itself (the
    # module guarantees p >= 2^-n). Verified: this vector gives 0.0 with tau = 0.
    d = [
        1.666666666666666,
        2.333333333333333,
        3.6666666666666665,
        0.3333333333333335,
        0.3333333333333335,
    ]
    assert sign_flip_pvalue(d) == 2 / 2**5  # the identity pattern and its mirror


@pytest.mark.anyio
async def test_analyze_run_single_variant_rubric_has_no_family_to_correct(store):
    # A one-variant rubric spec is legal (spec.py Field(min_length=1); the exactly-2
    # rule is pairwise-only) and the runner completes it. M7's correction pass must
    # not turn C(1,2) == 0 comparisons into "p_values is empty".
    spec = ExperimentSpec.model_validate(
        {
            "name": "solo",
            "variants": [{"name": "only", "system": "s", "user_template": "Q: {input}"}],
            "dataset": {"items": [{"id": f"q{i}", "input": f"i{i}"} for i in range(3)]},
            "sampling": {"model": "m"},
            "n_samples": 1,
            "limits": {"concurrency": 2, "requests_per_minute": 100_000},
            "judge": {"model": "judge-model", "mode": "rubric"},
        }
    )
    client = MockClient()
    client.add_rule(lambda request: request.model == "judge-model", "7")
    run_id = await run_experiment(spec, store, client)
    assert store.get_run(run_id)["status"] == "complete"

    result = analyze_run(store, run_id)
    assert result.comparisons == []
    assert result.scores["only"] == {"q0": 7.0, "q1": 7.0, "q2": 7.0}
    assert result.correction_method == "holm"


# --- M8/C2: the studentized (bootstrap-t) interval --------------------------------


def test_studentized_ci_not_estimable_for_constant_or_single_diffs():
    # Constant diffs are routine with quantized pairwise scores; a zero-width
    # "95% CI" asserts infinite precision, so the interval is withheld instead.
    assert studentized_ci([1.0] * 5) == (None, None)
    assert studentized_ci([1.0]) == (None, None)


def test_studentized_ci_brackets_the_point_estimate():
    d = [0.1, 0.4, -0.2, 0.9, 0.3, 0.05]
    lo, hi = studentized_ci(d, n_resamples=2000, seed=0)
    assert lo < sum(d) / len(d) < hi


def test_studentized_ci_is_deterministic_for_a_seed():
    d = [0.1, 0.4, -0.2, 0.9, 0.3, 0.05]
    assert studentized_ci(d, seed=3) == studentized_ci(d, seed=3)


@pytest.mark.parametrize("n", [pytest.param(6, id="n6"), pytest.param(12, id="n12")])
def test_studentized_ci_recovers_nominal_coverage_where_percentile_fails(n):
    # Pre-screened against this repo's numpy BEFORE pinning (seed 20260803, 300
    # trials, B=800): n=6 -> studentized 0.953 vs percentile 0.843; n=12 ->
    # studentized 0.937 vs percentile 0.900. The percentile bootstrap under-covers
    # a labelled 95% interval at exactly the item counts this harness targets —
    # that gap is why the interval changed. Never widen the band to make this pass.
    rng = np.random.default_rng(20260803)
    studentized = percentile = 0
    for _ in range(300):
        d = rng.normal(0.3, 1.0, n)
        lo, hi = studentized_ci(d, n_resamples=800, seed=0)
        studentized += lo <= 0.3 <= hi
        lo_p, hi_p = bootstrap_ci(d, n_resamples=800, seed=0)
        percentile += lo_p <= 0.3 <= hi_p
    assert studentized / 300 >= 0.92
    assert studentized > percentile


def test_analyze_run_withholds_the_interval_when_every_diff_is_identical(store):
    run_id, cond = make_run(store, mode="rubric")
    for item in ("q1", "q2", "q3"):
        for name, value in (("control", 5), ("treatment", 6)):
            sid = add_ok_sample(store, run_id, cond[name], item)
            add_rubric_judgment(store, run_id, item, sample_id=sid, score=value)
    comparison = analyze_run(store, run_id).comparisons[0]
    assert comparison.diffs == (1.0, 1.0, 1.0)
    assert (comparison.ci_low, comparison.ci_high) == (None, None)
    assert comparison.ci_method == "studentized"


# --- M8/M2+M3: power planned at the family alpha, one estimator behind both rows ---


def test_required_items_grows_under_a_family_adjusted_alpha():
    # Pre-screened before pinning: [-0.5, 0.6, 1.4, 2.5] needs 13 items at the raw
    # alpha and 17 at 0.05/3. A multi-arm run whose verdict uses a Holm-corrected p
    # must plan at the same stringency, or it under-states the budget.
    d = [-0.5, 0.6, 1.4, 2.5]
    assert required_items_for_power(d) == 13
    assert required_items_for_power(d, alpha=0.05 / 3) == 17


def test_multi_arm_run_plans_power_at_the_corrected_alpha(store):
    run_id, cond = make_run(store, mode="rubric", variants=("a", "b", "c"))
    for item, scores in (("q1", (3, 5, 8)), ("q2", (4, 5, 7)), ("q3", (2, 6, 9))):
        for name, value in zip(("a", "b", "c"), scores, strict=True):
            sid = add_ok_sample(store, run_id, cond[name], item)
            add_rubric_judgment(store, run_id, item, sample_id=sid, score=value)
    result = analyze_run(store, run_id)
    assert len(result.comparisons) == 3  # C(3,2)
    assert all(c.power_alpha == pytest.approx(0.05 / 3) for c in result.comparisons)

    two_arm, cond2 = make_run(store, mode="rubric")
    for item in ("q1", "q2"):
        for name, value in (("control", 3), ("treatment", 6)):
            sid = add_ok_sample(store, two_arm, cond2[name], item)
            add_rubric_judgment(store, two_arm, item, sample_id=sid, score=value)
    single = analyze_run(store, two_arm).comparisons[0]
    assert single.power_alpha == 0.05  # a lone comparison is its own family


def test_allocation_ladder_is_monotone_and_matches_the_headline_power_row():
    # Pre-screened against the shipped code, which returned current/double/limit =
    # 349/175/2 next to a headline power row of 3 items: the ladder sat on the
    # clipped (inflated) variance scale, so the report could claim that doubling
    # the samples per item INCREASES the items needed.
    diffs_by_item = {"q1": (-1.9, 2.1), "q2": (-1.5, 2.5), "q3": (-1.8, 2.2), "q4": (-1.6, 2.4)}
    means = [sum(v) / len(v) for v in diffs_by_item.values()]
    decomposition = decompose_variance(diffs_by_item, mean_diff=sum(means) / len(means))
    assert decomposition.var_between == 0.0  # clipped: noise dominates
    assert decomposition.n_required_items_current == required_items_for_power(means) == 3
    assert decomposition.n_required_items_double <= decomposition.n_required_items_current
    assert decomposition.n_required_items_limit <= decomposition.n_required_items_double


# --- M9: audit-minor remediations -------------------------------------------------


def test_analyze_run_judge_block_missing_mode_raises_value_error(store):
    # Reachable only via hand-edited spec_json (the pydantic spec requires mode);
    # analyze_run must fail like audit_judge does, never with a bare KeyError.
    run_id = store.create_run("greeting-tone", {"judge": {"model": "judge-model"}})
    cid = store.add_condition(
        run_id,
        variant_name="control",
        system_prompt="",
        user_template="A: {input}",
        sampling={"model": "m"},
    )
    sid = add_ok_sample(store, run_id, cid, "q1")
    add_rubric_judgment(store, run_id, "q1", sample_id=sid, score=5)
    with pytest.raises(ValueError, match="missing 'mode'"):
        analyze_run(store, run_id)


def test_decompose_share_between_none_when_no_variance():
    # A perfect CRN null: every replicate diff identical, no variance anywhere.
    # A share of 1.0 here would claim item dominance over zero variance; None is
    # the honest answer (share_between is never rendered - programmatic API only).
    v = decompose_variance({"q1": [0.0, 0.0], "q2": [0.0, 0.0]}, mean_diff=0.0)
    assert v is not None
    assert v.share_between is None
    assert v.n_required_items_current is None
    assert v.recommendation == "more_items"
