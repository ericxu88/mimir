"""Tests for mimir.stats — paired bootstrap CI, sign-flip p-value, power (DESIGN.md §7).

Pins the two brief proofs: identical synthetic distributions => CI contains 0 and
p > 0.05; shifted distributions => CI excludes 0, brackets the true shift, p < 0.05.
Those assertions are probabilistic in general but deterministic here: all data seeds
and resampling seeds are pinned literals, pre-screened so every assertion holds with
wide margin. Never change a seed to make a test pass — that inverts the oracle.
"""

import numpy as np
import pytest

from mimir.clients.mock import MockClient
from mimir.runner import run_experiment
from mimir.spec import ExperimentSpec
from mimir.stats import (
    analyze_run,
    bootstrap_ci,
    pairwise_item_scores,
    required_items_for_power,
    rubric_item_scores,
    sign_flip_pvalue,
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
    assert (comparison.ci_low, comparison.ci_high) == bootstrap_ci([-1.0, 0.0], seed=0)
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
    assert (comparison.ci_low, comparison.ci_high) == (0.0, 0.0)
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
    assert (comparison.ci_low, comparison.ci_high) == bootstrap_ci([-1.0, 1.0, 3.0], seed=0)


def test_analyze_run_constant_diffs_additional_clamped_to_zero(store):
    # Every item's diff is exactly 2 -> required = 2 (floor), additional = max(0, 2-3) = 0.
    run_id, cond = make_run(store, mode="rubric")
    for item in ("q1", "q2", "q3"):
        for name, value in (("control", 3), ("treatment", 5)):
            sid = add_ok_sample(store, run_id, cond[name], item)
            add_rubric_judgment(store, run_id, item, sample_id=sid, score=value)
    comparison = analyze_run(store, run_id).comparisons[0]
    assert (comparison.ci_low, comparison.ci_high) == (2.0, 2.0)
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
    assert (comparison.ci_low, comparison.ci_high) == (0.0, 0.0)
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
    assert (comparison.ci_low, comparison.ci_high) == (-1.0, -1.0)
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
