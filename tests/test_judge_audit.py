"""Tests for mimir.judge_audit — flip test, length bias, cross-judge kappa (DESIGN.md §8).

Exact `==` oracles are hand-computed and depend on the pinned formulas: Pearson r =
Sxy / sqrt(Sxx * Syy) (single sqrt over the product) and OLS slope = Sxy / Sxx over
mean-centered sums — np.polyfit / np.corrcoef / two separate sqrts are all inexact on
these constructions. A failed oracle means fix the construction or the implementation,
never loosen the assertion.
"""

import pytest

from mimir.clients.mock import MockClient
from mimir.judge_audit import (
    audit_judge,
    cohens_kappa,
    length_bias_pairwise,
    length_bias_rubric,
    length_regression,
    position_bias,
)
from mimir.runner import run_experiment
from mimir.spec import ExperimentSpec
from mimir.stats import analyze_run
from mimir.store import Store


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "mimir.db")
    yield s
    s.close()


def make_run(store, *, mode="pairwise", variants=("control", "treatment")):
    """Run + conditions in declared order; spec dict carries the judge block audit_judge reads."""
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


def add_ok_sample(store, run_id, condition_id, item_id, sample_index=0, response_text=None):
    return store.add_sample(
        run_id=run_id,
        condition_id=condition_id,
        item_id=item_id,
        sample_index=sample_index,
        cache_key="k" * 64,
        request_json="{}",
        raw_response="{}",
        response_text=f"response for {item_id}" if response_text is None else response_text,
        latency_ms=1.0,
        input_tokens=1,
        output_tokens=1,
    )


def add_pair_judgment(store, run_id, item_id, *, order, verdict, first_sid, second_sid, error=None):
    # Encodes the presentation rule ONCE, relative to the run's DECLARED variant
    # order: 'ab' presents the declared-first variant's sample in position A, 'ba'
    # presents the swap (runner.py pairwise_task). first_sid/second_sid are the
    # declared-order samples, so runs declaring variants in a different order pass
    # their own declared-first sample as first_sid.
    a_sid, b_sid = (first_sid, second_sid) if order == "ab" else (second_sid, first_sid)
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


def add_pair(store, run_id, cond, item_id, *, sample_index=0, texts=(None, None)):
    """One sample per variant for an item; returns (first_sid, second_sid) in declared order."""
    names = list(cond)
    first = add_ok_sample(
        store, run_id, cond[names[0]], item_id, sample_index, response_text=texts[0]
    )
    second = add_ok_sample(
        store, run_id, cond[names[1]], item_id, sample_index, response_text=texts[1]
    )
    return first, second


# --- pure math: Cohen's kappa -----------------------------------------------------


def test_kappa_partial_agreement_hand_oracle():
    # n=8, agree on 6 (positions 1-3 X, 6-8 Y). Marginals: a = 4X/4Y, b = 4X/4Y.
    # po = 6/8 = 0.75, pe = 0.5*0.5 + 0.5*0.5 = 0.5, kappa = 0.25/0.5 = 0.5 (dyadic).
    a = ["X", "X", "X", "X", "Y", "Y", "Y", "Y"]
    b = ["X", "X", "X", "Y", "X", "Y", "Y", "Y"]
    assert cohens_kappa(a, b) == 0.5


def test_kappa_perfect_agreement_two_categories():
    # Identical AND non-constant: pe = 0.5, kappa = (1 - 0.5)/(1 - 0.5) = 1.0. A
    # constant identical labeling instead hits the pe == 1 degenerate path (below).
    labels = ["X", "Y", "X", "Y"]
    assert cohens_kappa(labels, labels) == 1.0


def test_kappa_three_categories_exact_expression():
    # po = 3/4; marginals a = (X 1/2, Y 1/4, Z 1/4), b = (X 1/4, Y 1/2, Z 1/4);
    # pe = 1/8 + 1/8 + 1/16 = 0.3125. All operands dyadic, one final division —
    # the oracle expression reproduces the exact float.
    a = ["X", "Y", "Z", "X"]
    b = ["X", "Y", "Z", "Y"]
    assert cohens_kappa(a, b) == (0.75 - 0.3125) / (1.0 - 0.3125)


def test_kappa_disjoint_constant_judges_is_zero():
    # po = 0 and pe = 1*0 + 0*1 = 0 -> kappa exactly 0.0 (chance-level, defined).
    assert cohens_kappa(["X"] * 4, ["Y"] * 4) == 0.0


def test_kappa_exactly_opposite_balanced_is_minus_one():
    # po = 0, marginals both 0.5/0.5 -> pe = 0.5, kappa = -0.5/0.5 = -1.0.
    a = ["X", "Y", "X", "Y"]
    b = ["Y", "X", "Y", "X"]
    assert cohens_kappa(a, b) == -1.0


def test_kappa_both_constant_same_category_not_estimable():
    # pe = 1: agreement is guaranteed by the marginals, kappa is 0/0 -> None.
    assert cohens_kappa(["X"] * 3, ["X"] * 3) is None


def test_kappa_rejects_length_mismatch():
    with pytest.raises(ValueError, match="equal length"):
        cohens_kappa(["X", "Y"], ["X"])


def test_kappa_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        cohens_kappa([], [])


# --- pure math: length regression (OLS slope + Pearson r) -------------------------


def test_length_regression_exact_dyadic_line():
    # y = x/8 over x with an exactly representable mean (20): slope = Sxy/Sxx = 0.125
    # and r = Sxy/sqrt(Sxx*Syy) = 1.0, both exact with the pinned formulas.
    slope, r = length_regression([8, 16, 24, 32], [1.0, 2.0, 3.0, 4.0])
    assert slope == 0.125
    assert r == 1.0


def test_length_regression_exact_anti_correlated_line():
    slope, r = length_regression([8, 16, 24, 32], [4.0, 3.0, 2.0, 1.0])
    assert slope == -0.125
    assert r == -1.0


def test_length_regression_balanced_step_indicator():
    # The pairwise length-bias shape: x = +/-d balanced, y = the win indicator.
    # Centered: sxy = 4*4 = 16, sxx = 256, syy = 1.0 -> slope = 1/16, r = 16/16 = 1.0.
    slope, r = length_regression([8, -8, 8, -8], [1.0, 0.0, 1.0, 0.0])
    assert slope == 0.0625
    assert r == 1.0


def test_length_regression_mixed_winner_zero_correlation():
    # Length-indifferent construction: which side is longer varies while wins split
    # evenly -> cross products +4, +4, -4, -4 cancel exactly. (The naive "indifferent"
    # data — constant y — hits the zero-variance path instead, tested below.)
    slope, r = length_regression([8, -8, -8, 8], [1.0, 0.0, 1.0, 0.0])
    assert slope == 0.0
    assert r == 0.0


def test_length_regression_constant_y_slope_defined_r_not():
    # A constant-score judge HAS a length slope (exactly 0.0); only r is undefined.
    slope, r = length_regression([8, 16, 24], [5.0, 5.0, 5.0])
    assert slope == 0.0
    assert r is None


def test_length_regression_constant_x_not_estimable():
    assert length_regression([8, 8, 8], [1.0, 2.0, 3.0]) == (None, None)


def test_length_regression_single_point_not_estimable():
    assert length_regression([8], [1.0]) == (None, None)


def test_length_regression_rejects_length_mismatch():
    with pytest.raises(ValueError, match="equal length"):
        length_regression([1, 2], [1.0])


def test_length_regression_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        length_regression([], [])


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="inf"),
        pytest.param(float("-inf"), id="-inf"),
    ],
)
def test_length_regression_rejects_non_finite(bad):
    with pytest.raises(ValueError, match="finite"):
        length_regression([1.0, bad], [1.0, 2.0])
    with pytest.raises(ValueError, match="finite"):
        length_regression([1.0, 2.0], [1.0, bad])


def test_length_regression_correlation_clamped_to_unit_interval():
    # Perfectly separated two-level data with a NON-representable mean: the raw
    # single-sqrt formula lands at -1.0000000000000002, mathematically impossible
    # for a correlation. The result must be clamped to [-1, 1] (review finding).
    _, r = length_regression([-723.0] * 3 + [-417.0] * 4, [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    assert r == -1.0


# --- extraction: position bias (flip test + position-A win rate) ------------------


def test_position_bias_always_a_judge_flips_every_pair(store):
    # The brief's canonical biased judge: verdict 'A' in BOTH orders means the
    # content-level winner (via the sample join) differs between the twins.
    run_id, cond = make_run(store)
    for item in ("q1", "q2", "q3", "q4"):
        first, second = add_pair(store, run_id, cond, item)
        add_pair_judgment(
            store, run_id, item, order="ab", verdict="A", first_sid=first, second_sid=second
        )
        add_pair_judgment(
            store, run_id, item, order="ba", verdict="A", first_sid=first, second_sid=second
        )
    result = position_bias(*tables(store, run_id))
    assert result.n_rows_used == 8
    assert result.n_rows_errored == 0
    assert result.n_pairs == 4
    assert result.n_pairs_dropped == 0
    assert result.n_flips == 4
    assert result.flip_rate == 1.0
    assert result.position_a_win_rate == 1.0


def test_position_bias_consistent_judge_no_flips(store):
    # Content winner is the declared-first variant in both orders: 'ab' -> A, 'ba' -> B.
    run_id, cond = make_run(store)
    for item in ("q1", "q2", "q3", "q4"):
        first, second = add_pair(store, run_id, cond, item)
        add_pair_judgment(
            store, run_id, item, order="ab", verdict="A", first_sid=first, second_sid=second
        )
        add_pair_judgment(
            store, run_id, item, order="ba", verdict="B", first_sid=first, second_sid=second
        )
    result = position_bias(*tables(store, run_id))
    assert result.flip_rate == 0.0
    assert result.n_flips == 0
    # Each pair contributes one 'A' and one 'B' row: 4*(1.0) + 4*(0.0) over 8 rows.
    assert result.position_a_win_rate == 0.5


def test_position_bias_tie_handling(store):
    # TIE/TIE is consistent; TIE vs a win is a flip.
    run_id, cond = make_run(store)
    first, second = add_pair(store, run_id, cond, "t1")
    add_pair_judgment(
        store, run_id, "t1", order="ab", verdict="TIE", first_sid=first, second_sid=second
    )
    add_pair_judgment(
        store, run_id, "t1", order="ba", verdict="TIE", first_sid=first, second_sid=second
    )
    first, second = add_pair(store, run_id, cond, "t2")
    add_pair_judgment(
        store, run_id, "t2", order="ab", verdict="TIE", first_sid=first, second_sid=second
    )
    add_pair_judgment(
        store, run_id, "t2", order="ba", verdict="A", first_sid=first, second_sid=second
    )
    result = position_bias(*tables(store, run_id))
    assert result.n_pairs == 2
    assert result.n_flips == 1
    assert result.flip_rate == 0.5
    # Rows: TIE, TIE, TIE, A -> (0.5 + 0.5 + 0.5 + 1.0) / 4.
    assert result.position_a_win_rate == 0.625


def test_position_bias_quarter_flip_rate(store):
    # 3 consistent pairs + 1 flipped pair -> 1/4 exactly.
    run_id, cond = make_run(store)
    for item, ba_verdict in (("q1", "B"), ("q2", "B"), ("q3", "B"), ("q4", "A")):
        first, second = add_pair(store, run_id, cond, item)
        add_pair_judgment(
            store, run_id, item, order="ab", verdict="A", first_sid=first, second_sid=second
        )
        add_pair_judgment(
            store, run_id, item, order="ba", verdict=ba_verdict, first_sid=first, second_sid=second
        )
    result = position_bias(*tables(store, run_id))
    assert result.flip_rate == 0.25
    assert result.position_a_win_rate == 0.625  # five 'A' rows of eight


def test_position_bias_pairs_keyed_by_replicate_not_item(store):
    # n_samples = 2: the same item has TWO pairs, keyed by sample ids. Grouping by
    # item_id would merge them into one 4-row group and drop it.
    run_id, cond = make_run(store)
    r0 = add_pair(store, run_id, cond, "q1", sample_index=0)
    r1 = add_pair(store, run_id, cond, "q1", sample_index=1)
    add_pair_judgment(
        store, run_id, "q1", order="ab", verdict="A", first_sid=r0[0], second_sid=r0[1]
    )
    add_pair_judgment(
        store, run_id, "q1", order="ba", verdict="B", first_sid=r0[0], second_sid=r0[1]
    )
    add_pair_judgment(
        store, run_id, "q1", order="ab", verdict="A", first_sid=r1[0], second_sid=r1[1]
    )
    add_pair_judgment(
        store, run_id, "q1", order="ba", verdict="A", first_sid=r1[0], second_sid=r1[1]
    )
    result = position_bias(*tables(store, run_id))
    assert result.n_pairs == 2
    assert result.n_pairs_dropped == 0
    assert result.flip_rate == 0.5


def test_position_bias_missing_twin_dropped_and_counted(store):
    # position_swap: false stores only 'ab' rows — those pairs cannot be flip-tested,
    # but their rows still feed position_a_win_rate.
    run_id, cond = make_run(store)
    first, second = add_pair(store, run_id, cond, "q1")
    add_pair_judgment(
        store, run_id, "q1", order="ab", verdict="A", first_sid=first, second_sid=second
    )
    add_pair_judgment(
        store, run_id, "q1", order="ba", verdict="B", first_sid=first, second_sid=second
    )
    first, second = add_pair(store, run_id, cond, "q2")
    add_pair_judgment(
        store, run_id, "q2", order="ab", verdict="A", first_sid=first, second_sid=second
    )
    result = position_bias(*tables(store, run_id))
    assert result.n_pairs == 1
    assert result.n_pairs_dropped == 1
    assert result.flip_rate == 0.0
    assert result.position_a_win_rate == 2 / 3  # rows A, B, A


def test_position_bias_errored_twin_drops_pair_counts_row(store):
    run_id, cond = make_run(store)
    first, second = add_pair(store, run_id, cond, "q1")
    add_pair_judgment(
        store, run_id, "q1", order="ab", verdict="A", first_sid=first, second_sid=second
    )
    add_pair_judgment(
        store,
        run_id,
        "q1",
        order="ba",
        verdict="B",
        first_sid=first,
        second_sid=second,
        error="ClientError: 500",
    )
    result = position_bias(*tables(store, run_id))
    assert result.n_rows_used == 1
    assert result.n_rows_errored == 1
    assert result.n_pairs == 0
    assert result.n_pairs_dropped == 1
    assert result.flip_rate is None
    assert result.position_a_win_rate == 1.0


def test_position_bias_no_swapped_pairs_flip_rate_none(store):
    run_id, cond = make_run(store)
    for item in ("q1", "q2"):
        first, second = add_pair(store, run_id, cond, item)
        add_pair_judgment(
            store, run_id, item, order="ab", verdict="A", first_sid=first, second_sid=second
        )
    result = position_bias(*tables(store, run_id))
    assert result.n_pairs == 0
    assert result.flip_rate is None
    assert result.position_a_win_rate == 1.0


def test_position_bias_drifted_rows_counted_errored_no_crash(store):
    # Rows the runner never writes must count as errored, not crash (M3 precedent):
    # unknown verdict, NULL verdict, NULL sample_b_id, foreign samples in either
    # position (verdict chosen so the outcome would dereference the foreign side),
    # and a self-pair (sample_a_id == sample_b_id).
    run_id, cond = make_run(store)
    first, second = add_pair(store, run_id, cond, "q1")
    add_pair_judgment(
        store, run_id, "q1", order="ab", verdict="A", first_sid=first, second_sid=second
    )
    add_pair_judgment(
        store, run_id, "q1", order="ba", verdict="B", first_sid=first, second_sid=second
    )
    other_run, other_cond = make_run(store)
    foreign_sid = add_ok_sample(store, other_run, other_cond["control"], "q1")
    for drifted in (
        {"sample_a_id": first, "sample_b_id": second, "verdict": "C"},
        {"sample_a_id": first, "sample_b_id": second, "verdict": None},
        {"sample_a_id": first, "sample_b_id": None, "verdict": "B"},
        {"sample_a_id": foreign_sid, "sample_b_id": second, "verdict": "A"},
        {"sample_a_id": first, "sample_b_id": foreign_sid, "verdict": "B"},
        {"sample_a_id": first, "sample_b_id": first, "verdict": "A"},
    ):
        store.add_judgment(
            run_id=run_id,
            item_id="q1",
            judge_model="judge-model",
            mode="pairwise",
            position_order="ab",
            cache_key="j" * 64,
            **drifted,
        )
    result = position_bias(*tables(store, run_id))
    assert result.n_rows_used == 2
    assert result.n_rows_errored == 6
    assert result.n_pairs == 1
    assert result.flip_rate == 0.0
    # The same drifted rows must not dilute the length regression either.
    length = length_bias_pairwise(*tables(store, run_id))
    assert length.n_rows_used == 2
    assert length.n_rows_errored == 6


def test_position_bias_same_order_duplicate_groups_dropped(store):
    # Hand-inserted duplicates the runner can never write: a group of two usable
    # rows presenting the SAME order, and a full twin group plus one extra row.
    # Both must be dropped-and-counted, never evaluated (docstring promise).
    run_id, cond = make_run(store)
    first, second = add_pair(store, run_id, cond, "q1")
    add_pair_judgment(
        store, run_id, "q1", order="ab", verdict="A", first_sid=first, second_sid=second
    )
    add_pair_judgment(
        store, run_id, "q1", order="ba", verdict="B", first_sid=first, second_sid=second
    )
    first, second = add_pair(store, run_id, cond, "q2")
    for _ in range(2):  # two usable rows, same order, same pair
        add_pair_judgment(
            store, run_id, "q2", order="ab", verdict="A", first_sid=first, second_sid=second
        )
    first, second = add_pair(store, run_id, cond, "q3")
    add_pair_judgment(
        store, run_id, "q3", order="ab", verdict="A", first_sid=first, second_sid=second
    )
    add_pair_judgment(
        store, run_id, "q3", order="ba", verdict="B", first_sid=first, second_sid=second
    )
    add_pair_judgment(  # third usable row in the q3 group
        store, run_id, "q3", order="ab", verdict="B", first_sid=first, second_sid=second
    )
    result = position_bias(*tables(store, run_id))
    assert result.n_rows_used == 7
    assert result.n_pairs == 1
    assert result.n_pairs_dropped == 2
    assert result.flip_rate == 0.0


def test_position_bias_all_rows_errored_rates_none(store):
    run_id, cond = make_run(store)
    first, second = add_pair(store, run_id, cond, "q1")
    add_pair_judgment(
        store,
        run_id,
        "q1",
        order="ab",
        verdict="A",
        first_sid=first,
        second_sid=second,
        error="boom",
    )
    result = position_bias(*tables(store, run_id))
    assert result.n_rows_used == 0
    assert result.flip_rate is None
    assert result.position_a_win_rate is None


def test_position_bias_requires_exactly_two_conditions(store):
    run_id, _ = make_run(store, variants=("a", "b", "c"))
    with pytest.raises(ValueError, match="two"):
        position_bias(*tables(store, run_id))


# --- extraction: length bias ------------------------------------------------------


def test_length_bias_pairwise_longer_wins_perfect_correlation(store):
    # Both items: declared-first is 8 chars, declared-second 16. Longer always wins,
    # so each pair yields the points (-8, 0) and (+8, 1) -> r exactly 1.0.
    run_id, cond = make_run(store)
    for item in ("q1", "q2"):
        first, second = add_pair(store, run_id, cond, item, texts=("c" * 8, "t" * 16))
        add_pair_judgment(
            store, run_id, item, order="ab", verdict="B", first_sid=first, second_sid=second
        )
        add_pair_judgment(
            store, run_id, item, order="ba", verdict="A", first_sid=first, second_sid=second
        )
    result = length_bias_pairwise(*tables(store, run_id))
    assert result.n_points == 4
    assert result.slope == 0.0625  # win-indicator gain per char: 16/256
    assert result.correlation == 1.0


def test_length_bias_pairwise_indifferent_judge_zero_correlation(store):
    # Which variant is longer varies across items while the judge consistently
    # prefers the declared-first variant's CONTENT: cross products cancel exactly.
    # (A judge with constant verdicts would hit the zero-variance-y path instead.)
    run_id, cond = make_run(store)
    for item, texts in (("q1", ("c" * 8, "t" * 16)), ("q2", ("c" * 16, "t" * 8))):
        first, second = add_pair(store, run_id, cond, item, texts=texts)
        add_pair_judgment(
            store, run_id, item, order="ab", verdict="A", first_sid=first, second_sid=second
        )
        add_pair_judgment(
            store, run_id, item, order="ba", verdict="B", first_sid=first, second_sid=second
        )
    result = length_bias_pairwise(*tables(store, run_id))
    assert result.slope == 0.0
    assert result.correlation == 0.0


def test_length_bias_pairwise_null_response_text_point_skipped(store):
    run_id, cond = make_run(store)
    first, second = add_pair(store, run_id, cond, "q1", texts=("c" * 8, "t" * 16))
    add_pair_judgment(
        store, run_id, "q1", order="ab", verdict="B", first_sid=first, second_sid=second
    )
    add_pair_judgment(
        store, run_id, "q1", order="ba", verdict="A", first_sid=first, second_sid=second
    )
    # A judged sample with NULL response_text cannot happen through the runner
    # (errored samples are never judged) but must not crash the length join.
    null_first = store.add_sample(
        run_id=run_id,
        condition_id=cond["control"],
        item_id="q2",
        sample_index=0,
        cache_key="k" * 64,
        request_json="{}",
    )
    null_second = add_ok_sample(store, run_id, cond["treatment"], "q2")
    add_pair_judgment(
        store, run_id, "q2", order="ab", verdict="A", first_sid=null_first, second_sid=null_second
    )
    # Same shape with the NULL-text sample in presented position B.
    ok_first = add_ok_sample(store, run_id, cond["control"], "q3")
    null_second = store.add_sample(
        run_id=run_id,
        condition_id=cond["treatment"],
        item_id="q3",
        sample_index=0,
        cache_key="k" * 64,
        request_json="{}",
    )
    add_pair_judgment(
        store, run_id, "q3", order="ab", verdict="B", first_sid=ok_first, second_sid=null_second
    )
    result = length_bias_pairwise(*tables(store, run_id))
    assert result.n_rows_used == 4
    assert result.n_points == 2
    assert result.correlation == 1.0


def test_length_bias_pairwise_requires_exactly_two_conditions(store):
    run_id, _ = make_run(store, variants=("a", "b", "c"))
    with pytest.raises(ValueError, match="two"):
        length_bias_pairwise(*tables(store, run_id))


def test_length_bias_rubric_exact_line(store):
    # score = len/8 over lengths with an exactly representable mean.
    run_id, cond = make_run(store, mode="rubric", variants=("solo",))
    for item, length, score in (("q1", 8, 1), ("q2", 16, 2), ("q3", 24, 3), ("q4", 32, 4)):
        sid = add_ok_sample(store, run_id, cond["solo"], item, response_text="x" * length)
        add_rubric_judgment(store, run_id, item, sample_id=sid, score=score)
    result = length_bias_rubric(*tables(store, run_id))
    assert result.n_points == 4
    assert result.slope == 0.125
    assert result.correlation == 1.0


def test_length_bias_rubric_constant_scores_slope_zero_r_none(store):
    run_id, cond = make_run(store, mode="rubric", variants=("solo",))
    for item, length in (("q1", 8), ("q2", 16), ("q3", 24)):
        sid = add_ok_sample(store, run_id, cond["solo"], item, response_text="x" * length)
        add_rubric_judgment(store, run_id, item, sample_id=sid, score=5)
    result = length_bias_rubric(*tables(store, run_id))
    assert result.slope == 0.0
    assert result.correlation is None


def test_length_bias_rubric_errored_and_drifted_rows_skipped(store):
    run_id, cond = make_run(store, mode="rubric", variants=("solo",))
    for item, length, score in (("q1", 8, 1), ("q2", 16, 2)):
        sid = add_ok_sample(store, run_id, cond["solo"], item, response_text="x" * length)
        add_rubric_judgment(store, run_id, item, sample_id=sid, score=score)
    sid = add_ok_sample(store, run_id, cond["solo"], "q3", response_text="x" * 24)
    add_rubric_judgment(store, run_id, "q3", sample_id=sid, score=3, error="boom")
    # Drifted: score NULL without error.
    sid = add_ok_sample(store, run_id, cond["solo"], "q4", response_text="x" * 32)
    store.add_judgment(
        run_id=run_id,
        item_id="q4",
        judge_model="judge-model",
        mode="rubric",
        sample_a_id=sid,
        cache_key="j" * 64,
    )
    # Drifted: a judgment referencing another run's sample.
    other_run, other_cond = make_run(store, mode="rubric", variants=("solo",))
    foreign_sid = add_ok_sample(store, other_run, other_cond["solo"], "q1")
    add_rubric_judgment(store, run_id, "q5", sample_id=foreign_sid, score=5)
    # Drifted: usable score on a sample with NULL response_text (used, not a point).
    null_sid = store.add_sample(
        run_id=run_id,
        condition_id=cond["solo"],
        item_id="q6",
        sample_index=0,
        cache_key="k" * 64,
        request_json="{}",
    )
    add_rubric_judgment(store, run_id, "q6", sample_id=null_sid, score=5)
    result = length_bias_rubric(*tables(store, run_id))
    assert result.n_rows_used == 3
    assert result.n_rows_errored == 3
    assert result.n_points == 2


# --- orchestrator: audit_judge ----------------------------------------------------


def test_audit_judge_pairwise_card_assembly(store):
    # Default factory texts are identical within an item -> every length diff is 0
    # -> length bias not estimable; the card must say so instead of crashing.
    run_id, cond = make_run(store)
    for item in ("q1", "q2"):
        first, second = add_pair(store, run_id, cond, item)
        add_pair_judgment(
            store, run_id, item, order="ab", verdict="A", first_sid=first, second_sid=second
        )
        add_pair_judgment(
            store, run_id, item, order="ba", verdict="B", first_sid=first, second_sid=second
        )
    card = audit_judge(store, run_id)
    assert card.run_id == run_id
    assert card.judge_model == "judge-model"
    assert card.mode == "pairwise"
    assert card.n_judgments_used == 4
    assert card.n_judgments_errored == 0
    assert card.n_pairs == 2
    assert card.n_pairs_dropped == 0
    assert card.flip_rate == 0.0
    assert card.position_a_win_rate == 0.5
    assert card.n_length_points == 4
    assert card.length_slope is None
    assert card.length_correlation is None
    assert card.compare_run_id is None
    assert card.kappa is None
    assert card.kappa_n is None
    # Exact tuple pin: fixes note order AND wording (the slope is a separate field
    # and may exist when only the correlation is inestimable).
    assert card.notes == (
        "length correlation not estimable: zero variance or fewer than two points",
    )


def test_audit_judge_rubric_card(store):
    run_id, cond = make_run(store, mode="rubric", variants=("solo",))
    for item, length, score in (("q1", 8, 1), ("q2", 16, 2), ("q3", 24, 3), ("q4", 32, 4)):
        sid = add_ok_sample(store, run_id, cond["solo"], item, response_text="x" * length)
        add_rubric_judgment(store, run_id, item, sample_id=sid, score=score)
    card = audit_judge(store, run_id)
    assert card.mode == "rubric"
    assert card.n_judgments_used == 4
    assert card.n_pairs is None
    assert card.n_pairs_dropped is None
    assert card.flip_rate is None
    assert card.position_a_win_rate is None
    assert card.n_length_points == 4
    assert card.length_slope == 0.125
    assert card.length_correlation == 1.0
    assert card.notes == ("position bias flip test requires pairwise mode",)


def test_audit_judge_swap_off_notes(store):
    run_id, cond = make_run(store)
    for item in ("q1", "q2"):
        first, second = add_pair(store, run_id, cond, item)
        add_pair_judgment(
            store, run_id, item, order="ab", verdict="A", first_sid=first, second_sid=second
        )
    card = audit_judge(store, run_id)
    assert card.flip_rate is None
    assert card.position_a_win_rate == 1.0
    assert any("flip test skipped" in note for note in card.notes)
    assert any("conflates position and content" in note for note in card.notes)


def test_audit_judge_all_rows_errored_card(store):
    # Every judgment errored: the card must say "no usable judgments", NOT hint at
    # position_swap (review finding: the hint was misleading here), and must not
    # emit the conflates note (there is no win rate to conflate).
    run_id, cond = make_run(store)
    first, second = add_pair(store, run_id, cond, "q1")
    for order in ("ab", "ba"):
        add_pair_judgment(
            store,
            run_id,
            "q1",
            order=order,
            verdict="A",
            first_sid=first,
            second_sid=second,
            error="boom",
        )
    card = audit_judge(store, run_id)
    assert card.n_judgments_used == 0
    assert card.n_judgments_errored == 2
    assert card.flip_rate is None
    assert card.position_a_win_rate is None
    assert card.notes == (
        "no usable pairwise judgments",
        "length correlation not estimable: zero variance or fewer than two points",
    )


def test_audit_judge_judge_block_missing_mode_or_model(store):
    # Hand-edited spec_json without 'mode' or 'model' must raise the module's
    # ValueError family, not KeyError (review finding).
    no_mode = store.create_run("weird", {"name": "weird", "judge": {"model": "j"}})
    with pytest.raises(ValueError, match="missing"):
        audit_judge(store, no_mode)
    no_model = store.create_run("weird", {"name": "weird", "judge": {"mode": "pairwise"}})
    with pytest.raises(ValueError, match="missing"):
        audit_judge(store, no_model)


def test_audit_judge_unknown_run(store):
    with pytest.raises(ValueError, match="not found"):
        audit_judge(store, "nope")


def test_audit_judge_no_judge_configured(store):
    run_id = store.create_run("bare", {"name": "bare"})
    with pytest.raises(ValueError, match="no judge configured"):
        audit_judge(store, run_id)


def test_audit_judge_no_judgments(store):
    run_id, _ = make_run(store)
    with pytest.raises(ValueError, match="no judgments"):
        audit_judge(store, run_id)


def test_audit_judge_unknown_mode(store):
    run_id = store.create_run("weird", {"name": "weird", "judge": {"model": "j", "mode": "vibes"}})
    condition_id = store.add_condition(
        run_id,
        variant_name="solo",
        system_prompt="",
        user_template="Answer: {input}",
        sampling={"model": "claude-haiku-4-5-20251001"},
    )
    sid = add_ok_sample(store, run_id, condition_id, "q1")
    store.add_judgment(
        run_id=run_id,
        item_id="q1",
        judge_model="j",
        mode="vibes",
        sample_a_id=sid,
        cache_key="j" * 64,
        score=5.0,
    )
    with pytest.raises(ValueError, match="unknown judge mode"):
        audit_judge(store, run_id)


# --- orchestrator: cross-judge kappa ----------------------------------------------


def make_judged_run(store, item_winners, *, variants=("control", "treatment")):
    """A pairwise run where each item's content winner is fixed across both orders.

    item_winners maps item_id -> declared-FIRST variant wins (True) or declared-second
    (False). Content winner w means 'ab' verdict is A iff w is first, 'ba' the inverse.
    """
    run_id, cond = make_run(store, variants=variants)
    for item, first_wins in item_winners.items():
        first, second = add_pair(store, run_id, cond, item)
        add_pair_judgment(
            store,
            run_id,
            item,
            order="ab",
            verdict="A" if first_wins else "B",
            first_sid=first,
            second_sid=second,
        )
        add_pair_judgment(
            store,
            run_id,
            item,
            order="ba",
            verdict="B" if first_wins else "A",
            first_sid=first,
            second_sid=second,
        )
    return run_id


def test_audit_judge_kappa_identical_mixed_outcomes(store):
    # Outcomes vary across units (control wins q1, treatment wins q2) and the two
    # runs agree everywhere: po = 1, pe = 0.5 -> kappa exactly 1.0.
    run_a = make_judged_run(store, {"q1": True, "q2": False})
    run_b = make_judged_run(store, {"q1": True, "q2": False})
    card = audit_judge(store, run_a, compare_run_id=run_b)
    assert card.kappa == 1.0
    assert card.kappa_n == 4
    assert card.compare_run_id == run_b


def test_audit_judge_kappa_partial_agreement_hand_oracle(store):
    # run_a outcomes: control, control, treatment, treatment (4 units).
    # run_b flips exactly one treatment unit to control: po = 3/4, marginals
    # (1/2, 1/2) x (3/4, 1/4) -> pe = 1/2, kappa = (3/4 - 1/2)/(1/2) = 0.5.
    run_a = make_judged_run(store, {"q1": True, "q2": False})
    run_b_id, cond = make_run(store)
    first, second = add_pair(store, run_b_id, cond, "q1")
    add_pair_judgment(
        store, run_b_id, "q1", order="ab", verdict="A", first_sid=first, second_sid=second
    )
    add_pair_judgment(
        store, run_b_id, "q1", order="ba", verdict="B", first_sid=first, second_sid=second
    )
    first, second = add_pair(store, run_b_id, cond, "q2")
    add_pair_judgment(
        store, run_b_id, "q2", order="ab", verdict="B", first_sid=first, second_sid=second
    )
    add_pair_judgment(  # disagrees with run_a on this unit: control instead of treatment
        store, run_b_id, "q2", order="ba", verdict="B", first_sid=first, second_sid=second
    )
    card = audit_judge(store, run_a, compare_run_id=run_b_id)
    assert card.kappa == 0.5
    assert card.kappa_n == 4


def test_audit_judge_kappa_aligns_across_declaration_orders(store):
    # Same variant names declared in OPPOSITE orders; both judges prefer control's
    # content. Units are keyed by the presented-A variant (sample join), not by the
    # declaration-relative position_order, so agreement is perfect: kappa == 1.0.
    # A position_order-keyed alignment would report kappa == -1.0 here.
    run_a = make_judged_run(store, {"q1": True, "q2": True})
    run_b = make_judged_run(store, {"q1": False, "q2": False}, variants=("treatment", "control"))
    card = audit_judge(store, run_a, compare_run_id=run_b)
    # Both judges are constant (control always wins), so kappa is 0/0 -> None + note.
    assert card.kappa is None
    assert any("kappa not estimable" in note for note in card.notes)


def test_audit_judge_kappa_aligns_across_declaration_orders_mixed(store):
    # Same as above with non-constant outcomes so kappa is estimable: control wins
    # q1, treatment wins q2, in both runs, under opposite declaration orders.
    run_a = make_judged_run(store, {"q1": True, "q2": False})
    run_b = make_judged_run(store, {"q1": False, "q2": True}, variants=("treatment", "control"))
    card = audit_judge(store, run_a, compare_run_id=run_b)
    assert card.kappa == 1.0
    assert card.kappa_n == 4


def test_audit_judge_kappa_position_consistent_judge_across_declaration_orders(store):
    # THE discriminator for the unit-keying rule (found by mutation review): a judge
    # that always answers 'A' in BOTH orders, audited across runs whose variants are
    # declared in OPPOSITE orders. Keying units by the presented-A variant aligns
    # identical presentations -> kappa == 1.0. Keying by position_order would align
    # opposite presentations and report kappa == -1.0 for two byte-identical judges.
    runs = []
    for variants in (("control", "treatment"), ("treatment", "control")):
        run_id, cond = make_run(store, variants=variants)
        for item in ("q1", "q2"):
            first, second = add_pair(store, run_id, cond, item)
            for order in ("ab", "ba"):
                add_pair_judgment(
                    store,
                    run_id,
                    item,
                    order=order,
                    verdict="A",
                    first_sid=first,
                    second_sid=second,
                )
        runs.append(run_id)
    card = audit_judge(store, runs[0], compare_run_id=runs[1])
    assert card.kappa == 1.0
    assert card.kappa_n == 4


def test_audit_judge_kappa_conflicting_duplicate_unit_dropped(store):
    # A hand-inserted duplicate row whose outcome CONTRADICTS its unit's legitimate
    # row must remove that unit from the kappa alignment (drop-and-count policy,
    # review finding: last-write-wins silently flipped kappa 1.0 -> 0.5).
    run_a, cond = make_run(store)
    sids = {}
    for item, first_wins in (("q1", True), ("q2", False)):
        first, second = add_pair(store, run_a, cond, item)
        sids[item] = (first, second)
        add_pair_judgment(
            store,
            run_a,
            item,
            order="ab",
            verdict="A" if first_wins else "B",
            first_sid=first,
            second_sid=second,
        )
        add_pair_judgment(
            store,
            run_a,
            item,
            order="ba",
            verdict="B" if first_wins else "A",
            first_sid=first,
            second_sid=second,
        )
    # Conflicting duplicate for q1's 'ab' unit: legit outcome control, this says treatment.
    add_pair_judgment(
        store,
        run_a,
        "q1",
        order="ab",
        verdict="B",
        first_sid=sids["q1"][0],
        second_sid=sids["q1"][1],
    )
    run_b = make_judged_run(store, {"q1": True, "q2": False})
    card = audit_judge(store, run_a, compare_run_id=run_b)
    # The conflicted unit is excluded: 3 shared units, all agreeing and non-constant.
    assert card.kappa_n == 3
    assert card.kappa == 1.0
    # The flip test independently drops the now-3-row q1 group.
    assert card.n_pairs == 1
    assert card.n_pairs_dropped == 1


def test_audit_judge_kappa_tie_outcomes_hand_oracle(store):
    # Ties flow into kappa as their own category. run_a: q1 TIE/TIE, q2 control wins
    # both orders. run_b agrees except q2's 'ab' unit is TIE. Aligned labels:
    # a = [tie, tie, control, control], b = [tie, tie, tie, control] -> po = 3/4,
    # marginals (1/2, 1/2) x (3/4, 1/4) -> pe = 1/2, kappa = 0.5 exactly.
    run_ids = []
    for q2_ab_verdict in ("A", "TIE"):
        run_id, cond = make_run(store)
        first, second = add_pair(store, run_id, cond, "q1")
        for order in ("ab", "ba"):
            add_pair_judgment(
                store,
                run_id,
                "q1",
                order=order,
                verdict="TIE",
                first_sid=first,
                second_sid=second,
            )
        first, second = add_pair(store, run_id, cond, "q2")
        add_pair_judgment(
            store,
            run_id,
            "q2",
            order="ab",
            verdict=q2_ab_verdict,
            first_sid=first,
            second_sid=second,
        )
        add_pair_judgment(
            store, run_id, "q2", order="ba", verdict="B", first_sid=first, second_sid=second
        )
        run_ids.append(run_id)
    card = audit_judge(store, run_ids[0], compare_run_id=run_ids[1])
    assert card.kappa == 0.5
    assert card.kappa_n == 4


def test_audit_judge_kappa_compare_run_without_judge(store):
    run_a = make_judged_run(store, {"q1": True})
    bare = store.create_run("bare", {"name": "bare"})
    with pytest.raises(ValueError, match="no judge configured"):
        audit_judge(store, run_a, compare_run_id=bare)


def test_audit_judge_kappa_constant_same_outcome_not_estimable(store):
    run_a = make_judged_run(store, {"q1": True, "q2": True})
    run_b = make_judged_run(store, {"q1": True, "q2": True})
    card = audit_judge(store, run_a, compare_run_id=run_b)
    assert card.kappa is None
    assert card.kappa_n == 4
    assert any("kappa not estimable" in note for note in card.notes)


def test_audit_judge_kappa_variant_names_must_match(store):
    run_a = make_judged_run(store, {"q1": True})
    run_b = make_judged_run(store, {"q1": True}, variants=("control", "friendly"))
    with pytest.raises(ValueError, match="not comparable"):
        audit_judge(store, run_a, compare_run_id=run_b)


def test_audit_judge_kappa_no_shared_units(store):
    run_a = make_judged_run(store, {"q1": True})
    run_b = make_judged_run(store, {"zz": True})
    with pytest.raises(ValueError, match="share no judged units"):
        audit_judge(store, run_a, compare_run_id=run_b)


def test_audit_judge_kappa_compare_run_not_found(store):
    run_a = make_judged_run(store, {"q1": True})
    with pytest.raises(ValueError, match="not found"):
        audit_judge(store, run_a, compare_run_id="nope")


def test_audit_judge_kappa_requires_pairwise_primary(store):
    run_id, cond = make_run(store, mode="rubric", variants=("solo",))
    sid = add_ok_sample(store, run_id, cond["solo"], "q1")
    add_rubric_judgment(store, run_id, "q1", sample_id=sid, score=5)
    other = make_judged_run(store, {"q1": True})
    with pytest.raises(ValueError, match="pairwise mode in both"):
        audit_judge(store, run_id, compare_run_id=other)


def test_audit_judge_kappa_requires_pairwise_compare(store):
    run_a = make_judged_run(store, {"q1": True})
    rubric_id, cond = make_run(store, mode="rubric", variants=("solo",))
    sid = add_ok_sample(store, rubric_id, cond["solo"], "q1")
    add_rubric_judgment(store, rubric_id, "q1", sample_id=sid, score=5)
    with pytest.raises(ValueError, match="pairwise mode in both"):
        audit_judge(store, run_a, compare_run_id=rubric_id)


# --- end to end: audit of runner-written rows (anti-circularity) ------------------


def _pairwise_spec(judge_model="judge-model"):
    return ExperimentSpec.model_validate(
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
            "judge": {"model": judge_model, "mode": "pairwise"},
        }
    )


@pytest.mark.anyio
async def test_audit_judge_end_to_end_always_a_judge(store):
    # THE brief-mandated proof, on rows the real runner wrote: an always-prefers-
    # position-A judge must yield flip_rate ~ 1.0 (here exactly 1.0). stats.analyze_run
    # on the same rows reports "no difference" — the report card is what tells that
    # degenerate agreement apart from a real tie.
    client = MockClient()
    client.add_rule(lambda request: request.model == "judge-model", "A")
    run_id = await run_experiment(_pairwise_spec(), store, client)
    card = audit_judge(store, run_id)
    assert card.n_judgments_used == 4
    assert card.n_pairs == 2
    assert card.flip_rate == 1.0
    assert card.position_a_win_rate == 1.0
    # MockClient default texts are all 21 chars -> zero length variance.
    assert card.length_correlation is None
    comparison = analyze_run(store, run_id).comparisons[0]
    assert comparison.mean_diff == 0.0
    assert comparison.p_value == 1.0


@pytest.mark.anyio
async def test_audit_judge_end_to_end_content_consistent_judge(store):
    # The judge votes for whichever PRESENTED position holds control's canned text
    # (the M3 anti-circularity rule set, all four rules): no position bias. Control's
    # text is one char shorter than friendly's, so the content preference shows up
    # as a perfect NEGATIVE length correlation — pinning the presented-position
    # length join through real runner rows.
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
    run_id = await run_experiment(_pairwise_spec(), store, client)
    card = audit_judge(store, run_id)
    assert card.flip_rate == 0.0
    assert card.position_a_win_rate == 0.5
    # Points per pair: (16-17, win) and (17-16, loss) -> slope -0.5, r -1.0 exactly.
    assert card.n_length_points == 4
    assert card.length_slope == -0.5
    assert card.length_correlation == -1.0


@pytest.mark.anyio
async def test_audit_judge_end_to_end_longer_wins_judge(store):
    # A genuinely length-biased judge: parses both responses out of the rendered
    # default template and votes for the longer one. The B-side split must also cut
    # the template's fixed trailing question, or the comparison inverts for any
    # length difference below the tail length.
    def longer_a(request):
        if request.model != "judge-model":
            return False
        body = request.user.split("Response A:\n", 1)[1]
        response_a, rest = body.split("\n\nResponse B:\n", 1)
        response_b = rest.split("\n\nWhich response is better?", 1)[0]
        return len(response_a) > len(response_b)

    client = MockClient()
    client.add_rule(longer_a, "A")
    client.add_rule(lambda request: request.model == "judge-model", "B")
    client.add_rule(lambda request: request.system == "You are helpful.", "S" * 8)
    client.add_rule(lambda request: request.system == "You are warm.", "L" * 16)
    run_id = await run_experiment(_pairwise_spec(), store, client)
    card = audit_judge(store, run_id)
    # Longer (friendly) wins in BOTH orders: content-consistent, so no flips —
    # but the length preference is perfectly visible: (-8, 0) and (+8, 1) per pair.
    assert card.flip_rate == 0.0
    assert card.position_a_win_rate == 0.5
    assert card.length_slope == 0.0625
    assert card.length_correlation == 1.0


@pytest.mark.anyio
async def test_audit_judge_end_to_end_cross_judge_kappa(store):
    # Two runs in ONE store, differing only in judge model. Sample payloads are
    # identical -> run 2's samples are cache hits with byte-identical texts (exactly
    # what makes the runs comparable); judge payloads differ by model -> the judge-b
    # rules fire. always-A vs always-B invert every content outcome: kappa == -1.0
    # (exact because MockClient never errors, so the ab/ba unit counts stay balanced).
    client = MockClient()
    client.add_rule(lambda request: request.model == "judge-a", "A")
    client.add_rule(lambda request: request.model == "judge-b", "B")
    run_a = await run_experiment(_pairwise_spec(judge_model="judge-a"), store, client)
    run_b = await run_experiment(_pairwise_spec(judge_model="judge-b"), store, client)
    card = audit_judge(store, run_a, compare_run_id=run_b)
    assert card.compare_run_id == run_b
    assert card.kappa_n == 4
    assert card.kappa == -1.0
    # A run agrees perfectly with itself (outcomes alternate, so kappa is estimable).
    assert audit_judge(store, run_a, compare_run_id=run_a).kappa == 1.0
