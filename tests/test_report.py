"""Tests for mimir.report — terminal renderers, SVG histograms, static HTML (DESIGN.md §9).

All geometry and formatting oracles are exact strings hand-computed from the pinned
module constants (slot 24px, bar 22px, plot height 80, top 8, margin 8, integer
height mapping count*80//max). A failed golden means fix the renderer or the
construction, never loosen the oracle.
"""

import pytest

from mimir.judge_audit import JudgeReportCard
from mimir.report import (
    histogram_bins,
    render_analysis_text,
    render_audit_text,
    render_html,
    svg_histogram,
)
from mimir.stats import AnalysisResult, Comparison


def make_comparison(**overrides):
    fields = {
        "variant_a": "control",
        "variant_b": "friendly",
        "item_ids": ("q1", "q2", "q3", "q4"),
        "diffs": (0.25, 0.0, 0.25, 0.5),
        "n_items": 4,
        "n_items_dropped": 0,
        "mean_a": 0.375,
        "mean_b": 0.625,
        "mean_diff": 0.25,
        "ci_low": 0.125,
        "ci_high": 0.375,
        "ci_level": 0.95,
        "p_value": 0.125,
        "p_method": "exhaustive",
        "n_resamples": 10_000,
        "n_permutations": 16,
        "alpha": 0.05,
        "target_power": 0.8,
        "n_required_items": 23,
        "n_additional_items": 19,
        "seed": 0,
    }
    fields.update(overrides)
    return Comparison(**fields)


def make_result(**overrides):
    fields = {
        "run_id": "20260801-183042-a1b2",
        "experiment_name": "greeting-tone",
        "mode": "pairwise",
        "scores": {
            "control": {"q1": 0.25, "q2": 0.5, "q3": 0.25, "q4": 0.5},
            "friendly": {"q1": 0.5, "q2": 0.5, "q3": 0.5, "q4": 1.0},
        },
        "comparisons": [make_comparison()],
        "n_items": 4,
        "n_judgments_used": 16,
        "n_judgments_errored": 0,
    }
    fields.update(overrides)
    return AnalysisResult(**fields)


def make_card(**overrides):
    fields = {
        "run_id": "20260801-183042-a1b2",
        "judge_model": "judge-model",
        "mode": "pairwise",
        "n_judgments_used": 4,
        "n_judgments_errored": 0,
        "n_pairs": 2,
        "n_pairs_dropped": 0,
        "flip_rate": 1.0,
        "position_a_win_rate": 1.0,
        "n_length_points": 4,
        "length_slope": 0.0625,
        "length_correlation": 1.0,
        "compare_run_id": "20260801-190000-cd34",
        "kappa": -1.0,
        "kappa_n": 4,
        "notes": (),
    }
    fields.update(overrides)
    return JudgeReportCard(**fields)


# --- histogram binning ------------------------------------------------------------


def test_histogram_bins_pairwise_range():
    # 0.0 -> bin 0, 0.25 -> bin 2, 0.5 -> bin 5 (twice), 1.0 clamps into bin 9.
    bins = histogram_bins([0.0, 0.25, 0.5, 0.5, 1.0], lo=0.0, hi=1.0, n_bins=10)
    assert bins == [1, 0, 1, 0, 0, 2, 0, 0, 0, 1]


def test_histogram_bins_rubric_range():
    # Unit-width bins over [1, 10]: 1.0 -> bin 0, 5.5 -> bin 4, 10.0 clamps into bin 8.
    bins = histogram_bins([1.0, 5.5, 10.0], lo=1.0, hi=10.0, n_bins=9)
    assert bins == [1, 0, 0, 0, 1, 0, 0, 0, 1]


def test_histogram_bins_out_of_range_clamps_to_end_bins():
    # Drifted scores never crash a report: they land in the end bins.
    bins = histogram_bins([-0.5, 1.5], lo=0.0, hi=1.0, n_bins=10)
    assert bins == [1, 0, 0, 0, 0, 0, 0, 0, 0, 1]


def test_histogram_bins_rejects_empty():
    with pytest.raises(ValueError, match="no values"):
        histogram_bins([], lo=0.0, hi=1.0, n_bins=10)


# --- SVG histogram ----------------------------------------------------------------


def test_svg_histogram_exact_geometry():
    # Bins: [1, 0, 0, 0, 0, 2, 0, ...]; max = 2. Bin 0: h = max(1, 1*80//2) = 40,
    # x = 8 + 0*24 + 1 = 9, y = 88 - 40 = 48. Bin 5: h = 80, x = 129, y = 8.
    svg = svg_histogram([0.05, 0.55, 0.55], lo=0.0, hi=1.0, n_bins=10, fill="var(--series-1)")
    assert svg.count("<rect") == 2  # zero-count bins emit no rect
    assert '<rect x="9" y="48" width="22" height="40" fill="var(--series-1)">' in svg
    assert '<rect x="129" y="8" width="22" height="80" fill="var(--series-1)">' in svg
    assert '<line x1="8" y1="88" x2="248" y2="88"' in svg  # baseline across all 10 slots
    assert 'width="256" height="108"' in svg
    assert 'role="img"' in svg


def test_svg_histogram_tooltips_name_bin_ranges():
    svg = svg_histogram([0.05, 0.55, 0.55], lo=0.0, hi=1.0, n_bins=10, fill="var(--series-1)")
    assert "<title>1 of 3 scores in [0, 0.1)</title>" in svg
    assert "<title>2 of 3 scores in [0.5, 0.6)</title>" in svg


def test_svg_histogram_last_bin_interval_is_closed():
    svg = svg_histogram([1.0], lo=0.0, hi=1.0, n_bins=10, fill="var(--series-1)")
    assert "<title>1 of 1 scores in [0.9, 1]</title>" in svg


def test_svg_histogram_min_height_one_pixel():
    # count 1 vs max 200: 1*80//200 == 0, floored to a visible 1px bar at y = 87.
    values = [0.05] + [0.55] * 200
    svg = svg_histogram(values, lo=0.0, hi=1.0, n_bins=10, fill="var(--series-1)")
    assert '<rect x="9" y="87" width="22" height="1"' in svg


def test_svg_histogram_tick_labels():
    svg = svg_histogram([0.5], lo=0.0, hi=1.0, n_bins=10, fill="var(--series-1)")
    assert ">0<" in svg
    assert ">0.5<" in svg
    assert ">1<" in svg
    assert 'text-anchor="middle"' in svg


# --- terminal: render_analysis_text -----------------------------------------------

GOLDEN_ANALYSIS = """\
experiment: greeting-tone
run: 20260801-183042-a1b2 (pairwise, status: complete)
items: 4 | judgments used: 16 | errored: 0

variants:
  control           mean 0.375  (4 items scored)
  friendly          mean 0.625  (4 items scored)

friendly vs control (diff = friendly - control)
  paired items:     4 (0 dropped)
  mean control:     0.375
  mean friendly:    0.625
  mean diff:        0.250
  95% CI:           [0.125, 0.375]
  p-value:          0.1250 (exhaustive, 16 permutations)
  power:            23 items needed for 80% power (19 more than paired)"""


def test_render_analysis_text_golden():
    assert render_analysis_text(make_result(), status="complete") == GOLDEN_ANALYSIS


def test_render_analysis_text_without_status():
    text = render_analysis_text(make_result())
    assert "run: 20260801-183042-a1b2 (pairwise)" in text
    assert "status" not in text


def test_render_analysis_text_power_not_estimable():
    result = make_result(
        comparisons=[make_comparison(n_required_items=None, n_additional_items=None)]
    )
    text = render_analysis_text(result)
    assert "power:            not estimable" in text


def test_render_analysis_text_monte_carlo_method():
    result = make_result(
        comparisons=[make_comparison(p_method="monte_carlo", n_permutations=10_000)]
    )
    assert "(monte_carlo, 10000 permutations)" in render_analysis_text(result)


def test_render_analysis_text_variant_without_scores():
    result = make_result(scores={"control": {"q1": 0.25}, "friendly": {}})
    assert "  friendly          no scores" in render_analysis_text(result)


def test_render_analysis_text_multiple_comparisons():
    # Rubric k>2 yields C(k,2) comparison blocks, one header each.
    result = make_result(
        mode="rubric",
        comparisons=[
            make_comparison(variant_a="a", variant_b="b"),
            make_comparison(variant_a="a", variant_b="c"),
            make_comparison(variant_a="b", variant_b="c"),
        ],
    )
    text = render_analysis_text(result)
    assert "b vs a (diff = b - a)" in text
    assert "c vs a (diff = c - a)" in text
    assert "c vs b (diff = c - b)" in text


# --- terminal: render_audit_text --------------------------------------------------

GOLDEN_AUDIT = """\
judge report card
run: 20260801-183042-a1b2
judge: judge-model (pairwise)
judgments used: 4 | errored: 0
  flip rate:           1.000 (2 pairs, 0 dropped)
  position-A win rate: 1.000
  length bias:         slope 0.0625, correlation 1.000 (4 points)
  cross-judge kappa:   -1.000 vs run 20260801-190000-cd34 (n=4)"""


def test_render_audit_text_golden():
    assert render_audit_text(make_card()) == GOLDEN_AUDIT


def test_render_audit_text_rubric_card_uses_na():
    card = make_card(
        mode="rubric",
        n_pairs=None,
        n_pairs_dropped=None,
        flip_rate=None,
        position_a_win_rate=None,
        compare_run_id=None,
        kappa=None,
        kappa_n=None,
        notes=("position bias flip test requires pairwise mode",),
    )
    text = render_audit_text(card)
    assert "  flip rate:           n/a" in text
    assert "  position-A win rate: n/a" in text
    assert "cross-judge kappa" not in text
    assert "notes:\n  - position bias flip test requires pairwise mode" in text


def test_render_audit_text_no_notes_section_when_empty():
    assert "notes:" not in render_audit_text(make_card())


def test_render_audit_text_kappa_none_with_compare_run():
    card = make_card(kappa=None)
    assert "  cross-judge kappa:   n/a vs run 20260801-190000-cd34 (n=4)" in render_audit_text(card)


# --- HTML report ------------------------------------------------------------------


def test_render_html_document_shape():
    out = render_html(make_result(), make_card(), status="complete")
    assert out.startswith("<!DOCTYPE html>")
    assert out.rstrip().endswith("</html>")
    assert "<script" not in out
    assert out.count("<figure") == 2
    assert "var(--series-1)" in out
    assert "var(--series-2)" in out


def test_render_html_is_deterministic():
    result, card = make_result(), make_card()
    assert render_html(result, card, status="complete") == render_html(
        result, card, status="complete"
    )


def test_render_html_escapes_user_text():
    result = make_result(experiment_name='<b>&"x"')
    out = render_html(result)
    assert "&lt;b&gt;&amp;&quot;x&quot;" in out
    assert '<b>&"x"' not in out


def test_render_html_escapes_note_text():
    card = make_card(notes=('<script>alert("x")</script>',))
    out = render_html(make_result(), card)
    assert "<script" not in out
    assert "&lt;script&gt;" in out


def test_render_html_judge_section_only_with_card():
    with_card = render_html(make_result(), make_card())
    without = render_html(make_result(), None)
    assert "judge report card" in with_card
    assert "judge report card" not in without


def test_render_html_warning_banner_only_when_incomplete():
    failed = render_html(make_result(), status="failed")
    complete = render_html(make_result(), status="complete")
    none = render_html(make_result())
    assert "warning: run status is failed" in failed
    assert "warning:" not in complete
    assert "warning:" not in none


def test_render_html_significance_sentence():
    no = render_html(make_result())  # p = 0.125 >= 0.05
    yes = render_html(make_result(comparisons=[make_comparison(p_value=0.03125)]))
    assert "significant at alpha=0.05: no" in no
    assert "significant at alpha=0.05: yes" in yes


def test_render_html_variant_without_scores_falls_back():
    result = make_result(scores={"control": {"q1": 0.25}, "friendly": {}})
    out = render_html(result)
    assert "no scores" in out
    assert out.count("<figure") == 2


def test_render_html_contains_comparison_numbers():
    out = render_html(make_result())
    for fragment in ("0.250", "[0.125, 0.375]", "0.1250"):
        assert fragment in out


# --- review-driven hardening ------------------------------------------------------


def test_histogram_bins_non_finite_values_clamped():
    # Hand-crafted DBs can hold +/-inf REALs (the runner can't produce them, SQLite
    # stores NaN as NULL): they clamp to the end bins instead of OverflowError.
    bins = histogram_bins([float("inf"), float("-inf"), 0.5], lo=0.0, hi=1.0, n_bins=10)
    assert bins == [1, 0, 0, 0, 0, 1, 0, 0, 0, 1]


def test_render_html_significance_exact_alpha_boundary():
    # p == alpha is NOT significant: the comparison is strict < (0.05 is the same
    # float literal on both sides, so the boundary probe is exact).
    out = render_html(make_result(comparisons=[make_comparison(p_value=0.05)]))
    assert "significant at alpha=0.05: no" in out


def test_render_html_rubric_axis_wiring():
    # mode="rubric" must reach the (1, 10, 9) axis: unit-width bins and 1..10 ticks.
    result = make_result(mode="rubric", scores={"solo": {"q1": 5.5, "q2": 10.0}})
    out = render_html(result)
    assert ">1<" in out
    assert ">10<" in out
    assert "<title>1 of 2 scores in [5, 6)</title>" in out


def test_render_html_ninth_variant_falls_back_to_muted():
    scores = {f"v{i}": {"q1": 0.5} for i in range(9)}
    out = render_html(make_result(scores=scores))
    assert "var(--series-8)" in out
    assert "var(--muted)" in out
