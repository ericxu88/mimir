"""Tests for mimir.prereg — preregistration hash + deviation labeling (DESIGN §14).

The hash golden is a cross-session drift tripwire like the §3 cache-key goldens:
it was derived once (and cross-checked against an independent hashlib/json
expression) before being hardcoded. A failure means the canonicalization or the
PREREG_FIELDS allowlist changed — fix the code, never the golden, unless §14 and
PROGRESS.md are deliberately updated first.
"""

import json
import re

import pytest
from pydantic import ValidationError

import mimir.spec as spec_mod
import mimir.stats as stats_mod
from mimir.prereg import PREREG_FIELDS, evaluate_prereg, prereg_hash
from mimir.spec import ExperimentSpec
from mimir.stats import AnalysisResult, Comparison


def planned_dump(**overrides):
    """A resolved planned-spec dump, built through the real models."""
    data = {
        "name": "greeting-tone",
        "description": "does tone help",
        "variants": [
            {"name": "control", "system": "You are helpful.", "user_template": "A: {input}"},
            {"name": "friendly", "system": "You are warm.", "user_template": "A: {input}"},
        ],
        "dataset": {"items": [{"id": "q1", "input": "sky?"}, {"id": "q2", "input": "grass?"}]},
        "sampling": {"model": "claude-haiku-4-5-20251001", "seed": 7},
        "n_samples": 2,
        "judge": {"model": "judge-model", "mode": "pairwise"},
        "hypothesis": "friendly beats control on judged quality",
        "analysis_plan": {"primary": ["control", "friendly"]},
    }
    data.update(overrides)
    return ExperimentSpec.model_validate(data).model_dump()


def make_comparison(variant_a, variant_b, **overrides):
    fields = {
        "variant_a": variant_a,
        "variant_b": variant_b,
        "item_ids": ("q1", "q2"),
        "diffs": (0.25, 0.5),
        "n_items": 2,
        "n_items_dropped": 0,
        "mean_a": 0.375,
        "mean_b": 0.625,
        "mean_diff": 0.375,
        "ci_low": 0.125,
        "ci_high": 0.5,
        "ci_level": 0.95,
        "p_value": 0.5,
        "p_method": "exhaustive",
        "n_resamples": 10_000,
        "n_permutations": 4,
        "alpha": 0.05,
        "target_power": 0.8,
        "n_required_items": 23,
        "n_additional_items": 21,
        "seed": 0,
        "correction_method": "holm",
        "n_comparisons": 1,
    }
    fields.update(overrides)
    return Comparison(**fields)


def make_result(comparisons, *, correction_method="holm", mode="pairwise"):
    scores = {}
    for c in comparisons:
        scores.setdefault(c.variant_a, {"q1": 0.25})
        scores.setdefault(c.variant_b, {"q1": 0.5})
    return AnalysisResult(
        run_id="20260803-000000-abcd",
        experiment_name="greeting-tone",
        mode=mode,
        scores=scores,
        comparisons=list(comparisons),
        n_items=2,
        n_judgments_used=8,
        n_judgments_errored=0,
        correction_method=correction_method,
    )


# --- prereg_hash -------------------------------------------------------------------


def test_hash_none_for_planless_dumps():
    planless = planned_dump()
    planless["analysis_plan"] = None
    planless["hypothesis"] = None
    assert prereg_hash(planless) is None
    # Pre-M11 shaped dump: the keys are absent entirely.
    legacy = {k: v for k, v in planned_dump().items() if k not in ("hypothesis", "analysis_plan")}
    assert prereg_hash(legacy) is None


def test_hash_is_64_hex_and_deterministic():
    first = prereg_hash(planned_dump())
    assert re.fullmatch(r"[0-9a-f]{64}", first)
    assert prereg_hash(planned_dump()) == first


GOLDEN_PREREG_HASH = "e0cad927c9c5e214bc63841f33c66b70eb53685ba177b7c6f5354ef6f514da1c"


def test_hash_golden_matches_pinned_value():
    assert prereg_hash(planned_dump()) == GOLDEN_PREREG_HASH


@pytest.mark.parametrize("field", ["name", "description", "limits"])
def test_hash_ignores_labels_and_execution_knobs(field):
    base = prereg_hash(planned_dump())
    changed = planned_dump()
    changed[field] = {"concurrency": 9, "requests_per_minute": 9} if field == "limits" else "other"
    assert prereg_hash(changed) == base


@pytest.mark.parametrize("field", list(PREREG_FIELDS))
def test_hash_covers_every_prereg_field(field):
    base = prereg_hash(planned_dump())
    changed = planned_dump()
    if field == "variants":
        changed["variants"][0]["user_template"] = "B: {input}"
    elif field == "dataset":
        changed["dataset"]["items"][0]["input"] = "sea?"
    elif field == "sampling":
        changed["sampling"]["seed"] = 8
    elif field == "n_samples":
        changed["n_samples"] = 3
    elif field == "judge":
        changed["judge"]["model"] = "other-judge"
    elif field == "hypothesis":
        changed["hypothesis"] = "control beats friendly"
    else:
        changed["analysis_plan"]["correction"] = "bh"
    assert prereg_hash(changed) != base


def test_hash_identical_across_run_and_analyze_paths():
    # Run time hashes spec.model_dump(); analyze time hashes json.loads(spec_json).
    dump = planned_dump()
    via_json = json.loads(json.dumps(dump))
    assert prereg_hash(dump) == prereg_hash(via_json)


def test_plan_alpha_literal_matches_stats_alpha():
    # spec.py must never import stats (numpy at spec load); the two literals are
    # pinned equal here instead.
    assert spec_mod._PLAN_ALPHA == stats_mod.ALPHA


# --- evaluate_prereg ---------------------------------------------------------------


def test_evaluate_returns_none_without_plan():
    planless = planned_dump()
    planless["analysis_plan"] = None
    planless["hypothesis"] = None
    result = make_result([make_comparison("control", "friendly")])
    assert evaluate_prereg(planless, result) is None


def test_single_comparison_primary_confirmatory_with_reversed_plan_order():
    # Comparisons are oriented by declaration order; plan pairs match UNORDERED.
    dump = planned_dump()
    dump["analysis_plan"]["primary"] = ("friendly", "control")
    result = make_result([make_comparison("control", "friendly")])
    report = evaluate_prereg(dump, result)
    assert report.confirmatory is True
    assert report.deviations == ()
    assert report.labels == ("primary",)
    assert report.prereg_hash == prereg_hash(dump)
    assert report.hypothesis == "friendly beats control on judged quality"


def test_correction_mismatch_at_m1_is_not_a_deviation():
    # Holm and BH are numerically identical for a single comparison: the executed
    # analysis IS the planned analysis (human-approved m=1 identity rule).
    dump = planned_dump()
    result = make_result(
        [make_comparison("control", "friendly", correction_method="bh")],
        correction_method="bh",
    )
    report = evaluate_prereg(dump, result)
    assert report.confirmatory is True
    assert report.labels == ("primary",)


def multiarm_dump(extra_planned=True):
    data = {
        "name": "multi",
        "variants": [
            {"name": "a", "user_template": "A: {input}"},
            {"name": "b", "user_template": "B: {input}"},
            {"name": "c", "user_template": "C: {input}"},
        ],
        "dataset": {"items": [{"id": "q1", "input": "x"}]},
        "sampling": {"model": "m"},
        "judge": {"model": "j", "mode": "rubric"},
        "analysis_plan": {"primary": ["a", "b"]},
    }
    if extra_planned:
        data["analysis_plan"]["comparisons"] = [["a", "c"]]
    return ExperimentSpec.model_validate(data).model_dump()


def multiarm_result(correction_method="holm"):
    comparisons = [
        make_comparison("a", "b", correction_method=correction_method, n_comparisons=3),
        make_comparison("a", "c", correction_method=correction_method, n_comparisons=3),
        make_comparison("b", "c", correction_method=correction_method, n_comparisons=3),
    ]
    return make_result(comparisons, correction_method=correction_method, mode="rubric")


def test_multiarm_labels_primary_planned_and_exploratory():
    report = evaluate_prereg(multiarm_dump(), multiarm_result())
    assert report.confirmatory is True
    assert report.labels == ("primary", "planned", "exploratory")


def test_correction_mismatch_at_m3_makes_everything_exploratory():
    report = evaluate_prereg(multiarm_dump(), multiarm_result(correction_method="bh"))
    assert report.confirmatory is False
    assert report.labels == ("exploratory", "exploratory", "exploratory")
    assert any("bh" in reason and "holm" in reason for reason in report.deviations)


def test_missing_primary_comparison_is_a_run_level_deviation():
    # Hand-edited data: the plan's primary pair never shows up in the analysis.
    dump = multiarm_dump(extra_planned=False)
    dump["analysis_plan"]["primary"] = ("a", "b")
    result = make_result(
        [make_comparison("a", "c", n_comparisons=1)],
        mode="rubric",
    )
    report = evaluate_prereg(dump, result)
    assert report.confirmatory is False
    assert report.labels == ("exploratory",)
    assert any("primary" in reason for reason in report.deviations)


def test_malformed_plan_raises_value_error():
    dump = planned_dump()
    dump["analysis_plan"] = {"primary": ["control", "friendly"], "alpha": 0.5}
    result = make_result([make_comparison("control", "friendly")])
    with pytest.raises(ValidationError):
        evaluate_prereg(dump, result)
    assert issubclass(ValidationError, ValueError)  # cli's except ValueError catches it
