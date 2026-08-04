"""Pre-registration: hash the planned experiment, label deviations (DESIGN §14).

Layering mirrors judge_audit: pure functions over the stored spec_json dump and
an AnalysisResult — no store access, no I/O, and zero coupling into stats.py's
inference core (prereg is metadata + labeling, never new statistics).

The hash covers the SCIENTIFIC content of the spec (PREREG_FIELDS): the design
(variants, dataset, sampling, n_samples, judge) plus the hypothesis and the
analysis plan. Labels and execution knobs (name, description, limits) are
excluded, mirroring §3's labels-out logic. `dataset.path` datasets are covered
by reference, not content (§12 deferral). The hash is a pure function of the
spec_json stored at create_run time, so any run's hash is recomputable forever;
`mimir run` prints it at run time as the commitment device.

Deviation semantics: a run-level deviation (correction method mismatch when more
than one comparison exists — Holm and BH are numerically identical at m=1, so a
mismatch there is not a deviation; or the planned primary pair absent from the
analysis) marks the whole analysis EXPLORATORY. Otherwise the analysis is
confirmatory, and only comparisons outside the planned set are individually
exploratory. Correcting over all C(k,2) pairs when fewer were planned is
conservative, never a deviation.
"""

import hashlib
from dataclasses import dataclass
from typing import Any

from mimir.cache import canonical_json
from mimir.spec import AnalysisPlan
from mimir.stats import AnalysisResult

# The hashed sub-dict of the spec dump. Appending a new spec field here is a
# deliberate §14 change: it moves every future hash (never a stored one).
PREREG_FIELDS = (
    "variants",
    "dataset",
    "sampling",
    "n_samples",
    "judge",
    "hypothesis",
    "analysis_plan",
)


@dataclass(frozen=True)
class PreregReport:
    """Everything the report renderers need about a pre-registered analysis."""

    prereg_hash: str
    hypothesis: str | None
    primary: tuple[str, str]
    planned: tuple[tuple[str, str], ...]
    alpha: float
    correction: str
    confirmatory: bool
    deviations: tuple[str, ...]
    # One label per result.comparisons entry: "primary" | "planned" | "exploratory".
    labels: tuple[str, ...]


def prereg_hash(spec_dump: dict[str, Any]) -> str | None:
    """sha256 over the canonical JSON of PREREG_FIELDS; None when no plan exists."""
    if spec_dump.get("analysis_plan") is None:
        return None
    content = {field: spec_dump.get(field) for field in PREREG_FIELDS}
    return hashlib.sha256(canonical_json(content)).hexdigest()


def evaluate_prereg(spec_dump: dict[str, Any], result: AnalysisResult) -> PreregReport | None:
    """Label an analysis against the run's pre-registered plan.

    Raises pydantic ValidationError (a ValueError) on a malformed plan block in
    hand-edited spec_json — callers decide whether that is fatal.
    """
    plan_dict = spec_dump.get("analysis_plan")
    if plan_dict is None:
        return None
    plan = AnalysisPlan.model_validate(plan_dict)
    digest = prereg_hash(spec_dump)

    primary_key = frozenset(plan.primary)
    planned_keys = {frozenset(pair) for pair in plan.comparisons}
    comparison_keys = [frozenset((c.variant_a, c.variant_b)) for c in result.comparisons]

    deviations: list[str] = []
    if len(result.comparisons) > 1 and result.correction_method != plan.correction:
        deviations.append(
            f"correction {result.correction_method!r} differs from the planned {plan.correction!r}"
        )
    if primary_key not in comparison_keys:
        deviations.append(
            f"the planned primary comparison {plan.primary[0]!r} vs {plan.primary[1]!r}"
            " is absent from the analysis"
        )

    if deviations:
        labels = tuple("exploratory" for _ in comparison_keys)
    else:
        labels = tuple(
            "primary" if key == primary_key else "planned" if key in planned_keys else "exploratory"
            for key in comparison_keys
        )
    return PreregReport(
        prereg_hash=digest,
        hypothesis=spec_dump.get("hypothesis"),
        primary=plan.primary,
        planned=tuple(plan.comparisons),
        alpha=plan.alpha,
        correction=plan.correction,
        confirmatory=not deviations,
        deviations=tuple(deviations),
        labels=labels,
    )
