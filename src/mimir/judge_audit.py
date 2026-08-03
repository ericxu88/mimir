"""Judge auditor (docs/DESIGN.md §8) — position-bias flip test, length bias, cross-judge kappa.

Pure functions over stored judgment rows, mirroring stats.py's layering: math over
plain sequences, extraction over store row dicts, and `audit_judge(store, run_id)` as
the one public store-taking entry point. The content-level winner of a pairwise judgment
is always resolved through the sample join (sample_a_id -> condition -> variant_name);
`position_order` is provenance only and is never consulted — sample_a_id/sample_b_id
are the samples PRESENTED in positions A/B, which already encodes the order.
"""

import json
import math
from collections import Counter
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from mimir.stats import _VERDICT_VALUE, _sample_variants
from mimir.store import Store

# --- pure math --------------------------------------------------------------------

# Kappa's chance-agreement complement 1 - pe is either exactly 0.0 (both judges
# constant on one shared category) or >= 1/n^2; the tolerance only guards float noise.
_DEGENERATE_TOL = 1e-12


def cohens_kappa(labels_a: Sequence[Hashable], labels_b: Sequence[Hashable]) -> float | None:
    """Cohen's kappa between two aligned label sequences.

    Returns None (not estimable) when chance agreement pe is 1 — both raters constant
    on the same category — where kappa is 0/0.
    """
    if len(labels_a) != len(labels_b):
        raise ValueError("label sequences must have equal length")
    if not labels_a:
        raise ValueError("label sequences are empty")
    n = len(labels_a)
    observed = sum(a == b for a, b in zip(labels_a, labels_b, strict=True)) / n
    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)
    expected = sum((counts_a[cat] / n) * (counts_b[cat] / n) for cat in counts_a | counts_b)
    if 1.0 - expected <= _DEGENERATE_TOL:
        return None
    return (observed - expected) / (1.0 - expected)


def length_regression(
    x: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray
) -> tuple[float | None, float | None]:
    """OLS slope and Pearson r of y on x via mean-centered sums.

    The formulas are pinned for float-exactness on dyadic test constructions:
    slope = Sxy / Sxx and r = Sxy / sqrt(Sxx * Syy) — a SINGLE sqrt over the
    product, whose argument is a perfect square when the data are collinear.
    np.polyfit, np.corrcoef, and sqrt(Sxx) * sqrt(Syy) are all inexact there.

    Returns (None, None) when x has zero variance (nothing to regress on) and
    (slope, None) when only y is constant — a constant-score judge has a length
    slope of exactly 0.0, but its correlation is undefined.
    """
    xs = np.asarray(x, dtype=np.float64)
    ys = np.asarray(y, dtype=np.float64)
    if xs.size != ys.size:
        raise ValueError("x and y must have equal length")
    if xs.size == 0:
        raise ValueError("x and y are empty")
    if not (np.isfinite(xs).all() and np.isfinite(ys).all()):
        raise ValueError("x and y must be finite")
    xc = xs - xs.mean()
    yc = ys - ys.mean()
    sxx = float(np.dot(xc, xc))
    syy = float(np.dot(yc, yc))
    sxy = float(np.dot(xc, yc))
    # Exact zero checks are safe: variance sums of integer lengths / dyadic scores
    # carry no near-zero cancellation (unlike M3's mean-of-diffs power guard).
    if sxx == 0.0:
        return (None, None)
    slope = sxy / sxx
    if syy == 0.0:
        return (slope, None)
    # Clamp: the formula is exact on dyadic constructions but can land one ulp
    # outside [-1, 1] on perfectly separated data with a non-representable mean,
    # and an impossible correlation must never reach the report card.
    return (slope, max(-1.0, min(1.0, sxy / math.sqrt(sxx * syy))))


# --- extraction: stored rows -> bias measurements ---------------------------------


@dataclass(frozen=True)
class PositionBias:
    """Flip test + position-A win rate over one run's pairwise judgment rows."""

    n_rows_used: int
    n_rows_errored: int
    n_pairs: int  # order-twin pairs with both rows usable
    n_pairs_dropped: int  # groups with >= 1 usable row that could not be evaluated
    n_flips: int
    flip_rate: float | None  # None when no pairs (e.g. position_swap: false)
    position_a_win_rate: float | None  # mean verdict value over ALL usable rows


@dataclass(frozen=True)
class LengthBias:
    """score ~ length regression over one run's judgment rows (DESIGN §8)."""

    n_rows_used: int
    n_rows_errored: int
    n_points: int  # usable rows whose sample(s) have a response_text
    slope: float | None
    correlation: float | None


def _usable_pairwise(row: dict[str, Any], sample_variant: dict[int, str]) -> bool:
    # Beyond stats.py's error/verdict rule, guard the joins this module (unlike
    # stats.py) dereferences: sample_b_id NULL, a sample id outside this run's
    # samples, or a self-pair (the runner always presents two distinct samples) is
    # drifted data and must count as errored, never crash or dilute a metric.
    return (
        row["error"] is None
        and row["verdict"] in _VERDICT_VALUE
        and row["sample_b_id"] is not None
        and row["sample_a_id"] != row["sample_b_id"]
        and row["sample_a_id"] in sample_variant
        and row["sample_b_id"] in sample_variant
    )


def _content_outcome(row: dict[str, Any], sample_variant: dict[int, str]) -> str | None:
    """Content-level winner: the winning sample's variant name, or None for a tie.

    Resolved through the sample join only — sample_a_id/sample_b_id are the samples
    PRESENTED in positions A/B, so the verdict letter picks the winning sample and
    the join names its variant. position_order must never enter this computation.
    """
    if row["verdict"] == "A":
        return sample_variant[row["sample_a_id"]]
    if row["verdict"] == "B":
        return sample_variant[row["sample_b_id"]]
    return None


def position_bias(
    judgments: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    conditions: list[dict[str, Any]],
) -> PositionBias:
    """Position-bias flip test: a twin pair is consistent iff the content-level
    winner is identical across both presentation orders (tie == tie counts as
    consistent; tie vs win is a flip).

    Twins are grouped by the unordered sample-id pair — the 'ab' and 'ba' rows of one
    content pair satisfy ab.sample_a_id == ba.sample_b_id and vice versa, and sample
    ids are unique per (variant, item, replicate), so the group IS the (item,
    replicate) pair. item_id alone would merge replicates; row order is
    task-completion order, so adjacency is meaningless. A group is evaluated iff it
    holds exactly two usable rows presenting opposite orders; anything else
    (missing twin, errored twin, hand-inserted duplicates) is dropped and counted.
    """
    if len(conditions) != 2:
        raise ValueError(f"position bias requires exactly two variants, got {len(conditions)}")
    sample_variant = _sample_variants(samples, conditions)
    used = errored = 0
    verdict_values: list[float] = []
    groups: dict[frozenset[int], list[dict[str, Any]]] = {}
    for row in judgments:
        if not _usable_pairwise(row, sample_variant):
            errored += 1
            continue
        used += 1
        verdict_values.append(_VERDICT_VALUE[row["verdict"]])
        key = frozenset((row["sample_a_id"], row["sample_b_id"]))
        groups.setdefault(key, []).append(row)
    n_pairs = n_flips = n_dropped = 0
    for rows in groups.values():
        if len(rows) != 2 or rows[0]["sample_a_id"] == rows[1]["sample_a_id"]:
            n_dropped += 1
            continue
        n_pairs += 1
        if _content_outcome(rows[0], sample_variant) != _content_outcome(rows[1], sample_variant):
            n_flips += 1
    return PositionBias(
        n_rows_used=used,
        n_rows_errored=errored,
        n_pairs=n_pairs,
        n_pairs_dropped=n_dropped,
        n_flips=n_flips,
        flip_rate=n_flips / n_pairs if n_pairs else None,
        position_a_win_rate=float(np.mean(verdict_values)) if verdict_values else None,
    )


def length_bias_pairwise(
    judgments: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    conditions: list[dict[str, Any]],
) -> LengthBias:
    """Length bias for pairwise runs: win indicator (A=1, TIE=0.5, B=0) regressed on
    len_A - len_B in PRESENTED positions (chars of samples.response_text).

    Including both presentation orders is deliberate: a content-consistent judge's
    'ba' twin negates the length difference and flips the indicator, so pure
    position preference cancels out of the length signal.
    """
    if len(conditions) != 2:
        raise ValueError(f"length bias requires exactly two variants, got {len(conditions)}")
    sample_variant = _sample_variants(samples, conditions)
    sample_text = {row["id"]: row["response_text"] for row in samples}
    used = errored = 0
    diffs: list[float] = []
    values: list[float] = []
    for row in judgments:
        if not _usable_pairwise(row, sample_variant):
            errored += 1
            continue
        used += 1
        text_a = sample_text[row["sample_a_id"]]
        text_b = sample_text[row["sample_b_id"]]
        if text_a is None or text_b is None:
            continue  # drifted: the runner never judges a sample without text
        diffs.append(float(len(text_a) - len(text_b)))
        values.append(_VERDICT_VALUE[row["verdict"]])
    slope, correlation = length_regression(diffs, values) if len(diffs) >= 2 else (None, None)
    return LengthBias(
        n_rows_used=used,
        n_rows_errored=errored,
        n_points=len(diffs),
        slope=slope,
        correlation=correlation,
    )


def length_bias_rubric(
    judgments: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    conditions: list[dict[str, Any]],
) -> LengthBias:
    """Length bias for rubric runs: score regressed on len(response_text) in chars.

    `conditions` is accepted for signature symmetry with the other extraction
    functions and is not consulted (rubric rows join through samples alone).
    """
    sample_text = {row["id"]: row["response_text"] for row in samples}
    used = errored = 0
    lengths: list[float] = []
    scores: list[float] = []
    for row in judgments:
        usable = row["error"] is None and row["score"] is not None
        if not usable or row["sample_a_id"] not in sample_text:
            errored += 1
            continue
        used += 1
        text = sample_text[row["sample_a_id"]]
        if text is None:
            continue  # drifted: the runner never judges a sample without text
        lengths.append(float(len(text)))
        scores.append(float(row["score"]))
    slope, correlation = length_regression(lengths, scores) if len(lengths) >= 2 else (None, None)
    return LengthBias(
        n_rows_used=used,
        n_rows_errored=errored,
        n_points=len(lengths),
        slope=slope,
        correlation=correlation,
    )


def _content_outcomes(
    judgments: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    conditions: list[dict[str, Any]],
) -> dict[tuple[str, int, str], str | None]:
    """Kappa's per-run extraction: (item_id, replicate, presented-A variant) -> outcome.

    Keyed by the presented-A variant NAME via the sample join, not position_order:
    'ab'/'ba' are relative to each run's declared variant order, so a position label
    cannot align runs that declare the same variants in different orders. Both
    samples of a runner-written pair share sample_index, so sample_a_id recovers the
    replicate.
    """
    sample_variant = _sample_variants(samples, conditions)
    sample_index = {row["id"]: row["sample_index"] for row in samples}
    outcomes: dict[tuple[str, int, str], str | None] = {}
    conflicted: set[tuple[str, int, str]] = set()
    for row in judgments:
        if not _usable_pairwise(row, sample_variant):
            continue
        key = (row["item_id"], sample_index[row["sample_a_id"]], sample_variant[row["sample_a_id"]])
        outcome = _content_outcome(row, sample_variant)
        # Hand-inserted duplicates with contradicting outcomes poison the unit:
        # drop it entirely (last-write-wins would silently change kappa).
        if key in outcomes and outcomes[key] != outcome:
            conflicted.add(key)
        outcomes[key] = outcome
    for key in conflicted:
        del outcomes[key]
    return outcomes


# --- orchestration: the one store-taking entry point ------------------------------


@dataclass(frozen=True)
class JudgeReportCard:
    """DESIGN §8 report card; plain JSON-friendly types (M5 asdict()s it)."""

    run_id: str
    judge_model: str
    mode: str  # "pairwise" | "rubric"
    n_judgments_used: int
    n_judgments_errored: int
    n_pairs: int | None  # None in rubric mode (flip test not applicable)
    n_pairs_dropped: int | None
    flip_rate: float | None
    position_a_win_rate: float | None
    n_length_points: int
    length_slope: float | None
    length_correlation: float | None
    compare_run_id: str | None
    kappa: float | None
    kappa_n: int | None
    notes: tuple[str, ...]


def _cross_judge_kappa(
    store: Store,
    run_id: str,
    mode: str,
    judgments: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    conditions: list[dict[str, Any]],
    compare_run_id: str,
) -> tuple[float | None, int, list[str]]:
    if mode != "pairwise":
        raise ValueError("cross-judge kappa requires pairwise mode in both runs")
    compare = store.get_run(compare_run_id)
    if compare is None:
        raise ValueError(f"run {compare_run_id!r} not found")
    compare_judge = json.loads(compare["spec_json"]).get("judge")
    if compare_judge is None:
        raise ValueError(f"run {compare_run_id!r} has no judge configured; nothing to audit")
    if compare_judge.get("mode") != "pairwise":
        raise ValueError("cross-judge kappa requires pairwise mode in both runs")
    compare_conditions = store.get_conditions(compare_run_id)
    names = {row["variant_name"] for row in conditions}
    if names != {row["variant_name"] for row in compare_conditions}:
        raise ValueError(
            f"runs {run_id!r} and {compare_run_id!r} are not comparable: variant names differ"
        )
    ours = _content_outcomes(judgments, samples, conditions)
    theirs = _content_outcomes(
        store.get_judgments(compare_run_id),
        store.get_samples(compare_run_id),
        compare_conditions,
    )
    shared = sorted(ours.keys() & theirs.keys())
    if not shared:
        raise ValueError(f"runs {run_id!r} and {compare_run_id!r} share no judged units")
    kappa = cohens_kappa([ours[key] for key in shared], [theirs[key] for key in shared])
    notes = []
    if kappa is None:
        notes.append("kappa not estimable: both judges constant on the same outcome")
    return kappa, len(shared), notes


def audit_judge(store: Store, run_id: str, *, compare_run_id: str | None = None) -> JudgeReportCard:
    """Audit one stored run's judge; optionally compare against a second run's judge.

    Partial data from a `failed` or `running` run is audited as-is (M3 precedent).
    Raises ValueError when there is nothing to audit (unknown run, no judge, no
    judgments) or when a kappa comparison is invalid (non-pairwise mode, unknown or
    judgeless compare run, differing variant names, no shared judged units).
    """
    run = store.get_run(run_id)
    if run is None:
        raise ValueError(f"run {run_id!r} not found")
    judge = json.loads(run["spec_json"]).get("judge")
    if judge is None:
        raise ValueError(f"run {run_id!r} has no judge configured; nothing to audit")
    mode = judge.get("mode")
    judge_model = judge.get("model")
    if mode is None or judge_model is None:
        raise ValueError(f"run {run_id!r} judge block is missing 'mode' or 'model'")
    judgments = store.get_judgments(run_id)
    if not judgments:
        raise ValueError(f"run {run_id!r} has no judgments to audit")
    samples = store.get_samples(run_id)
    conditions = store.get_conditions(run_id)
    notes: list[str] = []
    if mode == "pairwise":
        position = position_bias(judgments, samples, conditions)
        length = length_bias_pairwise(judgments, samples, conditions)
        n_used, n_errored = position.n_rows_used, position.n_rows_errored
        if position.n_rows_used == 0:
            notes.append("no usable pairwise judgments")
        elif position.flip_rate is None:
            notes.append("no order-swapped judgment pairs; flip test skipped (position_swap off?)")
            notes.append(
                "position_a_win_rate conflates position and content preference"
                " without order-swapped pairs"
            )
        pair_fields = (position.n_pairs, position.n_pairs_dropped)
        rate_fields = (position.flip_rate, position.position_a_win_rate)
    elif mode == "rubric":
        length = length_bias_rubric(judgments, samples, conditions)
        n_used, n_errored = length.n_rows_used, length.n_rows_errored
        notes.append("position bias flip test requires pairwise mode")
        pair_fields = (None, None)
        rate_fields = (None, None)
    else:
        raise ValueError(f"unknown judge mode {mode!r}")
    if length.correlation is None:
        # The slope is a separate field and may still be reported (constant scores).
        notes.append("length correlation not estimable: zero variance or fewer than two points")
    kappa = None
    kappa_n = None
    if compare_run_id is not None:
        kappa, kappa_n, kappa_notes = _cross_judge_kappa(
            store, run_id, mode, judgments, samples, conditions, compare_run_id
        )
        notes.extend(kappa_notes)
    return JudgeReportCard(
        run_id=run_id,
        judge_model=judge_model,
        mode=mode,
        n_judgments_used=n_used,
        n_judgments_errored=n_errored,
        n_pairs=pair_fields[0],
        n_pairs_dropped=pair_fields[1],
        flip_rate=rate_fields[0],
        position_a_win_rate=rate_fields[1],
        n_length_points=length.n_points,
        length_slope=length.slope,
        length_correlation=length.correlation,
        compare_run_id=compare_run_id,
        kappa=kappa,
        kappa_n=kappa_n,
        notes=tuple(notes),
    )
