"""Stats engine (docs/DESIGN.md §7) — paired bootstrap CI, sign-flip p-value, power estimate.

Analysis is a pure read of the store: `analyze_run` is the only store-taking entry
point and uses only the pinned getters. Per-item paired differences are always
score(later-declared variant) - score(earlier-declared variant), items sorted by
item_id so seeded resampling is reproducible across invocations. Judgment rows with
an error are skipped and counted; items lacking a usable score for both variants of
a comparison are dropped and counted. No multiple-comparison correction in v1.
"""

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from statistics import NormalDist
from typing import Any, Literal

import numpy as np

from mimir.store import Store

DEFAULT_RESAMPLES = 10_000
DEFAULT_PERMUTATIONS = 10_000
# Exhaustive sign-flip up to 2^16 = 65,536 patterns: one small matmul, and exact
# precisely in the small-n regime where Monte Carlo error is worst.
EXHAUSTIVE_MAX_N = 16
CONFIDENCE = 0.95
ALPHA = 0.05
TARGET_POWER = 0.80

_VERDICT_VALUE = {"A": 1.0, "B": 0.0, "TIE": 0.5}


@dataclass(frozen=True)
class ScoreTable:
    """Per-item mean scores per variant, extracted from one run's stored rows."""

    scores: dict[str, dict[str, float]]  # variant_name -> item_id -> per-item mean score
    n_items: int  # distinct item_ids over the run's samples (the dataset as run)
    n_judgments_used: int
    n_judgments_errored: int


@dataclass(frozen=True)
class Comparison:
    """Paired analysis of one variant pair; all statistics are variant_b - variant_a."""

    variant_a: str  # earlier-declared variant
    variant_b: str
    item_ids: tuple[str, ...]  # sorted; the items actually paired
    diffs: tuple[float, ...]  # aligned with item_ids
    n_items: int  # len(item_ids)
    n_items_dropped: int
    mean_a: float  # means over the paired items only
    mean_b: float
    mean_diff: float
    ci_low: float
    ci_high: float
    ci_level: float
    p_value: float
    p_method: Literal["exhaustive", "monte_carlo"]
    n_resamples: int
    n_permutations: int  # 2**n when exhaustive
    alpha: float
    target_power: float
    n_required_items: int | None  # total items for TARGET_POWER at the observed effect
    n_additional_items: int | None  # max(0, required - n_items); None when not estimable
    seed: int


@dataclass(frozen=True)
class AnalysisResult:
    run_id: str
    experiment_name: str
    mode: str  # "pairwise" | "rubric"
    scores: dict[str, dict[str, float]]  # every scored (variant, item), even unpaired ones
    comparisons: list[Comparison]
    n_items: int
    n_judgments_used: int
    n_judgments_errored: int


# --- pure math over the per-item difference vector --------------------------------


def _as_diffs(diffs: Sequence[float] | np.ndarray) -> np.ndarray:
    d = np.asarray(diffs, dtype=np.float64)
    if d.size == 0:
        raise ValueError("diffs is empty")
    if not np.isfinite(d).all():
        raise ValueError("diffs must be finite")
    return d


def bootstrap_ci(
    diffs: Sequence[float] | np.ndarray,
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> tuple[float, float]:
    """95% percentile CI on the mean of the paired differences (resampling items)."""
    d = _as_diffs(diffs)
    if n_resamples < 1:
        raise ValueError("n_resamples must be >= 1")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_resamples, d.size))
    boot_means = d[idx].mean(axis=1)
    # method="linear" is today's numpy default, pinned so a future default change
    # cannot silently shift seeded CI values.
    tail = (1.0 - CONFIDENCE) / 2.0
    lo, hi = np.quantile(boot_means, [tail, 1.0 - tail], method="linear")
    return (float(lo), float(hi))


def sign_flip_pvalue(
    diffs: Sequence[float] | np.ndarray,
    *,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = 0,
) -> float:
    """Two-sided p-value for mean(diffs) == 0 under sign-flip symmetry.

    Exhaustive over all 2^n sign patterns when n <= EXHAUSTIVE_MAX_N (exact, no
    smoothing: the identity pattern guarantees p >= 2^-n). Monte Carlo otherwise,
    with add-one smoothing (Phipson & Smyth) so p is never 0.
    """
    d = _as_diffs(diffs)
    if n_permutations < 1:
        raise ValueError("n_permutations must be >= 1")
    n = d.size
    s_obs = abs(float(np.sum(d)))
    # signs @ d sums in a different order than np.sum(d), so the identity pattern can
    # miss its own comparison by an ulp; the tolerance is conservative (inflates p).
    tau = 1e-12 * max(1.0, s_obs)
    if n <= EXHAUSTIVE_MAX_N:
        patterns = np.arange(2**n, dtype=np.uint32)[:, None]
        bits = (patterns >> np.arange(n, dtype=np.uint32)) & 1
        signs = 1.0 - 2.0 * bits
        perm_sums = signs @ d
        count = int(np.count_nonzero(np.abs(perm_sums) >= s_obs - tau))
        return count / 2**n
    rng = np.random.default_rng(seed)
    signs = rng.integers(0, 2, size=(n_permutations, n)) * 2 - 1
    perm_sums = signs @ d
    count = int(np.count_nonzero(np.abs(perm_sums) >= s_obs - tau))
    return (1 + count) / (1 + n_permutations)


def required_items_for_power(diffs: Sequence[float] | np.ndarray) -> int | None:
    """Total items needed to detect the observed effect at TARGET_POWER, two-sided ALPHA.

    Normal approximation for the paired mean. Returns None when not estimable:
    zero observed mean (infinite n) or n < 2 (ddof=1 sd undefined).
    """
    d = _as_diffs(diffs)
    if d.size < 2:
        return None
    mean = float(np.mean(d))
    sd = float(np.std(d, ddof=1))
    # Same cancellation hazard as tau in sign_flip_pvalue: a perfectly balanced null
    # built from non-dyadic scores (e.g. pairwise means quantized at 1/12) yields a
    # mean of ~1e-17, not exactly 0.0, and the formula would report an astronomical
    # n (or overflow ceil). An effect that far below the data scale is not estimable.
    if abs(mean) <= 1e-12 * max(1.0, sd):
        return None
    # z_a = 1.959963984540054, z_p = 0.8416212335729143 (stdlib inverse normal CDF).
    z_a = NormalDist().inv_cdf(1.0 - ALPHA / 2.0)
    z_p = NormalDist().inv_cdf(TARGET_POWER)
    # A paired analysis needs at least 2 items, so a zero-variance effect floors there.
    return max(2, math.ceil(((z_a + z_p) * sd / abs(mean)) ** 2))


# --- extraction: stored rows -> per-item scores per variant -----------------------


def _sample_variants(
    samples: list[dict[str, Any]], conditions: list[dict[str, Any]]
) -> dict[int, str]:
    variant_names = {row["id"]: row["variant_name"] for row in conditions}
    return {row["id"]: variant_names[row["condition_id"]] for row in samples}


def _count_items(samples: list[dict[str, Any]]) -> int:
    return len({row["item_id"] for row in samples})


def pairwise_item_scores(
    judgments: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    conditions: list[dict[str, Any]],
) -> ScoreTable:
    """Per-item scores from pairwise verdicts: win 1, tie 0.5, loss 0, averaged over
    replicates and both position orders; the two variants' scores are complementary.

    The sample join (sample_a_id -> condition -> variant) is the source of truth for
    which variant sat in the presented position A; position_order is not consulted.
    """
    if len(conditions) != 2:
        raise ValueError(f"pairwise analysis requires exactly two variants, got {len(conditions)}")
    variant_a = conditions[0]["variant_name"]
    variant_b = conditions[1]["variant_name"]
    sample_variant = _sample_variants(samples, conditions)
    values_a: dict[str, list[float]] = {}
    used = errored = 0
    for row in judgments:
        # Defensive second clause: the runner never writes a NULL/unknown verdict
        # without an error, but analysis must not crash on drifted data.
        value = _VERDICT_VALUE.get(row["verdict"])
        if row["error"] is not None or value is None:
            errored += 1
            continue
        used += 1
        if sample_variant[row["sample_a_id"]] != variant_a:
            value = 1.0 - value  # presented-A was the later-declared variant
        values_a.setdefault(row["item_id"], []).append(value)
    scores_a = {item: float(np.mean(values)) for item, values in values_a.items()}
    scores_b = {item: 1.0 - score for item, score in scores_a.items()}
    return ScoreTable(
        scores={variant_a: scores_a, variant_b: scores_b},
        n_items=_count_items(samples),
        n_judgments_used=used,
        n_judgments_errored=errored,
    )


def rubric_item_scores(
    judgments: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    conditions: list[dict[str, Any]],
) -> ScoreTable:
    """Per-item scores from rubric judgments: mean of `score` over replicates,
    per variant (via the sample_a_id -> condition -> variant join)."""
    sample_variant = _sample_variants(samples, conditions)
    values: dict[str, dict[str, list[float]]] = {row["variant_name"]: {} for row in conditions}
    used = errored = 0
    for row in judgments:
        if row["error"] is not None or row["score"] is None:
            errored += 1
            continue
        used += 1
        variant = sample_variant[row["sample_a_id"]]
        values[variant].setdefault(row["item_id"], []).append(float(row["score"]))
    scores = {
        variant: {item: float(np.mean(item_values)) for item, item_values in per_item.items()}
        for variant, per_item in values.items()
    }
    return ScoreTable(
        scores=scores,
        n_items=_count_items(samples),
        n_judgments_used=used,
        n_judgments_errored=errored,
    )


# --- orchestration: the one store-taking entry point ------------------------------


def _compare(
    variant_a: str,
    variant_b: str,
    scores_a: dict[str, float],
    scores_b: dict[str, float],
    *,
    n_items_total: int,
    n_resamples: int,
    n_permutations: int,
    seed: int,
) -> Comparison:
    # Sorting is load-bearing: sample rows arrive in task-completion order, and a
    # seeded bootstrap over differently-ordered diffs draws different resamples.
    item_ids = tuple(sorted(scores_a.keys() & scores_b.keys()))
    if not item_ids:
        raise ValueError(f"comparison {variant_a!r} vs {variant_b!r} has no paired items")
    diffs = tuple(float(scores_b[item] - scores_a[item]) for item in item_ids)
    n = len(item_ids)
    ci_low, ci_high = bootstrap_ci(diffs, n_resamples=n_resamples, seed=seed)
    p_value = sign_flip_pvalue(diffs, n_permutations=n_permutations, seed=seed)
    exhaustive = n <= EXHAUSTIVE_MAX_N
    required = required_items_for_power(diffs)
    return Comparison(
        variant_a=variant_a,
        variant_b=variant_b,
        item_ids=item_ids,
        diffs=diffs,
        n_items=n,
        n_items_dropped=n_items_total - n,
        mean_a=float(np.mean([scores_a[item] for item in item_ids])),
        mean_b=float(np.mean([scores_b[item] for item in item_ids])),
        mean_diff=float(np.mean(diffs)),
        ci_low=ci_low,
        ci_high=ci_high,
        ci_level=CONFIDENCE,
        p_value=p_value,
        p_method="exhaustive" if exhaustive else "monte_carlo",
        n_resamples=n_resamples,
        n_permutations=2**n if exhaustive else n_permutations,
        alpha=ALPHA,
        target_power=TARGET_POWER,
        n_required_items=required,
        n_additional_items=None if required is None else max(0, required - n),
        seed=seed,
    )


def analyze_run(
    store: Store,
    run_id: str,
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = 0,
) -> AnalysisResult:
    """Extract per-item scores from a stored run and compare every variant pair.

    Partial data from a `failed` or `running` run is analyzed as-is (the store is
    append-only; what exists is valid). Raises ValueError when there is nothing to
    analyze: unknown run, no judge configured, no judgments, or a pair with no items.
    """
    run = store.get_run(run_id)
    if run is None:
        raise ValueError(f"run {run_id!r} not found")
    judge = json.loads(run["spec_json"]).get("judge")
    if judge is None:
        raise ValueError(f"run {run_id!r} has no judge configured; nothing to analyze")
    judgments = store.get_judgments(run_id)
    if not judgments:
        raise ValueError(f"run {run_id!r} has no judgments to analyze")
    samples = store.get_samples(run_id)
    conditions = store.get_conditions(run_id)
    mode = judge["mode"]
    if mode == "pairwise":
        table = pairwise_item_scores(judgments, samples, conditions)
    elif mode == "rubric":
        table = rubric_item_scores(judgments, samples, conditions)
    else:
        raise ValueError(f"unknown judge mode {mode!r}")
    variant_names = [row["variant_name"] for row in conditions]  # declared order
    comparisons = [
        _compare(
            variant_a,
            variant_b,
            table.scores[variant_a],
            table.scores[variant_b],
            n_items_total=table.n_items,
            n_resamples=n_resamples,
            n_permutations=n_permutations,
            seed=seed,
        )
        for variant_a, variant_b in combinations(variant_names, 2)
    ]
    return AnalysisResult(
        run_id=run_id,
        experiment_name=run["experiment_name"],
        mode=mode,
        scores=table.scores,
        comparisons=comparisons,
        n_items=table.n_items,
        n_judgments_used=table.n_judgments_used,
        n_judgments_errored=table.n_judgments_errored,
    )
