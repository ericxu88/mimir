"""Stats engine (docs/DESIGN.md §7) — paired bootstrap CI, sign-flip p-value, power estimate.

Analysis is a pure read of the store: `analyze_run` is the only store-taking entry
point and uses only the pinned getters. Per-item paired differences are always
score(later-declared variant) - score(earlier-declared variant), items sorted by
item_id so seeded resampling is reproducible across invocations. Judgment rows with
an error are skipped and counted; items lacking a usable score for both variants of
a comparison are dropped and counted. Multi-arm runs (M7): the correction family is
the run's C(k,2) comparisons — BH or Holm (default) adjusted p-values are always
populated (identity when m == 1); the raw p stays on the dataclass for math and
tests but is never rendered for a multi-arm family. Replicate-level extraction
feeds the variance decomposition (item-effect vs replicate noise -> "more items vs
more samples per item") and, for rubric runs, the 3-way score-variance shares.
"""

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
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
CORRECTION_METHODS: tuple[str, ...] = ("bh", "holm")
DEFAULT_CORRECTION: Literal["bh", "holm"] = "holm"

_VERDICT_VALUE = {"A": 1.0, "B": 0.0, "TIE": 0.5}

# z_a = 1.959963984540054, z_p = 0.8416212335729143 (stdlib inverse normal CDF).
# The SUM is hoisted — not the squares — so _required_items keeps the exact float
# expression ((z_a + z_p) * sd / |mean|) ** 2 pinned by the M3 power oracles.
_Z_SUM = NormalDist().inv_cdf(1.0 - ALPHA / 2.0) + NormalDist().inv_cdf(TARGET_POWER)


@dataclass(frozen=True)
class ScoreTable:
    """Per-item mean scores per variant, extracted from one run's stored rows."""

    scores: dict[str, dict[str, float]]  # variant_name -> item_id -> per-item mean score
    n_items: int  # distinct item_ids over the run's samples (the dataset as run)
    n_judgments_used: int
    n_judgments_errored: int


@dataclass(frozen=True)
class ReplicateTable:
    """The un-collapsed view behind ScoreTable: one score per (variant, item, replicate).

    `scores[variant][item_id][sample_index]`. Both samples of a runner-written
    pairwise judgment share `sample_index`, so `sample_a_id` recovers the replicate —
    the same join as judge_audit's kappa extraction (`_content_outcomes`),
    reimplemented here because judge_audit imports FROM stats. Rows are averaged
    WITHIN a replicate (the two position orders in pairwise mode, duplicate rows in
    either mode), never across replicates — collapsing across replicates is
    ScoreTable's job.
    """

    scores: dict[str, dict[str, dict[int, float]]]
    n_judgments_used: int
    n_judgments_errored: int


@dataclass(frozen=True)
class VarianceDecomposition:
    """Method-of-moments split of the paired difference into item-effect variance and
    replicate/sampling noise, and what that implies for the next run's budget.

    Estimated on the DIFF scale (the condition effect is the mean, not a variance
    component): d_ir = mean_diff + b_i + e_ir. `var_within` is the pooled one-way MSE
    over items with >= 2 replicates; `var_item_mean` is the ddof-1 variance of the
    per-item mean diffs (the sd^2 behind `Comparison.n_required_items`);
    `var_between` subtracts the sampling noise those means carry, floored at 0.
    `mean_replicates` is the HARMONIC mean of replicate counts, which makes
    var_between + var_within / mean_replicates == var_item_mean exact even when
    counts are unbalanced (pooling var_within across items stays an approximation
    there).
    """

    n_items: int
    n_items_with_replicates: int
    mean_replicates: float
    var_between: float
    var_within: float
    var_item_mean: float
    share_between: float
    n_required_items_current: int | None
    n_required_items_double: int | None
    n_required_items_limit: int | None
    recommendation: Literal["more_items", "more_samples_per_item"]


@dataclass(frozen=True)
class ScoreVarianceShares:
    """Run-level two-way method-of-moments split of raw-score variance into condition
    effect, item difficulty, and noise (interaction + replicate noise; confounded when
    every cell has a single replicate). Rubric mode only — pairwise scores are
    complementary (B = 1 - A) and the split degenerates. Conditions and items are
    treated as random effects for the share computation (a display approximation);
    complete-case items only (an item must have a score in every condition).
    """

    n_conditions: int
    n_items: int
    mean_replicates: float
    var_condition: float
    var_item: float
    var_noise: float
    share_condition: float
    share_item: float
    share_noise: float


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
    # M7 family/decomposition fields, DEFAULTED so hand-built Comparisons (report
    # tests) stay legal; analyze_run always populates the first three.
    p_value_corrected: float | None = None  # family-corrected p; == p_value when m == 1
    correction_method: Literal["bh", "holm"] | None = None
    n_comparisons: int = 1  # family size m = len(result.comparisons)
    variance: VarianceDecomposition | None = None  # None when replicates can't separate


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
    correction_method: Literal["bh", "holm"] | None = None
    score_variance: ScoreVarianceShares | None = None  # rubric only; pairwise degenerates


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
    return _required_items(float(np.mean(d)), float(np.std(d, ddof=1)))


def _required_items(mean: float, sd: float) -> int | None:
    """Items for TARGET_POWER at effect `mean` with per-item sd `sd`; None if not
    estimable.

    Same cancellation hazard as tau in sign_flip_pvalue: a perfectly balanced null
    built from non-dyadic scores (e.g. pairwise means quantized at 1/12) yields a
    mean of ~1e-17, not exactly 0.0, and the formula would report an astronomical
    n (or overflow ceil). An effect that far below the data scale is not estimable.
    NEVER rewrite the expression as z^2 * var / mean^2 — the algebraically equal
    form differs in the last ulp and can flip a ceil (M3 oracles pin this one).
    """
    if abs(mean) <= 1e-12 * max(1.0, sd):
        return None
    # A paired analysis needs at least 2 items, so a zero-variance effect floors there.
    return max(2, math.ceil((_Z_SUM * sd / abs(mean)) ** 2))


def _as_pvalues(p_values: Sequence[float] | np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=np.float64)
    if p.size == 0:
        raise ValueError("p_values is empty")
    if not np.isfinite(p).all():
        raise ValueError("p_values must be finite")
    if (p < 0.0).any() or (p > 1.0).any():
        raise ValueError("p_values must lie in [0, 1]")
    return p


def holm_bonferroni(p_values: Sequence[float] | np.ndarray) -> tuple[float, ...]:
    """Holm step-down adjusted p-values (family-wise error rate, any dependence).

    Sort ascending, scale the i-th smallest (0-based) by (m - i), enforce
    monotonicity with a running maximum (a later hypothesis can never be easier
    to reject than an earlier one), clip at 1, restore input order. Rejecting
    every hypothesis with adjusted p < alpha is exactly Holm's procedure. Ties
    get identical adjusted values through the running max — no special case.
    """
    p = _as_pvalues(p_values)
    m = p.size
    order = np.argsort(p, kind="stable")
    scaled = (m - np.arange(m)) * p[order]
    adjusted = np.maximum.accumulate(scaled)
    out = np.empty(m, dtype=np.float64)
    out[order] = np.minimum(adjusted, 1.0)
    return tuple(float(x) for x in out)


def benjamini_hochberg(p_values: Sequence[float] | np.ndarray) -> tuple[float, ...]:
    """Benjamini-Hochberg step-up adjusted p-values (false discovery rate).

    Sort ascending, scale the i-th smallest (1-based) by m / i, enforce
    monotonicity with a running minimum from the largest p downwards, clip at 1
    (defensive: the largest adjusted value is p_(m) itself), restore input
    order. Controls FDR under independence and positive regression dependence;
    a run's C(k,2) pairwise comparisons share arms and are positively dependent
    in practice but not provably PRDS — Holm (the default) is the
    assumption-free choice when a bogus winner must never ship.
    """
    p = _as_pvalues(p_values)
    m = p.size
    order = np.argsort(p, kind="stable")
    scaled = m * p[order] / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(scaled[::-1])[::-1]
    out = np.empty(m, dtype=np.float64)
    out[order] = np.minimum(adjusted, 1.0)
    return tuple(float(x) for x in out)


_CORRECTIONS = {"bh": benjamini_hochberg, "holm": holm_bonferroni}


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
        # Defensive clauses: the runner never writes a NULL/unknown verdict without
        # an error, nor a judgment referencing another run's sample (FK-valid but
        # drifted), and analysis must not crash on either.
        value = _VERDICT_VALUE.get(row["verdict"])
        presented_a = sample_variant.get(row["sample_a_id"])
        if row["error"] is not None or value is None or presented_a is None:
            errored += 1
            continue
        used += 1
        if presented_a != variant_a:
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
        # Third clause: a judgment referencing another run's sample is FK-valid
        # but drifted — skip-and-count, never KeyError (matches judge_audit).
        variant = sample_variant.get(row["sample_a_id"])
        if row["error"] is not None or row["score"] is None or variant is None:
            errored += 1
            continue
        used += 1
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


# --- M7: replicate-level extraction, variance decomposition, power planning -------


def pairwise_replicate_scores(
    judgments: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    conditions: list[dict[str, Any]],
) -> ReplicateTable:
    """Per-(item, replicate) pairwise scores, averaged over position orders only.

    Same verdict mapping and skip-and-count rules as pairwise_item_scores.
    """
    if len(conditions) != 2:
        raise ValueError(f"pairwise analysis requires exactly two variants, got {len(conditions)}")
    variant_a = conditions[0]["variant_name"]
    variant_b = conditions[1]["variant_name"]
    sample_variant = _sample_variants(samples, conditions)
    sample_index = {row["id"]: row["sample_index"] for row in samples}
    values_a: dict[tuple[str, int], list[float]] = {}
    used = errored = 0
    for row in judgments:
        value = _VERDICT_VALUE.get(row["verdict"])
        presented_a = sample_variant.get(row["sample_a_id"])
        if row["error"] is not None or value is None or presented_a is None:
            errored += 1
            continue
        used += 1
        if presented_a != variant_a:
            value = 1.0 - value  # presented-A was the later-declared variant
        values_a.setdefault((row["item_id"], sample_index[row["sample_a_id"]]), []).append(value)
    scores_a: dict[str, dict[int, float]] = {}
    scores_b: dict[str, dict[int, float]] = {}
    for (item, index), cell in values_a.items():
        mean = float(np.mean(cell))
        scores_a.setdefault(item, {})[index] = mean
        scores_b.setdefault(item, {})[index] = 1.0 - mean
    return ReplicateTable(
        scores={variant_a: scores_a, variant_b: scores_b},
        n_judgments_used=used,
        n_judgments_errored=errored,
    )


def rubric_replicate_scores(
    judgments: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    conditions: list[dict[str, Any]],
) -> ReplicateTable:
    """Per-(item, replicate) rubric scores: mean of `score` over duplicate rows."""
    sample_variant = _sample_variants(samples, conditions)
    sample_index = {row["id"]: row["sample_index"] for row in samples}
    values: dict[str, dict[tuple[str, int], list[float]]] = {
        row["variant_name"]: {} for row in conditions
    }
    used = errored = 0
    for row in judgments:
        variant = sample_variant.get(row["sample_a_id"])
        if row["error"] is not None or row["score"] is None or variant is None:
            errored += 1
            continue
        used += 1
        key = (row["item_id"], sample_index[row["sample_a_id"]])
        values[variant].setdefault(key, []).append(float(row["score"]))
    scores: dict[str, dict[str, dict[int, float]]] = {}
    for variant, per_cell in values.items():
        out: dict[str, dict[int, float]] = {}
        for (item, index), cell in per_cell.items():
            out.setdefault(item, {})[index] = float(np.mean(cell))
        scores[variant] = out
    return ReplicateTable(scores=scores, n_judgments_used=used, n_judgments_errored=errored)


def replicate_diffs(
    table: ReplicateTable, variant_a: str, variant_b: str
) -> dict[str, tuple[float, ...]]:
    """Per-item replicate-level differences variant_b - variant_a, ordered by
    sample_index. Items or replicate indices present for only one variant are
    dropped silently (drifted data must never crash analysis), so the result can
    cover fewer items than the collapsed comparison."""
    a = table.scores.get(variant_a, {})
    b = table.scores.get(variant_b, {})
    diffs: dict[str, tuple[float, ...]] = {}
    for item in sorted(a.keys() & b.keys()):
        shared = sorted(a[item].keys() & b[item].keys())
        if shared:
            diffs[item] = tuple(b[item][index] - a[item][index] for index in shared)
    return diffs


def decompose_variance(
    diffs_by_item: dict[str, Sequence[float]], *, mean_diff: float
) -> VarianceDecomposition | None:
    """Split replicate-level per-item differences; None when not separable
    (fewer than 2 items, or no item has >= 2 replicates — the n_samples: 1 case).

    Items are visited in sorted key order so the result never depends on caller
    insertion order. `mean_diff` is the caller's canonical effect estimate
    (Comparison.mean_diff), deliberately not recomputed: with unbalanced replicate
    counts it differs from the mean of per-item means.
    """
    items = {key: _as_diffs(diffs_by_item[key]) for key in sorted(diffs_by_item)}
    if len(items) < 2:
        return None
    counts = np.array([d.size for d in items.values()], dtype=np.float64)
    df = int((counts - 1).sum())
    if df == 0:
        return None
    means = np.array([float(d.mean()) for d in items.values()])
    ss = sum(float(((d - d.mean()) ** 2).sum()) for d in items.values())
    var_within = ss / df  # pooled one-way MSE, correctly weighted when unbalanced
    var_item_mean = float(np.var(means, ddof=1))
    inv_r = float(np.mean(1.0 / counts))  # E[MS_between] = var_b + var_w * mean(1/r_i)
    var_between = max(0.0, var_item_mean - var_within * inv_r)
    total_now = var_between + var_within * inv_r
    return VarianceDecomposition(
        n_items=len(items),
        n_items_with_replicates=int((counts > 1).sum()),
        mean_replicates=1.0 / inv_r,
        var_between=var_between,
        var_within=var_within,
        var_item_mean=var_item_mean,
        share_between=var_between / total_now if total_now > 0.0 else 1.0,
        n_required_items_current=_required_items(mean_diff, math.sqrt(total_now)),
        n_required_items_double=_required_items(
            mean_diff, math.sqrt(var_between + var_within * inv_r / 2.0)
        ),
        n_required_items_limit=_required_items(mean_diff, math.sqrt(var_between)),
        # Tie -> more_items: more items always helps, more replicates can never
        # shrink var_between, so items are the safe default when the split is a wash.
        recommendation=(
            "more_samples_per_item" if var_within * inv_r > var_between else "more_items"
        ),
    )


def score_variance_shares(table: ReplicateTable) -> ScoreVarianceShares | None:
    """Two-way MoM over the (condition, item, replicate) score array; None when
    fewer than 2 conditions or 2 complete-case items, or when all scores are
    constant (no variance to share)."""
    variants = sorted(table.scores)
    if len(variants) < 2:
        return None
    items = sorted(set.intersection(*(set(table.scores[v]) for v in variants)))
    if len(items) < 2:
        return None
    n_conditions, n_items = len(variants), len(items)
    cells = [[list(table.scores[v][item].values()) for item in items] for v in variants]
    x = np.array([[float(np.mean(cell)) for cell in row] for row in cells])
    counts = np.array([[len(cell) for cell in row] for row in cells], dtype=np.float64)
    inv_r = float(np.mean(1.0 / counts))
    grand = float(x.mean())
    cond_means = x.mean(axis=1)
    item_means = x.mean(axis=0)
    ms_cond = n_items * float(((cond_means - grand) ** 2).sum()) / (n_conditions - 1)
    ms_item = n_conditions * float(((item_means - grand) ** 2).sum()) / (n_items - 1)
    residuals = x - cond_means[:, None] - item_means[None, :] + grand
    ms_res = float((residuals**2).sum()) / ((n_conditions - 1) * (n_items - 1))
    df_within = int((counts - 1).sum())
    if df_within > 0:
        ss_within = sum(
            float(((np.asarray(cell) - np.mean(cell)) ** 2).sum()) for row in cells for cell in row
        )
        var_e = ss_within / df_within
        var_noise = max(0.0, ms_res - var_e * inv_r) + var_e
    else:
        var_noise = ms_res  # r == 1: interaction and replicate noise confounded
    var_condition = max(0.0, (ms_cond - ms_res) / n_items)
    var_item = max(0.0, (ms_item - ms_res) / n_conditions)
    total = var_condition + var_item + var_noise
    if total <= 0.0:
        return None
    return ScoreVarianceShares(
        n_conditions=n_conditions,
        n_items=n_items,
        mean_replicates=1.0 / inv_r,
        var_condition=var_condition,
        var_item=var_item,
        var_noise=var_noise,
        share_condition=var_condition / total,
        share_item=var_item / total,
        share_noise=var_noise / total,
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
    replicate_diffs_by_item: dict[str, tuple[float, ...]] | None = None,
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
    mean_diff = float(np.mean(diffs))
    variance = (
        None
        if replicate_diffs_by_item is None
        else decompose_variance(replicate_diffs_by_item, mean_diff=mean_diff)
    )
    return Comparison(
        variant_a=variant_a,
        variant_b=variant_b,
        item_ids=item_ids,
        diffs=diffs,
        n_items=n,
        n_items_dropped=n_items_total - n,
        mean_a=float(np.mean([scores_a[item] for item in item_ids])),
        mean_b=float(np.mean([scores_b[item] for item in item_ids])),
        mean_diff=mean_diff,
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
        variance=variance,
    )


def analyze_run(
    store: Store,
    run_id: str,
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = 0,
    correction: Literal["bh", "holm"] = DEFAULT_CORRECTION,
) -> AnalysisResult:
    """Extract per-item scores from a stored run and compare every variant pair.

    Partial data from a `failed` or `running` run is analyzed as-is (the store is
    append-only; what exists is valid). Raises ValueError when there is nothing to
    analyze: unknown run, no judge configured, no judgments, or a pair with no items.
    The correction family is the run's C(k,2) comparisons; `p_value_corrected` is
    always populated (identity when m == 1). `score_variance` is rubric-only:
    pairwise scores are complementary and the 3-way split degenerates.
    """
    if correction not in _CORRECTIONS:
        raise ValueError(
            f"unknown correction {correction!r}; expected one of {', '.join(CORRECTION_METHODS)}"
        )
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
        replicates = pairwise_replicate_scores(judgments, samples, conditions)
        score_variance = None
    elif mode == "rubric":
        table = rubric_item_scores(judgments, samples, conditions)
        replicates = rubric_replicate_scores(judgments, samples, conditions)
        score_variance = score_variance_shares(replicates)
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
            replicate_diffs_by_item=replicate_diffs(replicates, variant_a, variant_b),
        )
        for variant_a, variant_b in combinations(variant_names, 2)
    ]
    # The family statistic is only knowable once every pair is built, and
    # Comparison is frozen: build, then rewrite the three family fields in place.
    adjusted = _CORRECTIONS[correction]([c.p_value for c in comparisons])
    comparisons = [
        replace(c, p_value_corrected=q, correction_method=correction, n_comparisons=len(adjusted))
        for c, q in zip(comparisons, adjusted, strict=True)
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
        correction_method=correction,
        score_variance=score_variance,
    )
