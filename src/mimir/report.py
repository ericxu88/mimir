"""Report rendering (docs/DESIGN.md §9) — terminal text + single-file static HTML.

Pure renderers over the analysis/audit dataclasses: no store access, no I/O, no
timestamps — `render_html(x) == render_html(x)` is pinned by test. The CLI composes
(fetches, warns on status, writes files). SVG geometry and number formats are pinned
constants: exact-string test oracles depend on them, so a "nicer" refactor must fix
the oracle inputs, never loosen the assertions. Terminal output is ASCII-only.

All user-sourced text (experiment/variant names, run ids, judge model, notes) passes
through the single `_esc` chokepoint before landing in HTML.
"""

import html
import math
from collections.abc import Sequence

from mimir.judge_audit import JudgeReportCard
from mimir.stats import AnalysisResult, Comparison

# --- SVG histogram geometry (pinned; tests hand-compute attributes from these) ----

_SLOT_W = 24  # px per bin slot
_BAR_W = 22  # bar width; 1px inset each side -> 2px surface gap between bars
_PLOT_H = 80  # full bar height
_TOP = 8  # headroom above the tallest bar
_MARGIN = 8  # left/right margins
_BASE_Y = _TOP + _PLOT_H  # baseline y = 88
_SVG_H = 108  # baseline + label strip

_MAX_SERIES = 8  # categorical palette slots; later variants fall back to --muted


def histogram_bins(values: Sequence[float], *, lo: float, hi: float, n_bins: int) -> list[int]:
    """Bin counts over [lo, hi]; the hi edge and out-of-range values clamp into the
    end bins (drifted scores must never crash a report)."""
    if not values:
        raise ValueError("no values to bin")
    counts = [0] * n_bins
    for value in values:
        if not math.isfinite(value):
            # Hand-crafted DBs can hold +/-inf REALs (SQLite stores NaN as NULL,
            # so NaN never reaches here); clamp by sign instead of OverflowError.
            counts[0 if value < lo else n_bins - 1] += 1
            continue
        index = max(0, min(n_bins - 1, int((value - lo) / (hi - lo) * n_bins)))
        counts[index] += 1
    return counts


def _edge(lo: float, hi: float, n_bins: int, i: int) -> str:
    return format(lo + i * (hi - lo) / n_bins, "g")


def svg_histogram(values: Sequence[float], *, lo: float, hi: float, n_bins: int, fill: str) -> str:
    """One single-series histogram panel; identity is carried by the caller's caption.

    Integer-exact geometry: bar height = max(1, count * 80 // max(counts)); zero
    bins emit no rect. Each rect carries a <title> (native no-JS tooltip)."""
    counts = histogram_bins(values, lo=lo, hi=hi, n_bins=n_bins)
    peak = max(counts)
    total = len(values)
    right = _MARGIN + n_bins * _SLOT_W
    width = right + _MARGIN
    label_y = _SVG_H - 4
    parts = [
        f'<svg width="{width}" height="{_SVG_H}" viewBox="0 0 {width} {_SVG_H}"'
        ' role="img" aria-label="score distribution histogram">'
    ]
    for i, count in enumerate(counts):
        if count == 0:
            continue
        bar_h = max(1, count * _PLOT_H // peak)
        x = _MARGIN + i * _SLOT_W + 1
        y = _BASE_Y - bar_h
        close = "]" if i == n_bins - 1 else ")"
        interval = f"[{_edge(lo, hi, n_bins, i)}, {_edge(lo, hi, n_bins, i + 1)}{close}"
        parts.append(
            f'<rect x="{x}" y="{y}" width="{_BAR_W}" height="{bar_h}" fill="{fill}">'
            f"<title>{count} of {total} scores in {interval}</title></rect>"
        )
    parts.append(
        f'<line x1="{_MARGIN}" y1="{_BASE_Y}" x2="{right}" y2="{_BASE_Y}"'
        ' stroke="var(--axis)" stroke-width="1"/>'
    )
    ticks = (
        (_MARGIN, "start", _edge(lo, hi, n_bins, 0)),
        ((_MARGIN + right) // 2, "middle", format(lo + (hi - lo) / 2, "g")),
        (right, "end", _edge(lo, hi, n_bins, n_bins)),
    )
    for x, anchor, label in ticks:
        parts.append(
            f'<text x="{x}" y="{label_y}" text-anchor="{anchor}"'
            f' fill="var(--muted)" font-size="11">{label}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


# --- shared formatting ------------------------------------------------------------


def _fmt(value: float | None, spec: str = ".3f") -> str:
    return "n/a" if value is None else format(value, spec)


# Display names are the renderer's business; .get fallback so a future method
# can never KeyError a report.
_CORRECTION_LABELS = {"bh": "Benjamini-Hochberg FDR", "holm": "Holm-Bonferroni"}


def _p_value_text(c: Comparison, *, html: bool = False) -> str:
    # M7 mandate: a multi-arm family NEVER renders the raw p — corrected only.
    if c.n_comparisons > 1 and c.p_value_corrected is not None:
        family = f", {c.n_comparisons} comparisons" if html else ""
        return (
            f"{c.p_value_corrected:.4f} ({c.correction_method}-corrected{family},"
            f" {c.p_method}, {c.n_permutations} permutations)"
        )
    return f"{c.p_value:.4f} ({c.p_method}, {c.n_permutations} permutations)"


def _ci_text(c: Comparison) -> str:
    """The interval, or an honest refusal — never a zero-width 95% interval."""
    if c.ci_low is None or c.ci_high is None:
        return "not estimable (every difference identical)"
    return f"[{_fmt(c.ci_low)}, {_fmt(c.ci_high)}] ({c.ci_method} bootstrap)"


def _resolution_note(c: Comparison) -> str | None:
    """A permutation test over n items cannot return a p below 2/2**n; if that floor
    is above alpha the design can never reject, and silence would read as evidence
    of no effect."""
    smallest = 2 / 2**c.n_items if c.p_method == "exhaustive" else 1 / (1 + c.n_permutations)
    if smallest <= c.alpha:
        return None
    return (
        f"{c.n_items} items cannot reach significance at alpha={c.alpha:g}"
        f" (smallest achievable p = {smallest:.4f})"
    )


def _allocation_text(c: Comparison) -> tuple[str, str] | None:
    v = c.variance
    if v is None:
        return None
    # M8/M4: name the scale — these are per-difference variance components, not the
    # raw-score shares printed in the run-level block above.
    split = (
        f"item {v.var_between:.3f}, replicate noise {v.var_within:.3f}"
        f" (paired-difference scale, {v.mean_replicates:.1f} samples per item)"
    )
    if v.recommendation == "more_samples_per_item":
        phrase = "more samples per item"
    else:
        phrase = "more items"
    if v.n_required_items_double is None:
        detail = "items needed not estimable"
    else:
        detail = (
            f"{v.n_required_items_double} items at {2 * v.mean_replicates:.1f} samples,"
            f" {v.n_required_items_limit} at unlimited"
        )
    return split, f"{phrase} ({detail})"


def _score_axis(mode: str) -> tuple[float, float, int]:
    # Pairwise per-item scores live in [0, 1]; rubric scores in [1, 10].
    if mode == "rubric":
        return (1.0, 10.0, 9)
    return (0.0, 1.0, 10)


# --- terminal renderers -----------------------------------------------------------


def _comparison_lines(c: Comparison) -> list[str]:
    def row(label: str, value: str) -> str:
        return f"  {label:<18}{value}"

    if c.n_additional_items is None:
        power = "not estimable (mean diff indistinguishable from zero or n < 2)"
    else:
        power = (
            f"{c.n_required_items} items needed for {round(c.target_power * 100)}% power"
            f" ({c.n_additional_items} more than paired)"
        )
        if c.power_alpha != c.alpha:
            # M8/M2: name the level the plan was made at, since the verdict above it
            # is judged on a corrected p and the two must match.
            power += f", planned at alpha={c.power_alpha:.4g} for {c.n_comparisons} comparisons"
    lines = [
        f"{c.variant_b} vs {c.variant_a} (diff = {c.variant_b} - {c.variant_a})",
        row("paired items:", f"{c.n_items} ({c.n_items_dropped} dropped)"),
        row(f"mean {c.variant_a}:", _fmt(c.mean_a)),
        row(f"mean {c.variant_b}:", _fmt(c.mean_b)),
        row("mean diff:", _fmt(c.mean_diff)),
        row(f"{round(c.ci_level * 100)}% CI:", _ci_text(c)),
        row("p-value:", _p_value_text(c)),
    ]
    note = _resolution_note(c)
    if note is not None:
        lines.append(row("note:", note))  # qualifies the p-value directly above
    lines.append(row("power:", power))
    allocation = _allocation_text(c)
    if allocation is not None:
        split, advice = allocation
        lines.append(row("variance split:", split))
        lines.append(row("allocation:", advice))
    return lines


def render_analysis_text(result: AnalysisResult, *, status: str | None = None) -> str:
    """Plain-text analysis report; `status` (from the runs table) is optional."""
    run_line = f"run: {result.run_id} ({result.mode}"
    run_line += f", status: {status})" if status is not None else ")"
    lines = [
        f"experiment: {result.experiment_name}",
        run_line,
        f"items: {result.n_items} | judgments used: {result.n_judgments_used}"
        f" | errored: {result.n_judgments_errored}",
        "",
        "variants:",
    ]
    for variant, scores in result.scores.items():
        if scores:
            mean = sum(scores.values()) / len(scores)
            detail = f"mean {mean:.3f}  ({len(scores)} items scored)"
        else:
            detail = "no scores"
        lines.append(f"  {variant:<18}{detail}")
    shares = result.score_variance
    if shares is not None:
        lines.append("")
        lines.append(
            f"  {'variance shares:':<18}condition {shares.share_condition:.1%},"
            f" item {shares.share_item:.1%}, noise {shares.share_noise:.1%}"
            " (raw-score scale)"
        )
    if len(result.comparisons) > 1 and result.correction_method is not None:
        label = _CORRECTION_LABELS.get(result.correction_method, result.correction_method)
        lines.append("")
        lines.append(
            f"multiple comparisons: {len(result.comparisons)} pairs,"
            f" p-values corrected ({label}); CIs are uncorrected"
        )
    for comparison in result.comparisons:
        lines.append("")
        lines.extend(_comparison_lines(comparison))
    if result.comparisons:
        lines.append("")
        lines.append(
            # ASCII only (cp1252 consoles), and short enough not to wrap at 100.
            "interval: studentized bootstrap; p-value: sign-flip permutation - they can disagree"
        )
    return "\n".join(lines)


def render_audit_text(card: JudgeReportCard) -> str:
    """Plain-text judge report card; None fields render as n/a, notes verbatim."""

    def row(label: str, value: str) -> str:
        return f"  {label:<21}{value}"

    flip = _fmt(card.flip_rate)
    if card.n_pairs is not None:
        flip += f" ({card.n_pairs} pairs, {card.n_pairs_dropped} dropped)"
    length = (
        f"slope {_fmt(card.length_slope, '.4g')},"
        f" correlation {_fmt(card.length_correlation)}"
        f" ({card.n_length_points} points)"
    )
    lines = [
        "judge report card",
        f"run: {card.run_id}",
        f"judge: {card.judge_model} ({card.mode})",
        f"judgments used: {card.n_judgments_used} | errored: {card.n_judgments_errored}",
        row("flip rate:", flip),
        row("position-A win rate:", _fmt(card.position_a_win_rate)),
        row("length bias:", length),
    ]
    if card.compare_run_id is not None:
        kappa = f"{_fmt(card.kappa)} vs run {card.compare_run_id} (n={card.kappa_n})"
        lines.append(row("cross-judge kappa:", kappa))
    if card.notes:
        lines.append("notes:")
        lines.extend(f"  - {note}" for note in card.notes)
    return "\n".join(lines)


# --- HTML report ------------------------------------------------------------------


def _esc(text: object) -> str:
    return html.escape(str(text), quote=True)


# Palette: the dataviz reference instance, verbatim (validated for both modes;
# light-mode sub-3:1 series slots are covered by the relief rule — every chart
# ships beside its numbers as text/tables). Dark mode is pure CSS.
_STYLE = """\
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface-1: #fcfcfb;
  --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781; --axis: #c3c2b7;
  --critical: #d03b3b;
  --series-1: #2a78d6; --series-2: #eb6834; --series-3: #1baf7a;
  --series-4: #eda100; --series-5: #e87ba4; --series-6: #008300;
  --series-7: #4a3aa7; --series-8: #e34948;
}
@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark;
    --page: #0d0d0d; --surface-1: #1a1a19;
    --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781; --axis: #383835;
    --critical: #d03b3b;
    --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70;
    --series-4: #c98500; --series-5: #d55181; --series-6: #008300;
    --series-7: #9085e9; --series-8: #e66767;
  }
}
body {
  margin: 0; background: var(--page); color: var(--ink);
  font: 15px/1.5 system-ui, sans-serif;
}
main { max-width: 720px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 16px; margin: 28px 0 8px; }
h3 { font-size: 14px; margin: 16px 0 4px; }
.muted { color: var(--ink-2); }
.warn {
  border-left: 3px solid var(--critical);
  background: var(--surface-1); padding: 8px 12px; margin: 12px 0;
}
table { border-collapse: collapse; background: var(--surface-1); }
td, th {
  border: 1px solid var(--axis); padding: 4px 10px;
  font-size: 13px; text-align: left;
}
figure { margin: 12px 0; }
figcaption { font-size: 13px; margin-bottom: 4px; }
.chip {
  display: inline-block; width: 12px; height: 12px;
  border-radius: 2px; vertical-align: -1px; margin-right: 6px;
}
ul.notes { font-size: 13px; color: var(--ink-2); }
"""


def _summary_html(result: AnalysisResult) -> str:
    return (
        "<section><table><tr><th>items</th><th>judgments used</th><th>errored</th></tr>"
        f"<tr><td>{result.n_items}</td><td>{result.n_judgments_used}</td>"
        f"<td>{result.n_judgments_errored}</td></tr></table></section>"
    )


def _distributions_html(result: AnalysisResult) -> str:
    lo, hi, n_bins = _score_axis(result.mode)
    parts = ["<section><h2>score distributions</h2>"]
    for index, (variant, scores) in enumerate(result.scores.items()):
        fill = f"var(--series-{index + 1})" if index < _MAX_SERIES else "var(--muted)"
        parts.append("<figure>")
        if scores:
            mean = sum(scores.values()) / len(scores)
            caption = (
                f'<span class="chip" style="background:{fill}"></span>{_esc(variant)}'
                f' <span class="muted">mean {mean:.3f} - {len(scores)} items</span>'
            )
            parts.append(f"<figcaption>{caption}</figcaption>")
            parts.append(
                svg_histogram(list(scores.values()), lo=lo, hi=hi, n_bins=n_bins, fill=fill)
            )
        else:
            parts.append(f"<figcaption>{_esc(variant)}</figcaption>")
            parts.append('<p class="muted">no scores for this variant</p>')
        parts.append("</figure>")
    parts.append("</section>")
    return "".join(parts)


def _comparison_html(c: Comparison) -> str:
    if c.n_additional_items is None:
        power = "not estimable"
    else:
        power = (
            f"{c.n_required_items} items for {round(c.target_power * 100)}% power"
            f" ({c.n_additional_items} more than paired)"
        )
        if c.power_alpha != c.alpha:
            power += f", at alpha={c.power_alpha:.4g} for {c.n_comparisons} comparisons"
    rows = [
        (f"mean {_esc(c.variant_a)}", _fmt(c.mean_a)),
        (f"mean {_esc(c.variant_b)}", _fmt(c.mean_b)),
        ("mean diff", _fmt(c.mean_diff)),
        (f"{round(c.ci_level * 100)}% CI", _esc(_ci_text(c))),
        ("p-value", _p_value_text(c, html=True)),
        ("paired items", f"{c.n_items} ({c.n_items_dropped} dropped)"),
        ("power", power),
    ]
    note = _resolution_note(c)
    if note is not None:
        rows.append(("resolution", _esc(note)))
    allocation = _allocation_text(c)
    if allocation is not None:
        v = c.variance
        rows.append(("variance, item (diff scale)", f"{v.var_between:.3f}"))
        rows.append(("variance, replicate noise (diff scale)", f"{v.var_within:.3f}"))
        rows.append(("samples per item", f"{v.mean_replicates:.1f}"))
        rows.append(("allocation", _esc(allocation[1])))
    cells = "".join(f"<tr><th>{label}</th><td>{value}</td></tr>" for label, value in rows)
    corrected = c.n_comparisons > 1 and c.p_value_corrected is not None
    p_shown = c.p_value_corrected if corrected else c.p_value
    suffix = f" after {_esc(c.correction_method)} correction" if corrected else ""
    verdict = "yes" if p_shown < c.alpha else "no"
    return (
        f"<h3>{_esc(c.variant_b)} vs {_esc(c.variant_a)}</h3><table>{cells}</table>"
        f"<p>significant at alpha={c.alpha:g}{suffix}: {verdict}</p>"
    )


def _judge_html(card: JudgeReportCard) -> str:
    flip = _fmt(card.flip_rate)
    if card.n_pairs is not None:
        flip += f" ({card.n_pairs} pairs, {card.n_pairs_dropped} dropped)"
    rows = [
        ("judge model", _esc(card.judge_model)),
        ("mode", _esc(card.mode)),
        ("judgments used", str(card.n_judgments_used)),
        ("judgments errored", str(card.n_judgments_errored)),
        ("flip rate", flip),
        ("position-A win rate", _fmt(card.position_a_win_rate)),
        ("length slope", _fmt(card.length_slope, ".4g")),
        ("length correlation", _fmt(card.length_correlation)),
        ("length points", str(card.n_length_points)),
    ]
    if card.compare_run_id is not None:
        rows.append(("cross-judge kappa", f"{_fmt(card.kappa)} (n={card.kappa_n})"))
        rows.append(("compared with run", _esc(card.compare_run_id)))
    cells = "".join(f"<tr><th>{label}</th><td>{value}</td></tr>" for label, value in rows)
    notes = ""
    if card.notes:
        items = "".join(f"<li>{_esc(note)}</li>" for note in card.notes)
        notes = f'<ul class="notes">{items}</ul>'
    return f"<section><h2>judge report card</h2><table>{cells}</table>{notes}</section>"


def render_html(
    result: AnalysisResult,
    card: JudgeReportCard | None = None,
    *,
    status: str | None = None,
) -> str:
    """One self-contained HTML report: inline CSS, inline SVG, no JS, no external refs."""
    status_bit = f" - status {_esc(status)}" if status is not None else ""
    header = (
        f"<header><h1>{_esc(result.experiment_name)}</h1>"
        f'<p class="muted">run {_esc(result.run_id)} - {_esc(result.mode)}{status_bit}</p></header>'
    )
    warning = ""
    if status is not None and status != "complete":
        warning = (
            f'<p class="warn">warning: run status is {_esc(status)};'
            " this report reflects partial data</p>"
        )
    shares = result.score_variance
    shares_html = ""
    if shares is not None:
        shares_html = (
            f'<p class="muted">variance shares (raw-score scale):'
            f" condition {shares.share_condition:.1%},"
            f" item {shares.share_item:.1%}, noise {shares.share_noise:.1%}</p>"
        )
    family = ""
    if len(result.comparisons) > 1 and result.correction_method is not None:
        label = _CORRECTION_LABELS.get(result.correction_method, result.correction_method)
        family = (
            f'<p class="muted">multiple comparisons: {len(result.comparisons)} pairs,'
            f" p-values corrected ({_esc(label)}); CIs are uncorrected</p>"
        )
    procedures = (
        '<p class="muted">the interval is a studentized bootstrap and the p-value is a'
        " sign-flip permutation test — they can disagree</p>"
        if result.comparisons
        else ""
    )
    comparisons = (
        "<section><h2>comparisons</h2>"
        + family
        + "".join(_comparison_html(c) for c in result.comparisons)
        + procedures
        + "</section>"
    )
    judge = _judge_html(card) if card is not None else ""
    title = f"mimir report - {_esc(result.experiment_name)} - {_esc(result.run_id)}"
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{title}</title><style>{_STYLE}</style></head><body><main>"
        f"{header}{warning}{_summary_html(result)}{shares_html}{_distributions_html(result)}"
        f"{comparisons}{judge}"
        "</main></body></html>"
    )
