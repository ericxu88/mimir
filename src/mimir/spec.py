"""Pydantic models for the YAML experiment spec (DESIGN.md §2 — normative as of M2).

Spec defaults (temperature 1.0, max_tokens 1024, seed 0, n_samples 1, limits 4/60)
resolve HERE, before any hashing, so equal effective specs always produce equal
cache keys. The golden keys in tests/test_cache.py pin this indirectly.
"""

import json
import re
import string
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_DEFAULT_PAIRWISE_TEMPLATE = (
    "Question: {input}\n\n"
    "Response A:\n{response_a}\n\n"
    "Response B:\n{response_b}\n\n"
    "Which response is better? Reply with exactly A, B, or TIE on the final line."
)
_DEFAULT_RUBRIC_TEMPLATE = (
    "Question: {input}\n\n"
    "Response:\n{response}\n\n"
    "Rate the response quality from 1 to 10. Reply with only the integer on the final line."
)

# Placeholders judge templates may use beyond the dataset item's fields, per mode.
# These names are also RESERVED per mode: dataset items may not use them as field
# names when that judge mode is configured (they collide with the runner's
# str.format kwargs when the judge prompt is rendered).
_JUDGE_PLACEHOLDERS_BY_MODE = {
    "pairwise": frozenset({"response_a", "response_b"}),
    "rubric": frozenset({"response"}),
}


def _reject_lone_surrogates(value: str, what: str) -> str:
    # json.loads (and YAML) accept lone-surrogate escapes that canonical_json's
    # utf-8 encode rejects; caught at spec load they are one clear error instead of
    # an error row per unit (M9).
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError(
            f"{what} contains a lone surrogate (not valid UTF-8; it would poison"
            " cache keys and stored spec JSON)"
        ) from None
    return value


# Placeholder available to command-variant argv templates beyond the item's own
# fields; also RESERVED as a dataset field name when any command variant exists.
_COMMAND_PLACEHOLDERS = frozenset({"seed"})

# Import path for python variants: dotted module, colon, attribute.
_IMPORT_PATH_RE = re.compile(r"^[A-Za-z_]\w*(\.[A-Za-z_]\w*)*:[A-Za-z_]\w*$")


class Variant(BaseModel):
    """An LLM prompt condition (M10: one of three condition types)."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["llm"] = "llm"
    name: str
    system: str = ""
    user_template: str

    @field_validator("system", "user_template")
    @classmethod
    def _templates_utf8(cls, value: str, info) -> str:
        return _reject_lone_surrogates(value, f"variant {info.field_name}")


class CommandVariant(BaseModel):
    """A subprocess condition: argv-list template, no shell (M10).

    Each element is str.format-rendered with the item's fields plus {seed};
    rendered argv enters the cache key, so elements get the same UTF-8 guard
    as prompt templates. `timeout_s` is an execution limit, deliberately OUT
    of the cache key (DESIGN §3).
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["command"]
    name: str
    command: list[str] = Field(min_length=1)
    timeout_s: float = Field(default=60.0, gt=0, allow_inf_nan=False)

    @field_validator("command")
    @classmethod
    def _argv_utf8(cls, command: list[str]) -> list[str]:
        for element in command:
            _reject_lone_surrogates(element, "command argv element")
        return command


class PythonVariant(BaseModel):
    """A python-callable condition referenced by import path (M10)."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["python"]
    name: str
    callable: str

    @field_validator("callable")
    @classmethod
    def _import_path_format(cls, value: str) -> str:
        if not _IMPORT_PATH_RE.match(value):
            raise ValueError(f'callable must look like "pkg.module:function", got {value!r}')
        return value


class Dataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str | None = None
    items: list[dict[str, Any]] | None = None

    @field_validator("items")
    @classmethod
    def _items_json_safe(cls, items: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        # YAML parses unquoted dates/timestamps into datetime objects, which would
        # later break canonical JSON in create_run with a bare TypeError.
        if items is None:
            return items
        for index, item in enumerate(items):
            for field, value in item.items():
                try:
                    # Probe must match canonical_json semantics (ensure_ascii=False +
                    # utf-8 encode): default json.dumps accepts lone surrogates that
                    # would crash create_run.
                    json.dumps(value, ensure_ascii=False).encode("utf-8")
                except (TypeError, ValueError):
                    raise ValueError(
                        f"dataset item {item.get('id', index)!r} field {field!r}: value"
                        f" {value!r} is not JSON-serializable UTF-8 (quote YAML"
                        " dates/timestamps as strings; lone surrogates are rejected)"
                    ) from None
        return items

    @model_validator(mode="after")
    def _exactly_one_source(self) -> Self:
        if (self.path is None) == (self.items is None):
            raise ValueError("dataset requires exactly one of `path` or `items`")
        return self


class Sampling(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # None is valid only for specs with no llm variants (cross-field rule below);
    # command/python conditions have no model, temperature, or token budget.
    model: str | None = None
    temperature: float = Field(default=1.0, allow_inf_nan=False)
    max_tokens: int = Field(default=1024, ge=1)
    seed: int = 0


class Judge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["llm"] = "llm"
    model: str
    mode: Literal["pairwise", "rubric"]
    temperature: float = Field(default=0.0, allow_inf_nan=False)
    max_tokens: int = Field(default=512, ge=1)
    prompt_template: str | None = None
    position_swap: bool = True

    @field_validator("prompt_template")
    @classmethod
    def _template_utf8(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _reject_lone_surrogates(value, "judge prompt_template")

    def resolved_prompt_template(self) -> str:
        if self.prompt_template is not None:
            return self.prompt_template
        if self.mode == "pairwise":
            return _DEFAULT_PAIRWISE_TEMPLATE
        return _DEFAULT_RUBRIC_TEMPLATE


class ParseFloatScorerSpec(BaseModel):
    """Score each sample by parsing a float from its output text (M10).

    `mode` is the scoring SHAPE stats.py dispatches on: per-sample scalar scores
    are rubric-shaped, so the dumped spec_json satisfies analyze_run unchanged.
    Deliberately no `model` field — audit-judge on such a run raises its existing
    "judge block is missing 'mode' or 'model'" ValueError, which is honest: there
    is no LLM judge to audit.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["parse_float"]
    mode: Literal["rubric"] = "rubric"


# Discriminated unions: `type` selects the member. Untyped dicts get type "llm"
# injected by ExperimentSpec's before-validator so pre-M10 specs load unchanged.
AnyVariant = Annotated[Variant | CommandVariant | PythonVariant, Field(discriminator="type")]
AnyJudge = Annotated[Judge | ParseFloatScorerSpec, Field(discriminator="type")]

# v1 analyzes at stats.py's fixed module constant; a cross-module test pins the
# two literals equal (never import stats here — numpy at spec load, layering).
_PLAN_ALPHA = 0.05


class AnalysisPlan(BaseModel):
    """Pre-registered analysis plan (M11, DESIGN §14).

    `primary` names THE comparison the hypothesis rides on; `comparisons` are
    additional planned pairs. Pairs match analysis output as UNORDERED name
    sets (the report's diff orientation stays declaration-order). The resolved
    plan is part of the preregistration hash.
    """

    model_config = ConfigDict(extra="forbid")

    primary: tuple[str, str]
    comparisons: list[tuple[str, str]] = []
    alpha: float = _PLAN_ALPHA
    correction: Literal["holm", "bh"] = "holm"

    @field_validator("alpha")
    @classmethod
    def _alpha_is_the_module_constant(cls, value: float) -> float:
        if value != _PLAN_ALPHA:
            raise ValueError(
                f"alpha must be {_PLAN_ALPHA} in v1 — analysis runs at stats.py's fixed"
                " module constant (DESIGN §7); a configurable alpha is deferred (DESIGN §12)"
            )
        return value


class Limits(BaseModel):
    model_config = ConfigDict(extra="forbid")

    concurrency: int = Field(default=4, ge=1)
    requests_per_minute: int = Field(default=60, ge=1)


class ExperimentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    variants: list[AnyVariant] = Field(min_length=1)
    dataset: Dataset
    sampling: Sampling = Field(default_factory=Sampling)
    n_samples: int = Field(default=1, ge=1)
    judge: AnyJudge | None = None
    hypothesis: str | None = None
    analysis_plan: AnalysisPlan | None = None
    limits: Limits = Field(default_factory=Limits)

    @field_validator("hypothesis")
    @classmethod
    def _hypothesis_utf8(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _reject_lone_surrogates(value, "hypothesis")

    @model_validator(mode="before")
    @classmethod
    def _default_types(cls, data: Any) -> Any:
        # Discriminated unions need the tag present; pre-M10 specs have no `type`
        # keys. Inject "llm" on COPIES only — callers reuse their spec dicts.
        if not isinstance(data, dict):
            return data
        variants = data.get("variants")
        judge = data.get("judge")
        fix_variants = isinstance(variants, list) and any(
            isinstance(v, dict) and "type" not in v for v in variants
        )
        fix_judge = isinstance(judge, dict) and "type" not in judge
        if not (fix_variants or fix_judge):
            return data
        data = dict(data)
        if fix_variants:
            data["variants"] = [
                {**v, "type": "llm"} if isinstance(v, dict) and "type" not in v else v
                for v in variants
            ]
        if fix_judge:
            data["judge"] = {**judge, "type": "llm"}
        return data

    @model_validator(mode="after")
    def _validate_cross_field_rules(self) -> Self:
        names = [variant.name for variant in self.variants]
        if len(names) != len(set(names)):
            raise ValueError(f"variant names must be unique, got {names}")
        if self.judge is not None and self.judge.mode == "pairwise" and len(self.variants) != 2:
            raise ValueError(
                f"pairwise judging requires exactly 2 variants, got {len(self.variants)}"
            )
        if self.sampling.model is None and any(v.type == "llm" for v in self.variants):
            raise ValueError("sampling.model is required when any variant has type 'llm'")
        if self.hypothesis is not None and self.analysis_plan is None:
            raise ValueError(
                "hypothesis requires an analysis_plan — pre-register the analysis,"
                " not just the claim"
            )
        if self.analysis_plan is not None:
            if self.judge is None:
                raise ValueError(
                    "analysis_plan requires a judge/scorer block; a sample-only run"
                    " produces nothing to analyze"
                )
            declared = set(names)
            seen_pairs: set[frozenset[str]] = set()
            for a, b in [self.analysis_plan.primary, *self.analysis_plan.comparisons]:
                if a == b:
                    raise ValueError(
                        f"analysis_plan pair ({a!r}, {a!r}) must name two distinct variants"
                    )
                unknown = {a, b} - declared
                if unknown:
                    raise ValueError(
                        f"analysis_plan names {sorted(unknown)} are not declared variants"
                    )
                key = frozenset((a, b))
                if key in seen_pairs:
                    raise ValueError(
                        f"analysis_plan lists {a!r} vs {b!r} more than once (pairs are unordered)"
                    )
                seen_pairs.add(key)
        return self


def load_spec(path: str | Path) -> ExperimentSpec:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return ExperimentSpec.model_validate(data)


def _template_placeholders(template: str, context: str) -> set[str]:
    names = set()
    for _, field_name, format_spec, _ in string.Formatter().parse(template):
        if field_name is None:
            continue
        if field_name == "" or field_name.isdigit():
            raise ValueError(f"{context}: positional placeholders are not allowed")
        names.add(field_name)
        if format_spec and "{" in format_spec:
            # Nested replacement fields like {input:>{width}} also consume names.
            names |= _template_placeholders(format_spec, context)
    return names


def _validate_templates(spec: ExperimentSpec, items: list[dict[str, Any]]) -> None:
    # Intersection: a placeholder must exist in EVERY item, or rendering would
    # KeyError at run time on the items that lack it.
    field_sets = [{k for k in item if k != "id"} for item in items]
    item_fields = set.intersection(*field_sets) if field_sets else set()
    all_fields = set().union(*field_sets) if field_sets else set()

    llm_judge = spec.judge if spec.judge is not None and spec.judge.type == "llm" else None
    judge_placeholders: frozenset[str] = frozenset()
    if llm_judge is not None:
        judge_placeholders = _JUDGE_PLACEHOLDERS_BY_MODE[llm_judge.mode]
        reserved = all_fields & judge_placeholders
        if reserved:
            raise ValueError(
                f"dataset item fields {sorted(reserved)} are reserved for judge template"
                f" placeholders in {llm_judge.mode} mode; rename these dataset fields"
            )

    if any(v.type == "command" for v in spec.variants) and "seed" in all_fields:
        raise ValueError(
            "dataset item field 'seed' is reserved for the {seed} placeholder in"
            " command variants; rename this dataset field"
        )

    def check(template: str, context: str, extra_allowed: frozenset[str]) -> None:
        placeholders = _template_placeholders(template, context)
        if "id" in placeholders:
            raise ValueError(f"{context}: the `id` field may not be used as a placeholder")
        unknown = placeholders - item_fields - extra_allowed
        if unknown:
            raise ValueError(
                f"{context}: placeholders {sorted(unknown)} not found in dataset item fields"
            )

    for variant in spec.variants:
        if variant.type == "llm":
            check(variant.user_template, f"variant {variant.name!r} user_template", frozenset())
        elif variant.type == "command":
            for position, element in enumerate(variant.command):
                context = f"variant {variant.name!r} command[{position}]"
                check(element, context, _COMMAND_PLACEHOLDERS)
        # python variants render nothing: the callable receives (item, seed) directly.
    if llm_judge is not None:
        check(
            llm_judge.resolved_prompt_template(),
            "judge prompt_template",
            judge_placeholders,
        )


def load_items(spec: ExperimentSpec, base_dir: str | Path) -> list[dict[str, Any]]:
    """Load and validate dataset items (inline, or JSONL relative to base_dir)."""
    if spec.dataset.items is not None:
        items = spec.dataset.items
    else:
        items = []
        path = Path(base_dir) / spec.dataset.path
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: each JSONL line must be an object")
            for field, value in record.items():
                try:
                    # Same probe as Dataset._items_json_safe: json.loads accepts lone
                    # "\ud800" escapes that canonical_json cannot utf-8 encode.
                    json.dumps(value, ensure_ascii=False).encode("utf-8")
                except (TypeError, ValueError):
                    raise ValueError(
                        f"{path}:{line_number}: field {field!r}: value {value!r} is not"
                        " JSON-serializable UTF-8 (lone surrogates are rejected)"
                    ) from None
            items.append(record)

    if not items:
        raise ValueError("dataset is empty: at least one item is required")

    seen = set()
    for item in items:
        item_id = item.get("id")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError(f"every dataset item needs a non-empty string `id`, got {item!r}")
        if item_id in seen:
            raise ValueError(f"duplicate item id {item_id!r}")
        seen.add(item_id)

    _validate_templates(spec, items)
    return items
