"""Pydantic models for the YAML experiment spec (DESIGN.md §2 — normative as of M2).

Spec defaults (temperature 1.0, max_tokens 1024, seed 0, n_samples 1, limits 4/60)
resolve HERE, before any hashing, so equal effective specs always produce equal
cache keys. The golden keys in tests/test_cache.py pin this indirectly.
"""

import json
import string
from pathlib import Path
from typing import Any, Literal, Self

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


class Variant(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    system: str = ""
    user_template: str


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
                    json.dumps(value)
                except (TypeError, ValueError):
                    raise ValueError(
                        f"dataset item {item.get('id', index)!r} field {field!r}: value"
                        f" {value!r} is not JSON-serializable (quote YAML dates/timestamps"
                        " as strings)"
                    ) from None
        return items

    @model_validator(mode="after")
    def _exactly_one_source(self) -> Self:
        if (self.path is None) == (self.items is None):
            raise ValueError("dataset requires exactly one of `path` or `items`")
        return self


class Sampling(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    temperature: float = Field(default=1.0, allow_inf_nan=False)
    max_tokens: int = Field(default=1024, ge=1)
    seed: int = 0


class Judge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    mode: Literal["pairwise", "rubric"]
    temperature: float = Field(default=0.0, allow_inf_nan=False)
    max_tokens: int = Field(default=512, ge=1)
    prompt_template: str | None = None
    position_swap: bool = True

    def resolved_prompt_template(self) -> str:
        if self.prompt_template is not None:
            return self.prompt_template
        if self.mode == "pairwise":
            return _DEFAULT_PAIRWISE_TEMPLATE
        return _DEFAULT_RUBRIC_TEMPLATE


class Limits(BaseModel):
    model_config = ConfigDict(extra="forbid")

    concurrency: int = Field(default=4, ge=1)
    requests_per_minute: int = Field(default=60, ge=1)


class ExperimentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    variants: list[Variant] = Field(min_length=1)
    dataset: Dataset
    sampling: Sampling
    n_samples: int = Field(default=1, ge=1)
    judge: Judge | None = None
    limits: Limits = Field(default_factory=Limits)

    @model_validator(mode="after")
    def _validate_cross_field_rules(self) -> Self:
        names = [variant.name for variant in self.variants]
        if len(names) != len(set(names)):
            raise ValueError(f"variant names must be unique, got {names}")
        if self.judge is not None and self.judge.mode == "pairwise" and len(self.variants) != 2:
            raise ValueError(
                f"pairwise judging requires exactly 2 variants, got {len(self.variants)}"
            )
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

    judge_placeholders: frozenset[str] = frozenset()
    if spec.judge is not None:
        judge_placeholders = _JUDGE_PLACEHOLDERS_BY_MODE[spec.judge.mode]
        all_fields = set().union(*field_sets) if field_sets else set()
        reserved = all_fields & judge_placeholders
        if reserved:
            raise ValueError(
                f"dataset item fields {sorted(reserved)} are reserved for judge template"
                f" placeholders in {spec.judge.mode} mode; rename these dataset fields"
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
        check(variant.user_template, f"variant {variant.name!r} user_template", frozenset())
    if spec.judge is not None:
        check(
            spec.judge.resolved_prompt_template(),
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
