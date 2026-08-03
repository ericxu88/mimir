"""Tests for mimir.spec — pydantic models for the YAML experiment spec (DESIGN.md §2).

As of M2 the pydantic models are normative; spec defaults (temperature 1.0,
max_tokens 1024, seed 0, n_samples 1, limits 4/60) resolve HERE, before hashing,
so the golden keys in test_cache.py must never move.
"""

import json

import pytest
import yaml
from pydantic import ValidationError

from mimir.spec import ExperimentSpec, load_items, load_spec


def spec_dict(**overrides):
    data = {
        "name": "greeting-tone",
        "variants": [
            {
                "name": "control",
                "system": "You are a helpful assistant.",
                "user_template": "Answer the question: {input}",
            },
            {
                "name": "friendly",
                "system": "You are a warm, encouraging assistant.",
                "user_template": "Answer the question: {input}",
            },
        ],
        "dataset": {
            "items": [
                {"id": "q1", "input": "Why is the sky blue?"},
                {"id": "q2", "input": "Why is grass green?"},
            ]
        },
        "sampling": {"model": "claude-haiku-4-5-20251001"},
    }
    data.update(overrides)
    return data


def test_minimal_spec_resolves_documented_defaults():
    spec = ExperimentSpec.model_validate(spec_dict())
    assert spec.description == ""
    assert spec.sampling.temperature == 1.0
    assert spec.sampling.max_tokens == 1024
    assert spec.sampling.seed == 0
    assert spec.n_samples == 1
    assert spec.judge is None
    assert spec.limits.concurrency == 4
    assert spec.limits.requests_per_minute == 60


def test_variant_system_defaults_to_empty_string():
    data = spec_dict()
    del data["variants"][0]["system"]
    spec = ExperimentSpec.model_validate(data)
    assert spec.variants[0].system == ""


def test_judge_defaults_and_default_templates():
    spec = ExperimentSpec.model_validate(
        spec_dict(judge={"model": "judge-model", "mode": "pairwise"})
    )
    judge = spec.judge
    assert judge.temperature == 0.0
    assert judge.max_tokens == 512
    assert judge.position_swap is True
    assert judge.prompt_template is None
    template = judge.resolved_prompt_template()
    assert "{response_a}" in template
    assert "{response_b}" in template

    rubric = ExperimentSpec.model_validate(
        spec_dict(judge={"model": "judge-model", "mode": "rubric"})
    )
    assert "{response}" in rubric.judge.resolved_prompt_template()


def test_load_spec_from_yaml_file(tmp_path):
    path = tmp_path / "spec.yaml"
    path.write_text(yaml.safe_dump(spec_dict()), encoding="utf-8")
    spec = load_spec(path)
    assert spec.name == "greeting-tone"
    assert [v.name for v in spec.variants] == ["control", "friendly"]


def test_duplicate_variant_names_rejected():
    data = spec_dict()
    data["variants"][1]["name"] = "control"
    with pytest.raises(ValidationError, match="variant"):
        ExperimentSpec.model_validate(data)


def test_at_least_one_variant_required():
    with pytest.raises(ValidationError):
        ExperimentSpec.model_validate(spec_dict(variants=[]))


def test_dataset_requires_exactly_one_source(tmp_path):
    both = spec_dict()
    both["dataset"] = {"path": "data.jsonl", "items": [{"id": "q1", "input": "x"}]}
    with pytest.raises(ValidationError, match="exactly one"):
        ExperimentSpec.model_validate(both)
    neither = spec_dict()
    neither["dataset"] = {}
    with pytest.raises(ValidationError, match="exactly one"):
        ExperimentSpec.model_validate(neither)


def test_pairwise_judge_requires_exactly_two_variants():
    three = spec_dict(judge={"model": "j", "mode": "pairwise"})
    three["variants"].append(
        {"name": "third", "system": "", "user_template": "Answer the question: {input}"}
    )
    with pytest.raises(ValidationError, match="exactly 2"):
        ExperimentSpec.model_validate(three)
    one = spec_dict(judge={"model": "j", "mode": "pairwise"})
    one["variants"] = one["variants"][:1]
    with pytest.raises(ValidationError, match="exactly 2"):
        ExperimentSpec.model_validate(one)


def test_rubric_judge_allows_other_variant_counts():
    three = spec_dict(judge={"model": "j", "mode": "rubric"})
    three["variants"].append(
        {"name": "third", "system": "", "user_template": "Answer the question: {input}"}
    )
    spec = ExperimentSpec.model_validate(three)
    assert len(spec.variants) == 3


def test_unknown_fields_rejected():
    with pytest.raises(ValidationError):
        ExperimentSpec.model_validate(spec_dict(bogus=1))
    typo = spec_dict()
    typo["sampling"] = {"model": "m", "temperture": 0.5}
    with pytest.raises(ValidationError):
        ExperimentSpec.model_validate(typo)


def test_counts_must_be_positive():
    with pytest.raises(ValidationError):
        ExperimentSpec.model_validate(spec_dict(n_samples=0))
    with pytest.raises(ValidationError):
        ExperimentSpec.model_validate(spec_dict(limits={"concurrency": 0}))
    with pytest.raises(ValidationError):
        ExperimentSpec.model_validate(spec_dict(limits={"requests_per_minute": 0}))
    zero_tokens = spec_dict()
    zero_tokens["sampling"] = {"model": "m", "max_tokens": 0}
    with pytest.raises(ValidationError):
        ExperimentSpec.model_validate(zero_tokens)


def test_load_items_inline_returns_items(tmp_path):
    spec = ExperimentSpec.model_validate(spec_dict())
    items = load_items(spec, tmp_path)
    assert [item["id"] for item in items] == ["q1", "q2"]
    assert items[0]["input"] == "Why is the sky blue?"


def test_load_items_duplicate_ids_rejected(tmp_path):
    data = spec_dict()
    data["dataset"]["items"][1]["id"] = "q1"
    spec = ExperimentSpec.model_validate(data)
    with pytest.raises(ValueError, match="duplicate item id"):
        load_items(spec, tmp_path)


def test_load_items_requires_string_id(tmp_path):
    data = spec_dict()
    data["dataset"]["items"][0]["id"] = 1
    spec = ExperimentSpec.model_validate(data)
    with pytest.raises(ValueError, match="id"):
        load_items(spec, tmp_path)
    missing = spec_dict()
    del missing["dataset"]["items"][0]["id"]
    spec = ExperimentSpec.model_validate(missing)
    with pytest.raises(ValueError, match="id"):
        load_items(spec, tmp_path)


def test_load_items_from_jsonl_relative_to_base_dir(tmp_path):
    lines = [
        json.dumps({"id": "q1", "input": "Why is the sky blue?"}),
        "",
        json.dumps({"id": "q2", "input": "Why is grass green?"}),
        "   ",
    ]
    (tmp_path / "data.jsonl").write_text("\n".join(lines), encoding="utf-8")
    data = spec_dict()
    data["dataset"] = {"path": "data.jsonl"}
    spec = ExperimentSpec.model_validate(data)
    items = load_items(spec, tmp_path)
    assert [item["id"] for item in items] == ["q1", "q2"]


def test_user_template_placeholder_must_exist_in_item_fields(tmp_path):
    data = spec_dict()
    data["variants"][0]["user_template"] = "Answer: {question}"
    spec = ExperimentSpec.model_validate(data)
    with pytest.raises(ValueError, match="question"):
        load_items(spec, tmp_path)


def test_user_template_id_placeholder_rejected(tmp_path):
    data = spec_dict()
    data["variants"][0]["user_template"] = "Item {id}: {input}"
    spec = ExperimentSpec.model_validate(data)
    with pytest.raises(ValueError, match="id"):
        load_items(spec, tmp_path)


def test_user_template_positional_placeholder_rejected(tmp_path):
    data = spec_dict()
    data["variants"][0]["user_template"] = "Answer: {}"
    spec = ExperimentSpec.model_validate(data)
    with pytest.raises(ValueError, match="positional"):
        load_items(spec, tmp_path)


def test_default_judge_template_placeholders_validated_against_items(tmp_path):
    # The default pairwise template references {input}; items without that field
    # must fail at load so the user knows to supply a custom prompt_template.
    data = spec_dict(judge={"model": "j", "mode": "pairwise"})
    data["dataset"] = {"items": [{"id": "q1", "question": "Why?"}]}
    for variant in data["variants"]:
        variant["user_template"] = "Answer: {question}"
    spec = ExperimentSpec.model_validate(data)
    with pytest.raises(ValueError, match="input"):
        load_items(spec, tmp_path)


def test_custom_judge_template_may_use_response_placeholders(tmp_path):
    data = spec_dict(
        judge={
            "model": "j",
            "mode": "pairwise",
            "prompt_template": "Q: {question}\nA: {response_a}\nB: {response_b}\nReply A or B.",
        }
    )
    data["dataset"] = {"items": [{"id": "q1", "question": "Why?"}]}
    for variant in data["variants"]:
        variant["user_template"] = "Answer: {question}"
    spec = ExperimentSpec.model_validate(data)
    items = load_items(spec, tmp_path)
    assert len(items) == 1


def test_spec_model_dump_is_json_safe():
    spec = ExperimentSpec.model_validate(spec_dict(judge={"model": "j", "mode": "pairwise"}))
    dumped = spec.model_dump()
    text = json.dumps(dumped)
    assert '"temperature": 1.0' in text
    assert dumped["limits"] == {"concurrency": 4, "requests_per_minute": 60}


def test_placeholder_missing_from_some_items_rejected(tmp_path):
    # Placeholders must exist in EVERY item (intersection), or rendering would
    # KeyError mid-run on the items that lack the field.
    data = spec_dict()
    data["dataset"]["items"] = [
        {"id": "q1", "input": "a", "hint": "b"},
        {"id": "q2", "input": "c"},
    ]
    data["variants"][0]["user_template"] = "Q: {input} Hint: {hint}"
    spec = ExperimentSpec.model_validate(data)
    with pytest.raises(ValueError, match="hint"):
        load_items(spec, tmp_path)


@pytest.mark.parametrize(
    ("mode", "field"),
    [("pairwise", "response_a"), ("pairwise", "response_b"), ("rubric", "response")],
)
def test_judge_reserved_item_fields_rejected(tmp_path, mode, field):
    data = spec_dict(judge={"model": "j", "mode": mode})
    data["dataset"]["items"][0][field] = "preexisting"
    spec = ExperimentSpec.model_validate(data)
    with pytest.raises(ValueError, match="reserved"):
        load_items(spec, tmp_path)


def test_reserved_names_are_mode_specific_and_judge_only(tmp_path):
    # "response" is only reserved in rubric mode; without a judge nothing is reserved.
    no_judge = spec_dict()
    no_judge["dataset"]["items"][0]["response"] = "fine"
    assert load_items(ExperimentSpec.model_validate(no_judge), tmp_path)

    pairwise = spec_dict(judge={"model": "j", "mode": "pairwise"})
    pairwise["dataset"]["items"][0]["response"] = "fine"
    assert load_items(ExperimentSpec.model_validate(pairwise), tmp_path)


def test_judge_template_placeholders_are_mode_specific(tmp_path):
    rubric = spec_dict(
        judge={
            "model": "j",
            "mode": "rubric",
            "prompt_template": "Q: {input}\nA: {response_a}\nScore 1-10.",
        }
    )
    with pytest.raises(ValueError, match="response_a"):
        load_items(ExperimentSpec.model_validate(rubric), tmp_path)

    pairwise = spec_dict(
        judge={
            "model": "j",
            "mode": "pairwise",
            "prompt_template": "Q: {input}\nR: {response}\nReply A or B.",
        }
    )
    with pytest.raises(ValueError, match="response"):
        load_items(ExperimentSpec.model_validate(pairwise), tmp_path)


def test_non_json_safe_item_values_rejected():
    # yaml.safe_load parses unquoted dates into datetime.date, which would break
    # canonical JSON at create_run time; reject at spec validation instead.
    data = spec_dict()
    data["dataset"]["items"][0]["published"] = yaml.safe_load("2024-01-01")
    with pytest.raises(ValidationError, match="JSON-serializable"):
        ExperimentSpec.model_validate(data)


def test_inline_item_with_lone_surrogate_rejected():
    # json.dumps with the default ensure_ascii=True accepts lone surrogates, but
    # canonical_json (ensure_ascii=False + utf-8 encode) raises on them at
    # create_run time; the validator's probe must match canonical_json semantics.
    data = spec_dict()
    data["dataset"]["items"][0]["input"] = "bad \ud800 value"
    with pytest.raises(ValidationError, match="JSON-serializable"):
        ExperimentSpec.model_validate(data)


def test_jsonl_item_with_lone_surrogate_rejected_with_location(tmp_path):
    # JSONL lines come from json.loads, which accepts "\ud800" escapes — the same
    # probe must run on the path dataset, with the file:line error convention.
    path = tmp_path / "items.jsonl"
    path.write_text('{"id": "q1", "input": "bad \\ud800 value"}\n', encoding="utf-8")
    spec = ExperimentSpec.model_validate(spec_dict(dataset={"path": "items.jsonl"}))
    with pytest.raises(ValueError, match=r"items\.jsonl:1"):
        load_items(spec, tmp_path)


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
def test_non_finite_temperatures_rejected(bad):
    data = spec_dict()
    data["sampling"]["temperature"] = bad
    with pytest.raises(ValidationError):
        ExperimentSpec.model_validate(data)
    judged = spec_dict(judge={"model": "j", "mode": "pairwise", "temperature": bad})
    with pytest.raises(ValidationError):
        ExperimentSpec.model_validate(judged)


def test_nested_format_spec_placeholder_validated(tmp_path):
    data = spec_dict()
    data["variants"][0]["user_template"] = "Q: {input:>{width}}"
    spec = ExperimentSpec.model_validate(data)
    with pytest.raises(ValueError, match="width"):
        load_items(spec, tmp_path)


def test_jsonl_invalid_json_line_reports_location(tmp_path):
    lines = [json.dumps({"id": "q1", "input": "x"}), "{not json"]
    (tmp_path / "data.jsonl").write_text("\n".join(lines), encoding="utf-8")
    data = spec_dict()
    data["dataset"] = {"path": "data.jsonl"}
    spec = ExperimentSpec.model_validate(data)
    with pytest.raises(ValueError, match=r"data\.jsonl:2: invalid JSON"):
        load_items(spec, tmp_path)


def test_jsonl_non_object_line_rejected(tmp_path):
    (tmp_path / "data.jsonl").write_text("[1, 2]\n", encoding="utf-8")
    data = spec_dict()
    data["dataset"] = {"path": "data.jsonl"}
    spec = ExperimentSpec.model_validate(data)
    with pytest.raises(ValueError, match="object"):
        load_items(spec, tmp_path)


def test_empty_dataset_rejected(tmp_path):
    inline = spec_dict()
    inline["dataset"] = {"items": []}
    with pytest.raises(ValueError, match="empty"):
        load_items(ExperimentSpec.model_validate(inline), tmp_path)

    (tmp_path / "data.jsonl").write_text("\n", encoding="utf-8")
    from_file = spec_dict()
    from_file["dataset"] = {"path": "data.jsonl"}
    with pytest.raises(ValueError, match="empty"):
        load_items(ExperimentSpec.model_validate(from_file), tmp_path)


# --- M9: templates must be UTF-8 (lone surrogates poison cache keys) ---------------


def test_variant_system_rejects_lone_surrogate():
    data = spec_dict()
    data["variants"][0]["system"] = "You are helpful.\ud800"
    with pytest.raises(ValidationError, match="lone surrogate"):
        ExperimentSpec.model_validate(data)


def test_variant_user_template_rejects_lone_surrogate():
    data = spec_dict()
    data["variants"][1]["user_template"] = "Answer the question: {input}\udfff"
    with pytest.raises(ValidationError, match="lone surrogate"):
        ExperimentSpec.model_validate(data)


def test_judge_prompt_template_rejects_lone_surrogate():
    data = spec_dict(
        judge={
            "model": "judge-model",
            "mode": "rubric",
            "prompt_template": "Q: {input}\nA: {response}\nRate 1-10.\ud800",
        }
    )
    with pytest.raises(ValidationError, match="lone surrogate"):
        ExperimentSpec.model_validate(data)


# --- M10: condition/scorer type discriminators (llm | command | python) ------------


def command_spec_dict(**overrides):
    data = {
        "name": "bench",
        "variants": [
            {
                "type": "command",
                "name": "fast",
                "command": [
                    "python3",
                    "bench.py",
                    "--algo",
                    "fast",
                    "--seed",
                    "{seed}",
                    "--case",
                    "{input}",
                ],
            },
            {
                "type": "command",
                "name": "slow",
                "command": [
                    "python3",
                    "bench.py",
                    "--algo",
                    "slow",
                    "--seed",
                    "{seed}",
                    "--case",
                    "{input}",
                ],
            },
        ],
        "dataset": {
            "items": [
                {"id": "q1", "input": "alpha"},
                {"id": "q2", "input": "beta"},
            ]
        },
        "judge": {"type": "parse_float"},
    }
    data.update(overrides)
    return data


def test_untyped_variant_defaults_to_llm_type():
    spec = ExperimentSpec.model_validate(spec_dict())
    assert spec.variants[0].type == "llm"
    assert spec.model_dump()["variants"][0]["type"] == "llm"


def test_untyped_judge_defaults_to_llm_type():
    spec = ExperimentSpec.model_validate(spec_dict(judge={"model": "j", "mode": "rubric"}))
    assert spec.judge.type == "llm"
    assert spec.model_dump()["judge"]["type"] == "llm"


def test_type_default_injection_does_not_mutate_input():
    # The before-validator that injects type: "llm" must copy, never mutate:
    # tests (and callers) reuse spec dicts across validations.
    data = spec_dict(judge={"model": "j", "mode": "rubric"})
    ExperimentSpec.model_validate(data)
    assert "type" not in data["variants"][0]
    assert "type" not in data["judge"]
    spec = ExperimentSpec.model_validate(data)
    assert spec.variants[0].type == "llm"
    assert spec.judge.type == "llm"


def test_unknown_variant_type_rejected():
    data = spec_dict()
    data["variants"][0]["type"] = "carrier-pigeon"
    with pytest.raises(ValidationError):
        ExperimentSpec.model_validate(data)


def test_command_variant_validates_with_defaults():
    spec = ExperimentSpec.model_validate(command_spec_dict())
    assert spec.variants[0].type == "command"
    assert spec.variants[0].command[0] == "python3"
    assert spec.variants[0].timeout_s == 60.0


def test_command_variant_requires_nonempty_argv():
    data = command_spec_dict()
    data["variants"][0]["command"] = []
    with pytest.raises(ValidationError):
        ExperimentSpec.model_validate(data)


@pytest.mark.parametrize("bad", [0, -1, float("inf"), float("nan")])
def test_command_timeout_must_be_positive_and_finite(bad):
    data = command_spec_dict()
    data["variants"][0]["timeout_s"] = bad
    with pytest.raises(ValidationError):
        ExperimentSpec.model_validate(data)


def test_command_argv_rejects_lone_surrogate():
    # Rendered argv enters cache keys via canonical_json, same as templates.
    data = command_spec_dict()
    data["variants"][0]["command"][1] = "bench\ud800.py"
    with pytest.raises(ValidationError, match="lone surrogate"):
        ExperimentSpec.model_validate(data)


def test_python_variant_accepts_import_path():
    data = command_spec_dict()
    data["variants"][0] = {"type": "python", "name": "fast", "callable": "examples.bench:run"}
    spec = ExperimentSpec.model_validate(data)
    assert spec.variants[0].type == "python"
    assert spec.variants[0].callable == "examples.bench:run"


@pytest.mark.parametrize("bad", ["no_colon", ":fn", "mod:", "mod:fn:extra", "mod path:fn"])
def test_python_variant_rejects_malformed_import_path(bad):
    data = command_spec_dict()
    data["variants"][0] = {"type": "python", "name": "fast", "callable": bad}
    with pytest.raises(ValidationError, match="callable"):
        ExperimentSpec.model_validate(data)


def test_parse_float_judge_dumps_rubric_mode_without_model():
    # stats.py reads spec_json["judge"]["mode"]; per-sample scalar scores are
    # rubric-shaped. judge_audit additionally needs "model", which parse_float
    # deliberately lacks — audit-judge on such a run errors cleanly.
    spec = ExperimentSpec.model_validate(command_spec_dict())
    assert spec.judge.type == "parse_float"
    assert spec.judge.mode == "rubric"
    dumped = spec.model_dump()["judge"]
    assert dumped["mode"] == "rubric"
    assert "model" not in dumped


def test_parse_float_judge_rejects_llm_judge_fields():
    data = command_spec_dict(judge={"type": "parse_float", "model": "m"})
    with pytest.raises(ValidationError):
        ExperimentSpec.model_validate(data)


def test_parse_float_judge_allows_more_than_two_variants():
    data = command_spec_dict()
    data["variants"].append({"type": "command", "name": "third", "command": ["x", "{seed}"]})
    assert len(ExperimentSpec.model_validate(data).variants) == 3


def test_sampling_optional_for_non_llm_specs():
    spec = ExperimentSpec.model_validate(command_spec_dict())  # no sampling block at all
    assert spec.sampling.model is None
    assert spec.sampling.seed == 0
    seeded = command_spec_dict(sampling={"seed": 7})
    assert ExperimentSpec.model_validate(seeded).sampling.seed == 7


def test_llm_variant_requires_sampling_model():
    data = spec_dict(sampling={})
    with pytest.raises(ValidationError, match=r"sampling\.model"):
        ExperimentSpec.model_validate(data)
    missing = spec_dict()
    del missing["sampling"]
    with pytest.raises(ValidationError, match=r"sampling\.model"):
        ExperimentSpec.model_validate(missing)


def test_mixed_llm_and_command_variants_allowed():
    data = command_spec_dict(sampling={"model": "m"})
    data["variants"][1] = {"name": "llm-arm", "user_template": "Do: {input}"}
    spec = ExperimentSpec.model_validate(data)
    assert [v.type for v in spec.variants] == ["command", "llm"]


def test_command_argv_placeholders_validated_against_items(tmp_path):
    data = command_spec_dict()
    data["variants"][0]["command"] = ["python3", "bench.py", "--case", "{qestion}"]
    spec = ExperimentSpec.model_validate(data)
    with pytest.raises(ValueError, match="qestion"):
        load_items(spec, tmp_path)


def test_command_argv_seed_placeholder_is_always_available(tmp_path):
    # command_spec_dict argv uses {seed}; items deliberately lack a seed field.
    spec = ExperimentSpec.model_validate(command_spec_dict())
    assert load_items(spec, tmp_path)


def test_command_argv_id_placeholder_rejected(tmp_path):
    data = command_spec_dict()
    data["variants"][0]["command"] = ["python3", "bench.py", "--case", "{id}"]
    spec = ExperimentSpec.model_validate(data)
    with pytest.raises(ValueError, match="id"):
        load_items(spec, tmp_path)


def test_item_field_named_seed_reserved_for_command_variants(tmp_path):
    data = command_spec_dict()
    data["dataset"]["items"][0]["seed"] = 3
    spec = ExperimentSpec.model_validate(data)
    with pytest.raises(ValueError, match="reserved"):
        load_items(spec, tmp_path)


def test_item_field_named_seed_fine_without_command_variants(tmp_path):
    data = spec_dict()
    data["dataset"]["items"] = [
        {"id": "q1", "input": "a", "seed": 1},
        {"id": "q2", "input": "b", "seed": 2},
    ]
    assert load_items(ExperimentSpec.model_validate(data), tmp_path)


def test_llm_judge_over_command_variants_validates(tmp_path):
    # Cross-type composition: an LLM judge scoring subprocess outputs is legitimate.
    data = command_spec_dict(judge={"model": "j", "mode": "pairwise"})
    spec = ExperimentSpec.model_validate(data)
    assert spec.judge.type == "llm"
    assert load_items(spec, tmp_path)
