"""Tests for mimir.cache — the LOCKED cache-key definition (docs/DESIGN.md §3).

Not testable in M1 (arrives with M2's spec→payload rendering, per docs/PROGRESS.md):
spec-default resolution before hashing, label/limits exclusion from the key, and
rendered-text-not-templates.

All non-ASCII characters inside hashed string literals are written as escape sequences
so those literals stay byte-exact (immune to NFC/NFD editor normalization, which would
silently corrupt the expected hashes).
"""

import re

import pytest

from mimir.cache import (
    build_command_payload,
    build_payload,
    build_python_payload,
    cache_key,
    canonical_json,
)


def base_payload():
    # The docs/DESIGN.md §3 example payload, rebuilt fresh on every call so callers
    # may mutate the result without aliasing.
    return {
        "model": "claude-haiku-4-5-20251001",
        "system": "You are a helpful assistant.",
        "messages": [{"role": "user", "content": "Answer the question: Why is the sky blue?"}],
        "params": {"temperature": 1.0, "max_tokens": 1024},
        "seed": 42,
        "sample_index": 0,
    }


def test_identical_payload_gives_identical_key():
    assert cache_key(base_payload()) == cache_key(base_payload())
    payload = base_payload()
    assert cache_key(payload) == cache_key(payload)


def test_key_is_64_char_lowercase_hex():
    assert re.fullmatch(r"[0-9a-f]{64}", cache_key(base_payload()))


@pytest.mark.parametrize(
    "override",
    [
        pytest.param({"model": "claude-sonnet-4-5"}, id="model"),
        pytest.param({"system": "You are a helpful assistant. "}, id="system"),
        pytest.param(
            {"messages": [{"role": "user", "content": "Why is the sea blue?"}]},
            id="messages",
        ),
        pytest.param({"params": {"temperature": 0.7, "max_tokens": 1024}}, id="params.temperature"),
        pytest.param({"params": {"temperature": 1.0, "max_tokens": 2048}}, id="params.max_tokens"),
        pytest.param({"seed": 43}, id="seed"),
        pytest.param({"sample_index": 1}, id="sample_index"),
    ],
)
def test_each_field_perturbation_changes_key(override):
    perturbed = base_payload()
    perturbed.update(override)
    assert cache_key(perturbed) != cache_key(base_payload())


def test_top_level_dict_insertion_order_irrelevant():
    reversed_order = {
        "sample_index": 0,
        "seed": 42,
        "params": {"temperature": 1.0, "max_tokens": 1024},
        "messages": [{"role": "user", "content": "Answer the question: Why is the sky blue?"}],
        "system": "You are a helpful assistant.",
        "model": "claude-haiku-4-5-20251001",
    }
    assert cache_key(reversed_order) == cache_key(base_payload())


def test_nested_params_insertion_order_irrelevant():
    payload = base_payload()
    payload["params"] = {"max_tokens": 1024, "temperature": 1.0}
    assert cache_key(payload) == cache_key(base_payload())


def test_message_list_order_matters():
    m1 = {"role": "user", "content": "first"}
    m2 = {"role": "user", "content": "second"}
    a = base_payload()
    a["messages"] = [m1, m2]
    b = base_payload()
    b["messages"] = [m2, m1]
    assert cache_key(a) != cache_key(b)


def test_int_vs_float_temperature_same_key():
    kwargs = {
        "model": "claude-haiku-4-5-20251001",
        "user": "hi",
        "max_tokens": 1024,
        "seed": 42,
        "sample_index": 0,
    }
    from_int = build_payload(temperature=1, **kwargs)
    from_float = build_payload(temperature=1.0, **kwargs)
    assert cache_key(from_int) == cache_key(from_float)
    assert b'"temperature":1.0' in canonical_json(from_int)


def test_negative_zero_temperature_same_key_as_zero():
    kwargs = {"model": "m", "user": "u", "max_tokens": 16, "seed": 0, "sample_index": 0}
    from_neg = build_payload(temperature=-0.0, **kwargs)
    from_pos = build_payload(temperature=0.0, **kwargs)
    assert cache_key(from_neg) == cache_key(from_pos)
    assert b'"temperature":0.0' in canonical_json(from_neg)


def test_build_payload_none_or_omitted_system_becomes_empty_string():
    kwargs = {
        "model": "m",
        "user": "hi",
        "temperature": 1.0,
        "max_tokens": 16,
        "seed": 0,
        "sample_index": 0,
    }
    from_none = build_payload(system=None, **kwargs)
    from_omitted = build_payload(**kwargs)
    from_empty = build_payload(system="", **kwargs)
    assert from_none["system"] == ""
    assert cache_key(from_none) == cache_key(from_omitted) == cache_key(from_empty)


def test_build_payload_has_exactly_six_keys():
    payload = build_payload(
        model="m", user="hi", temperature=1.0, max_tokens=16, seed=0, sample_index=0
    )
    assert set(payload) == {"model", "system", "messages", "params", "seed", "sample_index"}
    assert set(payload["params"]) == {"temperature", "max_tokens"}
    assert payload["messages"] == [{"role": "user", "content": "hi"}]


def test_build_payload_returns_fresh_dict():
    kwargs = {
        "model": "m",
        "user": "hi",
        "temperature": 1.0,
        "max_tokens": 16,
        "seed": 0,
        "sample_index": 0,
    }
    first = build_payload(**kwargs)
    first["params"]["temperature"] = 9.9
    second = build_payload(**kwargs)
    assert second["params"]["temperature"] == 1.0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_build_payload_rejects_non_finite_temperature(bad):
    with pytest.raises(ValueError, match="temperature"):
        build_payload(model="m", user="u", temperature=bad, max_tokens=16, seed=0, sample_index=0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_tokens", 1024.0),
        ("max_tokens", True),
        ("seed", 42.0),
        ("seed", False),
        ("sample_index", 0.0),
        ("sample_index", True),
        ("temperature", True),
        ("temperature", "1.0"),
    ],
)
def test_build_payload_rejects_wrongly_typed_numeric_fields(field, value):
    kwargs = {
        "model": "m",
        "user": "u",
        "temperature": 1.0,
        "max_tokens": 16,
        "seed": 0,
        "sample_index": 0,
        field: value,
    }
    with pytest.raises(TypeError):
        build_payload(**kwargs)


def test_canonical_json_exact_bytes():
    # One assertion pins all four LOCKED knobs: separators (no spaces), sort_keys
    # recursing into nested dicts (a<b, x<y), ensure_ascii=False (raw UTF-8), and
    # float serialization (1.0 stays 1.0).
    result = canonical_json({"b": [1, 2], "a": {"y": 1.0, "x": "h\u00e9llo \u4e16\u754c"}})
    assert isinstance(result, bytes)
    assert result == b'{"a":{"x":"h\xc3\xa9llo \xe4\xb8\x96\xe7\x95\x8c","y":1.0},"b":[1,2]}'


def test_golden_key_matches_locked_spec():
    # Cross-session drift tripwire: every other test checks only relative properties,
    # so a "harmless" canonicalization change would pass them all while silently
    # invalidating every on-disk cache. Payload built inline (not via base_payload)
    # so a helper edit cannot shift this test. Expected value re-derived from the
    # DESIGN.md §3 one-liner on CPython 3.13 before being hardcoded here.
    payload = {
        "model": "claude-haiku-4-5-20251001",
        "system": "You are a helpful assistant.",
        "messages": [{"role": "user", "content": "Answer the question: Why is the sky blue?"}],
        "params": {"temperature": 1.0, "max_tokens": 1024},
        "seed": 42,
        "sample_index": 0,
    }
    assert cache_key(payload) == "06464804f4ac20a8508e62e2df909e64000a90265670b0505015866cb1aef92b"


def test_unicode_payload_raw_utf8_and_golden_key():
    def build():
        return build_payload(
            model="claude-haiku-4-5-20251001",
            system="h\u00e9llo \u4e16\u754c",
            user="\U0001f989 say hi",
            temperature=0.0,
            max_tokens=16,
            seed=0,
            sample_index=0,
        )

    blob = canonical_json(build())
    assert "h\u00e9llo \u4e16\u754c".encode() in blob
    assert b"\\u" not in blob
    assert cache_key(build()) == cache_key(build())
    expected = "cbc9c58be3eac9bc46083cda70c0d65ec8e3b75ceb945b26c5f503f9d0180942"
    assert cache_key(build()) == expected


# --- M10: command/python payload shapes (additive; the six-key LLM shape is LOCKED) ---


def test_command_payload_shape_and_values():
    # Exact key set doubles as the exclusion pin: timeout_s and base_dir are
    # execution limits and must never enter the cache key (DESIGN \u00a73).
    payload = build_command_payload(
        argv=["python3", "bench.py", "--seed", "7"], seed=7, sample_index=0
    )
    assert set(payload) == {"type", "argv", "seed", "sample_index"}
    assert payload["type"] == "command"
    assert payload["argv"] == ["python3", "bench.py", "--seed", "7"]
    assert payload["seed"] == 7
    assert payload["sample_index"] == 0


def test_python_payload_shape_and_values():
    item = {"id": "q1", "input": "x"}
    payload = build_python_payload(
        callable_path="examples.bench:run", item=item, seed=3, sample_index=1
    )
    assert set(payload) == {"type", "callable", "item", "seed", "sample_index"}
    assert payload["type"] == "python"
    assert payload["callable"] == "examples.bench:run"
    assert payload["item"] == item


@pytest.mark.parametrize(
    "override",
    [
        pytest.param({"argv": ["python3", "bench.py", "--seed", "8"]}, id="argv"),
        pytest.param({"seed": 8}, id="seed"),
        pytest.param({"sample_index": 1}, id="sample_index"),
    ],
)
def test_command_payload_perturbation_changes_key(override):
    kwargs = {"argv": ["python3", "bench.py", "--seed", "7"], "seed": 7, "sample_index": 0}
    base = cache_key(build_command_payload(**kwargs))
    kwargs.update(override)
    assert cache_key(build_command_payload(**kwargs)) != base


@pytest.mark.parametrize(
    "override",
    [
        pytest.param({"callable_path": "examples.bench:walk"}, id="callable"),
        pytest.param({"item": {"id": "q2"}}, id="item"),
        pytest.param({"seed": 8}, id="seed"),
        pytest.param({"sample_index": 1}, id="sample_index"),
    ],
)
def test_python_payload_perturbation_changes_key(override):
    kwargs = {"callable_path": "examples.bench:run", "item": {"id": "q1"}, "seed": 7}
    base = cache_key(build_python_payload(sample_index=0, **kwargs))
    kwargs.update(override)
    sample_index = kwargs.pop("sample_index", 0)
    assert cache_key(build_python_payload(sample_index=sample_index, **kwargs)) != base


def test_new_payload_shapes_cannot_collide_with_the_llm_shape():
    # The LOCKED six-key LLM payload has no "type" key; both new shapes carry one.
    # canonical_json of different key sets can never be byte-equal, so no command
    # or python payload can hash onto an existing LLM cache entry.
    llm = base_payload()
    assert "type" not in llm
    command = build_command_payload(argv=["x"], seed=0, sample_index=0)
    python = build_python_payload(callable_path="m:f", item={"id": "i"}, seed=0, sample_index=0)
    assert command["type"] == "command"
    assert python["type"] == "python"
    assert cache_key(command) != cache_key(python)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seed", 7.0),
        ("seed", True),
        ("sample_index", 0.0),
        ("sample_index", False),
        ("argv", "not-a-list"),
        ("argv", ["ok", 1]),
    ],
)
def test_command_payload_rejects_wrongly_typed_fields(field, value):
    kwargs = {"argv": ["python3"], "seed": 0, "sample_index": 0, field: value}
    with pytest.raises(TypeError):
        build_command_payload(**kwargs)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seed", 7.0),
        ("seed", True),
        ("sample_index", 0.0),
        ("callable_path", 3),
        ("item", ["not", "a", "dict"]),
    ],
)
def test_python_payload_rejects_wrongly_typed_fields(field, value):
    kwargs = {"callable_path": "m:f", "item": {"id": "i"}, "seed": 0, "sample_index": 0}
    kwargs[field] = value
    with pytest.raises(TypeError):
        build_python_payload(**kwargs)


def test_command_payload_returns_fresh_structures():
    argv = ["python3", "bench.py"]
    payload = build_command_payload(argv=argv, seed=0, sample_index=0)
    payload["argv"].append("--extra")
    assert argv == ["python3", "bench.py"]


def test_python_payload_copies_item():
    item = {"id": "q1"}
    payload = build_python_payload(callable_path="m:f", item=item, seed=0, sample_index=0)
    payload["item"]["id"] = "changed"
    assert item["id"] == "q1"
