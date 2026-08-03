"""Tests for mimir.conditions — Condition/Scorer protocols and adapters (M10).

Subprocess tests run real local child processes via sys.executable (stdlib only,
deterministic, no network) — the "no real API calls" rule concerns providers, not
subprocesses. Python-callable tests import functions from THIS module: pytest's
rootdir-style collection puts tests/ on sys.path, so "test_conditions:<fn>" is a
valid import path inside the test process.
"""

import sys

import pytest

from mimir.cache import build_payload
from mimir.clients.base import ClientError, CompletionResponse
from mimir.clients.mock import MockClient
from mimir.conditions import (
    CommandCondition,
    ConditionError,
    LlmCondition,
    ParseFloatScorer,
    build_conditions,
    needs_client,
    render_argv,
)
from mimir.spec import ExperimentSpec


def llm_spec(**overrides):
    data = {
        "name": "llm-exp",
        "variants": [
            {"name": "control", "system": "Be brief.", "user_template": "Q: {input}"},
            {"name": "friendly", "system": "Be warm.", "user_template": "Q: {input}"},
        ],
        "dataset": {"items": [{"id": "q1", "input": "Why?"}]},
        "sampling": {"model": "test-model", "seed": 7},
    }
    data.update(overrides)
    return ExperimentSpec.model_validate(data)


def command_spec(command=None, **overrides):
    data = {
        "name": "bench-exp",
        "variants": [
            {
                "type": "command",
                "name": "fast",
                "command": command or [sys.executable, "-c", "print(1.5)"],
            }
        ],
        "dataset": {"items": [{"id": "q1", "input": "alpha"}]},
    }
    data.update(overrides)
    return ExperimentSpec.model_validate(data)


def output_with_text(text: str) -> CompletionResponse:
    return CompletionResponse(
        text=text, raw={}, input_tokens=0, output_tokens=0, latency_ms=0.0, model="command:x"
    )


# --- render_argv (pure) ------------------------------------------------------------


def test_render_argv_substitutes_fields_and_seed():
    argv = render_argv(
        ["bench.py", "--case", "{input}", "--seed", "{seed}"],
        {"id": "q1", "input": "alpha"},
        7,
    )
    assert argv == ["bench.py", "--case", "alpha", "--seed", "7"]


def test_render_argv_literal_braces_render_literally():
    assert render_argv(["{{x}}"], {"id": "q1"}, 0) == ["{x}"]


def test_render_argv_formats_non_string_values():
    assert render_argv(["--n", "{count}"], {"id": "q1", "count": 3}, 0) == ["--n", "3"]


def test_render_argv_missing_field_raises():
    with pytest.raises(KeyError):
        render_argv(["{missing}"], {"id": "q1"}, 0)


def test_render_argv_id_is_not_a_field():
    # Consistent with prompt rendering: `id` is identity, never template input.
    with pytest.raises(KeyError):
        render_argv(["{id}"], {"id": "q1"}, 0)


# --- ParseFloatScorer (pure) -------------------------------------------------------


@pytest.mark.anyio
async def test_parse_float_scorer_reads_last_nonempty_line():
    scorer = ParseFloatScorer()
    assert await scorer(output_with_text("log line\n3.25\n\n"), {"id": "q1"}) == 3.25
    assert await scorer(output_with_text("4"), {"id": "q1"}) == 4.0
    assert await scorer(output_with_text("-2.5"), {"id": "q1"}) == -2.5
    assert await scorer(output_with_text("progress 10%\n1e-3"), {"id": "q1"}) == 0.001


@pytest.mark.anyio
async def test_parse_float_scorer_rejects_empty_output():
    with pytest.raises(ValueError, match="score"):
        await ParseFloatScorer()(output_with_text(""), {"id": "q1"})
    with pytest.raises(ValueError, match="score"):
        await ParseFloatScorer()(output_with_text("\n  \n"), {"id": "q1"})


@pytest.mark.anyio
async def test_parse_float_scorer_rejects_non_numeric_final_line():
    with pytest.raises(ValueError, match="score"):
        await ParseFloatScorer()(output_with_text("3.5\ndone"), {"id": "q1"})


@pytest.mark.anyio
@pytest.mark.parametrize("bad", ["nan", "inf", "-inf", "Infinity"])
async def test_parse_float_scorer_rejects_non_finite(bad):
    # float() parses these happily; a NaN score would silently corrupt every
    # mean and CI downstream (stats.py never guards — it never had to).
    with pytest.raises(ValueError, match="score"):
        await ParseFloatScorer()(output_with_text(bad), {"id": "q1"})


# --- LlmCondition ------------------------------------------------------------------


def test_llm_condition_payload_matches_build_payload_exactly():
    # Byte-identical arguments to the pre-M10 sample_unit: golden cache keys and
    # every pre-M10 cache entry stay valid.
    spec = llm_spec()
    condition = LlmCondition(spec.variants[0], spec.sampling, MockClient())
    expected = build_payload(
        model="test-model",
        system="Be brief.",
        user="Q: Why?",
        temperature=1.0,
        max_tokens=1024,
        seed=9,
        sample_index=2,
    )
    assert condition.payload({"id": "q1", "input": "Why?"}, seed=9, sample_index=2) == expected


@pytest.mark.anyio
async def test_llm_condition_execute_calls_client_with_payload_fields():
    spec = llm_spec()
    client = MockClient()
    condition = LlmCondition(spec.variants[0], spec.sampling, client)
    payload = condition.payload({"id": "q1", "input": "Why?"}, seed=5, sample_index=1)
    response = await condition.execute(payload)
    request = client.calls[0]
    assert (request.seed, request.sample_index) == (5, 1)
    assert request.user == "Q: Why?"
    assert request.system == "Be brief."
    assert request.model == "test-model"
    assert response.text.startswith("mock:")


@pytest.mark.anyio
async def test_llm_condition_client_error_propagates():
    # The runner's retry loop owns 429/5xx handling; conditions must not swallow.
    spec = llm_spec()
    client = MockClient()
    client.queue_error(429)
    condition = LlmCondition(spec.variants[0], spec.sampling, client)
    payload = condition.payload({"id": "q1", "input": "Why?"}, seed=0, sample_index=0)
    with pytest.raises(ClientError):
        await condition.execute(payload)


# --- CommandCondition --------------------------------------------------------------


def test_command_condition_payload_uses_rendered_argv(tmp_path):
    spec = command_spec(command=["prog", "--case", "{input}", "--seed", "{seed}"])
    condition = CommandCondition(spec.variants[0], tmp_path)
    payload = condition.payload({"id": "q1", "input": "alpha"}, seed=7, sample_index=3)
    assert payload["argv"] == ["prog", "--case", "alpha", "--seed", "7"]
    assert (payload["seed"], payload["sample_index"]) == (7, 3)


@pytest.mark.anyio
async def test_command_condition_executes_and_captures_stdout(tmp_path):
    spec = command_spec(command=[sys.executable, "-c", "print('hello'); print(0.5)"])
    condition = CommandCondition(spec.variants[0], tmp_path)
    payload = condition.payload({"id": "q1", "input": "alpha"}, seed=0, sample_index=0)
    output = await condition.execute(payload)
    assert output.text == "hello\n0.5\n"
    assert output.model == f"command:{sys.executable}"
    assert output.input_tokens == 0
    assert output.output_tokens == 0
    assert output.latency_ms > 0
    assert output.raw["returncode"] == 0
    assert output.raw["argv"][0] == sys.executable


@pytest.mark.anyio
async def test_command_condition_nonzero_exit_raises_condition_error(tmp_path):
    code = "import sys; sys.stderr.write('boom'); sys.exit(3)"
    spec = command_spec(command=[sys.executable, "-c", code])
    condition = CommandCondition(spec.variants[0], tmp_path)
    payload = condition.payload({"id": "q1", "input": "alpha"}, seed=0, sample_index=0)
    with pytest.raises(ConditionError, match="3") as excinfo:
        await condition.execute(payload)
    assert "boom" in str(excinfo.value)


@pytest.mark.anyio
async def test_command_condition_timeout_kills_child(tmp_path):
    spec = ExperimentSpec.model_validate(
        {
            "name": "bench-exp",
            "variants": [
                {
                    "type": "command",
                    "name": "fast",
                    "command": [sys.executable, "-c", "import time; time.sleep(60)"],
                    "timeout_s": 0.2,
                }
            ],
            "dataset": {"items": [{"id": "q1", "input": "alpha"}]},
        }
    )
    condition = CommandCondition(spec.variants[0], tmp_path)
    payload = condition.payload({"id": "q1", "input": "alpha"}, seed=0, sample_index=0)
    with pytest.raises(ConditionError, match="timed out"):
        await condition.execute(payload)


@pytest.mark.anyio
async def test_command_condition_missing_binary_raises_condition_error(tmp_path):
    spec = command_spec(command=["definitely-not-a-real-binary-xyz"])
    condition = CommandCondition(spec.variants[0], tmp_path)
    payload = condition.payload({"id": "q1", "input": "alpha"}, seed=0, sample_index=0)
    with pytest.raises(ConditionError, match="start"):
        await condition.execute(payload)


@pytest.mark.anyio
async def test_command_condition_runs_in_base_dir(tmp_path):
    spec = command_spec(command=[sys.executable, "-c", "import os; print(os.getcwd())"])
    condition = CommandCondition(spec.variants[0], tmp_path)
    payload = condition.payload({"id": "q1", "input": "alpha"}, seed=0, sample_index=0)
    output = await condition.execute(payload)
    assert output.text.strip() == str(tmp_path.resolve())


@pytest.mark.anyio
async def test_command_condition_replaces_invalid_utf8_stdout(tmp_path):
    code = r"import sys; sys.stdout.buffer.write(b'\xff\xfe ok\n1.5\n')"
    spec = command_spec(command=[sys.executable, "-c", code])
    condition = CommandCondition(spec.variants[0], tmp_path)
    payload = condition.payload({"id": "q1", "input": "alpha"}, seed=0, sample_index=0)
    output = await condition.execute(payload)
    assert "�" in output.text
    assert output.text.endswith("1.5\n")


@pytest.mark.anyio
async def test_command_condition_stdin_is_devnull(tmp_path):
    # A child that reads stdin must see EOF immediately, not hang the run.
    code = "import sys; print(len(sys.stdin.read()))"
    spec = command_spec(command=[sys.executable, "-c", code])
    condition = CommandCondition(spec.variants[0], tmp_path)
    payload = condition.payload({"id": "q1", "input": "alpha"}, seed=0, sample_index=0)
    output = await condition.execute(payload)
    assert output.text.strip() == "0"


@pytest.mark.anyio
async def test_command_condition_truncates_stored_stderr(tmp_path):
    code = "import sys; sys.stderr.write('x' * 10000); print(2.0)"
    spec = command_spec(command=[sys.executable, "-c", code])
    condition = CommandCondition(spec.variants[0], tmp_path)
    payload = condition.payload({"id": "q1", "input": "alpha"}, seed=0, sample_index=0)
    output = await condition.execute(payload)
    assert len(output.raw["stderr"]) <= 4096


# --- PythonCondition ---------------------------------------------------------------


def sync_bench(item, seed):
    return f"item={item['id']}\n{seed + 0.5}"


async def async_bench(item, seed):
    return f"async item={item['id']}\n{seed + 0.25}"


def output_bench(item, seed):
    return CompletionResponse(
        text="custom\n9.0",
        raw={"custom": True},
        input_tokens=0,
        output_tokens=0,
        latency_ms=1.0,
        model="python:custom",
    )


def not_a_string_bench(item, seed):
    return 42


def python_spec(callable_path):
    return ExperimentSpec.model_validate(
        {
            "name": "py-exp",
            "variants": [{"type": "python", "name": "fast", "callable": callable_path}],
            "dataset": {"items": [{"id": "q1", "input": "alpha"}]},
        }
    )


@pytest.mark.anyio
async def test_python_condition_calls_sync_fn():
    spec = python_spec("test_conditions:sync_bench")
    conditions = build_conditions(spec, None)
    condition = conditions["fast"]
    payload = condition.payload({"id": "q1", "input": "alpha"}, seed=4, sample_index=0)
    assert payload["item"] == {"id": "q1", "input": "alpha"}  # full item, id included
    output = await condition.execute(payload)
    assert output.text == "item=q1\n4.5"
    assert output.model == "python:test_conditions:sync_bench"


@pytest.mark.anyio
async def test_python_condition_calls_async_fn():
    spec = python_spec("test_conditions:async_bench")
    condition = build_conditions(spec, None)["fast"]
    payload = condition.payload({"id": "q1", "input": "alpha"}, seed=4, sample_index=0)
    output = await condition.execute(payload)
    assert output.text == "async item=q1\n4.25"


@pytest.mark.anyio
async def test_python_condition_fn_may_return_output_directly():
    spec = python_spec("test_conditions:output_bench")
    condition = build_conditions(spec, None)["fast"]
    payload = condition.payload({"id": "q1", "input": "alpha"}, seed=0, sample_index=0)
    output = await condition.execute(payload)
    assert output.text == "custom\n9.0"
    assert output.raw == {"custom": True}


@pytest.mark.anyio
async def test_python_condition_non_string_return_raises():
    spec = python_spec("test_conditions:not_a_string_bench")
    condition = build_conditions(spec, None)["fast"]
    payload = condition.payload({"id": "q1", "input": "alpha"}, seed=0, sample_index=0)
    with pytest.raises(ConditionError, match="int"):
        await condition.execute(payload)


def test_python_condition_bad_module_fails_at_build():
    spec = python_spec("definitely_not_a_module_xyz:fn")
    with pytest.raises(ValueError, match="import"):
        build_conditions(spec, None)


def test_python_condition_missing_attr_fails_at_build():
    spec = python_spec("json:not_there_xyz")
    with pytest.raises(ValueError, match="not_there_xyz"):
        build_conditions(spec, None)


# --- build_conditions / needs_client ----------------------------------------------


def test_build_conditions_maps_llm_variants():
    spec = llm_spec()
    client = MockClient()
    conditions = build_conditions(spec, client)
    assert set(conditions) == {"control", "friendly"}
    assert all(isinstance(c, LlmCondition) for c in conditions.values())


def test_build_conditions_requires_client_for_llm_variants():
    with pytest.raises(ValueError, match="client"):
        build_conditions(llm_spec(), None)


def test_build_conditions_requires_client_for_llm_judge_over_commands():
    spec = command_spec(judge={"model": "j", "mode": "rubric"})
    with pytest.raises(ValueError, match="client"):
        build_conditions(spec, None)


def test_build_conditions_command_spec_needs_no_client(tmp_path):
    conditions = build_conditions(command_spec(), None, base_dir=tmp_path)
    assert isinstance(conditions["fast"], CommandCondition)


def test_needs_client_true_only_for_llm_parts():
    assert needs_client(llm_spec()) is True
    assert needs_client(command_spec()) is False
    assert needs_client(command_spec(judge={"model": "j", "mode": "rubric"})) is True
    assert needs_client(command_spec(judge={"type": "parse_float"})) is False
