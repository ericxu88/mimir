"""Tests for mimir.clients.anthropic — the real Messages API adapter (DESIGN.md §6).

Every AnthropicClient here is constructed with an httpx.MockTransport and a fake
key, so live calls are structurally impossible. httpx.MockTransport accepts sync
handlers even under AsyncClient. The autouse fixture strips any real
ANTHROPIC_API_KEY from the environment so no test can silently inherit one.
"""

import json

import httpx
import pytest

import mimir.runner as runner_mod
from mimir.clients.anthropic import AnthropicClient
from mimir.clients.base import ClientError, CompletionRequest
from mimir.runner import run_experiment
from mimir.spec import ExperimentSpec
from mimir.store import Store


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    # Dev machines may export a real key; no test in this file may inherit it.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "mimir.db")
    yield s
    s.close()


def make_request(**overrides):
    base = {
        "model": "claude-haiku-4-5-20251001",
        "system": "You are a helpful assistant.",
        "user": "hello",
        "temperature": 1.0,
        "max_tokens": 16,
        "seed": 0,
        "sample_index": 0,
    }
    base.update(overrides)
    return CompletionRequest(**base)


def anthropic_body(
    *,
    text="hi",
    model="claude-haiku-4-5-20251001",
    input_tokens=10,
    output_tokens=5,
    content=None,
):
    return {
        "id": "msg_01",
        "type": "message",
        "role": "assistant",
        "content": content if content is not None else [{"type": "text", "text": text}],
        "model": model,
        "stop_reason": "end_turn",
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


def capture_transport(status=200, body=None):
    """A transport that records the last request and its decoded JSON body."""
    captured = {}

    def handler(request):
        captured["request"] = request
        captured["body"] = json.loads(request.content)
        return httpx.Response(status, json=body if body is not None else anthropic_body())

    return httpx.MockTransport(handler), captured


def make_client(monkeypatch, transport, key="sk-test-123"):
    monkeypatch.setenv("ANTHROPIC_API_KEY", key)
    return AnthropicClient(transport=transport)


# --- construction: env-only key --------------------------------------------------


def test_missing_api_key_raises_value_error():
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        AnthropicClient()


def test_empty_api_key_raises_value_error(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        AnthropicClient()


@pytest.mark.anyio
async def test_api_key_captured_at_construction(monkeypatch):
    transport, captured = capture_transport()
    client = make_client(monkeypatch, transport)
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    await client.complete(make_request())
    assert captured["request"].headers["x-api-key"] == "sk-test-123"


# --- wire pins --------------------------------------------------------------------


@pytest.mark.anyio
async def test_wire_body_never_contains_seed_or_sample_index(monkeypatch):
    transport, captured = capture_transport()
    client = make_client(monkeypatch, transport)
    await client.complete(make_request(seed=123, sample_index=7, temperature=0.7))
    # Exact-set equality: also forbids any future field from leaking on the wire
    # unreviewed (the Messages API has no seed parameter — DESIGN §3).
    assert set(captured["body"]) == {"model", "max_tokens", "messages", "system", "temperature"}


@pytest.mark.anyio
async def test_headers_and_endpoint_exact(monkeypatch):
    transport, captured = capture_transport()
    client = make_client(monkeypatch, transport)
    await client.complete(make_request())
    request = captured["request"]
    assert request.method == "POST"
    assert str(request.url) == "https://api.anthropic.com/v1/messages"
    assert request.headers["x-api-key"] == "sk-test-123"
    assert request.headers["anthropic-version"] == "2023-06-01"
    assert request.headers["content-type"].startswith("application/json")


@pytest.mark.anyio
async def test_system_omitted_when_empty_present_otherwise(monkeypatch):
    transport, captured = capture_transport()
    client = make_client(monkeypatch, transport)
    await client.complete(make_request(system=""))
    assert "system" not in captured["body"]
    await client.complete(make_request(system="You are terse."))
    assert captured["body"]["system"] == "You are terse."


@pytest.mark.anyio
async def test_temperature_omitted_at_default_one(monkeypatch):
    transport, captured = capture_transport()
    client = make_client(monkeypatch, transport)
    await client.complete(make_request(temperature=1.0))
    assert "temperature" not in captured["body"]


@pytest.mark.anyio
@pytest.mark.parametrize("temperature", [0.0, 0.7])
async def test_temperature_sent_when_not_default(monkeypatch, temperature):
    # 0.0 is the judge sampling default — it must go on the wire.
    transport, captured = capture_transport()
    client = make_client(monkeypatch, transport)
    await client.complete(make_request(temperature=temperature))
    assert captured["body"]["temperature"] == temperature


@pytest.mark.anyio
async def test_body_maps_model_max_tokens_messages(monkeypatch):
    transport, captured = capture_transport()
    client = make_client(monkeypatch, transport)
    await client.complete(make_request(user="what is up", max_tokens=99))
    body = captured["body"]
    assert body["model"] == "claude-haiku-4-5-20251001"
    assert body["max_tokens"] == 99
    assert body["messages"] == [{"role": "user", "content": "what is up"}]


# --- response parsing ---------------------------------------------------------------


@pytest.mark.anyio
async def test_multi_text_block_concat_skips_non_text(monkeypatch):
    content = [
        {"type": "text", "text": "foo"},
        {"type": "thinking", "thinking": "..."},
        {"type": "text", "text": "bar"},
    ]
    transport, _ = capture_transport(body=anthropic_body(content=content))
    client = make_client(monkeypatch, transport)
    response = await client.complete(make_request())
    assert response.text == "foobar"


@pytest.mark.anyio
async def test_usage_and_model_come_from_response_body(monkeypatch):
    body = anthropic_body(model="claude-haiku-4-5-20251001", input_tokens=17, output_tokens=42)
    transport, _ = capture_transport(body=body)
    client = make_client(monkeypatch, transport)
    # Request uses the undated alias; the envelope must record what actually served.
    response = await client.complete(make_request(model="claude-haiku-4-5"))
    assert response.model == "claude-haiku-4-5-20251001"
    assert response.input_tokens == 17
    assert response.output_tokens == 42


@pytest.mark.anyio
async def test_raw_is_full_payload_verbatim(monkeypatch):
    body = anthropic_body()
    transport, _ = capture_transport(body=body)
    client = make_client(monkeypatch, transport)
    response = await client.complete(make_request())
    assert response.raw == body


@pytest.mark.anyio
async def test_latency_is_measured_float(monkeypatch):
    transport, _ = capture_transport()
    client = make_client(monkeypatch, transport)
    response = await client.complete(make_request())
    assert isinstance(response.latency_ms, float)
    assert response.latency_ms >= 0.0


@pytest.mark.anyio
async def test_envelope_field_set_locked(monkeypatch):
    transport, _ = capture_transport()
    client = make_client(monkeypatch, transport)
    response = await client.complete(make_request())
    assert set(response.model_dump()) == {
        "text",
        "raw",
        "input_tokens",
        "output_tokens",
        "latency_ms",
        "model",
    }


# --- error mapping ------------------------------------------------------------------


@pytest.mark.anyio
@pytest.mark.parametrize("status", [400, 401, 429, 500, 529])
async def test_error_status_passthrough_with_body_message(monkeypatch, status):
    error_body = {"type": "error", "error": {"type": "some_error", "message": "kaboom"}}
    transport, _ = capture_transport(status=status, body=error_body)
    client = make_client(monkeypatch, transport)
    with pytest.raises(ClientError) as excinfo:
        await client.complete(make_request())
    assert excinfo.value.status_code == status
    assert "kaboom" in str(excinfo.value)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "response_kwargs",
    [{"text": "<html>gateway exploded</html>"}, {"json": {"unexpected": True}}],
)
async def test_malformed_error_body_falls_back_to_default_message(monkeypatch, response_kwargs):
    def handler(request):
        return httpx.Response(500, **response_kwargs)

    client = make_client(monkeypatch, httpx.MockTransport(handler))
    with pytest.raises(ClientError) as excinfo:
        await client.complete(make_request())
    assert excinfo.value.status_code == 500
    assert "500" in str(excinfo.value)  # ClientError's built-in default message


@pytest.mark.anyio
async def test_transport_error_escapes_unmapped(monkeypatch):
    # Pinned decision: no synthetic retryable ClientError for network failures —
    # the real exception reaches the runner and becomes an error row.
    def handler(request):
        raise httpx.ConnectError("boom")

    client = make_client(monkeypatch, httpx.MockTransport(handler))
    with pytest.raises(httpx.ConnectError):
        await client.complete(make_request())


# --- integration: adapter errors drive the runner's retry loop -----------------------


@pytest.mark.anyio
async def test_run_experiment_retries_through_adapter(store, monkeypatch):
    monkeypatch.setattr(runner_mod, "_BASE_DELAY_S", 0.001)
    monkeypatch.setattr(runner_mod, "_MAX_DELAY_S", 0.002)
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        if len(requests) <= 2:
            error = {"type": "error", "error": {"type": "rate_limit_error", "message": "slow"}}
            return httpx.Response(429, json=error)
        return httpx.Response(200, json=anthropic_body(text="hi", input_tokens=10))

    client = make_client(monkeypatch, httpx.MockTransport(handler))
    spec = ExperimentSpec.model_validate(
        {
            "name": "adapter-retry",
            "variants": [{"name": "control", "system": "s", "user_template": "Answer: {input}"}],
            "dataset": {"items": [{"id": "q1", "input": "only one"}]},
            "sampling": {"model": "claude-haiku-4-5-20251001"},
            "limits": {"concurrency": 4, "requests_per_minute": 100_000},
        }
    )
    run_id = await run_experiment(spec, store, client)

    assert len(requests) == 3  # 429, 429, then 200
    assert store.get_run(run_id)["status"] == "complete"
    (row,) = store.get_samples(run_id)
    assert row["error"] is None
    assert row["cache_hit"] == 0
    assert row["response_text"] == "hi"
    assert row["input_tokens"] == 10
    assert all("seed" not in body for body in requests)
