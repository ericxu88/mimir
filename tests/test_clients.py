"""Tests for mimir.clients — base contract and the deterministic MockClient."""

import pytest

from mimir.clients.base import ClientError, CompletionRequest, CompletionResponse
from mimir.clients.mock import MockClient


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


@pytest.mark.anyio
async def test_mock_is_deterministic_across_instances():
    first = await MockClient().complete(make_request())
    second = await MockClient().complete(make_request())
    assert first.text == second.text
    assert first.text.startswith("mock:")


@pytest.mark.anyio
async def test_mock_text_differs_by_sample_index_and_seed():
    base = await MockClient().complete(make_request())
    replicate = await MockClient().complete(make_request(sample_index=1))
    reseeded = await MockClient().complete(make_request(seed=99))
    assert base.text != replicate.text
    assert base.text != reseeded.text


@pytest.mark.anyio
async def test_mock_records_call_log():
    client = MockClient()
    await client.complete(make_request(user="one"))
    await client.complete(make_request(user="two"))
    assert [request.user for request in client.calls] == ["one", "two"]


@pytest.mark.anyio
async def test_mock_rule_overrides_text():
    client = MockClient()
    client.add_rule(lambda request: "judge" in request.user, "A")
    ruled = await client.complete(make_request(user="judge this pair"))
    unruled = await client.complete(make_request(user="plain question"))
    assert ruled.text == "A"
    assert unruled.text.startswith("mock:")


@pytest.mark.anyio
async def test_mock_queued_error_then_recovery():
    client = MockClient()
    client.queue_error(429, times=1)
    with pytest.raises(ClientError) as excinfo:
        await client.complete(make_request())
    assert excinfo.value.status_code == 429
    response = await client.complete(make_request())
    assert response.text.startswith("mock:")
    assert len(client.calls) == 2


def test_completion_response_dump_matches_envelope():
    response = CompletionResponse(
        text="hi",
        raw={"content": [{"type": "text", "text": "hi"}]},
        input_tokens=3,
        output_tokens=2,
        latency_ms=1.5,
        model="m",
    )
    dumped = response.model_dump()
    assert set(dumped) == {"text", "raw", "input_tokens", "output_tokens", "latency_ms", "model"}


def test_client_error_carries_status_code():
    error = ClientError(429)
    assert error.status_code == 429
    assert "429" in str(error)


def test_client_error_carries_retry_after():
    # Provider backpressure (M9): the adapter forwards the retry-after header so
    # the runner can floor its backoff on it; absent by default.
    assert ClientError(429).retry_after is None
    error = ClientError(429, retry_after=30.0)
    assert error.retry_after == 30.0
    assert error.status_code == 429
