"""Anthropic Messages API adapter (DESIGN.md §6) — the one module that touches the wire.

Wire contract: POST https://api.anthropic.com/v1/messages with the x-api-key and
anthropic-version headers. `system` is omitted when empty; `temperature` is omitted
when it equals 1.0 (the API default — newer models reject the parameter outright, so
default-temperature specs stay runnable everywhere, while an explicit non-default
temperature is always sent and may fail honestly as 400 error rows). `seed` and
`sample_index` are Mimir-internal and never sent (§3). Non-2xx responses raise
ClientError(status, message-from-error-body); transport errors (connect/timeout)
escape unmapped — the runner records them as error rows without retry, and a
cache-first re-run re-executes only the failed units. The API key comes from
ANTHROPIC_API_KEY only, read at construction so the CLI fails before touching the db.
"""

import os
import time
from typing import Any

import httpx

from mimir.clients.base import ClientError, CompletionRequest, CompletionResponse

_API_URL = "https://api.anthropic.com/v1/messages"
_API_VERSION = "2023-06-01"
_DEFAULT_TIMEOUT_S = 120.0  # httpx's 5s default is far too short for LLM calls


class AnthropicClient:
    def __init__(
        self,
        *,
        timeout: float = _DEFAULT_TIMEOUT_S,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set; export it or pass --mock for offline runs"
            )
        self._api_key = api_key
        self._timeout = httpx.Timeout(timeout)
        self._transport = transport

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        body = _build_body(request)
        headers = {"x-api-key": self._api_key, "anthropic-version": _API_VERSION}
        # Per-call client: the Client protocol has no close(), and everything runs
        # inside one asyncio.run — a shared AsyncClient would leak or need an owner.
        start = time.monotonic()
        async with httpx.AsyncClient(transport=self._transport, timeout=self._timeout) as http:
            response = await http.post(_API_URL, json=body, headers=headers)
        latency_ms = (time.monotonic() - start) * 1000.0

        if not (200 <= response.status_code < 300):
            raise ClientError(response.status_code, _error_message(response))

        data = response.json()
        text = "".join(
            block["text"] for block in data["content"] if block.get("type") == "text"
        )
        return CompletionResponse(
            text=text,
            raw=data,
            input_tokens=data["usage"]["input_tokens"],
            output_tokens=data["usage"]["output_tokens"],
            latency_ms=latency_ms,
            model=data["model"],
        )


def _build_body(request: CompletionRequest) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": request.model,
        "max_tokens": request.max_tokens,
        "messages": [{"role": "user", "content": request.user}],
    }
    if request.system:
        body["system"] = request.system
    if request.temperature != 1.0:
        body["temperature"] = request.temperature
    return body


def _error_message(response: httpx.Response) -> str:
    # ValueError covers json.JSONDecodeError across httpx versions.
    try:
        return response.json()["error"]["message"]
    except (ValueError, KeyError, TypeError):
        return ""
