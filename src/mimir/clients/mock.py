"""Deterministic MockClient — the only client tests ever use (brief rule 3).

Response text is derived from the cache key of the request's payload, so the same
request always yields the same text while any content change (including seed or
sample_index) yields a different one. The call log lets tests assert that cache
hits skip client calls entirely.
"""

import asyncio
from collections.abc import Callable

from mimir.cache import build_payload, cache_key
from mimir.clients.base import ClientError, CompletionRequest, CompletionResponse


class MockClient:
    def __init__(self, *, latency_s: float = 0.0) -> None:
        self.calls: list[CompletionRequest] = []
        self.max_in_flight = 0
        self._in_flight = 0
        self._latency_s = latency_s
        self._rules: list[tuple[Callable[[CompletionRequest], bool], str]] = []
        self._errors: list[ClientError] = []

    def add_rule(self, predicate: Callable[[CompletionRequest], bool], text: str) -> None:
        """First matching rule supplies the response text; otherwise it is derived."""
        self._rules.append((predicate, text))

    def queue_error(self, status_code: int, times: int = 1) -> None:
        """The next `times` calls raise ClientError(status_code), FIFO."""
        self._errors.extend(ClientError(status_code) for _ in range(times))

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.calls.append(request)
        self._in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self._in_flight)
        try:
            await asyncio.sleep(self._latency_s)
            if self._errors:
                raise self._errors.pop(0)
            text = self._response_text(request)
            return CompletionResponse(
                text=text,
                raw={"content": [{"type": "text", "text": text}], "model": request.model},
                input_tokens=max(1, len(request.user) // 4),
                output_tokens=max(1, len(text) // 4),
                latency_ms=1.0,
                model=request.model,
            )
        finally:
            self._in_flight -= 1

    def _response_text(self, request: CompletionRequest) -> str:
        for predicate, text in self._rules:
            if predicate(request):
                return text
        payload = build_payload(
            model=request.model,
            system=request.system,
            user=request.user,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            seed=request.seed,
            sample_index=request.sample_index,
        )
        return f"mock:{cache_key(payload)[:16]}"
