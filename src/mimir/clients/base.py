"""Client contract (DESIGN.md §6): request/response models and the async protocol.

CompletionResponse.model_dump() IS the cache envelope stored by Store.cache_put —
its field set must stay exactly {text, raw, input_tokens, output_tokens,
latency_ms, model}.
"""

from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict


class ClientError(Exception):
    """A provider-level failure. 429 and 5xx are retryable; everything else is not.

    `retry_after` (seconds, M9) carries the provider's retry-after header when one
    was sent; the runner uses it as a floor under its exponential backoff.
    """

    def __init__(
        self, status_code: int, message: str = "", *, retry_after: float | None = None
    ) -> None:
        self.status_code = status_code
        self.retry_after = retry_after
        super().__init__(message or f"client error: status {status_code}")


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    system: str = ""
    user: str
    temperature: float
    max_tokens: int
    seed: int  # Mimir-internal: never sent over the wire
    sample_index: int  # Mimir-internal: never sent over the wire


class CompletionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    raw: dict[str, Any]
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model: str


class Client(Protocol):
    async def complete(self, request: CompletionRequest) -> CompletionResponse: ...
