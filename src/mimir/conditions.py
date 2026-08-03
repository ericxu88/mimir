"""Condition and Scorer protocols + adapters — M10's generalized core (DESIGN §13).

A Condition is a stochastic system under test: conceptually ``(item, seed) ->
Output``. The protocol splits that call into ``payload()`` (the content-addressed
identity hashed by cache_key) and ``execute(payload)`` (one draw), because the
runner's cache-first fetch and in-flight coalescing (M8) are only sound when the
hashed payload FULLY determines the call — ``execute(payload(item, seed, i))`` IS
the ``(item, seed) -> Output`` call. Output is the existing envelope
(CompletionResponse): one universal shape for the cache, the store, and every
downstream reader.

A Scorer is a PURE per-sample scorer: ``(output, item) -> float``. The LLM judge
is deliberately NOT a Scorer — it is a second stochastic actor with the same
needs as a Condition (cached calls, replicate coordinates, row provenance) and
keeps its own engine in the runner. Comparator is the pairwise analogue's
protocol (two outputs, one verdict); no pure implementation ships in M10
(DESIGN §12).

Errors: LlmCondition lets ClientError propagate (the runner's retry loop owns
429/5xx); command/python failures raise ConditionError, which the runner records
as an error row without retrying.
"""

import asyncio
import importlib
import inspect
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from mimir.cache import build_command_payload, build_payload, build_python_payload
from mimir.clients.base import Client, CompletionRequest, CompletionResponse
from mimir.spec import CommandVariant, ExperimentSpec, PythonVariant, Sampling, Variant

Output = CompletionResponse

# Stored stderr is provenance, not measurement; keep failing chatty programs from
# bloating error rows and cache envelopes.
_STDERR_KEEP_CHARS = 4096


class ConditionError(Exception):
    """A condition failed to produce an output (bad exit, timeout, bad return)."""


class Condition(Protocol):
    """One arm of an experiment: (item, seed) -> Output, split for cache-first."""

    def payload(self, item: dict[str, Any], seed: int, sample_index: int) -> dict[str, Any]: ...

    async def execute(self, payload: dict[str, Any]) -> Output: ...


class Scorer(Protocol):
    """Pure per-sample scoring: (output, item) -> float."""

    async def __call__(self, output: Output, item: dict[str, Any]) -> float: ...


class Comparator(Protocol):
    """Pure pairwise comparison: (output_a, output_b, item) -> "A" | "B" | "TIE".

    Declaration only in M10: the LLM pairwise judge is this shape's cached-actor
    analogue and runs on the runner's judge engine instead.
    """

    async def __call__(self, output_a: Output, output_b: Output, item: dict[str, Any]) -> str: ...


def _item_fields(item: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in item.items() if key != "id"}


def render_argv(command: list[str], item: dict[str, Any], seed: int) -> list[str]:
    """Render each argv element with the item's fields plus {seed} (no shell)."""
    fields = _item_fields(item)
    return [element.format(**fields, seed=seed) for element in command]


async def execute_llm_payload(client: Client, payload: dict[str, Any]) -> Output:
    """One LLM draw from a §3 LLM payload; shared by LlmCondition and the judge.

    Reconstructing the request FROM the payload keeps the hashed identity and the
    executed call provably coherent (the invariant cache coalescing relies on).
    """
    request = CompletionRequest(
        model=payload["model"],
        system=payload["system"],
        user=payload["messages"][0]["content"],
        temperature=payload["params"]["temperature"],
        max_tokens=payload["params"]["max_tokens"],
        seed=payload["seed"],
        sample_index=payload["sample_index"],
    )
    return await client.complete(request)


class LlmCondition:
    """The pre-M10 prompt-variant path as a Condition adapter.

    payload() delegates to build_payload with byte-identical arguments to the old
    sample_unit, so golden cache keys, CRN seed sets, and every pre-M10 cache
    entry stay valid.
    """

    def __init__(self, variant: Variant, sampling: Sampling, client: Client) -> None:
        self._variant = variant
        self._sampling = sampling
        self._client = client

    def payload(self, item: dict[str, Any], seed: int, sample_index: int) -> dict[str, Any]:
        return build_payload(
            model=self._sampling.model,
            system=self._variant.system,
            user=self._variant.user_template.format(**_item_fields(item)),
            temperature=self._sampling.temperature,
            max_tokens=self._sampling.max_tokens,
            seed=seed,
            sample_index=sample_index,
        )

    async def execute(self, payload: dict[str, Any]) -> Output:
        return await execute_llm_payload(self._client, payload)


class CommandCondition:
    """A subprocess condition: seeded argv, stdout is the output text.

    Runs with cwd=base_dir (relative program paths resolve against the spec's
    directory), stdin=DEVNULL (a stdin-reading child sees EOF, never hangs), and
    a kill-on-timeout guard. Output decoding uses errors="replace" so arbitrary
    program output cannot poison UTF-8 invariants downstream.
    """

    def __init__(self, variant: CommandVariant, base_dir: str | Path) -> None:
        self._variant = variant
        self._base_dir = Path(base_dir)

    def payload(self, item: dict[str, Any], seed: int, sample_index: int) -> dict[str, Any]:
        argv = render_argv(self._variant.command, item, seed)
        return build_command_payload(argv=argv, seed=seed, sample_index=sample_index)

    async def execute(self, payload: dict[str, Any]) -> Output:
        argv = payload["argv"]
        start = time.monotonic()
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.DEVNULL,
                cwd=self._base_dir,
            )
        except OSError as exc:
            raise ConditionError(f"failed to start command {argv[0]!r}: {exc}") from exc
        try:
            async with asyncio.timeout(self._variant.timeout_s):
                stdout, stderr = await process.communicate()
        except TimeoutError:
            raise ConditionError(
                f"command {argv[0]!r} timed out after {self._variant.timeout_s}s"
            ) from None
        finally:
            # Reaps the child on timeout AND on TaskGroup cancellation.
            if process.returncode is None:
                process.kill()
                await process.wait()
        latency_ms = (time.monotonic() - start) * 1000.0
        text = stdout.decode("utf-8", errors="replace")
        stderr_text = stderr.decode("utf-8", errors="replace")[-_STDERR_KEEP_CHARS:]
        if process.returncode != 0:
            raise ConditionError(
                f"command {argv[0]!r} exited {process.returncode}; stderr: {stderr_text[-500:]!r}"
            )
        return CompletionResponse(
            text=text,
            raw={"argv": list(argv), "returncode": 0, "stderr": stderr_text},
            input_tokens=0,
            output_tokens=0,
            latency_ms=latency_ms,
            model=f"command:{argv[0]}",
        )


class PythonCondition:
    """A python-callable condition: fn(item, seed) -> str | Output.

    The callable is resolved at build_conditions time so a bad import path fails
    before any run row exists. Sync callables run in a thread (they must not
    block the event loop); async callables are awaited directly.
    """

    def __init__(self, variant: PythonVariant, fn: Callable[..., Any]) -> None:
        self._variant = variant
        self._fn = fn

    def payload(self, item: dict[str, Any], seed: int, sample_index: int) -> dict[str, Any]:
        return build_python_payload(
            callable_path=self._variant.callable, item=item, seed=seed, sample_index=sample_index
        )

    async def execute(self, payload: dict[str, Any]) -> Output:
        start = time.monotonic()
        if inspect.iscoroutinefunction(self._fn):
            result = await self._fn(payload["item"], payload["seed"])
        else:
            result = await asyncio.to_thread(self._fn, payload["item"], payload["seed"])
        latency_ms = (time.monotonic() - start) * 1000.0
        if isinstance(result, CompletionResponse):
            return result
        if not isinstance(result, str):
            raise ConditionError(
                f"python condition {self._variant.callable!r} returned"
                f" {type(result).__name__}, expected str or Output"
            )
        return CompletionResponse(
            text=result,
            raw={"callable": self._variant.callable},
            input_tokens=0,
            output_tokens=0,
            latency_ms=latency_ms,
            model=f"python:{self._variant.callable}",
        )


def _resolve_callable(path: str) -> Callable[..., Any]:
    module_name, _, attr = path.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ValueError(f"cannot import python condition {path!r}: {exc}") from exc
    try:
        fn = getattr(module, attr)
    except AttributeError:
        raise ValueError(
            f"module {module_name!r} has no attribute {attr!r} (python condition {path!r})"
        ) from None
    if not callable(fn):
        raise ValueError(f"python condition {path!r} is not callable: {fn!r}")
    return fn


def needs_client(spec: ExperimentSpec) -> bool:
    """True iff the spec has any LLM part (llm variant or llm judge)."""
    return any(variant.type == "llm" for variant in spec.variants) or (
        spec.judge is not None and spec.judge.type == "llm"
    )


def build_conditions(
    spec: ExperimentSpec, client: Client | None, base_dir: str | Path = "."
) -> dict[str, Condition]:
    """Map each spec variant to its Condition adapter; validates before any run row."""
    if client is None and needs_client(spec):
        raise ValueError(
            "spec requires an LLM client (llm variants or an llm judge) but none was provided;"
            " pass a client or remove the llm parts"
        )
    conditions: dict[str, Condition] = {}
    for variant in spec.variants:
        if variant.type == "llm":
            conditions[variant.name] = LlmCondition(variant, spec.sampling, client)
        elif variant.type == "command":
            conditions[variant.name] = CommandCondition(variant, base_dir)
        else:
            conditions[variant.name] = PythonCondition(variant, _resolve_callable(variant.callable))
    return conditions


class ParseFloatScorer:
    """Scorer: parse the final non-empty line of the output text as a finite float."""

    async def __call__(self, output: Output, item: dict[str, Any]) -> float:
        lines = [line.strip() for line in output.text.splitlines() if line.strip()]
        if not lines:
            raise ValueError("empty output: no score line")
        try:
            value = float(lines[-1])
        except ValueError:
            raise ValueError(
                f"unparseable score {lines[-1]!r}: expected a float on the final line"
            ) from None
        if not math.isfinite(value):
            raise ValueError(f"non-finite score {lines[-1]!r}: scores must be finite")
        return value
