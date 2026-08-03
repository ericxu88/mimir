"""Async cache-first runner (DESIGN.md §5).

Work units = variants x items x n_samples completion calls, plus judge units when a
judge is configured. Every unit is cache-first: on a hit the client is never called
and the sample row copies latency/tokens from the cached envelope. Concurrency is
capped by a semaphore, request starts are paced by a token bucket, and 429/5xx are
retried with exponential backoff + jitter. Unit failures are recorded as error rows,
never raised — a run always reaches a terminal status (running -> complete | failed).
"""

import asyncio
import json
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mimir.cache import build_payload, cache_key, canonical_json
from mimir.clients.base import Client, ClientError, CompletionRequest, CompletionResponse
from mimir.spec import ExperimentSpec, load_items
from mimir.store import Store

_MAX_ATTEMPTS = 5
_BASE_DELAY_S = 1.0
_MAX_DELAY_S = 60.0
_SQLITE_INT_MAX = 2**63 - 1

_sleep = asyncio.sleep


def _sanitized_envelope(envelope: dict[str, Any]) -> dict[str, Any]:
    """Apply the store's lone-surrogate replacement up front (json.loads accepts
    "\\ud800" escapes from provider payloads) so the in-memory envelope is identical
    to what a later cache hit returns — judge prompts and their cache keys must not
    differ between a first run and a re-run."""
    text = json.dumps(envelope, ensure_ascii=False)
    return json.loads(text.encode("utf-8", "replace").decode("utf-8"))


def _validated_envelope(envelope: dict[str, Any]) -> dict[str, Any]:
    """Type-check an envelope (cache rows can be corrupted out of band) and bound the
    token counts, which SQLite INTEGER columns cannot hold beyond 64 bits — both
    failure modes must land in the unit error-row path, not abort the TaskGroup."""
    checked = CompletionResponse.model_validate(envelope).model_dump()
    for field in ("input_tokens", "output_tokens"):
        if abs(checked[field]) > _SQLITE_INT_MAX:
            raise ValueError(f"envelope {field} out of SQLite integer range: {checked[field]}")
    return checked


def parse_pairwise_verdict(text: str) -> str:
    """Final non-empty line must be A, B, or TIE (case-insensitive)."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("empty judge output: no verdict line")
    verdict = lines[-1].upper()
    if verdict not in ("A", "B", "TIE"):
        raise ValueError(f"unparseable verdict {lines[-1]!r}: expected final line A, B, or TIE")
    return verdict


def parse_rubric_score(text: str) -> float:
    """Final non-empty line must be an integer 1-10."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("empty judge output: no score line")
    try:
        value = int(lines[-1])
    except ValueError:
        raise ValueError(
            f"unparseable score {lines[-1]!r}: expected an integer 1-10 on the final line"
        ) from None
    if not 1 <= value <= 10:
        raise ValueError(f"score {value} out of range 1-10")
    return float(value)


class TokenBucket:
    """Classic token bucket: burst up to `capacity`, then sustained rate_per_minute.

    Default capacity equals rate_per_minute (full first-minute burst). `clock` and
    `sleep` are injectable for deterministic tests.
    """

    def __init__(
        self,
        rate_per_minute: float,
        capacity: float | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Any] | None = None,
    ) -> None:
        self._rate = rate_per_minute / 60.0
        self._capacity = float(capacity if capacity is not None else max(1.0, rate_per_minute))
        self._tokens = self._capacity
        self._clock = clock
        self._sleep = sleep if sleep is not None else asyncio.sleep
        self._updated = clock()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            while True:
                now = self._clock()
                self._tokens = min(
                    self._capacity, self._tokens + (now - self._updated) * self._rate
                )
                self._updated = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                await self._sleep((1.0 - self._tokens) / self._rate)


async def _call_with_retry(
    client: Client,
    request: CompletionRequest,
    semaphore: asyncio.Semaphore,
    bucket: TokenBucket,
):
    for attempt in range(_MAX_ATTEMPTS):
        retry_after = None
        async with semaphore:
            await bucket.acquire()
            try:
                return await client.complete(request)
            except ClientError as exc:
                retryable = exc.status_code == 429 or 500 <= exc.status_code < 600
                if not retryable or attempt == _MAX_ATTEMPTS - 1:
                    raise
                retry_after = exc.retry_after
        # Semaphore released during backoff so other units can proceed.
        delay = min(_MAX_DELAY_S, _BASE_DELAY_S * 2**attempt)
        wait = delay * random.uniform(0.5, 1.0)
        if retry_after is not None:
            # Provider backpressure floors the jittered backoff (M9); the cap keeps
            # an absurd retry-after from parking a unit past _MAX_DELAY_S.
            wait = max(wait, min(_MAX_DELAY_S, retry_after))
        await _sleep(wait)
    raise AssertionError("unreachable")


def _item_fields(item: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in item.items() if key != "id"}


async def run_experiment(
    spec: ExperimentSpec,
    store: Store,
    client: Client,
    *,
    base_dir: str | Path = ".",
) -> str:
    """Execute a spec against the store cache-first; returns the new run id."""
    items = load_items(spec, base_dir)  # validation failures create no run row
    run_id = store.create_run(spec.name, spec.model_dump())
    try:
        await _execute_run(spec, store, client, items, run_id)
    except BaseException:
        # Backstop: unit handlers record errors instead of raising, but if
        # anything unexpected escapes, the run must still reach a terminal state.
        store.set_run_status(run_id, "failed")
        raise
    return run_id


async def _execute_run(
    spec: ExperimentSpec,
    store: Store,
    client: Client,
    items: list[dict[str, Any]],
    run_id: str,
) -> None:
    condition_ids = {
        variant.name: store.add_condition(
            run_id,
            variant_name=variant.name,
            system_prompt=variant.system,
            user_template=variant.user_template,
            sampling=spec.sampling.model_dump(),
        )
        for variant in spec.variants
    }

    semaphore = asyncio.Semaphore(spec.limits.concurrency)
    bucket = TokenBucket(spec.limits.requests_per_minute)
    # (variant_name, item_id, sample_index) -> (sample_id, response_text)
    results: dict[tuple[str, str, int], tuple[int, str]] = {}
    had_error = False

    judge = spec.judge
    judge_template = judge.resolved_prompt_template() if judge is not None else ""

    # M8/M5: units that share a cache key must share ONE provider call. Without
    # this, concurrent duplicates each called the provider and INSERT OR IGNORE kept
    # only the first envelope, leaving every other row pointing at a cache entry
    # whose text differed from the row's own response. Identical keys mean identical
    # requests (replicate and presentation order are IN the key), so coalescing can
    # never merge two measurements that were supposed to be independent draws.
    inflight: dict[str, asyncio.Future[dict[str, Any]]] = {}

    async def fetch(payload: dict[str, Any], key: str) -> tuple[dict[str, Any], bool]:
        """Cache-first envelope fetch; a hit never touches the client."""
        envelope = store.cache_get(key)
        if envelope is not None:
            return _validated_envelope(envelope), True
        pending = inflight.get(key)
        if pending is not None:
            # Another unit is already fetching this exact payload; share its result.
            return _validated_envelope(await asyncio.shield(pending)), True
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        # Consume the exception if nobody waited, so a failed unit cannot emit a
        # spurious "Future exception was never retrieved" warning.
        future.add_done_callback(lambda f: f.cancelled() or f.exception())
        inflight[key] = future
        try:
            envelope = await _fetch_uncached(payload, key)
        except BaseException as exc:
            future.set_exception(exc)
            raise
        finally:
            inflight.pop(key, None)
        future.set_result(envelope)
        return envelope, False

    async def _fetch_uncached(payload: dict[str, Any], key: str) -> dict[str, Any]:
        request = CompletionRequest(
            model=payload["model"],
            system=payload["system"],
            user=payload["messages"][0]["content"],
            temperature=payload["params"]["temperature"],
            max_tokens=payload["params"]["max_tokens"],
            seed=payload["seed"],
            sample_index=payload["sample_index"],
        )
        response = await _call_with_retry(client, request, semaphore, bucket)
        # Validate BEFORE cache_put: a garbage envelope must not poison the cache.
        envelope = _validated_envelope(_sanitized_envelope(response.model_dump()))
        store.cache_put(key, envelope)
        return envelope

    async def sample_unit(variant, item: dict[str, Any], index: int) -> None:
        nonlocal had_error
        key = ""
        request_json = ""
        try:
            payload = build_payload(
                model=spec.sampling.model,
                system=variant.system,
                user=variant.user_template.format(**_item_fields(item)),
                temperature=spec.sampling.temperature,
                max_tokens=spec.sampling.max_tokens,
                # CRN (M7): one seed per replicate, shared across variants and
                # items, so every condition faces the same labeled random states
                # (DESIGN §3/§5). Index 0 still carries sampling.seed exactly.
                seed=spec.sampling.seed + index,
                sample_index=index,
            )
            key = cache_key(payload)
            request_json = canonical_json(payload).decode("utf-8")
            envelope, hit = await fetch(payload, key)
            raw_response = json.dumps(envelope["raw"], ensure_ascii=False)
            response_text = envelope["text"]
            latency_ms = envelope["latency_ms"]
            input_tokens = envelope["input_tokens"]
            output_tokens = envelope["output_tokens"]
        except Exception as exc:
            # Record, never raise: a run must always reach a terminal status.
            had_error = True
            store.add_sample(
                run_id=run_id,
                condition_id=condition_ids[variant.name],
                item_id=item["id"],
                sample_index=index,
                cache_key=key,
                request_json=request_json,
                error=f"{type(exc).__name__}: {exc}",
            )
            return
        sample_id = store.add_sample(
            run_id=run_id,
            condition_id=condition_ids[variant.name],
            item_id=item["id"],
            sample_index=index,
            cache_key=key,
            request_json=request_json,
            raw_response=raw_response,
            response_text=response_text,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_hit=hit,
        )
        results[(variant.name, item["id"], index)] = (sample_id, response_text)

    async def judge_unit(
        item: dict[str, Any],
        render: Callable[[], str],
        row: dict[str, Any],
        parse: Callable[[str], Any],
        result_field: str,
        judge_index: int,
    ) -> None:
        nonlocal had_error
        row = {**row, "run_id": run_id, "item_id": item["id"], "cache_key": ""}
        try:
            payload = build_payload(
                model=judge.model,
                system="",
                user=render(),
                temperature=judge.temperature,
                max_tokens=judge.max_tokens,
                seed=spec.sampling.seed,
                # M8/C1: the judge's payload coordinate carries the replicate and
                # the presentation order (DESIGN §5). Without it every judge unit
                # of an item shared sample_index=0, so units whose prompts happened
                # to coincide — a temp-0 model repeating text across replicates, or
                # two variants answering identically — collapsed onto ONE cache
                # entry and the single cached verdict was consumed as if it were
                # independent draws. That silently zeroed M7's replicate noise and
                # pinned M4's flip rate at 1.0.
                sample_index=judge_index,
            )
            row["cache_key"] = cache_key(payload)
            envelope, _ = await fetch(payload, row["cache_key"])
            row["raw_response"] = json.dumps(envelope["raw"], ensure_ascii=False)
            row["latency_ms"] = envelope["latency_ms"]
            row["input_tokens"] = envelope["input_tokens"]
            row["output_tokens"] = envelope["output_tokens"]
            text = envelope["text"]
        except Exception as exc:
            # Record, never raise: a run must always reach a terminal status.
            had_error = True
            store.add_judgment(**row, error=f"{type(exc).__name__}: {exc}")
            return
        try:
            row[result_field] = parse(text)
        except ValueError as exc:
            had_error = True
            store.add_judgment(**row, error=str(exc))
            return
        store.add_judgment(**row)

    def pairwise_task(item: dict[str, Any], index: int, order: str):
        first = results[(spec.variants[0].name, item["id"], index)]
        second = results[(spec.variants[1].name, item["id"], index)]
        if order == "ba":
            first, second = second, first

        def render() -> str:
            # Rendered inside judge_unit's try so format errors become error rows.
            return judge_template.format(
                **_item_fields(item), response_a=first[1], response_b=second[1]
            )

        row = {
            "judge_model": judge.model,
            "mode": "pairwise",
            "sample_a_id": first[0],  # the sample PRESENTED in position A
            "sample_b_id": second[0],
            "position_order": order,
        }
        # 'ab' of replicate 0 keeps coordinate 0, so pre-M8 judge cache entries for
        # that unit still hit; every other unit re-executes (re-spent, never wrong).
        judge_index = 2 * index + (0 if order == "ab" else 1)
        return judge_unit(item, render, row, parse_pairwise_verdict, "verdict", judge_index)

    def rubric_task(variant_name: str, item: dict[str, Any], index: int):
        sample_id, text = results[(variant_name, item["id"], index)]

        def render() -> str:
            return judge_template.format(**_item_fields(item), response=text)

        row = {"judge_model": judge.model, "mode": "rubric", "sample_a_id": sample_id}
        return judge_unit(item, render, row, parse_rubric_score, "score", index)

    async with asyncio.TaskGroup() as task_group:
        for variant in spec.variants:
            for item in items:
                for index in range(spec.n_samples):
                    task_group.create_task(sample_unit(variant, item, index))

    if judge is not None:
        async with asyncio.TaskGroup() as task_group:
            if judge.mode == "pairwise":
                orders = ("ab", "ba") if judge.position_swap else ("ab",)
                name_a, name_b = spec.variants[0].name, spec.variants[1].name
                for item in items:
                    for index in range(spec.n_samples):
                        pair_complete = (name_a, item["id"], index) in results and (
                            name_b,
                            item["id"],
                            index,
                        ) in results
                        if pair_complete:  # pairs with failed samples are skipped
                            for order in orders:
                                task_group.create_task(pairwise_task(item, index, order))
            else:
                for variant in spec.variants:
                    for item in items:
                        for index in range(spec.n_samples):
                            if (variant.name, item["id"], index) in results:
                                task_group.create_task(rubric_task(variant.name, item, index))

    store.set_run_status(run_id, "failed" if had_error else "complete")
