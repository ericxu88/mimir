"""Tests for mimir.runner — async cache-first scheduler (DESIGN.md §5).

The MockClient call log is the oracle for the milestone's two core proofs:
cache hits skip client calls, and a re-run with one changed variant executes
only the delta.
"""

import json

import pytest

import mimir.runner as runner_mod
from mimir.cache import build_payload, cache_key
from mimir.clients.mock import MockClient
from mimir.runner import (
    TokenBucket,
    parse_pairwise_verdict,
    parse_rubric_score,
    run_experiment,
)
from mimir.spec import ExperimentSpec
from mimir.store import Store


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "mimir.db")
    yield s
    s.close()


def make_spec(
    *,
    name="greeting-tone",
    variants=None,
    items=None,
    n_samples=1,
    judge=None,
    limits=None,
):
    data = {
        "name": name,
        "variants": variants
        or [
            {
                "name": "control",
                "system": "You are a helpful assistant.",
                "user_template": "Answer: {input}",
            },
            {
                "name": "friendly",
                "system": "You are warm.",
                "user_template": "Answer: {input}",
            },
        ],
        "dataset": {
            "items": items
            or [
                {"id": "q1", "input": "Why is the sky blue?"},
                {"id": "q2", "input": "Why is grass green?"},
            ]
        },
        "sampling": {"model": "claude-haiku-4-5-20251001"},
        "n_samples": n_samples,
        "limits": limits or {"concurrency": 4, "requests_per_minute": 100_000},
    }
    if judge is not None:
        data["judge"] = judge
    return ExperimentSpec.model_validate(data)


def condition_names(store, run_id):
    return {row["id"]: row["variant_name"] for row in store.get_conditions(run_id)}


# --- pure logic: verdict parsing -------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("A", "A"),
        ("b", "B"),
        ("TIE", "TIE"),
        ("tie", "TIE"),
        ("Reasoning about both...\nB", "B"),
        ("verdict follows\n\n  a  \n", "A"),
    ],
)
def test_parse_pairwise_verdict_accepts_final_line(text, expected):
    assert parse_pairwise_verdict(text) == expected


@pytest.mark.parametrize("text", ["", "  \n ", "MAYBE", "A is better", "Response A\nAB"])
def test_parse_pairwise_verdict_rejects_garbage(text):
    with pytest.raises(ValueError, match="verdict"):
        parse_pairwise_verdict(text)


@pytest.mark.parametrize(
    ("text", "expected"),
    [("7", 7.0), ("1", 1.0), ("10", 10.0), ("Reasoning...\n9", 9.0), ("  8  \n", 8.0)],
)
def test_parse_rubric_score_accepts_final_line_integer(text, expected):
    assert parse_rubric_score(text) == expected


@pytest.mark.parametrize("text", ["", "0", "11", "7.5", "seven", "8/10"])
def test_parse_rubric_score_rejects_out_of_range_or_non_integer(text):
    with pytest.raises(ValueError, match="score"):
        parse_rubric_score(text)


# --- pure logic: token bucket -----------------------------------------------------


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


@pytest.mark.anyio
async def test_token_bucket_allows_burst_then_paces():
    clock = FakeClock()
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)
        clock.advance(seconds)

    bucket = TokenBucket(60, capacity=2, clock=clock.now, sleep=fake_sleep)  # 1 token/s
    await bucket.acquire()
    await bucket.acquire()
    assert sleeps == []
    await bucket.acquire()
    assert sum(sleeps) == pytest.approx(1.0, rel=0.01)


@pytest.mark.anyio
async def test_token_bucket_refills_while_idle():
    clock = FakeClock()
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)
        clock.advance(seconds)

    bucket = TokenBucket(60, capacity=1, clock=clock.now, sleep=fake_sleep)
    await bucket.acquire()
    clock.advance(5.0)  # idle: bucket refills (capped at capacity)
    await bucket.acquire()
    assert sleeps == []
    await bucket.acquire()  # only ONE token was banked while idle: must pace ~1s
    assert sum(sleeps) == pytest.approx(1.0, rel=0.01)


@pytest.mark.anyio
async def test_token_bucket_sustained_pacing_after_burst():
    clock = FakeClock()
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)
        clock.advance(seconds)

    bucket = TokenBucket(60, capacity=2, clock=clock.now, sleep=fake_sleep)  # 1 token/s
    await bucket.acquire()
    await bucket.acquire()
    assert sleeps == []
    for _ in range(3):
        await bucket.acquire()
    assert len(sleeps) == 3
    assert all(pause == pytest.approx(1.0, rel=0.01) for pause in sleeps)


# --- runner integration (MockClient + tmp store) ----------------------------------


@pytest.mark.anyio
async def test_basic_run_stores_all_samples(store):
    spec = make_spec()
    client = MockClient()
    run_id = await run_experiment(spec, store, client)

    assert store.get_run(run_id)["status"] == "complete"
    samples = store.get_samples(run_id)
    assert len(samples) == 4  # 2 variants x 2 items x 1 sample
    assert len(client.calls) == 4
    for row in samples:
        assert row["cache_hit"] == 0
        assert row["error"] is None
        assert row["response_text"].startswith("mock:")
        assert row["latency_ms"] > 0
        assert row["input_tokens"] > 0
        assert row["output_tokens"] > 0
        raw = json.loads(row["raw_response"])
        assert raw["content"][0]["text"] == row["response_text"]
        assert raw["model"] == "claude-haiku-4-5-20251001"
        payload = json.loads(row["request_json"])
        assert cache_key(payload) == row["cache_key"]
    # Each sample's rendered prompt must correspond to ITS item, and all four
    # (variant x item) payloads must be pairwise distinct.
    assert len({row["cache_key"] for row in samples}) == 4
    input_of = {"q1": "Why is the sky blue?", "q2": "Why is grass green?"}
    for row in samples:
        content = json.loads(row["request_json"])["messages"][0]["content"]
        assert content == f"Answer: {input_of[row['item_id']]}"


@pytest.mark.anyio
async def test_rerun_hits_cache_and_skips_client_calls(store):
    spec = make_spec()
    client = MockClient()
    first_run = await run_experiment(spec, store, client)
    first_by_key = {row["cache_key"]: row for row in store.get_samples(first_run)}
    client.calls.clear()

    second_run = await run_experiment(spec, store, client)
    assert client.calls == []
    samples = store.get_samples(second_run)
    assert len(samples) == 4
    for row in samples:
        assert row["cache_hit"] == 1
        original = first_by_key[row["cache_key"]]
        assert row["response_text"] == original["response_text"]
        assert row["latency_ms"] == original["latency_ms"]
        assert row["input_tokens"] == original["input_tokens"]
        assert row["output_tokens"] == original["output_tokens"]


@pytest.mark.anyio
async def test_rerun_with_one_changed_variant_executes_only_the_delta(store):
    client = MockClient()
    await run_experiment(make_spec(), store, client)
    client.calls.clear()

    changed = make_spec(
        variants=[
            {
                "name": "control",
                "system": "You are a helpful assistant.",
                "user_template": "Answer: {input}",
            },
            {
                "name": "friendly",
                "system": "You are extremely warm and effusive.",  # changed content
                "user_template": "Answer: {input}",
            },
        ]
    )
    run_id = await run_experiment(changed, store, client)

    assert len(client.calls) == 2  # only the changed variant's 2 items
    assert all(request.system == "You are extremely warm and effusive." for request in client.calls)
    names = condition_names(store, run_id)
    for row in store.get_samples(run_id):
        expected_hit = 1 if names[row["condition_id"]] == "control" else 0
        assert row["cache_hit"] == expected_hit


@pytest.mark.anyio
async def test_labels_and_limits_are_not_in_the_cache_key(store):
    client = MockClient()
    await run_experiment(make_spec(), store, client)
    client.calls.clear()

    relabeled = make_spec(
        name="renamed-experiment",
        variants=[
            {
                "name": "baseline",  # renamed label, same content
                "system": "You are a helpful assistant.",
                "user_template": "Answer: {input}",
            },
            {
                "name": "cheerful",  # renamed label, same content
                "system": "You are warm.",
                "user_template": "Answer: {input}",
            },
        ],
        limits={"concurrency": 9, "requests_per_minute": 55_000},
    )
    run_id = await run_experiment(relabeled, store, client)
    assert client.calls == []
    assert all(row["cache_hit"] == 1 for row in store.get_samples(run_id))


@pytest.mark.anyio
async def test_concurrency_capped_by_semaphore(store):
    items = [{"id": f"q{i}", "input": f"question {i}"} for i in range(4)]
    spec = make_spec(items=items, limits={"concurrency": 2, "requests_per_minute": 100_000})
    client = MockClient(latency_s=0.005)
    await run_experiment(spec, store, client)
    assert len(client.calls) == 8
    assert client.max_in_flight == 2


@pytest.mark.anyio
async def test_retryable_error_backs_off_then_succeeds(store, monkeypatch):
    monkeypatch.setattr(runner_mod, "_BASE_DELAY_S", 0.001)
    monkeypatch.setattr(runner_mod, "_MAX_DELAY_S", 0.002)
    spec = make_spec(items=[{"id": "q1", "input": "only one"}])
    client = MockClient()
    client.queue_error(429)
    client.queue_error(503)
    run_id = await run_experiment(spec, store, client)

    assert store.get_run(run_id)["status"] == "complete"
    assert len(client.calls) == 4  # 2 units + 2 retried attempts
    assert all(row["error"] is None for row in store.get_samples(run_id))


@pytest.mark.anyio
async def test_exhausted_retries_record_error_row_and_fail_run(store, monkeypatch):
    monkeypatch.setattr(runner_mod, "_BASE_DELAY_S", 0.001)
    monkeypatch.setattr(runner_mod, "_MAX_DELAY_S", 0.002)
    spec = make_spec(
        variants=[{"name": "control", "system": "s", "user_template": "Answer: {input}"}],
        items=[{"id": "q1", "input": "only one"}],
    )
    client = MockClient()
    client.queue_error(429, times=runner_mod._MAX_ATTEMPTS)
    run_id = await run_experiment(spec, store, client)

    assert len(client.calls) == runner_mod._MAX_ATTEMPTS
    assert store.get_run(run_id)["status"] == "failed"
    (row,) = store.get_samples(run_id)
    assert row["error"] is not None
    assert "429" in row["error"]
    assert row["raw_response"] is None


@pytest.mark.anyio
async def test_non_retryable_error_fails_immediately(store):
    spec = make_spec(
        variants=[{"name": "control", "system": "s", "user_template": "Answer: {input}"}],
        items=[{"id": "q1", "input": "only one"}],
    )
    client = MockClient()
    client.queue_error(400)
    run_id = await run_experiment(spec, store, client)

    assert len(client.calls) == 1  # no retry on 4xx other than 429
    assert store.get_run(run_id)["status"] == "failed"
    (row,) = store.get_samples(run_id)
    assert "400" in row["error"]


@pytest.mark.anyio
async def test_pairwise_judge_scores_both_orders_and_caches(store):
    spec = make_spec(judge={"model": "judge-model", "mode": "pairwise"})
    client = MockClient()
    client.add_rule(lambda request: request.model == "judge-model", "A")
    run_id = await run_experiment(spec, store, client)

    judgments = store.get_judgments(run_id)
    assert len(judgments) == 4  # 2 items x 1 replicate x 2 orders
    assert {(j["item_id"], j["position_order"]) for j in judgments} == {
        ("q1", "ab"),
        ("q1", "ba"),
        ("q2", "ab"),
        ("q2", "ba"),
    }
    assert all(j["verdict"] == "A" for j in judgments)
    assert all(j["mode"] == "pairwise" for j in judgments)
    assert all(j["error"] is None for j in judgments)

    # sample_a_id is the sample PRESENTED in position A: variant[0] for "ab",
    # variant[1] for "ba".
    names = condition_names(store, run_id)
    sample_variant = {row["id"]: names[row["condition_id"]] for row in store.get_samples(run_id)}
    for judgment in judgments:
        presented_first = sample_variant[judgment["sample_a_id"]]
        expected = "control" if judgment["position_order"] == "ab" else "friendly"
        assert presented_first == expected

    # The rendered judge PROMPT must present sample_a_id's text as {response_a}
    # and sample_b_id's as {response_b}; the stored cache_key hashes that prompt,
    # so recomputing it per row proves the swap is real, not just relabeled ids.
    text_by_id = {row["id"]: row["response_text"] for row in store.get_samples(run_id)}
    input_of = {"q1": "Why is the sky blue?", "q2": "Why is grass green?"}
    template = spec.judge.resolved_prompt_template()
    for judgment in judgments:
        expected_prompt = template.format(
            input=input_of[judgment["item_id"]],
            response_a=text_by_id[judgment["sample_a_id"]],
            response_b=text_by_id[judgment["sample_b_id"]],
        )
        expected_key = cache_key(
            build_payload(
                model="judge-model",
                system="",
                user=expected_prompt,
                temperature=0.0,
                max_tokens=512,
                seed=0,
                sample_index=0,
            )
        )
        assert judgment["cache_key"] == expected_key

    # Judge calls are cached too: a full re-run issues zero client calls but
    # still records fresh judgment rows for the new run.
    client.calls.clear()
    second_run = await run_experiment(
        make_spec(judge={"model": "judge-model", "mode": "pairwise"}), store, client
    )
    assert client.calls == []
    assert len(store.get_judgments(second_run)) == 4


@pytest.mark.anyio
async def test_position_swap_false_scores_single_order(store):
    spec = make_spec(judge={"model": "judge-model", "mode": "pairwise", "position_swap": False})
    client = MockClient()
    client.add_rule(lambda request: request.model == "judge-model", "TIE")
    run_id = await run_experiment(spec, store, client)
    judgments = store.get_judgments(run_id)
    assert len(judgments) == 2  # 2 items x 1 replicate x 1 order
    assert all(j["position_order"] == "ab" for j in judgments)


@pytest.mark.anyio
async def test_rubric_judge_scores_each_sample(store):
    spec = make_spec(judge={"model": "judge-model", "mode": "rubric"})
    client = MockClient()
    client.add_rule(lambda request: request.model == "judge-model", "8")
    run_id = await run_experiment(spec, store, client)
    judgments = store.get_judgments(run_id)
    assert len(judgments) == 4  # one per sample: 2 variants x 2 items x 1 replicate
    for judgment in judgments:
        assert judgment["score"] == 8.0
        assert judgment["verdict"] is None
        assert judgment["sample_b_id"] is None
        assert judgment["position_order"] is None


@pytest.mark.anyio
async def test_judge_parse_failure_records_error_and_fails_run(store):
    spec = make_spec(judge={"model": "judge-model", "mode": "pairwise"})
    client = MockClient()
    client.add_rule(lambda request: request.model == "judge-model", "MAYBE")
    run_id = await run_experiment(spec, store, client)

    judgments = store.get_judgments(run_id)
    assert len(judgments) == 4
    assert all(j["verdict"] is None for j in judgments)
    assert all(j["error"] is not None for j in judgments)
    assert all(j["raw_response"] is not None for j in judgments)  # raw kept for audit
    assert store.get_run(run_id)["status"] == "failed"


@pytest.mark.anyio
async def test_replicates_get_distinct_cache_keys(store):
    spec = make_spec(n_samples=2)
    client = MockClient()
    run_id = await run_experiment(spec, store, client)
    samples = store.get_samples(run_id)
    assert len(samples) == 8  # 2 variants x 2 items x 2 replicates
    assert len(client.calls) == 8
    assert len({row["cache_key"] for row in samples}) == 8
    for row in samples:
        assert json.loads(row["request_json"])["sample_index"] == row["sample_index"]


@pytest.mark.anyio
async def test_raising_n_samples_reuses_replicates_and_executes_only_the_delta(store):
    client = MockClient()
    await run_experiment(make_spec(n_samples=3), store, client)
    client.calls.clear()

    run_id = await run_experiment(make_spec(n_samples=5), store, client)
    # Replicates 0-2 are cache hits; only 3-4 execute (4 variant-item units each).
    assert sorted(request.sample_index for request in client.calls) == [3] * 4 + [4] * 4
    for row in store.get_samples(run_id):
        assert row["cache_hit"] == (1 if row["sample_index"] < 3 else 0)


@pytest.mark.anyio
async def test_pairwise_judge_pairs_replicate_i_with_replicate_i(store):
    spec = make_spec(n_samples=2, judge={"model": "judge-model", "mode": "pairwise"})
    client = MockClient()
    client.add_rule(lambda request: request.model == "judge-model", "A")
    run_id = await run_experiment(spec, store, client)

    judgments = store.get_judgments(run_id)
    assert len(judgments) == 8  # 2 items x 2 replicates x 2 orders (DESIGN §5)
    index_of = {row["id"]: row["sample_index"] for row in store.get_samples(run_id)}
    for judgment in judgments:
        assert index_of[judgment["sample_a_id"]] == index_of[judgment["sample_b_id"]]
    assert {(j["item_id"], index_of[j["sample_a_id"]], j["position_order"]) for j in judgments} == {
        (i, n, o) for i in ("q1", "q2") for n in (0, 1) for o in ("ab", "ba")
    }


@pytest.mark.anyio
async def test_rubric_judge_with_three_variants_scores_every_sample(store):
    spec = make_spec(
        variants=[
            {
                "name": "control",
                "system": "You are a helpful assistant.",
                "user_template": "Answer: {input}",
            },
            {"name": "friendly", "system": "You are warm.", "user_template": "Answer: {input}"},
            {"name": "terse", "system": "You are terse.", "user_template": "Answer: {input}"},
        ],
        judge={"model": "judge-model", "mode": "rubric"},
    )
    client = MockClient()
    client.add_rule(lambda request: request.model == "judge-model", "8")
    run_id = await run_experiment(spec, store, client)

    assert store.get_run(run_id)["status"] == "complete"
    assert len(store.get_samples(run_id)) == 6  # 3 variants x 2 items
    judgments = store.get_judgments(run_id)
    assert len(judgments) == 6  # one per sample
    assert all(j["score"] == 8.0 for j in judgments)


@pytest.mark.anyio
async def test_backoff_delays_follow_exponential_schedule_with_jitter_and_cap(store, monkeypatch):
    recorded = []

    async def record_sleep(seconds):
        recorded.append(seconds)

    monkeypatch.setattr(runner_mod, "_sleep", record_sleep)
    monkeypatch.setattr(runner_mod, "_MAX_DELAY_S", 3.0)  # bring the cap into range
    spec = make_spec(
        variants=[{"name": "control", "system": "s", "user_template": "Answer: {input}"}],
        items=[{"id": "q1", "input": "only one"}],
    )
    client = MockClient()
    client.queue_error(429, times=4)  # attempts 0-3 fail, attempt 4 succeeds
    run_id = await run_experiment(spec, store, client)

    assert store.get_run(run_id)["status"] == "complete"
    assert len(client.calls) == 5
    assert len(recorded) == 4  # one backoff sleep per failed attempt
    for delay, ceiling in zip(recorded, [1.0, 2.0, 3.0, 3.0], strict=True):
        assert 0.5 * ceiling <= delay <= ceiling  # min(cap, base*2^n) * jitter(0.5..1.0)


@pytest.mark.anyio
async def test_runner_acquires_one_token_per_request_attempt(store, monkeypatch):
    acquires = 0

    class CountingBucket(TokenBucket):
        async def acquire(self):
            nonlocal acquires
            acquires += 1
            await super().acquire()

    monkeypatch.setattr(runner_mod, "TokenBucket", CountingBucket)
    client = MockClient()
    await run_experiment(make_spec(), store, client)
    assert acquires == len(client.calls) == 4


@pytest.mark.anyio
async def test_judge_client_error_records_judgment_error_row(store):
    client = MockClient()
    await run_experiment(make_spec(), store, client)  # warm the completion cache
    client.calls.clear()

    client.queue_error(400)  # completions all hit cache: the 400 lands on a judge call
    client.add_rule(lambda request: request.model == "judge-model", "A")
    run_id = await run_experiment(
        make_spec(judge={"model": "judge-model", "mode": "pairwise"}), store, client
    )

    assert store.get_run(run_id)["status"] == "failed"
    assert all(request.model == "judge-model" for request in client.calls)
    judgments = store.get_judgments(run_id)
    assert len(judgments) == 4  # the errored unit still writes its row
    errored = [j for j in judgments if j["error"] is not None]
    assert len(errored) == 1
    assert "400" in errored[0]["error"]
    assert errored[0]["raw_response"] is None
    assert errored[0]["verdict"] is None


class ExplodingClient:
    async def complete(self, request):
        raise RuntimeError("boom")


@pytest.mark.anyio
async def test_unexpected_client_exception_records_error_and_fails_run(store):
    spec = make_spec(items=[{"id": "q1", "input": "only one"}])
    run_id = await run_experiment(spec, store, ExplodingClient())

    assert store.get_run(run_id)["status"] == "failed"
    samples = store.get_samples(run_id)
    assert len(samples) == 2
    for row in samples:
        assert "RuntimeError" in row["error"]
        assert row["cache_key"] != ""  # failure happened after key computation


@pytest.mark.anyio
async def test_corrupted_cache_envelope_records_error_row_not_crash(store):
    spec = make_spec(
        variants=[{"name": "control", "system": "s", "user_template": "Answer: {input}"}],
        items=[{"id": "q1", "input": "only one"}],
    )
    payload = build_payload(
        model="claude-haiku-4-5-20251001",
        system="s",
        user="Answer: only one",
        temperature=1.0,
        max_tokens=1024,
        seed=0,
        sample_index=0,
    )
    store.cache_put(cache_key(payload), {"text": "hi"})  # incomplete envelope
    client = MockClient()
    run_id = await run_experiment(spec, store, client)

    assert client.calls == []  # the (corrupted) cache hit still skipped the client
    assert store.get_run(run_id)["status"] == "failed"
    (row,) = store.get_samples(run_id)
    assert "KeyError" in row["error"]


@pytest.mark.anyio
async def test_judge_skips_pairs_with_failed_samples(store):
    spec = make_spec(
        items=[{"id": "q1", "input": "only one"}],
        judge={"model": "judge-model", "mode": "pairwise"},
    )
    client = MockClient()
    client.queue_error(400)  # first completion call fails, non-retryable
    run_id = await run_experiment(spec, store, client)

    assert store.get_run(run_id)["status"] == "failed"
    errored = [row for row in store.get_samples(run_id) if row["error"] is not None]
    assert len(errored) == 1
    assert store.get_judgments(run_id) == []  # incomplete pair is skipped, not crashed
