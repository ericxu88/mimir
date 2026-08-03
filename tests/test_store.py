"""Tests for mimir.store — SQLite results store (docs/DESIGN.md §4).

Append-only: result rows are inserted, never updated or deleted; the sole sanctioned
mutation is the runs.status transition running -> complete | failed.
"""

import re
import sqlite3
from datetime import datetime
from hashlib import sha256

import pytest

import mimir.store as store_mod
from mimir.cache import canonical_json
from mimir.store import _RUN_ID_ATTEMPTS, Store

RUN_ID_RE = r"\d{8}-\d{6}-[0-9a-f]{4}"
SPEC = {"name": "greeting-tone", "n_samples": 2}


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "mimir.db")
    yield s
    s.close()


def make_run_with_condition(s):
    run_id = s.create_run("greeting-tone", SPEC)
    condition_id = s.add_condition(
        run_id,
        variant_name="control",
        system_prompt="You are a helpful assistant.",
        user_template="Answer: {input}",
        sampling={"model": "claude-haiku-4-5-20251001", "temperature": 1.0},
    )
    return run_id, condition_id


def test_schema_init_is_idempotent(tmp_path):
    path = tmp_path / "mimir.db"
    with Store(path) as first:
        run_id = first.create_run("exp", SPEC)
    with Store(path) as second:
        assert second.get_run(run_id) is not None


def test_two_open_stores_see_committed_writes(tmp_path):
    path = tmp_path / "mimir.db"
    first = Store(path)
    second = Store(path)
    try:
        run_id = first.create_run("exp", SPEC)
        assert second.get_run(run_id) is not None
    finally:
        first.close()
        second.close()


def test_run_ids_unique_in_fast_loop(store):
    ids = {store.create_run("exp", SPEC) for _ in range(50)}
    assert len(ids) == 50


def test_context_manager_closes_connection(tmp_path):
    with Store(tmp_path / "mimir.db") as s:
        pass
    with pytest.raises(sqlite3.ProgrammingError):
        s.get_run("anything")


def test_create_run_get_run_roundtrip(store):
    run_id = store.create_run("greeting-tone", SPEC)
    assert re.fullmatch(RUN_ID_RE, run_id)
    run = store.get_run(run_id)
    assert run["id"] == run_id
    assert run["experiment_name"] == "greeting-tone"
    assert run["status"] == "running"
    expected_json = canonical_json(SPEC).decode("utf-8")
    assert run["spec_json"] == expected_json
    assert run["spec_hash"] == sha256(expected_json.encode("utf-8")).hexdigest()
    assert datetime.fromisoformat(run["created_at"]).tzinfo is not None


def test_get_run_missing_returns_none(store):
    assert store.get_run("20990101-000000-dead") is None


def test_run_status_running_to_complete(store):
    run_id = store.create_run("exp", SPEC)
    store.set_run_status(run_id, "complete")
    assert store.get_run(run_id)["status"] == "complete"


def test_run_status_running_to_failed(store):
    run_id = store.create_run("exp", SPEC)
    store.set_run_status(run_id, "failed")
    assert store.get_run(run_id)["status"] == "failed"


@pytest.mark.parametrize(
    ("setup_status", "target"),
    [
        ("complete", "running"),
        ("complete", "failed"),
        ("failed", "complete"),
        (None, "running"),
        (None, "finished"),
    ],
)
def test_run_status_illegal_transitions_rejected(store, setup_status, target):
    run_id = store.create_run("exp", SPEC)
    if setup_status is not None:
        store.set_run_status(run_id, setup_status)
    with pytest.raises(ValueError, match="status"):
        store.set_run_status(run_id, target)
    assert store.get_run(run_id)["status"] == (setup_status or "running")


def test_set_status_on_missing_run_raises(store):
    with pytest.raises(ValueError, match="status"):
        store.set_run_status("20990101-000000-dead", "complete")


def test_add_condition_roundtrip(store):
    run_id, condition_id = make_run_with_condition(store)
    (row,) = store.get_conditions(run_id)
    assert row["id"] == condition_id
    assert row["run_id"] == run_id
    assert row["variant_name"] == "control"
    assert row["system_prompt"] == "You are a helpful assistant."
    assert row["user_template"] == "Answer: {input}"
    assert row["sampling_json"] == canonical_json(
        {"model": "claude-haiku-4-5-20251001", "temperature": 1.0}
    ).decode("utf-8")


def test_duplicate_variant_name_in_run_rejected(store):
    run_id, _ = make_run_with_condition(store)
    with pytest.raises(sqlite3.IntegrityError):
        store.add_condition(
            run_id,
            variant_name="control",
            system_prompt="different",
            user_template="different: {input}",
            sampling={},
        )


def test_add_sample_roundtrip_preserves_mandated_fields(store):
    run_id, condition_id = make_run_with_condition(store)
    key = "ab" * 32
    raw = '{"content": "世界 \U0001f989", "note": "line\\nbreak"}'
    sample_id = store.add_sample(
        run_id=run_id,
        condition_id=condition_id,
        item_id="q1",
        sample_index=0,
        cache_key=key,
        request_json='{"model":"m"}',
        raw_response=raw,
        response_text="世界",
        latency_ms=123.456,
        input_tokens=17,
        output_tokens=42,
        cache_hit=True,
    )
    (row,) = store.get_samples(run_id)
    assert row["id"] == sample_id
    assert row["run_id"] == run_id
    assert row["condition_id"] == condition_id
    assert row["item_id"] == "q1"
    assert row["sample_index"] == 0
    assert row["cache_key"] == key
    assert row["request_json"] == '{"model":"m"}'
    assert row["raw_response"] == raw
    assert row["response_text"] == "世界"
    assert row["latency_ms"] == 123.456
    assert row["input_tokens"] == 17
    assert row["output_tokens"] == 42
    assert row["cache_hit"] == 1
    assert row["error"] is None
    assert datetime.fromisoformat(row["created_at"]).tzinfo is not None


def test_add_sample_error_row_allows_null_response(store):
    run_id, condition_id = make_run_with_condition(store)
    store.add_sample(
        run_id=run_id,
        condition_id=condition_id,
        item_id="q1",
        sample_index=0,
        cache_key="cd" * 32,
        request_json="{}",
        error="timeout after 5 retries",
    )
    (row,) = store.get_samples(run_id)
    assert row["raw_response"] is None
    assert row["response_text"] is None
    assert row["latency_ms"] is None
    assert row["input_tokens"] is None
    assert row["output_tokens"] is None
    assert row["cache_hit"] == 0
    assert row["error"] == "timeout after 5 retries"


def test_get_samples_returns_insertion_order(store):
    run_id, condition_id = make_run_with_condition(store)
    for index in range(3):
        store.add_sample(
            run_id=run_id,
            condition_id=condition_id,
            item_id="q1",
            sample_index=index,
            cache_key=f"{index:064d}",
            request_json="{}",
        )
    rows = store.get_samples(run_id)
    assert [r["sample_index"] for r in rows] == [0, 1, 2]


def test_duplicate_sample_coordinates_rejected(store):
    run_id, condition_id = make_run_with_condition(store)
    kwargs = {
        "run_id": run_id,
        "condition_id": condition_id,
        "item_id": "q1",
        "sample_index": 0,
        "cache_key": "ef" * 32,
        "request_json": "{}",
    }
    store.add_sample(**kwargs)
    with pytest.raises(sqlite3.IntegrityError):
        store.add_sample(**kwargs)


def test_store_public_surface_is_pinned_to_append_only_allowlist(store):
    # Append-only is enforced at the API surface (no DB triggers, by decision);
    # any new public method must be consciously added here and reviewed against
    # DESIGN.md §4 before it ships.
    expected = {
        "add_condition",
        "add_judgment",
        "add_sample",
        "cache_get",
        "cache_put",
        "close",
        "create_run",
        "get_conditions",
        "get_judgments",
        "get_run",
        "get_samples",
        "set_run_status",
    }
    assert {a for a in dir(store) if not a.startswith("_")} == expected


def test_cache_put_get_roundtrip(store):
    envelope = {
        "text": "héllo 世界",
        "raw": {"content": [{"type": "text", "text": "héllo 世界"}], "ok": True},
        "input_tokens": 12,
        "output_tokens": 34,
        "latency_ms": 456.7,
        "model": "claude-haiku-4-5-20251001",
    }
    key = "12" * 32
    store.cache_put(key, envelope)
    assert store.cache_get(key) == envelope


def test_cache_get_missing_key_returns_none(store):
    assert store.cache_get("0" * 64) is None


def test_cache_put_duplicate_key_is_first_write_wins(store):
    key = "34" * 32
    store.cache_put(key, {"text": "first"})
    store.cache_put(key, {"text": "second"})
    assert store.cache_get(key) == {"text": "first"}


def test_same_item_and_index_allowed_across_conditions(store):
    # M2 pairwise shape: the same (item_id, sample_index) exists under BOTH
    # variants' conditions within one run.
    run_id, condition_a = make_run_with_condition(store)
    condition_b = store.add_condition(
        run_id,
        variant_name="friendly",
        system_prompt="You are a warm, encouraging assistant.",
        user_template="Answer: {input}",
        sampling={"model": "claude-haiku-4-5-20251001", "temperature": 1.0},
    )
    for condition_id in (condition_a, condition_b):
        store.add_sample(
            run_id=run_id,
            condition_id=condition_id,
            item_id="q1",
            sample_index=0,
            cache_key="ab" * 32,
            request_json="{}",
        )
    assert len(store.get_samples(run_id)) == 2


def test_same_variant_name_allowed_across_runs(store):
    run_a, condition_a = make_run_with_condition(store)
    run_b, condition_b = make_run_with_condition(store)
    assert run_a != run_b
    assert condition_a != condition_b


def test_get_conditions_returns_insertion_order(store):
    run_id = store.create_run("exp", SPEC)
    for name in ("zeta", "alpha"):  # non-alphabetical: index order would be alpha, zeta
        store.add_condition(
            run_id, variant_name=name, system_prompt="", user_template="t", sampling={}
        )
    assert [r["variant_name"] for r in store.get_conditions(run_id)] == ["zeta", "alpha"]


def test_getters_unknown_run_id_return_empty(store):
    assert store.get_conditions("20990101-000000-dead") == []
    assert store.get_samples("20990101-000000-dead") == []


def test_create_run_retries_collision_then_succeeds(store, monkeypatch):
    taken = store.create_run("exp", SPEC)
    ids = iter([taken, "20260101-000000-beef"])
    monkeypatch.setattr(store_mod, "_new_run_id", lambda: next(ids))
    assert store.create_run("exp", SPEC) == "20260101-000000-beef"


def test_create_run_raises_after_exhausting_attempts(store, monkeypatch):
    taken = store.create_run("exp", SPEC)
    calls = 0

    def stuck():
        nonlocal calls
        calls += 1
        return taken

    monkeypatch.setattr(store_mod, "_new_run_id", stuck)
    with pytest.raises(sqlite3.IntegrityError):
        store.create_run("exp", SPEC)
    assert calls == _RUN_ID_ATTEMPTS


def test_cache_put_survives_lone_surrogate_text(store):
    # json.loads accepts "\ud800" escapes from provider payloads; the store must
    # persist (sanitized) rather than crash on UTF-8 encoding.
    key = "9a" * 32
    store.cache_put(key, {"text": "pre\ud800post"})
    assert store.cache_get(key) == {"text": "pre?post"}


def test_add_sample_survives_lone_surrogate_response(store):
    run_id, condition_id = make_run_with_condition(store)
    store.add_sample(
        run_id=run_id,
        condition_id=condition_id,
        item_id="q1",
        sample_index=0,
        cache_key="bc" * 32,
        request_json="{}",
        raw_response="pre\ud800post",
    )
    (row,) = store.get_samples(run_id)
    assert row["raw_response"] == "pre?post"


def make_sample(s, run_id, condition_id, *, item_id="q1", sample_index=0, key="ab" * 32):
    return s.add_sample(
        run_id=run_id,
        condition_id=condition_id,
        item_id=item_id,
        sample_index=sample_index,
        cache_key=key,
        request_json="{}",
        response_text="text",
    )


def test_add_judgment_roundtrip_pairwise(store):
    run_id, condition_id = make_run_with_condition(store)
    sample_a = make_sample(store, run_id, condition_id, sample_index=0)
    sample_b = make_sample(store, run_id, condition_id, sample_index=1)
    judgment_id = store.add_judgment(
        run_id=run_id,
        item_id="q1",
        judge_model="judge-model",
        mode="pairwise",
        sample_a_id=sample_a,
        sample_b_id=sample_b,
        position_order="ab",
        cache_key="fe" * 32,
        raw_response='{"content": "A"}',
        verdict="A",
        latency_ms=9.5,
        input_tokens=50,
        output_tokens=1,
    )
    (row,) = store.get_judgments(run_id)
    assert row["id"] == judgment_id
    assert row["run_id"] == run_id
    assert row["item_id"] == "q1"
    assert row["judge_model"] == "judge-model"
    assert row["mode"] == "pairwise"
    assert row["sample_a_id"] == sample_a
    assert row["sample_b_id"] == sample_b
    assert row["position_order"] == "ab"
    assert row["cache_key"] == "fe" * 32
    assert row["raw_response"] == '{"content": "A"}'
    assert row["verdict"] == "A"
    assert row["score"] is None
    assert row["latency_ms"] == 9.5
    assert row["input_tokens"] == 50
    assert row["output_tokens"] == 1
    assert row["error"] is None


def test_add_judgment_rubric_allows_null_pair_fields(store):
    run_id, condition_id = make_run_with_condition(store)
    sample_a = make_sample(store, run_id, condition_id)
    store.add_judgment(
        run_id=run_id,
        item_id="q1",
        judge_model="judge-model",
        mode="rubric",
        sample_a_id=sample_a,
        cache_key="fd" * 32,
        raw_response="8",
        score=8.0,
    )
    (row,) = store.get_judgments(run_id)
    assert row["sample_b_id"] is None
    assert row["position_order"] is None
    assert row["verdict"] is None
    assert row["score"] == 8.0


def test_judgment_with_bogus_run_id_rejected(store):
    run_id, condition_id = make_run_with_condition(store)
    sample_a = make_sample(store, run_id, condition_id)
    with pytest.raises(sqlite3.IntegrityError):
        store.add_judgment(
            run_id="20990101-000000-dead",
            item_id="q1",
            judge_model="j",
            mode="pairwise",
            sample_a_id=sample_a,
            cache_key="fc" * 32,
        )


def test_judgment_with_bogus_sample_id_rejected(store):
    run_id, _ = make_run_with_condition(store)
    with pytest.raises(sqlite3.IntegrityError):
        store.add_judgment(
            run_id=run_id,
            item_id="q1",
            judge_model="j",
            mode="pairwise",
            sample_a_id=999999,
            cache_key="fb" * 32,
        )


def test_sample_with_condition_from_other_run_rejected(store):
    # The composite FK ties a sample's condition to the sample's own run.
    run_a = store.create_run("exp-a", SPEC)
    run_b, condition_b = make_run_with_condition(store)
    assert run_a != run_b
    with pytest.raises(sqlite3.IntegrityError):
        store.add_sample(
            run_id=run_a,
            condition_id=condition_b,
            item_id="q1",
            sample_index=0,
            cache_key="de" * 32,
            request_json="{}",
        )


def test_sample_with_bogus_condition_id_rejected(store):
    run_id, _ = make_run_with_condition(store)
    with pytest.raises(sqlite3.IntegrityError):
        store.add_sample(
            run_id=run_id,
            condition_id=999999,
            item_id="q1",
            sample_index=0,
            cache_key="78" * 32,
            request_json="{}",
        )


def test_sample_with_bogus_run_id_rejected(store):
    _, condition_id = make_run_with_condition(store)
    with pytest.raises(sqlite3.IntegrityError):
        store.add_sample(
            run_id="20990101-000000-dead",
            condition_id=condition_id,
            item_id="q1",
            sample_index=0,
            cache_key="56" * 32,
            request_json="{}",
        )


def test_condition_with_bogus_run_id_rejected(store):
    with pytest.raises(sqlite3.IntegrityError):
        store.add_condition(
            "20990101-000000-dead",
            variant_name="control",
            system_prompt="",
            user_template="t",
            sampling={},
        )


def test_new_databases_reject_cross_run_judgments(store):
    # Run-scoped judgment FKs (M9): on databases created from this schema, a
    # judgment's samples must belong to the judgment's own run. Pre-M9 files keep
    # the old DDL (CREATE IF NOT EXISTS never alters), which is why the
    # skip-and-count drifted-row guards in stats/judge_audit stay load-bearing.
    run_a, cond_a = make_run_with_condition(store)
    run_b, cond_b = make_run_with_condition(store)
    sid_b = store.add_sample(
        run_id=run_b,
        condition_id=cond_b,
        item_id="q1",
        sample_index=0,
        cache_key="k" * 64,
        request_json="{}",
    )
    with pytest.raises(sqlite3.IntegrityError):
        store.add_judgment(
            run_id=run_a,
            item_id="q1",
            judge_model="judge-model",
            mode="rubric",
            sample_a_id=sid_b,
            cache_key="j" * 64,
        )
    sid_a = store.add_sample(
        run_id=run_a,
        condition_id=cond_a,
        item_id="q1",
        sample_index=0,
        cache_key="k" * 64,
        request_json="{}",
    )
    with pytest.raises(sqlite3.IntegrityError):
        store.add_judgment(
            run_id=run_a,
            item_id="q1",
            judge_model="judge-model",
            mode="pairwise",
            sample_a_id=sid_a,
            sample_b_id=sid_b,
            position_order="ab",
            cache_key="j" * 64,
        )
    # Same-run judgments (and NULL sample_b_id) still insert.
    assert store.add_judgment(
        run_id=run_a,
        item_id="q1",
        judge_model="judge-model",
        mode="rubric",
        sample_a_id=sid_a,
        cache_key="j" * 64,
    )
