"""SQLite results store (docs/DESIGN.md §4) — append-only, raw responses never discarded.

Result rows (conditions, samples, cache) are inserted, never updated or deleted.
The sole sanctioned mutation is set_run_status: runs.status running -> complete | failed.
The `judgments` table is added to _SCHEMA in M2 (CREATE TABLE IF NOT EXISTS makes that
a zero-migration change for existing DB files).
"""

import hashlib
import json
import secrets
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Self

from mimir.cache import canonical_json

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  id              TEXT PRIMARY KEY,
  experiment_name TEXT NOT NULL,
  spec_json       TEXT NOT NULL,
  spec_hash       TEXT NOT NULL,
  created_at      TEXT NOT NULL,
  status          TEXT NOT NULL CHECK (status IN ('running', 'complete', 'failed'))
);

CREATE TABLE IF NOT EXISTS conditions (
  id            INTEGER PRIMARY KEY,
  run_id        TEXT NOT NULL REFERENCES runs(id),
  variant_name  TEXT NOT NULL,
  system_prompt TEXT NOT NULL,
  user_template TEXT NOT NULL,
  sampling_json TEXT NOT NULL,
  UNIQUE (run_id, variant_name),
  UNIQUE (run_id, id)
);

CREATE TABLE IF NOT EXISTS samples (
  id            INTEGER PRIMARY KEY,
  run_id        TEXT NOT NULL REFERENCES runs(id),
  condition_id  INTEGER NOT NULL,
  item_id       TEXT NOT NULL,
  sample_index  INTEGER NOT NULL,
  cache_key     TEXT NOT NULL,
  request_json  TEXT NOT NULL,
  raw_response  TEXT,
  response_text TEXT,
  latency_ms    REAL,
  input_tokens  INTEGER,
  output_tokens INTEGER,
  cache_hit     INTEGER NOT NULL DEFAULT 0,
  error         TEXT,
  created_at    TEXT NOT NULL,
  UNIQUE (condition_id, item_id, sample_index),
  UNIQUE (run_id, id),
  -- Composite FK: a sample's condition must belong to the sample's own run.
  FOREIGN KEY (run_id, condition_id) REFERENCES conditions (run_id, id)
);

CREATE INDEX IF NOT EXISTS idx_samples_run_id ON samples(run_id);

CREATE TABLE IF NOT EXISTS cache (
  key           TEXT PRIMARY KEY,
  response_json TEXT NOT NULL,
  created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS judgments (
  id             INTEGER PRIMARY KEY,
  run_id         TEXT NOT NULL REFERENCES runs(id),
  item_id        TEXT NOT NULL,
  judge_model    TEXT NOT NULL,
  mode           TEXT NOT NULL,
  sample_a_id    INTEGER NOT NULL,
  sample_b_id    INTEGER,
  position_order TEXT,
  cache_key      TEXT NOT NULL,
  raw_response   TEXT,
  verdict        TEXT,
  score          REAL,
  latency_ms     REAL,
  input_tokens   INTEGER,
  output_tokens  INTEGER,
  error          TEXT,
  created_at     TEXT NOT NULL,
  -- Run-scoped FKs (M9): a judgment's samples must belong to the judgment's own
  -- run (a NULL sample_b_id leaves that FK unenforced, as SQLite defines).
  -- Pre-M9 databases keep the old DDL (CREATE IF NOT EXISTS never alters), so the
  -- skip-and-count drifted-row guards in stats/judge_audit remain load-bearing.
  FOREIGN KEY (run_id, sample_a_id) REFERENCES samples (run_id, id),
  FOREIGN KEY (run_id, sample_b_id) REFERENCES samples (run_id, id)
);

CREATE INDEX IF NOT EXISTS idx_judgments_run_id ON judgments(run_id);
"""

_RUN_ID_ATTEMPTS = 5


def _new_run_id() -> str:
    return f"{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}"


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _utf8_safe(text: str | None) -> str | None:
    # json.loads accepts lone-surrogate escapes ("\ud800") that UTF-8/SQLite cannot
    # store; replace them so provider text is always persisted instead of crashing.
    if text is None:
        return None
    return text.encode("utf-8", "replace").decode("utf-8")


class Store:
    """One SQLite file per project; owns the single connection (cache table included)."""

    def __init__(self, path: str | Path) -> None:
        # autocommit=True: every method is a single statement, durable immediately;
        # no transaction is ever held open across the M2 runner's await points.
        self._conn = sqlite3.connect(str(path), autocommit=True)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        # sqlite3.Connection's own context manager handles transactions, not closing;
        # Store's closes the connection.
        self.close()

    def create_run(self, experiment_name: str, spec: dict[str, Any]) -> str:
        # Takes the spec as a dict so the store is the single canonicalization point
        # ("spec_hash = sha256 of canonical spec_json", DESIGN.md §4).
        spec_json = canonical_json(spec).decode("utf-8")
        spec_hash = hashlib.sha256(spec_json.encode("utf-8")).hexdigest()
        for attempt in range(_RUN_ID_ATTEMPTS):
            run_id = _new_run_id()
            try:
                self._conn.execute(
                    "INSERT INTO runs"
                    " (id, experiment_name, spec_json, spec_hash, created_at, status)"
                    " VALUES (?, ?, ?, ?, ?, 'running')",
                    (run_id, experiment_name, spec_json, spec_hash, _utc_now_iso()),
                )
            except sqlite3.IntegrityError:
                if attempt == _RUN_ID_ATTEMPTS - 1:
                    raise
                continue
            return run_id
        raise AssertionError("unreachable")

    def add_condition(
        self,
        run_id: str,
        *,
        variant_name: str,
        system_prompt: str,
        user_template: str,
        sampling: dict[str, Any],
    ) -> int:
        cursor = self._conn.execute(
            "INSERT INTO conditions"
            " (run_id, variant_name, system_prompt, user_template, sampling_json)"
            " VALUES (?, ?, ?, ?, ?)",
            (
                run_id,
                variant_name,
                system_prompt,
                user_template,
                canonical_json(sampling).decode("utf-8"),
            ),
        )
        return int(cursor.lastrowid)  # type: ignore[arg-type]

    def add_sample(
        self,
        *,
        run_id: str,
        condition_id: int,
        item_id: str,
        sample_index: int,
        cache_key: str,
        request_json: str,
        raw_response: str | None = None,
        response_text: str | None = None,
        latency_ms: float | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cache_hit: bool = False,
        error: str | None = None,
    ) -> int:
        cursor = self._conn.execute(
            "INSERT INTO samples"
            " (run_id, condition_id, item_id, sample_index, cache_key, request_json,"
            "  raw_response, response_text, latency_ms, input_tokens, output_tokens,"
            "  cache_hit, error, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                condition_id,
                item_id,
                sample_index,
                cache_key,
                _utf8_safe(request_json),
                _utf8_safe(raw_response),
                _utf8_safe(response_text),
                latency_ms,
                input_tokens,
                output_tokens,
                int(cache_hit),
                _utf8_safe(error),
                _utc_now_iso(),
            ),
        )
        return int(cursor.lastrowid)  # type: ignore[arg-type]

    def set_run_status(self, run_id: str, status: Literal["complete", "failed"]) -> None:
        """The sole UPDATE in the codebase: runs.status running -> complete | failed."""
        if status not in ("complete", "failed"):
            raise ValueError(
                f"invalid target status {status!r}; only 'complete' or 'failed' are allowed"
            )
        cursor = self._conn.execute(
            "UPDATE runs SET status = ? WHERE id = ? AND status = 'running'",
            (status, run_id),
        )
        if cursor.rowcount != 1:
            raise ValueError(f"run {run_id!r} does not exist or is not in 'running' status")

    def cache_get(self, key: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT response_json FROM cache WHERE key = ?", (key,)).fetchone()
        return None if row is None else json.loads(row["response_json"])

    def cache_put(self, key: str, envelope: dict[str, Any]) -> None:
        """Store a response envelope under its content-addressed key.

        Envelope shape (DESIGN.md §6 CompletionResponse; opaque to the store):
        {"text", "raw", "input_tokens", "output_tokens", "latency_ms", "model"}.
        INSERT OR IGNORE: first write wins — same key means same request content,
        and overwriting would violate the append-only rule. Lone surrogates in the
        envelope are replaced (not backslash-escaped: json.loads would resurrect
        them on cache_get) so provider text can always be persisted.
        """
        text = json.dumps(envelope, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        self._conn.execute(
            "INSERT OR IGNORE INTO cache (key, response_json, created_at) VALUES (?, ?, ?)",
            (key, _utf8_safe(text), _utc_now_iso()),
        )

    def add_judgment(
        self,
        *,
        run_id: str,
        item_id: str,
        judge_model: str,
        mode: str,
        sample_a_id: int,
        sample_b_id: int | None = None,
        position_order: str | None = None,
        cache_key: str,
        raw_response: str | None = None,
        verdict: str | None = None,
        score: float | None = None,
        latency_ms: float | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        error: str | None = None,
    ) -> int:
        # sample_a_id/sample_b_id are the samples PRESENTED in positions A/B;
        # position_order records which variant order that was ('ab' = declared order).
        cursor = self._conn.execute(
            "INSERT INTO judgments"
            " (run_id, item_id, judge_model, mode, sample_a_id, sample_b_id, position_order,"
            "  cache_key, raw_response, verdict, score, latency_ms, input_tokens, output_tokens,"
            "  error, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                item_id,
                judge_model,
                mode,
                sample_a_id,
                sample_b_id,
                position_order,
                cache_key,
                _utf8_safe(raw_response),
                verdict,
                score,
                latency_ms,
                input_tokens,
                output_tokens,
                _utf8_safe(error),
                _utc_now_iso(),
            ),
        )
        return int(cursor.lastrowid)  # type: ignore[arg-type]

    def get_judgments(self, run_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM judgments WHERE run_id = ? ORDER BY id", (run_id,)
        ).fetchall()
        return [dict(row) for row in rows]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return None if row is None else dict(row)

    def get_conditions(self, run_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM conditions WHERE run_id = ? ORDER BY id", (run_id,)
        ).fetchall()
        return [dict(row) for row in rows]

    def get_samples(self, run_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM samples WHERE run_id = ? ORDER BY id", (run_id,)
        ).fetchall()
        return [dict(row) for row in rows]
