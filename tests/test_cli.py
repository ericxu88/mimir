"""Tests for mimir.cli — argument grammar, exit codes, end-to-end subcommands (DESIGN.md §9).

Exit-code scheme: 0 success, 1 domain errors (and `run` ending failed), 2 argparse
usage errors (SystemExit, deliberately uncaught). CLI tests are sync; stores are
seeded via asyncio.run(run_experiment(...)) with rigged MockClients (constructions
copied from test_judge_audit.py — always-A judge gives exactly-known analyze/audit
numbers) and the seeding Store is always closed before main() runs.
"""

import asyncio
import json
import re
import sqlite3
import sys

import pytest
import yaml

import mimir
from mimir.cli import main
from mimir.clients.mock import MockClient
from mimir.runner import run_experiment
from mimir.spec import ExperimentSpec
from mimir.store import Store


def spec_dict(*, judge=None, dataset=None):
    spec = {
        "name": "greeting-tone",
        "variants": [
            {"name": "control", "system": "You are helpful.", "user_template": "A: {input}"},
            {"name": "friendly", "system": "You are warm.", "user_template": "A: {input}"},
        ],
        "dataset": dataset
        or {"items": [{"id": "q1", "input": "sky?"}, {"id": "q2", "input": "grass?"}]},
        "sampling": {"model": "claude-haiku-4-5-20251001"},
        "n_samples": 1,
        "limits": {"concurrency": 4, "requests_per_minute": 100_000},
    }
    if judge is not None:
        spec["judge"] = judge
    return spec


def seed_judged_run(db_path, *, judge_model="judge-model", client=None):
    """A complete always-A judged run: flip rate 1.0, mean diff 0.0, p 1.0 exactly."""
    spec = ExperimentSpec.model_validate(
        spec_dict(judge={"model": judge_model, "mode": "pairwise"})
    )
    if client is None:
        client = MockClient()
        client.add_rule(lambda request: request.model == judge_model, "A")
    with Store(db_path) as store:
        return asyncio.run(run_experiment(spec, store, client))


# --- grammar and exit codes -------------------------------------------------------


def test_bare_invocation_prints_version(capsys):
    assert main([]) == 0
    assert capsys.readouterr().out.strip() == f"mimir {mimir.__version__}"


@pytest.mark.parametrize(
    "flag", [pytest.param("--version", id="long"), pytest.param("-V", id="short")]
)
def test_version_flag(flag, capsys):
    with pytest.raises(SystemExit) as excinfo:
        main([flag])
    assert excinfo.value.code == 0
    assert capsys.readouterr().out.strip() == f"mimir {mimir.__version__}"


def test_unknown_subcommand_usage_error():
    with pytest.raises(SystemExit) as excinfo:
        main(["nope"])
    assert excinfo.value.code == 2


def test_analyze_without_run_id_usage_error():
    with pytest.raises(SystemExit) as excinfo:
        main(["analyze"])
    assert excinfo.value.code == 2


# --- mimir run --------------------------------------------------------------------


def test_run_without_key_and_without_mock_errors(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec_dict()), encoding="utf-8")
    db = tmp_path / "results.db"
    assert main(["run", str(spec_path), "--db", str(db)]) == 1
    captured = capsys.readouterr()
    assert "error:" in captured.err
    assert "ANTHROPIC_API_KEY" in captured.err
    assert not db.exists()  # client construction fails before the store is opened


def test_run_mock_needs_no_api_key(tmp_path, monkeypatch, capsys):
    # The mock branch must never read the environment.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec_dict()), encoding="utf-8")
    assert main(["run", str(spec_path), "--db", str(tmp_path / "r.db"), "--mock"]) == 0


def test_run_without_mock_uses_real_client(tmp_path, monkeypatch, capsys):
    # Stub the class at the cli seam (never let a real key on the dev machine
    # construct a live client). MockClient's zero-arg constructor fits the call.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("mimir.cli.AnthropicClient", MockClient)
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec_dict()), encoding="utf-8")
    assert main(["run", str(spec_path), "--db", str(tmp_path / "r.db")]) == 0
    captured = capsys.readouterr()
    assert "mock client" not in captured.err  # the notice belongs to --mock only


def test_run_mock_judgeless_end_to_end(tmp_path, monkeypatch, capsys):
    # dataset.path resolves relative to the SPEC file's parent, not the cwd —
    # the test chdirs elsewhere to prove it.
    proj = tmp_path / "proj"
    (proj / "data").mkdir(parents=True)
    items = [{"id": "q1", "input": "sky?"}, {"id": "q2", "input": "grass?"}]
    (proj / "data" / "q.jsonl").write_text(
        "\n".join(json.dumps(item) for item in items), encoding="utf-8"
    )
    spec_path = proj / "spec.yaml"
    spec_path.write_text(
        yaml.safe_dump(spec_dict(dataset={"path": "data/q.jsonl"})), encoding="utf-8"
    )
    db = tmp_path / "results.db"
    monkeypatch.chdir(tmp_path)
    assert main(["run", str(spec_path), "--db", str(db), "--mock"]) == 0
    captured = capsys.readouterr()
    match = re.search(r"run (\d{8}-\d{6}-[0-9a-f]{4}) complete", captured.out)
    assert match is not None
    assert "samples: 4 (0 errors)" in captured.out
    assert "judgments: 0 (0 errors)" in captured.out
    assert (  # the exact canned-client notice is contract, pinned on stderr
        "note: using the deterministic mock client; responses are canned" in captured.err
    )
    assert "lands in M6" not in captured.err  # stale-suffix guard
    with Store(db) as store:
        assert store.get_run(match.group(1))["status"] == "complete"


def test_run_judged_spec_under_mock_fails(tmp_path, capsys):
    # MockClient's derived texts don't parse as verdicts: honest `failed`, exit 1
    # (by design: use the real client for judged runs).
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(
        yaml.safe_dump(spec_dict(judge={"model": "judge-model", "mode": "pairwise"})),
        encoding="utf-8",
    )
    db = tmp_path / "results.db"
    assert main(["run", str(spec_path), "--db", str(db), "--mock"]) == 1
    captured = capsys.readouterr()
    assert " failed" in captured.out
    assert "judgments: 4 (4 errors)" in captured.out


def test_run_missing_spec_file(tmp_path, capsys):
    assert main(["run", str(tmp_path / "nope.yaml"), "--db", str(tmp_path / "x.db"), "--mock"]) == 1
    assert "error:" in capsys.readouterr().err


def test_run_invalid_spec(tmp_path, capsys):
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump({"name": "broken"}), encoding="utf-8")
    assert main(["run", str(spec_path), "--db", str(tmp_path / "x.db"), "--mock"]) == 1
    assert "error:" in capsys.readouterr().err


# --- mimir analyze ----------------------------------------------------------------


def test_analyze_end_to_end(tmp_path, capsys):
    db = tmp_path / "results.db"
    run_id = seed_judged_run(db)
    assert main(["analyze", run_id, "--db", str(db)]) == 0
    captured = capsys.readouterr()
    # Always-A judge washes out exactly (known from test_judge_audit e2e).
    assert "mean diff:        0.000" in captured.out
    assert "p-value:          1.0000" in captured.out
    assert captured.err == ""  # complete run: no status warning


def test_analyze_html_explicit_path(tmp_path, capsys):
    db = tmp_path / "results.db"
    run_id = seed_judged_run(db)
    out_path = tmp_path / "report.html"
    assert main(["analyze", run_id, "--db", str(db), "--html", str(out_path)]) == 0
    assert f"wrote {out_path}" in capsys.readouterr().out
    html_text = out_path.read_text(encoding="utf-8")
    assert html_text.startswith("<!DOCTYPE html>")
    assert "control" in html_text
    assert "friendly" in html_text
    assert "judge report card" in html_text


def test_analyze_html_default_filename(tmp_path, monkeypatch, capsys):
    db = tmp_path / "results.db"
    run_id = seed_judged_run(db)
    monkeypatch.chdir(tmp_path)
    assert main(["analyze", run_id, "--db", str(db), "--html"]) == 0
    expected = tmp_path / f"mimir-report-{run_id}.html"
    assert expected.exists()
    assert f"wrote mimir-report-{run_id}.html" in capsys.readouterr().out


def test_analyze_unknown_run(tmp_path, capsys):
    db = tmp_path / "results.db"
    seed_judged_run(db)
    assert main(["analyze", "nope", "--db", str(db)]) == 1
    assert "not found" in capsys.readouterr().err


def test_analyze_missing_db_is_not_created(tmp_path, capsys):
    db = tmp_path / "nope.db"
    assert main(["analyze", "x", "--db", str(db)]) == 1
    assert "not found" in capsys.readouterr().err
    assert not db.exists()  # Store(path) would have created it silently


def test_analyze_failed_run_warns_on_stderr(tmp_path, capsys):
    # Judge answers only for q1; q2's judgments are parse errors -> run `failed`,
    # but q1 still analyzes: report renders, warning goes to stderr.
    db = tmp_path / "results.db"
    client = MockClient()
    client.add_rule(lambda request: request.model == "judge-model" and "sky?" in request.user, "A")
    run_id = seed_judged_run(db, client=client)
    assert main(["analyze", run_id, "--db", str(db)]) == 0
    captured = capsys.readouterr()
    assert "warning" in captured.err
    assert "partial" in captured.err
    assert "experiment: greeting-tone" in captured.out


# --- mimir audit-judge ------------------------------------------------------------


def test_audit_judge_end_to_end(tmp_path, capsys):
    db = tmp_path / "results.db"
    run_id = seed_judged_run(db)
    assert main(["audit-judge", run_id, "--db", str(db)]) == 0
    captured = capsys.readouterr()
    assert "flip rate:           1.000" in captured.out
    assert "position-A win rate: 1.000" in captured.out


def test_audit_judge_compare_kappa(tmp_path, capsys):
    # Two runs differing only in judge model: always-A vs always-B -> kappa -1.0
    # exactly (construction from test_judge_audit's cross-judge e2e).
    db = tmp_path / "results.db"
    client = MockClient()
    client.add_rule(lambda request: request.model == "judge-a", "A")
    client.add_rule(lambda request: request.model == "judge-b", "B")
    run_a = seed_judged_run(db, judge_model="judge-a", client=client)
    run_b = seed_judged_run(db, judge_model="judge-b", client=client)
    assert main(["audit-judge", run_a, "--compare", run_b, "--db", str(db)]) == 0
    assert f"cross-judge kappa:   -1.000 vs run {run_b} (n=4)" in capsys.readouterr().out


def test_audit_judge_unknown_run(tmp_path, capsys):
    db = tmp_path / "results.db"
    seed_judged_run(db)
    assert main(["audit-judge", "nope", "--db", str(db)]) == 1
    assert "not found" in capsys.readouterr().err


# --- review-driven hardening ------------------------------------------------------


def test_run_missing_dataset_file_errors(tmp_path, capsys):
    # A typo'd dataset.path is a routine user error: clean exit 1, never a traceback
    # (load_items raises OSError inside run_experiment, before any run row exists).
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(
        yaml.safe_dump(spec_dict(dataset={"path": "data/nope.jsonl"})), encoding="utf-8"
    )
    assert main(["run", str(spec_path), "--db", str(tmp_path / "r.db"), "--mock"]) == 1
    assert "error:" in capsys.readouterr().err


def test_run_db_path_is_directory_errors(tmp_path, capsys):
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec_dict()), encoding="utf-8")
    target = tmp_path / "adir"
    target.mkdir()
    assert main(["run", str(spec_path), "--db", str(target), "--mock"]) == 1
    assert "not a usable mimir database" in capsys.readouterr().err


def test_analyze_corrupt_db_file_errors(tmp_path, capsys):
    db = tmp_path / "corrupt.db"
    db.write_text("x" * 100, encoding="utf-8")
    assert main(["analyze", "whatever", "--db", str(db)]) == 1
    assert "not a usable mimir database" in capsys.readouterr().err


def test_analyze_html_unwritable_path_errors(tmp_path, capsys):
    db = tmp_path / "results.db"
    run_id = seed_judged_run(db)
    target = tmp_path / "adir"
    target.mkdir()
    assert main(["analyze", run_id, "--db", str(db), "--html", str(target)]) == 1
    captured = capsys.readouterr()
    assert "experiment: greeting-tone" in captured.out  # report still printed
    assert "cannot write" in captured.err


def test_analyze_html_judgeless_run_writes_nothing(tmp_path, monkeypatch, capsys):
    db = tmp_path / "results.db"
    spec = ExperimentSpec.model_validate(spec_dict())
    with Store(db) as store:
        run_id = asyncio.run(run_experiment(spec, store, MockClient()))
    monkeypatch.chdir(tmp_path)
    assert main(["analyze", run_id, "--db", str(db), "--html"]) == 1
    assert "no judge" in capsys.readouterr().err
    assert not list(tmp_path.glob("mimir-report-*.html"))


def test_analyze_html_survives_audit_failure(tmp_path, monkeypatch, capsys):
    # The audit is a defensive add-on to the HTML report: if it raises ValueError,
    # the report is still written, just without the judge section.
    db = tmp_path / "results.db"
    run_id = seed_judged_run(db)

    def boom(*args, **kwargs):
        raise ValueError("boom")

    monkeypatch.setattr("mimir.cli.audit_judge", boom)
    out_path = tmp_path / "report.html"
    assert main(["analyze", run_id, "--db", str(db), "--html", str(out_path)]) == 0
    assert out_path.exists()
    assert "judge report card" not in out_path.read_text(encoding="utf-8")


def test_audit_judge_failed_run_warns_on_stderr(tmp_path, capsys):
    db = tmp_path / "results.db"
    client = MockClient()
    client.add_rule(lambda request: request.model == "judge-model" and "sky?" in request.user, "A")
    run_id = seed_judged_run(db, client=client)
    assert main(["audit-judge", run_id, "--db", str(db)]) == 0
    captured = capsys.readouterr()
    assert "warning" in captured.err
    assert "partial" in captured.err
    assert "judge report card" in captured.out


def test_run_malformed_yaml_spec(tmp_path, capsys):
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text("name: [", encoding="utf-8")
    assert main(["run", str(spec_path), "--db", str(tmp_path / "r.db"), "--mock"]) == 1
    assert "error:" in capsys.readouterr().err


def test_run_default_db_filename(tmp_path, monkeypatch, capsys):
    # The recorded default: mimir.db in the working directory.
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec_dict()), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert main(["run", str(spec_path), "--mock"]) == 0
    assert (tmp_path / "mimir.db").exists()


# --- M7: analyze --correction -----------------------------------------------------


def seed_rubric_multiarm_run(db_path):
    """Three rubric variants seeded directly through the store: C(3,2) = 3 pairs."""
    spec = {"name": "multi", "judge": {"model": "judge-model", "mode": "rubric"}}
    scores = {
        "a": {"q1": 2.0, "q2": 4.0},
        "b": {"q1": 5.0, "q2": 7.0},
        "c": {"q1": 9.0, "q2": 3.0},
    }
    with Store(db_path) as store:
        run_id = store.create_run("multi", spec)
        for name, per_item in scores.items():
            cid = store.add_condition(
                run_id,
                variant_name=name,
                system_prompt="",
                user_template="A: {input}",
                sampling={"model": "m"},
            )
            for item, score in per_item.items():
                sid = store.add_sample(
                    run_id=run_id,
                    condition_id=cid,
                    item_id=item,
                    sample_index=0,
                    cache_key="k" * 64,
                    request_json="{}",
                    raw_response="{}",
                    response_text="r",
                    latency_ms=1.0,
                    input_tokens=1,
                    output_tokens=1,
                )
                store.add_judgment(
                    run_id=run_id,
                    item_id=item,
                    judge_model="judge-model",
                    mode="rubric",
                    sample_a_id=sid,
                    cache_key="j" * 64,
                    score=score,
                )
        store.set_run_status(run_id, "complete")
    return run_id


def test_analyze_multiarm_reports_corrected(tmp_path, capsys):
    db = tmp_path / "results.db"
    run_id = seed_rubric_multiarm_run(db)
    assert main(["analyze", run_id, "--db", str(db)]) == 0
    captured = capsys.readouterr()
    assert "multiple comparisons: 3 pairs" in captured.out
    assert "holm-corrected" in captured.out
    assert captured.err == ""


def test_analyze_correction_flag_bh(tmp_path, capsys):
    db = tmp_path / "results.db"
    run_id = seed_rubric_multiarm_run(db)
    assert main(["analyze", run_id, "--db", str(db), "--correction", "bh"]) == 0
    assert "bh-corrected" in capsys.readouterr().out


def test_analyze_correction_flag_invalid_usage_error(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        main(["analyze", "some-run", "--db", str(tmp_path / "x.db"), "--correction", "bonferroni"])
    assert excinfo.value.code == 2


def test_analyze_correction_flag_on_two_variant_run(tmp_path, capsys):
    db = tmp_path / "results.db"
    run_id = seed_judged_run(db)
    assert main(["analyze", run_id, "--db", str(db), "--correction", "holm"]) == 0
    assert capsys.readouterr().err == ""


# --- M9: sqlite3.Error surfaces as exit 1, never a traceback ----------------------


def test_analyze_sqlite_error_exits_one(tmp_path, monkeypatch, capsys):
    db = tmp_path / "results.db"
    Store(db).close()  # analyze refuses a missing db before touching analyze_run
    def boom(*args, **kwargs):
        raise sqlite3.OperationalError("database disk image is malformed")

    monkeypatch.setattr("mimir.cli.analyze_run", boom)
    assert main(["analyze", "some-run", "--db", str(db)]) == 1
    assert "database error:" in capsys.readouterr().err


def test_audit_sqlite_error_exits_one(tmp_path, monkeypatch, capsys):
    db = tmp_path / "results.db"
    Store(db).close()

    def boom(*args, **kwargs):
        raise sqlite3.OperationalError("database disk image is malformed")

    monkeypatch.setattr("mimir.cli.audit_judge", boom)
    assert main(["audit-judge", "some-run", "--db", str(db)]) == 1
    assert "database error:" in capsys.readouterr().err


def test_run_sqlite_error_exits_one(tmp_path, monkeypatch, capsys):
    async def boom(*args, **kwargs):
        raise sqlite3.OperationalError("database disk image is malformed")

    monkeypatch.setattr("mimir.cli.run_experiment", boom)
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec_dict()), encoding="utf-8")
    assert main(["run", str(spec_path), "--db", str(tmp_path / "r.db"), "--mock"]) == 1
    assert "database error:" in capsys.readouterr().err


# --- M10: non-LLM specs run keyless, no client, no mock notice ---------------------


def command_spec_yaml(tmp_path, *, judge=None):
    code = "import sys; print(float(sys.argv[1]) + int(sys.argv[2]) / 10)"
    spec = {
        "name": "bench",
        "variants": [
            {
                "type": "command",
                "name": "fast",
                "command": [sys.executable, "-c", code, "5.0", "{seed}"],
            },
            {
                "type": "command",
                "name": "slow",
                "command": [sys.executable, "-c", code, "2.0", "{seed}"],
            },
        ],
        "dataset": {"items": [{"id": "q1", "input": "a"}, {"id": "q2", "input": "b"}]},
        "limits": {"concurrency": 4, "requests_per_minute": 100_000},
    }
    if judge is not None:
        spec["judge"] = judge
    path = tmp_path / "bench.yaml"
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return path


def test_run_command_spec_keyless_and_offline(tmp_path, monkeypatch, capsys):
    # No llm parts -> no client is constructed at all; a missing key cannot matter.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    spec_path = command_spec_yaml(tmp_path)
    db = tmp_path / "results.db"
    assert main(["run", str(spec_path), "--db", str(db)]) == 0
    captured = capsys.readouterr()
    assert " complete" in captured.out
    assert "samples: 4 (0 errors)" in captured.out
    assert captured.err == ""


def test_run_command_spec_mock_flag_is_silent_noop(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    spec_path = command_spec_yaml(tmp_path)
    assert main(["run", str(spec_path), "--db", str(tmp_path / "r.db"), "--mock"]) == 0
    assert "mock client" not in capsys.readouterr().err


def test_analyze_parse_float_run_end_to_end(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    spec_path = command_spec_yaml(tmp_path, judge={"type": "parse_float"})
    db = tmp_path / "results.db"
    assert main(["run", str(spec_path), "--db", str(db)]) == 0
    out = capsys.readouterr().out
    match = re.search(r"run (\S+) complete", out)
    assert match is not None
    assert "judgments: 4 (0 errors)" in out
    assert main(["analyze", match.group(1), "--db", str(db)]) == 0
    captured = capsys.readouterr()
    assert "(rubric" in captured.out
    assert "fast" in captured.out
    assert "slow" in captured.out
    assert "-3.000" in captured.out  # slow (2.0) minus fast (5.0), seed 0 replicate 0
    assert captured.err == ""


def test_audit_judge_parse_float_run_errors_cleanly(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    spec_path = command_spec_yaml(tmp_path, judge={"type": "parse_float"})
    db = tmp_path / "results.db"
    assert main(["run", str(spec_path), "--db", str(db)]) == 0
    match = re.search(r"run (\S+) complete", capsys.readouterr().out)
    assert main(["audit-judge", match.group(1), "--db", str(db)]) == 1
    captured = capsys.readouterr()
    assert "error:" in captured.err
    assert re.search(r"mode|model", captured.err)


# --- M11: preregistration through the CLI ------------------------------------------


def planned_command_spec_yaml(tmp_path, *, three_arms=False, plan_correction="holm"):
    code = "import sys; print(float(sys.argv[1]) + int(sys.argv[2]) / 10)"

    def arm(name, base):
        return {
            "type": "command",
            "name": name,
            "command": [sys.executable, "-c", code, str(base), "{seed}"],
        }

    variants = [arm("fast", 5.0), arm("slow", 2.0)]
    if three_arms:
        variants.append(arm("mid", 3.5))
    spec = {
        "name": "bench-prereg",
        "variants": variants,
        "dataset": {"items": [{"id": "q1", "input": "a"}, {"id": "q2", "input": "b"}]},
        "judge": {"type": "parse_float"},
        "hypothesis": "fast beats slow on the benchmark score",
        "analysis_plan": {"primary": ["fast", "slow"], "correction": plan_correction},
        "limits": {"concurrency": 4, "requests_per_minute": 100_000},
    }
    path = tmp_path / "planned.yaml"
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return path


def run_planned(tmp_path, capsys, **kwargs):
    spec_path = planned_command_spec_yaml(tmp_path, **kwargs)
    db = tmp_path / "results.db"
    assert main(["run", str(spec_path), "--db", str(db)]) == 0
    out = capsys.readouterr().out
    run_id = re.search(r"run (\S+) complete", out).group(1)
    return db, run_id, out


def test_run_planned_spec_prints_prereg_hash_last(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    spec_path = planned_command_spec_yaml(tmp_path)
    db = tmp_path / "results.db"
    assert main(["run", str(spec_path), "--db", str(db)]) == 0
    captured = capsys.readouterr()
    lines = captured.out.rstrip().splitlines()
    # Last line, so `out.split()[1]`-style run-id extraction stays valid.
    assert re.fullmatch(r"preregistered: [0-9a-f]{64}", lines[-1])
    assert lines[0].startswith("run ")
    assert captured.err == ""


def test_analyze_planned_run_shows_confirmatory_prereg(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    db, run_id, run_out = run_planned(tmp_path, capsys)
    run_hash = re.search(r"preregistered: ([0-9a-f]{64})", run_out).group(1)
    assert main(["analyze", run_id, "--db", str(db)]) == 0
    captured = capsys.readouterr()
    assert f"preregistration: {run_hash} (confirmatory)" in captured.out
    assert "hypothesis:" in captured.out
    assert "[PRIMARY]" in captured.out
    assert captured.err == ""


def test_analyze_uses_planned_correction_by_default(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    db, run_id, _ = run_planned(tmp_path, capsys, three_arms=True, plan_correction="bh")
    assert main(["analyze", run_id, "--db", str(db)]) == 0
    captured = capsys.readouterr()
    assert "bh-corrected" in captured.out  # the plan's choice, no flag passed
    assert "(confirmatory)" in captured.out
    assert "[EXPLORATORY - not pre-registered]" in captured.out  # unplanned pairs


def test_analyze_explicit_flag_against_plan_deviates_at_m3(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    db, run_id, _ = run_planned(tmp_path, capsys, three_arms=True, plan_correction="holm")
    assert main(["analyze", run_id, "--db", str(db), "--correction", "bh"]) == 0
    captured = capsys.readouterr()
    assert "EXPLORATORY:" in captured.out
    assert "[EXPLORATORY - deviates from plan]" in captured.out
    assert "[PRIMARY]" not in captured.out


def test_analyze_explicit_flag_at_m1_stays_confirmatory(tmp_path, monkeypatch, capsys):
    # Holm and BH are identical for a single comparison (human-approved m=1 rule).
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    db, run_id, _ = run_planned(tmp_path, capsys)
    assert main(["analyze", run_id, "--db", str(db), "--correction", "bh"]) == 0
    captured = capsys.readouterr()
    assert "(confirmatory)" in captured.out
    assert "[PRIMARY]" in captured.out


def test_analyze_garbage_stored_plan_warns_and_renders_without_prereg(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    db, run_id, _ = run_planned(tmp_path, capsys)
    with sqlite3.connect(db) as connection:
        row = connection.execute(
            "SELECT spec_json FROM runs WHERE id = ?", (run_id,)
        ).fetchone()
        spec_json = json.loads(row[0])
        spec_json["analysis_plan"]["alpha"] = 0.5  # invalid: v1 pins alpha
        connection.execute(
            "UPDATE runs SET spec_json = ? WHERE id = ?",
            (json.dumps(spec_json), run_id),
        )
    assert main(["analyze", run_id, "--db", str(db)]) == 0
    captured = capsys.readouterr()
    assert "preregistration" not in captured.out
    assert "warning" in captured.err
    assert "analysis_plan" in captured.err


def test_analyze_planless_run_has_no_prereg_output(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    spec_path = command_spec_yaml(tmp_path, judge={"type": "parse_float"})
    db = tmp_path / "results.db"
    assert main(["run", str(spec_path), "--db", str(db)]) == 0
    out = capsys.readouterr().out
    assert "preregistered" not in out
    run_id = re.search(r"run (\S+) complete", out).group(1)
    assert main(["analyze", run_id, "--db", str(db)]) == 0
    captured = capsys.readouterr()
    assert "preregistration" not in captured.out
    assert captured.err == ""
