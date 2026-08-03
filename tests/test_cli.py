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


def test_run_without_mock_flag_errors(tmp_path, capsys):
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec_dict()), encoding="utf-8")
    db = tmp_path / "results.db"
    assert main(["run", str(spec_path), "--db", str(db)]) == 1
    captured = capsys.readouterr()
    assert "error:" in captured.err
    assert "--mock" in captured.err
    assert not db.exists()  # rejected before the store is opened


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
        "note: using the deterministic mock client; responses are canned"
        " (real client lands in M6)" in captured.err
    )
    with Store(db) as store:
        assert store.get_run(match.group(1))["status"] == "complete"


def test_run_judged_spec_under_mock_fails(tmp_path, capsys):
    # MockClient's derived texts don't parse as verdicts: honest `failed`, exit 1
    # (documented as expected until M6's real client).
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
