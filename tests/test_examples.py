"""Shipped examples stay valid: schema + dataset load, and the README quickstart
command runs end-to-end under --mock. Paths anchor on this file, never the cwd."""

from pathlib import Path

import pytest

from mimir.cli import main
from mimir.spec import load_items, load_spec

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@pytest.mark.parametrize("name", ["quickstart.yaml", "judged.yaml"])
def test_example_spec_and_dataset_validate(name):
    spec = load_spec(EXAMPLES / name)
    items = load_items(spec, EXAMPLES)
    assert len(items) == 6


def test_quickstart_is_judgeless_so_it_runs_under_mock():
    # PROGRESS constraint: judged runs under --mock end `failed` by design, so
    # the spec the README tells people to run offline must not have a judge.
    spec = load_spec(EXAMPLES / "quickstart.yaml")
    assert spec.judge is None


def test_judged_example_is_pairwise():
    spec = load_spec(EXAMPLES / "judged.yaml")
    assert spec.judge is not None
    assert spec.judge.mode == "pairwise"


def test_quickstart_runs_end_to_end_under_mock(tmp_path, capsys):
    # The literal README quickstart command, with an isolated --db.
    db = tmp_path / "results.db"
    assert main(["run", str(EXAMPLES / "quickstart.yaml"), "--db", str(db), "--mock"]) == 0
    captured = capsys.readouterr()
    assert " complete" in captured.out
    assert "samples: 24 (0 errors)" in captured.out  # 2 variants x 6 items x n_samples 2


def test_quickstart_analyze_exits_with_the_documented_error(tmp_path, capsys):
    # M8/M6: the example used to tell the reader to run `mimir analyze` next, which
    # can never succeed on a judgeless run. The two halves were each tested and
    # never composed, so the contradiction shipped. This pins the real behavior the
    # header comment now documents.
    db = tmp_path / "results.db"
    assert main(["run", str(EXAMPLES / "quickstart.yaml"), "--db", str(db), "--mock"]) == 0
    run_id = capsys.readouterr().out.split()[1]

    assert main(["analyze", run_id, "--db", str(db)]) == 1
    assert "no judge configured" in capsys.readouterr().err

    header = (EXAMPLES / "quickstart.yaml").read_text(encoding="utf-8")
    assert "COLLECTS SAMPLES ONLY" in header
    assert "examples/judged.yaml" in header


# --- M10: the subprocess example — full statistics, no API key ---------------------


def test_subprocess_example_validates():
    spec = load_spec(EXAMPLES / "subprocess.yaml")
    items = load_items(spec, EXAMPLES)
    assert len(items) == 5
    assert all(variant.type == "command" for variant in spec.variants)
    assert spec.judge.type == "parse_float"
    assert spec.judge.mode == "rubric"


def test_subprocess_example_runs_and_analyzes_keyless(tmp_path, monkeypatch, capsys):
    # The repo's first fully-offline path to real statistical output: run and
    # analyze with no ANTHROPIC_API_KEY, no --mock, no client at all.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    db = tmp_path / "bench.db"
    assert main(["run", str(EXAMPLES / "subprocess.yaml"), "--db", str(db)]) == 0
    captured = capsys.readouterr()
    assert " complete" in captured.out
    assert "samples: 120 (0 errors)" in captured.out  # 2 variants x 5 items x 12
    assert "judgments: 120 (0 errors)" in captured.out
    assert captured.err == ""  # no mock notice, no warnings
    run_id = captured.out.split()[1]

    assert main(["analyze", run_id, "--db", str(db)]) == 0
    out = capsys.readouterr().out
    assert "(rubric" in out
    assert "fast" in out
    assert "slow" in out
    assert "95% CI" in out


# --- M11: the pre-registered non-LLM example ---------------------------------------


def test_non_llm_example_validates():
    spec = load_spec(EXAMPLES / "non_llm" / "experiment.yaml")
    items = load_items(spec, EXAMPLES / "non_llm")
    assert len(items) == 8  # sign-flip floor 2/2^8 < 0.05: the design CAN reject
    assert all(variant.type == "command" for variant in spec.variants)
    assert spec.judge.type == "parse_float"
    assert spec.hypothesis is not None
    assert spec.analysis_plan.primary == ("restart", "anneal")
    assert spec.analysis_plan.correction == "holm"


def test_non_llm_example_runs_preregistered_end_to_end(tmp_path, monkeypatch, capsys):
    # The full M11 story, keyless: run prints the commitment hash; analyze shows
    # the same hash, confirmatory status, and the [PRIMARY] tag. Assertions are
    # structural — algorithm quality numbers live in the tuning, not in tests.
    import re

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    db = tmp_path / "knap.db"
    assert main(["run", str(EXAMPLES / "non_llm" / "experiment.yaml"), "--db", str(db)]) == 0
    captured = capsys.readouterr()
    assert "samples: 192 (0 errors)" in captured.out  # 2 arms x 8 instances x 12
    assert "judgments: 192 (0 errors)" in captured.out
    assert captured.err == ""
    run_id = re.search(r"run (\S+) complete", captured.out).group(1)
    run_hash = re.search(r"preregistered: ([0-9a-f]{64})", captured.out).group(1)

    assert main(["analyze", run_id, "--db", str(db)]) == 0
    captured = capsys.readouterr()
    assert f"preregistration: {run_hash} (confirmatory)" in captured.out
    assert "[PRIMARY]" in captured.out
    assert "cannot reach significance" not in captured.out
    assert captured.err == ""
