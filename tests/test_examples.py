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
