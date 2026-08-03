"""Command-line entry point (docs/DESIGN.md §9) — run / analyze / audit-judge.

Exit codes: 0 success; 1 domain errors (ValueError from spec/stats/audit, missing
files, and a run that ends `failed`); 2 argparse usage errors (its own SystemExit,
deliberately uncaught). Reports go to stdout; warnings and notices go to stderr so
stdout stays parseable. Bare `mimir` prints the version and exits 0 (M0 contract).

`run` uses the real Anthropic client by default (its constructor reads
ANTHROPIC_API_KEY; a missing key raises ValueError, caught by `_cmd_run`'s
standard exit-1 path before the store is created). `--mock` swaps in the
deterministic MockClient and never touches the environment.
"""

import argparse
import asyncio
import re
import sqlite3
import sys
from pathlib import Path

import yaml

from mimir import __version__
from mimir.clients.anthropic import AnthropicClient
from mimir.clients.base import Client
from mimir.clients.mock import MockClient
from mimir.judge_audit import audit_judge
from mimir.report import render_analysis_text, render_audit_text, render_html
from mimir.runner import run_experiment
from mimir.spec import load_spec
from mimir.stats import CORRECTION_METHODS, DEFAULT_CORRECTION, analyze_run
from mimir.store import Store

_DB_HELP = "SQLite results database (default: %(default)s)"


def _fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 1


def _new_store(path: str) -> Store:
    # A corrupt file or a directory raises sqlite3.Error at open (the schema
    # script runs in Store.__init__); surface it as the CLI's ValueError family.
    try:
        return Store(path)
    except sqlite3.Error as exc:
        raise ValueError(f"{path} is not a usable mimir database: {exc}") from exc


def _open_store(path: str) -> Store:
    # Store(path) creates the file on open; analyze/audit must not litter empty
    # databases when the user typos --db.
    if not Path(path).exists():
        raise ValueError(f"database {path} not found; run `mimir run` first")
    return _new_store(path)


def _make_client(args: argparse.Namespace) -> Client:
    if args.mock:  # mock branch first: --mock must never read the environment
        print(
            "note: using the deterministic mock client; responses are canned",
            file=sys.stderr,
        )
        return MockClient()
    return AnthropicClient()


def _warn_on_status(status: str, run_id: str) -> None:
    if status != "complete":
        print(
            f"warning: run {run_id} status is {status}; results reflect partial data",
            file=sys.stderr,
        )


def _cmd_run(args: argparse.Namespace) -> int:
    spec_path = Path(args.spec)
    try:
        spec = load_spec(spec_path)
        client = _make_client(args)
        store_cm = _new_store(args.db)
    except (ValueError, OSError, yaml.YAMLError) as exc:
        return _fail(str(exc))
    with store_cm as store:
        try:
            run_id = asyncio.run(run_experiment(spec, store, client, base_dir=spec_path.parent))
        except (ValueError, OSError) as exc:
            # OSError: a missing/unreadable dataset file raises inside
            # run_experiment (load_items), before any run row exists.
            return _fail(str(exc))
        status = store.get_run(run_id)["status"]
        samples = store.get_samples(run_id)
        judgments = store.get_judgments(run_id)
    sample_errors = sum(1 for row in samples if row["error"] is not None)
    judgment_errors = sum(1 for row in judgments if row["error"] is not None)
    print(f"run {run_id} {status}")
    print(
        f"samples: {len(samples)} ({sample_errors} errors)"
        f" | judgments: {len(judgments)} ({judgment_errors} errors)"
    )
    return 0 if status == "complete" else 1


def _html_path(args: argparse.Namespace, run_id: str) -> Path:
    if args.html:
        return Path(args.html)
    # Native run ids (YYYYMMDD-HHMMSS-hex4) are already filesystem-safe; the sub
    # guards hand-crafted DB ids against path traversal or reserved characters.
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", run_id)
    return Path(f"mimir-report-{safe}.html")


def _cmd_analyze(args: argparse.Namespace) -> int:
    try:
        with _open_store(args.db) as store:
            result = analyze_run(store, args.run_id, correction=args.correction)
            status = store.get_run(args.run_id)["status"]
            card = None
            if args.html is not None:
                # Defensive: analyze succeeding implies a judged run, but the audit
                # must never kill the report — the judge section is simply omitted.
                try:
                    card = audit_judge(store, args.run_id)
                except ValueError:
                    card = None
    except ValueError as exc:
        return _fail(str(exc))
    _warn_on_status(status, args.run_id)
    print(render_analysis_text(result, status=status))
    if args.html is not None:
        path = _html_path(args, args.run_id)
        try:
            path.write_text(render_html(result, card, status=status), encoding="utf-8")
        except OSError as exc:
            return _fail(f"cannot write {path}: {exc}")
        print(f"wrote {path}")
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    try:
        with _open_store(args.db) as store:
            card = audit_judge(store, args.run_id, compare_run_id=args.compare)
            status = store.get_run(args.run_id)["status"]
    except ValueError as exc:
        return _fail(str(exc))
    _warn_on_status(status, args.run_id)
    print(render_audit_text(card))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    # prog is pinned: under pytest, sys.argv[0] is the pytest binary.
    parser = argparse.ArgumentParser(
        prog="mimir", description="statistically rigorous LLM experiments"
    )
    parser.add_argument("-V", "--version", action="version", version=f"mimir {__version__}")
    sub = parser.add_subparsers(dest="command")  # not required: bare call prints version

    run_p = sub.add_parser("run", help="execute an experiment spec")
    run_p.add_argument("spec", help="path to the experiment spec YAML")
    run_p.add_argument("--db", default="mimir.db", help=_DB_HELP)
    run_p.add_argument(
        "--mock",
        action="store_true",
        help="run offline against the deterministic mock client (no API key needed)",
    )
    run_p.set_defaults(func=_cmd_run)

    analyze_p = sub.add_parser("analyze", help="statistical analysis of a stored run")
    analyze_p.add_argument("run_id")
    analyze_p.add_argument("--db", default="mimir.db", help=_DB_HELP)
    analyze_p.add_argument(
        "--correction",
        choices=CORRECTION_METHODS,
        default=DEFAULT_CORRECTION,
        help="multiple-comparison correction across the run's variant pairs"
        " (default: %(default)s; no effect on a 2-variant run)",
    )
    analyze_p.add_argument(
        "--html",
        nargs="?",
        const="",
        default=None,
        metavar="PATH",
        help="also write a self-contained HTML report (default name: mimir-report-<run_id>.html)",
    )
    analyze_p.set_defaults(func=_cmd_analyze)

    audit_p = sub.add_parser("audit-judge", help="judge reliability report card for a stored run")
    audit_p.add_argument("run_id")
    audit_p.add_argument(
        "--compare", metavar="RUN_ID", default=None, help="second run for cross-judge kappa"
    )
    audit_p.add_argument("--db", default="mimir.db", help=_DB_HELP)
    audit_p.set_defaults(func=_cmd_audit)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    if args.command is None:
        print(f"mimir {__version__}")
        return 0
    return args.func(args)
