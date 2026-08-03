<p align="center">
  <img src="assets/logo.png" alt="mimir" width="520">
</p>

# mimir (WIP)

in Norse mythology, Mimir guarded the well of wisdom, Mímisbrunnr, beneath the world tree. Odin gave up an eye for a single drink from it.

real knowledge comes at a cost.

same is true with llm experimentation where you can't just eyeball a few outputs and know which prompt is better

mimir is a small harness for running LLM experiments properly. You describe an experiment in a YAML file (a few prompt variants, a dataset, how many samples). mimir runs it and tells you whether the difference between variants is statistically real or just noise. mimir also checks whether the LLM judge scoring your outputs is actually reliable and robust

## install

not on PyPI yet (it will ship as `mimisbrunnr` — plain `mimir` was taken; the CLI and import name stay `mimir`). for now, install from source:

```sh
git clone https://github.com/ericxu88/mimir.git
cd mimir
uv sync
uv run mimir  # prints the version
```

## quickstart

run the bundled example offline — a deterministic canned client, no API key needed:

```sh
uv run mimir run examples/quickstart.yaml --mock
```

for real runs (and anything with an LLM judge), export your key and use the judged example:

```sh
export ANTHROPIC_API_KEY=sk-ant-...
uv run mimir run examples/judged.yaml
uv run mimir analyze <run-id>          # terminal report: CIs, p-values, power
uv run mimir analyze <run-id> --html   # plus a self-contained HTML report
uv run mimir audit-judge <run-id>      # judge report card: position/length bias
```

results append to `mimir.db` in the current directory (override with `--db`). responses are cached by content, so re-running a spec only executes what changed. judged specs need the real client — under `--mock` the canned responses never parse as verdicts, so a judged run honestly ends `failed`.

## cli

| command | what it does |
| --- | --- |
| `mimir run SPEC [--db PATH] [--mock]` | run the experiment (and its judge, if configured) |
| `mimir analyze RUN_ID [--db PATH] [--html [PATH]] [--correction {bh,holm}]` | paired-bootstrap stats report, optionally as HTML; `--correction` picks the multi-arm p-value correction (default holm) |
| `mimir audit-judge RUN_ID [--compare RUN_ID] [--db PATH]` | judge reliability report card |

exit codes: `0` success, `1` domain errors (bad spec, missing files, missing API key, a run ending `failed`), `2` usage errors. reports go to stdout; warnings and notices go to stderr.

## development

```sh
uv sync
uv run pytest
uv run ruff check .
```
