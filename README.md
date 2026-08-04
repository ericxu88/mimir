<p align="center">
  <img src="assets/logo.png" alt="mimir" width="100%">
</p>

# mimir (WIP)

in Norse mythology, Mimir guarded the well of wisdom, Mímisbrunnr, beneath the world tree. Odin gave up an eye for a single drink from it.

real knowledge comes at a cost.

same is true for any stochastic system — an LLM prompt, a randomized algorithm, a noisy benchmark — where you can't just eyeball a few outputs and know which variant is better

mimir is a statistically rigorous experiment harness for stochastic systems. You describe an experiment in a YAML file (a few variants, a dataset, how many samples). mimir runs it and tells you whether the difference between variants is statistically real or just noise. variants can be LLM prompts (the flagship adapter — with a built-in auditor that checks whether the LLM judge scoring your outputs is actually reliable), shell commands, or python callables. experiments can be pre-registered: declare a hypothesis and an analysis plan up front, get a hash to commit to, and any analysis that deviates from the plan is labeled EXPLORATORY

## install

not on PyPI yet. for now, install from source:

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

## beyond LLMs: any stochastic system

no API key needed — compare two seeded randomized algorithms end to end, with a pre-registered hypothesis:

```sh
uv run mimir run examples/non_llm/experiment.yaml --db knap.db
# run 20260804-021035-6678 complete
# samples: 192 (0 errors) | judgments: 192 (0 errors)
# preregistered: 717a237bb2f1...   <- sha256 of the hypothesis + analysis plan + design
uv run mimir analyze <run-id> --db knap.db
```

the example pre-registers a hypothesis (simulated annealing beats random-restart hill climbing on seeded knapsack instances) plus an analysis plan (primary comparison, alpha, correction). `analyze` shows the preregistration hash, says whether the analysis was confirmatory, and tags the planned primary comparison `[PRIMARY]`; overriding the planned correction relabels everything `EXPLORATORY`. a variant here is a `type: command` argv template — any program that takes a seed and prints a score works. `examples/subprocess.yaml` is the minimal judgeless-scoring version of the same idea

## cli

| command | what it does |
| --- | --- |
| `mimir run SPEC [--db PATH] [--mock]` | run the experiment (and its judge/scorer, if configured); prints the preregistration hash for planned specs |
| `mimir analyze RUN_ID [--db PATH] [--html [PATH]] [--correction {bh,holm}]` | paired-bootstrap stats report, optionally as HTML; `--correction` picks the multi-arm p-value correction (default: the spec's pre-registered choice, else holm) |
| `mimir audit-judge RUN_ID [--compare RUN_ID] [--db PATH]` | judge reliability report card |

exit codes: `0` success, `1` domain errors (bad spec, missing files, missing API key, a run ending `failed`), `2` usage errors. reports go to stdout; warnings and notices go to stderr.

## development

```sh
uv sync
uv run pytest
uv run ruff check .
```
