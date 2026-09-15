# Talos

Single-user autoresearch for [TIG](https://tig.foundation). Run one setup wizard and one
run wizard, and an LLM agent iterates on the current state-of-the-art algorithm for a TIG
challenge, benchmarking each candidate on TIG's own Modal-hosted harness, until it beats
the baseline or runs out of budget. You get back a submit-ready package.

## Requirements

- Python 3.10+ and Git.
- `uv` is the recommended way to create the virtualenv on this machine.
- A Modal account (free tier works) with an API token.
- One LLM credential: an API key for `anthropic`, `openai`, `google`, `openrouter`, or a
  custom OpenAI-compatible endpoint — or a logged-in `claude` or `codex` CLI session
  (`claude-cli` / `codex-cli` providers).

## Install

```bash
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python -e '.[dev]'
```

## Commands

### `talos setup`

Run once. Asks for the provider kind, a model id (a sensible default is offered per
provider), an API key for API providers (nothing for CLI providers, beyond checking the
binary is on `PATH` and running one trivial call to confirm a logged-in session), a
default mode (`single-shot` or `agentic`) for CLI providers, and a Modal token id/secret
(create one at modal.com/settings/tokens). Every credential is validated with one cheap
call before anything is written. On success it writes `talos.config.json` and
`.talos/secrets.json` (mode 0600, holds only the LLM key; CLI providers store no secret)
and deploys the benchmark app to your Modal account.

### `talos run`

Run per job. Prompts interactively for anything not given as a flag:

- `--challenge` — one of the TIG challenges; the interactive prompt lists them if omitted.
- `--direction` / `--direction-file` — free text describing what to explore; becomes the
  first entry in the job's tacit knowledge.
- `--mode {single-shot,agentic}` — overrides the configured mode for this run; `agentic`
  is only valid for a CLI provider.
- `--budget-usd`, `--budget-hours`, `--budget-iterations`, `--budget-compute-usd` — see
  Budget below.
- `--resume <job_id>` — reloads `runs/<job_id>/job.json` and `state.json` and continues a
  job that was interrupted, cancelled, or failed.
- `--yes` — accept defaults instead of prompting; still requires at least one budget
  dimension unless one is passed as a flag.

Once running, the terminal streams one line per event and, after every finished
iteration, a status line with the job id, iteration, best delta versus baseline, LLM and
Modal spend, and wall-clock time left. `Ctrl-C` stops cleanly at the next safe point and
packages the best candidate found so far.

There is also a hidden `--fake` flag: `talos run --challenge knapsack --direction "..."
--budget-iterations 5 --yes --fake` runs the whole loop against a canned in-process
provider and benchmark — no config file, no network, no Modal, no LLM credential. It
exists to demo and smoke-test the CLI on a machine with none of the above configured.

### Agentic mode

`--mode agentic` (CLI providers only) hands each iteration to a headless `claude` or `codex`
session in a throwaway worktree outside `runs/`, instead of asking an API for one edit.

With `claude-cli`, Talos writes a `.claude/settings.json` that the CLI enforces: reads, `Glob`
and `Grep` are allowed only over `algorithm/**`, `CHALLENGE.md`, `tacit.md`, `AGENTS.md` and
`.talos/hypothesis.json`; the only writes allowed are `Edit` on the algorithm files and on the
hypothesis file; the only command allowed is `talos compile`; `WebFetch`, `WebSearch`, `Write`
and the usual network/shell escapes are denied; and `defaultMode` is `dontAsk`, so any tool not
on the allow list is refused outright rather than prompted for. There is no network access at
the tool level. The child process gets an environment allowlist rather than your environment:
no LLM keys, no Modal tokens.

`codex-cli` is opt-in. Codex ignores `.claude/settings.json`, and its own `--sandbox
workspace-write` restricts writes only — under it the agent can execute arbitrary
agent-authored commands on your machine and read any file you can read. Talos therefore refuses
to start an agentic codex run unless you set `TALOS_ALLOW_CODEX_AGENTIC=1`. In both modes an
edit outside the algorithm files fails the iteration.

### `talos compile`

`talos compile --challenge <name> --dir <path>` (default `--dir algorithm`) uploads the
`.rs`/`.cu` files under `<path>` to Modal, compiles them, and prints the compiler output,
exiting non-zero on failure. It is the command agentic mode uses to check its own edits.

### `talos status`

Lists every job under `runs/`, one line each, with status, iteration, and spend.

## Where results land

Each job gets `runs/<job_id>/`: `job.json` (immutable inputs), `state.json` (mutable
progress, including the best candidate found), `timeline.jsonl` (one JSON event per
line), `tacit.md` (the direction and anything learned), and `iterations/<n>/` (each
candidate's files and hypothesis). On exit, `runs/<job_id>/package/` holds: the best
algorithm's files, `diff_vs_baseline.patch`, `scores.md` (per-nonce tables for baseline
and candidate on training and held-out nonces), `hypotheses.md` (the full log with
outcomes), `evidence_draft.md` (a partially filled-in TIG advance-evidence template), and
`README.md` explaining how to submit — also zipped as `package.zip`.

The measured baseline is cached separately from the run directory, keyed by challenge,
monorepo ref, algorithm, nonce sets, fuel, and hardware class: real runs share
`~/.talos/baselines/<challenge>/<key>.json` across jobs, while `--fake` runs (which use no
real network or Modal) keep theirs under `runs/<job_id>/baseline_cache/<challenge>/<key>.json`
instead.

## Budget

At least one of `--budget-usd`, `--budget-hours`, or `--budget-iterations` must be set
(directly, or through the wizard); zero is a valid, real cap, not "unset". Compute spend is
always capped separately in dollars — `--budget-compute-usd`, defaulting to $20 when `--yes`
is passed without it — and is checked before every compute call regardless of the other
dimensions. The wall-clock budget (`--budget-hours`) counts elapsed time from the job's
start, including time spent resumed. Compute spend shown in status lines and the final
report is an **estimate**, computed from measured container seconds times list prices in
a table shipped with Talos — not a billed amount.

## Live smoke test

`tests/test_live.py` (marker `live`, excluded from `make check`) compiles and scores the
real mainnet top algorithm for one challenge on your deployed Modal app:

```bash
TALOS_LIVE_CHALLENGE=knapsack .venv/bin/pytest -m live tests/test_live.py -s
```

Run this once yourself after `talos setup`, before trusting a real run — it has not been
run by the developers of this repository (no Modal or LLM credentials were available in
the development environment).

## How it works

Talos measures the current top-adoption mainnet algorithm as its baseline, then loops: an
LLM proposes a hypothesis and an edit, `talos compile` + the Modal bench harness score it
against training nonces, promising candidates are confirmed against held-out nonces, and
the loop stops when a candidate beats baseline on both, or the budget runs out. See
[docs/ai/specs/2026-09-11-talos-design.md](docs/ai/specs/2026-09-11-talos-design.md)
for the full design.

## Licence

GPLv3 (see `LICENSE`). `talos/search_replace.py` is lifted from
[tig-foundation/prometheus-swarm](https://github.com/tig-foundation/prometheus-swarm)
(its file scripts/search_replace.py), also GPLv3; `talos/agentic.py`'s sandbox settings mirror
Prometheus's `_build_sandbox_settings` design without copying its code.
