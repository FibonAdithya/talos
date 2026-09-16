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

## Compute backend

Talos compiles and scores candidates on one of two backends, chosen once at `talos setup`:

- **Modal** (default): a Modal account and API token, set up as in Requirements above.
- **C3** ([cthree.cloud](https://cthree.cloud)): a logged-in `c3` CLI (`c3 login`) and
  credit on the account; `talos setup` warns if the balance is below £1.

On C3, one iteration is one batch job with about 12 minutes of fixed overhead before any
nonce is scored — MEASURED 2026-09-14 on a spike run (knapsack, 4 vCPU): about 4 minutes
to script start, 466 seconds to build the candidate, then 1 to 2 seconds per nonce at
mainnet fuel. The release smoke test on 2026-09-15 (MEASURED, knapsack, 4 vCPU, two
training and two held-out nonces) took 12 min 20 s from submission to result: about 2 minutes
queued, 470 seconds to build, 1.3 to 1.7 seconds per nonce; the client's cost estimate was
$0.027 and the account balance fell by £0.02. Modal has no equivalent per-job overhead.

C3 pulls only public Docker Hub images, never GHCR. Maintainers mirror the TIG dev images
to `docker.io/fibonadithya/tig-<challenge>-dev:<tag>` with `make mirror-images`
(`scripts/mirror_images.sh`); this only needs running once per `DEV_IMAGE_TAG`, but a tag
bump means re-running it before release. Tags currently mirrored: knapsack 0.0.7 (verified
2026-09-15). Users never touch GHCR or the mirror script themselves. `talos run` on C3
checks the Hub tag before the baseline is measured and fails with a mirror hint if it is
missing. Maintainers can point at a test mirror instead of the real one with
`TALOS_IMAGE_NAMESPACE`.

The mirror script needs `talos` importable, which the system `python3` does not have: from
a fresh shell run `make mirror-images PYTHON=.venv/bin/python`, or activate the venv first
and run plain `make mirror-images` (`Makefile`'s `PYTHON ?= python3` otherwise picks the
system interpreter and fails with "No module named talos").

## Install

```bash
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python -e '.[dev]'
```

## Commands

### `talos setup`

Run once. Asks first for the compute backend (`modal` or `c3`), then the provider kind, a
model id (a sensible default is offered per provider; for `codex-cli` the wizard reads
the catalog from `codex debug models`, prints the models your login accepts, and offers
the first as the default; for `claude-cli` an alias such as `fable` works), an API key
for API providers
(nothing for CLI providers, beyond checking the binary is on `PATH` and running one
trivial call to confirm a logged-in session), and a default mode (`single-shot` or
`agentic`) for CLI providers. The Modal token id/secret prompt (create one at
modal.com/settings/tokens) appears only when the backend is `modal`; for `c3` there is no
token prompt, but setup runs `c3 whoami` to confirm a logged-in session and warns if the
credit balance is below £1. Every credential is validated with one cheap call before
anything is written. On success it writes `talos.config.json` and `.talos/secrets.json`
(mode 0600, holds only the LLM key; CLI providers store no secret), and, for the Modal
backend, deploys the benchmark app to your Modal account.

### `talos run`

Run per job. Prompts interactively for anything not given as a flag:

- `--challenge` — one of the TIG challenges; the interactive prompt lists them if omitted.
- `--direction` / `--direction-file` — free text describing what to explore; becomes the
  first entry in the job's tacit knowledge.
- `--mode {single-shot,agentic}` — overrides the configured mode for this run; `agentic`
  is only valid for a CLI provider.
- `--track <name>` — one active track of the challenge to optimise (the interactive prompt
  lists them; default all). Training scores that track only. When a candidate wins on
  training, the confirmation job scores the track's held-out nonces plus every other track's
  training nonces as a regression guard: no other track may get worse. The model still sees
  and may edit every file; the flag narrows what is scored and what it is told to target.
- `--budget-usd`, `--budget-hours`, `--budget-iterations`, `--budget-compute-usd` — see
  Budget below.
- `--resume <job_id>` — reloads `runs/<job_id>/job.json` and `state.json` and continues a
  job that was interrupted, cancelled, or failed.
- `--yes` — accept defaults instead of prompting; still requires at least one budget
  dimension unless one is passed as a flag.

Once running, the terminal streams one line per event and, after every finished
iteration, a status line with the job id, iteration, best delta versus baseline, LLM and
compute spend, and wall-clock time left. `Ctrl-C` stops cleanly at the next safe point and
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

On the C3 backend, the sandbox has no `talos.config.json` to read, so the configured
backend is passed to it as `TALOS_BACKEND`; each `talos compile` the agent runs from the
sandbox is one C3 job of about 12 minutes. If the agent's 30-minute timeout kills a sandbox
compile while its C3 job is still running, the job is not cancelled: it runs on to its own
time limit and bills for it, which bounds the cost but does not avoid it.

### `talos compile`

`talos compile --challenge <name> --dir <path>` (default `--dir algorithm`) uploads the
`.rs`/`.cu` files under `<path>` and compiles them, and prints the compiler output, exiting
non-zero on failure. It is the command agentic mode uses to check its own edits. It runs
on the configured backend: `--backend`, then `TALOS_BACKEND` (set for the agentic
sandbox), then `talos.config.json`, then `modal`. On Modal this is one function call; on
C3 it is one batch job of about 12 minutes, written to a fixed job directory that every
`talos compile` run in the same directory overwrites — concurrent `talos compile` runs in
one directory are unsupported.

### `talos status`

Lists every job under `runs/`, one line each, with status, iteration, and spend.

## Where results land

Each job gets `runs/<job_id>/`: `job.json` (immutable inputs), `state.json` (mutable
progress, including the best candidate found), `timeline.jsonl` (one JSON event per
line), `tacit.md` (the direction and anything learned), and `iterations/<n>/` (each
candidate's files and hypothesis). On exit, `runs/<job_id>/package/` holds: the best
algorithm's files, `diff_vs_baseline.patch`, `scores.md` (per-nonce tables for baseline
and candidate on training and held-out nonces; a focused job adds a regression-guard table
for the other tracks), `hypotheses.md` (the full log with
outcomes), `evidence_draft.md` (a partially filled-in TIG advance-evidence template), and
`README.md` explaining how to submit — also zipped as `package.zip`.

On the C3 backend, each job also gets `runs/<job_id>/c3/<n>/` (and
`runs/<job_id>/c3/baseline/` for the baseline measurement): the files C3 generates and
uploads for that job — a .c3 config, a job.sh entrypoint, a payload.json, and copies of
the modules the container needs — plus the pulled artifacts, which land under
`runs/<job_id>/c3/<n>/<c3-job-id>/artifacts/`, where `<c3-job-id>` is C3's own job id from
`c3 deploy`, distinct from the Talos `<job_id>`. The payload carries the job's rand hash
and is uploaded to C3's workspace store as part of the job directory.

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

Compute spent by `talos compile` from the agentic sandbox is not counted against
`--budget-compute-usd`; on C3 each of those is one job (about 12 minutes of overhead,
MEASURED 2026-09-14) billed at the profile's rate. A focused job's confirmation scores the
other tracks' training nonces too, about a minute more per winning iteration on C3
(ESTIMATE, unverified).

## Live smoke test

`tests/test_live.py` (marker `live`, excluded from `make check`) compiles and scores the
real mainnet top algorithm for one challenge on your deployed Modal app:

```bash
TALOS_LIVE_CHALLENGE=knapsack .venv/bin/pytest -m live tests/test_live.py -s
```

For the C3 backend, a separate test runs one real C3 job end to end; it needs `c3 login`
and a few pence of credit (MEASURED 2026-09-15: £0.02 billed, 12 min 20 s wall clock, job
`SUCCEEDED`, `1 passed`):

```bash
TALOS_LIVE_BACKEND=c3 .venv/bin/pytest -m live tests/test_live.py -k c3 -s
```

Run these once yourself after `talos setup`, before trusting a real run. The C3 test was
run by the maintainers on 2026-09-15 (see above); the Modal test has not been run by the
developers of this repository (no Modal or LLM credentials were available in the
development environment).

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
