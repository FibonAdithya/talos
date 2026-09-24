# Talos

Single-user autoresearch for [TIG](https://tig.foundation). You pick a TIG challenge and
describe what to explore. An LLM then repeatedly edits the current top mainnet algorithm for
that challenge, and each candidate is compiled and scored with TIG's own `tig-runtime` and
`tig-verifier` on your own Modal or C3 account. The run stops when a candidate beats the
baseline on both training and held-out nonces, or when the budget runs out. The best
candidate is written to a local, submit-ready package.

Talos never submits to TIG for you, and LLM-authored code never runs on your machine (with
one opt-in exception: [agentic codex](#agentic-mode)).

## Contents

- [How it works](#how-it-works)
- [Try it with no accounts](#try-it-with-no-accounts)
- [Setup](#setup)
- [Running on Windows](#running-on-windows)
- [Running a job](#running-a-job)
- [Command reference](#command-reference)
- [Agentic mode](#agentic-mode)
- [Where results land](#where-results-land)
- [Budget](#budget)
- [Live smoke test](#live-smoke-test)
- [Development](#development)
- [Licence](#licence)

## How it works

`talos run` measures the current top-adoption mainnet algorithm once as the baseline. Then,
until the budget runs out, it asks the LLM for an edit, compiles it on the compute backend,
and scores it on training nonces. A candidate that beats the baseline on training is scored
again on held-out nonces, and wins only if it still beats the baseline there. Baseline and
candidates always run on the same nonces, fuel, hyperparameters and hardware class, and the
LLM never sees the `rand_hash` that seeds the nonces.

[docs/architecture.md](docs/architecture.md) has the full flowchart, the guards between
build and scoring, the agentic sandbox and the baseline cache.
[docs/compute-backends.md](docs/compute-backends.md) covers how Modal and C3 are used, C3
timings and the C3 dev images.

## Try it with no accounts

You can run the whole loop end to end before creating any account. The hidden `--fake` flag
replaces the LLM, mainnet and the compute backend with in-process stand-ins: no config file,
no network, no credentials.

```bash
git clone https://github.com/FibonAdithya/talos.git
cd talos
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python -e .
.venv/bin/talos run --challenge knapsack --direction "demo" --budget-iterations 3 --yes --fake
```

It finishes in under a second and prints the same event stream a real run does, ending with:

```
Status: won (beat baseline on training and held-out nonces)
Best delta vs baseline: +1.000%
LLM spend: $0.02   Compute spend (estimated): $1.28
Package: .../runs/<job_id>/package
```

The spend figures in a fake run are made up by the stand-ins. Delete `runs/` afterwards if
you do not want the demo job listed by `talos status`.

## Setup

### 1. Prerequisites

- Linux, macOS or Windows. CI runs the test suite on all three. Compiling and scoring happen
  on the compute backend, so nothing Rust- or CUDA-related is needed locally. The `local`
  backend needs Docker (and, for GPU challenges, the NVIDIA container toolkit); the compiler
  still runs inside the container.
- Python 3.10 or newer, and Git.
- [`uv`](https://docs.astral.sh/uv/) to create the virtualenv (recommended; plain `pip`
  works too).
- One compute backend account (step 3).
- One LLM credential (step 4).

### 2. Install

```bash
git clone https://github.com/FibonAdithya/talos.git
cd talos
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python -e .
source .venv/bin/activate      # puts `talos` on PATH for this shell
talos --help
```

These commands are for bash or zsh. On Windows, follow
[Running on Windows](#running-on-windows) instead: it has every step as PowerShell
commands.

`talos setup` and `talos run` read and write `talos.config.json`, `.talos/secrets.json` and
`runs/` in the **current directory**. Always run Talos from the same directory, normally
the repository root.

### 3. Pick a compute backend

Candidates are compiled and scored on one of three backends. You choose one in `talos setup`.

| Backend | What you need before setup | Cost per iteration |
|---|---|---|
| `modal` (default) | A [Modal](https://modal.com) account (the free tier works) and an API token created at modal.com/settings/tokens. Keep the token id and secret to hand. | Container seconds only; no fixed per-job overhead. |
| `c3` | With a C3 API key (`c3 apikey create`): nothing to install — Talos talks to C3 over HTTPS. Without a key: the `c3` CLI ([cthree.cloud](https://cthree.cloud)) installed and logged in with `c3 login`. Either way, credit on the account; top up with `c3 topup`. | One batch job of about 12 minutes before the first nonce is scored. See [docs/compute-backends.md](docs/compute-backends.md#c3-timings). |
| `local` | Docker running on this machine. The first run per challenge pulls the 13 GB dev image and does one warm-up build (MEASURED 2026-09-23: about 13 minutes on a 16-core machine with the image already pulled). Every iteration then builds the candidate (MEASURED 2026-09-23: 14 to 15 minutes for one build and 4 nonces over two runs, of which the nonces took under 2 s each): the dev image re-instruments every dependency on each build, so this is not incremental. GPU challenges need the NVIDIA container toolkit and are unverified; see [docs/compute-backends.md](docs/compute-backends.md#local-backend). | Nothing. The compute budget question is skipped. |

### 4. Pick an LLM provider

| Provider | Credential | Environment variable used if `.talos/secrets.json` has no key | Default model |
|---|---|---|---|
| `anthropic` | API key | `ANTHROPIC_API_KEY` | `claude-opus-5` |
| `openai` | API key | `OPENAI_API_KEY` | `gpt-5` |
| `google` | API key | `GEMINI_API_KEY` | `gemini-2.5-pro` |
| `openrouter` | API key | `OPENROUTER_API_KEY` | `anthropic/claude-opus-5` |
| `custom` | API key and base URL of an OpenAI-compatible endpoint | `TALOS_CUSTOM_API_KEY` | none |
| `claude-cli` | A logged-in `claude` CLI on `PATH` | none | `claude-opus-5` |
| `codex-cli` | A logged-in `codex` CLI on `PATH` | none | the first model `codex debug models` lists |

Only the CLI providers (`claude-cli`, `codex-cli`) support [agentic mode](#agentic-mode).
CLI providers bill through your CLI subscription rather than per API call, so the `talos run`
wizard asks them for an iteration budget instead of a dollar budget.

### 5. Run `talos setup`

```bash
talos setup
```

It asks, in order:

1. **Compute backend**: `modal`, `c3` or `local`.
2. **Provider**: one of the kinds in the table above.
3. **Model**: press Enter for the default. For `codex-cli` it first prints the models your
   login accepts; for `claude-cli` an alias such as `fable`, `opus` or `sonnet` works.
4. **API base URL**: `custom` provider only.
5. **API key**: API providers only. Each character you type or paste shows as `*`, so you can
   see that a paste arrived; the key itself is never shown.
6. **Mode** (`single-shot` or `agentic`): CLI providers only. This is the default; `talos
   run --mode` overrides it per job.
7. **Modal token id and secret**: `modal` backend only; the secret shows as `*`.
8. **C3 API key**: `c3` backend only; shows as `*`. Leave it blank to use your `c3 login`
   session. With it blank, a `C3_API_KEY` environment variable is used if set.
9. **CPUs and memory for the local container**: `local` backend only. The defaults are what
   Docker reports: every CPU the daemon has and its memory minus 4 GiB. On Docker Desktop
   those are the VM's figures (Settings > Resources), not the machine's, and values above them
   are refused, because Docker rejects a container with more CPUs than the daemon has. Both
   are part of the local baseline cache key: change them and local baselines are measured
   again.

It then checks everything before writing anything:

- the provider, with one cheap call (for CLI providers: that the binary is on `PATH` and a
  trivial call succeeds);
- on `modal`: sets your Modal token and deploys the benchmark app to your account;
- on `c3`: runs `c3 whoami` and `c3 balance` (with the API key, if you gave one), and warns
  if the balance is below £1;
- on `local`: `docker info` is run before the limits are asked (its CPU and memory figures
  are their defaults and ceilings), and the summary says whether the NVIDIA runtime is present.

On success it prints ``Setup complete. Run `talos run` to start a job.`` and writes:

- `talos.config.json`;
- `.talos/secrets.json` (mode 0600), holding only the LLM API key and the C3 API key, if you
  gave either. If you gave neither, it deletes any `.talos/secrets.json` left by an earlier
  setup.

If any check fails it prints the reason, exits non-zero, and writes neither file.

Run `talos setup` again to change provider, model or backend. On the Modal backend, also
run it again after pulling a new version of Talos: the deployed app must match the client,
because the score function's arguments change between versions. A client that reaches an
older deploy stops with a message naming `talos setup`. The C3 backend ships its code with
each job, so it needs no redeploy.

### 6. Before your first real run

Run the [live smoke test](#live-smoke-test) for your backend once. It spends a small amount
of real compute and confirms the backend can compile and score the real mainnet algorithm.

## Running on Windows

Talos runs natively on Windows; WSL is not needed. (Inside WSL, follow the Linux instructions
instead.) CI runs the full test suite on `windows-latest` for every pull request. No
maintainer has run a real job end to end on a Windows machine yet, so if something fails
there, please open an issue with the command and its output.

Every block below is for PowerShell and can be pasted whole. The commands call
`.venv\Scripts\talos.exe` directly, so they work without activating the virtualenv and
without changing PowerShell's script execution policy.

### 1. Install Git and uv

```powershell
winget install --id Git.Git -e --source winget
winget install --id astral-sh.uv -e --source winget
```

Close PowerShell and open a new window, so that `git` and `uv` are on `PATH`. You do not
need to install Python: `uv venv` in the next step downloads it if the machine has none.

If `winget` is not available, install [Git for Windows](https://git-scm.com/download/win)
from its installer, and `uv` with:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Install Talos

```powershell
git clone https://github.com/FibonAdithya/talos.git
cd talos
uv venv --python 3.10 .venv
uv pip install --python .venv\Scripts\python.exe -e .
.venv\Scripts\talos.exe --help
```

Run every later block from this `talos` folder.

### 3. Try it with no accounts

```powershell
.venv\Scripts\talos.exe run --challenge knapsack --direction "demo" --budget-iterations 3 --yes --fake
```

See [Try it with no accounts](#try-it-with-no-accounts) for what it prints.

### 4. Configure

Pick a [compute backend](#3-pick-a-compute-backend) and an
[LLM provider](#4-pick-an-llm-provider), then:

```powershell
.venv\Scripts\talos.exe setup
```

It asks the same questions on every OS; they are listed under
[Run `talos setup`](#5-run-talos-setup). On the C3 backend, give it a C3 API key (see
[What differs on Windows](#what-differs-on-windows)).

### 5. Run the live smoke test once

For the Modal backend:

```powershell
uv pip install --python .venv\Scripts\python.exe pytest
$env:TALOS_LIVE_CHALLENGE = "knapsack"
.venv\Scripts\python.exe -m pytest -m live tests/test_live.py -s
```

For the C3 backend:

```powershell
uv pip install --python .venv\Scripts\python.exe pytest
$env:TALOS_LIVE_BACKEND = "c3"
.venv\Scripts\python.exe -m pytest -m live tests/test_live.py -k c3 -s
```

Both spend a small amount of real compute; see [Live smoke test](#live-smoke-test).

### 6. Run a job

Interactively:

```powershell
.venv\Scripts\talos.exe run
```

Or with every answer given as a flag:

```powershell
.venv\Scripts\talos.exe run --challenge knapsack --direction "Try a tighter upper bound in the branch-and-bound pruning" --budget-usd 20 --budget-hours 4 --budget-compute-usd 10 --yes
```

List jobs, and resume one by its id:

```powershell
.venv\Scripts\talos.exe status
.venv\Scripts\talos.exe run --resume 20260917-101006-knapsack
```

[Running a job](#running-a-job) and the [command reference](#command-reference) apply
unchanged.

### Optional: activate the virtualenv

Activation lets you type `talos` instead of `.venv\Scripts\talos.exe`. The first line allows
the activation script to run in this window only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned -Force
.venv\Scripts\activate
talos --help
```

### What differs on Windows

- **Modal backend**: nothing extra. `talos setup` runs the `modal` client that was installed
  into the virtualenv, whether or not the virtualenv is activated.
- **C3 backend**: use a C3 API key. Create one on the
  [C3 dashboard settings page](https://cthree.cloud/dashboard/settings) and paste it at the
  **C3 API key** prompt in `talos setup`. With a key, Talos talks to C3 over HTTPS and nothing
  from C3 needs installing. This path is covered by unit tests, and one real C3 job ran over
  it from Linux on 2026-09-18 (MEASURED, `docs/ai/specs/2026-09-17-c3-mcp-transport-design.md`
  §8). None has run from Windows, so run the smoke test in step 5 before a real run. The `c3`
  CLI path (key left blank) has not been tried on Windows: C3's documented installer is a
  shell script, and Windows cannot mark job.sh executable on disk, so it is unknown whether
  a job uploaded by the CLI from Windows would start.
- **CLI providers**: `npm` installs `claude` and `codex` as `.cmd` wrappers. Talos finds them
  through `PATH`, so `claude --version` (or `codex --version`) must work in the same window
  you run Talos from. [Agentic mode](#agentic-mode) uses the same sandbox as on other
  systems. The codex opt-in is:

  ```powershell
  $env:TALOS_ALLOW_CODEX_AGENTIC = "1"
  ```

- **Secrets**: Windows has no owner-only file permissions, so `.talos\secrets.json` is not
  restricted to your user the way it is on Linux and macOS (mode `0600`). Keep the repository
  in a folder only you can read.
- **Baseline cache**: `~/.talos/baselines/` is `$env:USERPROFILE\.talos\baselines\`.

### Using `cmd.exe` instead of PowerShell

The `.venv\Scripts\...` commands are the same. Two things differ:

| PowerShell | `cmd.exe` |
|---|---|
| `$env:VAR = "value"` | `set VAR=value` |
| `.venv\Scripts\activate` | `.venv\Scripts\activate.bat` (no execution policy step) |

### Development on Windows

`make` is not installed by default. This is the [development](#development) install, followed
by the three commands `make check` runs:

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements-dev.txt -e .
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe -m pytest -q -m "not live"
.venv\Scripts\python.exe -m agentify check .
```

## Running a job

### Start

Interactively, answering each prompt:

```bash
talos run
```

Or with every answer given as a flag:

```bash
talos run --challenge knapsack \
  --direction "Try a tighter upper bound in the branch-and-bound pruning" \
  --budget-usd 20 --budget-hours 4 --budget-compute-usd 10 --yes
```

Before the job starts, Talos fetches the challenge's active tracks and fuel from mainnet,
draws the nonces, and prints a line such as:

```
Job 20260917-101006-knapsack: 5 tracks, fuel ..., budget {...}; hyperparameters: 5/5 tracks from mainnet
```

The job id is the start time plus the challenge name.

### Watch

The terminal prints one line per event, prefixed with the time and the iteration number.
Each iteration ends with a summary line: the outcome, then the best delta so far, the spend
and the time left.

```
[10:41:02] #1 trying: bump k (local_search)
[10:49:12] #1 scored +1.000% vs baseline (worst track +1.000%, errors 0.0%, runtime x1.00)
[10:49:12] #1 new best | best +1.000% | llm $0.02 | compute ≈$1.28 | 3.9h left
[10:49:12] #1 confirming on held-out nonces
[10:49:12] #1 won: held-out +1.000% vs baseline (worst track +1.000%)
```

`best` is the best candidate's mean relative delta versus the baseline on training nonces.
`compute ≈` is an estimate (see [Budget](#budget)). The last field is the wall-clock budget
remaining, or `∞ left` without `--budget-hours`. An iteration that did not improve reads
`no improvement (4 in a row)` or `no candidate: did not compile (2 in a row)`; a failed
compile prints the first `error` line of the compiler output. Lines are cut to the terminal
width. `runs/<job_id>/timeline.jsonl` records every event with every field in full,
including hypothesis descriptions and the compiler output.

### Stop

Press `Ctrl-C` once. Talos stops at the next safe point, packages the best candidate found
so far, and prints the final status. The job can be resumed later.

### Resume

```bash
talos status                              # find the job id
talos run --resume 20260917-101006-knapsack
```

A resumed job keeps the provider, model, mode, track, hyperparameters and nonces it started
with, whatever `talos.config.json` says now. A C3 job that was still running when Talos
stopped is reattached rather than paid for twice.

### Submit

When a job ends with `Status: won`, open `runs/<job_id>/package/README.md`. It explains how
to submit the candidate to TIG. `evidence_draft.md` in the same directory is a partly
filled-in advance-evidence template. You submit it yourself; Talos does not.

A job that ends any other way still writes a package with its best candidate, but that
candidate has not beaten the baseline on held-out nonces, and the package README says so.

## Command reference

| Command | What it does |
|---|---|
| `talos setup` | One-time configuration: backend, LLM provider, credentials. Validates them and deploys the Modal app. |
| `talos run` | Starts or resumes a research job and writes its package when it stops. |
| `talos compile` | Compiles a directory of algorithm files on the configured backend and prints the compiler output. Scores nothing. |
| `talos status` | Lists every job under `runs/` with its status, iteration and spend. |

### `talos setup`

No flags. See [Setup, step 5](#5-run-talos-setup) for every prompt and what is written.
Exit code 0 on success, 1 when a credential or backend check fails, 2 for an unknown
backend or provider.

### `talos run`

Prompts for anything not given as a flag, unless `--yes` is passed.

| Flag | Meaning |
|---|---|
| `--challenge NAME` | `satisfiability`, `vehicle_routing`, `knapsack`, `job_scheduling`, `energy_arbitrage` (CPU), or `vector_search`, `hypergraph`, `neuralnet_optimizer` (GPU, on an L40S). The prompt shows an estimated hourly price for the GPU challenges. |
| `--direction TEXT` | What to explore, in free text. It becomes the first entry in the job's `tacit.md`. |
| `--direction-file PATH` | The same, read from a file. Pass one of `--direction` or `--direction-file`, not both. |
| `--track NAME` | Optimise one active track of the challenge instead of all of them (`all` is the default; the prompt lists the tracks). Training scores that track only. When a candidate wins on training, the confirmation scores that track's held-out nonces plus every other track's training nonces, and no other track may get worse. The LLM still sees and may edit every file. |
| `--hyperparameters mainnet\|none` | `mainnet` (the default) runs the baseline and every candidate with the per-track hyperparameters of the baseline algorithm's best-quality mainnet benchmark at the job's fuel, fixed at job start. A track with no such benchmark runs without any. `none` runs every nonce without hyperparameters. The package lists the values and the benchmark they came from. |
| `--mode single-shot\|agentic` | Overrides the configured mode for this job. `agentic` needs a CLI provider and uses roughly 5 to 20 times the tokens of `single-shot`. |
| `--budget-usd N` | LLM spend cap in USD. |
| `--budget-hours N` | Wall-clock cap in hours. |
| `--budget-iterations N` | Iteration cap. |
| `--budget-compute-usd N` | Estimated compute spend cap in USD. |
| `--resume JOB_ID` | Continue a job that was interrupted, cancelled, paused or failed. Cannot change its mode, track or hyperparameters. |
| `--yes` | Accept defaults instead of prompting. At least one of `--budget-usd`, `--budget-hours` or `--budget-iterations` must still be given; compute defaults to $20. |

Without flags, the wizard asks for challenge, direction, an LLM budget (USD for metered
providers, default 20; iterations for CLI providers, default 50), a wall-clock budget
(default 4 hours), a compute budget (default $20), the mode (CLI providers only), the
track, and the hyperparameters (`mainnet` or `none`).

Talos refuses to start, before spending anything, when: there is no API key for the
provider; the model has no entry in the price table and the only budget is `--budget-usd`
(a dollar cap it could not enforce); the mainnet challenge id no longer matches
`talos/challenges.py`; or, on C3, the challenge's dev image tag is not on GHCR.

Exit code 0 when the job ends `won`; 1 when it ends any other way, or when the mainnet or
dev image check stops it; 2 for invalid arguments, a missing config, or a missing API key.

### `talos compile`

```bash
talos compile --challenge knapsack --dir algorithm
```

| Flag | Meaning |
|---|---|
| `--challenge NAME` | Required. |
| `--dir PATH` | Directory whose `.rs` and `.cu` files are compiled (default `algorithm`). |
| `--backend modal\|c3` | Backend to use. Without it: `TALOS_BACKEND`, then `talos.config.json`, then `modal`. |

Prints the last 4000 characters of compiler output. Exit code 0 if the build succeeded, 1 if
it failed, 2 if the directory has no `.rs`/`.cu` files. Agentic mode uses this command to
check its own edits.

On Modal it is one function call. On C3 it is one batch job
([about 12 minutes](docs/compute-backends.md#c3-timings)) whose job directory is
`.talos/compile/`. Each run overwrites that directory, so do not run two `talos compile`
commands at once from the same directory.

### `talos status`

```bash
talos status
```

One line per job under `runs/`:

```
20260917-101006-knapsack: won it=1 llm=$0.02 compute=$1.28
```

Status is one of `queued`, `measuring_baseline`, `researching`, `confirming`, `won`,
`exhausted` (a budget ran out), `cancelled` (Ctrl-C), `paused` (the compute backend was
unreachable) or `failed` (the reason is in `state.json` and the last lines of
`timeline.jsonl`).

## Agentic mode

`--mode agentic` (CLI providers only) hands each iteration to a headless `claude` or `codex`
session in a throwaway worktree outside `runs/`, instead of asking an API for one edit.

With `claude-cli`, the session runs under a sandbox that the CLI enforces: it can read and
edit only the algorithm files and its notes, the only command it can run is
`talos compile`, it has no network tools, and its environment holds no LLM keys or Modal
tokens. The exact rules are in [docs/architecture.md](docs/architecture.md#the-agentic-sandbox).

`codex-cli` has no equivalent sandbox: under it the agent can execute arbitrary
agent-authored commands on your machine and read any file you can read. Talos therefore
refuses to start an agentic codex run unless you opt in:

```bash
export TALOS_ALLOW_CODEX_AGENTIC=1
```

With either CLI, an edit outside the algorithm files fails the iteration.

On the C3 backend, the sandbox has no `talos.config.json` to read, so the configured backend
is passed to it as `TALOS_BACKEND`, and a C3 API key, if you use one, as `C3_API_KEY`. Each
`talos compile` the agent runs from the sandbox is one C3 job
([about 12 minutes](docs/compute-backends.md#c3-timings)). If the agent's 30-minute timeout
kills a sandbox compile while its C3 job is still running, the job is not cancelled: it runs
on to its own time limit and bills for it, which bounds the cost but does not avoid it.

## Where results land

Each job gets a directory under `runs/`:

| Path | Contents |
|---|---|
| `runs/<job_id>/job.json` | The job's fixed inputs: challenge, direction, provider, model, mode, budget, nonces, fuel, track, baseline algorithm, hyperparameters. Written once. |
| `runs/<job_id>/state.json` | Progress: status, iteration, spend, the best candidate, the hypothesis log. |
| `runs/<job_id>/timeline.jsonl` | One JSON event per line; the same events the terminal prints. |
| `runs/<job_id>/tacit.md` | Your direction, plus lessons the LLM distilled from failed attempts. |
| `runs/<job_id>/baseline/` | The baseline algorithm's files and its per-nonce results. |
| `runs/<job_id>/iterations/<n>/` | Each candidate's files and its hypothesis. |
| `runs/<job_id>/best/` | The best candidate's files so far. |
| `runs/<job_id>/package/` | The hand-back package, written when the job stops. |
| `runs/<job_id>/package.zip` | The same package, zipped. |

The package directory holds:

| File | Contents |
|---|---|
| the best algorithm's files | The candidate to submit. |
| `diff_vs_baseline.patch` | The candidate as a patch against the baseline. |
| `scores.md` | Per-nonce tables for baseline and candidate on training and held-out nonces. A focused job (`--track`) adds a regression-guard table for the other tracks. |
| `hypotheses.md` | The full hypothesis log with outcomes. |
| `evidence_draft.md` | A partly filled-in TIG advance-evidence template. |
| `hyperparameters.json` | The per-track hyperparameters, when the job used them. |
| `README.md` | How to submit, with a Hyperparameters section when the job used them. |

Benchmark the submitted algorithm with the same hyperparameters: the measured improvement
holds only with them.

On the C3 backend each iteration also gets a `runs/<job_id>/c3/<n>/` directory, described in
[docs/compute-backends.md](docs/compute-backends.md#c3-job-directories). The measured baseline
is cached outside the run directory, in `~/.talos/baselines/`; see
[docs/architecture.md](docs/architecture.md#the-baseline-cache).

## Budget

- At least one of `--budget-usd`, `--budget-hours` or `--budget-iterations` must be set,
  directly or through the wizard. Zero is a valid, real cap, not "unset".
- Compute spend is always capped separately by `--budget-compute-usd`, which defaults to $20
  when `--yes` is passed without it. It is checked before every compute call, whatever the
  other caps are.
- The wall-clock budget (`--budget-hours`) counts elapsed time from the job's start,
  including time spent resumed.

Every spend figure in the status lines and the final report is an **estimate**, not a billed
amount. Compute spend is measured running seconds times list prices in a table shipped with
Talos. LLM spend is measured tokens times a price table. A model with no entry in that table
is shown as `unpriced`, and `--budget-usd` cannot be enforced for it: add `--budget-hours`
or `--budget-iterations`.

Compute spent by `talos compile` from the agentic sandbox is not counted against
`--budget-compute-usd`. On C3 each of those is one job with the fixed overhead described
under [C3 timings](docs/compute-backends.md#c3-timings), billed at the profile's rate. A
focused job's confirmation scores the other tracks' training nonces too, about a minute more
per winning iteration on C3 (ESTIMATE, unverified).

## Live smoke test

`tests/test_live.py` (marker `live`, excluded from `make check`) compiles and scores the
real mainnet top algorithm for one challenge on your deployed Modal app:

```bash
TALOS_LIVE_CHALLENGE=knapsack .venv/bin/pytest -m live tests/test_live.py -s
```

For the C3 backend, a separate test runs one real C3 job end to end. It needs `c3 login` (or
`C3_API_KEY` set) and a few pence of credit (MEASURED 2026-09-15: £0.02 billed, 12 min 20 s
wall clock, job `SUCCEEDED`, `1 passed`; the breakdown is under
[C3 timings](docs/compute-backends.md#c3-timings)):

```bash
TALOS_LIVE_BACKEND=c3 .venv/bin/pytest -m live tests/test_live.py -k c3 -s
```

For the local backend, a third test runs one real job in Docker on this machine. It costs
time, not money: the first run pulls the image and does the warm-up build (MEASURED
2026-09-23: about 13 minutes before the job, image already pulled), a later run finds
everything in place and takes only the job (MEASURED 2026-09-23, two runs: `1 passed` in
14 min 7 s and in 15 min 3 s, `prepare_s: 1`, `job_s: 844` and `899`):

```bash
TALOS_LIVE_BACKEND=local .venv/bin/pytest -m live tests/test_live.py -k local -s
```

Both need `pytest`: install it with `uv pip install --python .venv/bin/python pytest`, or use
the development install below.

Run the one for your backend once after `talos setup`, before trusting a real run. The
maintainers ran the C3 test on 2026-09-15. They have not run the Modal test: no Modal or LLM
credentials were available in the development environment.

## Development

The gate needs Python 3.11 or newer, because the pinned `agentify` in `requirements-dev.txt`
does. CI uses 3.12.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-dev.txt -e .
make check PYTHON=.venv/bin/python
```

`make check` runs ruff, pytest without the `live` marker, and the agentify contract check.
It is the same command CI runs. See `AGENTS.md` for the invariants a change must keep.

## Licence

GPLv3 (see `LICENSE`). `talos/search_replace.py` is lifted from
[tig-foundation/prometheus-swarm](https://github.com/tig-foundation/prometheus-swarm)
(its file scripts/search_replace.py), also GPLv3; `talos/agentic.py`'s sandbox settings mirror
Prometheus's `_build_sandbox_settings` design without copying its code.
