# Talos

Single-user autoresearch for [TIG](https://tig.foundation). You pick a TIG challenge and
describe what to explore. An LLM then repeatedly edits the current top mainnet algorithm for
that challenge, and each candidate is compiled and scored with TIG's own `tig-runtime` and
`tig-verifier` on your own Modal or C3 account, or in Docker on your own machine. The run
stops when a candidate beats the
baseline on both training and held-out nonces, or when the budget runs out. The best
candidate is written to a local, submit-ready package.

Talos never submits to TIG for you, and LLM-authored code never runs outside a container.
Only the `local` backend runs that container on your machine, with networking off,
capabilities dropped, and CPU and memory limits; the one opt-in exception is
[agentic codex](#agentic-mode).

## Contents

- [How it works](#how-it-works)
- [Prerequisites](#prerequisites)
- [Try it with no accounts](#try-it-with-no-accounts)
- [Setup](#setup)
- [Running on Windows](#running-on-windows)
- [Running a job](#running-a-job)
- [Command reference](#command-reference)
- [Agentic mode](#agentic-mode)
- [Where results land](#where-results-land)
- [Budget](#budget)
- [Live smoke tests (maintainers)](#live-smoke-tests-maintainers)
- [Development](#development)
- [Licence](#licence)

## How it works

`talos run` measures the current top-adoption mainnet algorithm once as the baseline. Then,
until the budget runs out, it asks the LLM for an edit, compiles it on the compute backend,
and scores it on training nonces. A candidate that beats the baseline on training is scored
again on held-out nonces, and wins only if it still beats the baseline there. Baseline and
candidates always run on the same nonces, fuel and hardware class, and on the same
hyperparameters except in a `--start template` job, whose candidates run without them (see
[From-scratch start and annealing search](docs/architecture.md#from-scratch-start-and-annealing-search)).
The LLM never sees the `rand_hash` that seeds the nonces.

[docs/architecture.md](docs/architecture.md) has the full flowchart, the guards between
build and scoring, the agentic sandbox and the baseline cache.
[docs/compute-backends.md](docs/compute-backends.md) covers how Modal, C3 and local Docker
are used, C3 and local timings, and the dev images.

## Prerequisites

- Linux, macOS or Windows. CI runs the test suite on all three. Compiling and scoring happen
  on the compute backend, so nothing Rust- or CUDA-related is needed locally. The `local`
  backend needs Docker (and, for GPU challenges, the NVIDIA container toolkit); the compiler
  still runs inside the container.
- Python 3.10 or newer, with `pip` and the `venv` module, and Git. On Debian and Ubuntu,
  `venv` is a separate package: `sudo apt install python3-venv`.
- For a real run, not the no-account demo below: one compute backend account
  ([Setup, step 2](#2-pick-a-compute-backend)) and one LLM credential
  ([Setup, step 3](#3-pick-an-llm-provider)).

## Try it with no accounts

`talos demo` runs a scripted job end to end before you create any account: no config file,
no network, no credentials and no spend. It uses the real research loop with a scripted LLM
and an in-process stand-in for the compute backend.

```bash
git clone https://github.com/FibonAdithya/talos.git
cd talos
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/talos demo
```

It takes a few seconds (it pauses 0.4 s after each event line; `--delay 0` removes the
pause) and prints the same event stream a real run does. The script shows, in order, a
compile failure and its fix, two iterations with no gain, a false positive (a candidate that
wins on training and loses on the held-out nonces) and a win. It ends with:

```
Status: won (beat baseline on training and held-out nonces)
Best delta vs baseline: +4.000%
LLM spend: $0.00   Compute spend (estimated): $0.00
Package: .../runs/demo-<timestamp>-knapsack/package
You can find the code to submit at .../runs/demo-<timestamp>-knapsack/package/submission, fill out the template README.md to make it ready to submit to The Innovation Game!
```

It then lists every file in the package with one line on what it holds, prints
`talos report <job_id>` for the one-page view, and ends with
`Next: talos setup, then talos doctor`. All figures are scripted, so spend is $0.00. The
demo job is written to `runs/demo-<timestamp>-<challenge>/` and `talos status` lists it;
delete that folder if you do not want it listed. The hidden `--fake` flag of `talos run`
is for the tests and is not part of this path.

## Setup

### 1. Install

```bash
git clone https://github.com/FibonAdithya/talos.git
cd talos
python3 -m venv .venv
.venv/bin/pip install -e .
source .venv/bin/activate      # puts `talos` on PATH for this shell
talos --help
```

These commands are for bash or zsh. On Windows, follow
[Running on Windows](#running-on-windows) instead: it has every step as PowerShell
commands.

`talos setup` and `talos run` read and write `talos.config.json`, `.talos/secrets.json` and
`runs/` in the **current directory**. Always run Talos from the same directory, normally
the repository root.

### 2. Pick a compute backend

Candidates are compiled and scored on one of three backends. You choose one in `talos setup`.

| Backend | What you need before setup | Cost per iteration |
|---|---|---|
| `modal` (default) | A [Modal](https://modal.com) account (the free tier works) and an API token created at modal.com/settings/tokens. Keep the token id and secret to hand. | Container seconds only; no fixed per-job overhead. |
| `c3` | With a C3 API key (`c3 apikey create`): nothing to install — Talos talks to C3 over HTTPS. Without a key: the `c3` CLI ([cthree.cloud](https://cthree.cloud)), version 0.20.0 or newer, installed and logged in with `c3 login` (older releases refuse the GHCR dev image before it reaches C3; check with `c3 version`). Either way, credit on the account; top up with `c3 topup`. | One batch job of about 12 minutes before the first nonce is scored. See [docs/compute-backends.md](docs/compute-backends.md#c3-timings). |
| `local` | Docker running on this machine. The first run per challenge pulls the 13 GB dev image and does one warm-up build (MEASURED 2026-09-25: 90 s including the clone, on a 16-core machine with the image already pulled). Every iteration then builds the candidate alone in its crate (MEASURED 2026-09-25: 3 minutes for one knapsack build and 4 nonces of under 2 s each; 16.5 minutes for job_scheduling, whose nonces take 150 to 165 s each): the dev image re-instruments every dependency on each build, so this is not incremental. GPU challenges need the NVIDIA container toolkit and are unverified; see [docs/compute-backends.md](docs/compute-backends.md#local-backend). | Nothing. The compute budget question is skipped. |

### 3. Pick an LLM provider

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

### 4. Run `talos setup`

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

### 5. Before your first real run

Run `talos doctor`. It spends no compute, but its provider check makes one small LLM test
call. It checks the config, the LLM credential, the backend, mainnet and the GHCR dev images,
and says what to fix for each failure. Then run
`talos doctor --full` once. It compiles and scores one real mainnet algorithm on your backend
to confirm that the whole path works, and it shows the cost and asks `Spend this?` (default
no) before it spends anything. The one measured figure is for knapsack on C3: MEASURED
2026-09-15: £0.02 and 12 min 20 s. Every other challenge and backend gets a figure labelled
`ESTIMATE (unverified)`, and the local backend has no compute charge. See
[`talos doctor`](#talos-doctor).

## Running on Windows

Talos runs natively on Windows; WSL is not needed. (Inside WSL, follow the Linux instructions
instead.) CI runs the full test suite on `windows-latest` for every pull request. No
maintainer has run a real job end to end on a Windows machine yet, so if something fails
there, please open an issue with the command and its output.

Every block below is for PowerShell and can be pasted whole. The commands call
`.venv\Scripts\talos.exe` directly, so they work without activating the virtualenv and
without changing PowerShell's script execution policy.

### 1. Install Git and Python

```powershell
winget install --id Git.Git -e --source winget
winget install --id Python.Python.3.12 -e --source winget
```

Close PowerShell and open a new window, so that `git` and `python` are on `PATH`.

If `winget` is not available, install [Git for Windows](https://git-scm.com/download/win)
and [Python](https://www.python.org/downloads/windows/) from their installers. In the
Python installer, tick "Add python.exe to PATH".

### 2. Install Talos

```powershell
git clone https://github.com/FibonAdithya/talos.git
cd talos
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e .
.venv\Scripts\talos.exe --help
```

Run every later block from this `talos` folder.

### 3. Try it with no accounts

```powershell
.venv\Scripts\talos.exe demo
```

See [Try it with no accounts](#try-it-with-no-accounts) for what it prints.

### 4. Configure

Pick a [compute backend](#2-pick-a-compute-backend) and an
[LLM provider](#3-pick-an-llm-provider), then:

```powershell
.venv\Scripts\talos.exe setup
```

It asks the same questions on every OS; they are listed under
[Run `talos setup`](#4-run-talos-setup). On the C3 backend, give it a C3 API key (see
[What differs on Windows](#what-differs-on-windows)).

### 5. Check the setup, then run the smoke check once

```powershell
.venv\Scripts\talos.exe doctor
.venv\Scripts\talos.exe doctor --full
```

`doctor` spends no compute; its provider check makes one small LLM test call. `doctor --full`
compiles and scores one real baseline on your backend; it shows the cost and asks `Spend this?`
(default no) first. The measured figure is for knapsack
on C3 only (MEASURED 2026-09-15: £0.02 and 12 min 20 s); see [`talos doctor`](#talos-doctor).

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
  §8). None has run from Windows, so run `talos doctor --full` (step 5) before a real run. The `c3`
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
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt -e .
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

To start from notes you already have (what has been tried, what failed, what to avoid), write
them to a file and pass `--direction-file notes.md` instead of `--direction`. The whole file
is sent to the LLM as the direction on every iteration.

Before the job starts, Talos fetches the challenge's active tracks and fuel from mainnet,
draws the nonces, and prints a job summary such as this one (from a `--fake` job; the figures
differ per run):

```
Job summary
  challenge: knapsack (CPU, tested)
  scope:     1 tracks, 8 nonces per track, fuel 1
  provider:  fake / fake, single-shot mode
  search:    start mainnet, greedy, 1 chain(s), scoring native
  hyperparameters: 1/1 tracks from mainnet
  budget:    1 iterations, compute $20.00
  backend:   fake (in-process, no charge)
  results:   runs/20261026-142231-knapsack/
```

It then asks `Start this job?` (default yes). Answering no prints `Not started.`, exits 0 and
leaves nothing under `runs/`. `--yes` skips the question but still prints the summary. The job
id is the start time plus the challenge name.

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
including hypothesis descriptions and the compiler output (except the baseline's own compiler
output in a `--start template` job, which is left out so its source cannot leak).

With `--search anneal`, or with several chains or a template start, each scored candidate
also prints an `accept` line (chain, node, parent, design family, gap to mainnet, and the
decision), and a chain that stalls prints a `reheat` line. A candidate kept as its chain's
next step without improving the job's best reads `kept as the chain's next step`:

```
[11:02:40] #4 chain 1 #4 <- #2 [rolling-horizon LP] -12.000% vs mainnet: accepted (p=0.42, T=0.031)
[11:09:15] #6 chain 1 #6 <- #4 [greedy dispatch] -20.000% vs mainnet: rejected (p=0.05, T=0.029)
[11:31:03] #9 chain 1 reheats from #3, T=0.050
```

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

An `exhausted` job continues when you add budget to it:

```bash
talos run --resume 20260917-101006-knapsack --add-iterations 20
```

Nothing is re-measured: the job keeps its baseline, nonces, hardware, calibration, best
candidate and lineage. An annealing job continues at the temperature its spend gives on the
new total budget, rather than reheating. If another cap is still spent (say `--add-iterations`
on a job out of hours), the extension is refused and names the flag to add. The `--add-*`
flags also work on any job `--resume` accepts. A `won` job cannot be continued.

Talos now pins TIG dev image `0.0.8` and source with the upstream seed-obfuscation fix.
After upgrading from `0.0.7`, start a new job: saved runs from the old build cannot resume
against the new runtime. Baselines are measured again automatically. Modal users must
rerun `talos setup` to redeploy before starting; C3 and local Docker use the new image on
the next job. Run `talos doctor --full` after upgrading.

### Submit

When a job ends with `Status: won`, the code to submit is in `runs/<job_id>/package/submission/`,
next to a copy of TIG's submission `README.md` template. Fill out the template (algorithm name,
copyright, submitter), then copy the folder's contents into
`tig-algorithms/src/<challenge>/<your_name>/` in a monorepo checkout.
`runs/<job_id>/package/README.md` lists the steps. `evidence_draft.md` in the same directory
is a partly filled-in advance-evidence template. You submit it yourself; Talos does not.

A job that ends any other way still writes a package with its best candidate in
`runs/<job_id>/package/submission/`, but that candidate has not beaten the baseline on
held-out nonces, and the package README says so. The last line of the run's output says the
same instead of telling you to submit it.

## Command reference

| Command | What it does |
|---|---|
| `talos setup` | One-time configuration: backend, LLM provider, credentials. Validates them and deploys the Modal app. |
| `talos run` | Starts or resumes a research job and writes its package when it stops. |
| `talos compile` | Compiles a directory of algorithm files on the configured backend and prints the compiler output. Scores nothing. |
| `talos status [JOB_ID]` | Lists every job under `runs/`, or one job's detail with `JOB_ID`; `--json` for scripts. |
| `talos show JOB_ID` | Shows one iteration (the best by default): hypothesis, outcome, scores and file paths; `--json` for scripts. |
| `talos diff JOB_ID` | Prints one iteration's unified diff against the baseline or its parent. |
| `talos report JOB_ID` | Writes `runs/<job_id>/package/report.html`, a one-page view of the job's search, from the files already in the run directory. |
| `talos demo` | Runs a scripted job with no account, network or spend, and prints its package guide. |
| `talos doctor` | Runs health checks on the config, credential, provider (one small LLM test call), backend, mainnet and dev images, with no compute spend; `--full` also scores one real baseline after asking. |
| `talos check-updates` | Checks GHCR for newer published TIG dev image versions and prints upgrade guidance. |

### `talos setup`

No flags. See [Setup, step 4](#4-run-talos-setup) for every prompt and what is written.
Exit code 0 on success, 1 when a credential or backend check fails, 2 for an unknown
backend or provider.

### `talos demo`

```bash
talos demo                         # a knapsack-labelled demo
talos demo --challenge hypergraph  # the same script, labelled with another challenge
talos demo --delay 0               # no pause between event lines
```

| Flag | Meaning |
|---|---|
| `--challenge NAME` | The challenge to label the demo with (default `knapsack`). The script and scores do not change. |
| `--delay SECONDS` | Pause after each event line (default 0.4; 0 disables). |

Needs no config, account, network or compute backend, and spends nothing: every figure is
scripted and priced at $0.00. It writes a job to `runs/demo-<timestamp>-<challenge>/`, runs
the real research loop on it, and prints the event stream, the final status (`won`), the
package path, a one-line description of each package file, `talos report <job_id>` and
`Next: talos setup, then talos doctor`. Exit code 0 on success.

### `talos doctor`

```bash
talos doctor                                    # no compute; one small LLM test call
talos doctor --full --challenge knapsack        # also score one baseline, after asking
talos doctor --json                             # one JSON document instead of text lines
```

| Flag | Meaning |
|---|---|
| `--full` | After the free checks, compile and score mainnet's top algorithm for the challenge on 2 training nonces of its first track on your backend. It prints the cost, then asks `Spend this?` (default no; a closed stdin counts as no). |
| `--challenge NAME` | Challenge for `--full` (default `knapsack`). |
| `--yes` | Run `--full` without asking. `--json` works without it, but `--full --json` needs it. |
| `--json` | Print `{"checks": [{"name", "status", "detail"}, ...], "ok": true or false}` and nothing else on stdout. With `--full` and no `--yes` the `full` check is `skip` with detail `needs --yes`. |

The free checks run in this order and print one line each, `name: ok (detail)` or
`name: STATUS: detail`, where STATUS is `warn`, `FAIL` or `skip`:

- `config`: `talos.config.json` loads. With none in the directory, `credential`, `provider`
  and `backend` print `skip` and the other checks still run.
- `credential`: the provider's API key, or its CLI (`claude`, `codex`) on `PATH`.
- `provider`: one test call to the LLM.
- `backend`: Modal has every function of the `talos-bench` app deployed; C3 is logged in and
  has credit (`warn` under £1); Docker answers on a local backend.
- `mainnet`: mainnet is reachable, and each challenge's id and CPU/GPU flag match
  `talos/challenges.py`.
- `images`: the pinned dev image is on GHCR for every challenge.

Most `FAIL` lines name the command that fixes it (such as `talos setup`); the rest say what to check, such as your network, or pass on the backend's own error text. With `--full` a last line, `full`, reports
the compile and the nonces scored. If a free check failed, `full` is `skip` and does not run.
The cost it prints is MEASURED 2026-09-15: £0.02 and 12 min 20 s for knapsack on C3;
for every other challenge and backend it is labelled `ESTIMATE (unverified)` and says
how many minutes it assumes. The local backend has no compute charge but pulls a 13 GB image
the first time.

Exit code 0 when no check is `FAIL`. Exit code 1 when any check is `FAIL`, and also when
`--full` was requested and did not run and pass (declined, skipped, or failed).

### `talos check-updates`

```bash
talos check-updates                       # all eight challenges
talos check-updates --challenge knapsack  # one challenge
```

Shows the configured image version and the newest published stable `X.Y.Z` tag for each
challenge. It reads only registry metadata: Docker, sudo, project configuration and
compute credentials are not needed. Aliases, prereleases and architecture-specific tags
are excluded. A newer tag is an upgrade candidate; source compatibility still needs review.
Exit code 0 means every lookup succeeded (including when updates are available), 1 means
at least one lookup failed and its update status is unknown, and 2 means invalid arguments.

Each new real job also checks its challenge and prints a recommendation if a newer image
is available. Each check has a five-second request budget; a failed automatic check is
silently skipped. Resumes, compile calls and fake runs do not make this advisory check.

To apply an upgrade:

1. Review the upstream release and its matching source revision. Check that the proposed
   image tag exists for all eight challenges and the architectures you use.
2. Update `DEV_IMAGE_TAG` and, when needed, `MONOREPO_REF` in `talos/challenges.py` together.
3. Run `make check`. For Modal, redeploy with `talos setup` before running the backend's
   `talos doctor --full`. C3 and local Docker use the new pin on their next job; run
   `talos doctor --full` for them too when that is your backend.
4. Start new jobs. Existing measurements are tied to the old pins; the upgrade automatically
   gets fresh cache keys and local build volumes.

### `talos run`

Prompts for anything not given as a flag, unless `--yes` is passed. Before it writes anything
or spends anything, it prints a job summary and asks `Start this job?`; `--yes` skips that
question. Answering no, or a closed stdin, ends the command with nothing under `runs/`.

The table below gives each challenge's status. `talos run` repeats the status as a warning
before it starts a job on a challenge that is not tested. "Tested" means at least one Talos job got past its baseline on some backend.

| Challenge | Hardware | Status |
|---|---|---|
| `knapsack` | CPU | tested |
| `energy_arbitrage` | CPU | tested |
| `hypergraph` | GPU | tested |
| `satisfiability` | CPU | untested |
| `vehicle_routing` | CPU | untested |
| `vector_search` | GPU | untested |
| `neuralnet_optimizer` | GPU | untested |
| `job_scheduling` | CPU | known broken ([#36](https://github.com/FibonAdithya/talos/issues/36)) |

| Flag | Meaning |
|---|---|
| `--challenge NAME` | Status of each challenge: see the table above. One of `satisfiability`, `vehicle_routing`, `knapsack`, `job_scheduling`, `energy_arbitrage` (CPU), or `vector_search`, `hypergraph`, `neuralnet_optimizer` (GPU: an L40S, or when none is free the next of A100-80GB and H100). On C3 a CPU challenge likewise runs on `cpu-d3-4vcpu-16gb`, or `cpu-e2-4vcpu-16gb` when that is out of stock; see [docs/compute-backends.md](docs/compute-backends.md#hardware-fallback). |
| `--direction TEXT` | What to explore, in free text. It becomes the first entry in the job's `tacit.md`. |
| `--direction-file PATH` | The same, read from a file. Pass one of `--direction` or `--direction-file`, not both. |
| `--track NAME` | Optimise one active track of the challenge instead of all of them (`all` is the default; the prompt lists the tracks). Training scores that track only. When a candidate wins on training, the confirmation scores that track's held-out nonces plus every other track's training nonces, and no other track may get worse. The LLM still sees and may edit every file. |
| `--nonces N` | Nonces per track in each of the training and held-out sets (8 is the default; the prompt asks). Every nonce is scored for the baseline on both sets and for every candidate on training, then again on held-out when it wins on training, so the count sets the cost of every evaluation. On hypergraph an L40 spends about 10 minutes per nonce-per-track on the baseline's two sets at the current fuel (measured 2026-09-25); 32 made that baseline 11 hours against the 6-hour C3 job limit. A resume keeps the sets the job was drawn with. |
| `--hyperparameters mainnet\|none` | `mainnet` (the default) runs the baseline and every candidate with the per-track hyperparameters of the baseline algorithm's best-quality mainnet benchmark at the job's fuel, fixed at job start. A track with no such benchmark runs without any. `none` runs every nonce without hyperparameters. The package lists the values and the benchmark they came from. With `--start template`, only the baseline runs with these values: candidates run with none, and the package has no `hyperparameters.json` (its README lists the baseline's values). |
| `--mode single-shot\|agentic` | Overrides the configured mode for this job. `agentic` needs a CLI provider and uses roughly 5 to 20 times the tokens of `single-shot`. |
| `--scoring native\|metered` | `native` (the default) scores research candidates with an unmetered runner under a per-track time budget calibrated from the baseline, which is faster; a candidate that improves natively is then scored again on the metered runtime and only becomes the best if it holds up there ([how it works](docs/architecture.md#native-research-scoring)). `metered` scores every candidate with TIG's own `tig-runtime` under the job's fuel limit. The live parity test has not been run: the CPU runner is measured on knapsack only, and the GPU runner has never been compiled; a native build that fails at calibration makes the job score metered. A job keeps its scoring on resume; `--resume` with a different `--scoring` is refused. A job whose `job.json` predates the `scoring` field resumes metered. The calibration is cached at `~/.talos/calibration/<challenge>/<key>.json`, keyed by `talos/calibration.py::calibration_key` (challenge, hardware class, both pins, the baseline's code, the native runner and the baseline's hyperparameters). If it cannot be made, the job scores metered and says so in the timeline. |
| `--start mainnet\|template` | `mainnet` (the default) starts from the top-adoption mainnet algorithm. `template` starts from TIG's empty template: the LLM sees the challenge's `README.md` and its non-baseline source files from the pinned monorepo (`talos/challenges.py::TEMPLATE_START_FILES`, not the baseline algorithms' directory) and the template, and never the baseline's code. The baseline is still measured and every candidate is scored against it. Candidates run without hyperparameters while the baseline keeps the mainnet ones. `energy_arbitrage` only; refused for other challenges. Refused with `codex-cli` in agentic mode, even with `TALOS_ALLOW_CODEX_AGENTIC=1`. See [From-scratch start and annealing search](docs/architecture.md#from-scratch-start-and-annealing-search). |
| `--search greedy\|anneal` | `greedy` (the default) edits the best candidate so far. `anneal` edits the chain's current candidate and sometimes accepts a worse one so the search can leave a local optimum. The annealing settings (start temperature 0.05, final temperature 0.1 of that, reheat after 5 iterations without a new chain best) are estimates with no measurement behind them. |
| `--chains N` | Independent search chains (default 1; a value below 1 is refused). Chains take turns: iteration n runs on chain (n-1) mod N, one at a time; running them concurrently comes in a later change. |
| `--budget-usd N` | LLM spend cap in USD. |
| `--budget-hours N` | Wall-clock cap in hours. |
| `--budget-iterations N` | Iteration cap. |
| `--budget-compute-usd N` | Estimated compute spend cap in USD. |
| `--resume JOB_ID` | Continue a job that was interrupted, cancelled, paused or failed, or an `exhausted` job given an `--add-*` flag. Cannot change its mode, track, hyperparameters, nonces, start, search or chains. |
| `--add-iterations N` | Adds N to the job's iteration cap. Only with `--resume`; refused for a cap the job was started without. |
| `--add-hours N` | Adds N to the job's wall-clock cap. Only with `--resume`; refused for a cap the job was started without. |
| `--add-usd N` | Adds N to the job's LLM spend cap. Only with `--resume`; refused for a cap the job was started without. |
| `--add-compute-usd N` | Adds N to the job's compute spend cap. Only with `--resume`; refused for a cap the job was started without. |
| `--yes` | Accept defaults instead of prompting, and skip the `Start this job?` confirmation. At least one of `--budget-usd`, `--budget-hours` or `--budget-iterations` must still be given; compute defaults to $20 (not asked and not set on the `local` backend, where it is always zero). |

Without flags, the wizard asks for challenge, direction, an LLM budget (USD for metered
providers, default 20; iterations for CLI providers, default 50), a wall-clock budget
(default 4 hours), a compute budget (default $20), the mode (CLI providers only), the
track, the hyperparameters (`mainnet` or `none`), and the nonces per track (default 8).

Talos refuses to start, before spending anything, when: there is no API key for the
provider; the model has no entry in the price table, or the provider is `codex-cli` (which
reports no cost), and the only budget is `--budget-usd` (a dollar cap it could not enforce);
the mainnet challenge id no longer matches `talos/challenges.py`; mainnet gives no fuel
budget for the challenge; or, on C3, the challenge's dev image tag is not on GHCR.

Exit code 0 when the job ends `won`, or when you decline the confirmation; 1 when it ends any other way, or when the mainnet or
dev image check stops it; 2 for invalid arguments, a missing config, or a missing API key.

### `talos compile`

```bash
talos compile --challenge knapsack --dir algorithm
```

| Flag | Meaning |
|---|---|
| `--challenge NAME` | Required. One of the names in the challenge status table under `talos run`; any other name exits 2 with a usage error. |
| `--dir PATH` | Directory whose `.rs` and `.cu` files are compiled (default `algorithm`). |
| `--backend modal\|c3\|local` | Backend to use. Without it: `TALOS_BACKEND`, then `talos.config.json`, then `modal`. |
| `--native` | Build the unmetered `talos-native` runner instead of TIG's metered build. This is the build a `--scoring native` job (the default) scores with, and its agent is told to pass this flag. Without it the build is TIG's metered one, the build a submission uses. |

A candidate that compiles with `--native` can still fail TIG's metered build. A native job's
validation catches that and demotes the candidate as `native_metered_build_mismatch`.

Prints the last 4000 characters of compiler output. Exit code 0 if the build succeeded, 1 if
it failed, 2 if the directory has no `.rs`/`.cu` files. Agentic mode uses this command to
check its own edits.

On Modal it is one function call. On C3 it is one batch job
([about 12 minutes](docs/compute-backends.md#c3-timings)) whose job directory is
`.talos/compile/`. On `local` it is one Docker container on this machine, after the same
prepare step as `talos run` (image pull, clone and warm-up build on the first run per
challenge; [about 3 minutes for knapsack](docs/compute-backends.md#local-timings) per build after that),
with its job directory under `.talos/compile/` too; the container is removed when the command
returns or is interrupted. Each run overwrites that directory, so do not run two
`talos compile` commands at once from the same directory.

### `talos status`

```bash
talos status [JOB_ID] [--json]
```

Without a job id, one row per job under `runs/` (the row below is illustrative, not a real
run):

```
JOB                             CHALLENGE       STATUS     IT  BEST     LLM          COMPUTE  LAST EVENT
20260917-101006-knapsack        knapsack        won        1   +0.412%  $0.02        $1.28    2026-09-17 10:41 UTC
20261026-120733-job_scheduling  job_scheduling  exhausted  6   n/a      not tracked  $12.40   2026-10-26 21:14 UTC
```

BEST is the best candidate's mean training delta, `n/a` with no candidate. LLM reads
`not tracked` for a `codex-cli` job and `unpriced` for a model with no price-table entry, as in
the status line during a run. LAST EVENT is the time of the last event in `timeline.jsonl`. A
directory whose `state.json` or `job.json` is missing or cannot be parsed is listed as
`unreadable`, with the reason on stderr; it does not stop the other jobs from being listed.

With a job id, that job's settings, status and stop reason, its best candidate (training delta
and held-out result), its budget (each cap after `--add-*` extensions, what was spent and what
is left), the compute job in flight and the last event. Hours spent run to the last event for
a job that has stopped (`won`, `exhausted`, `failed`, `cancelled` or `paused`) and to now for
any other. Talos writes no lock file, so a `researching` job whose last event is old may belong
to a process that has died.

`--json` prints one JSON object instead; its schema is in
[docs/json-output.md](docs/json-output.md). The command reads `runs/` only: no config,
credentials or network. Exit code 0, or 1 with a message on stderr for an unknown job id.

Status is one of `queued`, `measuring_baseline`, `researching`, `confirming`, `won`,
`exhausted` (a budget ran out; resumable with `--add-*`), `cancelled` (Ctrl-C), `paused`
(the compute backend was unreachable) or `failed` (the reason is in `state.json` and the last
lines of `timeline.jsonl`).

### `talos show`

```bash
talos show JOB_ID [--iteration N|best] [--json]
```

One iteration of a job: its parent in the search tree, outcome, hypothesis (title, strategy,
design family and the full description), training delta with the worst track and per-track
deltas, the held-out result, and the paths of its saved files. For the best candidate it also
prints the per-track tables, baseline against best, on the training and held-out nonces.
`--iteration` defaults to `best`. `--json` prints one JSON object; see
[docs/json-output.md](docs/json-output.md).

Exit code 0; 1 with a message on stderr when the job, the iteration or a best candidate does
not exist (an iteration still in flight has no record yet); 2 for a bad flag.

### `talos diff`

```bash
talos diff JOB_ID [--iteration N|best] [--against baseline|parent]
```

Prints the unified diff of one iteration's files to stdout, untruncated. `--iteration`
defaults to `best`. `--against baseline` (the default) diffs against the root the job started
from: the mainnet baseline, or the template for a `--start template` job. `--against parent`
diffs against the iteration it was built on. An empty diff prints nothing.

Exit code 0; 1 with a message on stderr when the job, the iteration or a best candidate does
not exist, or when the iteration saved no files (an edit or compile failure); 2 for a bad flag.

### `talos report`

```bash
talos report 20260917-101006-knapsack
```

Writes `runs/<job_id>/package/report.html` and prints its path. The page is one file with no
network access: a header (status, best delta, held-out result, spend, times, settings), a plot of
every iteration's training delta against the mainnet baseline with lines from each node to its
parent, a panel with any node's details and its diff against its parent, per-track tables for
the best candidate, the best diff, and the hypothesis list. It is built only from files already in
the run directory, so it works on old runs and on a job that is still running; it needs no
credentials and writes nothing else. The package written when a job stops includes it too.
Diffs are shortened to keep the page under about 2 MB; a shortened diff ends with the `diff -u`
command that prints it in full.

## Agentic mode

`--mode agentic` (CLI providers only) hands each iteration to a headless `claude` or `codex`
session in a throwaway worktree outside `runs/`, instead of asking an API for one edit.

With `claude-cli`, the session runs under a sandbox that the CLI enforces: it can read only
inside its worktree and edit only the algorithm files and its notes, the only commands it can
run are `talos compile` and read-only ones such as `ls` and `grep`, it has no network tools,
and its environment holds no LLM keys or Modal tokens. The exact rules are in [docs/architecture.md](docs/architecture.md#the-agentic-sandbox).

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
| `runs/<job_id>/job.json` | The job's fixed inputs: challenge, direction, provider, model, mode, budget, nonces, fuel, track, baseline algorithm, hyperparameters, scoring, start, search, chains and the annealing settings. Written once. Its budget is the starting budget; a job resumed with `--add-*` runs on that budget plus the extensions in `state.json`. |
| `runs/<job_id>/state.json` | Progress: status, iteration, spend, the best candidate, the hypothesis log, and the lineage tree (`nodes`, one per iteration, and `chains`, each chain's current node, best node and stall count). A job resumed with `--add-*` records each extension in `budget_extensions` (when, at which iteration, the status it was resumed from, and what it added). A `--scoring native` job also records its scoring, its per-track time budgets and a `validations` list: one entry per candidate that improved natively, with its iteration, its outcome (`validated` or `demoted`) and the reason it was demoted. |
| `runs/<job_id>/timeline.jsonl` | One JSON event per line; the same events the terminal prints. A resume with `--add-*` adds `budget_extended` (the additions and the resulting budget). A native job adds `calibrated` (the budgets), `validating`, `validated` and `demoted` (with its reason); `calibration_fallback` says the job scores metered and why. A tree job (anneal, several chains or a template start) adds `accept` (one per scored candidate: chain, node, parent, family, score, decision, probability, temperature) and `reheat`. |
| `runs/<job_id>/tacit.md` | Your direction, plus lessons the LLM distilled from failed attempts. |
| `runs/<job_id>/baseline/` | The baseline algorithm's files and its per-nonce results. |
| `runs/<job_id>/challenge/` | A `--start template` job only: the challenge files and the template the LLM was shown. |
| `runs/<job_id>/iterations/<n>/` | Each candidate's files and its hypothesis. |
| `runs/<job_id>/best/` | The best candidate's files so far. |
| `runs/<job_id>/package/` | The hand-back package, written when the job stops. |
| `runs/<job_id>/package.zip` | The same package, zipped. |

The package directory holds:

| File | Contents |
|---|---|
| `submission` | A folder with the candidate to submit: the best algorithm's `.rs` and `.cu` files and TIG's submission `README.md` template, to fill out. Absent when the job produced no candidate. |
| `diff_vs_baseline.patch` | The candidate as a patch against the baseline. A `--start template` job writes `diff_vs_template.patch` instead, against the empty template. |
| lineage.json, lineage.md | The search tree: every iteration's node and each chain's state (lineage.json), and the path from the root to the best node with a per-design-family table (lineage.md). Written when the job recorded any node. |
| `scores.md` | Per-nonce tables for baseline and candidate on training and held-out nonces. A focused job (`--track`) adds a regression-guard table for the other tracks. |
| `hypotheses.md` | The full hypothesis log with outcomes. |
| `report.html` | The search on one page: a plot of every iteration with its lineage, each node's diff against its parent, per-track tables and the hypothesis log. Open it in a browser; it loads nothing from the network. Written on every package, including one with no candidate. If the page fails to render, the package prints `report.html not written: …` and is written without it; `talos report` rewrites it. |
| `evidence_draft.md` | A partly filled-in TIG advance-evidence template. |
| `hyperparameters.json` | The per-track hyperparameters, when the job used them. Not written for a `--start template` job, whose candidate ran without them. |
| `README.md` | The submission steps, with a Hyperparameters section when the job used them. |

Benchmark the submitted algorithm with the same hyperparameters: the measured improvement
holds only with them. A `--start template` candidate was scored without hyperparameters, so
benchmark it without any.

On the C3 backend each iteration also gets a `runs/<job_id>/c3/<n>/` directory, described in
[docs/compute-backends.md](docs/compute-backends.md#c3-job-directories). The measured baseline
is cached outside the run directory, in `~/.talos/baselines/`; see
[docs/architecture.md](docs/architecture.md#the-baseline-cache).

## Budget

- At least one of `--budget-usd`, `--budget-hours` or `--budget-iterations` must be set,
  directly or through the wizard. Zero is a valid, real cap, not "unset".
- Compute spend is always capped separately by `--budget-compute-usd`, which defaults to $20
  when `--yes` is passed without it. It is checked before every compute call, whatever the
  other caps are. The one exception is the `local` backend, where compute costs nothing: the
  question is skipped, no default is applied, and the cap stays unset.
- The wall-clock budget (`--budget-hours`) counts elapsed time from the job's start,
  including time spent resumed.
- `talos run --resume JOB_ID --add-*` adds to the job's caps (see [Resume](#resume)). The time
  between a stop and the continuation is not counted against `--budget-hours`.

Every spend figure in the status lines and the final report is an **estimate**, not a billed
amount. Compute spend is measured running seconds times list prices in a table shipped with
Talos. LLM spend is measured tokens times a price table. With `claude-cli`, in both modes, it
is the `total_cost_usd` the CLI reports for each call or agentic session, the CLI's own estimate
at API prices, whatever your subscription actually charges. A model with no entry in that
table is shown as `unpriced`, and `--budget-usd` cannot be enforced for it: add
`--budget-hours` or `--budget-iterations`. The same applies to `codex-cli`, which reports no
cost at all and is shown as `not tracked`.

`--budget-usd` is checked before each LLM call or agentic session, so the call that crosses
the cap still finishes; the run stops before the next one. A `claude-cli` call that fails is
still charged the cost it printed. A call or agentic session killed at its 30-minute timeout
prints no cost, so it is not counted; the timeline records it, and any other call that ends
without a cost, as an `llm_cost_unknown` event.

Compute spent by `talos compile` from the agentic sandbox is not counted against
`--budget-compute-usd`. On C3 each of those is one job with the fixed overhead described
under [C3 timings](docs/compute-backends.md#c3-timings), billed at the profile's rate. A
focused job's confirmation scores the other tracks' training nonces too, about a minute more
per winning iteration on C3 (ESTIMATE, unverified).

## Live smoke tests (maintainers)

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
time, not money: the first run pulls the image, clones the pin and does the warm-up build
(MEASURED 2026-09-25: `prepare_s: 90` with the image already pulled), a later run finds
everything in place and takes only the job (MEASURED 2026-09-25: knapsack `1 passed` in
4 min 33 s with `job_s: 177`; job_scheduling `1 passed` in 18 min 16 s with `job_s: 990`):

```bash
TALOS_LIVE_BACKEND=local .venv/bin/pytest -m live tests/test_live.py -k local -s
```

For agentic mode, a fourth test makes one real `claude-cli` call on the smallest model, so it
spends LLM tokens, not compute. The agent changes directory into its worktree's algorithm
folder and must still be allowed to edit there (MEASURED 2026-09-29: `1 passed` in 12.2 s):

```bash
TALOS_LIVE_AGENT=claude-cli .venv/bin/pytest -m live tests/test_live.py -k claude_agent -s
```

All of them need `pytest`: install it with `.venv/bin/pip install pytest`, or use
the development install below.

Users run `talos doctor --full` instead (see [`talos doctor`](#talos-doctor)). The
maintainers ran the C3 test on 2026-09-15 and the Modal test on 2026-10-26 (MEASURED: knapsack,
mainnet algorithm `satchel`, both nonces `ok`, `1 passed` in 260.71 s, at commit `76e3ccd`).

## Development

The gate needs Python 3.11 or newer, because the pinned `agentify` in `requirements-dev.txt`
does. CI uses 3.12. If `python3` is older, create the virtualenv with a newer interpreter,
for example `python3.12 -m venv .venv`.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt -e .
make check PYTHON=.venv/bin/python
```

`make check` runs ruff, pytest without the `live` marker, and the agentify contract check.
It is the same command CI runs. See `AGENTS.md` for the invariants a change must keep.

## Licence

GPLv3 (see `LICENSE`). `talos/search_replace.py` is lifted from
[tig-foundation/prometheus-swarm](https://github.com/tig-foundation/prometheus-swarm)
(its file scripts/search_replace.py), also GPLv3; `talos/agentic.py`'s sandbox settings mirror
Prometheus's `_build_sandbox_settings` design without copying its code.
