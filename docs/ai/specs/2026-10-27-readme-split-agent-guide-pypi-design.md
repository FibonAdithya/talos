# README split, agent guide, PyPI install: design

Issues #66, #71 and #70. Three PRs, merged in that order, each branched from the
latest `origin/main`. Later PRs edit files the earlier ones create.

## Name check

`talos-tig` is free on PyPI (MEASURED 2026-10-27: `curl https://pypi.org/pypi/talos-tig/json` → 404).
`project.name` is already `talos-tig`.

## Why

- The README is 875 lines (MEASURED 2026-10-27: `wc -l README.md`). It serves new users, regular users
  and maintainers at once (#66).
- Coding agents will drive Talos for many users. Today an agent has to read the whole README to find the
  non-interactive flags, which commands spend money and where results land. The repo's `AGENTS.md`
  is for agents changing Talos, not using it (#71).
- Talos only installs from a clone, and it only works from the directory holding `talos.config.json`.
  Comparable tools install in one command (#70).

## Decisions taken by the owner (2026-10-27)

| Question | Decision |
|---|---|
| Order | #66, then #71, then #70; one PR each |
| #71's `talos show`, `talos diff`, `talos status JOB_ID --json` (none exist) | Add `talos status [JOB_ID] [--json]` only. No `show` or `diff`: results are read from the package files |
| #70: where state lives | Walk up from the current directory to the nearest `talos.config.json`, like git finds `.git`. No `talos init` command. An existing clone keeps working with no migration |
| #70: how far to go | Everything up to the first release: code, a tag-triggered publish workflow, CI that installs the built wheel on 3 OSes. The owner registers the PyPI trusted publisher and pushes the first tag |
| #70: extras | `modal` and `anthropic` stay default dependencies |
| #71 agent trial permissions | Throwaway clone with no config or credentials; Claude limited to `Bash(talos:*)` plus read-only tools; Codex in `workspace-write`; no bypass |

## Goals and non-goals

Goals: the three issues' "done when" lists, with the adjustments below.

Non-goals:
- Tool-specific agent wrappers (a Claude Code skill, a Cursor rule). Out of scope per #71.
- Removing `rich` or other dependency cleanup (separate issue).
- A docs website. Plain markdown in `docs/`.
- New live measurements for the README. Figures come from text already in the repo.

Anti-goals:
- Losing content. Every README section ends up in exactly one place.
- A number without its MEASURED date or `ESTIMATE (unverified)` label.
- Breaking a current clone user: same commands, same files, same `runs/`.

---

## PR 1: split the README (#66)

### Target layout

| File | Holds | Source sections of today's README |
|---|---|---|
| `README.md` (under 200 lines) | What Talos does and does not do; install; `talos demo`; backend and provider choice (short tables); `talos setup` and `talos doctor` (one paragraph each); first real run on knapsack; **what to expect**; challenge status table; where results land (short); links to the docs below; licence | Intro, How it works (shortened to one paragraph plus links), Prerequisites, Try it, Setup 1–5 (condensed) |
| `docs/usage.md` | Every prompt of `talos setup`; running a job (start, watch, stop, resume, submit); the full command reference with every flag table; agentic mode; where results land (full tables); budget | Setup step 4 details, Running a job, Command reference, Agentic mode, Where results land, Budget |
| `docs/troubleshooting.md` | Known failure modes, each as symptom (the exact message Talos prints, copied from the code) → cause → fix | See the list below |
| `docs/maintainers.md` | Pinning a new TIG dev image; the `0.0.7 → 0.0.8` migration note; live smoke tests and their history; development setup (Linux/macOS and Windows) | `check-updates` "To apply an upgrade", Resume's migration paragraph, Live smoke tests, Development, Development on Windows |
| `docs/windows.md` | Windows install and run, PowerShell blocks, what differs on Windows, `cmd.exe` | Running on Windows, minus its Development subsection |

### Rules

- Flag rows in `docs/usage.md` are one or two sentences. The detail moves to a paragraph under the
  table. This applies to `--scoring`, `--start`, `--nonces`, `--hyperparameters`, `--track` and
  `--challenge`. Internal code paths such as `talos/calibration.py::calibration_key` leave the flag
  table. They go into that paragraph, or into `docs/architecture.md` where they already have a home.
- **What to expect** is a small table: per backend, time per iteration and estimated compute cost per
  iteration. Then one sentence saying most jobs end `exhausted`, not `won`, and that an `exhausted`
  job's package is not a submission. Every cell is either a figure already in the repo with its
  MEASURED date (C3: about 12 minutes of fixed overhead per iteration, since each iteration is one batch job,
  `docs/compute-backends.md`; local: 3 minutes per knapsack
  build) or `ESTIMATE (unverified)`. A cell with no figure in the repo says so. Do not make one up.
  The dated figures also stay in the reference docs.
- The challenge status table stays in `README.md`, because `tests/test_challenges.py::test_readme_status_table_matches_the_code`
  reads it there and it is what a user needs to pick a first challenge. (Builder's call: if it moves,
  the test must move with it in the same commit.)
- Troubleshooting entries, at least: `modal deploy failed`; a client reaching an older Modal deploy;
  C3 CLI older than 0.20.0 refusing the GHCR image; C3 balance warning; local container CPU or memory
  above Docker Desktop's VM limits; an `unpriced` model or `codex-cli` with only `--budget-usd`;
  agentic codex refused without `TALOS_ALLOW_CODEX_AGENTIC=1`; a job `paused` (backend unreachable);
  a job `failed` (where the reason is); resume refused after the dev image upgrade; a sandbox compile
  on C3 that keeps running after the agent timeout; Windows (link to `docs/windows.md`). Each entry's
  symptom is the string the code prints, found with `grep` in `talos/`. If the README describes a
  failure the code no longer produces, leave it out and say so in the PR.
- Links: update every link into a moved README anchor. Known ones: `AGENTS.md` (the
  `README.md#where-results-land`, `#budget` and `#live-smoke-tests-maintainers` references, and the
  source-of-truth list, which gains `docs/usage.md`, `docs/troubleshooting.md`, `docs/maintainers.md`
  and `docs/windows.md` at level 2 beside `README.md`), `docs/architecture.md` (`#agentic-mode`, the
  "README's budget section"), `docs/compute-backends.md` (`#2-pick-a-compute-backend`,
  `#where-results-land`). Grep for `README.md#` and `../README` to find the rest.

### Done when

- `wc -l README.md` < 200.
- A section map in the PR body: every `##`/`###` heading of the old README → its new file and heading.
- `make check PYTHON=.venv/bin/python` passes. It runs agentify's reference check, which resolves
  doc anchors.

---

## PR 2: `talos guide` and `talos status --json` (#71)

### `talos status [JOB_ID] [--json]`

- No arguments: the same lines as today, unchanged.
- `JOB_ID`: that job only. An unknown id prints `no job <id> under runs/` to stderr and exits 1, like
  `talos report`.
- `--json`: one JSON document on stdout, nothing else. Without `JOB_ID`: `{"jobs": [...]}`. With it: the
  job object. Each job object has at least:
  `job_id`, `status`, `stop_reason`, `iteration`, `challenge`, `spend` (`llm_usd`, `compute_usd`,
  and `llm_tracking`: `tracked`, `unpriced` or `not tracked`, from the same logic as
  `_llm_spend_text`), `budget` (the effective caps, including `budget_extensions`), `best` (iteration
  and training delta, or `null`), `heldout` (see below), and `package` (the absolute path of
  `runs/<id>/package`, or `null` if not written yet). The builder reads `talos/state.py` for the
  source of each field.
- `heldout` is `{"result": "won" | "false_positive" | "not_run", "delta": float | null}` for the
  best candidate. `state.json` does not hold the held-out delta: `state.confirmed` and
  `state.false_positives` are lists of iteration numbers (`talos/state.py::JobState`). The delta is
  only in the `won` and `false_positive` events of `timeline.jsonl` (`holdout=`, in
  `talos/loop.py`). Read it from the last such event for the best candidate's iteration. (Builder's
  call, recorded here: a false positive's held-out delta is reported, because an `exhausted` job's
  package reports it too and the guide's report template needs it.)
- `state.json` is written atomically (`talos/state.py::_atomic_write`), so polling a running job is
  safe. A job whose `job.json` cannot be read still gets an entry, with the fields that come from it
  set to `null`.

### `talos guide`

- Prints `talos/data/agent_guide.md` to stdout and exits 0. It needs no config, network or
  credentials. `talos/data/*.md` is already package data in `pyproject.toml`.
- Read with `importlib.resources`, not `Path(__file__)`, so it works from a wheel.

### Guide content

Plain markdown, written for an agent. Sections from #71, with these corrections:

1. What Talos does and does not do: it never submits, and it spends the user's LLM and compute money.
2. Spending rules. Free: `guide`, `demo`, `status`, `report`, `check-updates`, `compile --backend local`.
   `doctor` spends no compute but makes one small LLM call. Spends: `run`, `doctor --full`, and
   `compile` on modal or c3. Confirm every budget with the user before a spending command. Never pass
   an `--add-*` flag without asking.
3. Starting a job non-interactively: the minimal `talos run --challenge ... --direction-file ...
   --budget-... --yes`. Point to the status table for choosing a challenge. How to write a
   direction file (prior attempts, constraints, ideas) with one short worked example. Budget presets
   per backend, labelled ESTIMATE where they are not measured.
4. Monitoring: launch in the background with output to a log file, poll `talos status JOB_ID --json`,
   what each status means. Stop with **one** `kill -INT <pid>` on Linux and macOS. Never `kill -9` or a bare
   `kill` (SIGTERM is not handled, so no package is written). `talos run` installs its own SIGINT
   handler only around the research loop (`cmd_run` in `talos/cli.py`). Before that point, a process
   launched with `&` from a non-interactive shell inherits SIGINT as ignored, so an early
   `kill -INT` does nothing. Fix (builder's call, recorded here): `cmd_run` sets
   `signal.default_int_handler` on entry, so an early SIGINT raises `KeyboardInterrupt` and the
   command exits before the job starts. A test covers this. The builder verifies both windows on Linux
   with a `--fake` job launched with `&` from a non-interactive shell before writing the guide text.
   Windows: say plainly what is and is not possible, after checking.
5. Reading results: the package files (`README.md`, `scores.md`, `hypotheses.md`, `report.html`) and
   what `won` vs `exhausted` means for submission. Use the held-out result and the worst-track delta.
   No "noise ratio": the report has none.
6. Reporting back to the user: a short template (outcome, best training delta, held-out result, spend
   with its tracking label, package path, suggested next step).
7. Troubleshooting: run `talos doctor` first, then `docs/troubleshooting.md`. Link it by its GitHub
   URL, because an installed copy has no `docs/`.

`README.md` gets one line near the top: an agent can be told "Run `talos guide` and follow it."

### Drift test

`tests/test_guide.py`:
- every `talos <word>` in the guide (code spans and fenced blocks) names a subcommand of
  `build_parser()`;
- every `--flag` that appears in a `talos <subcommand> ...` command in the guide is an option of that
  subparser;
- `talos guide` output equals the packaged file.

Mutations it must catch: a renamed flag in the parser, a misspelt subcommand in the guide, `talos guide`
printing a different file.

### Agent trial

Run Claude Code (`claude -p`) and Codex (`codex exec`), both installed on this machine, with only this
prompt: "Run `talos guide` and follow it to run the Talos demo, then report back to me as the guide
says." Use a throwaway clone in the session scratchpad with no `talos.config.json` and no
`.talos/secrets.json`, so a spending `talos run` fails for lack of config. Permissions (owner's
decision, 2026-10-27): Claude gets `--allowedTools` limited to `Bash(talos:*)` plus read-only tools;
Codex runs in its `workspace-write` sandbox. No permission bypass for either. This spends a little of the owner's CLI subscription usage, and no
compute. Record where each agent got stuck in `docs/ai/reports/2026-10-27-agent-guide-trial.md` and in
the PR body. Fix the guide for what the trial shows, then re-run once.

### Done when

- `talos guide` prints the guide from an installed package. A CI step for this lands in PR 3. In
  PR 2, a test runs it through `main()`.
- The drift test passes and the mutation check above is shown.
- The trial notes exist.

---

## PR 3: install from PyPI, state found by walking up (#70)

### Workspace resolution

New `talos/workspace.py::find_root(start: Path | None = None) -> Path`:

1. If `TALOS_WORKSPACE` is set, use that directory, without walking.
2. Otherwise walk from the current directory up through its parents. Return the first directory that
   holds `talos.config.json`.
3. Otherwise return the current directory, which is today's behaviour.

The marker is `talos.config.json` only. A demo-only clone (with `runs/` but no config) behaves as it
does today.

Every `Path.cwd()` in `talos/cli.py` (9 call sites, MEASURED 2026-10-27 with
`grep -o 'Path.cwd()' talos/cli.py | wc -l`; an earlier draft said 8, which was wrong) goes through `find_root()`. This covers setup, run, demo, status,
report, compile and doctor. `talos setup` writes to the found root if there is one, otherwise to the
current directory. It prints the workspace path. The `talos run` job summary's `results:` line
becomes an absolute path.

**Agentic sandbox**: `talos compile` runs inside a temporary worktree (`talos/agentic.py::prepare_worktree`).
Walking up from a temp directory could find a `talos.config.json` above it and read that workspace's
config or `.talos/secrets.json`. The sandbox environment must therefore set
`TALOS_WORKSPACE=<worktree>`. The sandbox environment is an allowlist
(`talos/agentic.py::_AGENT_ENV_ALLOWLIST`), so the variable is set explicitly in the agent's env,
not inherited. A test asserts this, and a mutation that drops it must fail.

### Modal deploy from an installed package

`deploy_bench` runs `shutil.which("modal")` first. Under `uv tool install`, the tool's virtualenv is not
on `PATH`, so that finds some other `modal` client of any version, or none. Change the order:
`[sys.executable, "-m", "modal"]` first, then `shutil.which("modal")`. The deploy then uses the client
pinned with Talos. `cwd=MODAL_APP_ROOT` resolves to `site-packages` in a wheel install. `modal_app`
is a packaged module, so this should work. Verify it once by hand from a wheel install, with the
owner's go-ahead at that point. A deploy replaces the owner's deployed app with the same code.

### Python version

Modal refuses to deploy a `serialized=True` function whose image Python differs in minor version from
the deploying interpreter (`modal_app/talos_bench.py::_python`). `uv tool install` picks whichever
Python uv prefers, which may be newer than Modal supports. Builder's call, recorded here: the README
install line pins it (`uv tool install --python 3.12 talos-tig`), and the wheel-install CI job uses
the same command. `requires-python` gets no upper bound.

### Packaging

- Version: single source in `talos/__init__.py::__version__`, read by `pyproject.toml`. Add
  `talos --version`. The first release is `0.1.0`.
- Check that every non-Python file Talos reads at runtime is in the wheel. Unzip the built wheel and
  compare it with the files read via `Path(__file__)` and `importlib.resources`, including
  `talos/c3_jobdir.py::JOB_MODULES`, `talos/data/*.md`, and any native-runner sources.
- `pyproject.toml` gains `readme`, `urls` and `classifiers`.

### CI

New job in `ci.yml` (or a new workflow), on the 3-OS matrix:
`uv build` → `uv tool install dist/*.whl` → in a fresh empty directory: `talos --version`,
`talos guide`, `talos demo --delay 0`, `talos status --json` (it must list the demo job). The clone
job stays unchanged. That is the "clone install still works" check.

New `.github/workflows/publish.yml`, triggered on `v*` tags. One job, with `permissions: id-token:
write` on that job only and `environment: pypi`. Steps: check that the tag equals `v` +
`__version__`, `uv build`, then `pypa/gh-action-pypi-publish` with trusted publishing (no token
stored). Pushing a workflow file needs the SSH remote: the HTTPS token lacks the `workflow` scope.

### Docs

- `README.md` install: `uv tool install talos-tig` first, with `pipx install talos-tig` and
  `pip install talos-tig` as alternatives. Explain the workspace in one paragraph: run `talos setup`
  in the directory you want to work from, and Talos finds it from any directory below.
- `docs/windows.md`: the same with `uv` on Windows.
- `docs/maintainers.md`: the clone install for development, and a release section (bump
  `__version__`, tag, push, check the PyPI page, then `uvx --from talos-tig talos demo` in a clean
  directory).
- `talos guide` mentions `TALOS_WORKSPACE` for agents running outside the workspace.

### What the owner does

1. On PyPI: create the `talos-tig` project's trusted publisher (repo `FibonAdithya/talos`, workflow
   `publish.yml`, environment `pypi`).
2. On GitHub: create the `pypi` environment.
3. After PR 3 merges: push tag `v0.1.0`.

Between PR 3's merge and that tag, the README names a package that is not on PyPI yet. Keep that gap
short: merge, then tag.

### Done when

- The wheel-install CI job passes on Linux, macOS and Windows.
- The clone CI job passes unchanged.
- `make check` passes, including tests for `find_root` (env override, walk-up, fallback to the
  current directory, sandbox override).
- The Modal deploy from a wheel install was checked by hand, or that check is listed as open in the
  PR.
- After the owner publishes: `uvx --from talos-tig talos demo` in a clean directory (recorded as
  MEASURED with the date).

## Left to the builder

- Exact wording and order of sections inside each new doc.
- The exact JSON field names beyond those listed, as long as they are documented in `docs/usage.md`.
- Whether the wheel-install CI is a job in `ci.yml` or its own workflow.
