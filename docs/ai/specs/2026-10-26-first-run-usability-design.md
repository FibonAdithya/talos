# First-run usability: challenge status, CLI help, `talos demo`, `talos doctor`, wizard menus

Issues #60, #62, #63, #64, #65 (tracking issue #72, group 1 and 2). Not authoritative once
merged: the code and README win (AGENTS.md "Source of truth").

## Why

A new user (or a coding agent) meets Talos through `talos --help`, the `setup`/`run` wizards,
the no-account demo and the "Before your first real run" README section. Each of those
currently misleads or stalls them:

- the run wizard defaults to `vehicle_routing`, which has never been run, and lists all eight
  challenges as equal (#60);
- 4 of 5 subcommands and 14 of 22 `run` flags have no help text (#62);
- the demo is a hidden `--fake` flag and always wins on iteration 1 (#63);
- checking a backend works means running pytest live tests with env vars (#64);
- fixed-choice prompts are free text, a typo aborts the wizard, and nothing is confirmed
  before money is spent (#65).

## Decisions taken with the repo owner (2026-10-26)

- Branch `first-run-usability`, stacked on `run-report-html` (PR #80, unmerged). One PR for
  all five issues, at least one commit per issue. The PR targets `run-report-html` until #80
  merges, then is retargeted to `main`.
- Challenge statuses: **tested** = knapsack, hypergraph, energy_arbitrage; **known broken** =
  job_scheduling (#36); **untested** = satisfiability, vehicle_routing, vector_search,
  neuralnet_optimizer. "Tested" means at least one Talos job got past its baseline on some
  backend.
- The pre-run summary shows the budget caps and one line on how the backend bills. It shows
  **no per-iteration dollar figure**, because none has been measured for Modal.

## Goals

1. Every subcommand and every non-suppressed flag has help text; a test enforces it.
2. A visible `talos demo` that needs no config or network and shows each outcome a real job
   produces.
3. `talos doctor` that checks config, provider, backend, mainnet and GHCR for free, and
   optionally (`--full`, confirmed) compiles and scores one baseline.
4. Fixed-choice wizard prompts are numbered menus that re-prompt on a typo; `talos run`
   confirms a summary before it writes a job or touches a backend.
5. Each challenge carries a status that the wizard, the summary and the README show.

## Non-goals

- No change to scoring, budgets, nonce counts, beat rules, hardware classes, prices or the
  sandbox (AGENTS.md "What requires a human").
- No README split (#66), no `--json` for `status` (#68), no PyPI packaging (#70).
- No new dependency (no `rich`, no `questionary`): menus are plain `print`/`input`.
- No change to what a `--fake` run does; `--fake` stays hidden and stays the test path.
- Nobody, agent or test, runs `talos doctor --full` against a real backend in this work. It
  spends the user's compute (AGENTS.md: running live tests needs a human).

## Anti-goals

- A summary or doctor line that states a cost as fact when it is an estimate. Estimates are
  labelled `ESTIMATE`, measured figures carry their date.
- A confirmation that can be bypassed by a blank or closed stdin into spending: EOF on the
  confirmation counts as "no".
- Breaking scripted, non-interactive use: every new prompt is skipped by `--yes`, and every
  wizard answer can still be given by name.

## Part 1 — challenge status (#60)

- `talos/challenges.py`: `ChallengeSpec` gains `status: str` (one of `tested`, `untested`,
  `known broken`) and `issue: int | None` (the issue that explains a `known broken` status).
  A module-level `STATUSES = ("tested", "untested", "known broken")`. Values as decided above.
- Wizard default challenge becomes `knapsack`.
- The challenge menu (Part 4) shows each challenge's status and GPU marker, e.g.
  `3) knapsack — CPU, tested`, `7) job_scheduling — CPU, known broken (#36)`.
- Before a job starts (new job only, not resume) on an `untested` or `known broken`
  challenge, `talos run` prints one warning line to stderr, also under `--yes`:
  `warning: vehicle_routing has never been run end to end with Talos` /
  `warning: job_scheduling is known broken: see issue #36`.
- README: the `--challenge` row in the command reference gains a table (or a sub-table near
  it) of challenge, CPU/GPU, status.
- Tests: every `CHALLENGES` entry has a status in `STATUSES`; a `known broken` entry has an
  issue number; the README table lists exactly the challenges in `CHALLENGES` with the same
  statuses (parse the table; `tests/test_docs_references.py` is the existing home for
  README-vs-code checks).

## Part 2 — CLI help (#62)

- `main()` builds the parser with a top-level `description`, and every subparser gets `help=`
  (shown in `talos --help`) and `description=`.
- Every non-suppressed action gets one line of help. `--fake` stays `argparse.SUPPRESS`.
- `--challenge` on `run` and `compile` uses `choices=list(CHALLENGES)`. `compile --backend`
  already has choices. Note: argparse rejects an unknown `--challenge` with exit 2 before
  `cmd_run`; the existing "unknown challenge" message path in `cmd_run` stays for the wizard
  answer.
- `talos run` flags are grouped with `add_argument_group`: "what to research" (challenge,
  direction, direction-file, track, start, hyperparameters), "budget" (budget-*), "search"
  (mode, scoring, search, chains, nonces), "resume" (resume, add-*), and `--yes` in a group or
  left in the default options.
- `talos run` gets an `epilog` (with `RawDescriptionHelpFormatter`) with three examples: a
  first real run, a resume with `--add-iterations`, a run with `--direction-file`.
- Test: walk the parser tree from the one `main()` builds (expose a `build_parser()` so the
  test does not have to call `main`), and fail on any action, at any level, whose `help` is
  empty or `None` and not `SUPPRESS`, and on any subparser without help. The test must fail if
  one flag's help is removed (mutation check).

## Part 3 — `talos demo` (#63)

`talos demo [--challenge NAME] [--delay SECONDS]`, listed in `talos --help`.

- Needs no config, credentials or network: same fake provider/bench/mainnet as `--fake`,
  but its own scripted provider and scores. `--challenge` defaults to `knapsack`; any
  challenge in `CHALLENGES` is accepted (the fake bench handles GPU ones already).
- Writes a real job under `runs/demo-<YYYYmmdd-HHMMSS>-<challenge>/`, so `talos status` lists
  it and `talos report <job_id>` works on it. Checked 2026-10-26: nothing in `talos/`
  parses a job id as a timestamp.
- Scripted run of about six iterations, budget `iterations=6`, greedy, one chain, native
  scoring (the default, so the metered validation event appears), every event printed with
  the same `on_event` line format a real run uses. The script must produce, in order:
  1. an edit that does not compile, then a compile-fix round that repairs it, then a score
     with no improvement;
  2. at least one more no-improvement iteration (the "N in a row" counter visible);
  3. a candidate that improves on training, validates on the metered runtime, then fails
     held-out (`false_positive`);
  4. a candidate that improves, validates and is confirmed on held-out (`won`).
  The job ends `won` before the iteration budget is used up.
  How the script is keyed (markers in the file text, the training/held-out nonce range, a
  counter) is the builder's choice; it must be deterministic.
- `--delay` (default 0.4 s, `0` allowed) is the pause after each printed event, so the
  stream can be read as it scrolls. Tests pass `--delay 0` or patch the sleep.
- At the end, after the usual summary block: the package path, one line per package file
  saying what it is (the file list is whatever `build_package` wrote; a fixed description
  table with a fallback for an unknown name), `talos report <job_id>` for the HTML report,
  and the next step: `talos setup`, then `talos doctor`.
- Exit 0 when the demo ends `won` (it always should); non-zero otherwise.
- Tests: the timeline (`timeline.jsonl`) of a demo run contains, in order, `compile_failed`,
  an `iteration_done` with outcome `failed:score` and `runs_since_improvement >= 2`,
  `false_positive`, `won`; final status `won`; it ran with no `talos.config.json` present and
  with network access blocked (patch `urllib.request.urlopen` and `socket.create_connection`
  to raise). The package directory exists and the printed text names `talos setup`.

## Part 4 — wizard menus and pre-run confirmation (#65)

### Menus

One helper, e.g. `_ask_choice(ask, title, options, default) -> str`, where `options` is a
sequence of `(value, explanation)`. It builds the menu text:

```
Compute backend:
  1) modal  — your Modal account; pay per container second (default)
  2) c3     — your C3 account; ~12 min fixed overhead per job
  3) local  — Docker on this machine; free, slow builds
Choose
```

and calls `ask(menu_text, default_number)`. The whole menu goes in the `prompt` argument, so
the `ask` injection point is unchanged and a recording `ask` in tests sees the options. The
answer may be the number, the value (exact match after `strip()`; case-insensitive), or blank
for the default. Anything else prints `not one of the choices: 'x'` to stderr and asks again;
after 3 invalid answers the command exits 2 with that message (like `_ask_number`, which
raises `ConfigError`).

Prompts that become menus:

| Wizard | Prompt | Options |
|---|---|---|
| setup | Compute backend | `BACKENDS`, default modal |
| setup | Provider | `KINDS` minus `fake`, default anthropic |
| setup | Mode (CLI providers only) | single-shot, agentic; default single-shot |
| run | Challenge | `CHALLENGES` with CPU/GPU and status, default knapsack |
| run | Mode (CLI providers only) | single-shot, agentic; default the config's mode; the existing "5-20x tokens" line goes into agentic's explanation |
| run | Track | `all` then each of `info.tracks`, default all |
| run | Hyperparameters | mainnet, none; default mainnet |

Free-text prompts stay free text: model, API base, keys, tokens, direction, numbers.
Explanations per option are the builder's to write, one line each, factual, and must not
state a cost figure that is not already in the README or docs.

Existing tests that script answers by name keep working; tests whose prompt-recording
assertions check old prompt strings are updated. New tests: a typo then a valid answer
re-prompts and succeeds; three typos exit 2; a number selects the right value; blank selects
the default.

### Pre-run summary and confirmation

After every wizard question is answered and the read-only mainnet lookups the wizard needs
have run (challenge info, hyperparameters, template docs), and **before** the image-update
check, the job directory, `job.json`, or any backend call, `talos run` (new job, not
resume) prints:

```
Job summary
  challenge     knapsack (CPU, tested)
  scope         all 5 tracks, 8 nonces per track, fuel 2000000000
  provider      anthropic / claude-... , single-shot
  search        start mainnet, greedy, 1 chain, scoring native
  hyperparams   5/5 tracks from mainnet
  budget        LLM $20.00, 4.0 h, compute $20.00
  backend       modal: billed per container second, no fixed per-job overhead
  results       runs/20261026-153000-knapsack/
Start this job? [Y/n]
```

(Exact layout is the builder's; the fields are not.) The backend line per backend:
modal "billed per container second, no fixed per-job overhead"; c3 "one batch job per
iteration, about 12 minutes of fixed overhead each (docs/compute-backends.md#c3-timings)";
local "Docker on this machine, no compute charge". Unpriced/unmetered LLM spend is shown
the way `_llm_spend_text` words it. A challenge warning from Part 1 is part of the summary.

- The confirmation goes through `ask`, default `y`. `y`/`yes` (any case) proceeds; `n`/`no`
  exits 0 with `Not started.` and nothing written: no `runs/<job_id>/`, no bench built, no
  `recommend_image_update` call. Any other answer re-asks (3 tries, then exit 2). EOF from
  `default_ask` (closed stdin) is "no" — `default_ask` currently lets `EOFError` escape;
  the builder decides where to catch it, but a closed stdin must never start a job.
- `--yes` skips the confirmation and prints the summary anyway (so a log shows what ran).
  `--fake` runs follow the same rule (tests pass `--yes` or answer `y`).
- The job id shown is the one the job is created under (compute it once, before the summary).
- Tests: declining exits 0, leaves no `runs/` directory, and never constructs a bench or
  calls `recommend_image_update` (spy on both); accepting proceeds as today; `--yes` prints
  the summary without asking; the summary includes the challenge status and each budget cap.

## Part 5 — `talos doctor` (#64)

`talos doctor [--full] [--challenge NAME] [--yes] [--json]`. New module `talos/doctor.py`
holds the checks; `cli.py` only wires the command.

### Free checks (always run, in this order)

Each check returns `(name, status, detail)` with status `ok`, `warn`, `FAIL` or `skip`; a FAIL
detail ends with the command that fixes it. A check that depends on an earlier FAIL is `skip`
with the reason.

1. **config** — `talos.config.json` loads (`talos.config.load`). FAIL: `run talos setup`.
2. **credential** — for API providers, `resolve_api_key(cfg)` is not None (never print it);
   for `claude-cli`/`codex-cli`, the binary is on `PATH`. FAIL names the env var or the
   install.
3. **provider** — `validate_provider` (one tiny LLM call, as `talos setup` does). The detail
   says it made one call.
4. **backend** — by `cfg.backend`:
   - modal: look up (hydrate, no call) every function the deployed app should have for the
     current code (`compile_*`/`score_batch_*` per CPU challenge, per GPU per GPU challenge;
     builder derives the list from the same source `modal_app/talos_bench.py::register`
     uses). Any missing → FAIL `run talos setup to redeploy`. This is the version check the
     issue asks for, to the extent the code can make one without a container call: a deploy
     from older code lacks newer functions. A deploy whose functions exist but take older
     arguments is still caught only at the first real call (`talos/bench.py::_stale_deploy`);
     the doctor line says "deployed" not "up to date". Modal not installed or no token →
     FAIL with `talos setup`.
   - c3: `check_c3` (whoami + balance). Balance under £1 → `warn` with `c3 topup`.
   - local: `check_local` (docker info); detail includes CPUs, memory and whether the NVIDIA
     runtime is present (no NVIDIA runtime is `ok` with a note, not a warn).
5. **mainnet** — one fetch of the challenge list; every `CHALLENGES` entry's id and is_gpu
   match (the same comparison `cmd_run` does for one challenge). Unreachable → FAIL;
   mismatch → FAIL naming the challenge and `update talos/challenges.py`. Builder may add a
   `mainnet` helper that returns all challenges from one block lookup.
6. **images** — `image_available` for every challenge, in parallel; any missing → FAIL
   naming them and pointing at `talos check-updates`. Applies to every backend (Modal builds
   from the same GHCR image).

Checks 2–4 need the config; if check 1 fails they are `skip`. Checks 5–6 always run.

### `--full` (paid)

Compiles and scores the mainnet top algorithm for `--challenge` (default knapsack) on the
configured backend, 2 training nonces of the first track, no held-out — the body of
`tests/test_live.py::test_baseline_compiles_and_scores`, generalised to the bench
`make_bench` returns (with a temporary run dir, an in-memory pending store, and
`select_hardware` first). Runs only after every free check passed (any FAIL → `--full` is
skipped and the command exits 1).

Before spending, it prints what it will do and what it may cost, then asks
`Spend this? [y/N]` through `ask`, default **no**:
- c3: "MEASURED 2026-09-15: £0.02 and 12 min 20 s for one knapsack job".
- modal: an `ESTIMATE (unverified)` from the rates in `talos/bench.py` for the challenge's
  container class and an assumed duration the builder states in the text.
- local: "no compute charge; the first run pulls a 13 GB image".
- Any challenge in `CHALLENGES` is accepted. The C3 figure above is printed **only** for
  knapsack on C3; every other challenge/backend pair says "not measured for <challenge>" plus
  an `ESTIMATE (unverified)` from the rates in `talos/bench.py` / `talos/c3_bench.py`. For a
  GPU challenge the estimate is a range across the GPU options (the GPU is chosen by
  `select_hardware`, which itself launches paid probes and so runs only after confirmation),
  and the text says GPU nonces can take many minutes each. (Builder-check question; my call.)
`--yes` skips the question. EOF or anything but y/yes → exit 1 with `--full not run`, and no
bench is constructed. Output: compile ok/failed, nonces ok/errored, and the bench's
`cost_usd_since` as "compute spend (estimated)".

### Output

Text: one line per check, `name: ok`, `name: warn: <detail>`, `name: FAIL: <detail; fix>`,
`name: skip: <reason>`. `--json`: one JSON object
`{"checks": [{"name", "status", "detail"}], "ok": bool}` on stdout, nothing else on stdout
(warnings that existing helpers print to stderr may stay on stderr). Exit 0 when no check
is FAIL, else 1. With `--json --full` and no `--yes`, the confirmation is refused (no
interactive prompt in JSON mode) and `--full` is reported as `skip: needs --yes`. A `--full`
that was requested but did not run (declined, refused, or skipped after a failed free check)
exits 1 in both modes, so a script never reads success when nothing was compiled.
(Builder-check question; my call.)

### Tests

Every backend and network call is stubbed: `validate_provider`, `check_c3`, `check_local`,
the Modal lookup, the mainnet fetch, `image_available`, `make_bench`. Memory note: a cli test
that lets a real `c3` subprocess run has created a real C3 job before; the doctor tests must
not reach `subprocess.run` for `c3`, `modal` or `docker` (assert by patching
`subprocess.run` to raise in the doctor test module).

- all checks pass → exit 0, one `ok` line each;
- each check's failure → exit 1 and its FAIL line contains the fix command;
- no config → checks 2–4 `skip`, 5–6 still run;
- `--full` without `--yes`, answer blank/no/EOF → exit 1 and `make_bench` never called;
- `--full --yes` → bench evaluated once with 2 training nonces and no held-out;
- `--full` with a failing free check → bench never built;
- `--json` output parses and has every check.

### README

- "Before your first real run" (and the Windows section's live smoke test step) becomes
  `talos doctor`, then `talos doctor --full` once. The pytest "Live smoke test" section stays,
  retitled for maintainers.
- "Try it with no accounts" (both OS sections) uses `talos demo`.
- Command reference gains `talos demo` and `talos doctor` entries.
- Before pushing, grep the README, AGENTS.md and docs for backend enumerations and command
  lists the change extends (the docs review bot flags stale ones).

## Constraints

- `ask` stays the single injection point; tests never read real stdin.
- `make check` (ruff, pytest without `live`, agentify) stays green on all three OSes in CI;
  menu and summary text is plain ASCII (`-` rather than the em dash in the examples above).
  The status line already prints `∞`/`≈`, but ASCII costs nothing here. (My call.)
- New tests are mutation-checked: break the code they guard, see them fail, restore.

## Left to the builder

- Module layout beyond `talos/doctor.py` (e.g. a `talos/wizard.py` for the menu helper).
- How the demo script is keyed, and its exact iteration count (about six).
- Option explanation wording, summary layout, help strings.
- The Modal duration assumption for the `--full` estimate (stated in the printed text).
