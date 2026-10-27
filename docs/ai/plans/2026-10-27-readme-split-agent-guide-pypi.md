# README split, agent guide, PyPI install: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship issues #66, #71 and #70 as three PRs: a short README with reference docs, a `talos guide` for coding agents plus `talos status --json`, and a PyPI-installable Talos that finds its workspace by walking up.

**Architecture:** Part A is documentation only. Part B adds `talos/status.py` (job summaries for `talos status`), a `guide` subcommand that prints packaged markdown, and a drift test that ties the guide to `build_parser()`. Part C adds `talos/workspace.py::find_root` and routes every state path through it, then adds packaging metadata, a wheel-install CI job and a tag-triggered publish workflow.

**Tech Stack:** Python 3.10+, argparse, pytest, setuptools, uv, GitHub Actions.

**Spec:** `docs/ai/specs/2026-10-27-readme-split-agent-guide-pypi-design.md`

## Global Constraints

- Each part is its own branch from the latest `origin/main`, one PR each. Merge order: A (`readme-split`), B (`agent-guide`), C (`pypi-install`). B and C branch only after the previous PR has merged.
- Gate: `make check PYTHON=.venv/bin/python` (ruff, `pytest -m "not live"`, `agentify check`) passes at every commit.
- Every number written into any doc is MEASURED (with date and the command or file it came from) or `ESTIMATE (unverified)`.
- Stage explicit paths only. Never `git add -A` / `git add .`.
- No test may reach Modal, C3, Docker, GHCR, mainnet or an LLM. Before mutation-checking any `tests/test_cli.py` test, confirm `subprocess.run` and `deploy_bench` are stubbed in it.
- Mutation checks run with `PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=$(mktemp -d)`, because a stale `.pyc` gives a false MISSED.
- `docs/ai/` is not authoritative. Do not link to it from user docs.

## Review Focus

1. **A user who runs `talos` from a subdirectory of a clone** (Part C) expects the clone's config and `runs/`. Test: `test_status_from_a_subdirectory_lists_the_workspace_jobs` in Task C1.
2. **`TALOS_WORKSPACE` set to a directory that does not exist** (Part C). A silent fallback to the current directory would write `runs/` in the wrong place. Expect exit 2 with `TALOS_WORKSPACE=<path> is not a directory`. Test: `test_find_root_rejects_a_missing_workspace` in Task C1.
3. **User-supplied relative paths (`--dir`, `--direction-file`) from a subdirectory** (Part C). They must stay relative to the current directory, not the workspace root. Test: `test_compile_dir_stays_relative_to_cwd` in Task C1.
4. **An agent polling `status --json` while the job writes** (Part B). `state.json` is atomic, but `timeline.jsonl` can have a half-written last line. Expect the partial line to be skipped, not an exception. Test: `test_heldout_ignores_a_truncated_last_timeline_line` in Task B1.
5. **A `kill -INT` sent to a background `talos run` before its loop starts** (Part B). Expect it to stop the command, not be ignored. Test: `test_run_restores_the_default_sigint_handler_on_entry` in Task B2.

---

# Part A: split the README (#66), branch `readme-split`

Section map. Every heading of today's `README.md` goes to exactly one place.

| Old heading | New home |
|---|---|
| intro paragraphs, How it works, Prerequisites, Try it with no accounts | `README.md` (How it works shortened to one paragraph plus links) |
| Setup 1 Install, 2 backend, 3 provider | `README.md` (tables condensed: the `local` row's MEASURED timings move to "What to expect") |
| Setup 4 `talos setup` (the prompt list, checks, files written) | `docs/usage.md#talos-setup`; README keeps one paragraph |
| Setup 5 Before your first real run | `README.md`, one paragraph |
| Running on Windows (all but Development on Windows) | `docs/windows.md` |
| Development on Windows | `docs/maintainers.md` |
| Running a job: Start, Watch, Stop, Resume, Submit | `docs/usage.md` (Resume's `0.0.7 → 0.0.8` paragraph goes to `docs/maintainers.md#upgrading-the-tig-dev-image`) |
| Command reference and every `### talos …` | `docs/usage.md` (`check-updates`' "To apply an upgrade" list goes to `docs/maintainers.md#upgrading-the-tig-dev-image`) |
| Challenge status table (now under `talos run`) | `README.md` (read there by `tests/test_challenges.py::test_readme_status_table_matches_the_code`) |
| Agentic mode | `docs/usage.md` |
| Where results land | `docs/usage.md`; README keeps a 5-line summary |
| Budget | `docs/usage.md` |
| Live smoke tests (maintainers), Development | `docs/maintainers.md` |
| Licence | `README.md` |

### Task A1: reference docs (usage, windows, maintainers)

**Files:**
- Create: `docs/usage.md`, `docs/windows.md`, `docs/maintainers.md`

**Interfaces:**
- Produces: anchors that Task A3 links to. Use these exact headings: `docs/usage.md` → `## talos setup`, `## Running a job`, `### Start`, `### Watch`, `### Stop`, `### Resume`, `### Submit`, `## Command reference`, one `### talos <command>` per command, `## Agentic mode`, `## Where results land`, `## Budget`. `docs/maintainers.md` → `## Upgrading the TIG dev image`, `## Live smoke tests`, `## Development`, `### Development on Windows`. `docs/windows.md` keeps today's step headings.

- [ ] **Step 1: Branch.** `git fetch origin && git switch -c readme-split origin/main`. The untracked spec and plan carry over across the switch. Commit them alone: `git add docs/ai/specs/2026-10-27-readme-split-agent-guide-pypi-design.md docs/ai/plans/2026-10-27-readme-split-agent-guide-pypi.md`, message `docs: design and plan for #66, #71, #70`.
- [ ] **Step 2: Write the three files** by moving text from `README.md` per the section map. Move it, do not rewrite it, except for these changes:
  - Each long flag row in the `talos run` table (`--challenge`, `--track`, `--nonces`, `--hyperparameters`, `--scoring`, `--start`) becomes one or two sentences. The rest goes into a paragraph under the table, headed by the flag name in bold. `talos/calibration.py::calibration_key` and `talos/challenges.py::TEMPLATE_START_FILES` leave the table and go into those paragraphs.
  - Fix relative links for the new depth: `docs/x.md` becomes `x.md`, and `README.md#…` becomes `../README.md#…`.
  - `docs/usage.md` opens with one line saying it is the full reference and the README is the quickstart.
- [ ] **Step 3: Gate.** Run `make check PYTHON=.venv/bin/python`. Expected: PASS. The README still has everything at this point, so nothing is broken yet.
- [ ] **Step 4: Commit** `docs/usage.md docs/windows.md docs/maintainers.md`, message `docs: move the run reference, Windows and maintainer material into docs/ (#66)`.

### Task A2: troubleshooting

**Files:**
- Create: `docs/troubleshooting.md`

- [ ] **Step 1: Collect symptoms from the code.** For each entry below, `grep -rn` `talos/` for the message and copy it verbatim as the entry's **Symptom**. If no message exists, write down the grep you ran and drop the entry. The PR body lists dropped entries.
  - `modal deploy failed` (`talos/cli.py::deploy_bench`)
  - a client reaching an older Modal deploy (grep `talos setup` in `talos/bench.py` and `modal_app/`)
  - C3 CLI older than 0.20.0 refusing the GHCR image, and `IMAGE_HINT`
  - C3 balance below £1 warning
  - local container CPUs or memory above Docker's limits (`talos/cli.py`, setup's local branch)
  - `unpriced` model or `codex-cli` with only `--budget-usd` (refusal message in `cmd_run`)
  - agentic codex refused without `TALOS_ALLOW_CODEX_AGENTIC=1`
  - job `paused` (backend unreachable); job `failed` (the reason is in `state.json` `stop_reason`)
  - resume refused after the dev image upgrade
  - a sandbox compile on C3 still running after the agent's 30-minute timeout
  - Windows: one line linking `windows.md#what-differs-on-windows`
- [ ] **Step 2: Write `docs/troubleshooting.md`.** Start with "Run `talos doctor` first." Then one `###` per entry with **Symptom**, **Cause** and **Fix** (the command to run).
- [ ] **Step 3: Gate**, then commit `docs/troubleshooting.md`, message `docs: add troubleshooting with symptoms quoted from the code (#66)`.

### Task A3: the short README and links

**Files:**
- Modify: `README.md` (rewrite), `AGENTS.md`, `docs/architecture.md`, `docs/compute-backends.md`

- [ ] **Step 1: Snapshot** the old README: `git show origin/main:README.md > $SCRATCH/old_readme.md`. `$SCRATCH` is the session scratchpad.
- [ ] **Step 2: Rewrite `README.md`**, in this order: intro (the two paragraphs today); install (clone and venv, link to `docs/windows.md`); `talos demo`; backend table (one short sentence per row) and provider table; `talos setup` and `talos doctor` (one paragraph each, linking `docs/usage.md#talos-setup` and `docs/usage.md#talos-doctor`); first real run (the knapsack `talos run … --yes` command); **What to expect**; challenge status table (exact rows from today, same format: the test's regex is `^\| \`([a-z_]+)\` \| (CPU|GPU) \| ([^|]+) \|$`); where results land (5 lines); Documentation links (usage, troubleshooting, windows, maintainers, architecture, compute-backends); Licence.
  - **What to expect** table: columns Backend | Time per iteration | Compute cost per iteration. Fill only from figures in the repo:
    - C3: about 12 minutes of fixed overhead per iteration, from `docs/compute-backends.md`. Its cost is the measured £0.02 for one knapsack baseline (MEASURED 2026-09-15). Call it the closest measured figure; it is not a per-iteration cost.
    - local: about 3 minutes for one knapsack build and 4 nonces (MEASURED 2026-09-25), and £0.
    - Modal: if no figure in the repo exists, write `not measured` in both cells. Do not invent one.

    Then one sentence: most jobs end `exhausted` because a budget ran out, an `exhausted` job's package is not a candidate to submit, and `--add-*` continues it.
- [ ] **Step 3: Content check.** Every flag, MEASURED date and `talos` subcommand in the old README must survive somewhere:

```bash
python3 - "$SCRATCH/old_readme.md" README.md docs/usage.md docs/troubleshooting.md docs/maintainers.md docs/windows.md <<'EOF'
import re, sys
old = open(sys.argv[1]).read(); new = "".join(open(p).read() for p in sys.argv[2:])
missing = sorted({m for pat in (r"--[a-z][a-z-]+", r"MEASURED \d{4}-\d\d-\d\d", r"`talos [a-z-]+")
                  for m in re.findall(pat, old)} - {m for pat in (r"--[a-z][a-z-]+",
                  r"MEASURED \d{4}-\d\d-\d\d", r"`talos [a-z-]+") for m in re.findall(pat, new)})
print("missing:", missing); sys.exit(bool(missing))
EOF
```
  Expected: `missing: []`, exit 0.
- [ ] **Step 4: Fix links into moved anchors.** Run `grep -rn 'README.md#\|README)' AGENTS.md docs/*.md talos/`. Repoint each one, at least these:
  - `AGENTS.md`: `README.md#where-results-land` → `docs/usage.md#where-results-land`, `README.md#budget` → `docs/usage.md#budget`, `README.md#live-smoke-tests-maintainers` → `docs/maintainers.md#live-smoke-tests`.
  - `AGENTS.md` source-of-truth item 2 becomes: "`README.md` (quickstart), `docs/usage.md` (full command reference), `docs/troubleshooting.md`, `docs/windows.md` and `docs/maintainers.md`".
  - The lookup-table row "Set up and run day-to-day commands" points to `README.md`, `docs/usage.md`.
  - `docs/architecture.md`: `../README.md#agentic-mode` → `usage.md#agentic-mode`; "the README's budget section" → `[Budget](usage.md#budget)`.
  - `docs/compute-backends.md`: `../README.md#2-pick-a-compute-backend` → the README's new backend heading anchor; `../README.md#where-results-land` → `usage.md#where-results-land`.
- [ ] **Step 5: Verify.** `wc -l README.md` must print under 200. `make check PYTHON=.venv/bin/python` must pass; agentify's reference check resolves every anchor.
- [ ] **Step 6: Commit** `README.md AGENTS.md docs/architecture.md docs/compute-backends.md`, message `docs: cut the README to a quickstart and repoint links (#66)`.
- [ ] **Step 7: PR.** Probe SSH first (`timeout 10 ssh -o BatchMode=yes -o ConnectTimeout=8 -T git@github.com`), then push. Open the PR against `main` with the section map above, the A2 dropped-entry list, and the measured `wc -l`. Closes #66. Follow the `review-pr` loop to merge.

---

# Part B: `talos status --json`, `talos guide` (#71), branch `agent-guide`

### Task B1: job summaries and `talos status [JOB_ID] [--json]`

**Files:**
- Create: `talos/status.py`, `tests/test_status.py`
- Modify: `talos/cli.py` (`cmd_status`; the `status` subparser in `build_parser`)

**Interfaces:**
- Consumes: `talos/state.py::JobState`, `JobSpec`; `talos/budget.py::effective_budget`; `talos/cli.py::_llm_spend_text` (move its label logic into `talos/status.py::llm_tracking(provider, model) -> str`, returning `"tracked"`, `"unpriced"` or `"not tracked"`, and make `_llm_spend_text` call it).
- Produces: `talos/status.py::job_summary(run_dir: Path) -> dict` returning exactly these keys:
  `job_id: str`, `status: str`, `stop_reason: str|None`, `iteration: int`, `challenge: str|None`,
  `spend: {"llm_usd": float, "compute_usd": float, "llm_tracking": str|None}`,
  `budget: {"usd", "hours", "iterations", "compute_usd"}|None` (the effective budget, or `None` when `job.json` is unreadable),
  `best: {"iteration": int, "delta": float}|None` (`delta` = `state.best.delta["mean_rel_delta"]`),
  `heldout: {"result": "won"|"false_positive"|"not_run", "delta": float|None}|None` (`None` when `best` is `None`),
  `package: str|None` (absolute path of `run_dir/"package"` if it is a directory).
  `heldout.result`: `won` if the best iteration is in `state.confirmed`, `false_positive` if it is in `state.false_positives`, otherwise `not_run`. `heldout.delta` is `holdout["mean_rel_delta"]` from the **last** `timeline.jsonl` row whose `kind` is `won` or `false_positive` and whose `iteration` equals the best iteration. If there is no such row, or the row has no `holdout` (a `false_positive` from a `ScoringError` carries `error` instead), it is `None`.

Test fixture: `demo_job(tmp_path, monkeypatch) -> Path`. It chdirs to `tmp_path`, runs `cli.main(["demo", "--delay", "0"])` and returns the one `runs/demo-*` directory. Tests then edit that directory's `state.json` / `timeline.jsonl` / `job.json` to set controlled values. Expected values are always the literals the test wrote, never values read back through `talos/status.py`.

- [ ] **Step 1: Write the failing tests** in `tests/test_status.py`:
  - `test_heldout_delta_comes_from_the_matching_timeline_event`: append `{"kind": "won", "iteration": <best>, "holdout": {"mean_rel_delta": 0.0421}}` and an earlier-iteration `won` row with `0.9`. Assert `job_summary(d)["heldout"] == {"result": "won", "delta": 0.0421}`. Catches reading the first event, or the wrong iteration.
  - `test_false_positive_reports_its_heldout_delta`: set `confirmed=[]`, `false_positives=[best]`, `status="exhausted"`, and append a `false_positive` row with `-0.0123`. Assert `{"result": "false_positive", "delta": -0.0123}`.
  - `test_unscored_best_is_not_run`: `confirmed=[]`, `false_positives=[]`. Assert `{"result": "not_run", "delta": None}`.
  - `test_scoring_error_false_positive_has_no_delta`: a `false_positive` row with `error` and no `holdout`. Assert `delta is None`.
  - `test_no_best_means_no_heldout`: `best=None`. Assert `best is None and heldout is None`.
  - `test_heldout_ignores_a_truncated_last_timeline_line`: append a complete `won` row for the best iteration with `0.0421`, then `'{"kind": "won", "iter'` with no newline. Assert no exception, and `heldout["delta"] == 0.0421`.
  - `test_budget_includes_extensions`: set `job.json` budget iterations to 5 and add `budget_extensions=[{"add": {"iterations": 7}}]` (plus whatever keys `effective_budget` reads). Assert `budget["iterations"] == 12`.
  - `test_codex_job_spend_is_not_tracked`: set `job.json` `provider="codex-cli"`. Assert `spend["llm_tracking"] == "not tracked"`.
  - `test_unreadable_job_json_still_listed`: write `{` to `job.json`. Assert `challenge is None`, `budget is None`, `spend["llm_tracking"] is None`, and that `status` is still read.
  - `test_package_path_is_absolute_or_null`: after the demo it equals `str((d / "package").resolve())`; after `shutil.rmtree(d / "package")` it is `None`.
  - `test_status_json_lists_every_job_sorted` (through `cli.main`): two demo jobs. `json.loads(capsys out)["jobs"]` gives their `job_id`s in sorted order, and stdout holds nothing else.
  - `test_status_job_id_json_prints_one_object`: `cli.main(["status", id, "--json"])` gives an object whose `job_id == id`.
  - `test_status_unknown_job_exits_1`: `cli.main(["status", "nope"]) == 1`, and stderr contains `no job nope under runs/`.
  - `test_status_text_line_unchanged`: `cli.main(["status"])` prints exactly `f"{id}: won it={it} llm=$0.00 compute=$0.00\n"`, where `it` is the integer the test wrote into `state.json`.
- [ ] **Step 2: Run** `.venv/bin/pytest tests/test_status.py -q`. Expected: FAIL (`ModuleNotFoundError: talos.status`).
- [ ] **Step 3: Implement** `talos/status.py` (`job_summary`, `llm_tracking`, a private timeline reader that skips lines failing `json.loads`). Add `status` arguments: `job_id` (`nargs="?"`) and `--json` (`action="store_true"`). `cmd_status` text mode keeps today's line, built from `job_summary`.
- [ ] **Step 4: Run** `tests/test_status.py` and `tests/test_cli.py`. Expected: PASS.
- [ ] **Step 5: Mutation check.** Apply each mutation, run `tests/test_status.py` and confirm it fails, then restore: (a) take the first matching event instead of the last; (b) ignore `iteration` when matching; (c) drop `effective_budget`, using the raw budget; (d) map `false_positives` to `not_run`; (e) `json.loads` without try. Record the results in the commit message body.
- [ ] **Step 6: Commit** `talos/status.py talos/cli.py tests/test_status.py`, message `feat: talos status [JOB_ID] [--json] for agents polling a job (#71)`.

### Task B2: a background `talos run` stops on SIGINT from its first line

**Files:**
- Modify: `talos/cli.py` (`cmd_run`, first statement)
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test** `test_run_restores_the_default_sigint_handler_on_entry`:
  - Set `signal.signal(signal.SIGINT, signal.SIG_IGN)`, and restore it in `finally`.
  - Monkeypatch `cli.fetch_challenge_info` with a function that records `signal.getsignal(signal.SIGINT)` and then raises `MainnetError("stop")`.
  - Run `cli.main(["run", "--challenge", "knapsack", "--direction", "go", "--budget-iterations", "1", "--yes"])` in a tmp workspace with a fake config, as in `test_run_requires_budget_and_creates_job`.
  - Assert the recorded handler `is signal.default_int_handler`.
- [ ] **Step 2: Run it.** Expected: FAIL (the recorded handler is `SIG_IGN`).
- [ ] **Step 3: Implement.**
  - The first line of `cmd_run` is `signal.signal(signal.SIGINT, signal.default_int_handler)`, with a one-line comment: a shell's `&` launch inherits SIGINT ignored.
  - Rename today's body to `_cmd_run(args, ask)`. The new `cmd_run` sets the handler, then calls `_cmd_run` inside `try/except KeyboardInterrupt`; the except prints `interrupted` to stderr and returns 130.
  - Scope the catch to `run` only, not `main()`: `tests/test_cli.py::test_compile_local_removes_its_containers_when_evaluate_raises` asserts that a `KeyboardInterrupt` escapes `talos compile`, and must keep passing unchanged.
  - Add the test `test_early_interrupt_exits_130_without_traceback`: `cli.fetch_challenge_info` raises `KeyboardInterrupt`. `cli.main(["run", …]) == 130` with the same args as the Step 1 test, stderr is `interrupted\n`, and no `runs/*/job.json` exists.
  - **Existing test to update:** `tests/test_cli.py::test_sigint_stops_the_bench_too` patches `cli.signal.signal` with `handlers.setdefault(sig, h)`, which keeps the *first* SIGINT handler set. After this change that handler is `default_int_handler`, so the test would call it and raise. Change it to record a list (`handlers.setdefault(sig, []).append(h)`). Assert `handlers[signal.SIGINT][0] is signal.default_int_handler`, then call `handlers[signal.SIGINT][1](None, None)`, which is the loop's stop handler. Keep `stops == ["bench"]`.
- [ ] **Step 4: Run** `tests/test_cli.py`. Expected: PASS.
- [ ] **Step 5: Real-process check (Linux).** `--fake` needs no config and no network (`cmd_run`), so both cases use it in an empty scratch directory.
  - **Early stop:** without `--yes`, `talos run` blocks at `Start this job?` before the loop. Give it a stdin that never closes: `mkfifo in; (sleep 30 > in &); bash -c '.venv/bin/talos run --fake --challenge knapsack --direction go --budget-iterations 1 < in > early.log 2>&1 & pid=$!; sleep 2; kill -INT $pid; wait $pid; echo rc=$?'`. Expected: `rc=130`, `early.log` ends with `interrupted`, and no `runs/` directory exists.
  - **Late stop:** run `bash -c '.venv/bin/talos run --fake --challenge knapsack --direction go --budget-iterations 100000 --yes > late.log 2>&1 & pid=$!; sleep 5; kill -INT $pid; wait $pid; echo rc=$?'`. Expected: `late.log` has `Status: cancelled`, and the package path is printed.
  - If the fake loop finishes 100000 iterations within 5 s, raise the count and record it.
  - Record both outputs (MEASURED). The guide's stop instructions in B3 are written from them.
- [ ] **Step 6: Commit** `talos/cli.py tests/test_cli.py`, message `fix: a background talos run honours SIGINT before its loop starts (#71)`.

### Task B3: `talos guide`, the guide, and its drift test

**Files:**
- Create: `talos/data/agent_guide.md`, `tests/test_guide.py`
- Modify: `talos/cli.py` (`guide` subparser, `cmd_guide`, `COMMANDS`), `README.md` (one line), `docs/usage.md` (`### talos guide` and the `status` flags)

**Interfaces:**
- Produces: `talos/cli.py::cmd_guide(args, ask) -> int`, which prints `resources.files("talos.data").joinpath("agent_guide.md").read_text(encoding="utf-8")` with `end=""` and returns 0.

- [ ] **Step 1: Write the failing tests** in `tests/test_guide.py`. The helper `unknown_references(text: str, parser: argparse.ArgumentParser) -> list[str]` lives in the test file:
  - Collect code from fenced blocks and from inline backtick spans.
  - In each code line, find every `talos <word>` (regex `\btalos ([a-z][a-z-]*)`), skipping `talos-` names.
  - The subcommand must be in the parser's subparser choices, or the result gets `"talos <word>"`.
  - Every `--flag` token after it on the same line, up to the first `|`, `;`, `&&`, `>` or `&`, must be in that subparser's `_option_string_actions`, or the result gets `"talos <word> --flag"`.

  Tests:
  - `test_checker_flags_unknown_subcommands_and_flags`: `unknown_references("`talos shw`\n```\ntalos run --budget-dollars 3 --yes\n```", build_parser())` `sorted(...) == ["talos run --budget-dollars", "talos shw"]`. The comparison is sorted because the order of collection (fenced blocks or inline spans first) is the implementer's choice. This proves the checker is not vacuous. It catches a checker that skips inline spans, skips fenced blocks or skips flags.
  - `test_checker_ignores_flags_after_a_pipe`: `"`talos status --json | jq --raw`"` returns `[]`.
  - `test_guide_names_only_real_commands_and_flags`: `unknown_references(<packaged guide>, build_parser()) == []`.
  - `test_guide_mentions_every_spending_command`: the guide contains `talos run`, `talos doctor --full` and `talos compile`, plus the stop instruction `kill -INT`. This is a content guard only: it catches deleting the spending or stop section, not wrong advice in it. Accepted as such.
  - `test_talos_guide_prints_the_packaged_file`: `cli.main(["guide"]) == 0`, and the captured stdout equals the packaged file's text.
- [ ] **Step 2: Run** them. Expected: FAIL (no `guide` subcommand, no file).
- [ ] **Step 3: Write `talos/data/agent_guide.md`** with the seven sections of spec "Guide content". Points that must be exact:
  - **Free:** `talos guide`, `talos demo`, `talos status`, `talos report`, `talos check-updates`, `talos compile --backend local`.
  - **No compute, one small LLM call:** `talos doctor`.
  - **Spends:** `talos run`, `talos doctor --full`, `talos compile` on modal or c3.
  - **Stop:** one `kill -INT <pid>`, worded from the B2 Step 5 outputs. Never `kill -9`.
  - **Windows:** state what was verified, or say it is unverified.
  - **Status values:** today's list from `docs/usage.md#talos-status`.
  - **Report template:** fields outcome, best training delta, held-out result and delta, spend with its `llm_tracking`, package path, next step.
  - **Troubleshooting link:** `https://github.com/FibonAdithya/talos/blob/main/docs/troubleshooting.md`.
  - **Direction file:** one worked example, under 15 lines.
  - **Budget presets:** each labelled MEASURED or `ESTIMATE (unverified)`.
- [ ] **Step 4: Add** the `guide` subparser (help: `print the guide for coding agents driving Talos`), `cmd_guide`, and the `COMMANDS` entry. Add a line to `README.md` under the intro: `Driving Talos from a coding agent? Tell it: "Run \`talos guide\` and follow it."` Add `### talos guide` and the new `status` arguments to `docs/usage.md`.
- [ ] **Step 5: Run** `make check PYTHON=.venv/bin/python`. Expected: PASS.
- [ ] **Step 6: Mutation check.** (a) Rename `--budget-iterations` to `--budget-iters` in the parser: `test_guide_names_only_real_commands_and_flags` fails. (b) Make `cmd_guide` read `rust_rules.md`: the print test fails. (c) Make the checker return `[]` always: the checker test fails. Restore each one.
- [ ] **Step 7: Commit** `talos/data/agent_guide.md tests/test_guide.py talos/cli.py README.md docs/usage.md`, message `feat: talos guide prints a guide for any coding agent (#71)`.

### Task B4: two-agent trial

**Files:**
- Create: `docs/ai/reports/2026-10-27-agent-guide-trial.md`. Stage this file only: `docs/ai/reports/` holds other untracked work that must stay out of the commit.

- [ ] **Step 1: Throwaway clone.**
  - Run `git clone <repo path> $SCRATCH/trial && cd $SCRATCH/trial && git switch agent-guide && uv venv && uv pip install -e .`.
  - Confirm it has no `talos.config.json` and no `.talos/`.
  - Each agent gets its own fresh clone (`trial-claude`, `trial-codex`), so one agent's `runs/` does not affect the other.
- [ ] **Step 2: Claude Code.** Run it in the background with a 900 s timeout, output to `$SCRATCH/trial-claude.log`:
  `cd $SCRATCH/trial-claude && PATH=$PWD/.venv/bin:$PATH timeout 900 claude -p "Run \`talos guide\` and follow it to run the Talos demo, then report back to me as the guide says." --allowedTools "Bash(talos:*)" Read Grep Glob`.
- [ ] **Step 3: Codex.** Same setup, in `trial-codex`:
  `timeout 900 codex exec --sandbox workspace-write -C $SCRATCH/trial-codex "<same prompt>"`.
- [ ] **Step 4: Write the report.** Caveat for the report: both CLIs also load the owner's global instructions (`~/.claude/CLAUDE.md`, `~/.codex/`), so this is not a clean-room agent. Say so. For each agent: the commands it ran, where it got stuck or guessed, whether its final report followed the template, and its wall time (MEASURED). Fix the guide for each problem found. Re-run both once, then append the re-run results.
- [ ] **Step 5: Commit** the report and any guide fix, message `docs: two-agent trial of talos guide against the demo (#71)`.
- [ ] **Step 6: PR.** Push and open the PR (closes #71), with the trial summary in the body. Follow the `review-pr` loop to merge.

---

# Part C: PyPI install, workspace by walking up (#70), branch `pypi-install`

### Task C1: `find_root` and every state path through it

**Files:**
- Create: `talos/workspace.py`, `tests/test_workspace.py`, `tests/conftest.py`
- Modify: `talos/cli.py` (the 9 `Path.cwd()` sites at `cmd_setup`, `cmd_run`, `cmd_demo`, `cmd_compile` ×3, `cmd_status`, `cmd_report`, `cmd_doctor`; find them with `grep -n 'Path.cwd()' talos/cli.py`), `talos/status.py` if it builds `runs/` paths itself, `talos/agentic.py` (`_agent_env`)

**Interfaces:**
- Produces:
  - `talos/workspace.py::find_root(start: Path | None = None) -> Path`. Rules:
    1. `TALOS_WORKSPACE` is set → if it is a directory, return `Path(it).resolve()`; otherwise raise `WorkspaceError(f"TALOS_WORKSPACE={value} is not a directory")`. `class WorkspaceError(ConfigError)` lives in `talos/workspace.py`.
    2. Walk from `(start or Path.cwd()).resolve()` through its parents. Return the first directory containing `talos.config.json`.
    3. Otherwise return `(start or Path.cwd()).resolve()`.
  - `ENV_VAR = "TALOS_WORKSPACE"`.
  - `talos/agentic.py::_agent_env(worktree: Path) -> dict[str, str]`. It now sets `env["TALOS_WORKSPACE"] = str(worktree)`, overriding any inherited value, and its caller passes `wt`.
- `main()` catches `WorkspaceError` only (not every `ConfigError`, which commands already handle in their own ways), prints it to stderr and returns 2.

- [ ] **Step 1: Add `tests/conftest.py`** with an autouse fixture: `monkeypatch.delenv("TALOS_WORKSPACE", raising=False)`. Without it, a developer's exported variable breaks every chdir-based test.
- [ ] **Step 2: Write the failing tests** in `tests/test_workspace.py`:
  - `test_find_root_walks_up_to_the_config`: config at `tmp/ws`, start at `tmp/ws/a/b`. Expect `tmp/ws` resolved.
  - `test_find_root_nearest_config_wins`: configs at `tmp/ws` and `tmp/ws/a`, start at `tmp/ws/a/b`. Expect `tmp/ws/a`.
  - `test_find_root_falls_back_to_start`: no config, start at `tmp/x`. Expect `tmp/x`. Skip if any parent of `tmp_path` has `talos.config.json`.
  - `test_find_root_env_wins_over_walk`: env set to `tmp/e`, config at `tmp/ws`, start inside `tmp/ws`. Expect `tmp/e`.
  - `test_find_root_rejects_a_missing_workspace`: env set to `tmp/missing`. `WorkspaceError` mentions `TALOS_WORKSPACE=`. Through `cli.main(["status"])`, the exit code is 2.
  - `test_status_from_a_subdirectory_lists_the_workspace_jobs`: write a fake config at `tmp/ws`, chdir to `tmp/ws`, run the demo, chdir to `tmp/ws/sub/deeper`. `cli.main(["status", "--json"])` lists the demo job.
  - `test_setup_from_a_subdirectory_updates_the_workspace_config`: existing config at `tmp/ws`, chdir to `tmp/ws/sub`. Copy the stubs and scripted answers of `tests/test_cli.py::test_setup_writes_config_and_0600_secret`: `validate_provider` and `deploy_bench` are both stubbed, so no LLM call and no Modal deploy happen. `tmp/ws/talos.config.json` changed, and `tmp/ws/sub/talos.config.json` does not exist.
  - `test_compile_dir_stays_relative_to_cwd`: config at `tmp/ws` with `backend="local"`. `tmp/ws/sub/algorithm/mod.rs` holds a unique marker string; there is no `algorithm/` under `tmp/ws`. Chdir to `tmp/ws/sub`. Stub Docker the way `tests/test_cli.py::_compile_local` does: monkeypatch `cli.make_bench` to record its `run_dir` and return a bench whose `evaluate` records `request.files`. Assert that the recorded files contain the marker, and that the recorded `run_dir == tmp/ws/.talos/compile`. This catches resolving `--dir` against the root, and leaving the run dir on `Path.cwd()`.
  - `test_agent_env_pins_the_workspace_to_the_worktree` (in `tests/test_agentic.py`): with `TALOS_WORKSPACE=/elsewhere` in `os.environ`, `_agent_env(wt)["TALOS_WORKSPACE"] == str(wt)`.
- [ ] **Step 3: Run** them. Expected: FAIL (no `talos.workspace`).
- [ ] **Step 4: Implement** `talos/workspace.py`.
  - Replace each `Path.cwd()` used for **state** (config, `.talos/`, `runs/`) with `find_root()`. `--dir` and `--direction-file` stay relative to the current directory.
  - `cmd_setup` prints `Workspace: <root>` before writing.
  - The job summary `results:` line prints `root / "runs" / job_id`, now absolute.
  - Update `_agent_env` and its caller.
- [ ] **Step 5: Run** `make check PYTHON=.venv/bin/python`. Expected: PASS. Existing tests that assert the `results:` line text may need the absolute path. Update those assertions, and do not loosen them.
- [ ] **Step 6: Mutation check.** Run `tests/test_workspace.py tests/test_agentic.py`, which must fail for each of: (a) the walk returns the farthest config, not the nearest; (b) env ignored; (c) a missing env dir falls back silently; (d) `_agent_env` does not set the variable; (e) `cmd_status` keeps `Path.cwd()`.
- [ ] **Step 7: Commit** `talos/workspace.py talos/cli.py talos/agentic.py talos/status.py tests/conftest.py tests/test_workspace.py tests/test_agentic.py`. Message: `feat: find the Talos workspace by walking up to talos.config.json (#70)`.

### Task C2: deploy with Talos's own Modal client, and `talos --version`

**Files:**
- Modify: `talos/cli.py` (`deploy_bench`; a top-level `--version` in `build_parser`), `pyproject.toml`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing tests:**
  - `test_deploy_uses_the_running_interpreters_modal`: monkeypatch `shutil.which` on the `shutil` module itself to return `"/usr/bin/modal"`, and record calls with a fake `run` returning `returncode=0`. Every recorded argv starts with `[sys.executable, "-m", "modal"]`.
  - `test_version_flag_prints_the_package_version`: `cli.main(["--version"])` raises `SystemExit(0)`, and stdout is `f"talos {talos.__version__}\n"`. Then `importlib.metadata.version("talos-tig") == talos.__version__`.
- [ ] **Step 2: Run** them. Expected: FAIL.
- [ ] **Step 3: Implement.**
  - `deploy_bench` uses `base = [sys.executable, "-m", "modal"]` always. The `shutil.which` branch is removed: `modal` is a hard dependency, so `-m modal` always resolves, and a fallback would be dead code. This deviates from the spec's "then `shutil.which`"; say so in the PR.
  - `--version` uses `action="version"` and `version=f"talos {__version__}"`.
  - `pyproject.toml`: `dynamic = ["version"]`, `[tool.setuptools.dynamic] version = {attr = "talos.__version__"}`, `readme = "README.md"`, `[project.urls]` (`Homepage`, `Issues`, `Documentation` pointing at `docs/usage.md` on GitHub), and `classifiers`: Python 3.10–3.13, GPLv3+, OS Independent, Environment :: Console.
  - Reinstall with `uv pip install -e .` so the metadata picks up the dynamic version.
- [ ] **Step 4: Run** `make check`. Expected: PASS.
- [ ] **Step 5: Mutation check.** Restore the `which` branch: the deploy test fails. Hard-code `version="talos 0"`: the version test fails.
- [ ] **Step 6: Commit** `talos/cli.py pyproject.toml tests/test_cli.py`, message `feat: deploy with Talos's own modal client; talos --version (#70)`.

### Task C3: wheel-install CI and the publish workflow

**Files:**
- Create: `.github/workflows/publish.yml`, `scripts/check_wheel.py`
- Modify: `.github/workflows/ci.yml` (new job `wheel`)

- [ ] **Step 1: Write `scripts/check_wheel.py WHEEL`.** It exits non-zero unless the wheel holds:
  - `talos/data/agent_guide.md`, `talos/data/rust_rules.md`, `talos/data/evidence_template.md`, `talos/data/submission_readme_template.md`;
  - `modal_app/talos_bench.py`;
  - `talos/<m>.py` for every `m` in `JOB_MODULES`. Read it by parsing `talos/c3_jobdir.py` with `ast` (path relative to the script's own location), not by importing `talos`: in CI, `python scripts/check_wheel.py` has `scripts/` on `sys.path`, not the repo root, and the wheel job installs no dependencies.

  Run it locally on `uv build` output. Expected: exit 0. Then delete `agent_guide.md` from a copy of the wheel and run it again. Expected: exit 1, naming the file.
- [ ] **Step 2: Add the `wheel` job to `ci.yml`.**
  - Same 3-OS matrix and bash shell. `actions/setup-python@v5` (3.12) for `scripts/check_wheel.py` and the JSON check, and `astral-sh/setup-uv` pinned to a released version tag.
  - Steps:

```bash
uv build
python scripts/check_wheel.py dist/*.whl
uv tool install --python 3.12 dist/*.whl
echo "$(uv tool dir --bin)" >> "$GITHUB_PATH"
```
  - Then, in a fresh step, `cd "$(mktemp -d)"` and run:

```bash
talos --version
talos guide > guide.txt && head -n 3 guide.txt   # no pipe: Actions' bash runs with pipefail, and an early-closing head could fail the step
talos demo --delay 0
talos status --json | python -c "import json,sys; j=json.load(sys.stdin)['jobs']; assert len(j)==1 and j[0]['status']=='won', j"
```
  - `permissions: contents: read` stays.
- [ ] **Step 3: Write `publish.yml`.**
  - `on: push: tags: ['v*']`. Top-level `permissions: contents: read`.
  - Job `build`: checkout, setup-uv, then `test "v$(python -c 'import talos; print(talos.__version__)')" = "$GITHUB_REF_NAME"`, `uv build`, `python scripts/check_wheel.py dist/*.whl`, and `actions/upload-artifact` of `dist/`.
  - Job `publish`: `needs: build`, `environment: pypi`, `permissions: id-token: write` (this job only), `actions/download-artifact`, then `pypa/gh-action-pypi-publish@release/v1`.
- [ ] **Step 4: Commit** `.github/workflows/ci.yml .github/workflows/publish.yml scripts/check_wheel.py`, message `ci: install the built wheel on 3 OSes; publish to PyPI on v* tags (#70)`. Push over SSH: HTTPS refuses workflow files.

### Task C4: docs for the PyPI install and the workspace

**Files:**
- Modify: `README.md`, `docs/windows.md`, `docs/usage.md`, `docs/maintainers.md`, `talos/data/agent_guide.md`, `AGENTS.md` (only if it says "current directory" for state)

- [ ] **Step 1: Write the docs.**
  - **README install:** `uv tool install --python 3.12 talos-tig`. Give `pipx install talos-tig` and `pip install talos-tig` as alternatives, each with a one-line note that Modal needs the same Python minor version as the deploying interpreter. The clone install moves to `docs/maintainers.md#development`.
  - **Workspace paragraph:** "Run `talos setup` in the folder you want to work from. Talos finds that folder from any folder below it; `TALOS_WORKSPACE=<dir>` overrides."
  - **`docs/usage.md`:** a `## Workspace` section with the three resolution rules.
  - **`docs/windows.md`:** the uv install in PowerShell.
  - **`docs/maintainers.md`:** a `## Releasing` section: bump `talos/__init__.py::__version__`, merge, `git tag vX.Y.Z && git push origin vX.Y.Z`, watch `publish.yml`, then `uvx --from talos-tig talos demo` in an empty directory. It also lists the one-time PyPI trusted-publisher setup (repo `FibonAdithya/talos`, workflow `publish.yml`, environment `pypi`).
  - **Guide:** a line on `TALOS_WORKSPACE` for agents working outside the workspace.
- [ ] **Step 2: Gate:** `make check` (the drift test re-checks the guide).
- [ ] **Step 3: Commit** the modified files, message `docs: install from PyPI with uv; explain the workspace (#70)`.

### Task C5: hand checks and PR

- [ ] **Step 1: Local wheel smoke.** Run `uv build`, then `uv tool install --python 3.12 --force dist/*.whl`. In `mktemp -d`, run `talos demo --delay 0 && talos status --json`. Record the output (MEASURED).
- [ ] **Step 2: Modal deploy from the wheel install.** Stop and ask the owner first, because it redeploys their `talos-bench` app. With a yes:
  - In a scratch workspace, run the wheel-installed `talos setup` with the owner's existing backend `modal`. Record whether `modal deploy` succeeded.
  - Then run `talos doctor`. Its `backend` check confirms every function is deployed.
  - With a no, list it as open in the PR.
- [ ] **Step 3: PR.** It closes #70 only after publishing. Write "Part of #70" until the owner's tag push. The body has the owner's three steps from the spec, the C2 deviation, and the C1 behaviour change (`results:` is now absolute). Follow the `review-pr` loop to merge. After merging, tell the owner to register the trusted publisher and push `v0.1.0`. After their publish, run `uvx --from talos-tig talos demo` in an empty directory, record it as MEASURED, and close #70.
