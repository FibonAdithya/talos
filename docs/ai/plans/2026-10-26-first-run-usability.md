# First-run usability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Challenge status (#60), full CLI help (#62), `talos demo` (#63), `talos doctor` (#64), and wizard menus with a pre-run confirmation (#65).

**Architecture:** Two new modules: `talos/wizard.py` (menu and yes/no helpers built on the existing `ask` injection point) and `talos/doctor.py` (health checks). `talos/cli.py` gains `build_parser()`, `cmd_demo`, `cmd_doctor` and a job summary. `talos/challenges.py` gains per-challenge status. The demo reuses `execute_job` with a scripted fake provider and bench.

**Tech Stack:** Python 3.11, argparse, pytest, ruff. No new dependencies.

**Spec:** `docs/ai/specs/2026-10-26-first-run-usability-design.md`. Read it before any task.

## Global Constraints

- Branch `first-run-usability`, stacked on `run-report-html`. Commit at least once per issue, with the issue number in the message (`(#60)`). End every commit message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Stage explicit paths only (never `git add -A`, `git add .` or `commit -a`).
- Verification command: `make check PYTHON=.venv/bin/python` (ruff, `pytest -m "not live"`, agentify). Single tests: `.venv/bin/python -m pytest -q tests/<file>::<test>`.
- Every text-mode file IO in `talos/` sets `encoding=` and `newline=` (`tests/test_portability.py` fails otherwise).
- Menu, summary and doctor text are plain ASCII (use `-`, not an em dash).
- `ask(prompt, default=None, secret=False) -> str` stays the only way the wizards read input. Tests never read real stdin.
- No test or agent may run a real `c3`, `modal` or `docker` command, a real LLM call, or `talos doctor --full` against a real backend. Doctor tests patch `subprocess.run` to raise.
- Don't touch anything in AGENTS.md's "What requires a human" list: prices, rates, beat rules, nonce counts, hardware classes, sandbox flags, `MONOREPO_REF`, `DEV_IMAGE_TAG`.
- Cost text: a measured figure carries its date (`MEASURED 2026-09-15`); anything computed is labelled `ESTIMATE (unverified)`.
- Each new test gets a `# mutation: ...` comment naming the break it catches (the house style in `tests/test_cli.py`). After the task, apply that break, see the test fail, then restore. Run mutation checks with `PYTHONDONTWRITEBYTECODE=1` and `-p no:cacheprovider`: stale `.pyc` files have hidden mutations before.
- Statuses: tested = knapsack, hypergraph, energy_arbitrage. Known broken = job_scheduling, issue 36. Untested = satisfiability, vehicle_routing, vector_search, neuralnet_optimizer.

## Review Focus

1. **Closed stdin (EOF) at a menu or confirmation:** `default_ask` lets `EOFError` escape. EOF at the run confirmation or the `doctor --full` question must mean "no", never a start or a spend. EOF at a menu exits 2 with a message, not a traceback. Owned by Task 3 (`ask_choice`, `confirm`).
2. **Answers by name that are also digits, or cased differently:** a track named like `n=1`, or `Modal` typed in capitals, must still match. A number past the end (`9` of 3) is invalid, not an IndexError. Owned by Task 3.
3. **A `--yes` run must never block on a prompt:** the summary prints and the job starts with no `ask` call. Owned by Task 4.
4. **Declining after mainnet was read:** nothing is under `runs/`, no bench is built, and `recommend_image_update` is never called. Owned by Task 4.
5. **`doctor` with no config in the directory:** the config-dependent checks print `skip`, and mainnet and GHCR still run. A FAIL line always names a fix command. Owned by Task 6.

---

### Task 1: Challenge status (#60)

**Files:**
- Modify: `talos/challenges.py` (`ChallengeSpec`, `_cpu`, `_gpu`, `CHALLENGES`)
- Modify: `talos/cli.py` (`_challenge_prompt` default, `cmd_run` warning)
- Modify: `README.md` (status table next to the `--challenge` row in "talos run" reference)
- Create: `tests/test_challenges.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Produces: `STATUSES: tuple[str, ...] = ("tested", "untested", "known broken")`. `ChallengeSpec.status: str = "untested"` and `ChallengeSpec.issue: int | None = None`, added after `memory_mib` so positional construction is unchanged. `status_text(spec: ChallengeSpec) -> str` returns `"tested"`, `"untested"` or `"known broken (#36)"`. `status_warning(spec: ChallengeSpec) -> str | None` returns None for tested, `"warning: {name} has never been run end to end with Talos"` for untested, and `"warning: {name} is known broken: see issue #{issue}"` for known broken. `DEFAULT_CHALLENGE = "knapsack"`.

- [ ] **Step 1: Write failing tests in `tests/test_challenges.py`**

```python
from pathlib import Path
import re
from talos.challenges import CHALLENGES, STATUSES, status_text, status_warning

EXPECTED = {"knapsack": "tested", "hypergraph": "tested", "energy_arbitrage": "tested",
            "job_scheduling": "known broken", "satisfiability": "untested",
            "vehicle_routing": "untested", "vector_search": "untested",
            "neuralnet_optimizer": "untested"}

def test_every_challenge_has_the_decided_status():
    # mutation: a challenge added without a status, or a status flipped, fails this
    assert {n: s.status for n, s in CHALLENGES.items()} == EXPECTED
    assert all(s.status in STATUSES for s in CHALLENGES.values())

def test_known_broken_names_its_issue():
    # mutation: dropping issue=36 leaves the user no explanation
    assert CHALLENGES["job_scheduling"].issue == 36
    assert status_text(CHALLENGES["job_scheduling"]) == "known broken (#36)"

def test_status_warning_wording():
    assert status_warning(CHALLENGES["knapsack"]) is None
    assert status_warning(CHALLENGES["vehicle_routing"]) == \
        "warning: vehicle_routing has never been run end to end with Talos"
    assert status_warning(CHALLENGES["job_scheduling"]) == \
        "warning: job_scheduling is known broken: see issue #36"

def test_readme_status_table_matches_the_code():
    # mutation: editing a status in code or README alone fails this
    text = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")
    rows = re.findall(r"^\| `([a-z_]+)` \| (CPU|GPU) \| ([^|]+) \|$", text, re.M)
    table = {name: (hw, status.strip()) for name, hw, status in rows}
    assert set(table) == set(CHALLENGES)
    for name, spec in CHALLENGES.items():
        hw, status = table[name]
        assert hw == ("GPU" if spec.is_gpu else "CPU")
        assert status.startswith(spec.status)
        if spec.issue:
            assert f"#{spec.issue}" in status
```

In `tests/test_cli.py`, add:
- `test_wizard_defaults_to_knapsack`. With `fake_config` and `knapsack_info`, run `cli.main(["run"], ask=...)` with a recording ask whose challenge answer is `""` (default). Assert `job.json`'s `challenge == "knapsack"`. Mutation: the default reverted to `vehicle_routing`.
- `test_run_warns_before_an_untested_challenge`. A `--yes --fake` run on `vehicle_routing` and one on `knapsack`. Assert the stderr of the first contains the exact warning string, and the stderr of the second has no `warning: knapsack`. Mutation: the warning removed, or printed for tested challenges.

- [ ] **Step 2: Run and confirm they fail**

Run: `.venv/bin/python -m pytest -q tests/test_challenges.py tests/test_cli.py -k "status or knapsack or untested"`
Expected: ImportError for `STATUSES` and failures in the cli tests.

- [ ] **Step 3: Implement**

- Add the fields and `STATUSES` to `talos/challenges.py`. `_cpu`/`_gpu` take `status` and `issue` keywords. Set the statuses in `CHALLENGES`. Add `status_text`, `status_warning` and `DEFAULT_CHALLENGE`.
- `cmd_run`: the challenge default becomes `DEFAULT_CHALLENGE`. Right after the challenge is validated (new job only; the resume branch returns earlier), print `status_warning` to stderr when it is not None, under `--yes` too.
- README: add a table with header `| Challenge | Hardware | Status |`. Rows look like `` | `knapsack` | CPU | tested | `` and `` | `job_scheduling` | CPU | known broken ([#36](https://github.com/FibonAdithya/talos/issues/36)) | ``. Add one sentence defining "tested": at least one Talos job got past its baseline on some backend.

- [ ] **Step 4: Run tests, then `make check PYTHON=.venv/bin/python`**

Expected: all pass.

- [ ] **Step 5: Mutation-check each new test, then commit**

```bash
git add talos/challenges.py talos/cli.py README.md tests/test_challenges.py tests/test_cli.py
git commit -m "challenges: per-challenge status, knapsack default, warning on untested (#60)"
```

---

### Task 2: Help text for every command and flag (#62)

**Files:**
- Modify: `talos/cli.py` (`main` split into `build_parser()` + dispatch)
- Modify: `tests/test_cli.py`

**Interfaces:**
- Produces: `build_parser() -> argparse.ArgumentParser`, which builds the whole tree. `main(argv=None, ask=default_ask)` calls it, then dispatches through a dict keyed by `args.cmd`. Later tasks add subparsers inside `build_parser` and entries to that dict. Every subparser gets `help=` and `description=`.

- [ ] **Step 1: Write failing tests**

```python
def _walk(parser):
    import argparse
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            helped = {a.dest for a in action._choices_actions if a.help}
            for name, sub in action.choices.items():
                yield f"subcommand {name}", name in helped
                yield from _walk(sub)
        elif action.help is not argparse.SUPPRESS:
            yield f"{parser.prog} {'/'.join(action.option_strings) or action.dest}", bool(action.help)

def test_every_subcommand_and_flag_has_help():
    # mutation: deleting the help= of any one flag or subcommand fails this
    missing = [where for where, ok in _walk(cli.build_parser()) if not ok]
    assert missing == []

def test_run_help_groups_flags_and_shows_examples(capsys):
    # mutation: dropping the argument groups or the epilog fails this
    with pytest.raises(SystemExit):
        cli.main(["run", "--help"])
    out = capsys.readouterr().out
    for title in ("what to research", "budget", "search", "resume"):
        assert title + ":" in out
    assert "Examples:" in out
    assert "--add-iterations" in out.split("Examples:")[1]

def test_challenge_flag_lists_its_choices(capsys):
    # mutation: dropping choices= hides the valid values
    with pytest.raises(SystemExit):
        cli.main(["compile", "--help"])
    assert "{satisfiability,vehicle_routing,knapsack," in capsys.readouterr().out
```

Note: the dest of a `_choices_actions` entry is the subcommand name, so `helped` collects the names that have `help=`.

- [ ] **Step 2: Run and confirm they fail.** `build_parser` does not exist yet.

- [ ] **Step 3: Implement**

- Move the parser construction into `build_parser()`. Add `description` to the top-level parser: one line on what Talos does.
- Give each subcommand `help=` and `description=`: `setup`, `run`, `compile`, `status`, `report`, `check-updates`.
- `run` uses `formatter_class=argparse.RawDescriptionHelpFormatter` and four groups: "what to research" (challenge, direction, direction-file, track, start, hyperparameters), "budget" (the four `--budget-*` flags), "search" (mode, scoring, search, chains, nonces) and "resume" (resume and the four `--add-*` flags). `--yes` stays in the default options. `--fake` stays `SUPPRESS`.
- The `run` epilog has the heading `Examples:` and three examples:
  - `talos run --challenge knapsack --direction "..." --budget-usd 20 --budget-hours 4`
  - `talos run --resume <job_id> --add-iterations 10`
  - `talos run --challenge knapsack --direction-file notes.md --budget-iterations 30 --yes`
- `--challenge` on `run` and `compile` gets `choices=list(CHALLENGES)`. Keep `cmd_run`'s existing "unknown challenge" path, which handles a wizard answer.
- Help strings are one line each and factual. `--yes` reads: "skip every question and the confirmation; take defaults (compute cap $20 on hosted backends)". Name the default compute cap with `DEFAULT_COMPUTE_USD`, not a literal.
- Existing test `test_compile_*` at `tests/test_cli.py` (`cli.main(["compile", "--challenge", "knapsak"]) == 2`) will now raise `SystemExit` from argparse instead of returning 2. Rewrite it as `with pytest.raises(SystemExit) as e: ...; assert e.value.code == 2`, assert `"invalid choice: 'knapsak'"` in stderr, and keep its `make_bench` refusal stub (the bench must still never be built). No other test passes an unknown challenge on argv (checked by grep during the audit).

- [ ] **Step 4: Run tests, then `make check`.** Expected: all pass.

- [ ] **Step 5: Mutation-check, then commit** (`cli: help text, flag groups and examples for every command (#62)`).

---

### Task 3: Numbered menus that re-prompt (#65, part 1)

**Files:**
- Create: `talos/wizard.py`
- Create: `tests/test_wizard.py`
- Modify: `talos/cli.py` (`cmd_setup`, `cmd_run` prompts)
- Modify: `tests/test_cli.py` (scripted answers and prompt assertions)

**Interfaces:**
- Produces, in `talos/wizard.py`:
  - `CHOICE_TRIES = 3`.
  - `menu_text(title: str, options: Sequence[tuple[str, str]], default: str) -> str`. Returns `"{title}:\n"`, then one line per option, `"  {i}) {value:<w}  - {explanation}"` with ` (default)` appended on the default, then a final `"Choose"`. `w` is the longest value.
  - `ask_choice(ask, title: str, options: Sequence[tuple[str, str]], default: str, tries: int = CHOICE_TRIES) -> str`. Calls `ask(menu_text(...), str(default_index))`, with the index counted from 1. Accepts a blank answer (the default), an in-range number, or a value matched case-insensitively after `strip()`, and returns the value as it appears in `options`. On anything else it prints `not one of the choices: {raw!r}` to stderr and asks again. After `tries` invalid answers, or on `EOFError`, it raises `ConfigError(f"{title}: no valid choice after {tries} tries")` (for EOF: `f"{title}: no answer (input closed)"`).
  - `confirm(ask, prompt: str, default: bool, tries: int = CHOICE_TRIES) -> bool`. Calls `ask(prompt, "y" if default else "n")`. `y`/`yes` returns True and `n`/`no` returns False, in any case. A blank answer returns the default. `EOFError` returns False, whatever the default. Any other answer prints `please answer y or n` to stderr and asks again; after `tries` bad answers it returns False.
- Consumes: `ConfigError` from `talos.config`.

- [ ] **Step 1: Write failing tests in `tests/test_wizard.py`**

Use a scripted ask that returns the queued answers and records each prompt.
- `test_number_name_and_blank_select`: options `[("modal","a"),("c3","b"),("local","c")]` with default `modal`. `"2"` returns `"c3"`, `"LOCAL"` returns `"local"`, `""` returns `"modal"`. The recorded prompt contains `"  1) modal  - a (default)"` and ends with `"Choose"`. Mutation: the default marker dropped, or matching made case-sensitive.
- `test_typo_reprompts_then_accepts`: answers `["agentc", "agentic"]` return `"agentic"`, with 2 prompts recorded and `not one of the choices: 'agentc'` in stderr. Mutation: the first answer accepted unchecked.
- `test_out_of_range_number_is_invalid`: `["9", "0", "-1"]` against 3 options raises `ConfigError` with no IndexError. Mutation: `options[int(raw)-1]` without a bounds check. `-1` would index the last option.
- `test_three_typos_raise`: `["x","y","z"]` raises `ConfigError`, and a fourth answer queued is never asked for. Mutation: an unbounded loop.
- `test_eof_at_menu_raises_config_error`: an ask that raises `EOFError` gives `ConfigError` mentioning "input closed".
- `test_track_value_that_looks_like_a_number`: options `[("all",""),("n=1",""),("2","")]`. `"n=1"` returns `"n=1"`. `"3"` returns `"2"`: a number is an index first. Mutation: values matched before indexes.
- `test_confirm`: `""` with default True returns True. `"N"` returns False. `["maybe","yes"]` returns True after one re-ask. EOF with default True returns False. Three bad answers return False. Mutation: EOF falls back to the default.

- [ ] **Step 2: Run them and confirm they fail.** The module does not exist yet.

- [ ] **Step 3: Implement `talos/wizard.py`.**

- [ ] **Step 4: Switch the prompts to menus in `talos/cli.py`**

| Where | Title | Options (value, explanation) | Default |
|---|---|---|---|
| `cmd_setup` | `Compute backend` | modal: "your Modal account; billed per container second"; c3: "your C3 account; about 12 min fixed overhead per iteration"; local: "Docker on this machine; no compute charge, slow builds" | `modal` |
| `cmd_setup` | `Provider` | each of `KINDS` except `fake`, with a one-line explanation (API key vs local CLI login) | `anthropic` |
| `cmd_setup` | `Mode` (CLI providers) | single-shot: "one LLM call per edit"; agentic: "the CLI agent edits and compiles in a sandbox; roughly 5-20x the tokens" | `single-shot` |
| `cmd_run` | `Challenge` | each of `CHALLENGES`: `f"{'GPU' if cs.is_gpu else 'CPU'}, {status_text(cs)}"` | `DEFAULT_CHALLENGE` |
| `cmd_run` | `Mode` (CLI providers) | as in setup | `cfg.mode` |
| `cmd_run` | `Track to optimise` | `("all", "every active track")` then each of `info.tracks` with an empty explanation | `all` |
| `cmd_run` | `Hyperparameters` | mainnet: "per-track values from the baseline's best mainnet benchmark"; none: "run without hyperparameters" | `mainnet` |

- Catch `ConfigError` from `ask_choice` at each call site: print it to stderr and return 2, the same as the `_ask_number` handling.
- Remove the old free-text checks that the menu now makes unreachable (`unknown backend`, `unknown provider`, `unknown mode`, `unknown hyperparameters choice`) from the wizard paths only. `cmd_run` still validates `--hyperparameters` through argparse `choices`, and `cfg.backend` from the config file.
- Delete `_challenge_prompt`. Leave the `print("agentic mode uses roughly 5-20x ...")` line out, because the mode menu's explanation carries it.

- [ ] **Step 5: Update `tests/test_cli.py`**

- `test_setup_rejects_the_fake_provider`, and the setup test that answers `"aws"`, now need three bad answers. Each asserts exit 2, `not one of the choices` in stderr, and no `talos.config.json` written.
- `test_wizard_labels_gpu_challenges_asks_mode_and_survives_a_typo`: assert the recorded challenge prompt contains `"hypergraph"` and `"GPU, tested"`, and `"job_scheduling"` with `"known broken (#36)"`. The 5-20x assertion now checks the mode menu prompt.
- `test_wizard_hyperparameters_answer_none_and_a_bad_answer`: `"maybe"` now re-prompts. Give `["maybe","maybe","maybe"]` and assert exit 2.
- Any other test that fails only because of a changed prompt string: update the string. Do not weaken an assertion about behaviour.

- [ ] **Step 6: Run `make check`. Expected: all pass.**

- [ ] **Step 7: Mutation-check, then commit** (`wizard: numbered menus that re-prompt on a typo (#65)`).

---

### Task 4: Pre-run summary and confirmation (#65, part 2)

**Files:**
- Modify: `talos/cli.py` (`cmd_run`, new `_job_summary`, new `_new_job_id`)
- Modify: `tests/test_cli.py`

**Interfaces:**
- Consumes: `confirm` (Task 3), `status_text`/`status_warning` (Task 1).
- Produces:
  - `BACKEND_BILLING: dict[str, str]`:
    - modal: `"billed per container second, no fixed per-job overhead"`
    - c3: `"one batch job per iteration, about 12 minutes of fixed overhead each (docs/compute-backends.md#c3-timings)"`
    - local: `"Docker on this machine, no compute charge"`
  - `_new_job_id(root: Path, challenge: str, prefix: str = "") -> str`. The existing stamp-and-suffix logic in `cmd_run`, with `prefix` placed before the stamp. Task 5 uses `prefix="demo-"`.
  - `_job_summary(*, job_id, challenge, scope, nonces, fuel, cfg, budget, start, search, chains, scoring, hp_line, backend) -> str`. A multi-line block that starts with `Job summary` and has one field per line:
    - challenge: name, CPU/GPU and `status_text`, plus the `status_warning` text when there is one
    - scope: the `scope` string, `f"{nonces} nonces per track"` and `f"fuel {fuel}"`
    - provider: provider / model, and the mode
    - search: `f"start {start}, {search}, {chains} chain(s), scoring {scoring}"`
    - the `hp_line` text, verbatim
    - budget: the caps that are not None (test with `is not None`: Budget allows 0, so `if budget.compute_usd:` would hide a $0.00 cap), joined by `, ` from `f"LLM ${usd:.2f}"`, `f"{hours:g} h"`, `f"{iterations} iterations"` and `f"compute ${compute_usd:.2f}"`. An unpriced provider appends ` (LLM spend unpriced)`.
    - backend: `BACKEND_BILLING[backend]`, or `"fake (in-process, no charge)"` for a fake provider
    - results: `runs/{job_id}/`

- [ ] **Step 1: Write failing tests**

- `test_declining_the_summary_writes_nothing_and_builds_no_bench`. Use a **non-fake** config, because `cmd_run` skips `recommend_image_update` for provider `fake`, so a fake config cannot catch the call being moved before the confirmation. Save `Config(provider="anthropic", model="claude-opus-5", mode="single-shot", api_base=None)`, `monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")` and `knapsack_info`. Monkeypatch `cli.recommend_image_update` and `cli.execute_job` to record calls. `execute_job` returns 0 when called, so the test fails on the recorder, not a crash. Run `cli.main(["run","--challenge","knapsack","--direction","go","--budget-iterations","3","--budget-compute-usd","5","--hyperparameters","none","--nonces","2"], ask=scripted(["", "n"]))`; the answers are track and the confirmation. Confirm with a recording ask that the last prompt is `Start this job?`. Assert rc 0, `"Not started." in out`, `not (tmp_path/"runs").exists()`, and both recorders empty. Mutation: the confirmation placed after `JobStore(...)` (runs/ exists), after `recommend_image_update` (recorder non-empty), or the `return 0` dropped (`execute_job` called).
- `test_eof_at_the_confirmation_is_no`: the confirmation answer is an ask that raises `EOFError`. Same assertions as declining. Mutation: `confirm` returns the default on EOF.
- `test_accepting_the_summary_starts_the_job`: answer `"y"`. Assert `execute_job` was called once and `runs/<id>/job.json` exists. Assert the printed `results runs/<id>/` uses the id `execute_job` received, so the id is computed once and not re-stamped.
- `test_yes_prints_the_summary_and_never_asks`: run with `--yes` and an ask that raises `AssertionError` when called. Assert `Job summary` and `knapsack (CPU, tested)` are in the output. Mutation: the confirmation not skipped under `--yes`.
- `test_summary_lists_each_budget_cap_and_billing`: for `--budget-usd 7 --budget-hours 2 --budget-iterations 9 --budget-compute-usd 3 --yes`, the output contains `LLM $7.00`, `2 h`, `9 iterations`, `compute $3.00`, and `BACKEND_BILLING["modal"]` written as a literal string in the test. Use a config with provider `anthropic` and a priced model, and stub `execute_job`. A second run with `--budget-compute-usd 0` shows `compute $0.00`. Mutation: a cap dropped from the summary, or a truthiness check hiding the zero cap.
- `test_summary_shows_the_challenge_warning`: `vehicle_routing` with `--yes`. The output contains `untested`.

- [ ] **Step 2: Run the tests and confirm they fail.**

- [ ] **Step 3: Implement in `cmd_run`**

Compute `job_id = _new_job_id(root, challenge)` after the template docs fetch. Build the summary from the values `cmd_run` already computes: move the `scope`/`hp_line` computation above it. Print it. Then:

```python
if not args.yes and not confirm(ask, "Start this job?", True):
    print("Not started.")
    return 0
```

All of this goes before `recommend_image_update`, `JobStore(...)` and `write_spec`. Remove the old one-line `Job {job_id}: ...` print: the summary replaces it. The summary keeps the substrings that existing tests assert: `"{n} nonces per track"` and the `hp_line` texts.

- [ ] **Step 4: Fix the existing run tests that do not pass `--yes`**

These tests now meet a `Start this job?` prompt (`scripted` raises "unexpected prompt"). Append `""` (default yes) to their answer lists. Do not add `--yes` to them: that would skip the wizard paths they test.

- [ ] **Step 5: Run `make check`. Expected: all pass.**

- [ ] **Step 6: Mutation-check, then commit** (`run: confirm a job summary before anything is written or spent (#65)`).

---

### Task 5: `talos demo` (#63)

**Files:**
- Create: `talos/demo.py` (the scripted provider, scores, compile check and package file descriptions)
- Modify: `talos/cli.py` (`execute_job` gains injection points; new `cmd_demo`; parser entry)
- Create: `tests/test_demo.py`

**Interfaces:**
- Consumes: `_new_job_id(root, challenge, prefix="demo-")` (Task 4) and `build_parser` (Task 2).
- Produces:
  - `execute_job(spec, store, cfg, resume, add=None, *, fakes: tuple | None = None, event_delay: float = 0.0)`. When `fakes` is given, it is `(provider, bench)` and replaces the `FakeProvider(_fake_script)`/`FakeBench(_fake_scores)` pair, with `cfg.provider == "fake"`. `on_event` calls the module-level hook `_event_sleep(event_delay)` (`_event_sleep = time.sleep` in `cli.py`) after each printed line when `event_delay > 0`. Tests patch `cli._event_sleep`, never `time.sleep`, which is global.
  - In `talos/demo.py`:
    - `DemoScript`, a callable class `(system, user) -> str`. It keeps a counter of edit calls, so its output is deterministic.
    - `demo_scores(challenge, files, ns) -> list[int]`
    - `demo_compile_ok(files) -> bool`
    - `PACKAGE_FILES: dict[str, str]`, mapping each package file or dir name to a one-line description, for the names `build_package` writes: `README.md`, `scores.md`, `hypotheses.md`, `evidence_draft.md`, `lineage.json`, `lineage.md`, `diff_vs_baseline.patch`, `diff_vs_template.patch`, `hyperparameters.json`, `report.html`, `submission`
    - `describe_package(pkg: Path) -> list[str]`, which returns `"  {name}  - {description}"` per entry, sorted, with `"(no description)"` for an unknown name
  - `cmd_demo(args, ask) -> int`, plus the `demo` subparser: `--challenge` (choices, default `knapsack`) and `--delay` (float, default `0.4`, help says `0` disables).

**Demo script (the outcomes are fixed; adjust the mechanics if the loop disagrees):** the baseline is `fn solve() { let k = 1; }`. Training quality is `100 + k - 1` per nonce. `DemoScript` answers by prompt kind:
- Hypothesis prompt (`'"strategy_tag" (one of'` in system): a JSON hypothesis with a distinct title per edit index.
- Compile-fix prompt (`"You are fixing a Rust compile error"` in system): remove the ` BROKEN` marker.
- Edit prompt, by edit counter:
  - 1: add ` BROKEN` (the compile fails; the fix restores k=1, so no improvement)
  - 2: k := 0 (no improvement; 2 in a row)
  - 3: k := 3 (training +2%, validates; `demo_scores` returns `98` for nonces with `ns.start >= HOLDOUT_START` while k == 3, so held-out fails: `false_positive`)
  - 4: k := 5 (training +4%, held-out at `100 + k - 1`: `won`)
- Any other prompt kind (distil, dead code): return a harmless answer that the loop accepts. Find what each needs by running the test.
- Write each edit as SEARCH/REPLACE against the `let k = N;` actually present in `user`, the way `_fake_script` does.
- `demo_compile_ok(files)` is `"BROKEN" not in files["mod.rs"]`.

`cmd_demo` builds the spec the way the `--fake` path of `cmd_run` does:
- `ChallengeInfo` with `tracks=["n=1"]` and `max_fuel=1`
- the `FAKE_MAINNET` algorithm and hyperparameters through `_mainnet_hyperparameters`
- `NONCES_PER_TRACK` nonces, scoring `native`, start `mainnet`, search `greedy`, 1 chain
- `Budget(iterations=6)` and direction `"demo: raise k"`
- `job_id = _new_job_id(root, challenge, prefix="demo-")`

It then writes the spec and `tacit.md` like `cmd_run`. It prints one line saying this is a scripted demo with no account, network or spend, then calls `execute_job(..., fakes=(FakeProvider(DemoScript()), FakeBench(demo_scores, compile_ok=demo_compile_ok)), event_delay=args.delay)`. After that it prints:
- `What is in the package:` and `describe_package(...)`
- `See the run as a page: talos report {job_id}`
- `Next: talos setup, then talos doctor`

It returns `execute_job`'s code.

- [ ] **Step 1: Write failing tests in `tests/test_demo.py`**

- `test_demo_shows_every_outcome_in_order`. `monkeypatch.chdir(tmp_path)` with no config. Block the network: monkeypatch `urllib.request.urlopen` and `socket.create_connection` to raise `AssertionError("network")`. Run `cli.main(["demo","--delay","0"])` and assert rc 0. Read `runs/demo-*/timeline.jsonl` and assert, in order:
  - a `compile_failed` event
  - an `iteration_done` with outcome `failed:score` and `runs_since_improvement >= 2`
  - a `false_positive`
  - a `won`

  Also assert `state.json` has status `won` and iteration `< 6`. Mutation: any one scripted step removed (for example the BROKEN edit) fails the order check.
- `test_demo_prints_package_guide_and_next_step`: the output contains `What is in the package:`, `scores.md  - `, `talos report demo-` and `talos setup`, and does **not** contain `(no description)`. Also assert every entry in the demo's `package/` dir is a key of `PACKAGE_FILES`. `PACKAGE_FILES` is keyed on file names, so a file `build_package` starts writing later fails here, instead of showing "(no description)" silently. Mutation: the guide dropped, or a package file missing from `PACKAGE_FILES`.
- `test_demo_delay_sleeps_between_events`: patch `cli._event_sleep` to record. With `--delay 0.25` there is at least one call and every call is `0.25`. With `--delay 0` there are no calls. Mutation: the delay ignored.
- `test_demo_is_listed_in_help` is covered by Task 2's walker. Add the assertion that `"demo"` appears in the output of `talos --help`.
- `test_describe_package_unknown_name` (unit): a temp dir with `scores.md` and `zzz.bin` gives `["  scores.md  - <PACKAGE_FILES['scores.md']>", "  zzz.bin  - (no description)"]`. Write the description literal in the test.

- [ ] **Step 2: Run them and confirm they fail.**

- [ ] **Step 3: Implement `talos/demo.py`, the `execute_job` injection, `cmd_demo` and the parser entry. Iterate on `DemoScript` until the order test passes.**

- [ ] **Step 4: Run `make check`.** The existing `--fake` tests must pass unchanged.

- [ ] **Step 5: Mutation-check, then commit** (`demo: visible talos demo with a scripted mix of outcomes (#63)`).

---

### Task 6: `talos doctor` (#64)

**Files:**
- Modify: `talos/mainnet.py` (add `fetch_challenges`; `fetch_challenge_info` uses it)
- Create: `talos/doctor.py`
- Modify: `talos/cli.py` (`cmd_doctor` and the parser entry)
- Create: `tests/test_doctor.py`
- Modify: `tests/test_mainnet.py`

**Interfaces:**
- `talos/mainnet.py`: `fetch_challenges(get_json=_get_json) -> list[ChallengeInfo]` makes one block lookup and one `get-challenges` call, and returns every challenge. Tracks may be empty; `max_fuel` defaults to `0` when absent. `fetch_challenge_info(name, get_json)` keeps its current behaviour and errors, built on `fetch_challenges`.
- `talos/doctor.py`. It imports `from talos import cli, mainnet` at module level, and calls `cli.check_c3`, `cli.check_local`, `cli.image_available`, `cli.make_bench`, `cli.local_settings`, `cli.remove_compile_containers`, `cli.validate_provider`, `cli.make_provider`, `mainnet.fetch_challenges`, `mainnet.fetch_challenge_info`, `mainnet.top_algorithm` and `mainnet.fetch_algorithm_files` through the module, so tests can monkeypatch them. `cli.py` imports `talos.doctor` only inside `cmd_doctor`.
  - `@dataclass(frozen=True) class Check: name: str; status: str; detail: str = ""`, where status is one of `OK, WARN, FAIL, SKIP = "ok", "warn", "FAIL", "skip"`.
  - `MODAL_APP_NAME = "talos-bench"`. Assert in a test that it equals `ModalBench().app_name`.
  - `modal_function_names() -> list[str]`: for each challenge, `compile_{name}{suffix}` and `score_batch_{name}{suffix}`. The suffix is `""` for CPU and `_{gpu_slug(g)}` for each of `MODAL_GPUS` for GPU. Then `probe_{gpu_slug(g)}` for each GPU.
  - `modal_lookup(app: str, name: str) -> bool`: hydrate, True when found and False on `modal.exception.NotFoundError`. Other exceptions propagate.
  - The checks:
    - `check_config(root: Path) -> tuple[Check, Config | None]`
    - `check_credential(cfg) -> Check` (CLI providers: `shutil.which("claude")`/`shutil.which("codex")`)
    - `check_provider(cfg) -> Check`
    - `check_backend(cfg) -> Check`
    - `check_mainnet() -> Check`
    - `check_images() -> Check`
  - `free_checks(root: Path) -> tuple[list[Check], Config | None]` runs them in spec order. Checks 2-4 are `SKIP` with detail `"no config"` when check 1 fails.
  - `FULL_ASSUMED_S = {"modal_cpu": 600, "modal_gpu": 1800, "c3_cpu": 1200, "c3_gpu": 2400}`. These are assumptions, printed in the text.
  - `full_cost_text(backend: str, challenge: str) -> str`:
    - knapsack on c3: `"MEASURED 2026-09-15: £0.02 and 12 min 20 s for one knapsack job on C3"`
    - local: `"no compute charge; the first run per challenge pulls a 13 GB image and does a warm-up build"`
    - every other pair: `f"not measured for {challenge}; ESTIMATE (unverified): ..."`. Compute it from `talos.bench` rates (`CPU_USD_PER_CORE_SECOND`, `MEM_USD_PER_GIB_SECOND`, `GPU_USD_PER_SECOND`) or `talos.c3_bench.GBP_PER_HOUR` × the assumed seconds. A GPU challenge gives a min-max range across `MODAL_GPUS`/`C3_GPU_CLASSES` and adds `"; GPU nonces can take many minutes each"`. State the assumed duration in the text.
  - `full_check(cfg, challenge: str) -> Check` compiles and scores the top algorithm on 2 training nonces of the first track, with no held-out (`draw_nonce_sets(info.tracks[:1], new_rand_hash(), training_count=2, holdout_count=0)`). It uses `cli.make_bench(cfg.backend, run_dir, PendingJobStore.memory(), c3_api_key=resolve_c3_api_key(cfg) if c3, local=cli.local_settings(cfg) if local)` in a `tempfile.TemporaryDirectory()`. For local, it calls `local_transport.prepare(challenge)` first and `cli.remove_compile_containers(run_dir)` in a `finally`. It calls `select_hardware(challenge)` before `evaluate`. The result is OK when compile is ok and any nonce is ok. The detail gives compile ok/failed, `k/2 nonces ok` and `compute spend (estimated) $x.xx` from `cost_usd_since(cost_mark())`.
  - `run(root: Path, *, full: bool, challenge: str, yes: bool, json_out: bool, ask) -> int` returns 0 when no check is FAIL and no requested `--full` failed to run, and 1 otherwise. A `--full` that is not run records `Check("full", SKIP, reason)` and returns 1:
    - if a free check failed, the reason is `"a free check failed"`
    - in JSON mode without `yes`, the reason is `"needs --yes"`
    - when declined or at EOF, the reason is `"declined"`

    Otherwise it prints `full_cost_text`, then asks `confirm(ask, "Spend this?", False)` unless `yes`.
  - Text output is one line per check: `f"{name}: ok"`, plus `f" ({detail})"` when there is detail. Non-ok checks print `f"{name}: {status}: {detail}"`. JSON output is `json.dumps({"checks":[asdict(c)...], "ok": rc == 0})`, and stdout carries nothing else.
- `cmd_doctor(args, ask)` and the subparser: `--full`, `--challenge` (choices, default `knapsack`), `--yes`, `--json`.

FAIL details and the fix each one names:

| Check | Failure | Detail must contain |
|---|---|---|
| config | `ConfigError` | `talos setup` |
| credential | API key missing | the env var from `ENV_KEYS` (or `TALOS_CUSTOM_API_KEY`) and `talos setup` |
| credential | CLI missing | `install` and the binary name |
| provider | `validate_provider` message | that message and `talos setup` |
| backend modal | names missing, or modal import / token error | the missing count, the first few names, and `talos setup` |
| backend c3 | `ConfigError` from `check_c3` | its message (it already names the fix) |
| backend local | `ConfigError` from `check_local` | its message |
| mainnet | `MainnetError` | `unreachable`, plus `check your network` |
| mainnet | drift | the challenge name and `update talos/challenges.py` |
| images | missing | the challenge names and `talos check-updates` |

Non-FAIL results:
- c3: a balance of 0.0 is WARN with `could not read the balance, or it is £0.00; c3 topup`, because `check_c3` returns 0.0 for both; a balance under 1.0 is WARN with `c3 topup`; anything else is OK with `£x.xx`.
- local: OK with `{ncpu} CPUs, {mem} GiB, NVIDIA runtime: yes/no`.
- modal: OK with `deployed (N functions)`. The detail must not say "up to date".
- provider: OK with `one test call answered`.

- [ ] **Step 1: Write failing tests**

`tests/test_mainnet.py`:
- `test_fetch_challenges_returns_all_with_one_block_lookup`: with a fake `get_json` that counts calls, two challenges give 2 calls and both infos.
- The existing `fetch_challenge_info` tests must pass unchanged.

`tests/test_doctor.py`. Use an autouse fixture that makes every real process or remote path raise `AssertionError("no real subprocess")`:
- `subprocess.run`
- `cli.make_transport` and `cli.CliTransport`. `talos/c3_transport.py` binds `run=subprocess.run` as a default argument at import time, so patching `subprocess.run` alone does not cover it.
- `doctor.modal_lookup`

Tests that need any of these restubbed do so explicitly. A helper `all_ok(monkeypatch, backend)` stubs every dependency to succeed. Tests:
- `test_all_checks_pass_exit_0` (modal; `cli.main(["doctor"])`): rc 0, with lines starting `config: ok`, `credential: ok`, `provider: ok`, `backend: ok`, `mainnet: ok` and `images: ok`, in that order. Mutation: a check dropped from `free_checks`.
- `test_each_failure_names_its_fix`, parametrized over the FAIL table rows: rc 1, and the FAIL line contains the listed text. Mutation: a fix hint dropped.
- `test_no_config_skips_dependent_checks`: no `talos.config.json`. Assert `credential: skip`, `provider: skip` and `backend: skip`, and that mainnet and images ran (a recorder on each stub). Mutation: an early return after the config failure.
- `test_modal_function_names_match_register`: build a fake app whose `.function(name=..., **kw)` records names and returns `lambda f: f`. Call `modal_app.talos_bench.register(app, image=lambda n: None, probe_image=lambda: None)` and assert the recorded set equals `set(modal_function_names())`. Mutation: the probe names or a GPU suffix dropped. If importing `modal_app.talos_bench` needs `modal` and it is missing in the venv, skip with `pytest.importorskip("modal")`, and say so in the report.
- `test_modal_missing_function_fails`: `doctor.modal_lookup` returns False for one name. Assert FAIL, the line contains that name, and does not contain "up to date".
- `test_c3_balance_warns_and_still_exits_0`: `check_c3` returns 0.5. The line is `backend: warn` and contains `c3 topup`; rc 0. Mutation: a warn counted as FAIL.
- `test_full_declined_never_builds_a_bench`, parametrized over the answers `""`, `"n"` and EOF: `make_bench` records calls, the rc is 1, there are no calls, and the output contains `full: skip: declined`. Mutation: the default flipped to yes, or EOF treated as the default.
- `test_full_with_yes_scores_two_training_nonces_no_holdout`: `make_bench` returns a stub bench that records the `EvalRequest`. rc 0. Exactly one `evaluate` call: `sum(ns.count for ns in req.training) == 2`, `all(ns.count == 0 for ns in req.holdout)`, and `select_hardware` was called before `evaluate`. Mutation: holdout counts of 2, or `select_hardware` skipped.
- `test_full_skipped_after_a_free_check_fails`: images FAIL plus `--full --yes`. `make_bench` is never called, `full: skip: a free check failed`, rc 1.
- `test_json_full_without_yes_is_skip_and_exit_1`: stdout parses as JSON, `ok` is False, the `full` entry has status `skip` and detail `needs --yes`, and `ask` is never called.
- `test_json_has_every_check`: the names equal `["config","credential","provider","backend","mainnet","images"]`.
- `test_full_cost_text`:
  - `("c3","knapsack")` contains `MEASURED 2026-09-15`.
  - `("c3","vehicle_routing")` contains `not measured for vehicle_routing` and `ESTIMATE (unverified)`, and does not contain `MEASURED`.
  - `("modal","knapsack")` contains `$0.04`. Derivation, by hand: 600 s × (4 × 0.0000131 + 8 × 0.00000222) = $0.0421.
  - `("modal","hypergraph")` contains `$0.98` and `$1.97`. Derivation: 1800 × 0.000542 = 0.9756 and 1800 × 0.001097 = 1.9746.
  - `("c3","hypergraph")` contains `£0.99` and `£2.45`. Derivation: 1.49 × 2400/3600 = 0.993 and 3.67 × 2400/3600 = 2.447.
  - `("local","hypergraph")` contains `no compute charge`.

  Mutation: the knapsack figure reused for vehicle_routing, or the GPU range taken from one GPU.

- [ ] **Step 2: Run them and confirm they fail.**

- [ ] **Step 3: Implement `fetch_challenges`, `talos/doctor.py`, `cmd_doctor` and the parser entry with help.**

- [ ] **Step 4: Run `make check`. Expected: all pass.**

- [ ] **Step 5: Mutation-check, then commit** (`doctor: free health checks and a confirmed --full smoke run (#64)`).

---

### Task 7: README and docs

**Files:**
- Modify: `README.md`
- Modify: `docs/*.md`, but only where they name the commands or the live smoke test as the user's first check (grep shows where)

- [ ] **Step 1: Edit the README**

- "Try it with no accounts" (Linux/macOS and Windows): replace the `--fake` command with `talos demo` (Windows: `.venv\Scripts\talos.exe demo`). Describe the outcomes it shows. Keep `--fake` out of the user path, and leave one sentence noting that tests use it.
- "5. Before your first real run" and the Windows "5. Run the live smoke test once" become: run `talos doctor`, then `talos doctor --full` once. Say that `--full` asks before spending, and give the measured C3 figure only for knapsack.
- "Live smoke test" becomes "Live smoke tests (maintainers)", with its content unchanged.
- Command reference: add `talos demo` and `talos doctor` sections in the style of the existing ones, with flags, output and exit codes. Mention the summary and confirmation under `talos run`, and that `--yes` skips the confirmation (update the `--yes` row).
- The Contents list gets the new sections.
- In `check-updates`' printed upgrade guidance (`cmd_check_updates`), "run your backend's live smoke test" becomes "run `talos doctor --full`".

- [ ] **Step 2: Grep for stale text, then fix each hit or leave it deliberately**

```bash
grep -n -i "\-\-fake\|live smoke\|pytest -m live\|vehicle_routing\]\|first real run" README.md AGENTS.md docs/*.md
grep -n -i "modal or c3\|two compute\|modal and c3\|Modal, C3" README.md AGENTS.md docs/*.md
```

- [ ] **Step 3: Run `make check`.** `test_docs_references` checks the anchors.

- [ ] **Step 4: Open the help and run the demo by hand, and read the output**

```bash
.venv/bin/talos --help
.venv/bin/talos run --help
cd "$(mktemp -d)" && /home/fibonadithya/TIG/Talos/.venv/bin/talos demo --delay 0
```

Expected: the subcommands listed with help, the grouped flags and examples, and a demo ending with `Status: won`, the package guide and the next step. Do not run `talos doctor` against a real backend. A no-`--full` run in a scratch dir with no config is allowed: it touches only mainnet and GHCR (read-only).

- [ ] **Step 5: Commit** (`docs: talos demo and talos doctor in the README; live tests are for maintainers (#63, #64)`).
