# `--start template` for every challenge: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `talos run --start template` is accepted for all eight challenges, and GPU template
starts compile because the root carries a `.cu` file.

**Architecture:** Per-challenge file lists in `talos/challenges.py`. A new module
`talos/template_start.py` builds the template root (`mod.rs`, plus `algorithm.cu` on GPU)
from template text or from `runs/<job>/challenge/`, and loop, package and report all use it.
The CLI writes `challenge/template.cu` at job creation and hands the `.cu` text to the loop on
start and on resume.

**Tech Stack:** Python 3.10+, pytest. Run tests with `.venv/bin/python -m pytest` from the
worktree `/home/fibonadithya/TIG/Talos-template-all` (branch `template-start-all-challenges`).
The worktree's own `.venv` is created in Task 0. Never use the main checkout's
`/home/fibonadithya/TIG/Talos/.venv/bin/talos`: its editable install points at the main
checkout, so it would run the wrong code.

**Spec:** `docs/ai/specs/2026-10-28-template-start-all-challenges-design.md`

## Global Constraints

- Monorepo pin is `talos/challenges.py::MONOREPO_REF` (`839c7184afda9b64c4a9dca18bca5a647a1bb867`). Do not change it.
- No path may fetch or show a file under `baselines/`, or the mainnet baseline's source, in a template job.
- Challenge files are shown unmodified (no stripping, including hypergraph's `greedy_bipartition`).
- The GPU root file name is `algorithm.cu`. The CPU root stays `{"mod.rs": ...}`.
- `talos/edits.py` and `talos/agentic.py::read_back` are not changed: the file set never grows.
- codex agentic stays refused for template starts.
- Tests: never run a command that bills Modal/C3/an LLM. `make check` runs `-m "not live"`; new network-only tests carry `pytest.mark.live`.
- Every new test has a `# mutation:` comment naming the change it catches (repo convention).
- Stage explicit paths only; never `git add -A` / `.`.

## Review Focus

1. **Resume of a GPU template job whose `challenge/template.cu` is missing** (a job created by an older Talos, or a deleted file). Expected: a clear error naming the file, not a `KeyError` or a root without the `.cu`. Owned by Task 3 (test `test_gpu_template_resume_without_template_cu_fails_clearly`).
2. **Nested doc names on Windows** (`nn/linear.rs`). Expected: written and read back with `/` keys; `Path` joins handle it. Covered by Task 2's neuralnet test; no Windows-specific code.
3. **`--start template` on a challenge removed from the lists in future.** Expected: the CLI refusal still fires before any fetch. Task 1 keeps the guard and its test (monkeypatched list).
4. **Prompt size on neuralnet_optimizer** (~80 KB of docs, ESTIMATE (unverified) ~25k tokens per prompt). Expected: the job runs; no truncation logic is added. Noted in docs (Task 5), no test.
5. **A GPU candidate that empties `algorithm.cu`.** Expected: compile fails through the normal compile-fix path (TIG's build needs a `.cu`; an empty file still matches the glob). No new code; not tested here.

---

### Task 0: Worktree venv

- [ ] **Step 1:** `cd /home/fibonadithya/TIG/Talos-template-all && uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -e . -r requirements-dev.txt` (`.venv/` is already in `.gitignore`).
- [ ] **Step 2:** `.venv/bin/python -c "import talos; print(talos.__file__)"` prints a path under `/home/fibonadithya/TIG/Talos-template-all/`, and `make check PYTHON=.venv/bin/python` passes on the untouched branch. Record the passed-test count as the baseline.

### Task 1: File lists, refusal guard and leak checks

**Files:**
- Modify: `talos/challenges.py:11-17`
- Modify: `tests/test_mainnet.py:255-272`
- Modify: `tests/test_cli.py` (`test_start_template_refuses_an_unsupported_challenge`, ~3011)
- Create: `tests/test_template_start_live.py`

**Interfaces:**
- Produces: `TEMPLATE_START_FILES` with an entry for every key of `CHALLENGES`, values exactly as spec §4 (relative paths, `nn/...` for neuralnet_optimizer).

- [ ] **Step 1: Write the failing tests** in `tests/test_mainnet.py`

```python
def test_every_challenge_has_a_template_file_list():
    # mutation: leaving a challenge out of TEMPLATE_START_FILES, or a list without mod.rs
    assert set(TEMPLATE_START_FILES) == set(CHALLENGES)
    for files in TEMPLATE_START_FILES.values():
        assert {"README.md", "mod.rs"} <= set(files)

@pytest.mark.parametrize("challenge", sorted(TEMPLATE_START_FILES))
def test_challenge_docs_never_fetch_baselines(challenge):
    urls = []
    docs = mainnet.fetch_challenge_docs(challenge, get_text=lambda u: urls.append(u) or "x")
    # mutation: a baselines/ entry in a list, or walking the directory
    assert not any("/baselines/" in u for u in urls)
    assert set(docs) == set(TEMPLATE_START_FILES[challenge])
```

Replace `test_challenge_docs_refuse_a_challenge_without_a_file_list` so it uses
`"no_such_challenge"` instead of `"knapsack"` (same assertion, `pytest.raises(MainnetError)`).
Import `CHALLENGES` from `talos.challenges`.

In `tests/test_cli.py`, change `test_start_template_refuses_an_unsupported_challenge` to
`monkeypatch.delitem(TEMPLATE_START_FILES, "knapsack")` first (import from `talos.challenges`;
it is the same dict object `cli` imported), keep the rest, and assert the error names
`energy_arbitrage`. Keep its existing mutation comment.

- [ ] **Step 2: Run, expect FAIL** — `.venv/bin/python -m pytest tests/test_mainnet.py -q -k "template_file_list or never_fetch_baselines or without_a_file_list"`: the first test fails on the set comparison.

- [ ] **Step 3: Add the seven lists** to `TEMPLATE_START_FILES`, exactly spec §4's table. Update the comment above it to say every challenge has a list and that hypergraph's files include TIG's greedy partitioner on purpose (it sets the quality zero point).

- [ ] **Step 4: Write the live test** `tests/test_template_start_live.py` with `pytestmark = pytest.mark.live`:

```python
@pytest.mark.parametrize("challenge", sorted(TEMPLATE_START_FILES))
def test_file_list_matches_the_pinned_tree(challenge):
    base = f"tig-challenges/src/{challenge}"
    paths = mainnet._walk(base, MONOREPO_REF, mainnet._get_json)
    rel = {p[len(base) + 1:] for p in paths}
    # mutation: a missed file (scenarios.rs, nn/linear.rs) or a list stale after a pin bump
    assert set(TEMPLATE_START_FILES[challenge]) == {r for r in rel if not r.startswith("baselines/")}

def test_hypergraph_and_neuralnet_docs_are_the_real_files():
    hg = mainnet.fetch_challenge_docs("hypergraph")
    # mutation: dropping or filtering the challenge .cu
    assert "greedy_bipartition" in hg["kernels.cu"]
    nn = mainnet.fetch_challenge_docs("neuralnet_optimizer")
    assert "pub" in nn["nn/linear.rs"]
```

Docstring: "Manual: reads GitHub only, bills nothing. Run: `.venv/bin/python -m pytest -m live tests/test_template_start_live.py`."

- [ ] **Step 5: Run** the Step 2 command (PASS), `.venv/bin/python -m pytest tests/test_cli.py -q -k refuses_an_unsupported` (PASS), and the live file once with `-m live` (PASS; needs network only).

- [ ] **Step 6: Commit** `talos/challenges.py tests/test_mainnet.py tests/test_cli.py tests/test_template_start_live.py` — `feat: template-start file lists for every challenge`.

---

### Task 2: Nested doc names and CPU template jobs

**Files:**
- Modify: `talos/cli.py` (`FAKE_MAINNET.fetch_challenge_docs` ~504; doc write loop ~1273-1276)
- Test: `tests/test_cli.py` (next to `_template_job`, ~3005)

**Interfaces:**
- Consumes: `TEMPLATE_START_FILES` (Task 1).
- Produces: `FAKE_MAINNET.fetch_challenge_docs(ch)` returns `{name: "Fake challenge: maximise k.\n" if name == "README.md" else f"fake {ch}/{name}\n" for name in TEMPLATE_START_FILES[ch]}`. `_template_job(tmp_path, extra=(), challenge="energy_arbitrage")`.

- [ ] **Step 1: Write the failing tests**

```python
def test_template_start_runs_on_a_cpu_challenge(tmp_path, monkeypatch):
    # mutation: the CLI refusal left in place for knapsack
    monkeypatch.chdir(tmp_path)
    rc, run_dir = _template_job(tmp_path, challenge="knapsack")
    # 1 = exhausted: the fake first candidate is at baseline parity (k=1), as the
    # energy_arbitrage fake job measured on 2026-10-28 (exit=1)
    assert rc == 1
    assert (run_dir / "challenge" / "mod.rs").read_text() == "fake knapsack/mod.rs\n"
    assert (run_dir / "package" / "diff_vs_template.patch").is_file()

def test_template_start_writes_and_quotes_nested_docs(tmp_path, monkeypatch):
    # mutation: writing docs without mkdir(parents=True) raises for nn/linear.rs
    monkeypatch.chdir(tmp_path)
    prompts = []
    script = cli._fake_script
    monkeypatch.setattr(cli, "_fake_script",
                        lambda s, u: prompts.append(s + u) or script(s, u))
    rc, run_dir = _template_job(tmp_path, challenge="neuralnet_optimizer")
    assert (run_dir / "challenge" / "nn" / "linear.rs").read_text() == \
        "fake neuralnet_optimizer/nn/linear.rs\n"
    assert any("tig-challenges/src/neuralnet_optimizer/nn/linear.rs" in p for p in prompts)
```

The fake bench ignores `.cu` files (a `--fake` hypergraph mainnet job ran to exit 0 on
2026-10-28), so this GPU test passes in this task without Task 3. `rc` is not asserted.

- [ ] **Step 2: Run, expect FAIL** — `.venv/bin/python -m pytest tests/test_cli.py -q -k "cpu_challenge or nested_docs"`.

- [ ] **Step 3: Implement** — the `FAKE_MAINNET.fetch_challenge_docs` shape from Interfaces; `(docs_dir / name).parent.mkdir(parents=True, exist_ok=True)` before each write; add the `challenge` parameter to `_template_job`.

- [ ] **Step 4: Run** Step 2's command, then `.venv/bin/python -m pytest tests/test_cli.py -q -k template` (all PASS).

- [ ] **Step 5: Commit** `talos/cli.py tests/test_cli.py` — `fix: template start writes nested challenge docs`.

---

### Task 3: GPU template root (`algorithm.cu`)

**Files:**
- Create: `talos/template_start.py`
- Modify: `talos/mainnet.py` (beside `fetch_template`, ~209)
- Modify: `talos/cli.py` (`FAKE_MAINNET`; job creation ~1227-1276; `_run_job` ~834-853)
- Modify: `talos/loop.py` (`Loop.__init__` ~95-103, `_root_files` ~422-427, `_context` ~561)
- Modify: `talos/prompts.py` (`PromptContext`, contract line ~207)
- Modify: `talos/agentic.py::prepare_worktree` (~142-146)
- Test: `tests/test_template_start.py` (new), `tests/test_cli.py`, `tests/test_agentic.py`

**Interfaces:**
- Produces in `talos/template_start.py`:
  - `ROOT_CU = "algorithm.cu"`
  - `def template_root(challenge: str, template_rs: str, template_cu: str | None) -> dict[str, str]` — `{"mod.rs": template_rs}` for CPU; `{"mod.rs": ..., ROOT_CU: template_cu}` for GPU; `ValueError` naming `template.cu` when GPU and `template_cu is None`.
  - `def root_names(challenge: str) -> list[str]` — sorted names `template_root` would return.
  - `def read_template_root(challenge_dir: Path, challenge: str) -> dict[str, str] | None` — reads `template.rs` (and `template.cu` on GPU) with `read_bytes().decode("utf-8", errors="replace")`; `None` if any needed file is missing.
- Produces in `talos/mainnet.py`: `def fetch_template_cu(name: str, get_text=_get_text) -> str` — `{GH_RAW}/{MONOREPO_REF}/tig-algorithms/src/{name}/template.cu`.
- Produces: `FAKE_MAINNET.fetch_template_cu = lambda ch: "// fake template.cu\n"`.
- Produces: `Loop(..., template_cu: str | None = None)` stored as `self.template_cu`; `PromptContext.template_cu: str | None = None`.

- [ ] **Step 1: Write the failing unit tests** in `tests/test_template_start.py`

```python
def test_cpu_root_is_mod_rs_only():
    # mutation: adding a .cu for CPU challenges
    assert template_root("knapsack", "RS", None) == {"mod.rs": "RS"}

def test_gpu_root_carries_the_cu():
    # mutation: _root_files returning only mod.rs makes every GPU candidate fail to build
    assert template_root("hypergraph", "RS", "CU") == {"mod.rs": "RS", "algorithm.cu": "CU"}
    assert root_names("hypergraph") == ["algorithm.cu", "mod.rs"]

def test_gpu_root_without_cu_raises():
    # mutation: silently returning a root without the .cu
    with pytest.raises(ValueError, match="template.cu"):
        template_root("vector_search", "RS", None)

def test_read_template_root_needs_the_cu_on_gpu(tmp_path):
    (tmp_path / "template.rs").write_text("RS")
    # mutation: reading template.rs alone for a GPU challenge
    assert read_template_root(tmp_path, "hypergraph") is None
    (tmp_path / "template.cu").write_text("CU")
    assert read_template_root(tmp_path, "hypergraph") == {"mod.rs": "RS", "algorithm.cu": "CU"}
    assert read_template_root(tmp_path, "knapsack") == {"mod.rs": "RS"}
```

and `test_fetch_template_cu_is_pinned` in `tests/test_mainnet.py` (URL equals
`f"{mainnet.GH_RAW}/{MONOREPO_REF}/tig-algorithms/src/hypergraph/template.cu"`; mutation: a
branch name instead of the pin).

- [ ] **Step 2: Write the failing CLI tests** in `tests/test_cli.py`

```python
def test_gpu_template_job_roots_carry_algorithm_cu(tmp_path, monkeypatch):
    # mutation: cli not writing challenge/template.cu, or the loop root without it
    monkeypatch.chdir(tmp_path)
    rc, run_dir = _template_job(tmp_path, challenge="hypergraph")
    assert (run_dir / "challenge" / "template.cu").read_text() == "// fake template.cu\n"
    first = run_dir / "iterations" / "0001"
    assert (first / "algorithm.cu").read_text() == "// fake template.cu\n"  # whole-file edit left it unchanged

def test_gpu_template_resume_keeps_algorithm_cu(tmp_path, monkeypatch):
    # mutation: resume not passing template_cu to the Loop (_node_files calls _root_files
    # first, template_root raises, the job fails and iteration 2 writes no algorithm.cu).
    # Iteration 2's parent is node 1 = state.best, whose files come from state.best.files.
    ...
```

Resume setup (both resume tests): `monkeypatch.chdir(tmp_path)`, `fake_config(tmp_path)`
(a resume reads the config), run `_template_job(tmp_path, challenge="hypergraph")` (1
iteration, rc 1), then edit `job.json` `budget.iterations = 2` and `state.json`
`status = "cancelled"` exactly as `test_..._resumes_from_its_stored_challenge` does (~3214),
then `cli.main(["run", "--resume", run_dir.name])`. This test asserts
`(run_dir / "iterations" / "0002" / "algorithm.cu").read_text() == "// fake template.cu\n"`.

```python
def test_gpu_template_resume_without_template_cu_fails_clearly(tmp_path, monkeypatch, capsys):
    # mutation: a KeyError/ValueError traceback, or a resume that runs with a root missing
    # the .cu, instead of a message naming the file
    ...
```

Same setup, then delete `run_dir / "challenge" / "template.cu"` before resuming. Assert
`rc == 1`, `"template.cu" in capsys.readouterr().err`, and
`JobStore(run_dir).load().iteration == 1` (nothing ran).

- [ ] **Step 3: Write the failing agentic and prompt tests**
  - `tests/test_agentic.py::test_gpu_template_challenge_md_quotes_template_cu` — `prepare_worktree(ctx(...), parent=tmp_path)` (the file's existing `ctx()` helper, ~18; it fixes `files=` itself, so do NOT pass `files=`, that raises TypeError) with `start="template"`, `is_gpu=True`, `template_cu="CU_MARKER"` writes `CHALLENGE.md` containing `CU_MARKER`; a CPU context (`template_cu=None`) does not contain `template.cu`. Mutation: `prepare_worktree` quoting only `template.rs`.
  - `tests/test_prompts.py::test_gpu_template_hypothesis_prompt_shows_template_cu` — with the file's `ctx(**kw)` helper (`challenge="hypergraph"`, `start="template"`, `is_gpu=True`, `template_cu="CU_MARKER"`), `hypothesis_prompts(c)[0]` contains `algorithm.cu` and `CU_MARKER`; with `template_cu=None` it contains neither. Mutation: the LLM is not told which file holds its kernels.

- [ ] **Step 4: Run, expect FAIL** — `.venv/bin/python -m pytest tests/test_template_start.py tests/test_mainnet.py tests/test_cli.py tests/test_agentic.py tests/test_prompts.py -q -k "template or cu"`.

- [ ] **Step 5: Implement**
  - `talos/template_start.py` per Interfaces (uses `CHALLENGES[challenge].is_gpu`).
  - `mainnet.fetch_template_cu`; `FAKE_MAINNET.fetch_template_cu`.
  - Job creation: for a GPU challenge fetch `template.cu` with the other template fetches (same `MainnetError` handling), and write it as `challenge/template.cu`.
  - `_run_job`: for a template start on a GPU challenge, read `challenge/template.cu` and pass it as `Loop(template_cu=...)`; if missing, print `cannot resume: runs/<job>/challenge/template.cu is missing` (or similar, naming the file) to stderr and return 1 before any spend.
  - `Loop._root_files`: `template_root(self.spec.challenge, self.template_rs, self.template_cu)` for template starts. `_context` passes `template_cu=self.template_cu`.
  - `prompts.py`: after the contract block, for `ctx.start == "template" and ctx.template_cu`, one line: `algorithm.cu holds your CUDA kernels; it starts as TIG's template.cu:` followed by the text in a ```cuda block.
  - `agentic.prepare_worktree`: append the same `template.cu` section to `CHALLENGE.md` when `ctx.template_cu`.

- [ ] **Step 6: Run** Step 4's command (PASS), then `make check PYTHON=.venv/bin/python` (PASS).

- [ ] **Step 7: Commit** the files listed above — `feat: GPU template starts seed algorithm.cu from template.cu`.

---

### Task 4: Package and report use the template root

**Files:**
- Modify: `talos/package.py` (~352-359)
- Modify: `talos/report.py::run_sources` (~115-124), `root_location` (~110)
- Test: `tests/test_cli.py` (package diff, needs the fake CLI job), `tests/test_report.py`

**Interfaces:**
- Consumes: `read_template_root`, `root_names` (Task 3).

- [ ] **Step 1: Write the failing tests**
  - `test_gpu_template_package_diff_names_algorithm_cu` (in `tests/test_cli.py`, beside `_template_job`) — run `_template_job(tmp_path, challenge="hypergraph")` with `_fake_script` patched so that, when `WHOLE_FILE_FORMAT in system`, it returns the original reply plus `<<<<<<< FILE algorithm.cu\n// changed\n>>>>>>> END FILE\n`. Assert `package/diff_vs_template.patch` contains `-// fake template.cu` and `+// changed`. The `-` line is what discriminates: diffing against `{"mod.rs": template.rs}` only shows `algorithm.cu` as a new file with no removed line. Mutation: package diffing against the single-file root.
  - `test_run_sources_template_root_has_algorithm_cu` — `make_run(tmp_path, hyps, start="template", template="RS", iteration_files={1: {"mod.rs": "A", "algorithm.cu": "B"}}, nodes=...)` (`make_run` hardcodes `challenge="knapsack"`, so pass `dataclasses.replace(spec, challenge="hypergraph")` to `run_sources`) and write `challenge/template.cu` = `"CU"` by hand. Assert `sources(0) == {"mod.rs": "RS", "algorithm.cu": "CU"}` and that a non-best node 1 reads `algorithm.cu == "B"` from its iteration dir. Mutation: `run_sources` keeping the single-file root or `names = ["mod.rs"]`.
  - Keep the existing CPU template package/report tests passing unchanged.

- [ ] **Step 2: Run, expect FAIL** — `.venv/bin/python -m pytest tests/test_cli.py tests/test_report.py -q -k "algorithm_cu"`.

- [ ] **Step 3: Implement** — package: `root = read_template_root(store.run_dir / "challenge", spec.challenge)`; `template_missing = root is None`; diff against `root`. report: `root = read_template_root(...)`, `names = root_names(spec.challenge)`.

- [ ] **Step 4: Run** Step 2's command and `make check PYTHON=.venv/bin/python` (PASS).

- [ ] **Step 5: Commit** `talos/package.py talos/report.py tests/test_cli.py tests/test_report.py` — `feat: template package and report include algorithm.cu`.

---

### Task 5: Docs and verification

**Files:**
- Modify: `docs/usage.md` (`--start` row ~362 and the **`--start`** section ~413-421), `docs/architecture.md:100`, `talos/cli.py` `--start` help text (~1621) if it names energy_arbitrage
- Check: `git grep -n -i "energy_arbitrage. only\|energy_arbitrage only" -- docs/usage.md docs/architecture.md talos` returns nothing afterwards. The usage.md sentence "Refused for challenges other than `energy_arbitrage`" (~417) wraps a line, so the grep misses it: delete it by hand. Leave the 2026-10-19 spec and plan under `docs/ai/` unchanged (historical record).

- [ ] **Step 1: Edit docs.** Say `--start template` works for every challenge; the LLM sees the challenge's `README.md` and source files except `baselines/` (hypergraph's include TIG's greedy partitioner, which sets the quality zero point); on GPU challenges the root also has `algorithm.cu` from `template.cu`; every new best of a template job gets a metered validation, which costs more on GPU challenges; neuralnet_optimizer's docs make long prompts (ESTIMATE (unverified) ~25k tokens). Keep the codex refusal sentence. In `docs/usage.md` "Where results land", note `challenge/template.cu` for GPU template jobs.

- [ ] **Step 2: Run** the grep from **Check** (no output), `grep -n "other than" docs/usage.md` (no template-start hit), and `make check PYTHON=.venv/bin/python` (PASS).

- [ ] **Step 3: Fake runs for all eight challenges** from a scratch workspace:
  `for c in knapsack energy_arbitrage job_scheduling hypergraph satisfiability vehicle_routing vector_search neuralnet_optimizer; do .venv/bin/talos --workspace <scratch> run --fake --start template --challenge $c --direction x --budget-iterations 1 --yes; echo "$c exit=$?"; done`
  Expected: every exit is 0 or 1 (won / not won) and every job has `package/diff_vs_template.patch`; none prints a traceback. Paste the eight exit lines into the PR body.

- [ ] **Step 4: Commit** the doc files — `docs: --start template works on every challenge`.

- [ ] **Step 4b: Mutation-check every new test (orchestrator).** For each `# mutation:` comment added in Tasks 1-4, apply that mutation to the code under test, run only that test with `PYTHONPYCACHEPREFIX=$(mktemp -d)` (stale `.pyc` gives false MISSED), confirm it FAILS, restore with `git checkout -- <file>`, and confirm `git diff --stat` is empty. Record a CAUGHT/MISSED table; any MISSED test is rewritten before the PR.

- [ ] **Step 5: Local knapsack smoke (orchestrator only, ask the user first).** Needs Docker and a real LLM: the user's configured provider is `claude-cli` (subscription-billed), and the knapsack dev image 0.0.8 is not pulled yet (13 GB). With a scratch workspace whose config sets `backend: local`: `talos --workspace <scratch> run --start template --challenge knapsack --direction "write a greedy by value density" --budget-iterations 2 --yes`. Expected: at least one iteration gets past `compile` and has `scored` in `timeline.jsonl`. Record the result as MEASURED with the command. Not a subagent task.

---

## Self-review notes

- Task 0 sets up the worktree venv; all later commands assume it.
- Spec §2 goals 1-4 → Tasks 1-4; §6 items 1-8 → Tasks 1 (1), 3 (2, 4, 5, 7), 2 (3), 4 (6), 5 (8); §7 tests → Tasks 1-4; §8 → Task 5.
- Names used across tasks: `ROOT_CU`, `template_root`, `root_names`, `read_template_root`, `fetch_template_cu`, `template_cu` (Loop and PromptContext), `_template_job(..., challenge=)`.
