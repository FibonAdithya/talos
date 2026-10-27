# PyPI install, default workspace, extras, stale-deploy check: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `uv tool install talos-tig` works and `talos` finds its state from any directory (#70).

**Architecture:** New `talos/workspace.py::find_root` replaces every state `Path.cwd()`. The new
`talos/extras.py::require` turns a missing optional dependency into a message. The new
`talos/deploy_digest.py` names a marker Modal function that the client looks up before spending.
Packaging metadata, a wheel-install CI job and a tag-triggered publish workflow finish it.

**Tech Stack:** Python 3.10+, argparse, pytest, setuptools, uv, GitHub Actions, Modal.

**Spec:** `docs/ai/specs/2026-10-27-pypi-workspace-design.md` (amends PR 3 of
`docs/ai/specs/2026-10-27-readme-split-agent-guide-pypi-design.md`; read both).

Branch: `pypi-workspace-70` (from `origin/main` at `e0880a3`; the spec commit `2d5a986` is on it).

## Global Constraints

- Gate: `make check PYTHON=.venv/bin/python` passes at every commit.
- No test reads or writes the real `~/.talos/workspace`, or touches Modal, C3, Docker, GHCR, mainnet
  or an LLM. Stub `subprocess.run` and `deploy_bench` in any `tests/test_cli.py` test before
  mutation-checking it.
- Mutation checks run with `PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=$(mktemp -d)`.
- Default workspace: `Path.home() / ".talos" / "workspace"`. Env var: `TALOS_WORKSPACE`. Flag:
  `talos --workspace DIR <command>`. Marker file: `talos.config.json` only.
- Extras: `modal = ["modal>=1.5,<2"]`, `anthropic = ["anthropic>=1,<2"]`,
  `all = ["talos-tig[modal,anthropic]"]`, `dependencies = []`.
- Marker function name: `deployed_<12 hex digits>`.
- Every redeploy hint for Modal says `talos deploy`. "run `talos setup`" stays for a missing config.
- Every number in docs is MEASURED (date and command) or `ESTIMATE (unverified)`.
- Stage explicit paths only. Leave unrelated untracked files (`docs/ai/reports/`, `uv.lock`, ...)
  alone.

## Review Focus

1. **Running `talos status` from a subdirectory of a clone that has a config.** Expected: the clone's
   jobs. Test: `test_status_from_a_subdirectory_lists_the_workspace_jobs` (Task 1).
2. **`TALOS_WORKSPACE` pointing at a directory that does not exist, for a read command.** Expected:
   exit 2 and `TALOS_WORKSPACE=<path> is not a directory`, not a silent fallback. Test:
   `test_find_root_rejects_a_missing_workspace` (Task 1).
3. **A relative `--dir` given from a subdirectory.** Expected: resolved against the current
   directory, not the workspace root. Test: `test_compile_dir_stays_relative_to_cwd` (Task 1).
4. **A modal-backend config with the modal extra not installed, then `talos run`.** Expected: exit 2
   with the install command, and no job directory created. Test:
   `test_run_without_modal_extra_creates_no_job` (Task 2).
5. **A Windows clone with CRLF checkouts, deploying or checking against an app deployed from Linux.**
   Expected: the same digest. Test: `test_digest_ignores_crlf` (Task 3).

---

### Task 1: Workspace resolution

**Files:**
- Create: `talos/workspace.py`, `tests/test_workspace.py`, `tests/conftest.py`
- Modify: `talos/cli.py` (11 `Path.cwd()` sites: `grep -n 'Path.cwd()' talos/cli.py`; `build_parser`;
  `main`; the `results:` line at ~916; demo's "Results go to" line), `talos/agentic.py`
  (`_agent_env`, its call at ~235), `talos/inspect_run.py:240`, `tests/test_agentic.py` (the calls
  of `_agent_env()`)

**Interfaces:**
- Produces:
  - `talos/workspace.py`:
    - `ENV_VAR = "TALOS_WORKSPACE"`
    - `class WorkspaceError(ConfigError)`
    - `default_root() -> Path`, which returns `Path.home() / ".talos" / "workspace"`
    - `find_root(start: Path | None = None) -> Path`, following spec §1 rules 1–3. It never creates
      anything. Rule 3 calls `default_root()` at call time, so the conftest patch works.
  - `talos/agentic.py::_agent_env(worktree: Path) -> dict[str, str]`, with
    `env["TALOS_WORKSPACE"] = str(worktree)`.
  - `main()` sets `os.environ["TALOS_WORKSPACE"] = str(Path(args.workspace).expanduser().resolve())`
    when `--workspace` is given. It catches `WorkspaceError` (only), prints it to stderr and returns 2.
- `cmd_setup`, `cmd_demo` and `cmd_run` with `--fake` call `root.mkdir(parents=True, exist_ok=True)`
  before writing. For `cmd_setup` and `cmd_demo`, a `TALOS_WORKSPACE` directory that does not exist
  is created rather than refused. Resolve it with a helper,
  `find_root_for_write() -> Path`, in `workspace.py`: rule 1 without the is-dir check, then rules 2–3.

- [ ] **Step 1: `tests/conftest.py`**, with one autouse fixture. It runs
  `monkeypatch.setenv("TALOS_WORKSPACE", "")` then `monkeypatch.delenv("TALOS_WORKSPACE")`.
  A bare `delenv(raising=False)` on an absent variable records nothing, so the value `main()` writes
  for `--workspace` would leak into later tests; set-then-delete makes teardown restore the original
  state. It also runs
  `monkeypatch.setattr(talos.workspace, "default_root", lambda: Path.cwd())`. Run the full suite
  before writing `workspace.py`: the import fails. That is the expected RED for the whole task.
- [ ] **Step 2: Failing tests in `tests/test_workspace.py`.** Each test that exercises rule 3 sets
  `monkeypatch.setattr(workspace, "default_root", lambda: tmp_path / "home-ws")`.
  - `test_find_root_walks_up_to_the_config`: the config is at `ws`, the start is `ws/a/b`, expect
    `ws.resolve()`.
  - `test_find_root_nearest_config_wins`: configs at `ws` and `ws/a`, start `ws/a/b`, expect `ws/a`.
  - `test_find_root_ignores_a_dot_talos_dir`: `ws/.talos/` exists with no config, start `ws`. Expect
    `tmp_path/"home-ws"`. This catches using `.talos/` as the marker, as the agentic worktree would
    tempt.
  - `test_find_root_falls_back_to_default_root`: no config anywhere. Expect `tmp_path/"home-ws"`, and
    `!= start`. Skip if any parent of `tmp_path` holds `talos.config.json`.
  - `test_find_root_env_wins_over_walk`: env `tmp/e` (exists), config at `ws`, start inside `ws`.
    Expect `tmp/e`.
  - `test_find_root_rejects_a_missing_workspace`: env `tmp/missing`. Expect `WorkspaceError` with
    `"TALOS_WORKSPACE="` in its message, and `cli.main(["status"]) == 2`.
  - `test_workspace_flag_overrides_cwd`: chdir to `tmp/a`, which has no config. Run
    `cli.main(["--workspace", str(tmp/"w"), "demo", "--delay", "0"])` (stdout captured). Expect
    `(tmp/"w"/"runs").is_dir()` and no `tmp/a/runs`.
  - `test_demo_falls_back_to_default_workspace`: no config, default patched to `tmp/home-ws`, chdir
    to `tmp/x`. Run demo. Expect `tmp/home-ws/runs/<one job>` and no `tmp/x/runs`.
  - `test_status_from_a_subdirectory_lists_the_workspace_jobs`: write a config at `ws`, chdir to
    `ws`, run demo, chdir to `ws/sub/deeper`. `cli.main(["status", "--json"])` lists exactly that
    job.
  - `test_setup_from_a_subdirectory_updates_the_workspace_config`: a config at `ws`, chdir to `ws/sub`,
    and the stubs and answers of `tests/test_cli.py::test_setup_writes_config_and_0600_secret`. Expect
    `ws/talos.config.json` to have provider `anthropic`, no `ws/sub/talos.config.json`, and stdout to
    contain `f"Workspace: {ws.resolve()}"`.
  - `test_compile_dir_stays_relative_to_cwd`: a config at `ws` with `backend: "local"`, a marker
    string in `ws/sub/algorithm/mod.rs`, and no `ws/algorithm`. Chdir to `ws/sub`. Stub
    `cli.prepare`, `cli.docker_info` and `cli.DockerTransport` (its `remove_run` runs after every
    local compile) exactly as `tests/test_cli.py::_compile_local` does. Monkeypatch `cli.make_bench`
    to record its second positional argument (`run_dir`) and return a bench whose `evaluate` records
    `request.files`. Run
    `cli.main(["compile", "--challenge", "knapsack", "--dir", "algorithm"])`. Expect the marker in
    the recorded files and `run_dir == ws.resolve()/".talos"/"compile"`. Copy the bench-return shape
    from `tests/test_cli.py::_compile_local`.
  - In `tests/test_agentic.py`, `test_agent_env_pins_the_workspace_to_the_worktree`: with
    `TALOS_WORKSPACE=/elsewhere` set, `_agent_env(wt)["TALOS_WORKSPACE"] == str(wt)`. Update the
    existing `_agent_env()` calls to pass a path.
- [ ] **Step 3: Run** `.venv/bin/python -m pytest tests/test_workspace.py tests/test_agentic.py -q`.
  Expected: FAIL or ERROR (no `talos.workspace`).
- [ ] **Step 4: Implement.**
  - `workspace.py` as in Interfaces.
  - In `cli.py`, every state `Path.cwd()` becomes `find_root()`, or `find_root_for_write()` in setup,
    demo and `run --fake`.
  - `cmd_compile` keeps `Path(args.dir)` relative to the current directory.
  - `cmd_setup` prints `Workspace: <root>`. When `root == default_root()`, it adds
    `(the default workspace; talos --workspace DIR setup keeps it elsewhere)`.
  - The `results:` line prints `root / "runs" / job_id`. Demo's "Results go to" line prints the
    absolute run dir.
  - `cmd_status`'s "no jobs under runs/" message names the absolute `runs` path.
  - `config.load`'s not-found message includes the root:
    `f"talos.config.json not found in {root}; run \`talos setup\` first"`.
  - `inspect_run.py:240`: use `os.path.relpath` inside `try`, falling back to the absolute path on
    `ValueError` (Windows, different drive), or keep `relative_to` with a fallback.
  - `_agent_env(worktree)` and its caller.
  - Top-level `--workspace` in `build_parser` (`metavar="DIR"`, help: "the Talos workspace (default:
    nearest talos.config.json above the current directory, else ~/.talos/workspace)").
- [ ] **Step 5: Run** `make check PYTHON=.venv/bin/python`. Expected: PASS. Existing assertions on
  the `results:` text or the not-found message are updated to the new exact text, never loosened.
- [ ] **Step 6: Mutation check.** Each mutation must make `tests/test_workspace.py` or
  `tests/test_agentic.py` fail:
  - (a) the walk returns the farthest config;
  - (b) the env var is ignored;
  - (c) a missing env dir falls back silently;
  - (d) the marker is `.talos` instead of `talos.config.json`;
  - (e) rule 3 returns `start`;
  - (f) `_agent_env` drops the variable;
  - (g) `cmd_status` keeps `Path.cwd()`;
  - (h) `cmd_compile` resolves `--dir` against the root;
  - (i) `main` ignores `--workspace`.
- [ ] **Step 7: Commit** the files listed above. Message:
  `feat: find the workspace by walking up to talos.config.json, else ~/.talos/workspace (#70)`.

### Task 2: Optional extras

**Files:**
- Create: `talos/extras.py`, `tests/test_extras.py`
- Modify: `pyproject.toml`, `talos/cli.py` (`cmd_setup`, `cmd_run`, `cmd_compile`), `talos/doctor.py`
  (`check_backend`, `check_provider`), `.github/workflows/ci.yml` (the clone job's
  `pip install -e .` becomes `pip install -e ".[all]"`)

**Interfaces:**
- Produces: `talos/extras.py`:
  - `EXTRAS = {"modal": "modal", "anthropic": "anthropic"}` (extra name → import name)
  - `missing(extra: str) -> bool`, via `importlib.util.find_spec`
  - `install_hint(extra: str) -> str`
  - `require(extra: str, why: str) -> None`, which raises `ConfigError(f"{why} needs the {extra}
    extra: {install_hint(extra)}")`

  `install_hint` gives the uv form, using the command found to work in Task 6 step 1, and
  `(in a clone: pip install -e '.[<extra>]')`.

- [ ] **Step 0: Find the uv command that adds an extra to an installed tool** (MEASURED, scratch
  dir only). Build the wheel and `uv tool install --python 3.12 <wheel>` with
  `UV_TOOL_DIR`/`UV_TOOL_BIN_DIR` set to scratch directories, so the user's real tools are not
  touched. Then try `uv tool install --python 3.12 '<wheel>[modal]'`, first without and then with
  `--force`. After each, check `import modal` with the tool env's python. Record which form works;
  `install_hint` uses it.
- [ ] **Step 1: Failing tests** in `tests/test_extras.py`. They monkeypatch
  `talos.extras.importlib.util.find_spec` to return `None` for the named module and defer to the
  real function otherwise.
  - `test_require_names_the_extra_and_both_install_forms`: the message contains `talos-tig[modal]`
    and `pip install -e '.[modal]'`.
  - `test_setup_modal_without_extra_exits_2_before_asking_for_a_token`: answers
    `["modal", "anthropic", "", "sk", "SHOULD-NOT-BE-ASKED"]`. Expect rc 2 and no
    `talos.config.json`. Also wrap the asker to record its prompts, and expect exactly one
    prompt (the backend choice).
  - `test_setup_anthropic_without_extra_exits_2`: with anthropic missing and backend `local`, expect
    rc 2, no config, and `validate_provider` never called.
  - Every `run` test below monkeypatches `cli.execute_job`, `cli.make_bench` and `cli.make_provider`
    to raise `AssertionError("reached")`. Under a mutation that skips the check, the test then fails
    without touching Modal or an LLM. Without these stubs, mutation (b) would build a real
    Anthropic provider and mutation (d) a real `ModalBench`.
  - `test_run_without_modal_extra_creates_no_job`: config backend `modal`, provider `claude-cli`,
    modal missing. Run `cli.main(["run", "--challenge", "knapsack", "--direction", "x",
    "--budget-iterations", "1", "--yes"])`. Expect rc 2, no `runs/` directory, and `talos-tig[modal]`
    in stderr.
  - `test_resume_checks_the_jobs_provider_not_the_configs`: config provider `claude-cli`, a job dir
    whose `job.json` has provider `anthropic`, and anthropic missing. `run --resume <id>` returns 2,
    with `talos-tig[anthropic]` in stderr. Build the job dir the way existing resume tests in
    `tests/test_cli.py` do.
  - `test_fake_run_and_demo_need_no_extras`: both extras missing. `cli.main(["demo", "--delay", "0"])
    == 0`, and the `run --fake` invocation existing tests use returns 0.
  - `test_doctor_reports_a_missing_extra`: config backend `modal`, modal missing. The `backend`
    check is FAIL and its detail contains `talos-tig[modal]`. The same holds for provider
    `anthropic` in the `provider` row.
- [ ] **Step 2: Run** them. Expected: FAIL (no `talos.extras`).
- [ ] **Step 3: Implement.**
  - `require("modal", "the modal backend")` runs in setup right after the backend choice. In
    `cmd_run` it runs right after the config load, when `cfg.backend == "modal"` and not `--fake`.
    In `cmd_compile` it runs when the backend is `modal`.
  - `require("anthropic", "the anthropic provider")` runs in setup right after the provider choice.
    In `cmd_run` it runs for a new job when `cfg.provider == "anthropic"`, and for a resume when
    `spec.provider == "anthropic"`, before `execute_job`.
  - In doctor, `check_backend` and `check_provider` return FAIL with `str(e)` from `require`.
  - `pyproject.toml`: the extras exactly as in Global Constraints.
- [ ] **Step 4: Run** `make check`. Expected: PASS. Then check the base install in a scratch venv:
  `uv venv` → `uv pip install -e .` (no extras) → `python -c "import modal"` fails, and `talos demo
  --delay 0` in an empty dir exits 0.
- [ ] **Step 5: Mutation check.**
  - (a) Remove the setup modal check.
  - (b) Check `cfg.provider` on resume.
  - (c) Require modal for `--fake`.
  - (d) Move the run check after job creation.

  Each must make `tests/test_extras.py` fail.
- [ ] **Step 6: Commit** `talos/extras.py tests/test_extras.py pyproject.toml talos/cli.py
  talos/doctor.py .github/workflows/ci.yml`, with message
  `feat: modal and anthropic become optional extras (#70)`. Push over SSH, because it touches
  `.github/workflows`.

### Task 3: Stale-deploy check and `talos deploy`

**Files:**
- Create: `talos/deploy_digest.py`, `tests/test_deploy_digest.py`
- Modify: `modal_app/talos_bench.py` (`register`), `talos/bench.py` (`ModalBench.check_deploy`, the
  hints in `_fn` and `_with_retry`), `talos/cli.py` (`deploy_bench`, `execute_job`, `cmd_compile`,
  new `cmd_deploy`, `COMMANDS`, `build_parser`), `talos/doctor.py` (`modal_function_names`,
  `_check_modal`), `docs/troubleshooting.md` (redeploy hints), existing tests asserting
  `"talos setup"` in redeploy messages (`tests/test_bench.py::test_missing_deployment_says_run_talos_setup`
  and any found by `grep -n "talos setup" tests/test_bench.py tests/test_doctor.py`)

**Interfaces:**
- Produces: `talos/deploy_digest.py`:
  - `DEPLOYED_FILES: tuple[str, ...]`, holding `modal_app/__init__.py`, `modal_app/talos_bench.py`,
    `talos/__init__.py`, `talos/challenges.py`, `talos/deploy_digest.py`, `talos/inside.py` and
    `talos/native_runner.py`. Each path is relative to the parent of the `talos` package directory,
    which is the repo root in a clone and `site-packages` in a wheel.
  - `digest(root: Path | None = None) -> str`, which returns 12 hex digits.
  - `marker_name(root: Path | None = None) -> str`, which returns `f"deployed_{digest(root)}"`.
  - `talos/bench.py::ModalBench.check_deploy() -> None`, which raises `BenchUnavailable`.
  - `talos/cli.py::cmd_deploy(args, ask) -> int`.
- Hash: sha256 over each path in `sorted(DEPLOYED_FILES)`, fed `path.encode() + b"\0" +
  data.replace(b"\r\n", b"\n")`. Keep `[:12]` of the hexdigest.

- [ ] **Step 1: Failing tests** in `tests/test_deploy_digest.py`:
  - `test_deployed_files_are_the_import_closure_of_the_modal_app`: seed the closure with
    `modal_app/__init__.py` and `modal_app/talos_bench.py`. Ast-walk every `Import` and `ImportFrom`
    node with `ast.walk`, so function bodies are included, and recurse into each new file. Resolve
    them as follows:
    - `import talos.x` and `from talos.x import y` → `talos/x.py`.
    - `from talos import x` → `talos/x.py` when that file exists. This is how `talos_bench.py`
      imports `inside`; resolving only the module part would miss it.
    - Any `talos` import also adds `talos/__init__.py`. The same rules apply for `modal_app`. Assert `set(closure) ==
    set(DEPLOYED_FILES)`. Mutation: drop `talos/native_runner.py` from the tuple, or add a
    function-level `from talos import scoring` in `inside.py`.
  - `test_digest_ignores_crlf`: copy the `DEPLOYED_FILES` into `tmp/a` and `tmp/b`, and rewrite
    `tmp/b`'s with CRLF. Expect `digest(tmp/a) == digest(tmp/b)`.
  - `test_digest_changes_with_any_deployed_file`: for each path in `DEPLOYED_FILES`, append `b"#"`
    in a copy. The digest differs from the base.
  - `test_digest_is_12_hex`: `re.fullmatch(r"[0-9a-f]{12}", digest())`.
  - `test_register_adds_the_marker`: use the recording-app pattern of
    `tests/test_talos_bench.py`. `marker_name()` is among the registered function names.
  - In `tests/test_bench.py`:
    - `test_check_deploy_missing_marker_says_talos_deploy`: build a fake `modal` module like
      `_fake_modal`, but with a `from_name` that appends `(app, name)` to a list and then raises
      NotFoundError. `_fake_modal` itself records nothing. Expect `BenchUnavailable` with `"talos deploy"` in it, and
      the looked-up name `== marker_name()`.
    - `test_check_deploy_transient_error_propagates`: from_name raises `ConnectionError`. The same
      `ConnectionError` propagates; it is not turned into `BenchUnavailable`.
  - In `tests/test_cli.py`:
    - `test_execute_job_pauses_on_stale_deploy_before_any_spend`: config backend `modal`, and
      `cli.make_bench` returns a stub whose `check_deploy` raises
      `BenchUnavailable("... talos deploy")` and whose `select_hardware`/`evaluate` record calls.
      Expect state `paused`, `"talos deploy"` in `stop_reason`, and no `select_hardware` or
      `evaluate` call. Stub every network piece the way existing `execute_job` tests do.
    - `test_deploy_command_deploys_without_a_config`: chdir to an empty `tmp`, with `cli.deploy_bench`
      stubbed to record its args. `cli.main(["deploy"]) == 0`, the recorded args are `(None, None)`,
      and stdout contains `marker_name()`.
    - `test_deploy_uses_the_running_interpreters_modal`: `shutil.which` returns `"/usr/bin/modal"`,
      and the fake `run` records argv. Every argv starts with `[sys.executable, "-m", "modal"]`.
  - In `tests/test_doctor.py`, `test_modal_missing_marker_says_stale`: `modal_lookup` returns False
    only for `marker_name()`. The backend check is FAIL, with `"talos deploy"` and `"different Talos
    code"` in its detail. Update `test_modal_function_names_match_register` only if it needs the
    marker added to the expected set.
- [ ] **Step 2: Run** them. Expected: FAIL.
- [ ] **Step 3: Implement.**
  - `deploy_digest.py`. Root default: `Path(__file__).resolve().parent.parent`.
  - `register` adds `app.function(name=marker_name(), image=probe_image(), timeout=10,
    serialized=True)(_noop)`.
  - `ModalBench.check_deploy`: from_name the marker and hydrate it. Only `modal.exception.NotFoundError`
    becomes `BenchUnavailable`, carrying the spec §3 message.
  - `execute_job`: right after `make_bench`, when `cfg.backend == "modal"` and not fake, call
    `bench.check_deploy()` inside the same `try` whose `BenchUnavailable` handler pauses the job, so
    it runs before `freeze_hardware`.
  - `cmd_compile`: when the backend is modal, call `check_deploy` before evaluating. On
    `BenchUnavailable`, print it and return 1.
  - `doctor.modal_function_names()` appends `marker_name()`. In `_check_modal`, when the only missing
    name is the marker, FAIL with "the deployed talos-bench app was built from different Talos code
    than this client; run `talos deploy`". Other missing functions say `talos deploy` too.
  - `deploy_bench`: `base = [sys.executable, "-m", "modal"]`, and the `shutil.which` branch is
    removed.
  - `cmd_deploy`: `require("modal", "talos deploy")`, then `deploy_bench(None, None)`, then print
    `f"Deployed talos-bench ({marker_name()})"`. Register the subparser with help "redeploy the Modal
    app after upgrading Talos".
  - Redeploy hints: in `ModalBench._fn`, `_with_retry`, doctor and `docs/troubleshooting.md`, replace
    "Run `talos setup` to redeploy/deploy" with `talos deploy`. Keep `talos setup` where the token
    or config is missing; in `_check_modal`'s lookup-exception branch, keep "run `talos setup` to set
    the token".
- [ ] **Step 4: Run** `make check`. Expected: PASS.
- [ ] **Step 5: Mutation check.**
  - (a) Drop the CRLF replace.
  - (b) Drop one file from `DEPLOYED_FILES`.
  - (c) Don't call `check_deploy` in `execute_job`.
  - (d) Call it after `freeze_hardware`.
  - (e) Convert every exception in `check_deploy` to `BenchUnavailable`.
  - (f) Restore the `which` branch.
  - (g) `cmd_deploy` passes a token.

  Each must fail a test.
- [ ] **Step 6: Commit** the listed files, with message
  `feat: refuse a stale Modal deploy before spending; add talos deploy (#70)`.

### Task 4: Version, metadata, wheel CI, publish workflow

**Files:**
- Create: `scripts/check_wheel.py`, `.github/workflows/publish.yml`
- Modify: `pyproject.toml`, `talos/cli.py` (`build_parser`: `--version`), `.github/workflows/ci.yml`
  (new `wheel` job), `tests/test_cli.py`

- [ ] **Step 1: Failing test** `test_version_flag_prints_the_package_version`:
  `cli.main(["--version"])` raises `SystemExit(0)`, stdout is `f"talos {talos.__version__}\n"`, and
  `importlib.metadata.version("talos-tig") == talos.__version__`. The last check needs the editable
  reinstall in step 3.
- [ ] **Step 2: Run** it. Expected: FAIL.
- [ ] **Step 3: Implement.**
  - `--version` uses `action="version"`.
  - `pyproject.toml`: `dynamic = ["version"]`, `[tool.setuptools.dynamic] version = {attr =
    "talos.__version__"}`, `readme = "README.md"`, `[project.urls]` (Homepage, Issues,
    Documentation → `docs/usage.md` on GitHub), and classifiers for Python 3.10–3.13, GPLv3+,
    OS Independent and Environment :: Console.
  - Then `uv pip install -p .venv/bin/python -e ".[all,dev]"`.
- [ ] **Step 4: `scripts/check_wheel.py WHEEL`.** It exits 1, naming every missing entry, unless the
  wheel holds the three `talos/data/*.md`, every `talos/<m>.py` for `m` in `JOB_MODULES`, and every
  path in `DEPLOYED_FILES`. Read both tuples with `ast` from the repo files, relative to the script's
  own location, without importing `talos`.
  - Verify: `uv build` then the script gives exit 0.
  - Then make a copy of the wheel with `talos/inside.py` removed, using `zip -d` on a copy. The
    script must exit 1 and name `talos/inside.py`.
- [ ] **Step 5: `wheel` job in `ci.yml`.** It uses the 3-OS matrix and bash, `setup-python@v5`
  (3.12) and `astral-sh/setup-uv` pinned to a release tag. Steps:
  1. `uv build`, then `python scripts/check_wheel.py dist/*.whl`, then
     `uv tool install --python 3.12 dist/*.whl` (no extras). Add `uv tool dir --bin` to
     `$GITHUB_PATH`.
  2. Default-workspace step. Make every temporary directory with
     `python -c "import tempfile; print(tempfile.mkdtemp())"`, not `mktemp -d`. On Windows, Git
     Bash's `mktemp` gives a `/tmp/...` path, and that is not translated when it is passed in
     `USERPROFILE`, which is what `Path.home()` reads there. Then: `export HOME=<dir>
     USERPROFILE=<dir>`, `cd <another dir>`, `talos --version`, `talos demo --delay 0`, and
     `talos status --json | python -c` asserting one job with status `won`. Finally, a
     `python -c` check that `Path.home()/".talos"/"workspace"/"runs"` is a directory, with no
     bash `test -d` on a Windows path.
  3. Env step: `TALOS_WORKSPACE=<new dir>`, demo, then a `python -c` check that
     `<dir>/runs` is a directory.
- [ ] **Step 6: `publish.yml`**, as in the earlier spec's PR 3 CI section: `on: push: tags: ['v*']`,
  top-level `permissions: contents: read`.
  - Job `build`: tag == `v` + `__version__`, `uv build`, `check_wheel.py`, then upload `dist/`.
  - Job `publish`: `needs: build`, `environment: pypi`, `permissions: id-token: write` only there,
    download, then `pypa/gh-action-pypi-publish@release/v1`.
- [ ] **Step 7: Run** `make check`. Expected: PASS. Mutation: hard-code `version="talos 0"`, and the
  version test fails.
- [ ] **Step 8: Commit** `scripts/check_wheel.py .github/workflows/ci.yml .github/workflows/publish.yml
  pyproject.toml talos/cli.py tests/test_cli.py`, with message
  `ci: install the built wheel on 3 OSes; publish on v* tags; talos --version (#70)`. Push over SSH.

### Task 5: Docs

**Files:** `README.md`, `docs/usage.md`, `docs/windows.md`, `docs/maintainers.md`,
`docs/troubleshooting.md`, `docs/compute-backends.md` (only if it says "run `talos setup` to
redeploy"), `AGENTS.md`

- [ ] **Step 1:** Run `grep -rn "pip install -e\|talos setup.*redeploy\|redeploy.*talos setup\|current directory" README.md docs/*.md AGENTS.md`
  and fix each hit to match spec §5.
- [ ] **Step 2:** Write the content of spec §5:
  - the README install (`uv tool install --python 3.12 'talos-tig[modal,anthropic]'`, plus the
    `pipx` and `pip` alternatives, plus the table of which extra each backend or provider needs);
  - the workspace paragraph and the fresh-clone `talos --workspace . setup` note;
  - the upgrade note (`talos deploy`);
  - `docs/usage.md` `## Workspace` and `### talos deploy`;
  - `docs/windows.md` with the uv install in PowerShell;
  - `docs/maintainers.md`: development install `pip install -e ".[all]"` plus `requirements-dev.txt`,
    and `## Releasing` (bump `__version__`, merge, tag, watch `publish.yml`, then
    `uvx --python 3.12 --from talos-tig talos demo`, plus the one-time trusted-publisher setup);
  - one line in AGENTS.md's "Where to look" table for the workspace (`talos/workspace.py`).
- [ ] **Step 3:** Run `make check` (agentify checks doc links). Expected: PASS.
- [ ] **Step 4: Commit** the modified docs, with message
  `docs: PyPI install, extras, workspace, talos deploy (#70)`.

### Task 6: Hand checks and PR

- [ ] **Step 1: Local wheel smoke (MEASURED).**
  - Run `uv build`, then `uv tool install --python 3.12 --force dist/*.whl`. In a fresh `HOME` and
    an empty dir, run `talos demo --delay 0 && talos status --json`.
  - Re-check that the hint from Task 2 step 0 still installs the extra.
  - Uninstall the tool afterwards, unless the owner uses it.
- [ ] **Step 2: Modal hand check.** Ask the owner first: it redeploys their `talos-bench` app. With a
  yes, from the wheel install with `[modal]`:
  1. `talos deploy`, recording its printed marker;
  2. `talos doctor` in the owner's workspace (`--workspace /home/fibonadithya/TIG/Talos`); the
     backend row must be OK.

  With a no, list both as open in the PR.
- [ ] **Step 3: PR** to `main`.
  - The body has: "Part of #70" (it closes only after the publish), the claim table, the three owner
    steps (trusted publisher, `pypi` environment, tag `v0.1.0`), the behaviour changes (default
    workspace, absolute `results:` path, `talos deploy` hints, extras), and the spike result.
  - Then follow the `review-pr` loop to merge.
  - After the owner publishes, run `uvx --python 3.12 --from talos-tig talos demo` in an empty dir
    and record it as MEASURED. Then close #70.
