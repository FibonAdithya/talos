# PyPI install, default workspace, optional extras, stale-deploy check: design

Issue #70. This replaces "PR 3" of `2026-10-27-readme-split-agent-guide-pypi-design.md`. Everything
in that section still holds unless this document changes it. The owner's decisions of 2026-10-27
changed three things, and #70 now ships before #71:

| Topic | PR 3 of the earlier spec | This spec (owner, 2026-10-27) |
|---|---|---|
| No `talos.config.json` above the current directory | use the current directory | use the default workspace `~/.talos/workspace` |
| `modal`, `anthropic` | default dependencies | no default dependencies; extras `[modal]`, `[anthropic]`, `[all]` |
| Deployed Modal app built from different code than the client | not covered | refuse before any spend and tell the user to redeploy |
| Order | after #71 (`talos guide`) | before #71; no `talos guide` step in CI |
| Fresh clone | `talos setup` writes into the clone | `talos setup` writes into the default workspace unless `--workspace .` is given (owner accepted, 2026-10-27) |

## Facts this design rests on

- `talos-tig` is free on PyPI and `talos` is taken (MEASURED 2026-10-27:
  `curl -s -o /dev/null -w "%{http_code}" https://pypi.org/pypi/talos-tig/json` → 404, `.../talos/json` → 200).
- The current wheel already installs and runs from an unrelated directory (MEASURED 2026-10-27: `uv build
  --wheel`, installed into a fresh 3.11 venv, then `talos --help`, `talos status` and
  `python -I -c "import modal_app.talos_bench"` all succeeded from an empty directory). `talos/data/*.md`
  (3 files) is in the wheel.
- `talos demo` runs with neither `modal` nor `anthropic` installed (MEASURED 2026-10-27: wheel installed
  with `uv pip install --no-deps`, `python -c "import modal"` → ModuleNotFoundError, `talos demo --delay 0`
  → exit 0, `runs/` written). `modal` and `anthropic` are imported only inside functions
  (`talos/bench.py`, `talos/doctor.py`, `talos/providers/anthropic_provider.py`); the one top-level
  `import modal` is in `modal_app/talos_bench.py`, which only `modal deploy` and tests import.
- The test suite imports real `modal` and `anthropic` (`tests/test_cli.py`, `tests/test_doctor.py`,
  `tests/test_talos_bench.py`, `tests/test_providers.py`).
- About 130 test lines `chdir` into a temporary directory and expect state there (MEASURED 2026-10-27:
  `grep -c chdir tests/*.py`, 130 in total, 112 of them in `tests/test_cli.py`).
- `talos setup` asks every question again, including the LLM API key; a blank Modal token keeps the
  existing token (`deploy_bench(token_id or None, ...)`).

## 1. Workspace resolution

`talos/workspace.py::find_root(start: Path | None = None) -> Path`, in this order:

1. `TALOS_WORKSPACE` set: if it is a directory, return it resolved. Otherwise raise
   `WorkspaceError(f"TALOS_WORKSPACE={value} is not a directory")` (`WorkspaceError(ConfigError)`).
2. Walk from `(start or Path.cwd()).resolve()` through its parents. Return the first directory that
   holds `talos.config.json`. The marker is that file only. A `.talos/` directory is not a marker,
   because the agentic worktree has its own `.talos/` and `talos compile` runs inside it.
3. Otherwise return `default_root()`, which is `Path.home() / ".talos" / "workspace"`. That directory
   sits beside the existing global caches `~/.talos/baselines` and `~/.talos/calibration`.
   `find_root` never creates it.

The global flag is `talos --workspace DIR <command>`. `main()` puts its value in
`os.environ["TALOS_WORKSPACE"]` before dispatching, so rule 1 is the only code path for both the flag
and the variable. **My call.**

Missing directories:
- `talos setup`, `talos demo` and the hidden `talos run --fake` create the workspace they resolved,
  with parents. `--fake` needs no config today (`talos/cli.py:930`) and keeps working with none. That
  includes a
  `--workspace` / `TALOS_WORKSPACE` directory that does not exist yet, so `talos --workspace ~/tig setup`
  works. **My call.**
- Every other command exits 2 on a missing `TALOS_WORKSPACE` (via `WorkspaceError`). With the default
  workspace missing, `status` says there are no jobs (naming the `runs/` path) and `run` says
  `talos.config.json not found in <root>; run talos setup`.

Call sites: there are 11 `Path.cwd()` calls in `talos/cli.py`, not the 9 the earlier spec gave
(MEASURED 2026-10-27: `grep -o 'Path.cwd()' talos/cli.py | wc -l` → 11). Every one that locates state (config, `.talos/`, `runs/`) goes
through `find_root()`. User-supplied relative paths (`--dir`, `--direction-file`) stay relative to the
current directory. `talos/inspect_run.py:240` (`path.relative_to(Path.cwd())` for display) is left to
the builder, as long as it cannot raise when the workspace is not under the current directory.

Output: `talos setup` prints `Workspace: <root>` before writing. When `<root>` is the default
workspace, it adds that `talos --workspace DIR setup` keeps the workspace elsewhere. The `talos run`
`results:` line and the demo's `Results go to` line print absolute paths.

Agentic sandbox: the agent's environment sets `TALOS_WORKSPACE=<worktree>` explicitly
(`talos/agentic.py::_agent_env`), overriding any inherited value. `talos compile` in the sandbox then
behaves exactly as today: no config, so the backend comes from `TALOS_BACKEND` and the run directory is
`<worktree>/.talos/compile`. Without this, the sandbox's `talos compile` would fall through to the
default workspace and read the user's `talos.config.json` and `.talos/secrets.json`. `_agent_env()`
takes no arguments today (`talos/agentic.py:219`). It becomes `_agent_env(worktree: Path)`, and the
caller at `talos/agentic.py:235` and the existing tests (`tests/test_agentic.py`) pass the worktree.
The variable is overwritten with a per-worktree value; nothing else in the allowlist changes.

Tests: no test may read or write the real `~/.talos/workspace`. `tests/conftest.py` gets autouse
fixtures that delete `TALOS_WORKSPACE` and monkeypatch `talos.workspace.default_root` to return the
current directory at call time. That keeps the ~130 existing chdir-based tests meaning what they mean
today. The tests of rule 3 replace that patch with a temporary directory and assert the fallback lands
there, not in the current directory.

## 2. Optional dependencies

```toml
dependencies = []
[project.optional-dependencies]
modal = ["modal>=1.5,<2"]
anthropic = ["anthropic>=1,<2"]
all = ["talos-tig[modal,anthropic]"]
dev = ["pytest>=8", "ruff>=0.5"]
```

`talos/extras.py::require(extra: str, why: str) -> None` raises `ConfigError` when
`importlib.util.find_spec(<module>)` is None. The message names the reason and both install forms, for
example: `the modal backend needs the modal extra: uv tool install --python 3.12 'talos-tig[modal]'
(in a clone: pip install -e '.[modal]')`.

Where it is called, before anything is written or spent:
- `talos setup`: right after the backend choice (modal) and right after the provider choice
  (anthropic). Exit 2.
- `talos run`: before the job directory is created (new job) or before `execute_job` (resume). The
  check covers the backend from the config when it is `modal` and the provider is not `fake`.
  `--fake` and `talos demo` also build `Config(backend="modal")` (`talos/cli.py:933`, `:1209`) and
  must never require the extra. It also covers the provider the job will actually use when that is
  `anthropic`: the config's provider for a new job, `job.json`'s for a resume, because a resume keeps
  the provider it started with (`talos/cli.py:957`). Exit 2.
- `talos compile` with the modal backend, and `talos deploy`.
- `talos doctor`: a FAIL row naming the missing extra, instead of a traceback.

`demo`, `status`, `show`, `diff`, `report`, `check-updates` and `--version` need neither extra. The
wheel CI job proves this by installing the wheel with no extras.

The builder checks by hand which command adds an extra to an existing `uv tool` install, and the
message uses that command (`uv tool install ... 'talos-tig[modal]'`, with `--force` if it turns out to
be needed).

Clone installs: the clone CI job installs `-e ".[all]"`, and `docs/maintainers.md#development` says
the same. An existing clone venv keeps `modal` and `anthropic` (pip does not uninstall them).

## 3. Stale-deploy check (Modal only)

C3 and local copy the container modules into every job (`talos/c3_jobdir.py::JOB_MODULES`), so they
cannot run stale code. Modal runs whatever was deployed last.

- `talos/deploy_digest.py::DEPLOYED_FILES`: a fixed tuple of repo-relative paths, made of
  `modal_app/__init__.py`, `modal_app/talos_bench.py` and every `talos/*.py` they import, followed
  transitively. Today that is `talos/__init__.py`, `talos/challenges.py`, `talos/inside.py` and
  `talos/native_runner.py`, plus `talos/deploy_digest.py` itself once `register` imports it to name the
  marker (MEASURED 2026-10-27: `python -c "import modal_app.talos_bench, sys; print(sorted(m for m in
  sys.modules if m.split('.')[0] in ('talos','modal_app')))"`; a grep found no function-level
  `talos` imports in those files). The completeness test walks the `ast` of each file, imports
  inside functions included, and follows every `talos.*` import transitively. It asserts that the
  closure equals `DEPLOYED_FILES`. A `sys.modules` check would miss a lazy import added later.
  `deploy_digest() -> str` is the first 12 hex digits of sha256 over each file in sorted order: the
  path, a NUL, then the file bytes with CRLF turned into LF. The LF step is there because Windows
  runners check out with `autocrlf=true`. `talos/__init__.py` is in the set, so every version bump
  needs a redeploy. That is intended, because a release can move the pins.
- Marker: `modal_app/talos_bench.py::register` adds one more function, `deployed_<digest>`, on the
  probe image with `serialized=True` and a no-op body. Nothing ever calls it.
- Client: `talos/bench.py::ModalBench.check_deploy()` runs
  `modal.Function.from_name(app, f"deployed_{deploy_digest()}").hydrate()`. On `NotFoundError` it raises
  `BenchUnavailable("the deployed talos-bench app was built from different Talos code than this
  client; run `talos deploy`")`. Other exceptions propagate, as in `ModalBench._fn`. A lookup starts
  no container (MEASURED 2026-10-27; see the spike below).
- Where it runs:
  - `execute_job`, for the modal backend (not fake), before `freeze_hardware`, so before the probe
    and the baseline. The existing `BenchUnavailable` handler leaves the job `paused` and resumable
    after a redeploy.
  - `talos compile` with the modal backend.
  - `talos doctor`'s backend check: FAIL when the functions exist but the marker does not.
- The marker only works if a redeploy removes the previous `deployed_<old>` function. If the old
  marker survived, a client whose digest matched an older deploy (a downgrade, or a clone checked out
  at an older commit) would pass the check against newer code. Spike, run with the owner's go-ahead
  (MEASURED 2026-10-27, scratch scripts, `.venv` modal). A throwaway app `talos-marker-probe` was
  deployed holding only `deployed_a`. A lookup found `deployed_a`, and `deployed_b` was NotFound. It
  was then redeployed holding only `deployed_b`: `deployed_a` was NotFound and `deployed_b` was found.
  `modal app stop --yes talos-marker-probe` then reported "No containers are currently running", and
  `modal app list` showed it stopped with 0 tasks. So a redeploy removes the old marker, and a lookup
  starts no container.
- `talos deploy` is a new non-interactive command. It runs `require("modal")`, then
  `deploy_bench(None, None)` (the token already set by `talos setup` is kept), then prints the digest.
  It reads no config, so it works in any directory. **My call**: re-running `talos setup` after each
  upgrade would mean typing the LLM key again.
- Every Modal-backend message that says to redeploy says `talos deploy` instead of `talos setup`. That
  covers `ModalBench._fn`, `_with_retry`, doctor, and `docs/troubleshooting.md`.
  "Run `talos setup` first" stays for a missing config.
- `deploy_bench` uses `[sys.executable, "-m", "modal"]` only (from the earlier spec's C2). With modal
  now optional, `require("modal")` runs first, so `-m modal` resolves whenever deploy is reached.

## 4. Packaging and release

As PR 3 of the earlier spec says, with these changes:
- `talos --version` and a dynamic version from `talos/__init__.py::__version__`. The first release is
  `0.1.0`.
- `scripts/check_wheel.py` checks `talos/data/{rust_rules,evidence_template,submission_readme_template}.md`,
  `modal_app/talos_bench.py`, every `talos/<m>.py` in `JOB_MODULES`, and every path in `DEPLOYED_FILES`.
  `talos/data/agent_guide.md` is not checked; #71 adds it later.
- The CI `wheel` job runs on 3 OSes: `uv build`, `check_wheel.py`, then
  `uv tool install --python 3.12 dist/*.whl` (no extras). Then, in an empty directory with
  `HOME`/`USERPROFILE` pointed at a fresh temporary directory: `talos --version`, `talos demo --delay 0`,
  `talos status --json` (one job, status `won`, under `<fake home>/.talos/workspace/runs`). That last
  step is the end-to-end check of the default workspace. A second step with `TALOS_WORKSPACE` set
  to a different empty directory asserts the demo lands there.
- `publish.yml`, the `pypi` environment, `id-token: write` only on the publish job, the tag-equals-version
  check, and the owner's three one-time steps are as in the earlier spec.

## 5. Docs

As in the earlier spec, plus:
- README install: `uv tool install --python 3.12 'talos-tig[modal,anthropic]'` for the defaults
  (Modal + Anthropic SDK). Then one line on which extra each backend or provider needs: local, C3,
  claude-cli and codex-cli need none.
- A workspace paragraph covering the three rules and `--workspace`. The fresh-clone note says
  `talos --workspace . setup` keeps state in the clone, as before.
- Upgrade note: after upgrading Talos, run `talos deploy` if you use Modal.
- `docs/usage.md`: a `## Workspace` section and a `talos deploy` entry.
- `AGENTS.md`: the line "stores a credential anywhere other than `.talos/secrets.json`" stays true,
  because the path is relative to the workspace. Mention the workspace once in the router, without
  restating the rules.

## 6. Done when

- `make check PYTHON=.venv/bin/python` passes, with new tests for each rule of `find_root`, the
  sandbox override, the extras checks, `DEPLOYED_FILES` completeness, the digest's CRLF handling,
  `check_deploy`, and `talos deploy`. Each new test is mutation-checked.
- The CI wheel job passes on Linux, macOS and Windows, and the clone job passes with `.[all]`.
- Hand checks, each needing the owner's go-ahead because they touch the owner's Modal account: a
  `talos deploy` from the wheel install succeeds, `talos doctor` is OK, and the deploy's printed
  digest is found as `deployed_<digest>`. (Removal of an old marker on redeploy is already MEASURED by
  the spike in §3.) If any is not run, the PR lists it as open.
- After the owner publishes `v0.1.0`: `uvx --python 3.12 --from talos-tig talos demo` in a clean
  directory, recorded as MEASURED with the date.

## Non-goals

- `talos guide` and the agent guide (#71).
- Migrating existing clone state into the default workspace. A clone with a config keeps working
  because rule 2 finds it.
- Platform-specific data directories (XDG, `%APPDATA%`). `~/.talos/` is already where the caches live.
- Auto-redeploying a stale app (the owner chose refuse-and-tell).

## Calls I made (owner to check)

1. The `--workspace` flag works by setting `TALOS_WORKSPACE` in `main()`.
2. `setup`, `demo` and `run --fake` create a missing workspace; other commands do not.
3. A new `talos deploy` command. Redeploy messages point to it.
4. The stale-deploy marker is a function name carrying the digest, so the check costs no container.
5. The digest covers the files the Modal app imports, including `__init__.py`, so every version bump
   needs a redeploy.

## Left to the builder

- Whether the wheel job is a job in `ci.yml` or its own workflow.
- The exact wording of messages, within the content given above.
- `inspect_run.py`'s display path, as long as it cannot raise.
