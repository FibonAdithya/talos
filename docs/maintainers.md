# Maintaining Talos

Material for people changing Talos rather than using it. `AGENTS.md` lists the invariants a
change must keep and what needs a human.

## Contents

- [Upgrading the TIG dev image](#upgrading-the-tig-dev-image)
- [Live smoke tests](#live-smoke-tests)
- [Development](#development)
- [Releasing](#releasing)

## Upgrading the TIG dev image

`talos check-updates` (see [its reference](usage.md#talos-check-updates)) lists newer
published dev image tags. To apply an upgrade:

1. Review the upstream release and its matching source revision. Check that the proposed
   image tag exists for all eight challenges and the architectures you use.
2. Update `DEV_IMAGE_TAG` and, when needed, `MONOREPO_REF` in `talos/challenges.py` together.
3. Run `make check`. For Modal, run `talos deploy` before running the backend's
   `talos doctor --full`. C3 and local Docker use the new pin on their next job; run
   `talos doctor --full` for them too when that is your backend.
4. Start new jobs. Existing measurements are tied to the old pins; the upgrade automatically
   gets fresh cache keys and local build volumes.

Talos now pins TIG dev image `0.0.8` and source with the upstream seed-obfuscation fix.
After upgrading from `0.0.7`, start a new job: saved runs from the old build cannot resume
against the new runtime. Baselines are measured again automatically. Modal users must
run `talos deploy` to redeploy before starting; C3 and local Docker use the new image on
the next job. Run `talos doctor --full` after upgrading.

## Live smoke tests

`tests/test_live.py` (marker `live`, excluded from `make check`) compiles and scores the
real mainnet top algorithm for one challenge on your deployed Modal app:

```bash
TALOS_LIVE_CHALLENGE=knapsack .venv/bin/pytest -m live tests/test_live.py -s
```

For the C3 backend, a separate test runs one real C3 job end to end. It needs `c3 login` (or
`C3_API_KEY` set) and a few pence of credit (MEASURED 2026-09-15: £0.02 billed, 12 min 20 s
wall clock, job `SUCCEEDED`, `1 passed`; the breakdown is under
[C3 timings](compute-backends.md#c3-timings)):

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

Users run `talos doctor --full` instead (see [`talos doctor`](usage.md#talos-doctor)). The
maintainers ran the C3 test on 2026-09-15 and the Modal test on 2026-10-26 (MEASURED: knapsack,
mainnet algorithm `satchel`, both nonces `ok`, `1 passed` in 260.71 s, at commit `76e3ccd`).

## Development

The gate needs Python 3.11 or newer, because the pinned `agentify` in `requirements-dev.txt`
does. CI uses 3.12. If `python3` is older, create the virtualenv with a newer interpreter,
for example `python3.12 -m venv .venv`.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt -e ".[all]"
make check PYTHON=.venv/bin/python
```

`make check` runs ruff, pytest without the `live` marker, and the agentify contract check.
It is what CI's `check` job runs. CI also runs a `wheel` job (build, `scripts/check_wheel.py`,
install with no extras) that `make check` does not cover. See `AGENTS.md` for the invariants a
change must keep.

`".[all]"` installs the `modal` and `anthropic` extras, as CI does, because some tests
import those packages. Run `talos` from the clone with `talos --workspace . <command>` to
keep its state in the clone instead of `~/.talos/workspace` (see
[Workspace](usage.md#workspace)).

### Development on Windows

`make` is not installed by default. This is the [development](#development) install, followed
by the three commands `make check` runs:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt -e ".[all]"
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe -m pytest -q -m "not live"
.venv\Scripts\python.exe -m agentify check .
```

## Releasing

The package is `talos-tig` on PyPI. `.github/workflows/publish.yml` publishes it when a tag
matching `v*` is pushed, using PyPI trusted publishing (no token is stored).

1. Bump `__version__` in `talos/__init__.py` and merge the change to `main`. A new version
   needs a Modal redeploy for every user (`talos deploy`), because the deployed app is keyed
   to the code.
2. Tag the merge commit and push the tag:

   ```bash
   git tag v<version>
   git push origin v<version>
   ```

3. Watch the `publish` workflow. Its `build` job fails unless the tag equals `v` plus
   `__version__`; it then builds the wheel and runs `scripts/check_wheel.py` on it. The
   `publish` job runs in the `pypi` environment and uploads with
   `pypa/gh-action-pypi-publish`.
4. Check the release from PyPI, in a directory with no Talos config:

   ```bash
   uvx --python 3.12 --from talos-tig talos demo
   ```

   `talos demo` needs no extras, account or network.

One-time setup by the repository owner: create the `pypi` environment in the GitHub
repository's settings, and add a trusted publisher for `talos-tig` on PyPI with owner
`FibonAdithya`, repository `talos`, workflow `.github/workflows/publish.yml` (in PyPI's
form, the workflow name is "publish.yml", without the directories) and environment `pypi`.
Until the first release exists, add it as a pending publisher. This follows PyPI's
trusted-publishing documentation and has
not yet been done for this project.
