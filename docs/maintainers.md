# Maintaining Talos

Material for people changing Talos rather than using it. `AGENTS.md` lists the invariants a
change must keep and what needs a human.

## Contents

- [Upgrading the TIG dev image](#upgrading-the-tig-dev-image)
- [Live smoke tests](#live-smoke-tests)
- [Development](#development)

## Upgrading the TIG dev image

`talos check-updates` (see [its reference](usage.md#talos-check-updates)) lists newer
published dev image tags. To apply an upgrade:

1. Review the upstream release and its matching source revision. Check that the proposed
   image tag exists for all eight challenges and the architectures you use.
2. Update `DEV_IMAGE_TAG` and, when needed, `MONOREPO_REF` in `talos/challenges.py` together.
3. Run `make check`. For Modal, redeploy with `talos setup` before running the backend's
   `talos doctor --full`. C3 and local Docker use the new pin on their next job; run
   `talos doctor --full` for them too when that is your backend.
4. Start new jobs. Existing measurements are tied to the old pins; the upgrade automatically
   gets fresh cache keys and local build volumes.

Talos now pins TIG dev image `0.0.8` and source with the upstream seed-obfuscation fix.
After upgrading from `0.0.7`, start a new job: saved runs from the old build cannot resume
against the new runtime. Baselines are measured again automatically. Modal users must
rerun `talos setup` to redeploy before starting; C3 and local Docker use the new image on
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
.venv/bin/pip install -r requirements-dev.txt -e .
make check PYTHON=.venv/bin/python
```

`make check` runs ruff, pytest without the `live` marker, and the agentify contract check.
It is the same command CI runs. See `AGENTS.md` for the invariants a change must keep.

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
