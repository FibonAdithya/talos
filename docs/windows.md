# Running on Windows

Talos runs natively on Windows; WSL is not needed. (Inside WSL, follow the Linux instructions
in the [README](../README.md) instead.) CI runs the full test suite on `windows-latest` for
every pull request. No maintainer has run a real job end to end on a Windows machine yet, so
if something fails there, please open an issue with the command and its output.

Every block below is for PowerShell and can be pasted whole. The commands call
`.venv\Scripts\talos.exe` directly, so they work without activating the virtualenv and
without changing PowerShell's script execution policy.

## 1. Install Git and Python

```powershell
winget install --id Git.Git -e --source winget
winget install --id Python.Python.3.12 -e --source winget
```

Close PowerShell and open a new window, so that `git` and `python` are on `PATH`.

If `winget` is not available, install [Git for Windows](https://git-scm.com/download/win)
and [Python](https://www.python.org/downloads/windows/) from their installers. In the
Python installer, tick "Add python.exe to PATH".

## 2. Install Talos

```powershell
git clone https://github.com/FibonAdithya/talos.git
cd talos
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e .
.venv\Scripts\talos.exe --help
```

Run every later block from this `talos` folder.

## 3. Try it with no accounts

```powershell
.venv\Scripts\talos.exe demo
```

See [Try it with no accounts](../README.md#try-it-with-no-accounts) for what it prints.

## 4. Configure

Pick a [compute backend](../README.md#pick-a-compute-backend) and an
[LLM provider](../README.md#pick-an-llm-provider), then:

```powershell
.venv\Scripts\talos.exe setup
```

It asks the same questions on every OS; they are listed under
[`talos setup`](usage.md#talos-setup). On the C3 backend, give it a C3 API key (see
[What differs on Windows](#what-differs-on-windows)).

## 5. Check the setup, then run the smoke check once

```powershell
.venv\Scripts\talos.exe doctor
.venv\Scripts\talos.exe doctor --full
```

`doctor` spends no compute; its provider check makes one small LLM test call. `doctor --full`
compiles and scores one real baseline on your backend; it shows the cost and asks `Spend this?`
(default no) first. The measured figure is for knapsack
on C3 only (MEASURED 2026-09-15: £0.02 and 12 min 20 s); see [`talos doctor`](usage.md#talos-doctor).

## 6. Run a job

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

[Running a job](usage.md#running-a-job) and the [command reference](usage.md#command-reference)
apply unchanged.

## Optional: activate the virtualenv

Activation lets you type `talos` instead of `.venv\Scripts\talos.exe`. The first line allows
the activation script to run in this window only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned -Force
.venv\Scripts\activate
talos --help
```

## What differs on Windows

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
  you run Talos from. [Agentic mode](usage.md#agentic-mode) uses the same sandbox as on other
  systems. The codex opt-in is:

  ```powershell
  $env:TALOS_ALLOW_CODEX_AGENTIC = "1"
  ```

- **Secrets**: Windows has no owner-only file permissions, so `.talos\secrets.json` is not
  restricted to your user the way it is on Linux and macOS (mode `0600`). Keep the repository
  in a folder only you can read.
- **Baseline cache**: `~/.talos/baselines/` is `$env:USERPROFILE\.talos\baselines\`.

## Using `cmd.exe` instead of PowerShell

The `.venv\Scripts\...` commands are the same. Two things differ:

| PowerShell | `cmd.exe` |
|---|---|
| `$env:VAR = "value"` | `set VAR=value` |
| `.venv\Scripts\activate` | `.venv\Scripts\activate.bat` (no execution policy step) |

For development on Windows, see
[Development on Windows](maintainers.md#development-on-windows).
