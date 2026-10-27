# Running on Windows

Talos runs natively on Windows; WSL is not needed. (Inside WSL, follow the Linux instructions
in the [README](../README.md) instead.) CI runs the full test suite on `windows-latest` for
every pull request. No maintainer has run a real job end to end on a Windows machine yet, so
if something fails there, please open an issue with the command and its output.

Every block below is for PowerShell and can be pasted whole.

## 1. Install uv

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Close PowerShell and open a new window, so that `uv` is on `PATH`. uv downloads the Python
it needs, so you do not install Python separately. Other ways to install uv are on
[its installation page](https://docs.astral.sh/uv/getting-started/installation/).

## 2. Install Talos

```powershell
uv tool install --python 3.12 'talos-tig[modal,anthropic]'
uv tool update-shell
```

Close PowerShell and open a new window, then check:

```powershell
talos --version
```

`uv tool update-shell` puts uv's tool directory on `PATH`.
The `modal` extra is for the Modal backend and the `anthropic` extra is for the `anthropic`
provider; the [README](../README.md#install) says which extras you need and why to name them
all in one command. Talos keeps its config, secrets and runs in `%USERPROFILE%\.talos\workspace`
unless you pass `--workspace` (see [Workspace](usage.md#workspace)). This install command has not
been verified on Windows.

## 3. Try it with no accounts

```powershell
talos demo
```

See [Try it with no accounts](../README.md#try-it-with-no-accounts) for what it prints.

## 4. Configure

Pick a [compute backend](../README.md#pick-a-compute-backend) and an
[LLM provider](../README.md#pick-an-llm-provider), then:

```powershell
talos setup
```

It asks the same questions on every OS; they are listed under
[`talos setup`](usage.md#talos-setup). On the C3 backend, give it a C3 API key (see
[What differs on Windows](#what-differs-on-windows)).

## 5. Check the setup, then run the smoke check once

```powershell
talos doctor
talos doctor --full
```

`doctor` spends no compute; its provider check makes one small LLM test call. `doctor --full`
compiles and scores one real baseline on your backend; it shows the cost and asks `Spend this?`
(default no) first. The measured figure is for knapsack
on C3 only (MEASURED 2026-09-15: £0.02 and 12 min 20 s); see [`talos doctor`](usage.md#talos-doctor).

## 6. Run a job

Interactively:

```powershell
talos run
```

Or with every answer given as a flag:

```powershell
talos run --challenge knapsack --direction "Try a tighter upper bound in the branch-and-bound pruning" --budget-usd 20 --budget-hours 4 --budget-compute-usd 10 --yes
```

List jobs, and resume one by its id:

```powershell
talos status
talos run --resume 20260917-101006-knapsack
```

[Running a job](usage.md#running-a-job) and the [command reference](usage.md#command-reference)
apply unchanged.

## What differs on Windows

- **Modal backend**: the `modal` extra (step 2). `talos setup` and `talos deploy` run the
  `modal` client that was installed with Talos.
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

- **Secrets**: Windows has no owner-only file permissions, so `.talos\secrets.json` in the
  workspace is not restricted to your user the way it is on Linux and macOS (mode `0600`).
  The default workspace is `%USERPROFILE%\.talos\workspace`, inside your user folder; if you
  use `--workspace` elsewhere, choose a folder only you can read.
- **Baseline cache**: `~/.talos/baselines/` is `$env:USERPROFILE\.talos\baselines\`.

## Using `cmd.exe` instead of PowerShell

The `talos` commands are the same. Setting an environment variable differs:

| PowerShell | `cmd.exe` |
|---|---|
| `$env:VAR = "value"` | `set VAR=value` |

For development on Windows, see
[Development on Windows](maintainers.md#development-on-windows).
