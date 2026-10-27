# Two-agent trial of `talos guide` against `talos demo` (#71)

Non-authoritative working notes (see `docs/ai/README.md`).

## Setup

- Branch `agent-guide`. Run 1 at commit 845216d, run 2 at d8cfeb9 (the guide fix below).
- Each run used its own fresh clone in the session scratchpad, with its own venv
  (`uv pip install -e .`), no `talos.config.json` and no `.talos/`. So a spending `talos run`
  would have failed for lack of config.
- Prompt for both agents, and nothing else: "Run `talos guide` and follow it to run the Talos
  demo, then report back to me as the guide says."
- Claude Code: `claude -p "<prompt>" --allowedTools "Bash(talos:*)" Read Grep Glob
  --output-format stream-json --verbose < /dev/null`, `timeout 900`.
- Codex 0.147.0: `codex exec --sandbox workspace-write -C <clone> "<prompt>" < /dev/null`,
  `timeout 900`. The session header shows model `gpt-5.6-terra`, approval `never`.
- Caveat: both CLIs loaded the owner's global instructions and plugins (`~/.claude/`,
  `~/.codex/`). Codex spent its first commands reading a superpowers skill. These are not
  clean-room agents.

## Results (all MEASURED 2026-10-27)

| Run | Agent | Wall time | Usage | Commands it ran | Stuck? | Report followed the template? |
|---|---|---|---|---|---|---|
| 1 | Claude Code | 28 s | 5 turns, $0.32 as reported by the CLI | `talos guide`; `talos demo --help`; `talos demo --delay 0`; `talos status J --json`; `talos show J --json`; `talos diff J` | no | yes, and it said unprompted that the demo is scripted |
| 1 | Codex | 53 s | 27,090 tokens | skill files; `talos guide`; `talos demo --delay 0`; `talos status J --json && talos show J --json` | no | yes, but it did not say the demo's `won` is scripted, and it suggested `talos setup` without saying the user runs it |
| 2 | Claude Code | 30 s | 6 turns, $0.34 as reported by the CLI | as run 1, plus it read the package `README.md` | no | yes, with both fixes |
| 2 | Codex | 16 s | 35,238 tokens | skill files chained with `talos guide && talos demo --delay 0`; then status and show | no | yes, with both fixes |

Neither agent ran a spending command, raised a budget, or tried `talos setup`.

## What the trial changed in the guide

After run 1: section 2 says a demo's `won` status and package are scripted, with nothing in
them to submit. Section 6 says that after a demo, the user runs `talos setup` themselves and the
agent then runs `talos doctor` (d8cfeb9). Both agents' run-2 reports followed both points.

## Found, not fixed here

- **The demo's own output contradicts the guide.** Claude found this in run 2. `talos demo`
  ends with "You can find the code to submit at …/submission, fill out the template README.md
  to make it ready to submit", and the demo package's `README.md` has an ordinary "Submitting"
  section. A reader who sees only the package could take it for a real result. The fix belongs
  in `cmd_demo` / `talos/package.py`, not in the guide.
- **Codex ran the demo before it had read the guide.** In run 2 it chained
  `talos guide && talos demo --delay 0` into one command. That was harmless because the demo is
  free, but it shows an agent may act on the prompt's wording before the guide's rules.

## Not covered by this trial

The demo finishes in under a second. The guide's background launch, polling, stop and resume
instructions were not exercised by either agent. The stop behaviour itself was checked by hand
(plan Task B2, Step 5): `origin/main` ignored an early `kill -INT` to a background
`talos run --fake` for 20 s, this branch exited 130 at 2 s, and a `kill -INT` 6 s into a
background `talos demo --delay 1` gave `Status: cancelled` with a written package.
