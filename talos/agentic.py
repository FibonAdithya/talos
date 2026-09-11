"""Agentic mode: one headless claude or codex call per iteration inside a sandboxed worktree.
The agent edits algorithm files, may run `talos compile`, and must write .talos/hypothesis.json.
The loop reads the files back and owns compile, score and publish as in single-shot mode."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from talos.prompts import PromptContext, STRATEGY_TAGS, _rust_rules


class AgenticError(ValueError):
    """A ValueError so Loop.iterate records it as a failed iteration instead of crashing the run."""


def sandbox_settings(worktree: Path) -> dict:
    """Deny is evaluated before allow in Claude Code, so never deny a glob that covers an
    allowed path. Unlisted tools are refused by `dontAsk` mode rather than prompted for."""
    return {"permissions": {
        "allow": ["Read(algorithm/**)", "Read(CHALLENGE.md)", "Read(tacit.md)", "Read(AGENTS.md)",
                  "Read(.talos/hypothesis.json)", "Edit(algorithm/**)",
                  "Edit(.talos/hypothesis.json)", "Bash(talos compile:*)"],
        "deny": ["WebFetch", "WebSearch", "Write(**)", "Bash(curl:*)", "Bash(wget:*)", "Bash(git:*)",
                 "Bash(ssh:*)", "Bash(python:*)", "Bash(pip:*)", "Bash(nc:*)", "Bash(rm:*)"],
        "defaultMode": "dontAsk"}}


def claude_md(ctx: PromptContext) -> str:
    return f"""# Talos agentic iteration: {ctx.challenge}

You are improving a Rust solver for the TIG challenge "{ctx.challenge}". Beat the mainnet
baseline "{ctx.baseline_name}" on TIG's benchmark (higher verifier quality per nonce under a
fixed fuel budget on every active track). Your current best is {ctx.best_delta:+.3%} vs baseline.

Rules:
- Edit ONLY files under `algorithm/`. Do not create new files. Do not touch anything else.
- You may run `talos compile --challenge {ctx.challenge} --dir algorithm` to check the build.
  Nothing else may be executed. There is no network.
- Make ONE focused change per iteration that implements a single hypothesis.
- Before you stop, EDIT the existing file `.talos/hypothesis.json` (it starts as `{{}}`) so it
  holds keys "title", "description", "strategy_tag" (one of: {", ".join(STRATEGY_TAGS)}).
  Use the Edit tool; creating new files is not permitted.

Direction from the user:
{ctx.direction}

Tacit knowledge so far is in `tacit.md`. The solver contract is in `CHALLENGE.md`.

{_rust_rules()}
"""


def prepare_worktree(run_dir: Path, ctx: PromptContext) -> Path:
    wt = Path(run_dir) / "agentic"
    if wt.exists():
        shutil.rmtree(wt)
    (wt / "algorithm").mkdir(parents=True)
    (wt / ".talos").mkdir()
    (wt / ".talos" / "hypothesis.json").write_text("{}\n")  # agent must Edit, Write is denied
    (wt / ".claude").mkdir()
    for name, text in ctx.files.items():
        p = wt / "algorithm" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    (wt / "CHALLENGE.md").write_text(f"# {ctx.challenge} solver contract (template.rs)\n\n"
                                     f"```rust\n{ctx.template_rs}\n```\n")
    (wt / "tacit.md").write_text(ctx.tacit)
    md = claude_md(ctx)
    (wt / "CLAUDE.md").write_text(md)
    (wt / "AGENTS.md").write_text(md)
    (wt / ".claude" / "settings.json").write_text(json.dumps(sandbox_settings(wt), indent=1))
    return wt


def read_back(worktree: Path, ctx: PromptContext) -> tuple[dict, dict[str, str]]:
    hp = worktree / ".talos" / "hypothesis.json"
    if not hp.exists() or hp.read_text().strip() in ("", "{}"):
        raise AgenticError("agent did not fill in .talos/hypothesis.json")
    try:
        h = json.loads(hp.read_text())
        hypothesis = {"title": str(h["title"]), "description": str(h["description"]),
                      "strategy_tag": h.get("strategy_tag") if h.get("strategy_tag") in STRATEGY_TAGS else "hybrid"}
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise AgenticError(f"bad hypothesis.json: {e}") from None
    files = {name: (worktree / "algorithm" / name).read_text() for name in ctx.files
             if (worktree / "algorithm" / name).exists()}
    if files == ctx.files:
        raise AgenticError("agent changed no algorithm file")
    return hypothesis, files


def _agent_env() -> dict[str, str]:
    """The agent runs `talos compile`; make sure the interpreter that runs Talos is first on PATH so
    the console script resolves even when the venv is not activated in the agent's shell."""
    env = dict(os.environ)
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    return env


def _run_agent(cmd: list[str], wt: Path, timeout_s: int, run) -> None:
    try:
        r = run(cmd, cwd=wt, capture_output=True, text=True, timeout=timeout_s, env=_agent_env())
    except FileNotFoundError:
        raise AgenticError(f"{cmd[0]} CLI not found on PATH") from None
    except subprocess.TimeoutExpired:
        raise AgenticError(f"{cmd[0]} timed out after {timeout_s}s") from None
    (wt / ".talos" / "agent_stdout.txt").write_text(r.stdout or "")
    (wt / ".talos" / "agent_stderr.txt").write_text(r.stderr or "")
    if r.returncode != 0:
        raise AgenticError(f"{cmd[0]} exited {r.returncode}: {(r.stderr or '')[-500:]}")


def _run_claude(wt: Path, model: str, prompt: str, timeout_s: int, run) -> None:
    _run_agent(["claude", "-p", "--model", model, "--settings", str(wt / ".claude" / "settings.json"),
                "--permission-mode", "dontAsk", prompt], wt, timeout_s, run)


def _run_codex(wt: Path, model: str, prompt: str, timeout_s: int, run) -> None:
    _run_agent(["codex", "exec", "-m", model, "--sandbox", "workspace-write",
                "--skip-git-repo-check", prompt], wt, timeout_s, run)


def attach_agentic(loop, provider_kind: str, model: str, timeout_s: int = 1800,
                   run=subprocess.run) -> None:
    runner = {"claude-cli": _run_claude, "codex-cli": _run_codex}.get(provider_kind)
    if runner is None:
        raise AgenticError(f"agentic mode needs claude-cli or codex-cli, not {provider_kind}")

    def propose_and_edit(ctx: PromptContext) -> tuple[dict, dict[str, str]]:
        wt = prepare_worktree(loop.store.run_dir, ctx)
        prompt = ("Read CLAUDE.md (or AGENTS.md), then make one improvement to the solver under "
                  "algorithm/, check it with `talos compile`, and write .talos/hypothesis.json.")
        if ctx.failed_hypotheses:
            prompt += "\nAlready tried and failed against this code: " + "; ".join(
                h.get("title", "") for h in ctx.failed_hypotheses)
        if ctx.forced_tag:
            prompt += f"\nYou have stagnated: use strategy_tag \"{ctx.forced_tag}\"."
        loop._check_budget()
        runner(wt, model, prompt, timeout_s, run)
        hypothesis, files = read_back(wt, ctx)
        loop._event("hypothesis", **hypothesis)
        return hypothesis, files

    loop.propose_and_edit = propose_and_edit
