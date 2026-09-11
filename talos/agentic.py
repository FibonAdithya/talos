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
    allowed path. Unlisted tools are refused by `dontAsk` mode rather than prompted for.

    Reads alone do not let a `dontAsk` agent discover file names under `algorithm/`: Glob and
    Grep are separate tools from Read and need their own allow entries over the same read
    scope (mirrors Prometheus's `_build_sandbox_settings`)."""
    read_scope = ["algorithm/**", "CHALLENGE.md", "tacit.md", "AGENTS.md",
                  ".talos/hypothesis.json"]
    allow = []
    for tool in ("Read", "Glob", "Grep"):
        allow += [f"{tool}({p})" for p in read_scope]
    allow += ["Edit(algorithm/**)", "Edit(.talos/hypothesis.json)", "Bash(talos compile:*)"]
    return {"permissions": {
        "allow": allow,
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

The algorithm files are: {", ".join(f"algorithm/{name}" for name in sorted(ctx.files))}.

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
    """spec §9: an edit outside the algorithm files fails the iteration in both modes. For codex,
    which ignores .claude/settings.json, this scope check is the ONLY enforcement of that rule."""
    hp = worktree / ".talos" / "hypothesis.json"
    if not hp.exists() or hp.read_text().strip() in ("", "{}"):
        raise AgenticError("agent did not fill in .talos/hypothesis.json")
    try:
        h = json.loads(hp.read_text())
        hypothesis = {"title": str(h["title"])[:200], "description": str(h["description"])[:4000],
                      "strategy_tag": h.get("strategy_tag") if h.get("strategy_tag") in STRATEGY_TAGS else "hybrid"}
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise AgenticError(f"bad hypothesis.json: {e}") from None
    algo_dir = worktree / "algorithm"
    on_disk = {p.relative_to(algo_dir).as_posix() for p in algo_dir.rglob("*") if p.is_file()}
    extra = on_disk - set(ctx.files)
    if extra:
        raise AgenticError(f"agent created files outside the algorithm set: {sorted(extra)}")
    missing = set(ctx.files) - on_disk
    if missing:
        raise AgenticError(f"agent deleted algorithm files: {sorted(missing)}")
    files = {name: (algo_dir / name).read_text() for name in ctx.files}
    if files == ctx.files:
        raise AgenticError("agent changed no algorithm file")
    return hypothesis, files


# spec §7.6/§10: `talos compile` inside the sandbox runs agent-controlled build code, and
# anything left in the environment is exfiltratable through it. So the child gets an allowlist,
# not the scrubbed-copy Prometheus uses (Prometheus's `_scrubbed_env` removes only known secret
# keys/prefixes and passes the rest of os.environ through). We keep: PATH (interpreter's bin dir
# prepended, as before), the login/locale/terminal basics a CLI needs to run at all (HOME, USER,
# LOGNAME, SHELL, TERM, LANG, LC_*, TZ, TMPDIR, XDG_*), and proxy variables — Prometheus's comment
# on `_scrubbed_env` notes it deliberately lets proxy vars pass through so the claude/codex login
# session still works on networks that require an outbound proxy. No provider keys, no Modal
# tokens, no anything else.
_AGENT_ENV_ALLOWLIST = frozenset({
    "HOME", "USER", "LOGNAME", "SHELL", "TERM", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "TMPDIR",
    "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy",
})

_TRANSCRIPT_CAP = 200_000


def _agent_env() -> dict[str, str]:
    """The agent runs `talos compile`; make sure the interpreter that runs Talos is first on PATH so
    the console script resolves even when the venv is not activated in the agent's shell."""
    env = {k: v for k, v in os.environ.items()
           if k in _AGENT_ENV_ALLOWLIST or k.startswith("XDG_")}
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    return env


def _run_agent(cmd: list[str], wt: Path, timeout_s: int, run) -> None:
    try:
        r = run(cmd, cwd=wt, capture_output=True, text=True, timeout=timeout_s, env=_agent_env())
    except FileNotFoundError:
        raise AgenticError(f"{cmd[0]} CLI not found on PATH") from None
    except subprocess.TimeoutExpired:
        raise AgenticError(f"{cmd[0]} timed out after {timeout_s}s") from None
    (wt / ".talos" / "agent_stdout.txt").write_text((r.stdout or "")[-_TRANSCRIPT_CAP:])
    (wt / ".talos" / "agent_stderr.txt").write_text((r.stderr or "")[-_TRANSCRIPT_CAP:])
    if r.returncode != 0:
        raise AgenticError(f"{cmd[0]} exited {r.returncode}: {(r.stderr or '')[-500:]}")


def _run_claude(wt: Path, model: str, prompt: str, timeout_s: int, run) -> None:
    _run_agent(["claude", "-p", "--model", model, "--settings", str(wt / ".claude" / "settings.json"),
                "--permission-mode", "dontAsk", prompt], wt, timeout_s, run)


def _run_codex(wt: Path, model: str, prompt: str, timeout_s: int, run) -> None:
    _run_agent(["codex", "exec", "-m", model, "--sandbox", "workspace-write",
                "--skip-git-repo-check", prompt], wt, timeout_s, run)


def _copy_transcript(wt: Path, loop) -> None:
    """prepare_worktree wipes the agentic worktree on the next iteration (shutil.rmtree), so a
    failed iteration's stdout/stderr must be copied into the iteration dir now or it is lost.
    loop._n is only set on a real Loop mid-iterate; the test's SimpleNamespace stub has none."""
    n = getattr(loop, "_n", None)
    if n is None:
        return
    it_dir = loop.store.iteration_dir(n)
    for name in ("agent_stdout.txt", "agent_stderr.txt"):
        src = wt / ".talos" / name
        if src.exists():
            shutil.copy(src, it_dir / name)


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
        try:
            runner(wt, model, prompt, timeout_s, run)
        finally:
            _copy_transcript(wt, loop)
        hypothesis, files = read_back(wt, ctx)
        loop._event("hypothesis", **hypothesis)
        return hypothesis, files

    loop.propose_and_edit = propose_and_edit
