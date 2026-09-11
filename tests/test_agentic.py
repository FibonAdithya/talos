import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from talos.agentic import (AgenticError, attach_agentic, claude_md, prepare_worktree, read_back,
                           sandbox_settings)
from talos.prompts import PromptContext
from talos.state import JobStore


def ctx():
    return PromptContext(challenge="knapsack", template_rs="pub fn solve_challenge(", direction="go",
                         tacit="- USER: go", files={"mod.rs": "fn a(){}", "ls.rs": "fn b(){}"},
                         baseline_name="b", best_delta=0.0)


def test_prepare_worktree_layout(tmp_path):
    # mutation: dropping the algorithm/ subdir or the sandbox settings file breaks the sandbox layout
    wt = prepare_worktree(tmp_path, ctx())
    assert (wt / "algorithm" / "mod.rs").read_text() == "fn a(){}"
    assert (wt / "CHALLENGE.md").exists() and (wt / "tacit.md").read_text() == "- USER: go"
    assert (wt / ".claude" / "settings.json").exists()
    assert (wt / "AGENTS.md").exists()


def test_sandbox_denies_network_and_scopes_edits(tmp_path):
    # Claude Code applies deny before allow, so a broad deny like Edit(**) or Bash(*) would
    # silently block the very edits and compile command we allow.
    # mutation: adding "Edit(**)" or "Bash(*)" to deny breaks agentic mode entirely
    s = sandbox_settings(tmp_path)
    allow, deny = s["permissions"]["allow"], s["permissions"]["deny"]
    # Claude Code's documented prefix form is `Bash(cmd:*)`; a bare `*` is not a prefix match
    assert "Bash(talos compile:*)" in allow
    assert "Edit(algorithm/**)" in allow and "Edit(.talos/hypothesis.json)" in allow
    assert "WebFetch" in deny and "WebSearch" in deny and "Write(**)" in deny
    assert {"Bash(curl:*)", "Bash(wget:*)", "Bash(git:*)", "Bash(ssh:*)", "Bash(python:*)"} <= set(deny)
    assert "Edit(**)" not in deny and "Bash(*)" not in deny and "Bash(*:*)" not in deny
    assert s["permissions"]["defaultMode"] == "dontAsk"  # unlisted tools are refused, not prompted


def test_read_back_requires_hypothesis_and_returns_files(tmp_path):
    wt = prepare_worktree(tmp_path, ctx())
    with pytest.raises(AgenticError):
        read_back(wt, ctx())
    (wt / ".talos" / "hypothesis.json").write_text(json.dumps(
        {"title": "T", "description": "D", "strategy_tag": "local_search"}))
    (wt / "algorithm" / "mod.rs").write_text("fn a(){ 1 }")
    (wt / "algorithm" / "evil.rs").write_text("x")  # new file: not one of the algorithm's files
    hyp, files = read_back(wt, ctx())
    assert hyp["title"] == "T" and files["mod.rs"] == "fn a(){ 1 }"
    assert "evil.rs" not in files  # mutation: globbing the dir would pick it up


def test_claude_md_mentions_compile_and_hypothesis():
    # mutation: forgetting to mention the compile command or hypothesis file leaves the agent
    # with no way to know how to verify its edit or how to signal it is done
    text = claude_md(ctx())
    assert "talos compile" in text and ".talos/hypothesis.json" in text and "knapsack" in text


def test_agentic_error_is_a_failed_iteration_not_a_crash():
    # mutation: a RuntimeError base escapes Loop.iterate's except clause and kills the whole run
    assert issubclass(AgenticError, ValueError)


def test_attach_agentic_timeout_is_agentic_error_and_talos_is_on_path(tmp_path):
    seen = {}

    def run(cmd, **kw):
        seen.update(cmd=cmd, env=kw.get("env"), cwd=kw.get("cwd"))
        raise subprocess.TimeoutExpired(cmd, 1)

    loop = types.SimpleNamespace(store=JobStore(tmp_path), propose_and_edit=None,
                                 _check_budget=lambda: None, _event=lambda *a, **k: None)
    attach_agentic(loop, "claude-cli", "m", timeout_s=1, run=run)
    with pytest.raises(AgenticError):  # mutation: letting TimeoutExpired escape crashes the loop
        loop.propose_and_edit(ctx())
    assert seen["cmd"][0] == "claude"  # mutation: shutil.which() makes argv[0] machine-dependent
    assert "--permission-mode" in seen["cmd"] and seen["cwd"] == tmp_path / "agentic"
    # mutation: dropping the PATH prepend means `talos compile` is not found inside the sandbox
    assert seen["env"]["PATH"].split(os.pathsep)[0] == str(Path(sys.executable).parent)
