import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

from talos.agentic import (AgenticError, CODEX_AGENTIC_ENV, attach_agentic, claude_md,
                           prepare_worktree, read_back, sandbox_settings)
from talos.prompts import PromptContext
from talos.state import JobStore


def ctx(**kw):
    return PromptContext(challenge="knapsack", template_rs="pub fn solve_challenge(", direction="go",
                         tacit="- USER: go", files={"mod.rs": "fn a(){}", "ls.rs": "fn b(){}"},
                         baseline_name="b", best_delta=0.0, **kw)


def test_prepare_worktree_layout(tmp_path):
    # mutation: dropping the algorithm/ subdir or the sandbox settings file breaks the sandbox layout
    wt = prepare_worktree(ctx(), parent=tmp_path)
    assert (wt / "algorithm" / "mod.rs").read_text() == "fn a(){}"
    assert (wt / "CHALLENGE.md").exists() and (wt / "tacit.md").read_text() == "- USER: go"
    assert (wt / ".claude" / "settings.json").exists()
    assert (wt / "AGENTS.md").exists()
    # mutation: reusing one fixed path means a second iteration inherits the first one's files
    assert prepare_worktree(ctx(), parent=tmp_path) != wt


def test_worktree_is_not_under_the_run_dir(tmp_path):
    # mutation: a worktree under runs/ puts job.json one `..` away from the agent, and job.json
    # is the only file holding the job's rand_hash
    run_dir = tmp_path / "runs" / "job-1"
    run_dir.mkdir(parents=True)
    (run_dir / "job.json").write_text("{}")
    seen = {}

    def run(cmd, **kw):
        seen.update(cwd=kw.get("cwd"))
        raise subprocess.TimeoutExpired(cmd, 1)

    loop = types.SimpleNamespace(store=JobStore(run_dir), propose_and_edit=None,
                                 _check_budget=lambda: None, _event=lambda *a, **k: None)
    attach_agentic(loop, "claude-cli", "m", timeout_s=1, run=run)
    with pytest.raises(AgenticError):
        loop.propose_and_edit(ctx())
    wt = Path(seen["cwd"])
    assert wt == loop._agentic_wt and wt.name.startswith("talos-agentic-")
    assert run_dir.resolve() not in wt.resolve().parents
    shutil.rmtree(wt, ignore_errors=True)


def test_codex_agentic_is_opt_in(tmp_path, monkeypatch):
    # mutation: attaching codex without the opt-in runs agent-authored commands on this machine
    # under a sandbox that restricts neither command execution nor reads
    monkeypatch.delenv(CODEX_AGENTIC_ENV, raising=False)
    loop = types.SimpleNamespace(store=JobStore(tmp_path), propose_and_edit=None,
                                 _check_budget=lambda: None, _event=lambda *a, **k: None)
    with pytest.raises(AgenticError) as ei:
        attach_agentic(loop, "codex-cli", "m", timeout_s=1, run=lambda *a, **k: None)
    assert CODEX_AGENTIC_ENV in str(ei.value)
    assert loop.propose_and_edit is None  # nothing was attached
    monkeypatch.setenv(CODEX_AGENTIC_ENV, "1")
    attach_agentic(loop, "codex-cli", "m", timeout_s=1, run=lambda *a, **k: None)
    assert loop.propose_and_edit is not None


def test_sandbox_denies_network_and_scopes_edits(tmp_path):
    # Claude Code applies deny before allow, so a broad deny like Edit(**) or Bash(*) would
    # silently block the very edits and compile command we allow.
    # mutation: adding "Edit(**)" or "Bash(*)" to deny breaks agentic mode entirely
    s = sandbox_settings(tmp_path)
    allow, deny = s["permissions"]["allow"], s["permissions"]["deny"]
    # Claude Code's documented prefix form is `Bash(cmd:*)`; a bare `*` is not a prefix match
    assert "Bash(talos compile:*)" in allow
    assert "Edit(algorithm/**)" in allow and "Edit(.talos/hypothesis.json)" in allow
    # mutation: dropping Glob/Grep entries leaves a dontAsk agent unable to discover file names
    # under algorithm/ (Read alone does not grant Glob/Grep access)
    assert "Glob(algorithm/**)" in allow and "Grep(algorithm/**)" in allow
    assert "WebFetch" in deny and "WebSearch" in deny and "Write(**)" in deny
    assert {"Bash(curl:*)", "Bash(wget:*)", "Bash(git:*)", "Bash(ssh:*)", "Bash(python:*)"} <= set(deny)
    assert "Edit(**)" not in deny and "Bash(*)" not in deny and "Bash(*:*)" not in deny
    assert s["permissions"]["defaultMode"] == "dontAsk"  # unlisted tools are refused, not prompted


def test_read_back_requires_hypothesis_and_returns_files(tmp_path):
    wt = prepare_worktree(ctx(), parent=tmp_path)
    with pytest.raises(AgenticError):
        read_back(wt, ctx())
    (wt / ".talos" / "hypothesis.json").write_text(json.dumps(
        {"title": "T", "description": "D", "strategy_tag": "local_search"}))
    (wt / "algorithm" / "mod.rs").write_text("fn a(){ 1 }")
    (wt / "algorithm" / "evil.rs").write_text("x")  # new file: not one of the algorithm's files
    with pytest.raises(AgenticError):
        # mutation: silently dropping a new file lets a codex agent add files the loop never sees
        read_back(wt, ctx())
    (wt / "algorithm" / "evil.rs").unlink()
    hyp, files = read_back(wt, ctx())
    assert hyp["title"] == "T" and files["mod.rs"] == "fn a(){ 1 }" and files["ls.rs"] == "fn b(){}"


def test_read_back_rejects_deleted_algorithm_file(tmp_path):
    wt = prepare_worktree(ctx(), parent=tmp_path)
    (wt / ".talos" / "hypothesis.json").write_text(json.dumps(
        {"title": "T", "description": "D", "strategy_tag": "local_search"}))
    (wt / "algorithm" / "mod.rs").write_text("fn a(){ 1 }")
    (wt / "algorithm" / "ls.rs").unlink()  # the file set is fixed; a deletion is out of scope
    with pytest.raises(AgenticError):
        # mutation: not checking for deletions lets a shrunk algorithm file set through unnoticed
        read_back(wt, ctx())


def test_read_back_caps_hypothesis_title_and_description(tmp_path):
    wt = prepare_worktree(ctx(), parent=tmp_path)
    (wt / ".talos" / "hypothesis.json").write_text(json.dumps(
        {"title": "T" * 500, "description": "D" * 5000, "strategy_tag": "local_search"}))
    (wt / "algorithm" / "mod.rs").write_text("fn a(){ 1 }")
    hyp, _ = read_back(wt, ctx())
    # mutation: dropping the length cap lets an unbounded hypothesis blow up the timeline/event log
    assert len(hyp["title"]) == 200 and len(hyp["description"]) == 4000


def test_claude_md_mentions_compile_and_hypothesis():
    # mutation: forgetting to mention the compile command or hypothesis file leaves the agent
    # with no way to know how to verify its edit or how to signal it is done
    text = claude_md(ctx())
    assert "talos compile" in text and ".talos/hypothesis.json" in text and "knapsack" in text
    # mutation: without the file list, a dontAsk agent (no Glob-free way to browse) cannot find
    # the algorithm files by name
    assert "mod.rs" in text and "ls.rs" in text


def test_agentic_error_is_a_failed_iteration_not_a_crash():
    # mutation: a RuntimeError base escapes Loop.iterate's except clause and kills the whole run
    assert issubclass(AgenticError, ValueError)


def test_attach_agentic_timeout_is_agentic_error_and_talos_is_on_path(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-leak")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "ms-leak")
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
    # mutation: a worktree under runs/ puts job.json one `..` away from the agent
    assert "--permission-mode" in seen["cmd"] and seen["cwd"] == loop._agentic_wt
    assert tmp_path.resolve() not in Path(seen["cwd"]).resolve().parents
    # mutation: dropping the PATH prepend means `talos compile` is not found inside the sandbox
    assert seen["env"]["PATH"].split(os.pathsep)[0] == str(Path(sys.executable).parent)
    # mutation: inheriting the parent environment hands the agent every credential Talos holds
    assert "ANTHROPIC_API_KEY" not in seen["env"] and "MODAL_TOKEN_SECRET" not in seen["env"]
    assert seen["env"]["HOME"] == os.environ["HOME"]
    # mutation: dropping these makes the CLI look logged out wherever its config is not in $HOME
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/tmp/cc")
    monkeypatch.setenv("CODEX_HOME", "/tmp/ch")
    with pytest.raises(AgenticError):
        loop.propose_and_edit(ctx())
    assert seen["env"]["CLAUDE_CONFIG_DIR"] == "/tmp/cc" and seen["env"]["CODEX_HOME"] == "/tmp/ch"
    shutil.rmtree(loop._agentic_wt, ignore_errors=True)


def test_run_agent_caps_transcript_length(tmp_path):
    from talos.agentic import _run_agent
    wt = prepare_worktree(ctx(), parent=tmp_path)
    huge = "x" * 250_000

    def run(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 0, stdout=huge, stderr=huge)

    _run_agent(["claude"], wt, 10, run)
    # mutation: forgetting the cap lets a runaway agent transcript blow up disk and git history
    assert len((wt / ".talos" / "agent_stdout.txt").read_text()) == 200_000
    assert len((wt / ".talos" / "agent_stderr.txt").read_text()) == 200_000


def test_attach_agentic_copies_transcript_to_iteration_dir(tmp_path):
    def run(cmd, **kw):
        wt = kw["cwd"]
        (wt / ".talos" / "hypothesis.json").write_text(json.dumps(
            {"title": "T", "description": "D", "strategy_tag": "local_search"}))
        (wt / "algorithm" / "mod.rs").write_text("fn a(){ 1 }")
        return subprocess.CompletedProcess(cmd, 0, stdout="agent output", stderr="")

    loop = types.SimpleNamespace(store=JobStore(tmp_path), propose_and_edit=None,
                                 _check_budget=lambda: None, _event=lambda *a, **k: None, _n=3)
    attach_agentic(loop, "claude-cli", "m", timeout_s=1, run=run)
    loop.propose_and_edit(ctx())
    it_dir = loop.store.iteration_dir(3)
    shutil.rmtree(loop._agentic_wt, ignore_errors=True)
    # mutation: copying only after read_back succeeds (instead of in a finally) loses the
    # transcript whenever the iteration fails, which is exactly when it is needed for debugging
    assert (it_dir / "agent_stdout.txt").read_text() == "agent output"


def test_agent_env_passes_the_backend_through(monkeypatch):
    from talos.agentic import _agent_env
    monkeypatch.setenv("TALOS_BACKEND", "c3")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    env = _agent_env()
    # mutation: dropping TALOS_BACKEND from the allowlist sends agentic C3 compiles to Modal;
    # widening the allowlist leaks keys into the sandbox
    assert env["TALOS_BACKEND"] == "c3" and "ANTHROPIC_API_KEY" not in env


def test_claude_md_names_the_focus_track():
    # mutation: the agentic goal line still says "every active track" for a focused job
    c = ctx()
    c.track, c.guard_tracks = "n=1", ["n=2"]
    text = claude_md(c)
    assert "n=1" in text and "n=2" in text and "regression guard" in text
    assert "regression guard" not in claude_md(ctx())


def test_agentic_prompt_describes_failed_attempts_with_their_numbers(tmp_path):
    seen = {}

    def run(cmd, **kw):
        seen["prompt"] = cmd[-1]
        raise subprocess.TimeoutExpired(cmd, 1)

    loop = types.SimpleNamespace(store=JobStore(tmp_path), propose_and_edit=None,
                                 _check_budget=lambda: None, _event=lambda *a, **k: None)
    attach_agentic(loop, "claude-cli", "m", timeout_s=1, run=run)
    failed = [{"title": "Dynamic greedy", "outcome": "failed:score", "mean_rel_delta": -0.0005,
               "worst_track": "n_items=5000,budget=10", "worst_rel_delta": -0.0037,
               "runtime_ratio": 14.07}]
    with pytest.raises(AgenticError):
        loop.propose_and_edit(ctx(failed_hypotheses=failed))
    shutil.rmtree(loop._agentic_wt, ignore_errors=True)
    # mutation: joining titles alone drops the per-track delta and the runtime ratio
    assert "Dynamic greedy" in seen["prompt"] and "14.1x" in seen["prompt"]
    assert "n_items=5000,budget=10" in seen["prompt"]


def test_claude_md_shows_the_hyperparameters():
    from talos.agentic import claude_md
    text = claude_md(ctx(hyperparameters={"t": {"x": 1}}))
    # mutation: the agentic brief omitting the block while single-shot prompts carry it
    assert 'track t: {"x":1}' in text and "do not rename or remove" in text
