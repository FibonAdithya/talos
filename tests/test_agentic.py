import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

from talos.agentic import (AgenticError, CODEX_AGENTIC_ENV, SETTINGS_REL, attach_agentic,
                           claude_md, codex_agentic_refused, prepare_worktree, read_back,
                           rule_root, sandbox_settings)
from talos.prompts import PromptContext, gap_block
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
    assert json.loads((wt / SETTINGS_REL).read_text()) == sandbox_settings(wt)
    # mutation: a copy at .claude/settings.json is read as project settings, whose allow list
    # Claude Code ignores in an untrusted directory (and says so on every run)
    assert not (wt / ".claude").exists()
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
    root = rule_root(str(tmp_path))  # its exact form is pinned by the literal cases below
    # Claude Code's documented prefix form is `Bash(cmd:*)`; a bare `*` is not a prefix match
    assert "Bash(talos compile:*)" in allow
    assert f"Edit({root}/algorithm/**)" in allow and f"Edit({root}/.talos/hypothesis.json)" in allow
    assert f"Read({root}/algorithm/**)" in allow
    # Edit rules also grant Write, so the edit scope is exactly these two paths per root.
    # mutation: any wider edit rule (<wt>/**, <wt>/*, <wt>/.talos/*, or the settings file
    # itself) would let the agent rewrite its own permissions
    roots = {rule_root(str(tmp_path)), rule_root(str(tmp_path.resolve()))}
    assert {r for r in allow if r.startswith("Edit(")} == {
        f"Edit({r}/{p})" for r in roots for p in ("algorithm/**", ".talos/hypothesis.json")}
    assert "WebFetch" in deny and "WebSearch" in deny
    assert {"Bash(curl:*)", "Bash(wget:*)", "Bash(git:*)", "Bash(ssh:*)", "Bash(python:*)"} <= set(deny)
    assert "Edit(**)" not in deny and "Bash(*)" not in deny and "Bash(*:*)" not in deny
    assert s["permissions"]["defaultMode"] == "dontAsk"  # unlisted tools are refused, not prompted


def test_every_file_rule_is_absolute_to_the_worktree(tmp_path):
    # Claude Code resolves a relative rule against the shell's current directory, which moves
    # with every `cd`. Agents run `cd <wt>/algorithm && grep ...` before editing; afterwards
    # `Edit(algorithm/**)` matched nothing and 34 of 37 such sessions on tig-adi had Edit denied.
    # mutation: going back to relative rules (`Edit(algorithm/**)`) fails this test
    s = sandbox_settings(tmp_path)
    file_rules = [r for r in s["permissions"]["allow"] if r.startswith(("Read(", "Edit("))]
    assert file_rules
    root = rule_root(str(tmp_path))
    assert root.startswith("//")
    for r in file_rules:
        inner = r[r.index("(") + 1:-1]
        assert inner.startswith(root + "/"), r


@pytest.mark.parametrize("given, expected", [
    ("/tmp/talos-agentic-x", "//tmp/talos-agentic-x"),
    ("C:\\Users\\adi\\AppData\\Local\\Temp\\talos-agentic-x",
     "//c/Users/adi/AppData/Local/Temp/talos-agentic-x"),
    ("D:/work/wt", "//d/work/wt"),
])
def test_rule_root_uses_claude_codes_absolute_form(given, expected):
    # Claude Code writes an absolute rule path as `//path`, and matches Windows paths in their
    # POSIX form (C:\Users\a -> /c/Users/a).
    # mutation: leaving the backslashes or the `C:` drive in place yields a rule that never matches
    assert rule_root(given) == expected


def test_symlinked_worktree_gets_rules_for_both_paths(tmp_path):
    # macOS hands out /var/folders/... temp dirs that resolve to /private/var/...; the agent
    # may see either form, so a rule for only one of them fails on the other.
    # mutation: emitting rules for the given path alone fails the resolved-path assertion
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(real, target_is_directory=True)
    except OSError:
        pytest.skip("this account cannot create symlinks (Windows without developer mode)")
    allow = sandbox_settings(link)["permissions"]["allow"]
    assert f"Edit({rule_root(str(link))}/algorithm/**)" in allow
    assert f"Edit({rule_root(str(real.resolve()))}/algorithm/**)" in allow


def test_claude_is_pointed_at_the_worktree_settings_file(tmp_path):
    # mutation: passing the old .claude/settings.json path gives --settings a missing file, and
    # claude-cli then runs with no allow list at all
    seen = {}

    def run(cmd, **kw):
        path = Path(cmd[cmd.index("--settings") + 1])
        seen.update(path=path, cwd=Path(kw["cwd"]), exists=path.exists())
        raise subprocess.TimeoutExpired(cmd, 1)

    loop = types.SimpleNamespace(store=JobStore(tmp_path), propose_and_edit=None,
                                 _check_budget=lambda: None, _event=lambda *a, **k: None)
    attach_agentic(loop, "claude-cli", "m", timeout_s=1, run=run)
    with pytest.raises(AgenticError):
        loop.propose_and_edit(ctx())
    assert seen["exists"] and seen["path"] == seen["cwd"] / SETTINGS_REL
    shutil.rmtree(seen["cwd"], ignore_errors=True)


_NO_COST = object()


def _denied_run(denials, edit=False, cost=_NO_COST, returncode=0):
    """A fake claude: the JSON result only when asked for it, as the real CLI behaves. `cost`
    becomes the result's total_cost_usd; by default the result has none."""
    def run(cmd, **kw):
        wt = kw["cwd"]
        if edit:
            (wt / ".talos" / "hypothesis.json").write_text(json.dumps(
                {"title": "T", "description": "D", "strategy_tag": "local_search"}))
            (wt / "algorithm" / "mod.rs").write_text("fn a(){ 1 }")
        as_json = "json" in cmd and cmd[cmd.index("json") - 1] == "--output-format"
        result = {"type": "result", "subtype": "success", "is_error": False, "result": "done",
                  "permission_denials": denials}
        if cost is not _NO_COST:
            result["total_cost_usd"] = cost
        out = json.dumps(result) if as_json else "done"
        return subprocess.CompletedProcess(cmd, returncode, stdout=out, stderr="boom")
    return run


def _agentic_loop(tmp_path, run, kind="claude-cli"):
    """A loop stub that records what the agentic call charged and which events it raised."""
    charges, events = [], []
    loop = types.SimpleNamespace(store=JobStore(tmp_path), propose_and_edit=None,
                                 _check_budget=lambda: None,
                                 _event=lambda kind, **data: events.append((kind, data)),
                                 _charge_llm=charges.append, charges=charges, events=events)
    attach_agentic(loop, kind, "m", timeout_s=1, run=run)
    return loop


def test_a_claude_session_is_charged_its_cost_once(tmp_path):
    # issue 34: the session's total_cost_usd was never read, so --budget-usd could not cap
    # agentic mode. mutation: not charging leaves []; charging per retry or twice gives two
    loop = _agentic_loop(tmp_path, _denied_run([], edit=True, cost=1.25))
    loop.propose_and_edit(ctx())
    shutil.rmtree(loop._agentic_wt, ignore_errors=True)
    assert loop.charges == [1.25]
    assert not [k for k, _ in loop.events if k == "llm_cost_unknown"]


def test_a_free_claude_session_is_charged_zero_not_recorded_as_unknown(tmp_path):
    # mutation: `if cost:` treats a measured 0.0 as missing
    loop = _agentic_loop(tmp_path, _denied_run([], edit=True, cost=0))
    loop.propose_and_edit(ctx())
    shutil.rmtree(loop._agentic_wt, ignore_errors=True)
    assert loop.charges == [0.0]
    assert not [k for k, _ in loop.events if k == "llm_cost_unknown"]


def test_a_failed_claude_session_is_still_charged(tmp_path):
    # the tokens were spent whatever the exit code. mutation: checking the exit code before
    # reading the cost drops every failed session's spend
    loop = _agentic_loop(tmp_path, _denied_run([], cost=0.5, returncode=1))
    with pytest.raises(AgenticError) as ei:
        loop.propose_and_edit(ctx())
    shutil.rmtree(loop._agentic_wt, ignore_errors=True)
    assert loop.charges == [0.5]
    assert str(ei.value).startswith("claude exited 1: boom")


@pytest.mark.parametrize("cost", [_NO_COST, None, "1.0", True, -1.0, float("nan"),
                                  float("inf")])
def test_a_claude_session_without_a_usable_cost_is_recorded_as_unknown(tmp_path, cost):
    # mutation: charging a bool, a string, a negative or a non-finite number corrupts llm_usd;
    # skipping the event makes an uncounted session look like a free one
    loop = _agentic_loop(tmp_path, _denied_run([], edit=True, cost=cost))
    loop.propose_and_edit(ctx())
    shutil.rmtree(loop._agentic_wt, ignore_errors=True)
    assert loop.charges == []
    assert [k for k, _ in loop.events].count("llm_cost_unknown") == 1


def test_a_claude_transcript_that_is_not_json_is_recorded_as_unknown(tmp_path):
    loop = _agentic_loop(tmp_path, lambda cmd, **kw: subprocess.CompletedProcess(
        cmd, 0, stdout="plain prose", stderr=""))
    with pytest.raises(AgenticError):
        loop.propose_and_edit(ctx())
    shutil.rmtree(loop._agentic_wt, ignore_errors=True)
    assert loop.charges == [] and [k for k, _ in loop.events] == ["llm_cost_unknown"]


def test_the_cost_is_read_before_the_transcript_cap(tmp_path, monkeypatch):
    # a long session's JSON result outgrows the saved transcript. mutation: parsing the capped
    # agent_stdout.txt instead of the CLI's stdout loses the cost of exactly the costly sessions
    import talos.agentic as agentic
    monkeypatch.setattr(agentic, "_TRANSCRIPT_CAP", 10)
    loop = _agentic_loop(tmp_path, _denied_run([], edit=True, cost=2.0))
    loop.propose_and_edit(ctx())
    shutil.rmtree(loop._agentic_wt, ignore_errors=True)
    assert loop.charges == [2.0]


def test_a_timed_out_claude_session_is_recorded_as_unknown(tmp_path):
    # the json format prints its result only at the end, so a killed session's cost is lost.
    # mutation: dropping the event hides that the run's llm spend is short by a whole session
    def run(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 1)

    loop = _agentic_loop(tmp_path, run)
    with pytest.raises(AgenticError):
        loop.propose_and_edit(ctx())
    shutil.rmtree(loop._agentic_wt, ignore_errors=True)
    assert loop.charges == []
    [(kind, data)] = loop.events
    assert kind == "llm_cost_unknown" and "timed out after 1s" in data["reason"]


def test_a_missing_claude_binary_records_no_cost_event(tmp_path):
    # nothing ran, so nothing was spent. mutation: treating every AgenticError as a lost
    # session adds a spurious event
    def run(cmd, **kw):
        raise FileNotFoundError(cmd[0])

    loop = _agentic_loop(tmp_path, run)
    with pytest.raises(AgenticError):
        loop.propose_and_edit(ctx())
    shutil.rmtree(loop._agentic_wt, ignore_errors=True)
    assert loop.charges == [] and loop.events == []


def test_a_codex_session_charges_nothing_and_raises_no_event(tmp_path, monkeypatch):
    # codex reports no cost; the CLI marks the whole run "not tracked" instead
    # (cli.unpriced_reason). mutation: parsing codex's prose as claude JSON adds one
    # llm_cost_unknown event per iteration
    monkeypatch.setenv(CODEX_AGENTIC_ENV, "1")

    def run(cmd, **kw):
        wt = kw["cwd"]
        (wt / ".talos" / "hypothesis.json").write_text(json.dumps(
            {"title": "T", "description": "D", "strategy_tag": "local_search"}))
        (wt / "algorithm" / "mod.rs").write_text("fn a(){ 1 }")
        return subprocess.CompletedProcess(cmd, 0, stdout='{"total_cost_usd": 3.0}', stderr="")

    loop = _agentic_loop(tmp_path, run, kind="codex-cli")
    loop.propose_and_edit(ctx())
    shutil.rmtree(loop._agentic_wt, ignore_errors=True)
    assert loop.charges == [] and [k for k, _ in loop.events] == ["hypothesis"]


def test_failed_read_back_reports_the_tool_calls_the_cli_refused(tmp_path):
    # mutation: without this a session whose every Edit was denied is recorded as the agent
    # producing nothing, with no trace of the denial anywhere in the run's output
    edit = {"tool_name": "Edit", "tool_use_id": "t1",
            "tool_input": {"file_path": "/x/algorithm/mod.rs", "old_string": "a",
                           "new_string": "b"}}
    read = {"tool_name": "Read", "tool_use_id": "t3", "tool_input": {"file_path": "/etc/passwd"}}
    loop = _agentic_loop(tmp_path, _denied_run([edit, read, dict(edit, tool_use_id="t2")]))
    with pytest.raises(AgenticError) as ei:
        loop.propose_and_edit(ctx())
    shutil.rmtree(loop._agentic_wt, ignore_errors=True)
    assert str(ei.value) == ("agent did not fill in .talos/hypothesis.json; "
                             "the CLI refused 3 tool calls (Edit x2, Read x1)")


def test_failed_read_back_without_a_denial_says_nothing_about_one(tmp_path):
    # mutation: appending the note unconditionally blames the sandbox for an idle agent
    loop = _agentic_loop(tmp_path, _denied_run([]))
    with pytest.raises(AgenticError) as ei:
        loop.propose_and_edit(ctx())
    shutil.rmtree(loop._agentic_wt, ignore_errors=True)
    assert str(ei.value) == "agent did not fill in .talos/hypothesis.json"


def test_failed_read_back_survives_a_transcript_that_is_not_json(tmp_path, monkeypatch):
    # codex prints prose, and a JSON value that is not an object must not crash the loop either
    # mutation: letting JSONDecodeError or AttributeError escape kills the whole run
    monkeypatch.setenv(CODEX_AGENTIC_ENV, "1")
    for out in ("plain prose", "[1, 2]", '{"permission_denials": "nope"}',
                '{"permission_denials": [7]}'):
        loop = _agentic_loop(tmp_path, lambda cmd, out=out, **kw: subprocess.CompletedProcess(
            cmd, 0, stdout=out, stderr=""), kind="codex-cli")
        with pytest.raises(AgenticError) as ei:
            loop.propose_and_edit(ctx())
        shutil.rmtree(loop._agentic_wt, ignore_errors=True)
        assert str(ei.value) == "agent did not fill in .talos/hypothesis.json"


def test_a_denied_call_does_not_fail_an_iteration_that_produced_an_edit(tmp_path):
    # mutation: raising on any denial fails an agent that was refused one stray Read and then
    # did its work
    loop = _agentic_loop(tmp_path, _denied_run(
        [{"tool_name": "Read", "tool_use_id": "t", "tool_input": {"file_path": "/etc/passwd"}}],
        edit=True))
    hypothesis, files = loop.propose_and_edit(ctx())
    shutil.rmtree(loop._agentic_wt, ignore_errors=True)
    assert hypothesis["title"] == "T" and files["mod.rs"] == "fn a(){ 1 }"


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
    # mutation: without these, a Windows claude or codex cannot find its login under the user
    # profile, and Node cannot start at all without SYSTEMROOT
    windows = {"SYSTEMROOT": r"C:\Windows", "USERPROFILE": r"C:\Users\u",
               "APPDATA": r"C:\Users\u\AppData\Roaming",
               "LOCALAPPDATA": r"C:\Users\u\AppData\Local",
               "PATHEXT": ".COM;.EXE;.BAT;.CMD", "COMSPEC": r"C:\Windows\system32\cmd.exe",
               "TEMP": r"C:\Temp", "TMP": r"C:\Temp", "HOMEDRIVE": "C:", "HOMEPATH": r"\Users\u",
               "USERNAME": "u", "SYSTEMDRIVE": "C:", "WINDIR": r"C:\Windows"}
    for k, v in windows.items():
        monkeypatch.setenv(k, v)
    with pytest.raises(AgenticError):
        loop.propose_and_edit(ctx())
    assert {k: seen["env"].get(k) for k in windows} == windows
    shutil.rmtree(loop._agentic_wt, ignore_errors=True)


def test_run_agent_caps_transcript_length(tmp_path):
    from talos.agentic import _run_agent
    wt = prepare_worktree(ctx(), parent=tmp_path)
    huge = "x" * 250_000

    def run(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 0, stdout=huge, stderr=huge)

    _run_agent(["claude"], wt, "p", 10, run)
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


def test_agent_env_pins_the_workspace_to_the_worktree(tmp_path, monkeypatch):
    from talos.agentic import _agent_env
    monkeypatch.setenv("TALOS_WORKSPACE", "/elsewhere")
    assert _agent_env(tmp_path)["TALOS_WORKSPACE"] == str(tmp_path)


def test_agent_env_passes_the_backend_through(tmp_path, monkeypatch):
    from talos.agentic import _agent_env
    monkeypatch.setenv("TALOS_BACKEND", "c3")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    env = _agent_env(tmp_path)
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
        seen["prompt"] = kw["input"]
        raise subprocess.TimeoutExpired(cmd, 1)

    loop = types.SimpleNamespace(store=JobStore(tmp_path), propose_and_edit=None,
                                 _check_budget=lambda: None, _event=lambda *a, **k: None)
    attach_agentic(loop, "claude-cli", "m", timeout_s=1, run=run)
    failed = [{"title": "Dynamic greedy", "outcome": "failed:score", "mean_rel_delta": -0.0005,
               "worst_track": "n_items=5000,budget=10", "worst_rel_delta": -0.0037,
               "runtime_ratio": 14.07},
              {"title": "Old beam", "outcome": "failed:score", "against": 0}]
    with pytest.raises(AgenticError):
        loop.propose_and_edit(ctx(anchor=3, failed_hypotheses=failed))
    shutil.rmtree(loop._agentic_wt, ignore_errors=True)
    # mutation: joining titles alone drops the per-track delta and the runtime ratio
    assert "Dynamic greedy" in seen["prompt"] and "14.1x" in seen["prompt"]
    assert "n_items=5000,budget=10" in seen["prompt"]
    # issue #33. mutation: the agentic prompt keeping its own unlabelled list
    assert "- Old beam [failed:score, against the baseline]" in seen["prompt"]


def test_claude_md_shows_the_hyperparameters():
    from talos.agentic import claude_md
    text = claude_md(ctx(hyperparameters={"t": {"x": 1}}))
    # mutation: the agentic brief omitting the block while single-shot prompts carry it
    assert 'track t: {"x":1}' in text and "do not rename or remove" in text


def test_agent_env_passes_the_c3_api_key_through(tmp_path, monkeypatch):
    from talos.agentic import _agent_env
    monkeypatch.setenv("C3_API_KEY", "c3_key_1")
    monkeypatch.setenv("C3_AUTH_TOKEN", "raw-token")
    env = _agent_env(tmp_path)
    # mutation: dropping it from the allowlist fails every agentic `talos compile` on C3 for a
    # user with no `c3 login` session; the allowlist stays exact, so other C3_ vars stay out
    assert env["C3_API_KEY"] == "c3_key_1" and "C3_AUTH_TOKEN" not in env


def test_agent_env_passes_the_frozen_gpu_through(tmp_path, monkeypatch):
    from talos.agentic import _agent_env
    monkeypatch.setenv("TALOS_HARDWARE", "A100-80GB")
    # mutation: dropping TALOS_HARDWARE from the allowlist makes every sandbox `talos compile`
    # probe for a GPU of its own instead of using the job's
    assert _agent_env(tmp_path)["TALOS_HARDWARE"] == "A100-80GB"


def test_a_native_job_tells_its_agent_to_compile_natively():
    from dataclasses import replace
    c = ctx()
    metered = claude_md(c)
    native = claude_md(replace(c, scoring="native"))
    assert "`talos compile --challenge knapsack --dir algorithm`" in metered
    # mutation: the flag left out puts a native job's agent on the 287 s metered build
    assert "`talos compile --challenge knapsack --dir algorithm --native`" in native


def template_ctx(**kw):
    return PromptContext(challenge="energy_arbitrage", template_rs="TEMPLATE-RS", direction="go",
                         tacit="", files={"mod.rs": "fn empty(){}"}, baseline_name="b",
                         best_delta=float("-inf"), start="template",
                         challenge_docs={"README.md": "DOC-TEXT"}, **kw)


def test_template_worktree_carries_challenge_docs(tmp_path):
    # mutation: writing only template_rs to CHALLENGE.md leaves the agent with no problem
    # definition
    wt = prepare_worktree(template_ctx(), parent=tmp_path)
    text = (wt / "CHALLENGE.md").read_text()
    assert "DOC-TEXT" in text
    assert "tig-challenges/src/energy_arbitrage/README.md" in text
    assert "TEMPLATE-RS" in text


def test_template_claude_md_shows_the_gap_and_family_key():
    # mutation: keeping "Your current best is" beside the gap block prints "-inf%"
    c = template_ctx()
    md = claude_md(c)
    assert gap_block(c) in md
    assert "design_family" in md
    assert "Your current best is" not in md
    assert "-inf" not in md


def test_mainnet_challenge_md_is_unchanged(tmp_path):
    wt = prepare_worktree(ctx(), parent=tmp_path)
    assert (wt / "CHALLENGE.md").read_text() == (
        "# knapsack solver contract (template.rs)\n\n```rust\npub fn solve_challenge(\n```\n")


def _write_hypothesis(wt, body):
    (wt / ".talos" / "hypothesis.json").write_text(json.dumps(body))
    (wt / "algorithm" / "mod.rs").write_text("fn changed(){}")


def test_read_back_reads_design_family(tmp_path):
    # mutation: dropping design_family from read_back loses the family label
    c = template_ctx()
    wt = prepare_worktree(c, parent=tmp_path)
    _write_hypothesis(wt, {"title": "t", "description": "d", "strategy_tag": "hybrid",
                           "design_family": " greedy dispatch "})
    assert read_back(wt, c)[0]["design_family"] == "greedy dispatch"
    _write_hypothesis(wt, {"title": "t", "description": "d", "strategy_tag": "hybrid"})
    assert read_back(wt, c)[0]["design_family"] == "unlabelled"
    _write_hypothesis(wt, {"title": "t", "description": "d", "design_family": "x" * 200})
    assert read_back(wt, c)[0]["design_family"] == "x" * 80


def test_codex_refused_for_template_even_when_opted_in(monkeypatch):
    # mutation: reusing the env opt-in lets codex run agentic on a template start
    monkeypatch.setenv(CODEX_AGENTIC_ENV, "1")
    assert codex_agentic_refused("codex-cli", "agentic", "template") is True
    assert codex_agentic_refused("codex-cli", "agentic", "mainnet") is False
    assert codex_agentic_refused("claude-cli", "agentic", "template") is False


def test_attach_agentic_refuses_codex_for_a_template_job(monkeypatch):
    # mutation: attach_agentic not passing loop.spec.start
    monkeypatch.setenv(CODEX_AGENTIC_ENV, "1")
    loop = types.SimpleNamespace(spec=types.SimpleNamespace(start="template"))
    with pytest.raises(AgenticError, match="template"):
        attach_agentic(loop, "codex-cli", "m")


def test_claude_md_goal_sentence_closes_and_names_a_template_start():
    # mutation: dropping the ")" again, or keeping "improving a Rust solver" for a template start
    goal = ('baseline "b" on TIG\'s benchmark (higher verifier quality per nonce under a\n'
            'fixed fuel budget on every active track).')
    mainnet = claude_md(ctx())
    assert goal in mainnet
    assert 'You are improving a Rust solver for the TIG challenge "knapsack".' in mainnet
    template = claude_md(ctx(start="template"))
    assert goal in template and "improving a Rust solver" not in template
    assert ('You are writing a Rust solver from the empty template for the TIG challenge '
            '"knapsack".') in template
