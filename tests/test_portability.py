"""Talos runs on the user's own machine, which may be Linux, macOS or Windows. These tests pin
the parts that differ by OS, so the Linux run of the suite catches them too."""
import ast
import json
import shutil
import subprocess
import types
from pathlib import Path

import pytest

from talos import executables
from talos.agentic import CODEX_AGENTIC_ENV, AgenticError, attach_agentic
from talos.providers.claude_cli import ClaudeCli
from talos.providers.codex_cli import CodexCli, list_codex_models
from talos.state import JobStore

ROOT = Path(__file__).resolve().parent.parent


def _mode(node: ast.Call) -> str:
    for kw in node.keywords:
        if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
            return kw.value.value
    for arg in node.args:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return arg.value
    return "r"


def _text_io_violations(source: str) -> list[tuple[int, str]]:
    """Text IO that relies on a platform default. Without `encoding`, Windows reads and writes
    cp1252, and the first non-ASCII character raises. Without `newline`, Windows writes CRLF, and
    a CRLF `job.sh` does not run in the Linux container."""
    out = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
        kws = {kw.arg for kw in node.keywords}
        if name == "read_text":
            need = {"encoding"}
        elif name == "write_text":
            need = {"encoding", "newline"}
        elif name in ("open", "fdopen"):
            mode = _mode(node)
            if "b" in mode:
                continue
            need = {"encoding", "newline"} if set(mode) & set("wax+") else {"encoding"}
        elif any(kw.arg == "text" and isinstance(kw.value, ast.Constant) and kw.value.value is True
                 for kw in node.keywords):
            need = {"encoding"}
        else:
            continue
        if need - kws:
            out.append((node.lineno, f"{name} without {', '.join(sorted(need - kws))}"))
    return out


def test_the_checker_flags_each_platform_default():
    # mutation: a checker that flags nothing passes the scan below for any code at all
    bad = ("p.read_text()\np.write_text(s, encoding='utf-8')\nopen(p, 'w', encoding='utf-8')\n"
           "p.open('a')\nos.fdopen(fd, 'w')\nrun(cmd, text=True)\n")
    assert [line for line, _ in _text_io_violations(bad)] == [1, 2, 3, 4, 5, 6]
    good = ("p.read_text(encoding='utf-8')\np.write_text(s, encoding='utf-8', newline='\\n')\n"
            "open(p, 'rb')\nrun(cmd, text=True, encoding='utf-8')\nopen(p, encoding='utf-8')\n")
    assert _text_io_violations(good) == []


def test_text_io_names_its_encoding_and_line_ending():
    # mutation: dropping encoding= or newline= from any call in the package fails this
    found = [f"{p.relative_to(ROOT).as_posix()}:{line}: {what}"
             for d in ("talos", "modal_app") for p in sorted((ROOT / d).rglob("*.py"))
             for line, what in _text_io_violations(p.read_text(encoding="utf-8"))]
    assert found == []


def test_argv0_resolves_through_path_only_on_windows(monkeypatch):
    monkeypatch.setattr(executables.shutil, "which", lambda name: rf"C:\npm\{name}.cmd")
    monkeypatch.setattr(executables, "_NT", True)
    # mutation: a bare name on Windows cannot start an npm-installed `codex.cmd`
    assert executables.argv0("codex") == r"C:\npm\codex.cmd"
    monkeypatch.setattr(executables, "_NT", False)
    # mutation: resolving on POSIX too makes argv[0] depend on what this machine has installed
    assert executables.argv0("codex") == "codex"


def test_argv0_keeps_the_bare_name_when_windows_cannot_find_it(monkeypatch):
    monkeypatch.setattr(executables.shutil, "which", lambda name: None)
    monkeypatch.setattr(executables, "_NT", True)
    # mutation: returning None makes subprocess raise TypeError, not the FileNotFoundError that
    # every call site turns into "CLI not found on PATH"
    assert executables.argv0("claude") == "claude"


@pytest.fixture
def windows_path(monkeypatch):
    monkeypatch.setattr(executables.shutil, "which", lambda name: rf"C:\bin\{name}.cmd")
    monkeypatch.setattr(executables, "_NT", True)


class _Done:
    returncode = 0
    stdout = json.dumps({"result": "", "models": []})
    stderr = ""


def test_providers_start_the_resolved_cli(windows_path):
    seen = []

    def run(cmd, **kw):
        seen.append(cmd[0])
        return _Done()

    ClaudeCli(model="m", run=run).complete("S", "U")
    CodexCli(model="m", run=run).complete("S", "U")
    list_codex_models(run=run)
    # mutation: any provider building argv from the bare name
    assert seen == [r"C:\bin\claude.cmd", r"C:\bin\codex.cmd", r"C:\bin\codex.cmd"]


@pytest.mark.parametrize("kind,exe", [("claude-cli", "claude"), ("codex-cli", "codex")])
def test_agentic_starts_the_resolved_cli_and_sends_the_prompt_on_stdin(tmp_path, monkeypatch,
                                                                       windows_path, kind, exe):
    monkeypatch.setenv(CODEX_AGENTIC_ENV, "1")
    seen = {}

    def run(cmd, **kw):
        seen.update(cmd=cmd, input=kw.get("input"))
        raise subprocess.TimeoutExpired(cmd, 1)

    from tests.test_agentic import ctx
    loop = types.SimpleNamespace(store=JobStore(tmp_path), propose_and_edit=None,
                                 _check_budget=lambda: None, _event=lambda *a, **k: None)
    attach_agentic(loop, kind, "m", timeout_s=1, run=run)
    with pytest.raises(AgenticError):
        loop.propose_and_edit(ctx())
    shutil.rmtree(loop._agentic_wt, ignore_errors=True)
    assert seen["cmd"][0] == rf"C:\bin\{exe}.cmd"
    # mutation: a multi-line prompt in argv is cut at the first newline by cmd.exe, which is what
    # runs an npm `.cmd` wrapper on Windows
    assert "Read CLAUDE.md" in seen["input"]
    assert not any("Read CLAUDE.md" in a for a in seen["cmd"])


def test_c3_calls_start_the_resolved_cli(tmp_path, windows_path):
    from talos import cli
    from talos.c3_bench import C3Bench

    def recorder(seen):
        def run(cmd, **kw):
            seen.append(cmd[0])
            r = _Done()
            r.stdout = '{"balance": 1}'
            return r
        return run

    setup, bench = [], []
    try:
        cli.check_c3(run=recorder(setup))
    except Exception:
        pass  # only argv[0] matters here, not how the stub output parses
    try:
        C3Bench(tmp_path, run=recorder(bench))._c3("whoami")
    except Exception:
        pass
    # mutation: either c3 call site building argv from the bare name
    assert setup and set(setup) == {r"C:\bin\c3.cmd"}
    assert bench == [r"C:\bin\c3.cmd"]
