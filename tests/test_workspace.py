"""Workspace resolution (issue #70): where talos finds talos.config.json, .talos/ and runs/."""
import json
from pathlib import Path

import pytest

from talos import cli, workspace
from talos.bench import EvalResult
from talos.types import CompileResult
from talos.workspace import WorkspaceError, find_root

from tests.test_cli import _RefusingBench, scripted


def _config(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / "talos.config.json").write_text(
        json.dumps({"provider": "fake", "model": "fake", "backend": "local"}), encoding="utf-8")
    return d


def _home(monkeypatch, tmp_path) -> Path:
    home = tmp_path / "home-ws"
    monkeypatch.setattr(workspace, "default_root", lambda: home)
    return home


def test_find_root_walks_up_to_the_config(tmp_path, monkeypatch):
    _home(monkeypatch, tmp_path)
    ws = _config(tmp_path / "ws")
    start = ws / "a" / "b"
    start.mkdir(parents=True)
    assert find_root(start) == ws.resolve()


def test_find_root_nearest_config_wins(tmp_path, monkeypatch):
    _home(monkeypatch, tmp_path)
    _config(tmp_path / "ws")
    inner = _config(tmp_path / "ws" / "a")
    start = inner / "b"
    start.mkdir()
    assert find_root(start) == inner.resolve()


def test_find_root_ignores_a_dot_talos_dir(tmp_path, monkeypatch):
    home = _home(monkeypatch, tmp_path)
    if any((p / "talos.config.json").exists() for p in tmp_path.resolve().parents):
        pytest.skip("a parent of tmp_path holds a talos.config.json")
    ws = tmp_path / "ws"
    (ws / ".talos").mkdir(parents=True)
    assert find_root(ws) == home


def test_find_root_falls_back_to_default_root(tmp_path, monkeypatch):
    home = _home(monkeypatch, tmp_path)
    if any((p / "talos.config.json").exists() for p in tmp_path.resolve().parents):
        pytest.skip("a parent of tmp_path holds a talos.config.json")
    start = tmp_path / "x"
    start.mkdir()
    got = find_root(start)
    assert got == home and got != start


def test_find_root_env_wins_over_walk(tmp_path, monkeypatch):
    _home(monkeypatch, tmp_path)
    e = tmp_path / "e"
    e.mkdir()
    ws = _config(tmp_path / "ws")
    monkeypatch.setenv("TALOS_WORKSPACE", str(e))
    assert find_root(ws) == e.resolve()


def test_find_root_rejects_a_missing_workspace(tmp_path, monkeypatch, capsys):
    _home(monkeypatch, tmp_path)
    missing = tmp_path / "missing"
    monkeypatch.setenv("TALOS_WORKSPACE", str(missing))
    with pytest.raises(WorkspaceError, match="TALOS_WORKSPACE="):
        find_root(tmp_path)
    assert cli.main(["status"]) == 2
    assert f"TALOS_WORKSPACE={missing} is not a directory" in capsys.readouterr().err


def test_workspace_flag_overrides_cwd(tmp_path, monkeypatch, capsys):
    a = tmp_path / "a"
    a.mkdir()
    monkeypatch.chdir(a)
    w = tmp_path / "w"
    assert cli.main(["--workspace", str(w), "demo", "--delay", "0"]) == 0
    assert (w / "runs").is_dir() and not (a / "runs").exists()


def test_demo_falls_back_to_default_workspace(tmp_path, monkeypatch, capsys):
    home = _home(monkeypatch, tmp_path)
    x = tmp_path / "x"
    x.mkdir()
    monkeypatch.chdir(x)
    if any((p / "talos.config.json").exists() for p in x.resolve().parents):
        pytest.skip("a parent of tmp_path holds a talos.config.json")
    assert cli.main(["demo", "--delay", "0"]) == 0
    assert len(list((home / "runs").iterdir())) == 1 and not (x / "runs").exists()
    assert str(home / "runs") in capsys.readouterr().out


def test_status_from_a_subdirectory_lists_the_workspace_jobs(tmp_path, monkeypatch, capsys):
    _home(monkeypatch, tmp_path)
    ws = _config(tmp_path / "ws")
    monkeypatch.chdir(ws)
    assert cli.main(["demo", "--delay", "0"]) == 0
    (ws / "sub" / "deeper").mkdir(parents=True)
    monkeypatch.chdir(ws / "sub" / "deeper")
    capsys.readouterr()
    assert cli.main(["status", "--json"]) == 0
    jobs = json.loads(capsys.readouterr().out)["jobs"]
    assert [j["job_id"] for j in jobs] == [p.name for p in (ws / "runs").iterdir()]
    assert len(jobs) == 1


def test_setup_from_a_subdirectory_updates_the_workspace_config(tmp_path, monkeypatch, capsys):
    _home(monkeypatch, tmp_path)
    ws = _config(tmp_path / "ws")
    (ws / "sub").mkdir()
    monkeypatch.chdir(ws / "sub")
    monkeypatch.setattr(cli, "validate_provider", lambda p: None)
    monkeypatch.setattr(cli, "deploy_bench", lambda *a, **k: None)
    rc = cli.main(["setup"], ask=scripted(["", "anthropic", "", "sk-test", "ak-1", "as-1"]))
    assert rc == 0
    assert json.loads((ws / "talos.config.json").read_text())["provider"] == "anthropic"
    assert not (ws / "sub" / "talos.config.json").exists()
    assert f"Workspace: {ws.resolve()}" in capsys.readouterr().out


def test_compile_dir_stays_relative_to_cwd(tmp_path, monkeypatch):
    from talos.local_transport import DockerInfo
    _home(monkeypatch, tmp_path)
    ws = _config(tmp_path / "ws")
    (ws / "sub" / "algorithm").mkdir(parents=True)
    (ws / "sub" / "algorithm" / "mod.rs").write_text("fn marker_in_sub(){}")
    assert not (ws / "algorithm").exists()
    monkeypatch.chdir(ws / "sub")
    monkeypatch.setattr(cli, "prepare", lambda ch, **k: None)
    monkeypatch.setattr(cli, "docker_info", lambda: DockerInfo(["runc"], 4, 8))

    class T:
        def remove_run(self, run_dir):
            pass
    monkeypatch.setattr(cli, "DockerTransport", T)
    seen = {}

    class B(_RefusingBench):
        def evaluate(self, request):
            seen["files"] = request.files
            return EvalResult(CompileResult(ok=True, artifact_id="a", output="ok"), [], None,
                              "not_won")

    def make(backend, run_dir, *a, **k):
        seen["run_dir"] = run_dir
        return B("x")
    monkeypatch.setattr(cli, "make_bench", make)
    assert cli.main(["compile", "--challenge", "knapsack", "--dir", "algorithm"]) == 0
    assert seen["files"] == {"mod.rs": "fn marker_in_sub(){}"}
    assert seen["run_dir"] == ws.resolve() / ".talos" / "compile"


def test_write_commands_refuse_a_workspace_that_is_a_file(tmp_path, monkeypatch, capsys):
    _home(monkeypatch, tmp_path)
    f = tmp_path / "f"
    f.write_text("x", encoding="utf-8")
    monkeypatch.setenv("TALOS_WORKSPACE", str(f))
    assert cli.main(["demo", "--delay", "0"]) == 2
    assert f"TALOS_WORKSPACE={f} is not a directory" in capsys.readouterr().err
    assert f.is_file()
