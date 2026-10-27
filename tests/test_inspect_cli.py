"""`talos status`, `talos show` and `talos diff`: read-only views of runs/ (issue #68)."""
import json

import pytest

from talos import cli
from tests.run_fixture import make_run


def _runs(tmp_path):
    """J: finished, best #2 (confirmed). N: no candidate. P: in progress, a pending job."""
    runs = tmp_path / "runs"
    make_run(runs / "J", [{"iteration": 1}, {"iteration": 2, "against": 1}], best=2,
             confirmed=[2], status="won", iteration_files={1: {"mod.rs": "line a\nline b\n"}},
             timeline=[{"kind": "x"}, {"kind": "won", "iteration": 2,
                       "holdout": {"mean_rel_delta": 0.04, "worst_rel_delta": 0.01}}])
    make_run(runs / "N", [{"iteration": 1}], timeline=[{"kind": "x"}])
    make_run(runs / "P", [{"iteration": 1}], status="researching",
             pending_job={"purpose": 2, "hypothesis": {"title": "try X"}, "files": {}},
             timeline=[{"kind": "hypothesis", "iteration": 2, "title": "try X"}])
    return runs


def test_status_table_and_json(tmp_path, monkeypatch, capsys):
    _runs(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert cli.main(["status"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("JOB") and [ln.split()[0] for ln in lines[1:]] == ["J", "N", "P"]
    assert cli.main(["status", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["schema_version"] == 1 and [j["job_id"] for j in out["jobs"]] == ["J", "N", "P"]


def test_status_with_no_jobs(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert cli.main(["status"]) == 0 and "no jobs under runs/" in capsys.readouterr().out
    assert cli.main(["status", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {"schema_version": 1, "jobs": []}


def test_status_reports_an_unreadable_job_on_stderr(tmp_path, monkeypatch, capsys):
    # mutation: dropping the stderr line leaves the user with "unreadable" and no reason
    runs = _runs(tmp_path)
    (runs / "N" / "state.json").write_text("{")
    monkeypatch.chdir(tmp_path)
    assert cli.main(["status"]) == 0
    cap = capsys.readouterr()
    assert "unreadable" in cap.out and "N: job N: cannot read state.json" in cap.err


def test_status_job_detail(tmp_path, monkeypatch, capsys):
    _runs(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert cli.main(["status", "P", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["job_id"] == "P" and d["activity"]["pending"] == {"purpose": 2, "title": "try X"}
    assert d["budget"]["iterations"] == {"cap": 10, "spent": 1, "left": 9}
    assert cli.main(["status", "J"]) == 0
    assert "best: #2" in capsys.readouterr().out


def test_status_unknown_job(tmp_path, monkeypatch, capsys):
    _runs(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert cli.main(["status", "nope"]) == 1
    cap = capsys.readouterr()
    assert cap.out == "" and cap.err.strip() == "no job nope under runs/"


def test_show_and_diff_default_to_best(tmp_path, monkeypatch, capsys):
    # mutation: defaulting --iteration to the last iteration, or --against to parent
    _runs(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert cli.main(["show", "J", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["iteration"] == 2
    assert cli.main(["show", "J"]) == 0
    assert "iteration: 2" in capsys.readouterr().out
    assert cli.main(["diff", "J"]) == 0
    assert capsys.readouterr().out.startswith("--- baseline/mod.rs")
    assert cli.main(["diff", "J", "--against", "parent"]) == 0
    assert capsys.readouterr().out.startswith("--- #1/mod.rs")
    assert cli.main(["show", "J", "--iteration", "1", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["iteration"] == 1


def test_json_error_leaves_stdout_empty(tmp_path, monkeypatch, capsys):
    # mutation: printing the error (or a JSON error object) to stdout
    make_run(tmp_path / "runs" / "J", [{"iteration": 1}])
    monkeypatch.chdir(tmp_path)
    assert cli.main(["show", "J", "--json"]) == 1
    cap = capsys.readouterr()
    assert cap.out == "" and "has no best candidate" in cap.err
    assert cli.main(["diff", "J"]) == 1
    cap = capsys.readouterr()
    assert cap.out == "" and "has no best candidate" in cap.err


@pytest.mark.parametrize("bad", ["0", "-1", "abc", "1.5", "²"])  # "²".isdigit() is True
def test_iteration_flag_rejects_bad_values(bad, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    for cmd in ("show", "diff"):
        with pytest.raises(SystemExit) as e:
            cli.main([cmd, "J", "--iteration", bad])
        assert e.value.code == 2
    assert "expected a positive iteration number or 'best'" in capsys.readouterr().err


def test_parse_which():
    assert cli.parse_which("best") == "best"
    assert cli.parse_which("7") == 7 and cli.parse_which("07") == 7


def test_inspect_commands_write_nothing(tmp_path, monkeypatch):
    # mutation: any of the commands constructing JobStore, or writing report.html
    _runs(tmp_path)
    monkeypatch.chdir(tmp_path)
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    mtimes = {p: p.stat().st_mtime_ns for p in tmp_path.rglob("*") if p.is_file()}
    for argv in (["status"], ["status", "--json"], ["status", "J"], ["status", "nope"],
                 ["show", "J"], ["show", "J", "--iteration", "1", "--json"], ["diff", "J"],
                 ["diff", "J", "--against", "parent"], ["show", "N"], ["status", "P", "--json"],
                 ["show", "nope"], ["diff", "nope"]):
        cli.main(argv)
    assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*")) == before
    assert all(p.stat().st_mtime_ns == m for p, m in mtimes.items())


def test_inspect_commands_need_no_config_or_network(tmp_path, monkeypatch, capsys):
    # mutation: a cmd_* calling config.load() or make_bench, or reaching the network
    _runs(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "load", lambda *a, **k: pytest.fail("config loaded"))
    monkeypatch.setattr(cli, "make_bench", lambda *a, **k: pytest.fail("bench built"))
    monkeypatch.setattr(cli.urllib.request, "urlopen", lambda *a, **k: pytest.fail("network"))
    for argv in (["status"], ["status", "J", "--json"], ["show", "J"], ["diff", "J"]):
        assert cli.main(argv) == 0, argv
