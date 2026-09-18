import json
import types
from pathlib import Path

import pytest

from talos.c3_bench import C3CommandError
from talos.c3_transport import CliTransport, make_transport


def runner(script):
    """script: (cmd_word) -> (returncode, stdout). Records calls."""
    calls = []

    def run(cmd, **kw):
        calls.append((cmd, kw))
        rc, out = script(cmd[1], kw)
        return types.SimpleNamespace(returncode=rc, stdout=out, stderr="" if rc == 0 else out)
    return run, calls


def test_cli_transport_deploy_returns_the_job_id_from_the_job_dir(tmp_path):
    out = "Warning: experimental\n" + json.dumps({"id": "job_9"})
    run, calls = runner(lambda word, kw: (0, out))
    assert CliTransport(run=run).deploy(tmp_path) == "job_9"
    # mutation: deploying outside the job dir uploads the wrong workspace
    assert calls[0][1]["cwd"] == str(tmp_path)
    assert calls[0][0][:2] == ["c3", "deploy"]


def test_cli_transport_status_picks_this_job_and_raises_when_absent():
    rows = [{"job_id": "job_1", "status": "running"}, {"job_id": "job_2", "status": "PENDING"}]
    run, _ = runner(lambda word, kw: (0, json.dumps(rows)))
    # mutation: returning the first row reports another job's status; not upper-casing breaks
    # the ACTIVE/TERMINAL comparisons in C3Bench._wait
    assert CliTransport(run=run).status("job_1") == "RUNNING"
    with pytest.raises(C3CommandError):
        CliTransport(run=run).status("job_absent")


def test_cli_transport_fetch_copies_the_named_artifact_and_reports_absence(tmp_path):
    def script(word, kw):
        d = Path(kw["cwd"]) / "job_1" / "artifacts"
        d.mkdir(parents=True, exist_ok=True)
        (d / "results.json").write_text('{"compile": {}}', encoding="utf-8", newline="\n")
        return 0, json.dumps({"jobs": [{"job_id": "job_1", "directory": str(d.parent)}]})

    run, calls = runner(script)
    t = CliTransport(run=run)
    dest = tmp_path / "out" / "results.json"
    assert t.fetch("job_1", "results.json", dest) is True
    assert json.loads(dest.read_text(encoding="utf-8")) == {"compile": {}}
    # mutation: returning True for a file the job never wrote makes _collect read a stale path
    assert t.fetch("job_1", "build.log", tmp_path / "out" / "build.log") is False
    # mutation: one pull per file downloads every job twice
    assert [c[0][1] for c in calls].count("pull") == 1


def test_cli_transport_fetch_pulls_in_place_when_dest_is_where_c3_pull_writes(tmp_path):
    def script(word, kw):
        d = Path(kw["cwd"]) / "job_1" / "artifacts"
        d.mkdir(parents=True, exist_ok=True)
        (d / "results.json").write_text("{}", encoding="utf-8", newline="\n")
        return 0, json.dumps({"jobs": [{"job_id": "job_1", "directory": str(d.parent)}]})

    run, calls = runner(script)
    dest = tmp_path / "jobdir" / "job_1" / "artifacts" / "results.json"
    assert CliTransport(run=run).fetch("job_1", "results.json", dest) is True
    # mutation: pulling in dest.parent nests a second job_1/artifacts/ under the first
    assert calls[0][1]["cwd"] == str(tmp_path / "jobdir") and dest.read_text() == "{}"
    assert not (dest.parent / "job_1").exists()


def test_cli_transport_balance_parses_the_printed_amount_and_survives_garbage():
    run, _ = runner(lambda word, kw: (0, "Credit balance: £12.34 (free tier)"))
    assert CliTransport(run=run).balance_gbp() == 12.34
    run, _ = runner(lambda word, kw: (0, "no balance here"))
    # mutation: returning 0.0 invents a number the CLI never printed
    assert CliTransport(run=run).balance_gbp() is None


def test_cli_transport_missing_binary_is_a_c3commanderror():
    def run(cmd, **kw):
        raise FileNotFoundError(2, "No such file or directory: 'c3'")

    with pytest.raises(C3CommandError):
        CliTransport(run=run).whoami()


@pytest.mark.xfail(reason="McpTransport lands in Task 4", strict=True, raises=ImportError)
def test_make_transport_uses_the_cli_without_a_key_and_mcp_with_one():
    from talos.c3_mcp import McpTransport
    assert isinstance(make_transport(None), CliTransport)
    # mutation: ignoring the key keeps the CLI requirement on Windows
    assert isinstance(make_transport("c3_key_" + "a" * 20), McpTransport)
