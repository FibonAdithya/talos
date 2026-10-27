"""`talos demo`: a scripted, in-process run that shows every outcome a real job can have."""
import json
import re
import socket
import urllib.request
from pathlib import Path

import pytest

from talos import cli


def _no_network(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("network")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


def _run_demo(tmp_path: Path, monkeypatch, delay: str = "0") -> tuple[int, Path]:
    monkeypatch.chdir(tmp_path)
    _no_network(monkeypatch)
    rc = cli.main(["demo", "--delay", delay])
    run_dir = next((tmp_path / "runs").glob("demo-*/job.json")).parent
    return rc, run_dir


def _events(run_dir: Path) -> list[dict]:
    lines = (run_dir / "timeline.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _kind(e: dict) -> str:
    return e["kind"]


def test_demo_shows_every_outcome_in_order(tmp_path, monkeypatch):
    # mutation: removing any one scripted step (the BROKEN edit, say) fails the order check
    rc, run_dir = _run_demo(tmp_path, monkeypatch)
    assert rc == 0
    assert not (tmp_path / "talos.config.json").exists()
    events = _events(run_dir)
    wanted = [
        lambda e: _kind(e) == "compile_failed",
        lambda e: (_kind(e) == "iteration_done" and e.get("outcome") == "failed:score"
                   and e.get("runs_since_improvement", 0) >= 2),
        lambda e: _kind(e) == "false_positive",
        lambda e: _kind(e) == "won",
    ]
    i = 0
    for e in events:
        if i < len(wanted) and wanted[i](e):
            i += 1
    assert i == len(wanted), f"matched {i} of {len(wanted)}: {[_kind(e) for e in events]}"
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "won"
    assert state["iteration"] < 6


def test_demo_prints_package_guide_and_next_step(tmp_path, monkeypatch, capsys):
    # mutation: dropping the guide, or a package file missing from PACKAGE_FILES, fails this
    from talos.demo import PACKAGE_FILES

    rc, run_dir = _run_demo(tmp_path, monkeypatch)
    out = capsys.readouterr().out
    assert rc == 0
    assert "What is in the package:" in out
    assert "scores.md  - " in out
    assert "talos report demo-" in out
    assert "talos setup" in out
    assert "(no description)" not in out
    # mutation: the fakes' default prices print $0.09 / $0.88 under a line that promises no spend
    assert "LLM spend: $0.00   Compute spend (estimated): $0.00" in out
    names = sorted(p.name for p in (run_dir / "package").iterdir())
    assert names and all(n in PACKAGE_FILES for n in names), names


def test_demo_delay_sleeps_between_events(tmp_path, monkeypatch):
    # mutation: ignoring --delay (never calling _event_sleep, or with another value) fails this
    calls: list[float] = []
    monkeypatch.setattr(cli, "_event_sleep", calls.append)
    slow, fast = tmp_path / "slow", tmp_path / "fast"
    slow.mkdir()
    fast.mkdir()
    rc, _ = _run_demo(slow, monkeypatch, "0.25")
    assert rc == 0
    assert calls and all(c == 0.25 for c in calls)
    calls.clear()
    rc, _ = _run_demo(fast, monkeypatch, "0")
    assert rc == 0
    assert calls == []


def test_demo_is_listed_in_help(capsys):
    # mutation: leaving demo out of build_parser, or hiding its help (argparse.SUPPRESS prints
    # "demo  ==SUPPRESS=="), fails this: the line must be the name followed by words
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    assert re.search(r"^\s+demo\s+\w", capsys.readouterr().out, re.M)


def test_describe_package_unknown_name(tmp_path):
    # mutation: dropping the fallback raises KeyError; unsorted output puts zzz.bin first
    from talos.demo import describe_package

    (tmp_path / "zzz.bin").write_bytes(b"")
    (tmp_path / "scores.md").write_text("x", encoding="utf-8")
    assert describe_package(tmp_path) == [
        "  scores.md  - training and held-out scores of the best candidate against the "
        "baseline",
        "  zzz.bin  - (no description)",
    ]


def test_demo_run_survives_a_cp1252_stdout(tmp_path, monkeypatch):
    # Issue #85. On Windows a piped stdout is cp1252, which has no ≈ (every status line) or ≤
    # (an LLM-written title); before the fix the job ended "failed" at its first status line.
    # mutation: dropping the UTF-8 reconfigure from main() raises inside on_event
    import io
    import sys

    from talos import demo
    _, description, new_k = demo._STEPS[1]
    monkeypatch.setitem(demo._STEPS, 1, ("keep w ≤ cap", description, new_k))
    buf = io.BytesIO()
    out = io.TextIOWrapper(buf, encoding="cp1252", newline="\n")
    monkeypatch.setattr(sys, "stdout", out)
    rc, run_dir = _run_demo(tmp_path, monkeypatch)
    out.flush()
    text = buf.getvalue().decode("utf-8")
    assert rc == 0
    assert json.loads((run_dir / "state.json").read_text(encoding="utf-8"))["status"] != "failed"
    assert "trying: keep w ≤ cap" in text and "compute ≈$" in text
