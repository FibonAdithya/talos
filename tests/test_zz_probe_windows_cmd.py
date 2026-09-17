"""THROWAWAY PROBE, never merged: what a Windows .cmd wrapper does to argv and stdin."""
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(os.name != "nt", reason="probe of cmd.exe")
def test_probe(tmp_path):
    dump = tmp_path / "dump.py"
    dump.write_text("import sys, json\nprint(json.dumps({'argv': sys.argv[1:], 'stdin': sys.stdin.read()}))\n")
    # the shape npm writes: "%_prog%" "%dp0%\...\cli.js" %*
    shim = tmp_path / "tool.cmd"
    shim.write_text(f'@ECHO off\r\n"{sys.executable}" "{dump}" %*\r\n')
    arg = "line one\nline two & more \"quoted\" 100% ^caret"
    r = subprocess.run([str(shim), "-p", "--system-prompt", arg, "--model", "m"], input="in1\nin2 & é",
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    print("PROBE multiline-arg rc", r.returncode, "stdout", repr(r.stdout), "stderr", repr(r.stderr[-300:]))
    big = "x" * 9000
    r = subprocess.run([str(shim), big], capture_output=True, text=True, encoding="utf-8", errors="replace")
    print("PROBE 9000-char-arg rc", r.returncode, "stdout len", len(r.stdout), "stderr", repr(r.stderr[-300:]))
    assert False, "probe: read the printed output"
