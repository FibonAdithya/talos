"""Keep every test off the real workspace, the caller's TALOS_WORKSPACE and the caller's PATH."""
from pathlib import Path
from types import SimpleNamespace

import pytest

import talos.executables
import talos.workspace


@pytest.fixture(autouse=True)
def _isolated_workspace(monkeypatch):
    # Set-then-delete, not a bare delenv(raising=False): on an absent variable that records
    # nothing, so the value main() writes for --workspace would leak into later tests.
    monkeypatch.setenv("TALOS_WORKSPACE", "")
    monkeypatch.delenv("TALOS_WORKSPACE")
    # The existing chdir-based tests mean the current directory when they say "the workspace".
    monkeypatch.setattr(talos.workspace, "default_root", lambda: Path.cwd())


@pytest.fixture(autouse=True)
def _cli_lookup_finds_nothing(request, monkeypatch):
    # On Windows argv0 resolves through shutil.which, so a machine with claude, codex or c3
    # installed puts a full path in argv[0] where the subprocess-mocking tests expect the bare
    # name. Resolve as CI's windows-latest does, with none installed. Only argv0's lookup is
    # stubbed (shutil.which itself stays real), and live tests start the real CLIs.
    if request.node.get_closest_marker("live") is None:
        monkeypatch.setattr(talos.executables, "shutil", SimpleNamespace(which=lambda name: None))
