"""Keep every test off the real workspace and the caller's TALOS_WORKSPACE."""
from pathlib import Path

import pytest

import talos.workspace


@pytest.fixture(autouse=True)
def _isolated_workspace(monkeypatch):
    # Set-then-delete, not a bare delenv(raising=False): on an absent variable that records
    # nothing, so the value main() writes for --workspace would leak into later tests.
    monkeypatch.setenv("TALOS_WORKSPACE", "")
    monkeypatch.delenv("TALOS_WORKSPACE")
    # The existing chdir-based tests mean the current directory when they say "the workspace".
    monkeypatch.setattr(talos.workspace, "default_root", lambda: Path.cwd())
