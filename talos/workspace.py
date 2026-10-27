"""Where Talos keeps its state: talos.config.json, .talos/ and runs/ (issue #70)."""
from __future__ import annotations

import os
from pathlib import Path

from talos.config import ConfigError

ENV_VAR = "TALOS_WORKSPACE"
MARKER = "talos.config.json"


class WorkspaceError(ConfigError):
    """TALOS_WORKSPACE names something that is not a directory."""


def default_root() -> Path:
    return Path.home() / ".talos" / "workspace"


def _walk(start: Path | None) -> Path:
    here = (start or Path.cwd()).resolve()
    for d in (here, *here.parents):
        if (d / MARKER).is_file():
            return d
    # Called at call time, so a test can replace it.
    return default_root()


def find_root(start: Path | None = None) -> Path:
    """TALOS_WORKSPACE if it is a directory, else the nearest directory at or above `start`
    (default: the current directory) that holds talos.config.json, else the default workspace.
    Only the config file marks a workspace: the agentic worktree has a .talos/ of its own.
    Never creates anything."""
    value = os.environ.get(ENV_VAR)
    if value:
        p = Path(value).expanduser()
        if not p.is_dir():
            raise WorkspaceError(f"{ENV_VAR}={value} is not a directory")
        return p.resolve()
    return _walk(start)


def find_root_for_write(start: Path | None = None) -> Path:
    """For the commands that create the workspace: a TALOS_WORKSPACE that does not exist yet is
    accepted, not refused. The caller makes the directory."""
    value = os.environ.get(ENV_VAR)
    if value:
        return Path(value).expanduser().resolve()
    return _walk(start)
