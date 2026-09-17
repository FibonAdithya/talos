"""How Talos names the external CLIs it starts (claude, codex, c3)."""
import os
import shutil

_NT = os.name == "nt"


def argv0(name: str) -> str:
    """On POSIX, `subprocess` finds a bare name on PATH, and a bare argv[0] keeps commands the same
    on every machine. On Windows it looks for `name.exe` alone, so an npm-installed `codex.cmd`
    is never found; `shutil.which` honours PATHEXT. A name it cannot find stays bare, so the call
    still raises the FileNotFoundError each call site reports as "not on PATH"."""
    return (shutil.which(name) or name) if _NT else name
