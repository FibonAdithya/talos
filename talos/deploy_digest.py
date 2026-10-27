"""Names the Modal marker function after the source the app was deployed from, so a client can
tell a stale deploy from a current one. C3 and local copy their container modules into every
job; Modal runs whatever was deployed last."""
from __future__ import annotations

import hashlib
from pathlib import Path

# The import closure of modal_app/talos_bench.py, relative to the parent of the `talos` package
# (the repo root in a clone, site-packages in a wheel). tests/test_deploy_digest.py walks the
# imports and fails when this drifts. scripts/check_wheel.py reads it with `ast`, so it stays a
# literal tuple of string literals.
DEPLOYED_FILES = ("modal_app/__init__.py", "modal_app/talos_bench.py", "talos/__init__.py",
                  "talos/challenges.py", "talos/deploy_digest.py", "talos/inside.py",
                  "talos/native_runner.py")


def digest(root: Path | None = None) -> str:
    """12 hex digits over every deployed file: its path, a NUL, then its bytes with CRLF turned
    into LF (a Windows checkout with autocrlf must match a deploy made on Linux)."""
    root = root if root is not None else Path(__file__).resolve().parent.parent
    h = hashlib.sha256()
    for rel in sorted(DEPLOYED_FILES):
        h.update(rel.encode() + b"\0" + (root / rel).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:12]


def marker_name(root: Path | None = None) -> str:
    return f"deployed_{digest(root)}"
