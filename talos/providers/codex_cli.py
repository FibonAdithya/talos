"""Headless `codex exec` using the user's ChatGPT/Codex login. The system prompt is folded
into the prompt because codex exec has no system flag."""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from talos.executables import argv0
from talos.providers import ProviderAuthError, ProviderError
from talos.types import Completion, Usage


class CodexCli:
    metered = False

    def __init__(self, model: str, run=subprocess.run, timeout_s: int = 1800):
        self.name = "codex-cli"
        self.model = model
        self._run = run
        self.timeout_s = timeout_s

    def complete(self, system: str, user: str) -> Completion:
        prompt = f"{system}\n\n---\n\n{user}"
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "last.md"
            # "-" makes codex read the prompt from stdin. As an argv element the prompt would hit
            # the kernel's 128 KiB per-argument cap: a challenge's algorithm files alone exceed it.
            cmd = [argv0("codex"), "exec", "-m", self.model, "--skip-git-repo-check",
                   "-o", str(out), "-"]
            try:
                r = self._run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=self.timeout_s, cwd=td)
            except FileNotFoundError:
                raise ProviderAuthError("codex CLI not found on PATH") from None
            except subprocess.TimeoutExpired:
                raise ProviderError("codex CLI timed out") from None
            if r.returncode != 0:
                err = (r.stderr or r.stdout or "")[-500:]
                if "login" in err.lower() or "auth" in err.lower():
                    raise ProviderAuthError(f"codex CLI not logged in: {err}")
                raise ProviderError(f"codex CLI failed: {err}")
            text = out.read_text(encoding="utf-8") if out.exists() else r.stdout
        return Completion(text=text, usage=Usage())


def list_codex_models(run=subprocess.run, timeout_s: int = 60) -> list[str]:
    """Slugs the installed codex CLI will accept, best first. Empty when the catalog cannot be
    read (no binary, non-zero exit, unparseable output): setup then falls back to the static
    default rather than failing."""
    try:
        r = run([argv0("codex"), "debug", "models"], capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=timeout_s)
        if r.returncode != 0:
            return []
        models = json.loads(r.stdout)["models"]
        visible = [m for m in models if m.get("visibility") == "list" and m.get("slug")]
    except (OSError, subprocess.TimeoutExpired, ValueError, KeyError, TypeError, AttributeError):
        return []
    return [m["slug"] for m in sorted(visible, key=lambda m: m.get("priority", 0))]
