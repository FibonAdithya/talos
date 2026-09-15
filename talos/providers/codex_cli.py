"""Headless `codex exec` using the user's ChatGPT/Codex login. The system prompt is folded
into the prompt because codex exec has no system flag."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

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
            cmd = ["codex", "exec", "-m", self.model, "--skip-git-repo-check", "-o", str(out), prompt]
            try:
                r = self._run(cmd, capture_output=True, text=True, timeout=self.timeout_s, cwd=td)
            except FileNotFoundError:
                raise ProviderAuthError("codex CLI not found on PATH") from None
            except subprocess.TimeoutExpired:
                raise ProviderError("codex CLI timed out") from None
            if r.returncode != 0:
                err = (r.stderr or r.stdout or "")[-500:]
                if "login" in err.lower() or "auth" in err.lower():
                    raise ProviderAuthError(f"codex CLI not logged in: {err}")
                raise ProviderError(f"codex CLI failed: {err}")
            text = out.read_text() if out.exists() else r.stdout
        return Completion(text=text, usage=Usage())
