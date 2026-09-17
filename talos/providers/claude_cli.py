"""Headless `claude -p` using the user's Claude subscription login. No key is stored."""
from __future__ import annotations

import json
import subprocess

from talos.executables import argv0
from talos.providers import ProviderAuthError, ProviderError
from talos.types import Completion, Usage


class ClaudeCli:
    metered = False

    def __init__(self, model: str, run=subprocess.run, timeout_s: int = 1800):
        self.name = "claude-cli"
        self.model = model
        self._run = run
        self.timeout_s = timeout_s

    def complete(self, system: str, user: str) -> Completion:
        cmd = [argv0("claude"), "-p", "--output-format", "json", "--model", self.model,
               "--system-prompt", system]
        try:
            r = self._run(cmd, input=user, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=self.timeout_s)
        except FileNotFoundError:
            raise ProviderAuthError("claude CLI not found on PATH; install Claude Code") from None
        except subprocess.TimeoutExpired:
            raise ProviderError("claude CLI timed out") from None
        if r.returncode != 0:
            err = (r.stderr or r.stdout or "")[-500:]
            if "login" in err.lower() or "auth" in err.lower():
                raise ProviderAuthError(f"claude CLI not logged in: {err}")
            raise ProviderError(f"claude CLI failed: {err}")
        try:
            data = json.loads(r.stdout)
        except json.JSONDecodeError:
            return Completion(text=r.stdout, usage=Usage())
        u = data.get("usage") or {}
        usage = Usage(int(u.get("input_tokens", 0)), int(u.get("output_tokens", 0)),
                      cost_usd=data.get("total_cost_usd"))
        return Completion(text=data.get("result", ""), usage=usage)
