"""Headless `claude -p` using the user's Claude subscription login. No key is stored."""
from __future__ import annotations

import json
import math
import subprocess
import tempfile
from pathlib import Path

from talos.executables import argv0
from talos.providers import ProviderAuthError, ProviderError, ProviderTimeout
from talos.types import Completion, Usage


NO_COST = "claude printed no total_cost_usd"


def session_cost(stdout: str) -> float | None:
    """total_cost_usd from `claude -p --output-format json`: the CLI's own estimate of the call
    or session at API list prices. Single-shot calls (ClaudeCli.complete) and agentic sessions
    (talos/agentic.py) both charge it. None when stdout is not that result or the figure is not
    a finite, non-negative number. The agentic path reads it from the CLI's full stdout, never
    the transcript file, which _TRANSCRIPT_CAP may have cut."""
    try:
        cost = json.loads(stdout)["total_cost_usd"]
    except (ValueError, KeyError, TypeError, IndexError):
        return None
    if isinstance(cost, bool) or not isinstance(cost, (int, float)):
        return None
    if not math.isfinite(cost) or cost < 0:
        return None
    return float(cost)


class ClaudeCli:
    metered = False

    def __init__(self, model: str, run=subprocess.run, timeout_s: int = 1800):
        self.name = "claude-cli"
        self.model = model
        self._run = run
        self.timeout_s = timeout_s

    def complete(self, system: str, user: str) -> Completion:
        # The system prompt goes in a file, not argv. On Windows an npm-installed `claude.cmd`
        # runs under cmd.exe, which cuts an argument at its first newline, drops every argument
        # after it, and still exits 0; it also refuses a command line over 8191 characters.
        with tempfile.TemporaryDirectory() as td:
            sp = Path(td) / "system.md"
            sp.write_text(system, encoding="utf-8", newline="\n")
            cmd = [argv0("claude"), "-p", "--output-format", "json", "--model", self.model,
                   "--system-prompt-file", str(sp)]
            try:
                r = self._run(cmd, input=user, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=self.timeout_s)
            except FileNotFoundError:
                raise ProviderAuthError("claude CLI not found on PATH; install Claude Code") from None
            except subprocess.TimeoutExpired:
                # The json format prints its result only at the end, so the cost is lost.
                msg = f"claude CLI timed out after {self.timeout_s}s"
                raise ProviderTimeout(msg, cost_unknown=msg) from None
        # Read before the exit code: a call that fails can still have cost money.
        cost = session_cost(r.stdout or "")
        unknown = NO_COST if cost is None else None
        if r.returncode != 0:
            err = (r.stderr or r.stdout or "")[-500:]
            if "login" in err.lower() or "auth" in err.lower():
                raise ProviderAuthError(f"claude CLI not logged in: {err}", cost_usd=cost,
                                        cost_unknown=unknown)
            raise ProviderError(f"claude CLI failed: {err}", cost_usd=cost, cost_unknown=unknown)
        try:
            data = json.loads(r.stdout)
        except json.JSONDecodeError:
            return Completion(text=r.stdout, usage=Usage(), cost_unknown=unknown)
        u = data.get("usage") or {}
        usage = Usage(int(u.get("input_tokens", 0)), int(u.get("output_tokens", 0)),
                      cost_usd=cost)
        return Completion(text=data.get("result", ""), usage=usage, cost_unknown=unknown)
