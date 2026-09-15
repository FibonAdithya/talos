"""Scripted provider for tests and the fake end-to-end run."""
from __future__ import annotations

from talos.types import Completion, Usage


class FakeProvider:
    metered = True

    def __init__(self, script, usd_per_call: float = 0.01):
        self.name = "fake"
        self._script = script
        self._i = 0
        self.usd_per_call = usd_per_call
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> Completion:
        self.calls.append((system, user))
        if callable(self._script):
            text = self._script(system, user)
        else:
            text = self._script[min(self._i, len(self._script) - 1)]
            self._i += 1
        return Completion(text=text, usage=Usage(100, 50, cost_usd=self.usd_per_call))
