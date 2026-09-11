"""Bench client. ModalBench talks to the deployed `talos-bench` app; FakeBench drives the
loop in tests. Both return the same shapes."""
from __future__ import annotations

import hashlib
import time
from typing import Callable, Protocol

from talos.challenges import CHALLENGES
from talos.types import CompileResult, NonceResult, NonceSet

# Rough Modal list prices, $/second, used only for budget accounting (marked estimated).
CPU_USD_PER_CORE_SECOND = 0.0000131
MEM_USD_PER_GIB_SECOND = 0.00000222
GPU_USD_PER_SECOND = {"L40S": 0.000542}


class BenchUnavailable(Exception):
    pass


class Bench(Protocol):
    def compile(self, challenge: str, files: dict[str, str]) -> CompileResult: ...
    def score(self, challenge: str, artifact_id: str, nonce_sets: list[NonceSet],
              fuel: int) -> list[NonceResult]: ...
    def cost_mark(self) -> float: ...
    def cost_usd_since(self, mark: float) -> float: ...


def _seconds_cost(challenge: str, seconds: float) -> float:
    spec = CHALLENGES[challenge]
    if spec.is_gpu:
        return seconds * GPU_USD_PER_SECOND[spec.gpu]
    return seconds * (spec.cpu * CPU_USD_PER_CORE_SECOND
                      + spec.memory_mib / 1024 * MEM_USD_PER_GIB_SECOND)


class ModalBench:
    def __init__(self, app_name: str = "talos-bench", retry_window_s: int = 900):
        self.app_name = app_name
        self.retry_window_s = retry_window_s
        self._cost = 0.0

    def _fn(self, name: str):
        import modal
        try:
            fn = modal.Function.from_name(self.app_name, name)
            fn.hydrate()  # from_name is lazy; hydrate forces the lookup so "not deployed" fails here
            return fn
        except Exception as e:  # noqa: BLE001 - any lookup failure means not deployed
            raise BenchUnavailable(f"Modal function {name} not found in app {self.app_name}: "
                                   f"{e}. Run `talos setup` to deploy.") from None

    def _with_retry(self, call: Callable[[], object]):
        deadline = time.time() + self.retry_window_s
        delay = 5.0
        while True:
            try:
                return call()
            except BenchUnavailable:
                raise
            except Exception as e:  # noqa: BLE001 - Modal raises many transport types
                if time.time() + delay > deadline:
                    raise BenchUnavailable(f"Modal unreachable for {self.retry_window_s}s: {e}")
                time.sleep(delay)
                delay = min(delay * 2, 60)

    def compile(self, challenge: str, files: dict[str, str]) -> CompileResult:
        fn = self._fn(f"compile_{challenge}")
        t0 = time.time()
        out = self._with_retry(lambda: fn.remote(files))
        self._cost += _seconds_cost(challenge, time.time() - t0)
        return CompileResult(ok=out["ok"], artifact_id=out.get("artifact_id"), output=out["output"])

    def score(self, challenge: str, artifact_id: str, nonce_sets: list[NonceSet],
              fuel: int) -> list[NonceResult]:
        fn = self._fn(f"score_nonce_{challenge}")
        args = [(artifact_id, ns.track, ns.rand_hash, n, fuel) for ns in nonce_sets
                for n in ns.nonces()]
        rows = self._with_retry(lambda: list(fn.starmap(args)))
        results = [NonceResult.from_dict(r) for r in rows]
        self._cost += sum(_seconds_cost(challenge, r.runtime_ms / 1000) for r in results)
        return results

    def cost_mark(self) -> float:
        return self._cost

    def cost_usd_since(self, mark: float) -> float:
        return self._cost - mark


class FakeBench:
    """scores(challenge, files, nonce_set) -> list of quality per nonce, None = error."""

    def __init__(self, scores: Callable[[str, dict[str, str], NonceSet], list[int | None]],
                 compile_ok: Callable[[dict[str, str]], bool] = lambda files: True,
                 usd_per_nonce: float = 0.01):
        self._scores = scores
        self._compile_ok = compile_ok
        self._usd_per_nonce = usd_per_nonce
        self._files: dict[str, dict[str, str]] = {}
        self._cost = 0.0
        self.compile_calls = 0
        self.score_calls = 0

    def compile(self, challenge: str, files: dict[str, str]) -> CompileResult:
        self.compile_calls += 1
        if not self._compile_ok(files):
            return CompileResult(ok=False, artifact_id=None, output="error[E0308]: mismatched types")
        art = hashlib.sha256(repr(sorted(files.items())).encode()).hexdigest()[:16]
        self._files[art] = dict(files)
        return CompileResult(ok=True, artifact_id=art, output="ok")

    def score(self, challenge: str, artifact_id: str, nonce_sets: list[NonceSet],
              fuel: int) -> list[NonceResult]:
        self.score_calls += 1
        out: list[NonceResult] = []
        for ns in nonce_sets:
            qs = self._scores(challenge, self._files[artifact_id], ns)
            for n, q in zip(ns.nonces(), qs):
                self._cost += self._usd_per_nonce
                if q is None:
                    out.append(NonceResult(ns.track, n, False, None, 1, "no_solution"))
                else:
                    out.append(NonceResult(ns.track, n, True, q, 1, None))
        return out

    def cost_mark(self) -> float:
        return self._cost

    def cost_usd_since(self, mark: float) -> float:
        return self._cost - mark
