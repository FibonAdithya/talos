"""Bench client. ModalBench talks to the deployed `talos-bench` app; FakeBench drives the
loop in tests. Both return the same shapes."""
from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from typing import Callable, Protocol

from talos.challenges import CHALLENGES, BeatRule
from talos.diagnostics import dead_new_functions
from talos.inside import NONCE_TIMEOUT_S
from talos.scoring import holdout_decision
from talos.types import CompileResult, NonceResult, NonceSet

# Rough Modal list prices, $/second, used only for budget accounting (marked estimated).
CPU_USD_PER_CORE_SECOND = 0.0000131
MEM_USD_PER_GIB_SECOND = 0.00000222
GPU_USD_PER_SECOND = {"L40S": 0.000542}

HOLDOUT_REASONS = ("won", "not_won", "forced", "not_compiled", "timeout", "dead_code")

_HASH_RE = re.compile(r"[0-9a-f]{64}")


def _redact(text: str) -> str:
    """A job's rand_hash must never reach a log line or a timeline event. Modal exceptions
    quote the failing argv, so every message built from one goes through here first."""
    return _HASH_RE.sub("<hash>", text)


class BenchUnavailable(Exception):
    pass


def timeout_for(timeouts: dict[str, int] | None, track: str) -> int:
    return NONCE_TIMEOUT_S if timeouts is None else timeouts.get(track, NONCE_TIMEOUT_S)


def dead_code(request: "EvalRequest", compile_output: str) -> list[str]:
    if request.prior_functions is None:
        return []
    return dead_new_functions(compile_output, request.prior_functions)


@dataclass
class EvalRequest:
    challenge: str
    files: dict[str, str]
    training: list[NonceSet]
    holdout: list[NonceSet]
    fuel: int
    baseline_training: list[NonceResult] | None  # None = score held-out unconditionally
    rule: BeatRule
    # `fn` names per file in the code the candidate was edited from. A candidate that compiles
    # with a never-used function absent from here is not scored (holdout_reason "dead_code"):
    # its change is off the solve path and scoring it repeats the prior result. None = skip
    # the check (the baseline measurement, `talos compile`).
    prior_functions: dict[str, list[str]] | None = None
    # Per-track cap on one nonce's runtime, in seconds. None = the flat NONCE_TIMEOUT_S the
    # baseline ran under. The loop derives it from the baseline's measured runtime so a
    # candidate many times slower fails fast instead of holding the job for its full length.
    timeouts: dict[str, int] | None = None


@dataclass
class EvalResult:
    compile: CompileResult
    training: list[NonceResult]          # empty when compile.ok is False or dead code stopped it
    holdout: list[NonceResult] | None    # None when not scored
    holdout_reason: str

    def __post_init__(self) -> None:
        if self.holdout_reason not in HOLDOUT_REASONS:
            raise ValueError(f"unknown holdout reason {self.holdout_reason!r}")


class BenchCancelled(Exception):
    """The user asked for a stop while a job was in flight; the job has been cancelled."""


class PendingJobStore:
    """Where a backend records the job it has in flight, so a resumed run can reattach. The
    loop hands the bench closures over JobState.pending_job; `memory()` is for callers without
    a job, such as `talos compile`."""

    def __init__(self, get: Callable[[], dict | None], set: Callable[[dict | None], None]):
        self.get, self.set = get, set

    @classmethod
    def memory(cls) -> "PendingJobStore":
        box: dict = {"v": None}
        return cls(lambda: box["v"], lambda d: box.__setitem__("v", d))


class Bench(Protocol):
    def evaluate(self, request: EvalRequest) -> EvalResult: ...
    def request_stop(self) -> None: ...
    def cost_mark(self) -> float: ...
    def cost_usd_since(self, mark: float) -> float: ...


def _seconds_cost(challenge: str, seconds: float) -> float:
    spec = CHALLENGES[challenge]
    if spec.is_gpu:
        return seconds * GPU_USD_PER_SECOND[spec.gpu]
    return seconds * (spec.cpu * CPU_USD_PER_CORE_SECOND
                      + spec.memory_mib / 1024 * MEM_USD_PER_GIB_SECOND)


class ModalBench:
    def __init__(self, app_name: str = "talos-bench", retry_window_s: int = 900,
                 clock: Callable[[], float] = time.time,
                 sleep: Callable[[float], None] = time.sleep):
        self.app_name = app_name
        self.retry_window_s = retry_window_s
        self._clock = clock
        self._sleep = sleep
        self._cost = 0.0
        self._fns: dict[str, object] = {}

    def _fn(self, name: str):
        """Only a genuine "not deployed" becomes BenchUnavailable. Anything else — a network
        blip during hydrate — propagates so _with_retry treats it as an outage and retries,
        instead of telling the user to run `talos setup`."""
        cached = self._fns.get(name)
        if cached is not None:
            return cached
        import modal
        try:
            fn = modal.Function.from_name(self.app_name, name)
            fn.hydrate()  # from_name is lazy; hydrate forces the lookup so "not deployed" fails here
            self._fns[name] = fn
            return fn
        except modal.exception.NotFoundError as e:
            raise BenchUnavailable(
                f"Modal function {name} not found in app {self.app_name}: "
                f"{type(e).__name__}: {_redact(str(e))[:300]}. "
                f"Run `talos setup` to deploy.") from None

    def _with_retry(self, call: Callable[[], object]):
        """The window is anchored at the FIRST failure, not at call start: a starmap over 64
        nonces or a cold image pull can itself take longer than retry_window_s, and anchoring
        at call start would give that call zero retries."""
        deadline = None
        delay = 5.0
        while True:
            try:
                return call()
            except BenchUnavailable:
                raise
            except Exception as e:  # noqa: BLE001 - Modal raises many transport types
                if deadline is None:
                    deadline = self._clock() + self.retry_window_s
                if self._clock() + delay > deadline:
                    raise BenchUnavailable(
                        f"Modal unreachable for {self.retry_window_s}s: "
                        f"{type(e).__name__}: {_redact(str(e))[:300]}") from None
                self._sleep(delay)
                delay = min(delay * 2, 60)

    def _compile(self, challenge: str, files: dict[str, str]) -> CompileResult:
        t0 = self._clock()
        out = self._with_retry(lambda: self._fn(f"compile_{challenge}").remote(files))
        self._cost += _seconds_cost(challenge, self._clock() - t0)
        return CompileResult(ok=out["ok"], artifact_id=out.get("artifact_id"), output=out["output"])

    def _score(self, challenge: str, artifact_id: str, nonce_sets: list[NonceSet],
              fuel: int, timeouts: dict[str, int] | None) -> list[NonceResult]:
        args = [(artifact_id, ns.track, ns.rand_hash, n, fuel, timeout_for(timeouts, ns.track))
                for ns in nonce_sets for n in ns.nonces()]
        rows = self._with_retry(
            lambda: list(self._fn(f"score_nonce_{challenge}").starmap(args)))
        results = [NonceResult.from_dict(r) for r in rows]
        self._cost += sum(_seconds_cost(challenge, r.runtime_ms / 1000) for r in results)
        return results

    def evaluate(self, request: EvalRequest) -> EvalResult:
        c = self._compile(request.challenge, request.files)
        if not c.ok:
            return EvalResult(c, [], None, "not_compiled")
        if dead_code(request, c.output):
            return EvalResult(c, [], None, "dead_code")
        tr = (self._score(request.challenge, c.artifact_id, request.training, request.fuel,
                          request.timeouts)
              if request.training else [])
        go, reason = holdout_decision(request.baseline_training, tr, request.rule)
        ho = None
        if go:
            ho = (self._score(request.challenge, c.artifact_id, request.holdout, request.fuel,
                              request.timeouts)
                  if request.holdout else [])
        return EvalResult(c, tr, ho, reason)

    def request_stop(self) -> None:
        pass  # Modal calls are short; the loop's own stop check lands between them

    def cost_mark(self) -> float:
        return self._cost

    def cost_usd_since(self, mark: float) -> float:
        return self._cost - mark


class FakeBench:
    """scores(challenge, files, nonce_set) -> list of quality per nonce, None = error."""

    def __init__(self, scores: Callable[[str, dict[str, str], NonceSet], list[int | None]],
                 compile_ok: Callable[[dict[str, str]], bool] = lambda files: True,
                 usd_per_nonce: float = 0.01,
                 compile_output: Callable[[dict[str, str]], str] = lambda files: "ok"):
        self._scores = scores
        self._compile_ok = compile_ok
        self._compile_output = compile_output
        self._runtime_ms = 1
        self._usd_per_nonce = usd_per_nonce
        self._cost = 0.0
        self.calls: list[EvalRequest] = []
        self.holdout_runs = 0

    def evaluate(self, request: EvalRequest) -> EvalResult:
        self.calls.append(request)
        if not self._compile_ok(request.files):
            return EvalResult(CompileResult(ok=False, artifact_id=None,
                                            output="error[E0308]: mismatched types"),
                              [], None, "not_compiled")
        art = hashlib.sha256(repr(sorted(request.files.items())).encode()).hexdigest()[:16]
        comp = CompileResult(ok=True, artifact_id=art, output=self._compile_output(request.files))
        if dead_code(request, comp.output):
            return EvalResult(comp, [], None, "dead_code")
        tr = self._score(request.challenge, request.files, request.training)
        go, reason = holdout_decision(request.baseline_training, tr, request.rule)
        ho = None
        if go:
            self.holdout_runs += 1
            ho = self._score(request.challenge, request.files, request.holdout)
        return EvalResult(comp, tr, ho, reason)

    def _score(self, challenge, files, nonce_sets) -> list[NonceResult]:
        out: list[NonceResult] = []
        for ns in nonce_sets:
            qs = self._scores(challenge, files, ns)
            if len(qs) != ns.count:
                raise ValueError(f"scores callback returned {len(qs)} qualities "
                                 f"for {ns.count} nonces on track {ns.track}")
            for n, q in zip(ns.nonces(), qs):
                self._cost += self._usd_per_nonce
                if q is None:
                    out.append(NonceResult(ns.track, n, False, None, self._runtime_ms,
                                           "no_solution"))
                else:
                    out.append(NonceResult(ns.track, n, True, q, self._runtime_ms, None))
        return out

    def request_stop(self) -> None:
        pass

    def cost_mark(self) -> float:
        return self._cost

    def cost_usd_since(self, mark: float) -> float:
        return self._cost - mark
