"""Bench client. ModalBench talks to the deployed `talos-bench` app; FakeBench drives the
loop in tests. Both return the same shapes."""
from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from typing import Callable, Protocol

from talos.challenges import CHALLENGES, BeatRule, hardware_options, gpu_slug, modal_workers
from talos.diagnostics import dead_new_functions
from talos.inside import NONCE_TIMEOUT_S
from talos.scoring import holdout_decision
from talos.types import CompileResult, NonceResult, NonceSet

# Rough Modal list prices, $/second, used only for budget accounting (marked estimated).
CPU_USD_PER_CORE_SECOND = 0.0000131
MEM_USD_PER_GIB_SECOND = 0.00000222
# MEASURED from modal.com/pricing on 2026-09-24; one entry per GPU in MODAL_GPUS.
GPU_USD_PER_SECOND = {"L40S": 0.000542, "A100-80GB": 0.000694, "H100": 0.001097}

HOLDOUT_REASONS = ("won", "not_won", "forced", "not_compiled", "timeout", "dead_code")

_HASH_RE = re.compile(r"[0-9a-f]{64}")
_C3_KEY_RE = re.compile(r"c3_key_[A-Za-z0-9_-]+")


def _redact(text: str) -> str:
    """A job's rand_hash must never reach a log line or a timeline event. Modal exceptions
    quote the failing argv, so every message built from one goes through here first.
    A C3 API key must not reach one either."""
    return _HASH_RE.sub("<hash>", _C3_KEY_RE.sub("<c3-key>", text))


class BenchUnavailable(Exception):
    pass


def _stale_deploy(e: Exception) -> bool:
    """A remote function refusing its arguments is a deploy older than this client, never a
    transport outage. Modal re-raises the container's TypeError with its message intact."""
    return "positional argument" in str(e) or "unexpected keyword argument" in str(e)


def _timeout_types() -> tuple[type, ...]:
    """What FunctionCall.get raises when the client-side wait runs out. modal 1.5.5 raises the
    builtin TimeoutError; its own modal.exception.TimeoutError is caught too in case a later
    SDK switches (a fake `modal` module in tests may lack it)."""
    import modal
    own = getattr(getattr(modal, "exception", None), "TimeoutError", None)
    return (TimeoutError, own) if isinstance(own, type) else (TimeoutError,)


def timeout_for(timeouts: dict[str, int] | None, track: str) -> int:
    return NONCE_TIMEOUT_S if timeouts is None else timeouts.get(track, NONCE_TIMEOUT_S)


def hyperparameters_for(hyperparameters: dict[str, dict | None] | None,
                        track: str) -> dict | None:
    return None if hyperparameters is None else hyperparameters.get(track)


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
    # Per-track mainnet hyperparameters, passed to tig-runtime on every nonce of that track. A
    # track mapped to None, or absent, runs without the flag. None = no track gets any. The loop
    # and the baseline both take it from JobSpec.hyperparameters, never per request.
    hyperparameters: dict[str, dict | None] | None = None


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
    def select_hardware(self, challenge: str, chosen: str | None = None) -> str | None: ...
    def request_stop(self) -> None: ...
    def cost_mark(self) -> float: ...
    def cost_usd_since(self, mark: float) -> float: ...


def _seconds_cost(challenge: str, seconds: float, gpu: str | None) -> float:
    spec = CHALLENGES[challenge]
    if spec.is_gpu:
        return seconds * GPU_USD_PER_SECOND[gpu]
    return seconds * (spec.cpu * CPU_USD_PER_CORE_SECOND
                      + spec.memory_mib / 1024 * MEM_USD_PER_GIB_SECOND)


class ModalBench:
    def __init__(self, app_name: str = "talos-bench", retry_window_s: int = 900,
                 clock: Callable[[], float] = time.time,
                 sleep: Callable[[float], None] = time.sleep, probe_window_s: int = 120):
        self.app_name = app_name
        self.retry_window_s = retry_window_s
        self.probe_window_s = probe_window_s
        self._clock = clock
        self._sleep = sleep
        self._cost = 0.0
        self._fns: dict[str, object] = {}
        self.gpu: str | None = None  # set by select_hardware; None for CPU challenges

    def select_hardware(self, challenge: str, chosen: str | None = None) -> str | None:
        """Freezes the GPU this bench calls for `challenge`. With `chosen` (a resumed job, or
        the sandbox's `talos compile` given TALOS_HARDWARE) nothing is probed. Otherwise each GPU in
        preference order gets a probe call; the first whose probe returns within probe_window_s
        is the choice, a probe that has not started by then is cancelled (it would otherwise
        run, and bill, whenever that GPU freed up) and the next GPU is tried. None answering is
        BenchUnavailable, which pauses the run. CPU challenges have no options: None."""
        options = hardware_options(CHALLENGES[challenge])
        if not options:
            self.gpu = None
            return None
        if chosen is not None:
            if chosen not in options:
                raise ValueError(f"unknown GPU {chosen!r} for {challenge}; one of "
                                 f"{', '.join(options)}")
            self.gpu = chosen
            return chosen
        for gpu in options:
            if self._with_retry(lambda: self._probe(gpu)):
                self.gpu = gpu
                return gpu
        raise BenchUnavailable(f"no Modal capacity for any of {', '.join(options)} within "
                               f"{self.probe_window_s}s each; try again later")

    def _probe(self, gpu: str) -> bool:
        t0 = self._clock()
        call = self._fn(f"probe_{gpu_slug(gpu)}").spawn()
        try:
            call.get(timeout=self.probe_window_s)
        except _timeout_types():
            call.cancel()
            return False  # never started: nothing to charge
        # ESTIMATE: the whole wait at the GPU's rate. Queue time is in it, so this is never
        # below what the container billed.
        self._cost += GPU_USD_PER_SECOND[gpu] * (self._clock() - t0)
        return True

    def _name(self, kind: str, challenge: str) -> str:
        """The deployed function for `kind` ("compile" or "score_batch") of `challenge`: the
        bare name for a CPU challenge, the frozen GPU's variant for a GPU one."""
        if not CHALLENGES[challenge].is_gpu:
            return f"{kind}_{challenge}"
        if self.gpu is None:
            raise ValueError(f"{challenge} is a GPU challenge; select_hardware was not called")
        return f"{kind}_{challenge}_{gpu_slug(self.gpu)}"

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
                if _stale_deploy(e):
                    raise BenchUnavailable(
                        f"the deployed {self.app_name} app predates this client "
                        f"({type(e).__name__}: {_redact(str(e))[:200]}). "
                        f"Run `talos setup` to redeploy.") from None
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
        name = self._name("compile", challenge)  # outside the retry: a missing GPU is a bug
        out = self._with_retry(lambda: self._fn(name).remote(files))
        self._cost += _seconds_cost(challenge, self._clock() - t0, self.gpu)
        return CompileResult(ok=out["ok"], artifact_id=out.get("artifact_id"), output=out["output"])

    def _score(self, challenge: str, artifact_id: str, nonce_sets: list[NonceSet],
              fuel: int, timeouts: dict[str, int] | None,
              hyperparameters: dict[str, dict | None] | None) -> list[NonceResult]:
        """One `score_batch` call per `modal_workers` nonces, across tracks: a container scores
        its batch on every core at once, and a batch never holds more nonces than the
        container has workers, so its wall time is one nonce's, not a queue's."""
        tasks = [{"track": ns.track, "rand_hash": ns.rand_hash, "nonce": n, "fuel": fuel,
                  "timeout_s": timeout_for(timeouts, ns.track),
                  "hyperparameters": hyperparameters_for(hyperparameters, ns.track)}
                 for ns in nonce_sets for n in ns.nonces()]
        if not tasks:
            # A holdout count of 0 still yields one NonceSet per track, each with no nonces.
            # modal 1.5.5 never returns from starmap([]), so the client would hang here.
            return []
        size = modal_workers(CHALLENGES[challenge])
        args = [(artifact_id, tasks[i:i + size]) for i in range(0, len(tasks), size)]
        name = self._name("score_batch", challenge)
        batches = self._with_retry(lambda: list(self._fn(name).starmap(args)))
        # The container's wall seconds are what Modal bills: verifier time and the batch's
        # slowest nonce included, which one nonce's runtime_ms never was.
        self._cost += sum(_seconds_cost(challenge, b["seconds"], self.gpu) for b in batches)
        return [NonceResult.from_dict(r) for b in batches for r in b["rows"]]

    def evaluate(self, request: EvalRequest) -> EvalResult:
        c = self._compile(request.challenge, request.files)
        if not c.ok:
            return EvalResult(c, [], None, "not_compiled")
        if dead_code(request, c.output):
            return EvalResult(c, [], None, "dead_code")
        tr = (self._score(request.challenge, c.artifact_id, request.training, request.fuel,
                          request.timeouts, request.hyperparameters)
              if request.training else [])
        go, reason = holdout_decision(request.baseline_training, tr, request.rule)
        ho = None
        if go:
            ho = (self._score(request.challenge, c.artifact_id, request.holdout, request.fuel,
                              request.timeouts, request.hyperparameters)
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

    def select_hardware(self, challenge: str, chosen: str | None = None) -> str | None:
        """In-process, so nothing is probed; but a GPU challenge still gets a GPU (the first
        option, or the frozen one), because the hardware class the fake run keys its baseline
        under needs one just as a real run's does."""
        options = hardware_options(CHALLENGES[challenge])
        return (chosen or options[0]) if options else None

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
