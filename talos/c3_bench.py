"""C3 (cthree.cloud) bench backend: one evaluate call is one batch job, driven through the
`c3` CLI. Every subprocess call goes through the injected runner."""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from talos.bench import (BenchCancelled, BenchUnavailable, EvalRequest, EvalResult,
                         PendingJobStore, _redact)
from talos.c3_jobdir import LocalSettings, request_hash, write_job_dir
from talos.challenges import CHALLENGES, c3_profile
from talos.inside import NONCE_TIMEOUT_S
from talos.types import CompileResult, NonceResult, NonceSet

if TYPE_CHECKING:
    from talos.c3_transport import C3Transport

GBP_PER_HOUR = {"cpu-d3-4vcpu-16gb": 0.11, "l40": 1.49}  # ESTIMATE: `c3 list`, 2026-09-15
USD_PER_GBP = 1.35                                          # ESTIMATE
ACTIVE = ("PENDING", "SCHEDULING", "RUNNING")
QUEUED = ("PENDING", "SCHEDULING")
DONE = ("SUCCEEDED", "COMPLETED", "SYNCED")
TERMINAL = DONE + ("FAILED", "CANCELED", "CANCELLED", "TIMED_OUT")


class C3CommandError(RuntimeError):
    pass


class _JobFailed(RuntimeError):
    pass


_JOB_ID_RE = re.compile(r"[A-Za-z0-9._-]+")


def _safe_job_id(job_id: str) -> str:
    """A job id becomes a path component (job_dir/job_id/artifacts). C3 returns it, so a
    compromised or buggy server could hand back "../../x" and walk the local filesystem;
    reject anything that is not a plain relative name. Does not echo the id."""
    if job_id in (".", "..") or not _JOB_ID_RE.fullmatch(job_id):
        raise BenchUnavailable("C3 returned an unusable job id")
    return job_id


def parse_json_stdout(text: str):
    """`c3 deploy --json` prints a Warning line before the JSON on experimental hardware."""
    for i, ch in enumerate(text):
        if ch in "{[":
            return json.loads(text[i:])
    raise ValueError("no JSON in c3 output")


def c3_env(api_key: str | None) -> dict[str, str] | None:
    """The environment for a `c3` call. With a key, C3_API_KEY on top of this process's
    environment; the CLI reads it there, and argv would show it in `ps`. Without one, None:
    the child inherits the environment and uses the `c3 login` session."""
    return {**os.environ, "C3_API_KEY": api_key} if api_key else None


def fill_timeouts(rows: list[NonceResult], nonce_sets: list[NonceSet]) -> list[NonceResult]:
    have = {(r.track, r.nonce): r for r in rows}
    out = []
    for ns in nonce_sets:
        for n in ns.nonces():
            out.append(have.get((ns.track, n)) or
                       NonceResult(ns.track, n, False, None, NONCE_TIMEOUT_S * 1000, "timeout"))
    return out


class C3Bench:
    def __init__(self, run_dir: Path, pending: PendingJobStore | None = None,
                 run: Callable = subprocess.run, clock: Callable[[], float] = time.time,
                 sleep: Callable[[float], None] = time.sleep, poll_s: float = 20.0,
                 pending_timeout_s: int = 1800, poll_failures_max: int = 15,
                 api_key: str | None = None, transport: "C3Transport | None" = None,
                 local: LocalSettings | None = None, usd_per_hour: float | None = None):
        self.run_dir = Path(run_dir)
        self._pending = pending or PendingJobStore.memory()
        self._clock, self._sleep = clock, sleep
        self.poll_s, self.pending_timeout_s = poll_s, pending_timeout_s
        self.poll_failures_max = poll_failures_max
        self._cost = 0.0
        self._stop = False
        from talos.c3_transport import CliTransport, make_transport
        # An injected `run` (run is not subprocess.run) always means "drive the CLI" — that is
        # how every test stays off the network and the real `c3` binary. Only the default `run`
        # lets a configured key select MCP instead.
        self._t = transport or (CliTransport(run=run, api_key=api_key) if run is not subprocess.run
                                else make_transport(api_key, run=run))
        self.local = local
        self.subdir = "local" if local is not None else "c3"
        self.usd_per_hour = usd_per_hour
        # Error messages name the backend the user chose. Transports without a label are C3's.
        self._label = getattr(self._t, "label", "C3")

    def _wait(self, job_id: str, profile: str) -> tuple[str, float | None, float]:
        """Returns (status, first_running_time, terminal_time). Poll failures — a CLI error, an
        unparseable document, or a status outside ACTIVE/TERMINAL — are tolerated up to
        poll_failures_max in a row; a stop request cancels the job."""
        # The pending timeout restarts here on every reattach: the original submission time
        # is not stored in the pending record, only the job id and request hash.
        submitted = self._clock()
        first_running = None
        failures = 0
        while True:
            if self._stop:
                self._t.cancel(job_id)
                self._pending.set(None)
                raise BenchCancelled(job_id)
            try:
                status = self._t.status(job_id)
                if status not in ACTIVE and status not in TERMINAL:
                    raise C3CommandError(f"unrecognised job status {status!r}")
            except (C3CommandError, ValueError, KeyError, AttributeError, TypeError) as e:
                # a CLI error, a document shape we do not understand, or a status neither
                # ACTIVE nor TERMINAL: none of these are safe to poll on for ever
                failures += 1
                if failures >= self.poll_failures_max:
                    raise BenchUnavailable(f"{self._label} unreachable for {failures} polls: "
                                           f"{_redact(str(e))[:200]}") from None
                self._sleep(self.poll_s)
                continue
            failures = 0
            now = self._clock()
            if status == "RUNNING" and first_running is None:
                first_running = now
            if status in TERMINAL:
                return status, first_running, now
            if status in QUEUED and now - submitted > self.pending_timeout_s:
                self._t.cancel(job_id)
                self._pending.set(None)  # else the next resume reattaches to a cancelled job
                raise BenchUnavailable(f"no {self._label} capacity for {profile} in "
                                       f"{self.pending_timeout_s}s; job {job_id} cancelled")
            self._sleep(self.poll_s)

    def _forget_job(self) -> None:
        """Drop only the reattach keys of the pending record, keeping `purpose`/`hypothesis`/
        `files` so a resume re-enters the same iteration instead of reattaching for ever to a
        job that is already terminal and has no artifacts to collect."""
        rec = self._pending.get() or {}
        self._pending.set({k: v for k, v in rec.items()
                           if k not in ("job_id", "request_hash", "job_dir")})

    # ── evaluate ───────────────────────────────────────────────────────
    def evaluate(self, request: EvalRequest) -> EvalResult:
        pend = self._pending.get() or {}
        if self._stop:
            # a resume can restore a pending record naming a real, billing job before the
            # first poll ever runs; cancel it (best-effort, idempotent) whatever its
            # request_hash — only when no job_id is on record is there nothing to cancel
            stale_job_id = pend.get("job_id")
            if stale_job_id:
                self._t.cancel(stale_job_id)
            self._pending.set(None)
            raise BenchCancelled(stale_job_id or "stop requested before submission")
        purpose = str(pend.get("purpose", "adhoc"))
        job_dir = self.run_dir / self.subdir / purpose
        rh = request_hash(request)
        if pend.get("job_id") and pend.get("request_hash") == rh:
            job_id = _safe_job_id(pend["job_id"])
        else:
            job_id = self._submit(job_dir, request, purpose, rh)
        try:
            return self._collect(job_id, job_dir, request)
        except _JobFailed as first:
            job_id = self._submit(job_dir, request, purpose, rh)
            try:
                return self._collect(job_id, job_dir, request)
            except _JobFailed as second:
                raise BenchUnavailable(f"{self._label} job failed twice: {first}; "
                                       f"then {second}") from None

    def _submit(self, job_dir: Path, request: EvalRequest, purpose: str, rh: str) -> str:
        write_job_dir(job_dir, request, purpose, local=self.local)
        try:
            job_id = self._t.deploy(job_dir)
        except (C3CommandError, ValueError) as e:
            raise BenchUnavailable(f"{self._label} deploy failed: "
                                   f"{_redact(str(e))[:300]}") from None
        job_id = _safe_job_id(job_id)
        self._pending.set({**(self._pending.get() or {}), "backend": self.subdir, "job_id": job_id,
                           "job_dir": str(job_dir), "request_hash": rh})
        return job_id

    def _collect(self, job_id: str, job_dir: Path, request: EvalRequest) -> EvalResult:
        profile = c3_profile(CHALLENGES[request.challenge])
        status, t_run, t_end = self._wait(job_id, profile)
        if t_run is not None:
            # ESTIMATE. Reattaching to a job already RUNNING bills only from the reattach,
            # undercounting whatever ran before this process started polling; reattaching to
            # a job that is already terminal bills nothing for it at all.
            rate = (GBP_PER_HOUR[profile] * USD_PER_GBP if self.usd_per_hour is None
                    else self.usd_per_hour)
            self._cost += (t_end - t_run) / 3600 * rate
        artifacts = job_dir / job_id / "artifacts"
        try:
            have_results = self._t.fetch(job_id, "results.json", artifacts / "results.json")
            self._t.fetch(job_id, "build.log", artifacts / "build.log")
        except (C3CommandError, ValueError, KeyError, IndexError) as e:
            raise BenchUnavailable(
                f"{self._label} pull failed for {job_id}: {_redact(str(e))[:300]}") from None
        results = artifacts / "results.json"
        if not have_results:
            # every branch below is terminal for this job id: forget it, or the pause leaves a
            # record that makes every later resume reattach to a dead job and re-pause
            self._forget_job()
            if status in DONE:
                raise BenchUnavailable(f"{self._label} job {job_id} succeeded without results.json")
            if status == "TIMED_OUT":
                # a build/score run that burned its whole time budget is a property of the
                # files, not an infrastructure blip: resubmitting doubles the most expensive
                # failure mode instead of surfacing it
                raise BenchUnavailable(f"{self._label} job {job_id} timed out before writing "
                                       f"results")
            raise _JobFailed(f"{job_id} {status} with no results")
        try:
            return self._result_from(status, json.loads(results.read_text(encoding="utf-8")),
                                     request)
        except (ValueError, TypeError, KeyError, AttributeError) as e:
            raise BenchUnavailable(
                f"{self._label} job {job_id} wrote unreadable results: "
                f"{_redact(str(e))[:300]}") from None

    @staticmethod
    def _result_from(status: str, data: dict, request: EvalRequest) -> EvalResult:
        comp = CompileResult(**data["compile"])
        if not comp.ok:
            return EvalResult(comp, [], None, "not_compiled")
        started = data.get("started", {})
        tr = [NonceResult.from_dict(r) for r in data.get("training", [])]
        if started.get("training"):
            tr = fill_timeouts(tr, request.training)
        reason = data.get("holdout_reason", "not_won")
        if data.get("holdout") is not None:
            ho = fill_timeouts([NonceResult.from_dict(r) for r in data["holdout"]],
                               request.holdout)
        elif started.get("holdout"):
            ho = fill_timeouts([], request.holdout)
        elif status not in DONE and reason == "won":
            ho, reason = None, "timeout"  # the job decided to score held-out but was cut off
        else:
            ho = None
        return EvalResult(comp, tr, ho, reason)

    # ── protocol odds and ends ─────────────────────────────────────────
    def request_stop(self) -> None:
        self._stop = True

    def cost_mark(self) -> float:
        return self._cost

    def cost_usd_since(self, mark: float) -> float:
        return self._cost - mark
