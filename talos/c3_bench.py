"""C3 (cthree.cloud) bench backend: one evaluate call is one batch job, driven through the
`c3` CLI. Every subprocess call goes through the injected runner."""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable

from talos.bench import (BenchCancelled, BenchUnavailable, EvalRequest, EvalResult,
                         PendingJobStore, _redact)
from talos.c3_jobdir import request_hash, write_job_dir
from talos.challenges import CHALLENGES, c3_profile
from talos.inside import NONCE_TIMEOUT_S
from talos.types import CompileResult, NonceResult, NonceSet

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


def parse_json_stdout(text: str):
    """`c3 deploy --json` prints a Warning line before the JSON on experimental hardware."""
    for i, ch in enumerate(text):
        if ch in "{[":
            return json.loads(text[i:])
    raise ValueError("no JSON in c3 output")


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
                 pending_timeout_s: int = 1800, poll_failures_max: int = 15):
        self.run_dir = Path(run_dir)
        self._pending = pending or PendingJobStore.memory()
        self._run = run
        self._clock, self._sleep = clock, sleep
        self.poll_s, self.pending_timeout_s = poll_s, pending_timeout_s
        self.poll_failures_max = poll_failures_max
        self._cost = 0.0
        self._stop = False

    # ── CLI plumbing ───────────────────────────────────────────────────
    def _c3(self, *args: str, cwd: Path | None = None, timeout: int = 600) -> str:
        r = self._run(["c3", *args], capture_output=True, text=True, timeout=timeout,
                      cwd=str(cwd) if cwd else None)
        if r.returncode != 0:
            raise C3CommandError(f"c3 {args[0]} failed ({r.returncode}): "
                                 f"{_redact((r.stderr or r.stdout)[-500:])}")
        return r.stdout

    def _deploy(self, job_dir: Path) -> str:
        try:
            doc = parse_json_stdout(self._c3("deploy", "--json", cwd=job_dir))
        except (C3CommandError, ValueError) as e:
            raise BenchUnavailable(f"C3 deploy failed: {_redact(str(e))[:300]}") from None
        return doc["id"]

    def _status(self, job_id: str) -> str:
        for row in parse_json_stdout(self._c3("squeue", "--json", timeout=60)):
            if row.get("job_id") == job_id:
                return str(row.get("status", "UNKNOWN")).upper()
        raise C3CommandError(f"job {job_id} not listed by squeue")

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
                self._cancel(job_id)
                self._pending.set(None)
                raise BenchCancelled(job_id)
            try:
                status = self._status(job_id)
                if status not in ACTIVE and status not in TERMINAL:
                    raise C3CommandError(f"unrecognised job status {status!r}")
            except (C3CommandError, ValueError, KeyError, AttributeError, TypeError) as e:
                # a CLI error, a document shape we do not understand, or a status neither
                # ACTIVE nor TERMINAL: none of these are safe to poll on for ever
                failures += 1
                if failures >= self.poll_failures_max:
                    raise BenchUnavailable(f"C3 unreachable for {failures} polls: "
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
                self._cancel(job_id)
                self._pending.set(None)  # else the next resume reattaches to a cancelled job
                raise BenchUnavailable(f"no C3 capacity for {profile} in "
                                       f"{self.pending_timeout_s}s; job {job_id} cancelled")
            self._sleep(self.poll_s)

    def _cancel(self, job_id: str) -> None:
        try:
            self._c3("cancel", job_id, timeout=60)
        except C3CommandError:
            pass  # best effort: the job may already be terminal

    def _pull(self, job_id: str, job_dir: Path) -> Path:
        job_dir.mkdir(parents=True, exist_ok=True)  # a reattach never wrote the job dir
        pulled = job_dir / job_id
        d = pulled
        for attempt in range(2):
            try:
                doc = parse_json_stdout(self._c3("pull", job_id, "--json", cwd=job_dir))
            except (C3CommandError, ValueError):
                if attempt == 1:  # a transient pull failure gets one retry, same job id
                    raise
                continue
            jobs = doc.get("jobs") or []
            d = Path(jobs[0]["directory"]) if jobs and jobs[0].get("directory") else pulled
            if not d.is_absolute():
                d = job_dir / d
            for cand in (d / "artifacts", d):
                if (cand / "results.json").exists() or (cand / "build.log").exists():
                    return cand
            shutil.rmtree(pulled, ignore_errors=True)  # a "skipped" pull with nothing on disk
        return d

    # ── evaluate ───────────────────────────────────────────────────────
    def evaluate(self, request: EvalRequest) -> EvalResult:
        if self._stop:  # noticed before any deploy: nothing was submitted, so nothing to cancel
            self._pending.set(None)
            raise BenchCancelled("stop requested before submission")
        pend = self._pending.get() or {}
        purpose = str(pend.get("purpose", "adhoc"))
        job_dir = self.run_dir / "c3" / purpose
        rh = request_hash(request)
        if pend.get("job_id") and pend.get("request_hash") == rh:
            job_id = pend["job_id"]
        else:
            job_id = self._submit(job_dir, request, purpose, rh)
        try:
            return self._collect(job_id, job_dir, request)
        except _JobFailed as first:
            job_id = self._submit(job_dir, request, purpose, rh)
            try:
                return self._collect(job_id, job_dir, request)
            except _JobFailed as second:
                raise BenchUnavailable(f"C3 job failed twice: {first}; then {second}") from None

    def _submit(self, job_dir: Path, request: EvalRequest, purpose: str, rh: str) -> str:
        write_job_dir(job_dir, request, purpose)
        job_id = self._deploy(job_dir)
        self._pending.set({**(self._pending.get() or {}), "backend": "c3", "job_id": job_id,
                           "job_dir": str(job_dir), "request_hash": rh})
        return job_id

    def _collect(self, job_id: str, job_dir: Path, request: EvalRequest) -> EvalResult:
        profile = c3_profile(CHALLENGES[request.challenge])
        status, t_run, t_end = self._wait(job_id, profile)
        if t_run is not None:
            # ESTIMATE. Reattaching to a job already RUNNING bills only from the reattach,
            # undercounting whatever ran before this process started polling; reattaching to
            # a job that is already terminal bills nothing for it at all.
            self._cost += (t_end - t_run) / 3600 * GBP_PER_HOUR[profile] * USD_PER_GBP
        try:
            artifacts = self._pull(job_id, job_dir)
        except (C3CommandError, ValueError, KeyError, IndexError) as e:
            raise BenchUnavailable(
                f"C3 pull failed for {job_id}: {_redact(str(e))[:300]}") from None
        results = artifacts / "results.json"
        if not results.exists():
            if status in DONE:
                raise BenchUnavailable(f"C3 job {job_id} succeeded without results.json")
            if status == "TIMED_OUT":
                # a build/score run that burned its whole time budget is a property of the
                # files, not an infrastructure blip: resubmitting doubles the most expensive
                # failure mode instead of surfacing it
                raise BenchUnavailable(f"C3 job {job_id} timed out before writing results")
            raise _JobFailed(f"{job_id} {status} with no results")
        try:
            return self._result_from(status, json.loads(results.read_text()), request)
        except (ValueError, TypeError, KeyError, AttributeError) as e:
            raise BenchUnavailable(
                f"C3 job {job_id} wrote unreadable results: {_redact(str(e))[:300]}") from None

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
