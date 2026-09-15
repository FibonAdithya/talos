"""The research loop. One Loop per job; all state lives in JobState and is saved after every
step so the run can be resumed. No network code here beyond calling the provider and bench."""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from typing import Callable

from talos.baseline import resolve_baseline
from talos.bench import BenchCancelled, BenchUnavailable, EvalRequest, EvalResult
from talos.budget import BudgetExhausted, exhausted
from talos.challenges import CHALLENGES
from talos.edits import EditError, EditOutcome, apply_edit_response
from talos.prompts import (PromptContext, STRATEGY_TAGS, compile_fix_prompts, distill_prompts,
                           edit_prompts, edit_repair_prompts, hypothesis_prompts,
                           parse_distillation, parse_hypothesis)
from talos.providers import ProviderAuthError, ProviderError, ProviderRateLimited
from talos.scoring import ScoringError, beats, bundle_delta
from talos.search_replace import format_misses
from talos.state import Candidate, JobSpec, JobState, JobStore, TERMINAL
from talos.types import Completion


@dataclass
class Thresholds:
    recall: int = 2
    distill: int = 3
    reset: int = 5
    compile_fix_rounds: int = 3
    edit_repair_rounds: int = 1
    rate_limit_wait_s: int = 30
    rate_limit_max_waits: int = 20


class Interrupted(Exception):
    pass


class _BudgetedBench:
    """The bench, with the loop's budget check and cost accounting around every call.

    resolve_baseline compiles and scores both nonce sets in one evaluate of its own, and on a
    cold cache that is the most expensive compute call of the whole job. Handing it the raw bench
    let a baseline run to completion with --budget-compute-usd 0, and charged its cost only after
    the call had already happened."""

    def __init__(self, loop: "Loop"):
        self._loop = loop
        self._bench = loop.bench

    def _metered(self, call):
        loop = self._loop
        loop._check_budget()
        mark = self._bench.cost_mark()
        try:
            return call()
        finally:
            loop.state.spend.compute_usd += self._bench.cost_usd_since(mark)
            loop._save()

    def evaluate(self, request):
        return self._metered(lambda: self._bench.evaluate(request))

    def request_stop(self) -> None:
        self._bench.request_stop()

    def cost_mark(self) -> float:
        return self._bench.cost_mark()

    def cost_usd_since(self, mark: float) -> float:
        return self._bench.cost_usd_since(mark)


class Loop:
    def __init__(self, spec: JobSpec, state: JobState, store: JobStore, provider, bench,
                 template_rs: str, clock=time.time, sleep=time.sleep,
                 on_event: Callable[[str, dict], None] | None = None,
                 thresholds: Thresholds | None = None):
        self.spec, self.state, self.store = spec, state, store
        self.provider, self.bench = provider, bench
        self.template_rs = template_rs
        self.clock, self.sleep = clock, sleep
        self.on_event = on_event or (lambda kind, data: None)
        self.t = thresholds or Thresholds()
        self.rule = CHALLENGES[spec.challenge].beat
        self._stop = False
        # The iteration an event belongs to. state.iteration only catches up when the iteration
        # finishes, so events raised mid-iteration would otherwise carry the previous number.
        self._n = state.iteration
        self.propose_and_edit = self.single_shot_propose_and_edit

    # ── plumbing ──────────────────────────────────────────────────────

    def request_stop(self) -> None:
        self._stop = True

    def _event(self, kind: str, **data) -> None:
        self.store.event(kind, iteration=self._n, **data)
        self.on_event(kind, data)

    def _save(self) -> None:
        self.store.save(self.state)

    def _check_budget(self) -> None:
        dim = exhausted(self.spec.budget, self.state.spend, self.clock())
        if dim:
            raise BudgetExhausted(dim)
        if self._stop:
            raise Interrupted()

    def _llm(self, system: str, user: str) -> Completion:
        self._check_budget()
        waits = 0
        while True:
            try:
                c = self.provider.complete(system, user)
                break
            except ProviderRateLimited as e:
                waits += 1
                if waits > self.t.rate_limit_max_waits:
                    raise ProviderError(f"gave up after {waits} rate-limit waits: {e}")
                self._event("rate_limited", wait_s=self.t.rate_limit_wait_s)
                self.sleep(self.t.rate_limit_wait_s)
                self._check_budget()  # a stop or an hours cap must land during a rate-limit storm
        # `is not None`, not truthiness: a measured cost of exactly 0.0 is a measurement, and
        # None means "this model has no price-table entry" — which the CLI reports as unpriced
        # rather than as $0.00, because a dollar cap cannot be enforced against it.
        if c.usage.cost_usd is not None:
            self.state.spend.llm_usd += c.usage.cost_usd
        self._save()
        return c

    def _request(self, files: dict[str, str], baseline_training) -> EvalRequest:
        return EvalRequest(challenge=self.spec.challenge, files=files, training=self.spec.training,
                           holdout=self.spec.holdout, fuel=self.spec.fuel,
                           baseline_training=baseline_training, rule=self.rule)

    def _bench_evaluate(self, files: dict[str, str]) -> EvalResult:
        """pending_job carries the files BEFORE the call: a C3 job outlives this process, and a
        resume must be able to rebuild the exact request and reattach to it."""
        self._check_budget()
        self.state.pending_job = {**(self.state.pending_job or {}), "files": files}
        self._save()
        mark = self.bench.cost_mark()
        try:
            return self.bench.evaluate(self._request(files, self.state.baseline.training))
        finally:
            self.state.spend.compute_usd += self.bench.cost_usd_since(mark)
            self._save()

    # ── baseline ──────────────────────────────────────────────────────

    def measure_baseline(self, cache_dir, hardware_class: str, mainnet=None) -> None:
        self._check_budget()  # a zero (or already spent) compute cap must bite before the call
        self.state.status = "measuring_baseline"
        self._save()
        pend = self.state.pending_job
        if not (pend and pend.get("purpose") == "baseline"):
            pend = {"purpose": "baseline"}
        self.state.pending_job = pend  # keep a stored job_id: a resumed baseline reattaches
        self._save()
        kw = {"mainnet": mainnet} if mainnet else {}
        rec, template = resolve_baseline(self.spec.challenge, self.spec.training,
                                         self.spec.holdout, self.spec.fuel, _BudgetedBench(self),
                                         cache_dir, hardware_class, rule=self.rule,
                                         log=lambda m: self._event("baseline", message=m), **kw)
        self.state.pending_job = None
        self.state.baseline = rec
        self.template_rs = template
        self.state.status = "researching"
        self._save()
        self._write_files(self.store.run_dir / "baseline", rec.files)
        (self.store.run_dir / "baseline" / "results.json").write_text(json.dumps(
            {"training": [r.to_dict() for r in rec.training],
             "holdout": [r.to_dict() for r in rec.holdout]}, indent=1))
        self._event("baseline_ready", name=rec.name, adoption=rec.adoption)

    @staticmethod
    def _write_files(directory, files: dict[str, str]) -> None:
        """spec §5.4: `baseline/` and `best/` in the run directory hold the current sources as
        plain files, replaced whole so they never mix two candidates' files."""
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)
        for name, text in files.items():
            p = directory / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)

    # ── context ───────────────────────────────────────────────────────

    def _current_files(self) -> dict[str, str]:
        return dict(self.state.best.files if self.state.best else self.state.baseline.files)

    def _current_training(self):
        return self.state.best.training if self.state.best else self.state.baseline.training

    def _best_delta(self) -> float:
        return self.state.best.delta["mean_rel_delta"] if self.state.best else 0.0

    def _failed_for_current(self) -> list[dict]:
        anchor = self.state.best.iteration if self.state.best else 0
        return [h for h in self.state.hypotheses
                if h.get("against") == anchor and h.get("outcome", "").startswith("failed")]

    def _forced_tag(self) -> str | None:
        if self.state.runs_since_improvement < self.t.reset:
            return None
        counts = self.state.strategy_counts
        return min(STRATEGY_TAGS, key=lambda tag: counts.get(tag, 0))

    def _context(self) -> PromptContext:
        recall = self.state.runs_since_improvement >= self.t.recall
        failed = self._failed_for_current() if recall else []
        return PromptContext(challenge=self.spec.challenge, template_rs=self.template_rs,
                             direction=self.spec.direction, tacit=self.state.tacit,
                             files=self._current_files(),
                             baseline_name=self.state.baseline.name,
                             best_delta=self._best_delta(), failed_hypotheses=failed,
                             forced_tag=self._forced_tag(),
                             is_gpu=CHALLENGES[self.spec.challenge].is_gpu)

    # ── single-shot propose + edit ────────────────────────────────────

    def single_shot_propose_and_edit(self, ctx: PromptContext) -> tuple[dict, dict[str, str]]:
        system, user = hypothesis_prompts(ctx)
        hypothesis = parse_hypothesis(self._llm(system, user).text)
        self._event("hypothesis", **hypothesis)
        system, user = edit_prompts(ctx, hypothesis)
        text = self._llm(system, user).text
        outcome = apply_edit_response(ctx.files, text)
        for _ in range(self.t.edit_repair_rounds):
            if not outcome.misses:
                break
            system, user = edit_repair_prompts(ctx, outcome.files, format_misses(outcome.misses))
            try:
                repaired = apply_edit_response(outcome.files, self._llm(system, user).text)
            except EditError:
                break  # a repair reply with no blocks: keep what was applied, skip the misses
            # Accumulate, never replace: the repair round re-emits only the failed blocks, so its
            # own `applied` says nothing about the first response's blocks, and its `rejected`
            # would forget an out-of-scope block that the first response carried.
            outcome = EditOutcome(files=repaired.files, applied=outcome.applied + repaired.applied,
                                  misses=repaired.misses,
                                  rejected=outcome.rejected + repaired.rejected)
        if outcome.rejected:
            # spec §9: an edit outside the algorithm files fails the iteration in BOTH modes.
            # Applying the in-scope blocks of the same response and carrying on would let the
            # out-of-scope block go unpunished and reward a response that tried it.
            self._event("edits_rejected", paths=outcome.rejected)
            raise EditError("edit outside the algorithm files "
                            f"rejected: {sorted(outcome.rejected)}")
        if outcome.applied == 0:
            raise EditError("no edit block applied")
        return hypothesis, outcome.files

    # ── one iteration ─────────────────────────────────────────────────

    def iterate(self) -> None:
        n = self.state.iteration + 1
        self._n = n
        it_dir = self.store.iteration_dir(n)
        ctx = self._context()
        anchor = self.state.best.iteration if self.state.best else 0
        record = {"iteration": n, "against": anchor, "title": "", "description": "",
                  "strategy_tag": "", "outcome": "started"}
        try:
            hypothesis, files = self.propose_and_edit(ctx)
        except (EditError, ValueError) as e:
            record.update(outcome="failed:edit", error=str(e))
            self._finish_iteration(n, record, improved=False)
            return
        record.update(hypothesis)
        (it_dir / "hypothesis.json").write_text(json.dumps(hypothesis))
        self.state.pending_job = {"purpose": n, "hypothesis": hypothesis, "files": files}
        self._save()
        self._score_candidate(n, ctx, record, hypothesis, files)

    def _score_candidate(self, n: int, ctx: PromptContext, record: dict, hypothesis: dict,
                         files: dict[str, str]) -> None:
        """Everything after the LLM has produced files: compile-fix rounds, scoring, confirmation.
        Entered from iterate() and from a resume with a pending job."""
        it_dir = self.store.iteration_dir(n)
        res = self._bench_evaluate(files)
        for _ in range(self.t.compile_fix_rounds):
            if res.compile.ok:
                break
            self._event("compile_failed", output=res.compile.output[-2000:])
            system, user = compile_fix_prompts(ctx, files, res.compile.output)
            try:
                fixed = apply_edit_response(files, self._llm(system, user).text)
            except EditError:
                break
            if fixed.rejected:
                # spec §9 holds for the fix response too: never apply its in-scope blocks either.
                self._event("edits_rejected", paths=fixed.rejected)
                record.update(outcome="failed:edit", error="edit outside the algorithm files "
                              f"rejected: {sorted(fixed.rejected)}")
                self._finish_iteration(n, record, improved=False)
                return
            if fixed.applied == 0:
                break  # byte-identical files; re-evaluating them would repeat the same error
            files = fixed.files
            res = self._bench_evaluate(files)
        if not res.compile.ok:
            record.update(outcome="failed:compile")
            self._finish_iteration(n, record, improved=False)
            return
        for name, text in files.items():
            (it_dir / name).parent.mkdir(parents=True, exist_ok=True)
            (it_dir / name).write_text(text)
        results = res.training
        try:
            delta = bundle_delta(self.state.baseline.training, results)
        except ScoringError as e:
            record.update(outcome="failed:score", error=str(e))
            self._finish_iteration(n, record, improved=False)
            return
        cand = Candidate(iteration=n, files=files, artifact_id=res.compile.artifact_id,
                         training=results, delta=delta.to_dict(), hypothesis=hypothesis)
        self._event("scored", mean_rel_delta=delta.mean_rel_delta,
                    worst_rel_delta=delta.worst_rel_delta, error_rate=delta.error_rate)
        if delta.error_rate > self.rule.error_ceiling:  # spec §9: over the ceiling is a failure
            record.update(outcome="failed:runtime", error_rate=delta.error_rate)
            self._finish_iteration(n, record, improved=False)
            return
        # spec §7.7: a candidate that beats the baseline on training becomes the best, whether or
        # not the held-out set goes on to confirm it. Confirming a candidate that is not best would
        # report "won" with different code in state.best.
        wins = beats(self.state.baseline.training, results, self.rule)
        improved = delta.mean_rel_delta > self._best_delta() or wins
        if improved:
            self.state.best = cand
            self._write_files(self.store.run_dir / "best", cand.files)
            record["outcome"] = "improved"
        else:
            record["outcome"] = "failed:score"
        # The record and the iteration bump are persisted BEFORE the confirmation is read out of
        # the result, so a kill between the two never loses the iteration.
        self._finish_iteration(n, record, improved=improved)
        if wins:
            self._confirm(cand, res.holdout, res.holdout_reason)

    def _confirm(self, cand: Candidate, ho, reason: str) -> None:
        """The held-out results come back from the same evaluate call that scored training: the
        bench (or the C3 job) applies the beat rule itself and scores held-out only on a win."""
        self.state.status = "confirming"
        self._save()
        self._event("confirming")
        cand.holdout = ho
        if ho is None:
            # The job said "won on training" but produced no held-out results (a timeout inside
            # the container, say). That is not a win, and it must not strand the job at
            # "confirming".
            self.state.false_positives.append(cand.iteration)
            self.state.status = "researching"
            self._save()
            self._event("false_positive", error=f"held-out not scored ({reason})")
            return
        try:
            won = beats(self.state.baseline.holdout, ho, self.rule)
            holdout_delta = bundle_delta(self.state.baseline.holdout, ho).to_dict()
        except ScoringError as e:
            # An unscoreable held-out run is not a win; it must not leave the job at "confirming".
            self.state.false_positives.append(cand.iteration)
            self.state.status = "researching"
            self._save()
            self._event("false_positive", error=str(e))
            return
        if won:
            self.state.confirmed.append(cand.iteration)
            self.state.status = "won"
            self.state.stop_reason = "beat baseline on training and held-out nonces"
            self._mark_won(cand.iteration)
            self._event("won", holdout=holdout_delta)
        else:
            self.state.false_positives.append(cand.iteration)
            self.state.status = "researching"
            self._event("false_positive", holdout=holdout_delta)
        self._save()

    def _mark_won(self, iteration: int) -> None:
        """The record for this iteration was appended by _finish_iteration just before the
        held-out run, so it is the last one — but check rather than assume."""
        if self.state.hypotheses and self.state.hypotheses[-1].get("iteration") == iteration:
            self.state.hypotheses[-1]["outcome"] = "won"

    def _finish_iteration(self, n: int, record: dict, improved: bool) -> None:
        self.state.iteration = n
        self.state.spend.iterations += 1
        self.state.hypotheses.append(record)
        self._n = n
        if improved:
            self.state.runs_since_improvement = 0
        else:
            self.state.runs_since_improvement += 1
        # Counted here, not where the hypothesis is parsed: a resumed pending iteration re-enters
        # at _score_candidate, and counting at propose time would count that iteration twice.
        if record.get("strategy_tag"):
            self.state.strategy_counts[record["strategy_tag"]] = (
                self.state.strategy_counts.get(record["strategy_tag"], 0) + 1)
        self.state.pending_job = None
        self._save()
        self._event("iteration_done", outcome=record["outcome"],
                    runs_since_improvement=self.state.runs_since_improvement)
        if not improved and self.state.runs_since_improvement == self.t.distill:
            self._distill()
        if not improved and self.state.runs_since_improvement >= self.t.reset:
            self._event("reset", forced_tag=self._forced_tag())

    def _distill(self) -> None:
        failed = self._failed_for_current()
        if not failed:
            return
        system, user = distill_prompts(self._context(), failed)
        try:
            text = self._llm(system, user).text
        except ProviderAuthError:
            raise  # spec §9: a dead key or a billing failure stops the run at once
        except ProviderError:
            return
        lesson = parse_distillation(text)
        if lesson:
            self.state.tacit = (self.state.tacit.rstrip() + f"\n- LLM: {lesson}\n").lstrip()
            (self.store.run_dir / "tacit.md").write_text(self.state.tacit)
            self._save()
            self._event("distilled", lesson=lesson)

    # ── driver ────────────────────────────────────────────────────────

    def _discard_incomplete_iteration(self) -> None:
        n = self.state.iteration + 1
        if (self.state.pending_job or {}).get("purpose") == n:
            return  # that iteration is not abandoned: its job is still in flight and resumable
        d = self.store.run_dir / "iterations" / f"{n:04d}"
        if d.exists():
            shutil.rmtree(d)
            self._event("discarded_incomplete", discarded=n)

    def _resume_pending(self, pending: dict) -> None:
        n = pending["purpose"]
        self._n = n
        self._event("resumed_pending", job_id=pending.get("job_id"))
        anchor = self.state.best.iteration if self.state.best else 0
        hypothesis = pending["hypothesis"]
        record = {"iteration": n, "against": anchor, **hypothesis, "outcome": "started"}
        self._score_candidate(n, self._context(), record, hypothesis, pending["files"])

    def run(self) -> JobState:
        if self.state.status in TERMINAL:
            return self.state  # before the discard: a finished job keeps its winning directory
        self._discard_incomplete_iteration()
        try:
            self.state.status = "researching"
            self._save()
            pending = self.state.pending_job
            if pending is not None and pending.get("purpose") == "baseline":
                self.state.pending_job = None  # the baseline is recorded; the record is stale
                self._save()
            elif pending is not None and isinstance(pending.get("purpose"), int):
                # An evaluate killed mid-flight (a stop, a bench outage, a hard kill): re-enter
                # that iteration with its own files, so the backend can reattach to the job
                # instead of abandoning one C3 is still billing for.
                self._resume_pending(pending)
            while self.state.status not in TERMINAL:
                self._check_budget()
                self.iterate()
        except BudgetExhausted as e:
            self.state.status, self.state.stop_reason = "exhausted", e.dimension
        except Interrupted:
            self.state.status, self.state.stop_reason = "cancelled", "user requested stop"
        except BenchCancelled as e:
            self.state.status, self.state.stop_reason = ("cancelled",
                                                         f"stopped; bench job cancelled: {e}")
        except ProviderAuthError as e:
            self.state.status, self.state.stop_reason = "failed", f"provider auth: {e}"
        except ProviderError as e:
            self.state.status, self.state.stop_reason = "failed", f"provider: {e}"
        except BenchUnavailable as e:
            self.state.status, self.state.stop_reason = "paused", f"bench: {e}"
        self._save()
        self._event("stopped", status=self.state.status, reason=self.state.stop_reason)
        return self.state
