"""The research loop. One Loop per job; all state lives in JobState and is saved after every
step so the run can be resumed. No network code here beyond calling the provider and bench."""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from typing import Callable

from talos.baseline import resolve_baseline
from talos.bench import BenchUnavailable
from talos.budget import BudgetExhausted, exhausted
from talos.challenges import CHALLENGES
from talos.edits import EditError, apply_edit_response
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


class Loop:
    def __init__(self, spec: JobSpec, state: JobState, store: JobStore, provider, bench,
                 template_rs: str, clock=time.time, sleep=time.sleep,
                 on_event: Callable[[str, dict], None] | None = None,
                 thresholds: Thresholds = Thresholds()):
        self.spec, self.state, self.store = spec, state, store
        self.provider, self.bench = provider, bench
        self.template_rs = template_rs
        self.clock, self.sleep = clock, sleep
        self.on_event = on_event or (lambda kind, data: None)
        self.t = thresholds
        self.rule = CHALLENGES[spec.challenge].beat
        self._stop = False
        self.propose_and_edit = self.single_shot_propose_and_edit

    # ── plumbing ──────────────────────────────────────────────────────

    def request_stop(self) -> None:
        self._stop = True

    def _event(self, kind: str, **data) -> None:
        self.store.event(kind, iteration=self.state.iteration, **data)
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
        if c.usage.cost_usd:
            self.state.spend.llm_usd += c.usage.cost_usd
        self._save()
        return c

    def _bench_compile(self, files):
        self._check_budget()
        mark = self.bench.cost_mark()
        r = self.bench.compile(self.spec.challenge, files)
        self.state.spend.modal_usd += self.bench.cost_usd_since(mark)
        self._save()
        return r

    def _bench_score(self, artifact_id, nonce_sets):
        self._check_budget()
        mark = self.bench.cost_mark()
        r = self.bench.score(self.spec.challenge, artifact_id, nonce_sets, self.spec.fuel)
        self.state.spend.modal_usd += self.bench.cost_usd_since(mark)
        self._save()
        return r

    # ── baseline ──────────────────────────────────────────────────────

    def measure_baseline(self, cache_dir, hardware_class: str, mainnet=None) -> None:
        self.state.status = "measuring_baseline"
        self._save()
        kw = {"mainnet": mainnet} if mainnet else {}
        mark = self.bench.cost_mark()
        rec, template = resolve_baseline(self.spec.challenge, self.spec.training,
                                         self.spec.holdout, self.spec.fuel, self.bench,
                                         cache_dir, hardware_class,
                                         log=lambda m: self._event("baseline", message=m), **kw)
        self.state.spend.modal_usd += self.bench.cost_usd_since(mark)
        self.state.baseline = rec
        self.template_rs = template
        self.state.status = "researching"
        self._save()
        self._event("baseline_ready", name=rec.name, adoption=rec.adoption)

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
            repaired = apply_edit_response(outcome.files, self._llm(system, user).text)
            outcome = repaired
        if outcome.applied == 0:
            raise EditError("no edit block applied")
        return hypothesis, outcome.files

    # ── one iteration ─────────────────────────────────────────────────

    def iterate(self) -> None:
        n = self.state.iteration + 1
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
        self.state.strategy_counts[hypothesis["strategy_tag"]] = (
            self.state.strategy_counts.get(hypothesis["strategy_tag"], 0) + 1)

        # compile with bounded fix rounds
        comp = self._bench_compile(files)
        for _ in range(self.t.compile_fix_rounds):
            if comp.ok:
                break
            self._event("compile_failed", output=comp.output[-2000:])
            system, user = compile_fix_prompts(ctx, files, comp.output)
            try:
                files = apply_edit_response(files, self._llm(system, user).text).files
            except EditError:
                break
            comp = self._bench_compile(files)
        if not comp.ok:
            record.update(outcome="failed:compile")
            self._finish_iteration(n, record, improved=False)
            return
        for name, text in files.items():
            (it_dir / name).parent.mkdir(parents=True, exist_ok=True)
            (it_dir / name).write_text(text)

        # score on training
        results = self._bench_score(comp.artifact_id, self.spec.training)
        try:
            delta = bundle_delta(self.state.baseline.training, results)
        except ScoringError as e:
            record.update(outcome="failed:score", error=str(e))
            self._finish_iteration(n, record, improved=False)
            return
        cand = Candidate(iteration=n, files=files, artifact_id=comp.artifact_id,
                         training=results, delta=delta.to_dict(), hypothesis=hypothesis)
        self._event("scored", mean_rel_delta=delta.mean_rel_delta,
                    worst_rel_delta=delta.worst_rel_delta, error_rate=delta.error_rate)
        if delta.error_rate > self.rule.error_ceiling:  # spec §9: over the ceiling is a runtime failure
            record.update(outcome="failed:runtime", error_rate=delta.error_rate)
            self._finish_iteration(n, record, improved=False)
            return
        improved = (delta.mean_rel_delta > self._best_delta()
                    or self.state.best is None and delta.mean_rel_delta > 0)
        if improved:
            self.state.best = cand
            record["outcome"] = "improved"
        else:
            record["outcome"] = "failed:score"

        if beats(self.state.baseline.training, results, self.rule):
            self._confirm(cand)
        self._finish_iteration(n, record, improved=improved)

    def _confirm(self, cand: Candidate) -> None:
        self.state.status = "confirming"
        self._save()
        self._event("confirming")
        ho = self._bench_score(cand.artifact_id, self.spec.holdout)
        cand.holdout = ho
        if beats(self.state.baseline.holdout, ho, self.rule):
            self.state.confirmed.append(cand.iteration)
            self.state.status = "won"
            self.state.stop_reason = "beat baseline on training and held-out nonces"
            self._event("won", holdout=bundle_delta(self.state.baseline.holdout, ho).to_dict())
        else:
            self.state.false_positives.append(cand.iteration)
            self.state.status = "researching"
            self._event("false_positive",
                        holdout=bundle_delta(self.state.baseline.holdout, ho).to_dict())
        self._save()

    def _finish_iteration(self, n: int, record: dict, improved: bool) -> None:
        self.state.iteration = n
        self.state.spend.iterations += 1
        self.state.hypotheses.append(record)
        if improved:
            self.state.runs_since_improvement = 0
        else:
            self.state.runs_since_improvement += 1
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
            lesson = parse_distillation(self._llm(system, user).text)
        except ProviderError:
            return
        if lesson:
            self.state.tacit = (self.state.tacit.rstrip() + f"\n- LLM: {lesson}\n").lstrip()
            (self.store.run_dir / "tacit.md").write_text(self.state.tacit)
            self._save()
            self._event("distilled", lesson=lesson)

    # ── driver ────────────────────────────────────────────────────────

    def _discard_incomplete_iteration(self) -> None:
        n = self.state.iteration + 1
        d = self.store.run_dir / "iterations" / f"{n:04d}"
        if d.exists():
            shutil.rmtree(d)
            self._event("discarded_incomplete", discarded=n)

    def run(self) -> JobState:
        self._discard_incomplete_iteration()
        if self.state.status in TERMINAL:
            return self.state
        self.state.status = "researching"
        self._save()
        try:
            while self.state.status not in TERMINAL:
                self._check_budget()
                self.iterate()
        except BudgetExhausted as e:
            self.state.status, self.state.stop_reason = "exhausted", e.dimension
        except Interrupted:
            self.state.status, self.state.stop_reason = "cancelled", "user requested stop"
        except ProviderAuthError as e:
            self.state.status, self.state.stop_reason = "failed", f"provider auth: {e}"
        except ProviderError as e:
            self.state.status, self.state.stop_reason = "failed", f"provider: {e}"
        except BenchUnavailable as e:
            self.state.status, self.state.stop_reason = "paused", f"bench: {e}"
        self._save()
        self._event("stopped", status=self.state.status, reason=self.state.stop_reason)
        return self.state
