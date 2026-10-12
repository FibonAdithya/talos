"""Which training nonces an iteration scores. No I/O here."""
from __future__ import annotations

from dataclasses import dataclass

from talos.scoring import focus_sets
from talos.types import NonceSet


class FixedTraining:
    """Every iteration scores the job's own training sets. This is what the loop always did."""
    fresh = False

    def training(self, spec, iteration: int) -> list[NonceSet]:
        return focus_sets(spec.track, spec.training, spec.holdout)[0]


@dataclass(frozen=True)
class FreshTraining:
    """Iteration i, counted from 1, scores `count` nonces per track starting at count * (i - 1),
    so no two iterations share a nonce. The baseline and the current best have no results on
    them yet, so the loop solves both there before it scores the candidate.

    Limit: this policy is for benches that run a solve to completion inside `evaluate` and do
    not keep a job across processes. With a bench that reattaches to a pending job (C3), a
    resumed iteration submits its reference solves first, so the pending candidate job is not
    reattached."""
    count: int
    fresh = True

    def training(self, spec, iteration: int) -> list[NonceSet]:
        if iteration < 1:
            raise ValueError(f"iterations count from 1, got {iteration}")
        return [NonceSet(track=t, rand_hash=spec.rand_hash,
                         start=self.count * (iteration - 1), count=self.count)
                for t in spec.tracks]
