"""Lineage tree nodes, chains and the simulated-annealing maths. Pure: no I/O, no clock."""
from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass

from talos.budget import Budget, Spend

ROOT = 0  # node id of every chain's root


@dataclass(frozen=True)
class AnnealSettings:
    t0: float = 0.05
    floor: float = 0.1  # T at the end of a cooling run, as a fraction of t0
    stall: int = 5

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> AnnealSettings:
        return cls(**d)


@dataclass
class Node:
    id: int
    chain: int
    parent: int
    family: str
    title: str
    outcome: str
    score: float | None = None
    tracks: dict[str, float] | None = None
    metered_score: float | None = None
    accepted: bool = False
    probability: float | None = None
    draw: float | None = None
    temperature: float | None = None
    best: bool = False

    def rank_score(self) -> float | None:
        return self.metered_score if self.metered_score is not None else self.score

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Node:
        return cls(**d)


@dataclass
class Chain:
    index: int
    current: int = ROOT
    best: int = ROOT
    best_score: float | None = None
    stall: int = 0
    reheated_at: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Chain:
        return cls(**d)


def chain_for(iteration: int, chains: int) -> int:
    """Chains take turns; iterations are numbered from 1."""
    return (iteration - 1) % chains


def spent_fraction(budget: Budget, spend: Spend, now: float) -> float:
    """Largest spent fraction over the capped dimensions, clamped to [0, 1]; a cap of 0 is 1.0."""
    fractions: list[float] = []

    def add(used: float, cap: float | None) -> None:
        if cap is not None:
            fractions.append(1.0 if cap == 0 else used / cap)

    add(spend.llm_usd, budget.usd)
    add(now - spend.started_at, None if budget.hours is None else budget.hours * 3600)
    add(spend.iterations, budget.iterations)
    add(spend.compute_usd, budget.compute_usd)
    if not fractions:
        return 0.0
    return min(1.0, max(0.0, max(fractions)))


def temperature(settings: AnnealSettings, spent: float, reheated_at: float) -> float:
    """Linear cooling from t0 to t0 * floor over the budget left after the last reheat."""
    if reheated_at >= 1:
        progress = 1.0
    else:
        progress = min(1.0, max(0.0, (spent - reheated_at) / (1 - reheated_at)))
    return settings.t0 * (1 - (1 - settings.floor) * progress)


def accept_probability(drop: float, t: float) -> float:
    if drop <= 0:
        return 1.0
    if t <= 0:
        return 0.0
    return math.exp(-drop / t)


def draw(job_id: str, chain: int, iteration: int) -> float:
    """Deterministic uniform draw in [0, 1), so a resumed job decides as the original did."""
    digest = hashlib.sha256(f"{job_id}:{chain}:{iteration}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def decide(
    parent_score: float | None, score: float, t: float, u: float
) -> tuple[bool, float | None]:
    """Accept or reject a candidate. The probability is None when acceptance is automatic."""
    if parent_score is None or score >= parent_score:
        return True, None
    p = accept_probability(parent_score - score, t)
    return u < p, p
