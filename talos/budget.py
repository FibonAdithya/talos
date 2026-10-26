"""Budget caps and measured spend. Pure; the caller passes the clock."""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class Budget:
    usd: float | None
    hours: float | None
    iterations: int | None
    compute_usd: float | None

    def validate(self) -> None:
        # spec §13: the run budget (usd/hours/iterations) must be given; the compute cap is an
        # additional always-on cap, not a substitute, so it does not count toward "at least one".
        if all(v is None for v in (self.usd, self.hours, self.iterations)):
            raise ValueError("at least one budget dimension must be set")
        for name in ("usd", "hours", "iterations", "compute_usd"):
            v = getattr(self, name)
            if v is not None and v < 0:
                raise ValueError(f"budget {name} must be >= 0")

    def to_dict(self) -> dict:
        return asdict(self)


# The four caps a job can have, named as Budget's fields and as the --add-* flags name them.
ADD_DIMENSIONS: tuple[str, ...] = ("iterations", "hours", "usd", "compute_usd")


def effective_budget(budget: Budget, extensions: list[dict]) -> Budget:
    """The job's budget plus every extension (state.budget_extensions), cap by cap. A cap the
    job never had stays None: extensions add to caps, they do not create them."""
    caps = budget.to_dict()
    for e in extensions:
        for d, v in e["add"].items():
            if v is not None and caps[d] is not None:
                caps[d] += v
    return Budget(**caps)


@dataclass
class Spend:
    started_at: float
    llm_usd: float = 0.0
    compute_usd: float = 0.0
    iterations: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class BudgetExhausted(Exception):
    def __init__(self, dimension: str):
        super().__init__(f"budget exhausted: {dimension}")
        self.dimension = dimension


def exhausted(budget: Budget, spend: Spend, now: float) -> str | None:
    if budget.usd is not None and spend.llm_usd >= budget.usd:
        return "usd"
    if budget.hours is not None and (now - spend.started_at) >= budget.hours * 3600:
        return "hours"
    if budget.iterations is not None and spend.iterations >= budget.iterations:
        return "iterations"
    if budget.compute_usd is not None and spend.compute_usd >= budget.compute_usd:
        return "compute_usd"
    return None
