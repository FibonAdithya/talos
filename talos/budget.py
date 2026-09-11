"""Budget caps and measured spend. Pure; the caller passes the clock."""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class Budget:
    usd: float | None
    hours: float | None
    iterations: int | None
    modal_usd: float | None

    def validate(self) -> None:
        if all(v is None for v in (self.usd, self.hours, self.iterations, self.modal_usd)):
            raise ValueError("at least one budget dimension must be set")
        for name in ("usd", "hours", "iterations", "modal_usd"):
            v = getattr(self, name)
            if v is not None and v < 0:
                raise ValueError(f"budget {name} must be >= 0")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Spend:
    started_at: float
    llm_usd: float = 0.0
    modal_usd: float = 0.0
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
    if budget.modal_usd is not None and spend.modal_usd >= budget.modal_usd:
        return "modal_usd"
    return None
