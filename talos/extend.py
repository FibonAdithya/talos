"""Adding budget to a job on resume (`talos run --resume JOB --add-*`).

job.json is write-once, so an extension is an entry appended to state.budget_extensions;
talos/budget.py::effective_budget adds the entries to the job's original budget."""
from __future__ import annotations

from talos.budget import ADD_DIMENSIONS, Budget, effective_budget, exhausted
from talos.lineage import spent_fraction
from talos.state import JobState

ADD_FLAGS: dict[str, str] = {"iterations": "--add-iterations", "hours": "--add-hours",
                             "usd": "--add-usd", "compute_usd": "--add-compute-usd"}
BUDGET_FLAGS: dict[str, str] = {"iterations": "--budget-iterations", "hours": "--budget-hours",
                                "usd": "--budget-usd", "compute_usd": "--budget-compute-usd"}


class ExtensionRefused(ValueError):
    """The extension cannot be applied; str() is the message for the user."""


def extend(budget: Budget, state: JobState, add: dict[str, float], now: float,
           previous_status: str) -> dict:
    """Validates one extension against the job's budget and spend, then records it in
    state.budget_extensions and returns the entry. Refuses (ExtensionRefused) before changing
    anything. The caller saves.

    `now` is the clock after any resume shift of spend.started_at. Annealing reheat points are
    spent fractions of the budget in force when the chain reheated, so each is rescaled onto
    the new budget: an extension continues the cooling where the larger budget would have it,
    instead of reheating every chain to t0."""
    for d, v in add.items():
        if v <= 0:
            raise ExtensionRefused(f"{ADD_FLAGS[d]} must be greater than 0, not {v}")
    for d in add:
        if getattr(budget, d) is None:
            raise ExtensionRefused(
                f"{ADD_FLAGS[d]}: the job was started without {BUDGET_FLAGS[d]}, so it has no "
                f"{d} cap to add to")
    entry = {"at": now, "iteration": state.iteration, "previous_status": previous_status,
             "add": {d: add.get(d) for d in ADD_DIMENSIONS}}
    old = effective_budget(budget, state.budget_extensions)
    new = effective_budget(budget, state.budget_extensions + [entry])
    still = exhausted(new, state.spend, now)
    if still:
        raise ExtensionRefused(f"job is still out of budget on {still}; add {ADD_FLAGS[still]}")
    s_old, s_new = spent_fraction(old, state.spend, now), spent_fraction(new, state.spend, now)
    if s_old > 0:
        for c in state.chains:
            c.reheated_at *= s_new / s_old
    state.budget_extensions.append(entry)
    return entry
