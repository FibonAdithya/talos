import pytest

from talos.budget import Budget, Spend, effective_budget, exhausted
from talos.extend import ExtensionRefused, extend
from talos.lineage import AnnealSettings, Chain, temperature
from talos.state import JobState


def job(iterations_cap, spent, hours=None, started_at=0.0):
    b = Budget(usd=None, hours=hours, iterations=iterations_cap, compute_usd=None)
    st = JobState.fresh(Spend(started_at=started_at, iterations=spent))
    st.iteration = spent
    return b, st


def test_extend_appends_an_entry_and_lifts_the_cap():
    # mutation: not appending leaves the job exhausted
    b, st = job(5, 5)
    entry = extend(b, st, {"iterations": 2}, now=100.0, previous_status="exhausted")
    assert entry == {"at": 100.0, "iteration": 5, "previous_status": "exhausted",
                     "add": {"iterations": 2, "hours": None, "usd": None, "compute_usd": None}}
    assert st.budget_extensions == [entry]
    assert exhausted(effective_budget(b, st.budget_extensions), st.spend, 100.0) is None


def test_extend_refuses_when_another_cap_is_still_spent():
    # mutation: checking only the dimensions being added lets an hours-exhausted job through
    # mutation: appending before the check leaves a refused entry in state
    b, st = job(5, 5, hours=1.0)
    with pytest.raises(ExtensionRefused) as e:
        extend(b, st, {"iterations": 2}, now=3600.0, previous_status="exhausted")
    assert "still out of budget on hours" in str(e.value) and "--add-hours" in str(e.value)
    assert st.budget_extensions == []


def test_extend_refuses_a_cap_the_job_never_had():
    # mutation: adding to a None cap silently creates an hours limit the user never set
    b, st = job(5, 5)
    with pytest.raises(ExtensionRefused) as e:
        extend(b, st, {"hours": 1.0, "iterations": 1}, now=0.0, previous_status="exhausted")
    assert "--budget-hours" in str(e.value)
    assert st.budget_extensions == []


@pytest.mark.parametrize("value", [0, -1])
def test_extend_refuses_a_non_positive_addition(value):
    # mutation: `if v < 0` lets a zero extension through; `if not v` lets -1 through
    b, st = job(5, 4)
    with pytest.raises(ExtensionRefused) as e:
        extend(b, st, {"iterations": value}, now=0.0, previous_status="cancelled")
    assert "--add-iterations must be greater than 0" in str(e.value)


def test_extend_rescales_reheat_points_onto_the_new_budget():
    # 20 of 20 iterations spent (s_old = 1.0); +20 gives s_new = 0.5. By hand:
    # 0.8 * 0.5 / 1.0 = 0.4; progress (0.5 - 0.4) / 0.6 = 1/6; T = 0.05 * (1 - 0.9 / 6) = 0.0425
    # mutation: no rescale leaves 0.8 > s_new, progress clamps to 0 and T = t0 = 0.05
    # mutation: s_old / s_new instead gives 1.6
    b, st = job(20, 20)
    st.chains = [Chain(0, reheated_at=0.8), Chain(1, reheated_at=0.0)]
    extend(b, st, {"iterations": 20}, now=0.0, previous_status="exhausted")
    assert st.chains[0].reheated_at == pytest.approx(0.4)
    assert st.chains[1].reheated_at == 0.0
    assert temperature(AnnealSettings(), 0.5, st.chains[0].reheated_at) == pytest.approx(0.0425)


def test_extend_on_an_unspent_job_does_not_divide_by_zero():
    # mutation: dividing without the s_old > 0 guard raises ZeroDivisionError
    b, st = job(5, 0)
    st.chains = [Chain(0)]
    extend(b, st, {"iterations": 5}, now=0.0, previous_status="cancelled")
    assert st.chains[0].reheated_at == 0.0


def test_second_extension_rescales_against_the_extended_budget():
    # Review Focus 1. 10/10 spent, +10: s 1.0 -> 0.5, 0.5 -> 0.25. Then 15/20 spent
    # (s_old 0.75), +10: s_new 15/30 = 0.5, 0.25 * 0.5 / 0.75 = 1/6.
    # mutation: computing s_old from spec.budget alone (15/10, clamped to 1.0) gives 0.125
    b, st = job(10, 10)
    st.chains = [Chain(0, reheated_at=0.5)]
    extend(b, st, {"iterations": 10}, now=0.0, previous_status="exhausted")
    assert st.chains[0].reheated_at == pytest.approx(0.25)
    st.spend.iterations = st.iteration = 15
    extend(b, st, {"iterations": 10}, now=0.0, previous_status="cancelled")
    assert st.chains[0].reheated_at == pytest.approx(1 / 6)
    assert effective_budget(b, st.budget_extensions).iterations == 30


def test_extend_leaves_the_pending_job_alone():
    # Review Focus 4. mutation: clearing pending_job on extension orphans a job C3 still bills
    b, st = job(5, 5)
    st.pending_job = {"purpose": 6, "job_id": "c3-123"}
    extend(b, st, {"iterations": 1}, now=0.0, previous_status="paused")
    assert st.pending_job == {"purpose": 6, "job_id": "c3-123"}
