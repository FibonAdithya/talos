import pytest

from talos.budget import Budget, Spend, exhausted


def test_zero_budget_is_exhausted_before_first_call():
    # mutation: `if budget.usd:` treats 0 as unset and never stops
    b = Budget(usd=0.0, hours=None, iterations=None, modal_usd=None)
    assert exhausted(b, Spend(started_at=0.0), now=0.0) == "usd"


def test_boundary_at_cap_is_exhausted():
    # mutation: `>` instead of `>=` lets spend exactly at the cap continue
    b = Budget(usd=1.0, hours=None, iterations=3, modal_usd=None)
    assert exhausted(b, Spend(llm_usd=1.0, started_at=0.0), now=0.0) == "usd"
    assert exhausted(b, Spend(llm_usd=0.99, iterations=3, started_at=0.0), now=0.0) == "iterations"
    assert exhausted(b, Spend(llm_usd=0.99, iterations=2, started_at=0.0), now=0.0) is None


def test_hours_uses_clock_not_wall():
    # mutation: reading time.time() inside makes this untestable and flaky
    b = Budget(usd=None, hours=1.0, iterations=None, modal_usd=None)
    assert exhausted(b, Spend(started_at=100.0), now=100.0 + 3599) is None
    assert exhausted(b, Spend(started_at=100.0), now=100.0 + 3600) == "hours"


def test_all_none_is_invalid():
    # mutation: an `all(...)` check reading `or` instead of `and`, or a missing
    # call to validate(), would let a job with no cap at all reach the wizard
    with pytest.raises(ValueError):
        Budget(usd=None, hours=None, iterations=None, modal_usd=None).validate()
    Budget(usd=None, hours=2.0, iterations=None, modal_usd=None).validate()


def test_modal_usd_alone_is_not_a_budget():
    # mutation: counting modal_usd as a budget lets a run start with unbounded LLM spend
    with pytest.raises(ValueError):
        Budget(usd=None, hours=None, iterations=None, modal_usd=5.0).validate()


def test_zero_cap_is_a_valid_budget():
    # README: "zero is a valid, real cap, not unset". validate() must accept it so that exhausted()
    # can then refuse the first call.
    # mutation: `all(not v ...)` instead of `all(v is None ...)` refuses `--budget-usd 0` and
    # `--budget-iterations 0` as "no budget dimension set"
    Budget(usd=0.0, hours=None, iterations=None, modal_usd=None).validate()
    Budget(usd=None, hours=None, iterations=0, modal_usd=None).validate()
