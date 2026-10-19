import math
import pytest
from talos.budget import Budget, Spend
from talos.lineage import (AnnealSettings, Chain, Node, ROOT, accept_probability, chain_for,
                           decide, draw, spent_fraction, temperature)

S = AnnealSettings()


def test_chains_take_turns():
    # mutation: `iteration % chains` puts iteration 1 on chain 1
    assert [chain_for(n, 3) for n in range(1, 8)] == [0, 1, 2, 0, 1, 2, 0]
    assert chain_for(5, 1) == 0


def test_temperature_cools_linearly_to_floor():
    # mutation: dropping the floor term cools to 0, which rejects every worse candidate late on
    assert temperature(S, 0.0, 0.0) == pytest.approx(0.05)
    assert temperature(S, 0.5, 0.0) == pytest.approx(0.0275)
    assert temperature(S, 1.0, 0.0) == pytest.approx(0.005)


def test_reheat_restarts_cooling_over_the_remaining_budget():
    # mutation: ignoring reheated_at leaves T at the cooled value after a reheat
    assert temperature(S, 0.6, 0.6) == pytest.approx(0.05)
    assert temperature(S, 0.8, 0.6) == pytest.approx(0.0275)
    assert temperature(S, 1.0, 1.0) == pytest.approx(0.005)  # no division by zero at the end


def test_accept_probability():
    assert accept_probability(0.05, 0.05) == pytest.approx(math.exp(-1))
    assert accept_probability(0.0, 0.05) == 1.0
    assert accept_probability(-0.1, 0.05) == 1.0
    # mutation: exp(-drop / t) with t == 0 raises ZeroDivisionError
    assert accept_probability(0.01, 0.0) == 0.0


def test_spent_fraction_takes_the_largest_capped_dimension():
    b = Budget(usd=10, hours=None, iterations=20, compute_usd=None)
    assert spent_fraction(b, Spend(started_at=0.0, llm_usd=2.5, iterations=10), 0.0) == 0.5
    h = Budget(usd=None, hours=2, iterations=None, compute_usd=None)
    assert spent_fraction(h, Spend(started_at=0.0), 3600.0) == pytest.approx(0.5)
    # mutation: `if budget.compute_usd:` treats a 0 cap as no cap and returns 0.0
    z = Budget(usd=None, hours=None, iterations=5, compute_usd=0.0)
    assert spent_fraction(z, Spend(started_at=0.0), 0.0) == 1.0
    # clamped: an overshoot never gives a fraction above 1
    assert spent_fraction(b, Spend(started_at=0.0, llm_usd=50.0), 0.0) == 1.0


def test_draw_is_deterministic_and_keyed_by_chain_and_iteration():
    # mutation: random.random() makes a resumed job decide differently
    assert draw("j", 0, 3) == draw("j", 0, 3)
    assert 0.0 <= draw("j", 0, 3) < 1.0
    assert len({draw("j", c, n) for c in range(3) for n in range(1, 30)}) == 87


def test_decide():
    assert decide(None, -0.5, 0.05, 0.99) == (True, None)      # first scored node on a chain
    assert decide(-0.2, -0.1, 0.05, 0.99) == (True, None)      # improves on its parent
    # 0.0 - (-0.05) is exactly 0.05, so decide's p equals this p bit for bit. (-0.1 - -0.15 is
    # 0.04999999999999999, whose p is one ulp above exp(-1): a correct `u < p` would accept.)
    p = accept_probability(0.05, 0.05)
    assert p == math.exp(-1.0)
    assert decide(0.0, -0.05, 0.05, p - 1e-9) == (True, pytest.approx(p))
    # mutation: `u <= p` accepts a draw exactly at p
    assert decide(0.0, -0.05, 0.05, p) == (False, pytest.approx(p))


def test_round_trips():
    n = Node(id=4, chain=1, parent=2, family="lp", title="t", outcome="accepted", score=-0.1,
             tracks={"a": -0.1}, metered_score=-0.09, accepted=True, probability=0.4,
             draw=0.2, temperature=0.03, best=True)
    assert Node.from_dict(n.to_dict()) == n
    assert n.rank_score() == -0.09
    c = Chain(index=2, current=4, best=4, best_score=-0.09, stall=1, reheated_at=0.3)
    assert Chain.from_dict(c.to_dict()) == c
    assert AnnealSettings.from_dict(S.to_dict()) == S
    assert ROOT == 0


def test_no_drop_is_certain_even_when_frozen():
    # mutation: `drop < 0` makes a tie at t == 0 return 0.0
    assert accept_probability(0.0, 0.0) == 1.0


def test_draw_stays_below_one_for_every_key():
    # mutation: dividing by 2**63 gives draws up to 2.0
    assert all(0.0 <= draw("job", c, n) < 1.0 for c in range(4) for n in range(1, 200))
