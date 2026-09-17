"""Manual: compile and score the real mainnet top algorithm on Modal for one challenge.
Run: TALOS_LIVE_CHALLENGE=knapsack .venv/bin/pytest -m live tests/test_live.py -s
Needs `talos setup` done (Modal deployed) and network."""
import os

import pytest

from talos import mainnet
from talos.bench import ModalBench
from talos.nonces import draw_nonce_sets, new_rand_hash

pytestmark = pytest.mark.live


def test_baseline_compiles_and_scores():
    from talos.bench import EvalRequest
    from talos.challenges import CHALLENGES

    ch = os.environ.get("TALOS_LIVE_CHALLENGE", "knapsack")
    info = mainnet.fetch_challenge_info(ch)
    top = mainnet.top_algorithm(ch)
    assert top is not None, f"no adopted compiled algorithm for {ch}"
    name, _algorithm_id, adoption = top
    files = mainnet.fetch_algorithm_files(ch, name)
    bench = ModalBench()
    tr, ho = draw_nonce_sets(info.tracks[:1], new_rand_hash(), training_count=2, holdout_count=0)
    r = bench.evaluate(EvalRequest(ch, files, tr, ho, info.max_fuel, None,
                                   CHALLENGES[ch].beat))
    assert r.compile.ok, r.compile.output[-3000:]
    res = r.training
    assert len(res) == 2
    assert all(r.error != "panic" for r in res), [r.to_dict() for r in res]
    assert any(r.ok for r in res), [r.to_dict() for r in res]
    print({"algorithm": name, "adoption": adoption, "results": [r.to_dict() for r in res]})


def test_c3_knapsack_job(tmp_path):
    """Manual: one real C3 job. Run:
    TALOS_LIVE_BACKEND=c3 .venv/bin/pytest -m live tests/test_live.py -k c3 -s
    Needs `c3 login` and about £0.05 of credit; takes about 20 minutes."""
    if os.environ.get("TALOS_LIVE_BACKEND") != "c3":
        pytest.skip("set TALOS_LIVE_BACKEND=c3")
    from talos.bench import EvalRequest
    from talos.c3_bench import C3Bench
    from talos.challenges import CHALLENGES
    ch = "knapsack"
    info = mainnet.fetch_challenge_info(ch)
    name, _algorithm_id, _adoption = mainnet.top_algorithm(ch)
    files = mainnet.fetch_algorithm_files(ch, name)
    tr, ho = draw_nonce_sets(info.tracks[:1], new_rand_hash(), training_count=2, holdout_count=2)
    b = C3Bench(tmp_path)
    r = b.evaluate(EvalRequest(ch, files, tr, ho, info.max_fuel, None, CHALLENGES[ch].beat))
    assert r.compile.ok, r.compile.output[-3000:]
    assert len(r.training) == 2 and r.holdout is not None and len(r.holdout) == 2
    assert any(x.ok and x.quality > 0 for x in r.training), [x.to_dict() for x in r.training]
    assert r.holdout_reason == "forced"
    assert b.cost_mark() < 0.10 * 1.35
    print({"cost_usd_estimate": b.cost_mark(), "training": [x.to_dict() for x in r.training]})
