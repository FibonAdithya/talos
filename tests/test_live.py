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
    name, adoption = top
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
