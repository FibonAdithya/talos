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
    Needs either a C3 API key or `c3 login`, and about £0.05 of credit; takes about 20 minutes."""
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
    # With a C3 API key in .talos/secrets.json or C3_API_KEY this runs over MCP; with no key it
    # runs over the `c3` CLI and needs `c3 login`. Print which, so the run's evidence says so.
    from pathlib import Path

    from talos.config import ConfigError, load, resolve_c3_api_key
    try:
        cfg = load(Path.cwd())
    except ConfigError:
        cfg = None  # no `talos setup` here: C3_API_KEY alone still selects MCP
    key = resolve_c3_api_key(cfg)
    b = C3Bench(tmp_path, api_key=key)
    print({"transport": b._t.name, "keyed": bool(key)})
    # the CPU capacity probe, as `talos run` does it: a `true` job per profile until one starts
    print({"hardware": b.select_hardware(ch)})
    r = b.evaluate(EvalRequest(ch, files, tr, ho, info.max_fuel, None, CHALLENGES[ch].beat))
    assert r.compile.ok, r.compile.output[-3000:]
    assert len(r.training) == 2 and r.holdout is not None and len(r.holdout) == 2
    assert any(x.ok and x.quality > 0 for x in r.training), [x.to_dict() for x in r.training]
    assert r.holdout_reason == "forced"
    assert b.cost_mark() < 0.10 * 1.35
    print({"cost_usd_estimate": b.cost_mark(), "training": [x.to_dict() for x in r.training]})


def test_local_knapsack_job(tmp_path):
    """Manual: one real local Docker job. Run:
    TALOS_LIVE_BACKEND=local .venv/bin/pytest -m live tests/test_live.py -k local -s
    Needs Docker. Costs time, not money: the first run pulls a 13 GB image and does one warm-up
    build (MEASURED 2026-09-23: 12 minutes on 16 cores); the job is one candidate build (about
    as long: the image re-instruments every dependency on each build) plus 4 nonces."""
    if os.environ.get("TALOS_LIVE_BACKEND") != "local":
        pytest.skip("set TALOS_LIVE_BACKEND=local")
    import time

    from talos.bench import EvalRequest
    from talos.c3_bench import C3Bench
    from talos.c3_jobdir import LocalSettings
    from talos.challenges import CHALLENGES
    from talos.local_transport import DockerTransport, prepare
    ch = os.environ.get("TALOS_LIVE_CHALLENGE", "knapsack")
    info = mainnet.fetch_challenge_info(ch)
    name, _algorithm_id, _adoption = mainnet.top_algorithm(ch)
    files = mainnet.fetch_algorithm_files(ch, name)
    tr, ho = draw_nonce_sets(info.tracks[:1], new_rand_hash(), training_count=2, holdout_count=2)
    t0 = time.monotonic()
    gpu = prepare(ch)
    t_prep = time.monotonic() - t0
    local = LocalSettings(cpus=os.cpu_count() or 1, memory_gib=8)
    b = C3Bench(tmp_path, transport=DockerTransport(), local=local, usd_per_hour=0.0,
                poll_s=5.0)
    t1 = time.monotonic()
    r = b.evaluate(EvalRequest(ch, files, tr, ho, info.max_fuel, None, CHALLENGES[ch].beat))
    t_job = time.monotonic() - t1
    assert r.compile.ok, r.compile.output[-3000:]
    assert len(r.training) == 2 and r.holdout is not None and len(r.holdout) == 2
    assert any(x.ok and x.quality > 0 for x in r.training), [x.to_dict() for x in r.training]
    assert r.holdout_reason == "forced" and b.cost_mark() == 0.0
    art = next((tmp_path / "local" / "adhoc").glob("talos-*/artifacts/results.json"))
    if os.name != "nt":
        # copied out by this process, so owned by this user, never by root
        assert art.stat().st_uid == os.getuid(), "artifacts must belong to the user"
    print({"gpu": gpu, "prepare_s": round(t_prep), "job_s": round(t_job),
           "training": [x.to_dict() for x in r.training]})
