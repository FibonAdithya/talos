"""Manual: compile and score the real mainnet top algorithm on Modal for one challenge.
Run: TALOS_LIVE_CHALLENGE=knapsack .venv/bin/pytest -m live tests/test_live.py -s
Needs `talos setup` done (Modal deployed) and network."""
import os
import subprocess

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
    Needs Docker. Costs time, not money: the first run pulls a 13 GB image, clones the pin and
    does one warm-up build (MEASURED 2026-09-25: 90 s on 16 cores, image already pulled); the
    job is one candidate build plus 4 nonces (MEASURED 2026-09-25: 177 s for knapsack, 990 s
    for job_scheduling, whose nonces take 150 to 165 s each)."""
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


def test_c3_gpu_probe(tmp_path):
    """Manual: the C3 GPU capacity probe, for real. Run:
    TALOS_LIVE_BACKEND=c3 .venv/bin/pytest -m live tests/test_live.py -k c3_gpu_probe -s
    Submits a two-minute `true` job per class until one leaves the queue (docs/compute-backends.md
    #hardware-fallback); costs under £0.10 and takes from a minute to the capacity window per class."""
    if os.environ.get("TALOS_LIVE_BACKEND") != "c3":
        pytest.skip("set TALOS_LIVE_BACKEND=c3")
    import time
    from pathlib import Path

    from talos.c3_bench import C3Bench
    from talos.challenges import C3_GPU_CLASSES
    from talos.config import ConfigError, load, resolve_c3_api_key
    try:
        cfg = load(Path.cwd())
    except ConfigError:
        cfg = None
    # TALOS_LIVE_TRANSPORT=cli forces the `c3` CLI path even with a key configured
    key = None if os.environ.get("TALOS_LIVE_TRANSPORT") == "cli" else resolve_c3_api_key(cfg)
    b = C3Bench(tmp_path, api_key=key)
    t0 = time.monotonic()
    cls = b.select_hardware("hypergraph")
    assert cls in C3_GPU_CLASSES
    assert 0 < b.cost_mark() < 0.20
    print({"transport": b._t.name, "keyed": bool(key), "class": cls,
           "probe_s": round(time.monotonic() - t0), "cost_usd_estimate": round(b.cost_mark(), 4)})


def test_modal_gpu_probe():
    """Manual: the Modal GPU capacity probe, for real. Run:
    TALOS_LIVE_BACKEND=modal .venv/bin/pytest -m live tests/test_live.py -k modal_gpu_probe -s
    Needs the app deployed at this version (`talos setup`, or `modal deploy -m
    modal_app.talos_bench`): the probe functions are new. Costs seconds of one GPU."""
    if os.environ.get("TALOS_LIVE_BACKEND") != "modal":
        pytest.skip("set TALOS_LIVE_BACKEND=modal")
    import time

    from talos.challenges import MODAL_GPUS
    b = ModalBench()
    t0 = time.monotonic()
    gpu = b.select_hardware("hypergraph")
    assert gpu in MODAL_GPUS
    assert 0 < b.cost_mark() < 0.20
    print({"gpu": gpu, "probe_s": round(time.monotonic() - t0),
           "cost_usd_estimate": round(b.cost_mark(), 4)})


def test_claude_agent_can_edit_after_cd(tmp_path):
    """Manual: one real claude-cli call, which spends LLM tokens (the smallest model). Run:
    TALOS_LIVE_AGENT=claude-cli .venv/bin/pytest -m live tests/test_live.py -k claude_agent -s
    Agents `cd <wt>/algorithm` before editing; with relative permission rules every Edit after
    that was denied (tig-adi, 2026-09-26..29). The worktree's own settings must survive it."""
    import shutil as _shutil

    from talos.agentic import _run_claude, prepare_worktree
    from talos.prompts import PromptContext

    if os.environ.get("TALOS_LIVE_AGENT") != "claude-cli":
        pytest.skip("set TALOS_LIVE_AGENT=claude-cli")
    if _shutil.which("claude") is None:
        pytest.skip("claude CLI not on PATH")
    ctx = PromptContext(challenge="knapsack", template_rs="", direction="go", tacit="",
                        files={"mod.rs": "// hello\n"}, baseline_name="b", best_delta=0.0)
    wt = prepare_worktree(ctx, parent=tmp_path)
    mod = wt / "algorithm" / "mod.rs"
    prompt = (f"Step 1: run exactly this Bash command: cd {wt / 'algorithm'} && ls . "
              f"Step 2: use the Edit tool to replace the word hello with bye in {mod}. "
              "Use no other tools except Read. Then say DONE, or DENIED if the edit was refused.")
    model = os.environ.get("TALOS_LIVE_CLAUDE_MODEL", "claude-haiku-4-5-20251001")
    _run_claude(wt, model, prompt, 300, subprocess.run)
    out = (wt / ".talos" / "agent_stdout.txt").read_text()
    assert mod.read_text() == "// bye\n", out


def test_native_parity(tmp_path):
    """Manual: the baseline scored metered and native on the same nonces, one per track, must
    give the same quality on every nonce. Run:
    TALOS_LIVE_NATIVE=c3 TALOS_LIVE_CHALLENGE=hypergraph .venv/bin/pytest -m live \
        tests/test_live.py -k native_parity -s
    TALOS_LIVE_NATIVE is c3, modal or local. Spends real credit on c3 and modal."""
    backend = os.environ.get("TALOS_LIVE_NATIVE")
    if backend not in ("c3", "modal", "local"):
        pytest.skip("set TALOS_LIVE_NATIVE=c3|modal|local")
    from talos.bench import EvalRequest, PendingJobStore
    from talos.challenges import CHALLENGES
    from talos.cli import local_settings, make_bench
    ch = os.environ.get("TALOS_LIVE_CHALLENGE", "knapsack")
    info = mainnet.fetch_challenge_info(ch)
    name, _algorithm_id, _adoption = mainnet.top_algorithm(ch)
    files = mainnet.fetch_algorithm_files(ch, name)
    tr, _ = draw_nonce_sets(info.tracks, new_rand_hash(), training_count=1, holdout_count=0)
    if backend == "local":
        from talos.local_transport import prepare
        prepare(ch)
    bench = make_bench(backend, tmp_path, PendingJobStore.memory(),
                       c3_api_key=os.environ.get("C3_API_KEY"),
                       local=local_settings(None) if backend == "local" else None)
    bench.select_hardware(ch)
    out = {}
    for mode in ("metered", "native"):
        r = bench.evaluate(EvalRequest(ch, files, tr, [], info.max_fuel, None,
                                       CHALLENGES[ch].beat, mode=mode))
        assert r.compile.ok, r.compile.output[-3000:]
        out[mode] = {(x.track, x.nonce): x for x in r.training}
    for k, m in out["metered"].items():
        n = out["native"][k]
        assert m.quality == n.quality, (k, m.to_dict(), n.to_dict())
        assert m.fuel_consumed is not None and n.solve_us is not None
    print({"algorithm": name, "rows": {str(k): (v.to_dict(), out["native"][k].to_dict())
                                       for k, v in out["metered"].items()}})
