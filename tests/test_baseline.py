import json
import types

import pytest

from talos.baseline import BaselineError, cache_key, resolve_baseline
from talos.bench import FakeBench
from talos.challenges import MONOREPO_REF
from talos.types import NonceSet

TR = [NonceSet("t", "ab" * 32, 0, 2)]
HO = [NonceSet("t", "ab" * 32, 1_000_000, 2)]


def fake_mainnet(top=("algo_x", 55)):
    return types.SimpleNamespace(
        top_algorithm=lambda ch, **kw: top,
        fetch_algorithm_files=lambda ch, name, **kw: {"mod.rs": f"// {name}"},
        fetch_template=lambda ch, **kw: "pub fn solve_challenge(",
    )


def test_resolve_compiles_and_scores_both_sets(tmp_path):
    # mutation: swapping training/holdout order, or scoring only one set, or dropping adoption
    fb = FakeBench(lambda ch, files, ns: [10 + n for n in ns.nonces()])
    rec, template = resolve_baseline("knapsack", TR, HO, 5, fb, tmp_path, "cpu4",
                                     mainnet=fake_mainnet())
    assert rec.name == "algo_x" and rec.adoption == 55 and rec.artifact_id
    assert [r.quality for r in rec.training] == [10, 11]
    assert [r.nonce for r in rec.holdout] == [1_000_000, 1_000_001]
    assert "solve_challenge" in template
    assert fb.compile_calls == 1 and fb.score_calls == 2


def test_cache_hit_skips_bench(tmp_path):
    # mutation: ignoring the cache re-measures and charges the user twice
    fb = FakeBench(lambda ch, files, ns: [1 for _ in ns.nonces()])
    resolve_baseline("knapsack", TR, HO, 5, fb, tmp_path, "cpu4", mainnet=fake_mainnet())
    resolve_baseline("knapsack", TR, HO, 5, fb, tmp_path, "cpu4", mainnet=fake_mainnet())
    assert fb.compile_calls == 1 and fb.score_calls == 2


def test_cache_key_changes_with_fuel_and_hardware():
    # mutation: cache_key ignoring fuel or hardware_class would collide keys across
    # different measurement conditions and serve a stale/mismatched cached baseline
    a = cache_key("knapsack", "ref", "algo", TR, HO, 5, "cpu4")
    assert a != cache_key("knapsack", "ref", "algo", TR, HO, 6, "cpu4")
    assert a != cache_key("knapsack", "ref", "algo", TR, HO, 5, "l40s")


def test_no_top_algorithm_is_an_error(tmp_path):
    # mutation: proceeding with top=None would crash later with an unhelpful traceback
    # instead of a clear BaselineError naming the missing mainnet algorithm
    fb = FakeBench(lambda ch, files, ns: [1])
    with pytest.raises(BaselineError):
        resolve_baseline("knapsack", TR, HO, 5, fb, tmp_path, "cpu4", mainnet=fake_mainnet(top=None))


def test_baseline_compile_failure_is_an_error_with_output(tmp_path):
    # mutation: swallowing the compile error or omitting the raw compiler output would
    # violate spec §9, which requires the raw output on a baseline compile failure
    fb = FakeBench(lambda ch, files, ns: [1], compile_ok=lambda f: False)
    with pytest.raises(BaselineError) as ei:
        resolve_baseline("knapsack", TR, HO, 5, fb, tmp_path, "cpu4", mainnet=fake_mainnet())
    assert "E0308" in str(ei.value)


def test_corrupt_cache_is_remeasured(tmp_path):
    # mutation: letting JSONDecodeError escape aborts the job on a half-written cache file
    key = cache_key("knapsack", MONOREPO_REF, "algo_x", TR, HO, 5, "cpu4")
    cache_file = tmp_path / "knapsack" / f"{key}.json"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text("{not json")

    fb = FakeBench(lambda ch, files, ns: [10 + n for n in ns.nonces()])
    rec, _template = resolve_baseline("knapsack", TR, HO, 5, fb, tmp_path, "cpu4",
                                      mainnet=fake_mainnet())
    assert fb.compile_calls == 1 and fb.score_calls == 2
    assert [r.quality for r in rec.training] == [10, 11]
    assert json.loads(cache_file.read_text())["name"] == "algo_x"
