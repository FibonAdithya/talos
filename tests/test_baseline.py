import json
import types

import pytest

from talos.baseline import BaselineError, cache_key, resolve_baseline
from talos.bench import FakeBench
from talos.challenges import CHALLENGES, MONOREPO_REF, hardware_class
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
    rec, template = resolve_baseline("knapsack", TR, HO, 5, fb, tmp_path, "cpu4-mem8192",
                                     mainnet=fake_mainnet())
    assert rec.name == "algo_x" and rec.adoption == 55 and rec.artifact_id
    assert [r.quality for r in rec.training] == [10, 11]
    assert [r.nonce for r in rec.holdout] == [1_000_000, 1_000_001]
    assert "solve_challenge" in template
    assert fb.compile_calls == 1 and fb.score_calls == 2


def test_cache_hit_skips_bench(tmp_path):
    # mutation: ignoring the cache re-measures and charges the user twice
    fb = FakeBench(lambda ch, files, ns: [1 for _ in ns.nonces()])
    resolve_baseline("knapsack", TR, HO, 5, fb, tmp_path, "cpu4-mem8192", mainnet=fake_mainnet())
    resolve_baseline("knapsack", TR, HO, 5, fb, tmp_path, "cpu4-mem8192", mainnet=fake_mainnet())
    assert fb.compile_calls == 1 and fb.score_calls == 2


def test_cache_key_changes_with_fuel_and_hardware():
    # mutation: cache_key ignoring fuel or hardware_class would collide keys across
    # different measurement conditions and serve a stale/mismatched cached baseline
    a = cache_key("knapsack", "ref", "algo", TR, HO, 5, "cpu4-mem8192")
    assert a != cache_key("knapsack", "ref", "algo", TR, HO, 6, "cpu4-mem8192")
    assert a != cache_key("knapsack", "ref", "algo", TR, HO, 5, "gpu-L40S")


def test_no_top_algorithm_is_an_error(tmp_path):
    # mutation: proceeding with top=None would crash later with an unhelpful traceback
    # instead of a clear BaselineError naming the missing mainnet algorithm
    fb = FakeBench(lambda ch, files, ns: [1])
    with pytest.raises(BaselineError):
        resolve_baseline("knapsack", TR, HO, 5, fb, tmp_path, "cpu4-mem8192", mainnet=fake_mainnet(top=None))


def test_baseline_compile_failure_is_an_error_with_output(tmp_path):
    # mutation: swallowing the compile error or omitting the raw compiler output would
    # violate spec §9, which requires the raw output on a baseline compile failure
    fb = FakeBench(lambda ch, files, ns: [1], compile_ok=lambda f: False)
    with pytest.raises(BaselineError) as ei:
        resolve_baseline("knapsack", TR, HO, 5, fb, tmp_path, "cpu4-mem8192", mainnet=fake_mainnet())
    assert "E0308" in str(ei.value)


def test_corrupt_cache_is_remeasured(tmp_path):
    # mutation: letting JSONDecodeError escape aborts the job on a half-written cache file
    key = cache_key("knapsack", MONOREPO_REF, "algo_x", TR, HO, 5, "cpu4-mem8192")
    cache_file = tmp_path / "knapsack" / f"{key}.json"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text("{not json")

    fb = FakeBench(lambda ch, files, ns: [10 + n for n in ns.nonces()])
    rec, _template = resolve_baseline("knapsack", TR, HO, 5, fb, tmp_path, "cpu4-mem8192",
                                      mainnet=fake_mainnet())
    assert fb.compile_calls == 1 and fb.score_calls == 2
    assert [r.quality for r in rec.training] == [10, 11]
    assert json.loads(cache_file.read_text())["name"] == "algo_x"


def test_all_error_baseline_is_refused_and_not_cached(tmp_path):
    # mutation: accepting an all-error baseline burns the budget on unscoreable iterations --
    # every candidate is then compared against nothing and fails at scoring until the cap bites
    fb = FakeBench(lambda ch, files, ns: [None for _ in ns.nonces()])
    with pytest.raises(BaselineError) as ei:
        resolve_baseline("knapsack", TR, HO, 5, fb, tmp_path, "cpu4-mem8192",
                         mainnet=fake_mainnet())
    msg = str(ei.value)
    assert "knapsack" in msg and "fuel 5" in msg and "'t'" in msg and "no_solution" in msg
    assert not list(tmp_path.rglob("*.json"))  # an unusable baseline is never cached


def test_zero_quality_baseline_is_refused(tmp_path):
    # mutation: checking only `any ok` accepts a baseline that scores 0 everywhere, against
    # which every relative delta is a division by zero
    fb = FakeBench(lambda ch, files, ns: [0 for _ in ns.nonces()])
    with pytest.raises(BaselineError):
        resolve_baseline("knapsack", TR, HO, 5, fb, tmp_path, "cpu4-mem8192",
                         mainnet=fake_mainnet())


def test_partly_erroring_baseline_is_accepted(tmp_path):
    # mutation: requiring every nonce to score would reject a usable baseline; TIG algorithms
    # legitimately fail some nonces
    fb = FakeBench(lambda ch, files, ns: [None, 10])
    rec, _ = resolve_baseline("knapsack", TR, HO, 5, fb, tmp_path, "cpu4-mem8192",
                              mainnet=fake_mainnet())
    assert [r.ok for r in rec.training] == [False, True]


def test_hardware_class_separates_cpu_memory_and_gpu():
    # mutation: f"cpu{cpu}" alone lets a memory change reuse a cached baseline measured with a
    # different memory (and a different price per second)
    import dataclasses
    knapsack = CHALLENGES["knapsack"]
    assert hardware_class(knapsack) == f"cpu{knapsack.cpu}-mem{knapsack.memory_mib}"
    bigger = dataclasses.replace(knapsack, memory_mib=knapsack.memory_mib * 2)
    assert hardware_class(bigger) != hardware_class(knapsack)
    assert hardware_class(CHALLENGES["hypergraph"]) == "gpu-L40S"
