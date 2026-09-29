import pytest

from talos import calibration
from talos.types import NonceResult


def m(track, nonce, fuel):
    return NonceResult(track, nonce, True, 1, 1, None, fuel_consumed=fuel)


def n(track, nonce, us):
    return NonceResult(track, nonce, True, 1, 1, None, solve_us=us)


METERED = [m("a", 0, 1000), m("a", 1, 2000), m("a", 2, 4000)]
NATIVE = [n("a", 0, 10_000), n("a", 1, 30_000), n("a", 2, 400_000)]


def test_ratio_is_the_median_of_microseconds_per_fuel():
    # per nonce: 10.0, 15.0, 100.0 us per fuel. The median is 15.0; the mean would be 41.67
    # mutation: mean instead of median; ratio inverted (fuel per us: median 0.0667)
    assert calibration.track_ratios(METERED, NATIVE) == {"a": 15.0}


def test_budget_is_ratio_times_fuel_times_margin_in_microseconds():
    rec = calibration.new_record({"a": 15.0})
    # 15.0 us/fuel x 100_000 fuel x 0.8 = 1_200_000 us, worked by hand
    # mutation: margin dropped -> 1_500_000; ceil on a float product -> 1_200_001
    assert calibration.budgets_us(rec, 100_000) == {"a": 1_200_000}


def test_a_tiny_budget_is_raised_to_the_floor():
    rec = calibration.new_record({"a": 1.0})
    # 1.0 x 1000 x 0.8 = 800 us, under the 1 s floor
    # mutation: floor dropped leaves an 800 us budget that times out on timer noise alone
    assert calibration.budgets_us(rec, 1000) == {"a": 1_000_000}


def test_nonces_without_fuel_or_solve_time_are_skipped_and_a_track_with_none_left_is_absent():
    metered = [m("a", 0, 0), m("a", 1, None), m("b", 0, 1000)]
    native = [n("a", 0, 5000), n("a", 1, 5000), n("b", 0, None)]
    # mutation: dividing by a zero fuel_consumed raises ZeroDivisionError; keeping the track
    # with ratio 0 gives it the 1 s floor instead of the metered fallback
    assert calibration.track_ratios(metered, native) == {}


def test_a_nonce_pairs_only_with_its_own_track_and_nonce():
    metered = [m("a", 0, 1000), m("b", 0, 10)]
    native = [n("a", 0, 2000), n("b", 1, 99_999)]
    # mutation: pairing by position or by nonce alone puts b's solve time on a's fuel
    assert calibration.track_ratios(metered, native) == {"a": 2.0}


def test_a_pair_whose_metered_run_hit_the_fuel_limit_is_ignored():
    # The metered run saved q=100 at fuel 1000 and was cut off; the unbudgeted native run kept
    # going and saved q=150 at 9000 us. The two numbers are from different saves.
    # mutation: pairing them anyway gives a ratio of 9.0 for this track
    metered = [NonceResult("a", 0, True, 100, 1, None, fuel_consumed=1000, limit_hit=True)]
    native = [NonceResult("a", 0, True, 150, 1, None, solve_us=9000)]
    assert calibration.track_ratios(metered, native) == {}


def test_only_clean_pairs_with_equal_quality_are_kept():
    metered = [NonceResult("a", 0, True, 100, 1, None, fuel_consumed=1000),
               NonceResult("a", 1, True, 100, 1, None, fuel_consumed=1000),
               NonceResult("a", 2, True, 100, 1, None, fuel_consumed=1000),
               NonceResult("a", 3, False, None, 1, "no_solution", fuel_consumed=1000),
               NonceResult("a", 4, True, 100, 1, None, fuel_consumed=1000, limit_hit=True)]
    native = [NonceResult("a", 0, True, 100, 1, None, solve_us=5000),
              NonceResult("a", 1, True, 101, 1, None, solve_us=9000),
              NonceResult("a", 2, True, 100, 1, None, solve_us=9000, limit_hit=True),
              NonceResult("a", 3, False, None, 1, "no_solution", solve_us=9000),
              NonceResult("a", 4, True, 100, 1, None, solve_us=9000)]
    # nonce 0 alone is clean: 5000 / 1000 = 5.0. Nonce 1 drifted in quality, nonce 2 hit the
    # native limit, nonce 3 failed on both (equal None qualities), nonce 4 hit the metered
    # limit at the same quality.
    # mutation: dropping any one condition adds a 9.0 and moves the median off 5.0
    assert calibration.track_ratios(metered, native) == {"a": 5.0}


def test_three_misses_lower_the_margin_once_and_the_floor_holds():
    rec = calibration.new_record({"a": 1.0})
    # mutation: `>` instead of `>=` needs a fourth miss
    assert [calibration.record_miss(rec, "a") for _ in range(3)] == [False, False, True]
    assert rec["tracks"]["a"]["margin"] == pytest.approx(0.7)
    assert rec["tracks"]["a"]["misses"] == 0
    for _ in range(3 * 10):
        calibration.record_miss(rec, "a")
    # mutation: no floor drives the margin to zero or below, a budget every candidate fails
    assert rec["tracks"]["a"]["margin"] == pytest.approx(0.3)


def test_the_key_changes_with_hardware_baseline_and_hyperparameters():
    base = calibration.calibration_key("hypergraph", "c3-l40", {"mod.rs": "a"}, None)
    assert base == calibration.calibration_key("hypergraph", "c3-l40", {"mod.rs": "a"}, None)
    # mutation: dropping any one of these reuses an L40 budget on an A100, or after the
    # baseline or its hyperparameters changed
    assert base != calibration.calibration_key("hypergraph", "c3-a100", {"mod.rs": "a"}, None)
    assert base != calibration.calibration_key("hypergraph", "c3-l40", {"mod.rs": "b"}, None)
    assert base != calibration.calibration_key("hypergraph", "c3-l40", {"mod.rs": "a"},
                                               {"t": {"x": 1}})
    # a track mapped to None is the same run as no map (baseline.effective_hyperparameters)
    assert base == calibration.calibration_key("hypergraph", "c3-l40", {"mod.rs": "a"},
                                               {"t": None})


def test_the_key_depends_on_the_pins(monkeypatch):
    base = calibration.calibration_key("knapsack", "hw", {"mod.rs": "a"}, None)
    monkeypatch.setattr(calibration, "MONOREPO_REF", "other")
    assert calibration.calibration_key("knapsack", "hw", {"mod.rs": "a"}, None) != base


def test_the_key_changes_with_the_native_runner(monkeypatch):
    # The ratio is native solve time over metered fuel; a new runner (template, toolchain,
    # build profile) times differently.
    # mutation: leaving the runner digest out of the payload reuses a stale ratio
    from talos import native_runner
    base = calibration.calibration_key("knapsack", "hw", {"mod.rs": "a"}, None)
    monkeypatch.setattr(native_runner, "_CPU_RUN", native_runner._CPU_RUN + " ")
    assert calibration.calibration_key("knapsack", "hw", {"mod.rs": "a"}, None) != base


def test_save_then_load_round_trips(tmp_path):
    rec = calibration.new_record({"a": 15.0})
    calibration.save(tmp_path, "knapsack", "k1", rec)
    assert calibration.load(tmp_path, "knapsack", "k1") == rec
    assert calibration.load(tmp_path, "knapsack", "k2") is None


def test_load_treats_a_corrupt_record_as_missing(tmp_path):
    p = tmp_path / "knapsack" / "k1.json"
    p.parent.mkdir(parents=True)
    p.write_text('{"tracks": {"a": {"ratio": 1.0,')   # truncated mid-write
    assert calibration.load(tmp_path, "knapsack", "k1") is None
    p.write_text('{"tracks": {"a": {}}}')             # no ratio
    # mutation: a KeyError here tracebacks out of `talos run` instead of re-measuring
    assert calibration.load(tmp_path, "knapsack", "k1") is None


def test_note_demotion_counts_by_reason():
    rec = calibration.new_record({"a": 1.0})
    calibration.note_demotion(rec, "nondeterministic")
    calibration.note_demotion(rec, "nondeterministic")
    assert rec["demotions"] == {"nondeterministic": 2}
