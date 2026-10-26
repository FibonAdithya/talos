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


@pytest.mark.parametrize("field,value", [("ratio", "NaN"), ("ratio", "Infinity"),
                                         ("ratio", "0.0"), ("ratio", "-1.0"),
                                         ("margin", "NaN"), ("margin", "0"), ("margin", "-0.5")])
def test_load_treats_a_non_positive_or_non_finite_ratio_or_margin_as_missing(tmp_path, field,
                                                                            value):
    # json.loads accepts NaN and Infinity; float() accepts both and every non-positive value
    # mutation: keeping the record gives a zero, negative or NaN budget (NaN raises in round)
    rec = {"ratio": "1.5", "margin": "0.8"}
    rec[field] = value
    p = tmp_path / "knapsack" / "k1.json"
    p.parent.mkdir(parents=True)
    p.write_text('{"version": 1, "demotions": {}, "tracks": {"a": {"ratio": %s, "margin": %s, '
                 '"misses": 0}}}' % (rec["ratio"], rec["margin"]), encoding="utf-8")
    assert calibration.load(tmp_path, "knapsack", "k1") is None


def test_load_keeps_a_valid_record(tmp_path):
    p = tmp_path / "knapsack" / "k1.json"
    p.parent.mkdir(parents=True)
    p.write_text('{"version": 1, "demotions": {}, "tracks": {"a": {"ratio": 1.5, "margin": 0.8, '
                 '"misses": 0}}}', encoding="utf-8")
    assert calibration.load(tmp_path, "knapsack", "k1")["tracks"]["a"]["ratio"] == 1.5


def test_note_demotion_counts_by_reason():
    rec = calibration.new_record({"a": 1.0})
    calibration.note_demotion(rec, "nondeterministic")
    calibration.note_demotion(rec, "nondeterministic")
    assert rec["demotions"] == {"nondeterministic": 2}


def test_concurrent_counts_on_one_record_are_not_lost(tmp_path, monkeypatch):
    # Issue #42 case 1: two jobs demoting at once each load, add, and save. The slow load
    # makes every thread read before any writes, so without the lock all but one count is
    # overwritten.
    # mutation: count_validation without _locked ends with {"nondeterministic": 1}
    import threading
    import time
    calibration.save(tmp_path, "knapsack", "k1", calibration.new_record({"a": 1.0}))
    real_load = calibration.load

    def slow_load(*a):
        rec = real_load(*a)
        time.sleep(0.05)
        return rec

    monkeypatch.setattr(calibration, "load", slow_load)
    threads = [threading.Thread(target=calibration.count_validation,
                                args=(tmp_path, "knapsack", "k1", f"job{i}:1",
                                      "nondeterministic", []))
               for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    rec = real_load(tmp_path, "knapsack", "k1")
    assert rec["demotions"] == {"nondeterministic": 6}
    assert sorted(rec["counted"]) == [f"job{i}:1" for i in range(6)]


def test_a_validation_counted_once_is_not_counted_again(tmp_path):
    # Issue #42 case 2: a resume validates the same iteration again after a kill between the
    # record's save and state.json's. The repeat counts nothing but still reports that the
    # first count lowered the margin, so the resumed job recomputes its budgets.
    # mutation: dropping the token check counts a fourth miss (misses 1, demotions 4)
    # mutation: returning False for a repeat leaves the resumed job on the old budgets
    calibration.save(tmp_path, "knapsack", "k1", calibration.new_record({"a": 1.0}))
    for i in (1, 2):
        _, changed = calibration.count_validation(tmp_path, "knapsack", "k1", f"j:{i}",
                                                  "fuel_proxy_miss", ["a"])
        assert changed is False
    rec, changed = calibration.count_validation(tmp_path, "knapsack", "k1", "j:3",
                                                "fuel_proxy_miss", ["a"])
    assert changed is True and rec["tracks"]["a"]["margin"] == pytest.approx(0.7)
    rec, changed = calibration.count_validation(tmp_path, "knapsack", "k1", "j:3",
                                                "fuel_proxy_miss", ["a"])
    assert changed is True
    stored = calibration.load(tmp_path, "knapsack", "k1")
    assert stored["demotions"] == {"fuel_proxy_miss": 3}
    assert stored["tracks"]["a"] == {"ratio": 1.0, "margin": pytest.approx(0.7), "misses": 0}


def test_count_validation_ignores_tracks_the_record_lacks_and_a_missing_record(tmp_path):
    # mutation: record_miss on an unknown track raises KeyError out of the loop
    assert calibration.count_validation(tmp_path, "knapsack", "k1", "j:1", "x", ["a"]) is None
    calibration.save(tmp_path, "knapsack", "k1", calibration.new_record({"a": 1.0}))
    rec, changed = calibration.count_validation(tmp_path, "knapsack", "k1", "j:1",
                                                "fuel_proxy_miss", ["a", "zz"])
    assert changed is False and rec["tracks"]["a"]["misses"] == 1 and "zz" not in rec["tracks"]


def test_counted_tokens_keep_only_the_newest(tmp_path):
    # A resume needs only its own last token; the cap keeps the record from growing forever.
    # mutation: no cap leaves COUNTED_KEEP + 2 tokens; dropping the newest instead of the
    # oldest loses the token a resume looks for
    calibration.save(tmp_path, "knapsack", "k1", calibration.new_record({"a": 1.0}))
    n = calibration.COUNTED_KEEP + 2
    for i in range(n):
        calibration.count_validation(tmp_path, "knapsack", "k1", f"j:{i}", "x", [])
    rec = calibration.load(tmp_path, "knapsack", "k1")
    assert list(rec["counted"]) == [f"j:{i}" for i in range(2, n)]
    assert rec["demotions"] == {"x": n}


def test_a_malformed_counted_field_is_reset_not_fatal(tmp_path):
    # mutation: keeping a list here makes `token in counted` a list scan and the store a
    # TypeError
    rec = calibration.new_record({"a": 1.0})
    rec["counted"] = ["j:1"]
    calibration.save(tmp_path, "knapsack", "k1", rec)
    assert calibration.load(tmp_path, "knapsack", "k1")["counted"] == {}
    rec, _ = calibration.count_validation(tmp_path, "knapsack", "k1", "j:1", "x", [])
    assert rec["demotions"] == {"x": 1}


def test_create_keeps_a_record_another_job_saved_first(tmp_path):
    # Two jobs measuring the same key at once: the second must not overwrite the first's
    # record, which may already hold counted misses.
    # mutation: create() that always saves returns and stores the ratio-2.0 record
    first = calibration.new_record({"a": 1.0})
    first["tracks"]["a"]["misses"] = 2
    calibration.save(tmp_path, "knapsack", "k1", first)
    got = calibration.create(tmp_path, "knapsack", "k1", calibration.new_record({"a": 2.0}))
    assert got == first and calibration.load(tmp_path, "knapsack", "k1") == first
    fresh = calibration.new_record({"b": 3.0})
    assert calibration.create(tmp_path, "knapsack", "k2", fresh) == fresh
    assert calibration.load(tmp_path, "knapsack", "k2") == fresh
