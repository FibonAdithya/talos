import math

import pytest

from talos.challenges import BeatRule
from talos.scoring import (Bar, BundleDelta, ScoringError, TrackDelta, beats, beats_focused,
                           bundle_delta, focus_sets, fuel_proxy_misses, noise,
                           quality_mismatches,
                           improves, runtime_ratio, select, validation_failure,
                           worst_track_shortfall)
from talos.types import NonceResult, NonceSet


def R(track, nonce, q, err=None):
    return NonceResult(track=track, nonce=nonce, ok=err is None, quality=q,
                       runtime_ms=1, error=err)


BASE = [R("t1", 0, 100), R("t1", 1, 100), R("t2", 0, 200), R("t2", 1, 200)]


def test_track_and_bundle_delta():
    # mutation: using max() instead of min() for worst_rel_delta, or plain sum instead of mean()
    # for mean_rel_delta, gives the wrong aggregate across t1's +0.10 and t2's 0.0
    cand = [R("t1", 0, 110), R("t1", 1, 110), R("t2", 0, 200), R("t2", 1, 200)]
    d = bundle_delta(BASE, cand)
    by = {t.track: t for t in d.tracks}
    assert by["t1"].rel_delta == pytest.approx(0.10)
    assert by["t2"].rel_delta == pytest.approx(0.0)
    assert d.mean_rel_delta == pytest.approx(0.05)
    assert d.worst_rel_delta == pytest.approx(0.0)
    assert d.error_rate == 0.0


def test_error_scored_as_worst_observed_floored_at_zero():
    # mutation: scoring an error as None/skip inflates the candidate mean
    cand = [R("t1", 0, 110), R("t1", 1, None, "panic"), R("t2", 0, 200), R("t2", 1, 200)]
    d = bundle_delta(BASE, cand)
    by = {t.track: t for t in d.tracks}
    # worst observed on t1 across both = 100 -> error counts as 100
    assert by["t1"].cand_mean == pytest.approx(105)
    assert by["t1"].cand_errors == 1
    assert d.error_rate == pytest.approx(0.25)


def test_mismatched_nonces_raise():
    # mutation: silently zipping unequal lists compares different instances
    with pytest.raises(ScoringError):
        bundle_delta(BASE, BASE[:3])
    with pytest.raises(ScoringError):
        bundle_delta(BASE, [R("t1", 0, 1), R("t1", 5, 1), R("t2", 0, 1), R("t2", 1, 1)])


def test_beats_margin_boundary():
    # mutation: `>=` instead of `>` on the margin passes the equal case
    rule = BeatRule(margin=0.05, track_tolerance=0.0, error_ceiling=0.05)
    cand = [R("t1", 0, 110), R("t1", 1, 110), R("t2", 0, 200), R("t2", 1, 200)]
    assert not beats(BASE, cand, rule)  # mean delta exactly 0.05
    rule2 = BeatRule(margin=0.0499, track_tolerance=0.0, error_ceiling=0.05)
    assert beats(BASE, cand, rule2)


def test_default_rule_wins_on_any_improvement():
    # user decision 2026-10-02: any improvement over the baseline is a win.
    # mutation: a default margin above 0.0025 rejects the +0.25% candidate; `>=` accepts the tie
    step = [R("t1", 0, 100), R("t1", 1, 101), R("t2", 0, 200), R("t2", 1, 200)]
    assert bundle_delta(BASE, step).mean_rel_delta == pytest.approx(0.0025)
    assert beats(BASE, step, BeatRule())
    assert beats_focused(BASE, step, BeatRule(), "t1")
    assert not beats(BASE, BASE, BeatRule())
    assert not beats_focused(BASE, BASE, BeatRule(), "t1")


def test_beats_rejects_track_regression_and_errors():
    # mutation: dropping the worst-track check accepts a regression on t2
    rule = BeatRule(margin=0.01, track_tolerance=0.0, error_ceiling=0.05)
    regress = [R("t1", 0, 150), R("t1", 1, 150), R("t2", 0, 199), R("t2", 1, 199)]
    assert not beats(BASE, regress, rule)
    # mutation: dropping the error ceiling accepts a 25% error rate
    errs = [R("t1", 0, 150), R("t1", 1, 150), R("t2", 0, 250), R("t2", 1, None, "timeout")]
    assert not beats(BASE, errs, rule)


def test_zero_baseline_mean_raises():
    # mutation: dropping the `bm <= 0` guard divides by zero computing rel_delta
    zero = [R("t1", 0, 0), R("t1", 1, 0)]
    with pytest.raises(ScoringError):
        bundle_delta(zero, [R("t1", 0, 5), R("t1", 1, 5)])


H = "ab" * 32
TR2 = [NonceSet("t1", H, 0, 2), NonceSet("t2", H, 0, 2)]
HO2 = [NonceSet("t1", H, 1_000_000, 2), NonceSet("t2", H, 1_000_000, 2)]


def test_select_keeps_rows_inside_the_sets_in_input_order():
    # mutation: keying on track alone keeps held-out rows in a training slice; keying on nonce
    # alone keeps t2's rows in a t1 slice
    rows = [R("t2", 0, 1), R("t1", 1_000_000, 2), R("t1", 1, 3), R("t1", 0, 4), R("t1", 2, 5)]
    out = select(rows, [NonceSet("t1", H, 0, 2)])
    assert [(r.track, r.nonce) for r in out] == [("t1", 1), ("t1", 0)]


def test_focus_sets_narrows_training_and_appends_guards_to_holdout():
    # mutation: forgetting the guard leaves shared-code regressions on t2 unseen; using t2's
    # held-out set as the guard would compare against baseline rows the guard never measured
    tr, ho = focus_sets("t1", TR2, HO2)
    assert tr == [TR2[0]]
    assert ho == [HO2[0], TR2[1]]


def test_focus_sets_without_a_track_is_the_identity():
    # mutation: the focus path leaking into unfocused jobs changes every existing request
    assert focus_sets(None, TR2, HO2) == (TR2, HO2)


RULE = BeatRule(margin=0.005, track_tolerance=0.0, error_ceiling=0.05)


def test_beats_focused_fails_on_a_guard_regression():
    # mutation: dropping the guard check confirms a candidate that broke t2
    cand = [R("t1", 0, 120), R("t1", 1, 120), R("t2", 0, 199), R("t2", 1, 200)]
    assert not beats_focused(BASE, cand, RULE, "t1")
    # (the unfocused rule rejects this too, via worst_rel_delta; the point is that the guard
    # check, not the margin, is what fails here: the focus track is +20%)
    tol = BeatRule(margin=0.005, track_tolerance=0.01, error_ceiling=0.05)
    assert beats_focused(BASE, cand, tol, "t1")  # a 0.25% drop is inside a 1% tolerance


def test_beats_focused_applies_the_margin_to_the_focus_track_only():
    # mutation: applying the margin to the mean confirms a candidate whose focus track is flat
    # but whose guard track happened to rise
    cand = [R("t1", 0, 100), R("t1", 1, 100), R("t2", 0, 220), R("t2", 1, 220)]
    assert not beats_focused(BASE, cand, RULE, "t1")
    assert beats_focused(BASE, cand, RULE, "t2")


def test_beats_focused_counts_errors_on_the_focus_track_only():
    # mutation: using the bundle error rate lets a focus-track error rate of 50% pass because
    # 40 clean guard rows dilute it to 1/42 = 2.4%, under the 5% ceiling; ignoring errors
    # entirely passes it too (the focus track is +50% on quality)
    base = [R("t1", 0, 100), R("t1", 1, 100)] + [R("t2", n, 200) for n in range(40)]
    cand = ([R("t1", 0, 200), R("t1", 1, None, "panic")]
            + [R("t2", n, 200) for n in range(40)])
    assert bundle_delta(base, cand).error_rate < RULE.error_ceiling  # the diluted rate
    assert not beats_focused(base, cand, RULE, "t1")


def test_beats_focused_on_a_single_track_equals_beats():
    # mutation: the two rules diverging when there is nothing to guard
    # 200 -> 201 is rel_delta 0.005 exactly, the margin: a `>` in one rule and `>=` in the other
    # diverges here and nowhere else in this list
    base = [R("t1", 0, 200), R("t1", 1, 200)]
    for q in (200, 201, 202, 220):
        cand = [R("t1", 0, q), R("t1", 1, q)]
        assert beats_focused(base, cand, RULE, "t1") == beats(base, cand, RULE)
    assert not beats_focused(base, [R("t1", 0, 201), R("t1", 1, 201)], RULE, "t1")
    assert beats_focused(base, [R("t1", 0, 202), R("t1", 1, 202)], RULE, "t1")


def test_runtime_ratio_is_the_slowest_track_relative_to_baseline():
    # iteration 2 of run 20260916-095103 ran 14x slower on one track and the model never
    # heard. mutation: averaging across tracks hides a single slow track behind fast ones
    def T(track, nonce, ms):
        return NonceResult(track=track, nonce=nonce, ok=True, quality=1, runtime_ms=ms)
    base = [T("a", 0, 10), T("a", 1, 10), T("b", 0, 100), T("b", 1, 100)]
    cand = [T("a", 0, 10), T("a", 1, 10), T("b", 0, 1400), T("b", 1, 1400)]
    assert runtime_ratio(base, cand) == pytest.approx(14.0)
    assert runtime_ratio(base, base) == pytest.approx(1.0)
    # a baseline track with no measured runtime cannot be a divisor
    zero = [T("a", 0, 0), T("a", 1, 0)]
    assert runtime_ratio(zero, [T("a", 0, 5), T("a", 1, 5)]) == pytest.approx(1.0)


# The file defines BASE (two tracks, t1/t2) and RULE at module level, and the existing tests
# read them at call time. These names must not collide: a second `BASE = ...` below would
# silently replace the first and break every existing bundle_delta/beats test.
def vrows(qs, track="t", error=None, limit=()):
    return [NonceResult(track, i, q is not None, q, 1, None if q is not None else error,
                        limit_hit=i in limit)
            for i, q in enumerate(qs)]


VBASE = vrows([100, 100, 100, 100])
# a non-zero margin, so a stepping stone can improve without beats()
VRULE = BeatRule(margin=0.005, track_tolerance=0.0, error_ceiling=0.05)
# the bar before any best on a mainnet start: the baseline against itself
BAR0 = Bar(0.0, 0.0)


def test_validation_passes_a_stepping_stone_short_of_the_margin():
    # mean 100.25 vs 100: +0.25%, short of the 0.5% margin, above a best of 0.0
    # mutation: requiring beats() here (spec §5.3 as written) demotes every stepping stone
    step = vrows([100, 100, 100, 101])
    reason, d = validation_failure(True, step, step, VBASE, BAR0, VRULE)
    assert reason is None and d.mean_rel_delta == pytest.approx(0.0025)


def test_each_failure_reason_in_order():
    good = vrows([110, 110, 110, 110])
    assert validation_failure(False, [], good, VBASE, BAR0, VRULE)[0] == \
        "native_metered_build_mismatch"
    oof = vrows([None, 110, 110, 110], error="out_of_fuel", limit=(0,))
    # mutation: out_of_fuel read as an ordinary error lets a fuel-proxy miss through as long
    # as the error rate stays under the ceiling; checked before the quality comparison
    assert validation_failure(True, oof, good, VBASE, BAR0, VRULE)[0] == "fuel_proxy_miss"
    drift = vrows([110, 110, 110, 111])
    # mutation: comparing means instead of nonce by nonce hides a +1/-1 pair of drifts
    assert validation_failure(True, drift, good, VBASE, BAR0, VRULE)[0] == "nondeterministic"
    pair = vrows([111, 109, 110, 110])
    assert validation_failure(True, pair, good, VBASE, BAR0, VRULE)[0] == "nondeterministic"
    worse = vrows([100, 100, 100, 100])
    assert validation_failure(True, worse, worse, VBASE, BAR0, VRULE)[0] == "not_improved"
    timeouts = vrows([None, 110, 110, 110], error="timeout")
    # 1 error in 4 nonces is a rate of 0.25, above the 0.05 ceiling (by hand)
    assert validation_failure(True, timeouts, good, VBASE, BAR0, VRULE)[0] == "error_ceiling"
    short = vrows([110, 110])
    assert validation_failure(True, short, short, VBASE, BAR0, VRULE)[0] == "unscoreable"


def test_a_metered_run_out_of_fuel_with_a_verified_solution_is_a_proxy_miss():
    # tig-runtime exited 87 on nonce 0 after saving a solution that verified: ok, error None
    metered = vrows([108, 110, 110, 110], limit=(0,))
    native = vrows([110, 110, 110, 110])
    # mutation: detecting misses by `error == "out_of_fuel"` alone calls this
    # "nondeterministic" (108 vs 110), and the margin never adapts
    assert validation_failure(True, metered, native, VBASE, BAR0, VRULE)[0] == "fuel_proxy_miss"
    assert [(r.track, r.nonce) for r in fuel_proxy_misses(metered, native)] == [("t", 0)]


def test_an_out_of_fuel_row_without_limit_hit_is_still_a_proxy_miss():
    # A Modal app deployed before limit_hit existed sends error="out_of_fuel", limit_hit False.
    metered = vrows([None, 110, 110, 110], error="out_of_fuel")
    native = vrows([110, 110, 110, 110])
    assert not metered[0].limit_hit
    # mutation: detecting misses by limit_hit alone reads this as an ordinary error, which
    # the 0.05 ceiling then reports as "error_ceiling" and the margin never adapts
    assert validation_failure(True, metered, native, VBASE, BAR0, VRULE)[0] == "fuel_proxy_miss"


def test_a_native_timeout_is_a_cutoff_not_a_proxy_miss():
    # The candidate's outer timeout (max(60 s, 3 x the baseline's metered time)) can fire
    # before a budget calibrated from the full fuel does: the native run never reached its
    # budget, so the budget did not stand in for more fuel than TIG gives.
    metered = vrows([None, 110, 110, 110], error="out_of_fuel", limit=(0,))
    native = vrows([None, 110, 110, 110], error="timeout")
    # mutation: reading only native limit_hit charges this nonce as a miss against the
    # track's shared margin; the metered error is still counted, by the ceiling (1 in 4)
    assert fuel_proxy_misses(metered, native) == []
    assert validation_failure(True, metered, native, VBASE, BAR0, VRULE)[0] == "error_ceiling"


def test_a_proxy_miss_is_reported_before_a_quality_drift_on_another_nonce():
    # nonce 0 hit the metered limit (native had budget left); nonce 1 drifted 110 -> 111
    metered = vrows([108, 110, 110, 110], limit=(0,))
    native = vrows([110, 111, 110, 110])
    # mutation: checking quality before fuel reports "nondeterministic", which records no
    # miss, so the track's margin never drops
    assert validation_failure(True, metered, native, VBASE, BAR0, VRULE)[0] == "fuel_proxy_miss"


def test_a_nonce_where_both_runs_hit_their_limit_is_neither_a_miss_nor_a_mismatch():
    # the native budget stopped nonce 0 early (margin < 1 by design) and the metered run used
    # its whole fuel: consistent, and the two qualities are expected to differ
    metered = vrows([112, 110, 110, 110], limit=(0,))
    native = vrows([109, 110, 110, 110], limit=(0,))
    # mutation: comparing qualities on a limit-truncated nonce demotes a valid candidate
    assert validation_failure(True, metered, native, VBASE, BAR0, VRULE)[0] is None
    assert quality_mismatches(metered, native) == []


def test_a_native_budget_exit_is_not_a_quality_mismatch():
    # native stopped at its budget; metered finished inside its fuel with a better solution
    metered = vrows([112, 110, 110, 110])
    native = vrows([109, 110, 110, 110], limit=(0,))
    # mutation: ignoring native limit_hit demotes every candidate that uses 80-100% of its fuel
    assert validation_failure(True, metered, native, VBASE, BAR0, VRULE)[0] is None


def test_a_nonce_ok_on_only_one_path_is_not_a_quality_mismatch():
    native = vrows([110, None, 110, 110], error="no_solution")
    metered = vrows([110, 110, 110, 110])
    # mutation: comparing None to 110 calls a native error "nondeterministic"
    reason, _ = validation_failure(True, metered, native, VBASE, BAR0, VRULE)
    assert reason is None


def test_a_stepping_stone_must_also_beat_the_current_best():
    step = vrows([100, 100, 100, 101])  # +0.25%: not over a best of +0.3%, and no beats()
    # mutation: comparing against 0.0 instead of the current best keeps replacing the best
    # with worse stepping stones
    assert validation_failure(True, step, step, VBASE, Bar(0.003, 0.003), VRULE)[0] == "not_improved"
    # the loop's rule is mean-over-best OR beats: +2% beats the baseline and passes even
    # under a +3% best (tests/test_loop.py::test_win_with_lower_delta_than_a_false_positive_best)
    big = vrows([102, 102, 102, 102])
    assert validation_failure(True, big, big, VBASE, Bar(0.03, 0.03), VRULE)[0] is None


def test_quality_mismatches_lists_each_differing_nonce():
    native = vrows([110, 110, None, 110], error="no_solution")
    metered = vrows([110, 111, 105, 109])
    # nonce 2 is ok on one path only, so it is not a mismatch
    assert quality_mismatches(metered, native) == [
        {"track": "t", "nonce": 1, "native": 110, "metered": 111},
        {"track": "t", "nonce": 3, "native": 110, "metered": 109}]


# Issue #32: which candidate becomes the best, the code every later iteration edits.
# Five tracks at a baseline quality of 10000, so a quality of 9813 is exactly -1.87%.
ISSUE_TRACKS = ("10k", "20k", "50k", "100k", "200k")
IBASE = [NonceResult(t, 0, True, 10000, 1) for t in ISSUE_TRACKS]


def irows(*qs):
    return [NonceResult(t, 0, True, q, 1) for t, q in zip(ISSUE_TRACKS, qs)]


# run 20260928-093637-hypergraph: the best after iterations 1-5 (+0.37 +0.04 +0.07 +0.10 +0.18)
ISSUE_BEST = bundle_delta(IBASE, irows(10037, 10004, 10007, 10010, 10018))
# iteration 7: -1.87 -4.72 +8.81 -7.59 +14.53, mean +1.83%
ISSUE_SEVEN = bundle_delta(IBASE, irows(9813, 9528, 10881, 9241, 11453))
# no candidate here beats the baseline, so only the stepping-stone path can pass
NO_WIN = BeatRule(margin=0.5, track_tolerance=0.0, error_ceiling=0.05)


def test_a_higher_mean_bought_with_a_regressed_track_is_not_an_improvement():
    bar = Bar(ISSUE_BEST.mean_rel_delta, ISSUE_BEST.worst_rel_delta)
    assert ISSUE_SEVEN.mean_rel_delta == pytest.approx(0.01832)
    assert ISSUE_SEVEN.mean_rel_delta > bar.mean  # the old mean-only rule took it
    # mutation: dropping the worst-track clause from improves() makes iteration 7 the best
    assert not improves(ISSUE_SEVEN, bar, BeatRule())
    assert not improves(ISSUE_SEVEN, bar, NO_WIN)


def test_a_stepping_stone_whose_worst_track_holds_is_an_improvement():
    # every track +0.20% except 20k, which stays at the best's worst (+0.04%) exactly
    step = bundle_delta(IBASE, irows(10020, 10004, 10020, 10020, 10020))
    bar = Bar(ISSUE_BEST.mean_rel_delta, ISSUE_BEST.worst_rel_delta)
    assert step.worst_rel_delta == bar.worst
    # mutation: `>` instead of `>=` on the worst track rejects a candidate that only ties it
    # mutation: requiring beats() (spec §5.3) rejects every stepping stone under NO_WIN
    assert improves(step, bar, NO_WIN)


def test_track_tolerance_lets_the_worst_track_drop_that_far_below_the_best():
    # binary-exact values: worst -0.5 against a best's worst of -0.25, tolerance 0.25
    base = [NonceResult("t", 0, True, 100, 1), NonceResult("u", 0, True, 100, 1)]
    cand = [NonceResult("t", 0, True, 175, 1), NonceResult("u", 0, True, 50, 1)]
    d = bundle_delta(base, cand)
    assert d.worst_rel_delta == -0.5 and d.mean_rel_delta == 0.125
    tolerant = BeatRule(margin=0.5, track_tolerance=0.25, error_ceiling=0.05)
    # mutation: ignoring track_tolerance in improves() rejects this candidate
    assert improves(d, Bar(0.0, -0.25), tolerant)
    assert not improves(d, Bar(0.0, -0.2), tolerant)


def test_a_win_is_an_improvement_even_under_a_higher_best():
    # tests/test_loop.py::test_win_with_lower_delta_than_a_false_positive_best relies on this:
    # a candidate that beats the baseline becomes best so that confirming it confirms best
    win = bundle_delta(IBASE, irows(10100, 10100, 10100, 10100, 10100))
    # mutation: dropping the beats() disjunct rejects it, so a confirmed win is not the best
    assert improves(win, Bar(0.05, 0.05), BeatRule())


def test_no_improvement_over_the_error_ceiling():
    base = [NonceResult("t", n, True, 100, 1) for n in range(4)]
    cand = [NonceResult("t", 0, False, None, 1, "panic")] + [
        NonceResult("t", n, True, 200, 1) for n in (1, 2, 3)]
    d = bundle_delta(base, cand)
    assert d.error_rate == 0.25 and d.mean_rel_delta > 0
    # mutation: dropping the error ceiling from the stepping-stone path accepts it
    assert not improves(d, Bar(0.0, 0.0), NO_WIN)


def test_on_a_template_start_any_scored_candidate_improves_on_no_best():
    far_below = bundle_delta(IBASE, irows(1000, 500, 2000, 100, 3000))
    assert far_below.worst_rel_delta < -0.9
    # mutation: a no-best bar of (0.0, 0.0) on a template start keeps every candidate out
    assert improves(far_below, Bar(-math.inf, -math.inf), NO_WIN)


def test_the_shortfall_names_the_track_that_kept_a_candidate_out():
    bar = Bar(ISSUE_BEST.mean_rel_delta, ISSUE_BEST.worst_rel_delta)
    why = worst_track_shortfall(ISSUE_SEVEN, bar, BeatRule())
    # the recall prompt shows this text, so the model learns which track it broke
    assert why is not None and "100k" in why and "-7.59%" in why and "+0.04%" in why
    # mutation: a shortfall reported when the mean did not clear the best either
    worse = bundle_delta(IBASE, irows(9000, 9000, 9000, 9000, 9000))
    assert worst_track_shortfall(worse, bar, BeatRule()) is None
    # mutation: a shortfall reported for a candidate that does improve
    win = bundle_delta(IBASE, irows(10100, 10100, 10100, 10100, 10100))
    assert worst_track_shortfall(win, bar, BeatRule()) is None


def test_validation_demotes_a_metered_worst_track_regression():
    base = vrows([100, 100]) + vrows([100, 100], track="u")
    metered = vrows([110, 110]) + vrows([99, 99], track="u")  # t +10%, u -1%: mean +4.5%
    # the best is +1% mean with a worst of +0.5%; -1% < 0 so beats() fails too
    # mutation: validation_failure keeping the mean-only rule validates it
    assert validation_failure(True, metered, metered, base, Bar(0.01, 0.005),
                              VRULE)[0] == "not_improved"


def test_paired_se_matches_hand_computation_on_two_tracks():
    # spec Tests 1. mutations: population stdev; dividing by the candidate mean; summing
    # track SEs instead of root-sum-of-squares; dividing the combined SE by sqrt(T)
    base = [R("A", i, 100) for i in range(4)] + [R("B", 0, 200), R("B", 1, 200)]
    cand = [R("A", 0, 101), R("A", 1, 103), R("A", 2, 99), R("A", 3, 105),
            R("B", 0, 200), R("B", 1, 204)]
    d = bundle_delta(base, cand)
    by = {t.track: t for t in d.tracks}
    assert by["A"].rel_se == pytest.approx(math.sqrt(5 / 3) / 100, rel=1e-9)
    assert by["B"].rel_se == pytest.approx(0.01, rel=1e-9)
    assert d.mean_rel_se == pytest.approx(math.sqrt(2 / 3) / 100, rel=1e-9)
    ratio, band, state = noise(d, BeatRule())
    assert ratio == pytest.approx(1.5 / math.sqrt(2 / 3), rel=1e-9)
    assert (band, state) == ("marginal", "ok")
    assert d.to_dict()["mean_rel_se"] == d.mean_rel_se
    assert d.to_dict()["tracks"][0]["rel_se"] == by["A"].rel_se


def test_se_is_none_when_any_track_has_one_nonce():
    # spec Tests 2. mutation: averaging over the tracks that have an SE hides the n=1 track
    d = bundle_delta([R("a", 0, 100)], [R("a", 0, 102)])
    assert d.tracks[0].rel_se is None and d.mean_rel_se is None
    assert noise(d, BeatRule()) == (None, None, "n1")
    base = [R("a", 0, 100)] + [R("b", i, 100) for i in range(4)]
    cand = [R("a", 0, 102)] + [R("b", 0, 100), R("b", 1, 102), R("b", 2, 100), R("b", 3, 102)]
    d = bundle_delta(base, cand)
    by = {t.track: t for t in d.tracks}
    assert by["a"].rel_se is None and by["b"].rel_se is not None
    assert d.mean_rel_se is None and noise(d, BeatRule())[2] == "n1"


def test_se_scores_a_failed_nonce_at_the_worst_quality():
    # spec Tests 3. mutation: leaving the failed nonce out of the differences gives [2, 0]
    base = [R("t", 0, 10), R("t", 1, 20), R("t", 2, 30)]
    cand = [R("t", 0, 12), R("t", 1, None, "panic"), R("t", 2, 30)]
    d = bundle_delta(base, cand)
    assert d.tracks[0].rel_delta == pytest.approx(-2 / 15, rel=1e-9)
    assert d.tracks[0].rel_se == pytest.approx(0.1855921454276674, rel=1e-9)
    assert noise(d, BeatRule(error_ceiling=0.34))[2] == "ok"


def test_noise_is_withheld_over_the_error_ceiling():
    # spec Tests 4 and 10. mutation: checking n=1 before the ceiling reports "n1" for a
    # broken one-nonce candidate
    base = [R("t", 0, 10), R("t", 1, 20), R("t", 2, 30)]
    cand = [R("t", 0, 12), R("t", 1, None, "panic"), R("t", 2, 30)]
    d = bundle_delta(base, cand)
    assert noise(d, BeatRule(error_ceiling=0.05)) == (None, None, "over_ceiling")
    assert d.mean_rel_se is not None
    one = bundle_delta([R("t", 0, 100)], [R("t", 0, None, "panic")])
    assert noise(one, BeatRule())[2] == "over_ceiling"


def test_zero_spread_states():
    # spec Tests 5. mutation: dividing by a zero SE raises ZeroDivisionError or returns inf
    same = bundle_delta(BASE, list(BASE))
    assert same.mean_rel_se == 0.0
    assert noise(same, BeatRule()) == (None, None, "identical")
    plus = [R(r.track, r.nonce, r.quality + 1) for r in BASE]
    assert noise(bundle_delta(BASE, plus), BeatRule()) == (None, None, "no_spread")


def test_se_pairs_rows_by_nonce_not_by_input_order():
    # The bench may return rows in another order than the baseline's. mutation: pairing
    # b[track].values() with c[track].values() in dict order. Reversed, that pairs
    # [100, 110, 120, 130] with [135, 119, 113, 101]: differences [35, 9, -7, -29], a far
    # larger stdev than the true [1, 3, -1, 5]
    base = [R("A", 0, 100), R("A", 1, 110), R("A", 2, 120), R("A", 3, 130)]
    cand = [R("A", 0, 101), R("A", 1, 113), R("A", 2, 119), R("A", 3, 135)]
    fwd = bundle_delta(base, cand)
    rev = bundle_delta(base, list(reversed(cand)))
    # differences [1, 3, -1, 5] in either order: stdev sqrt(20/3), mean base 115
    expected = math.sqrt(20 / 3) / 2 / 115
    assert fwd.tracks[0].rel_se == pytest.approx(expected, rel=1e-9)
    assert rev.tracks[0].rel_se == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("delta,band", [(2.0, "clear"), (1.999, "marginal"),
                                        (1.0, "marginal"), (0.999, "noise"),
                                        (-2.0, "clear")])
def test_noise_band_boundaries(delta, band):
    # spec Tests 6. mutations: > for >= at either boundary; the band using the signed delta
    d = BundleDelta(tracks=[TrackDelta("t", 1.0, 1.0, delta, 0, 4, rel_se=1.0)],
                    mean_rel_delta=delta, worst_rel_delta=delta, error_rate=0.0,
                    mean_rel_se=1.0)
    ratio, got, state = noise(d, BeatRule())
    assert (ratio, got, state) == (abs(delta), band, "ok")
