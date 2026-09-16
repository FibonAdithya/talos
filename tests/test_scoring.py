import pytest

from talos.challenges import BeatRule
from talos.scoring import (ScoringError, beats, beats_focused, bundle_delta, focus_sets,
                           runtime_ratio, select)
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
    # mutation: `>` instead of `>=` on the margin fails the equal case
    rule = BeatRule(margin=0.05, track_tolerance=0.0, error_ceiling=0.05)
    cand = [R("t1", 0, 110), R("t1", 1, 110), R("t2", 0, 200), R("t2", 1, 200)]
    assert beats(BASE, cand, rule)  # mean delta exactly 0.05
    rule2 = BeatRule(margin=0.0501, track_tolerance=0.0, error_ceiling=0.05)
    assert not beats(BASE, cand, rule2)


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
    assert beats_focused(base, [R("t1", 0, 201), R("t1", 1, 201)], RULE, "t1")


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
