import pytest

from talos.challenges import BeatRule
from talos.scoring import ScoringError, beats, bundle_delta
from talos.types import NonceResult


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
