"""Compare a candidate's per-nonce results against the baseline's on identical nonces."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from statistics import mean

from talos.challenges import BeatRule
from talos.types import NonceResult, NonceSet


class ScoringError(ValueError):
    pass


@dataclass
class TrackDelta:
    track: str
    base_mean: float
    cand_mean: float
    rel_delta: float
    cand_errors: int
    n: int


@dataclass
class BundleDelta:
    tracks: list[TrackDelta]
    mean_rel_delta: float
    worst_rel_delta: float
    error_rate: float

    def to_dict(self) -> dict:
        return asdict(self)


def _by_track(results: list[NonceResult]) -> dict[str, dict[int, NonceResult]]:
    out: dict[str, dict[int, NonceResult]] = {}
    for r in results:
        out.setdefault(r.track, {})[r.nonce] = r
    return out


def _quality_or_worst(r: NonceResult, worst: int) -> int:
    return r.quality if r.ok and r.quality is not None else worst


def bundle_delta(baseline: list[NonceResult], candidate: list[NonceResult]) -> BundleDelta:
    b, c = _by_track(baseline), _by_track(candidate)
    if b.keys() != c.keys():
        raise ScoringError(f"track mismatch: {sorted(b)} vs {sorted(c)}")
    tracks: list[TrackDelta] = []
    total = errors = 0
    for track in sorted(b):
        if b[track].keys() != c[track].keys():
            raise ScoringError(f"nonce mismatch on {track}")
        observed = [r.quality for r in list(b[track].values()) + list(c[track].values())
                    if r.ok and r.quality is not None]
        worst = max(0, min(observed)) if observed else 0
        bq = [_quality_or_worst(r, worst) for r in b[track].values()]
        cq = [_quality_or_worst(r, worst) for r in c[track].values()]
        bm, cm = mean(bq), mean(cq)
        if bm <= 0:
            raise ScoringError(f"baseline mean quality on {track} is {bm}; cannot normalise")
        n_err = sum(1 for r in c[track].values() if not r.ok)
        tracks.append(TrackDelta(track=track, base_mean=bm, cand_mean=cm,
                                 rel_delta=(cm - bm) / bm, cand_errors=n_err, n=len(cq)))
        total += len(cq)
        errors += n_err
    return BundleDelta(tracks=tracks,
                       mean_rel_delta=mean(t.rel_delta for t in tracks),
                       worst_rel_delta=min(t.rel_delta for t in tracks),
                       error_rate=errors / total)


def beats(baseline: list[NonceResult], candidate: list[NonceResult], rule: BeatRule) -> bool:
    d = bundle_delta(baseline, candidate)
    return (d.mean_rel_delta >= rule.margin
            and d.worst_rel_delta >= -rule.track_tolerance
            and d.error_rate <= rule.error_ceiling)


def holdout_decision(baseline_training: list[NonceResult] | None, training: list[NonceResult],
                     rule: BeatRule) -> tuple[bool, str]:
    """Whether to score the held-out set, and the reason recorded with the result. Lives here,
    not in the bench, because the C3 job applies it inside the container with the same code."""
    if baseline_training is None:
        return True, "forced"
    try:
        return (True, "won") if beats(baseline_training, training, rule) else (False, "not_won")
    except ScoringError:
        return False, "not_won"


def select(results: list[NonceResult], sets: list[NonceSet]) -> list[NonceResult]:
    """The rows whose (track, nonce) fall inside one of `sets`, in input order. Strays are
    dropped, not raised on; bundle_delta still raises on any mismatch that survives."""
    wanted = {(s.track, n) for s in sets for n in s.nonces()}
    return [r for r in results if (r.track, r.nonce) in wanted]


def focus_sets(track: str | None, training: list[NonceSet],
               holdout: list[NonceSet]) -> tuple[list[NonceSet], list[NonceSet]]:
    """The nonce sets one iteration scores. Focused: training is the track's own training set;
    held-out is the track's held-out set followed by every other track's TRAINING set, the
    regression guard (already measured for the baseline). Unfocused: unchanged."""
    if track is None:
        return training, holdout
    focus_tr = [s for s in training if s.track == track]
    focus_ho = [s for s in holdout if s.track == track]
    guards = [s for s in training if s.track != track]
    return focus_tr, focus_ho + guards


def beats_focused(baseline: list[NonceResult], candidate: list[NonceResult], rule: BeatRule,
                  track: str) -> bool:
    """Confirmation rule for a focused job: the focus track clears the margin with its own error
    rate under the ceiling, and no guard track drops below -track_tolerance."""
    d = bundle_delta(baseline, candidate)
    by = {t.track: t for t in d.tracks}
    if track not in by:
        raise ScoringError(f"focus track {track!r} has no results")
    focus = by[track]
    guards_ok = all(t.rel_delta >= -rule.track_tolerance for t in d.tracks if t.track != track)
    return (focus.rel_delta >= rule.margin
            and focus.cand_errors / focus.n <= rule.error_ceiling
            and guards_ok)
