"""Compare a candidate's per-nonce results against the baseline's on identical nonces."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from statistics import mean

from talos.challenges import BeatRule
from talos.types import NonceResult


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
