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


def runtime_ratio(baseline: list[NonceResult], candidate: list[NonceResult]) -> float:
    """Mean candidate runtime over mean baseline runtime on the candidate's slowest track
    relative to the baseline, so one track running 14x slower is not hidden by four that did
    not change. Tracks whose baseline runtime is zero are skipped; no track leaves 1.0."""
    b, c = _by_track(baseline), _by_track(candidate)
    ratios = []
    for track in b.keys() & c.keys():
        bm = mean(r.runtime_ms for r in b[track].values())
        if bm <= 0:
            continue
        ratios.append(mean(r.runtime_ms for r in c[track].values()) / bm)
    return max(ratios) if ratios else 1.0


def beats(baseline: list[NonceResult], candidate: list[NonceResult], rule: BeatRule) -> bool:
    d = bundle_delta(baseline, candidate)
    return (d.mean_rel_delta > rule.margin
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
    """Confirmation rule for a focused job: the focus track is over the margin with its own error
    rate under the ceiling, and no guard track drops below -track_tolerance."""
    d = bundle_delta(baseline, candidate)
    by = {t.track: t for t in d.tracks}
    if track not in by:
        raise ScoringError(f"focus track {track!r} has no results")
    focus = by[track]
    guards_ok = all(t.rel_delta >= -rule.track_tolerance for t in d.tracks if t.track != track)
    return (focus.rel_delta > rule.margin
            and focus.cand_errors / focus.n <= rule.error_ceiling
            and guards_ok)


VALIDATION_REASONS = ("native_metered_build_mismatch", "fuel_proxy_miss", "nondeterministic",
                      "unscoreable", "error_ceiling", "not_improved")


def fuel_proxy_misses(metered: list[NonceResult],
                      native: list[NonceResult]) -> list[NonceResult]:
    """Metered rows whose fuel ran out (with or without a saved solution) where the same
    nonce's native run did not reach its budget: the budget stood in for more fuel than TIG
    gives. A nonce that hit its limit on both paths is consistent, not a miss. Nor is one whose
    native run hit the outer per-nonce timeout first: it never reached its budget."""
    nat = {(r.track, r.nonce): r for r in native}
    out = []
    for r in metered:
        if r.error == "out_of_fuel" or r.limit_hit:
            o = nat.get((r.track, r.nonce))
            if o is None or not (o.limit_hit or o.error == "timeout"):
                out.append(r)
    return out


def quality_mismatches(metered: list[NonceResult], native: list[NonceResult]) -> list[dict]:
    """Nonces both paths solved, neither cut off at its limit, with different qualities, in
    metered order. Scoring is deterministic per instance (MEASURED, spec §1), so any entry
    means the two paths ran different code or the algorithm depends on something other than
    its seed. A run stopped at its fuel or budget saved an earlier solution, so its quality
    is expected to differ and says nothing about determinism."""
    nat = {(r.track, r.nonce): r for r in native}
    out = []
    for r in metered:
        o = nat.get((r.track, r.nonce))
        if (o is not None and r.ok and o.ok and not r.limit_hit and not o.limit_hit
                and r.quality != o.quality):
            out.append({"track": r.track, "nonce": r.nonce, "native": o.quality,
                        "metered": r.quality})
    return out


def validation_failure(compiled: bool, metered: list[NonceResult], native: list[NonceResult],
                       baseline: list[NonceResult], best_delta: float,
                       rule: BeatRule) -> tuple[str | None, BundleDelta | None]:
    """Whether a candidate that improved natively still stands on TIG's metered runtime, and
    why not. It must build metered, run out of fuel on no nonce whose native run had budget
    left, give the native quality on every nonce both paths solved within their limits, stay
    under the error ceiling, and still count as improved by the loop's own rule (mean over
    the current best, or beats). beats() alone is not required: that would demote every
    stepping stone (user decision 2026-09-29)."""
    if not compiled:
        return "native_metered_build_mismatch", None
    if fuel_proxy_misses(metered, native):
        return "fuel_proxy_miss", None
    if quality_mismatches(metered, native):
        return "nondeterministic", None
    try:
        d = bundle_delta(baseline, metered)
    except ScoringError:
        return "unscoreable", None
    if d.error_rate > rule.error_ceiling:
        return "error_ceiling", d
    if not (d.mean_rel_delta > best_delta or beats(baseline, metered, rule)):
        return "not_improved", d
    return None, d
