"""Native research scoring has no fuel counter. This module turns the baseline's metered
fuel_consumed and its native solve time, measured on the same nonces, into a per-track
time budget that stands in for the fuel limit. One record per (challenge, hardware class,
monorepo pin, dev image tag, baseline code, hyperparameters), cached under
~/.talos/calibration/ and reused by every later job with the same key."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import median

from talos import inside, native_runner
from talos.baseline import effective_hyperparameters
from talos.challenges import DEV_IMAGE_TAG, MONOREPO_REF
from talos.state import _atomic_write
from talos.types import NonceResult

# ESTIMATES with no measurement behind them (spec §4.4 and §6); tuned from recorded misses.
MARGIN_START = 0.8
MARGIN_STEP = 0.1
MARGIN_FLOOR = 0.3
MISS_LIMIT = 3
# ESTIMATE: a guard against timer noise on very fast nonces, not a research economy cap.
# runtime_floor_s (60 s) would bind on most hypergraph tracks (user decision 2026-09-29).
BUDGET_FLOOR_US = 1_000_000


def calibration_key(challenge: str, hardware_class: str, baseline_files: dict[str, str],
                    hyperparameters: dict | None) -> str:
    """The baseline enters by its metered artifact hash, which covers its files and both pins;
    the native runner by its digest, since a new template or build profile times differently.
    The nonce sets do not enter: a ratio of time to fuel carries over between nonce draws,
    and the rand hash must never reach a file outside the job."""
    payload = {"challenge": challenge, "hw": hardware_class, "ref": MONOREPO_REF,
               "image": DEV_IMAGE_TAG,
               "baseline": inside.content_hash(baseline_files, MONOREPO_REF, DEV_IMAGE_TAG),
               "runner": native_runner.runner_digest(),
               "hp": effective_hyperparameters(hyperparameters)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]


def track_ratios(metered: list[NonceResult], native: list[NonceResult]) -> dict[str, float]:
    """Median microseconds of native solve per unit of fuel, per track, over nonces that have
    both numbers from the same save. Each is taken at the algorithm's last save_solution call,
    which is the same save only when both runs are ok, neither hit its limit and both reached
    the same quality: a metered run cut off by fuel saved earlier than the unbudgeted native
    run did (metered q=100 at fuel 1000 against native q=150 at 9000 us gave 9.0). A zero or
    missing fuel_consumed is skipped: `if m.fuel_consumed` is meant to drop 0 as well as
    None."""
    by_nonce = {(r.track, r.nonce): r for r in metered}
    per: dict[str, list[float]] = {}
    for r in native:
        m = by_nonce.get((r.track, r.nonce))
        if (m is None or not (m.ok and r.ok) or m.limit_hit or r.limit_hit
                or m.quality != r.quality):
            continue
        if m.fuel_consumed and r.solve_us is not None:
            per.setdefault(r.track, []).append(r.solve_us / m.fuel_consumed)
    return {t: median(v) for t, v in per.items()}


def new_record(ratios: dict[str, float]) -> dict:
    return {"version": 1, "demotions": {},
            "tracks": {t: {"ratio": r, "margin": MARGIN_START, "misses": 0}
                       for t, r in ratios.items()}}


def budgets_us(record: dict, fuel: int) -> dict[str, int]:
    """round, not ceil: ratio x fuel x margin is a float product, and ceil turns
    1200000.0000000002 into 1200001."""
    return {t: max(BUDGET_FLOOR_US, int(round(v["ratio"] * fuel * v["margin"])))
            for t, v in record["tracks"].items()}


def record_miss(record: dict, track: str) -> bool:
    """A validation found a nonce of `track` out of fuel on the metered path after it passed
    natively. Every MISS_LIMIT misses the track's margin drops by MARGIN_STEP, never below
    MARGIN_FLOOR. True when the margin changed, so the caller recomputes its budgets."""
    t = record["tracks"][track]
    t["misses"] += 1
    if t["misses"] < MISS_LIMIT:
        return False
    t["misses"] = 0
    lowered = max(MARGIN_FLOOR, round(t["margin"] - MARGIN_STEP, 6))
    changed = lowered != t["margin"]
    t["margin"] = lowered
    return changed


def note_demotion(record: dict, reason: str) -> None:
    record.setdefault("demotions", {})
    record["demotions"][reason] = record["demotions"].get(reason, 0) + 1


def _path(cache_dir: Path, challenge: str, key: str) -> Path:
    return Path(cache_dir) / challenge / f"{key}.json"


def load(cache_dir: Path, challenge: str, key: str) -> dict | None:
    """None for a missing, unreadable or malformed record: it is then measured again. A ratio
    or margin that is not a finite positive number is malformed (json reads NaN and
    Infinity): it would give a zero, negative or NaN budget."""
    p = _path(cache_dir, challenge, key)
    if not p.exists():
        return None
    try:
        rec = json.loads(p.read_text(encoding="utf-8"))
        for v in rec["tracks"].values():
            ratio, margin = float(v["ratio"]), float(v["margin"])
            int(v["misses"])
            if not all(math.isfinite(x) and x > 0 for x in (ratio, margin)):
                return None
    except (ValueError, KeyError, TypeError, AttributeError):
        return None
    return rec


def save(cache_dir: Path, challenge: str, key: str, record: dict) -> None:
    p = _path(cache_dir, challenge, key)
    p.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(p, json.dumps(record, indent=1))
