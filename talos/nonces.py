"""Draw the per-track training and held-out nonce sets for one job."""
from __future__ import annotations

import secrets

from talos.types import NonceSet

HOLDOUT_START = 1_000_000
# Per track, for each of the two sets. Sized for GPU challenges, where one nonce costs minutes:
# on hypergraph an L40 spends about 10 minutes per nonce-per-track on the baseline's two sets
# (measured 2026-09-25), so 32 was an 11-hour baseline against a 6-hour C3 job cap.
NONCES_PER_TRACK = 8


def new_rand_hash() -> str:
    return secrets.token_hex(32)


def draw_nonce_sets(tracks: list[str], rand_hash: str, training_count: int = NONCES_PER_TRACK,
                    holdout_count: int = NONCES_PER_TRACK) -> tuple[list[NonceSet], list[NonceSet]]:
    training = [NonceSet(track=t, rand_hash=rand_hash, start=0, count=training_count)
                for t in tracks]
    holdout = [NonceSet(track=t, rand_hash=rand_hash, start=HOLDOUT_START, count=holdout_count)
               for t in tracks]
    return training, holdout
