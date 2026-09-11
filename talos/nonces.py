"""Draw the per-track training and held-out nonce sets for one job."""
from __future__ import annotations

import secrets

from talos.types import NonceSet

HOLDOUT_START = 1_000_000


def new_rand_hash() -> str:
    return secrets.token_hex(32)


def draw_nonce_sets(tracks: list[str], rand_hash: str, training_count: int = 32,
                    holdout_count: int = 32) -> tuple[list[NonceSet], list[NonceSet]]:
    training = [NonceSet(track=t, rand_hash=rand_hash, start=0, count=training_count)
                for t in tracks]
    holdout = [NonceSet(track=t, rand_hash=rand_hash, start=HOLDOUT_START, count=holdout_count)
               for t in tracks]
    return training, holdout
