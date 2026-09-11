"""Per-model prices in USD per million tokens: (input, output). Unknown models cost None,
which the CLI reports as "unpriced" rather than zero."""
from __future__ import annotations

from talos.types import Usage

PRICES: dict[str, tuple[float, float]] = {
    "claude-fable-5-1": (10.0, 50.0),
    "claude-fable-5": (10.0, 50.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


def estimate_cost(model: str, usage: Usage) -> float | None:
    key = model.split("/", 1)[-1] if "/" in model else model  # openrouter "vendor/model"
    p = PRICES.get(key)
    if p is None:
        return None
    return usage.input_tokens / 1e6 * p[0] + usage.output_tokens / 1e6 * p[1]
