"""Core value types shared by every Talos module. No I/O here."""
from __future__ import annotations

from dataclasses import dataclass, asdict

# "not_run": a nonce a candidate's job never started, because it had already failed the error
# ceiling and stopped (talos/c3_job.py::_score). It counts as an error, as an unscored nonce must.
ERROR_KINDS = (None, "no_solution", "invalid", "out_of_fuel", "panic", "timeout", "compile",
               "not_run")

SCORING_MODES = ("metered", "native")


@dataclass(frozen=True)
class NonceSet:
    track: str
    rand_hash: str
    start: int
    count: int

    def nonces(self) -> range:
        return range(self.start, self.start + self.count)


@dataclass
class NonceResult:
    track: str
    nonce: int
    ok: bool
    quality: int | None
    runtime_ms: int
    error: str | None = None
    # Metered runs: the fuel tig-runtime recorded at the last save_solution call. Native runs:
    # microseconds from the start of solve_challenge to its last save. Calibration pairs the
    # two (talos/calibration.py::track_ratios). None when the run did not record it.
    fuel_consumed: int | None = None
    solve_us: int | None = None
    # The solver exited 87: tig-runtime's fuel ran out (metered) or the native budget did.
    # A run that saved a verified solution first is still ok, as TIG counts it, so `error`
    # cannot carry this; validation needs it to tell a truncated run from nondeterminism
    # (talos/scoring.py::validation_failure).
    limit_hit: bool = False

    def __post_init__(self) -> None:
        if self.error not in ERROR_KINDS:
            raise ValueError(f"unknown error kind {self.error!r}")
        if self.ok and self.quality is None:
            raise ValueError("ok result must carry a quality")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "NonceResult":
        return cls(**d)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None


@dataclass
class Completion:
    text: str
    usage: Usage
    # Why a call that cost something reported no cost; None when the cost is known, or when it
    # is None only because the model has no price-table entry (reported as "unpriced").
    cost_unknown: str | None = None


@dataclass
class CompileResult:
    ok: bool
    artifact_id: str | None
    output: str
