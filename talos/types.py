"""Core value types shared by every Talos module. No I/O here."""
from __future__ import annotations

from dataclasses import dataclass, asdict

ERROR_KINDS = (None, "no_solution", "invalid", "out_of_fuel", "panic", "timeout", "compile")


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


@dataclass
class CompileResult:
    ok: bool
    artifact_id: str | None
    output: str
