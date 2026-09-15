"""Static per-challenge facts. Everything a job needs that is not read from mainnet."""
from __future__ import annotations

from dataclasses import dataclass

MONOREPO_REF = "84a5787f5b14a630bdf40f52bccf37887d3d8464"
DEV_IMAGE_TAG = "0.0.7"


def dev_image(name: str) -> str:
    return f"ghcr.io/tig-foundation/tig-monorepo/{name}/dev:{DEV_IMAGE_TAG}"


@dataclass(frozen=True)
class BeatRule:
    margin: float = 0.005
    track_tolerance: float = 0.0
    error_ceiling: float = 0.05


@dataclass(frozen=True)
class ChallengeSpec:
    name: str
    id: str
    is_gpu: bool
    beat: BeatRule = BeatRule()
    cpu: int = 4
    memory_mib: int = 8192
    gpu: str | None = None


def _cpu(name: str, cid: str) -> ChallengeSpec:
    return ChallengeSpec(name=name, id=cid, is_gpu=False)


def _gpu(name: str, cid: str) -> ChallengeSpec:
    return ChallengeSpec(name=name, id=cid, is_gpu=True, gpu="L40S")


def hardware_class(spec: ChallengeSpec) -> str:
    """Part of the baseline cache key. A measurement is only reusable on the same hardware, and
    memory belongs in the key as much as cores do: changing memory_mib alone changes the timings
    (and the price per second) a cached baseline was measured under."""
    return f"gpu-{spec.gpu}" if spec.is_gpu else f"cpu{spec.cpu}-mem{spec.memory_mib}"


CHALLENGES: dict[str, ChallengeSpec] = {
    s.name: s
    for s in (
        _cpu("satisfiability", "c001"),
        _cpu("vehicle_routing", "c002"),
        _cpu("knapsack", "c003"),
        _gpu("vector_search", "c004"),
        _gpu("hypergraph", "c005"),
        _gpu("neuralnet_optimizer", "c006"),
        _cpu("job_scheduling", "c007"),
        _cpu("energy_arbitrage", "c008"),
    )
}
