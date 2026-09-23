"""Static per-challenge facts. Everything a job needs that is not read from mainnet."""
from __future__ import annotations

from dataclasses import dataclass

MONOREPO_REF = "84a5787f5b14a630bdf40f52bccf37887d3d8464"
DEV_IMAGE_TAG = "0.0.7"

C3_CPU_PROFILE = "cpu-d3-4vcpu-16gb"  # 4 vCPU / 16 GB, matches the 4-core Modal spec
C3_GPU_HARDWARE = "l40"               # class; closest to Modal's L40S
C3_CPU_WORKERS = 4


def dev_image(name: str) -> str:
    return f"ghcr.io/tig-foundation/tig-monorepo/{name}/dev:{DEV_IMAGE_TAG}"


def c3_image(name: str) -> str:
    """C3 pulls the official GHCR image directly, so both backends build in the same image."""
    return dev_image(name)


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


def c3_profile(spec: ChallengeSpec) -> str:
    return C3_GPU_HARDWARE if spec.is_gpu else C3_CPU_PROFILE


def c3_workers(spec: ChallengeSpec) -> int:
    return 1 if spec.is_gpu else C3_CPU_WORKERS


def c3_hardware_class(spec: ChallengeSpec) -> str:
    """Baseline cache key component. Prefixed so a C3 baseline never matches a Modal one."""
    return f"c3-{c3_profile(spec)}"


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
