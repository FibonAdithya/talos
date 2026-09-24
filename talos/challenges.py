"""Static per-challenge facts. Everything a job needs that is not read from mainnet."""
from __future__ import annotations

import re
from dataclasses import dataclass

MONOREPO_REF = "84a5787f5b14a630bdf40f52bccf37887d3d8464"
DEV_IMAGE_TAG = "0.0.7"

C3_CPU_WORKERS = 4
# Hardware preference order, tried in turn at job start until one has capacity
# (`talos/bench.py::ModalBench.select_hardware`, `talos/c3_bench.py::C3Bench.select_hardware`).
# The choice is then frozen in state.json for the whole job: the baseline and every candidate
# must score on the same class (AGENTS.md invariant 1), so a fallback can never happen per
# call. GPUs: only classes with at least the L40S's 48 GB of VRAM, cheapest first; a smaller
# card could OOM a candidate the baseline fit. C3's "l40" class spans both the L40 and the
# L40S. C3 CPU profiles: only the 4 vCPU / 16 GB ones (the 4-core Modal spec; the 4 GB
# cpu-n1 profile is under the 8 GB the spec declares), the profile the existing baselines were
# measured on first. Modal's CPU containers have one class and nothing to fall back to.
MODAL_GPUS = ("L40S", "A100-80GB", "H100")
C3_GPU_CLASSES = ("l40", "a100", "h100")
C3_CPU_PROFILES = ("cpu-d3-4vcpu-16gb", "cpu-e2-4vcpu-16gb")


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


def _cpu(name: str, cid: str) -> ChallengeSpec:
    return ChallengeSpec(name=name, id=cid, is_gpu=False)


def _gpu(name: str, cid: str) -> ChallengeSpec:
    return ChallengeSpec(name=name, id=cid, is_gpu=True)


def hardware_options(spec: ChallengeSpec) -> tuple[str, ...]:
    """The Modal GPUs a job for this challenge may be frozen to, in preference order."""
    return MODAL_GPUS if spec.is_gpu else ()


def c3_hardware_options(spec: ChallengeSpec) -> tuple[str, ...]:
    """The C3 GPU classes or CPU profiles a job for this challenge may be frozen to, in
    preference order."""
    return C3_GPU_CLASSES if spec.is_gpu else C3_CPU_PROFILES


def _chosen(spec: ChallengeSpec, hardware: str | None, options: tuple[str, ...]) -> str:
    """The hardware a job was frozen to, checked against the list. No default: a helper that
    fell back to the first option would let a job frozen to an A100 (or the e2 CPU profile)
    look up (and hit) an L40S (or d3) baseline whenever the caller forgot to pass the choice."""
    if hardware is None:
        raise ValueError(f"pass the hardware the {spec.name} job was frozen to; one of "
                         f"{', '.join(options)}")
    if hardware not in options:
        raise ValueError(f"unknown hardware {hardware!r} for {spec.name}; one of "
                         f"{', '.join(options)}")
    return hardware


def gpu_slug(gpu: str) -> str:
    """A GPU name as a Modal function-name suffix: lower case, runs of other characters become
    one underscore ("A100-80GB" -> "a100_80gb")."""
    return re.sub(r"[^a-z0-9]+", "_", gpu.lower()).strip("_")


def modal_workers(spec: ChallengeSpec) -> int:
    """Nonces scored at once inside one Modal container, and so the size of one `score_batch`
    call: every core of a CPU container, one on a GPU (the nonces would serialise on the
    device and the batch would outlive the function timeout)."""
    return 1 if spec.is_gpu else spec.cpu


def hardware_class(spec: ChallengeSpec, hardware: str | None = None) -> str:
    """Part of the baseline cache key. A measurement is only reusable on the same hardware, and
    memory belongs in the key as much as cores do: changing memory_mib alone changes the timings
    (and the price per second) a cached baseline was measured under. The packing is in it too:
    four nonces sharing a container's cores and memory time differently from one nonce alone,
    so a baseline measured one nonce per container (before `-x4`) is never a hit. For a GPU
    challenge the key carries the GPU the job was frozen to; `hardware` is ignored for a CPU
    challenge, which has one class on Modal."""
    if spec.is_gpu:
        return f"gpu-{_chosen(spec, hardware, MODAL_GPUS)}"
    return f"cpu{spec.cpu}-mem{spec.memory_mib}-x{modal_workers(spec)}"


def c3_profile(spec: ChallengeSpec, hardware: str | None = None) -> str:
    """The C3 `hardware:` setting: the CPU profile or GPU class the job was frozen to."""
    return _chosen(spec, hardware, c3_hardware_options(spec))


def c3_workers(spec: ChallengeSpec) -> int:
    return 1 if spec.is_gpu else C3_CPU_WORKERS


def c3_hardware_class(spec: ChallengeSpec, hardware: str | None = None) -> str:
    """Baseline cache key component. Prefixed so a C3 baseline never matches a Modal one."""
    return f"c3-{c3_profile(spec, hardware)}"


def host_slug(host: str) -> str:
    """A hostname (or a GPU name) as a cache-key component: lower case, every run of characters
    outside [a-z0-9] becomes one dash, no dash at either end. Empty input becomes "host"."""
    return re.sub(r"[^a-z0-9]+", "-", host.lower()).strip("-") or "host"


def local_workers(spec: ChallengeSpec, cpus: int) -> int:
    return 1 if spec.is_gpu else cpus


def local_hardware_class(spec: ChallengeSpec, cpus: int, memory_gib: int,
                         gpu_name: str | None, host: str) -> str:
    """Baseline cache key component for the local backend. Prefixed so a local baseline never
    matches a Modal or C3 one. The limits are in it, for GPU challenges too, because they change
    the timings a baseline was measured under (the build's instrumentation pass runs on the
    CPUs whatever the device); the host guards a home directory synced between machines."""
    limits = f"cpu{cpus}-mem{memory_gib}"
    if spec.is_gpu:
        if not gpu_name:
            raise ValueError(f"{spec.name} is a GPU challenge; its local hardware class needs "
                             f"the GPU name")
        return f"local-{host_slug(host)}-gpu-{host_slug(gpu_name)}-{limits}"
    return f"local-{host_slug(host)}-{limits}"


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
