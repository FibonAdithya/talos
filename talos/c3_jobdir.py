"""Generate the directory `c3 deploy` uploads for one evaluate call. Pure file writing."""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import stat
from dataclasses import asdict, dataclass
from pathlib import Path

from talos.bench import EvalRequest
from talos.challenges import (CHALLENGES, DEV_IMAGE_TAG, MONOREPO_REF, c3_image, c3_profile,
                              c3_workers, dev_image, local_workers)
from talos.inside import NONCE_TIMEOUT_S

BUILD_ALLOWANCE_S = 1200
# The local job's build allowance. MEASURED 2026-09-23: a candidate build took 14m44s on 16
# cores (the dev image re-instruments every dependency on each build), and fewer cores take
# longer; 20 minutes would time out every job on a smaller machine.
LOCAL_BUILD_ALLOWANCE_S = 3600
LOCAL_APP = "/app"  # where the local job container mounts the challenge's monorepo volume
LOCAL_LOCK = f"{LOCAL_APP}/.talos-lock"  # held by every container that writes the volume
TIME_CAP_S = 6 * 3600
JOB_MODULES = ("__init__", "inside", "scoring", "types", "challenges", "diagnostics", "c3_job")
_PKG = Path(__file__).resolve().parent


@dataclass(frozen=True)
class LocalSettings:
    """What the local backend fixes at `talos setup`: the job container's CPU and memory limits.
    Both are in the local hardware class, so changing either invalidates local baselines."""
    cpus: int
    memory_gib: int


def time_limit_s(nonce_count: int, workers: int,
                 build_allowance_s: int = BUILD_ALLOWANCE_S) -> int:
    raw = build_allowance_s + math.ceil(nonce_count * NONCE_TIMEOUT_S / workers)
    return min(TIME_CAP_S, math.ceil(raw / 60) * 60)


def hhmmss(seconds: int) -> str:
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def job_settings(challenge: str, purpose: str, seconds: int) -> dict:
    """The job's settings, named as C3's MCP `deploy` tool names them. `.c3` renders these same
    values, so the two submission paths cannot disagree about hardware, image or time limit."""
    spec = CHALLENGES[challenge]
    return {"project": "talos", "job_name": f"talos-{challenge}-{purpose}", "script": "job.sh",
            "hardware": c3_profile(spec), "walltime_seconds": seconds,
            "docker_image": c3_image(challenge),
            "docker_requires_accelerator": "cuda" if spec.is_gpu else "none"}


def c3_config_text(challenge: str, purpose: str, seconds: int) -> str:
    s = job_settings(challenge, purpose, seconds)
    return (f"project: {s['project']}\njob_name: {s['job_name']}\nscript: {s['script']}\n"
            f"hardware: {s['hardware']}\ntime: \"{hhmmss(s['walltime_seconds'])}\"\n"
            f"docker:\n  image: {s['docker_image']}\n"
            f"  requires_accelerator: {s['docker_requires_accelerator']}\n")


def job_sh_text(monorepo_ref: str) -> str:
    return ("#!/bin/bash\nset -euo pipefail\nmkdir -p /app\n"
            f"curl -fsSL \"https://codeload.github.com/tig-foundation/tig-monorepo/tar.gz/"
            f"{monorepo_ref}\" | tar xz -C /app --strip-components=1\n"
            "cd \"$C3_JOB_WORKDIR\"\nexec python3 -m talos.c3_job\n")


def local_job_sh_text() -> str:
    """The monorepo is already on the /app volume (talos/local_transport.py::prepare), so the
    local job only changes into the bind-mounted job dir and runs the same runner as C3, under
    the volume's lock: /app is shared by every job of the challenge on this machine (two runs,
    or a `talos compile` beside a run), and the runner stages the candidate into that checkout,
    so two jobs at once must take turns or one is built from the other's files. A waiting job's
    time limit still counts from its start."""
    return ("#!/bin/bash\nset -euo pipefail\ncd \"$C3_JOB_WORKDIR\"\n"
            f"exec flock {LOCAL_LOCK} python3 -m talos.c3_job\n")


def local_settings_doc(request: EvalRequest, local: LocalSettings, seconds: int) -> dict:
    """local.json: everything DockerTransport.deploy needs and nothing else reads. The request
    hash names the container; the rand hash must not be here (the name shows in `docker ps`)."""
    spec = CHALLENGES[request.challenge]
    return {"challenge": request.challenge, "image": dev_image(request.challenge),
            "cpus": local.cpus, "memory_gib": local.memory_gib, "gpu": spec.is_gpu,
            "workers": local_workers(spec, local.cpus), "time_limit_s": seconds,
            "request_hash": request_hash(request)}


def payload(request: EvalRequest, workers: int | None = None) -> dict:
    spec = CHALLENGES[request.challenge]
    p = {"challenge": request.challenge, "challenge_id": spec.id, "files": request.files,
         "training": [asdict(n) for n in request.training],
         "holdout": [asdict(n) for n in request.holdout], "fuel": request.fuel,
         "baseline_training": ([r.to_dict() for r in request.baseline_training]
                               if request.baseline_training is not None else None),
         "rule": asdict(request.rule), "monorepo_ref": MONOREPO_REF,
         "prior_functions": request.prior_functions, "timeouts": request.timeouts,
         "workers": c3_workers(spec) if workers is None else workers,
         "nonce_timeout_s": NONCE_TIMEOUT_S,
         "dev_image_tag": DEV_IMAGE_TAG}
    if request.hyperparameters is not None:
        # Only when present, so a request identical to a pre-upgrade one hashes the same way;
        # a resumed job would otherwise see a hash mismatch and submit a second, orphaning the
        # first (still-billing) one. `c3_job` reads it with `payload.get("hyperparameters") or {}`.
        p["hyperparameters"] = request.hyperparameters
    return p


def request_hash(request: EvalRequest) -> str:
    """Identifies a request across processes so a resume reattaches only to a job that was
    submitted for exactly these files and nonces. The hash covers the hash of the rand_hash,
    never the rand_hash itself, because it lands in state.json."""
    p = payload(request)
    p["baseline_training"] = None
    for key in ("training", "holdout"):
        p[key] = [{**n, "rand_hash": hashlib.sha256(n["rand_hash"].encode()).hexdigest()}
                  for n in p[key]]
    return hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()[:16]


def write_job_dir(job_dir: Path, request: EvalRequest, purpose: str,
                  local: LocalSettings | None = None) -> Path:
    """Writes the deploy directory, wiping `job_dir` first if it already exists. With `local`
    it is the local flavour: local.json instead of .c3, a job.sh that does not download the
    monorepo, and the local worker count in the payload."""
    job_dir = Path(job_dir)
    if job_dir.exists():
        shutil.rmtree(job_dir)
    (job_dir / "talos").mkdir(parents=True)
    spec = CHALLENGES[request.challenge]
    nonces = sum(n.count for n in request.training) + sum(n.count for n in request.holdout)
    if local is None:
        workers = c3_workers(spec)
        seconds = time_limit_s(max(nonces, 1), workers)
        (job_dir / ".c3").write_text(c3_config_text(request.challenge, purpose, seconds),
                                     encoding="utf-8", newline="\n")
        sh_text = job_sh_text(MONOREPO_REF)
    else:
        workers = local_workers(spec, local.cpus)
        seconds = time_limit_s(max(nonces, 1), workers, LOCAL_BUILD_ALLOWANCE_S)
        (job_dir / "local.json").write_text(
            json.dumps(local_settings_doc(request, local, seconds), indent=1),
            encoding="utf-8", newline="\n")
        sh_text = local_job_sh_text()
    sh = job_dir / "job.sh"
    sh.write_text(sh_text, encoding="utf-8", newline="\n")
    sh.chmod(sh.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    (job_dir / "payload.json").write_text(json.dumps(payload(request, workers), indent=1),
                                          encoding="utf-8", newline="\n")
    for mod in JOB_MODULES:
        shutil.copy2(_PKG / f"{mod}.py", job_dir / "talos" / f"{mod}.py")
    return job_dir
