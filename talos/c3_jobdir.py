"""Generate the directory `c3 deploy` uploads for one evaluate call. Pure file writing."""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import stat
from dataclasses import asdict
from pathlib import Path

from talos.bench import EvalRequest
from talos.challenges import (CHALLENGES, DEV_IMAGE_TAG, MONOREPO_REF, c3_image, c3_profile,
                              c3_workers)
from talos.inside import NONCE_TIMEOUT_S

BUILD_ALLOWANCE_S = 1200
TIME_CAP_S = 6 * 3600
JOB_MODULES = ("__init__", "inside", "scoring", "types", "challenges", "c3_job")
_PKG = Path(__file__).resolve().parent


def time_limit_s(nonce_count: int, workers: int) -> int:
    raw = BUILD_ALLOWANCE_S + math.ceil(nonce_count * NONCE_TIMEOUT_S / workers)
    return min(TIME_CAP_S, math.ceil(raw / 60) * 60)


def hhmmss(seconds: int) -> str:
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def c3_config_text(challenge: str, purpose: str, seconds: int) -> str:
    spec = CHALLENGES[challenge]
    accel = "cuda" if spec.is_gpu else "none"
    return (f"project: talos\njob_name: talos-{challenge}-{purpose}\nscript: job.sh\n"
            f"hardware: {c3_profile(spec)}\ntime: \"{hhmmss(seconds)}\"\n"
            f"docker:\n  image: {c3_image(challenge)}\n  requires_accelerator: {accel}\n")


def job_sh_text(monorepo_ref: str) -> str:
    return ("#!/bin/bash\nset -euo pipefail\nmkdir -p /app\n"
            f"curl -fsSL \"https://codeload.github.com/tig-foundation/tig-monorepo/tar.gz/"
            f"{monorepo_ref}\" | tar xz -C /app --strip-components=1\n"
            "cd \"$C3_JOB_WORKDIR\"\nexec python3 -m talos.c3_job\n")


def payload(request: EvalRequest) -> dict:
    spec = CHALLENGES[request.challenge]
    return {"challenge": request.challenge, "challenge_id": spec.id, "files": request.files,
            "training": [asdict(n) for n in request.training],
            "holdout": [asdict(n) for n in request.holdout], "fuel": request.fuel,
            "baseline_training": ([r.to_dict() for r in request.baseline_training]
                                  if request.baseline_training is not None else None),
            "rule": asdict(request.rule), "monorepo_ref": MONOREPO_REF,
            "workers": c3_workers(spec), "nonce_timeout_s": NONCE_TIMEOUT_S,
            "dev_image_tag": DEV_IMAGE_TAG}


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


def write_job_dir(job_dir: Path, request: EvalRequest, purpose: str) -> Path:
    job_dir = Path(job_dir)
    if job_dir.exists():
        shutil.rmtree(job_dir)
    (job_dir / "talos").mkdir(parents=True)
    spec = CHALLENGES[request.challenge]
    nonces = sum(n.count for n in request.training) + sum(n.count for n in request.holdout)
    (job_dir / ".c3").write_text(c3_config_text(request.challenge, purpose,
                                                time_limit_s(max(nonces, 1), c3_workers(spec))))
    sh = job_dir / "job.sh"
    sh.write_text(job_sh_text(MONOREPO_REF))
    sh.chmod(sh.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    (job_dir / "payload.json").write_text(json.dumps(payload(request), indent=1))
    for mod in JOB_MODULES:
        shutil.copy2(_PKG / f"{mod}.py", job_dir / "talos" / f"{mod}.py")
    return job_dir
