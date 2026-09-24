"""Modal app `talos-bench`. Deployed once into the user's Modal account by `talos setup`.
Every function runs inside the official TIG dev image for its challenge plus a pinned
monorepo checkout at /app. Artifacts live on the `talos-artifacts` Volume keyed by the
content hash of the submitted files."""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import modal

from talos import inside
from talos.challenges import (CHALLENGES, DEV_IMAGE_TAG, MODAL_GPUS, MONOREPO_REF, dev_image,
                              gpu_slug, modal_workers)
from talos.inside import NONCE_TIMEOUT_S

APP_NAME = "talos-bench"
ARTIFACTS = "/artifacts"
MONOREPO = Path("/app")

app = modal.App(APP_NAME)
volume = modal.Volume.from_name("talos-artifacts", create_if_missing=True)


def _python() -> str:
    # The functions below are serialized=True, and Modal refuses to deploy a serialized function
    # whose image Python differs in minor version from the interpreter that defined it.
    return "{}.{}".format(*sys.version_info[:2])


def _image(name: str) -> modal.Image:
    python = _python()
    return (
        modal.Image.from_registry(dev_image(name), add_python=python)
        .apt_install("git")
        .run_commands(
            "git clone https://github.com/tig-foundation/tig-monorepo.git /app",
            f"cd /app && git checkout {MONOREPO_REF}",
        )
        .env({"CHALLENGE": name})
        .add_local_python_source("modal_app", "talos")
    )


def _probe_image() -> modal.Image:
    """The capacity probe's image: stock, so its cold start measures the GPU queue and not a
    13 GB dev-image pull."""
    return modal.Image.debian_slim(python_version=_python())


def content_hash(files: dict[str, str]) -> str:
    return inside.content_hash(files, MONOREPO_REF, DEV_IMAGE_TAG)


def _compile_impl(name: str, files: dict[str, str]) -> dict:
    """Never raises for an application-level failure. A bad file map or a build that emits no
    .so is a failed compile the loop can act on; raising would instead burn the client's
    retry window on a deterministic error and pause the run."""
    try:
        volume.reload()
        art_id = content_hash(files)
        dest = Path(ARTIFACTS) / name / art_id
        if (dest / "algo.so").exists():
            return {"ok": True, "artifact_id": art_id, "output": "cached"}
        inside.stage_algorithm(MONOREPO, name, files, inside.ALGO_NAME)
        try:
            ok, out = inside.build(MONOREPO, name, inside.ALGO_NAME)
            if not ok:
                return {"ok": False, "artifact_id": None, "output": out}
            so, ptx = inside.artifact_paths(MONOREPO, name, inside.ALGO_NAME)
            if not so.exists():
                return {"ok": False, "artifact_id": None,
                        "output": out + f"\nbuild produced no .so at {so}"}
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(so, dest / "algo.so")
            if ptx:
                shutil.copy2(ptx, dest / "algo.ptx")
            volume.commit()
            return {"ok": True, "artifact_id": art_id, "output": out}
        finally:
            inside.unstage_algorithm(MONOREPO, name, inside.ALGO_NAME)
    except Exception as e:  # noqa: BLE001 - an in-container error is a failed compile, not an outage
        return {"ok": False, "artifact_id": None,
                "output": f"bench error: {type(e).__name__}: {e}"}


def _score_batch_impl(name: str, challenge_id: str, artifact_id: str, tasks: list[dict],
                      workers: int, pool_factory=None, clock=time.monotonic) -> dict:
    """Scores `tasks` (one dict each: track, rand_hash, nonce, fuel, timeout_s,
    hyperparameters) on `workers` processes at once. The client sends at most `workers`
    tasks per call, so the batch's wall time is bounded by its slowest nonce and fits the
    function timeout. Returns the rows in task order and the container's own wall seconds,
    which is what Modal bills for and what the client charges."""
    volume.reload()
    d = Path(ARTIFACTS) / name / artifact_id
    so, ptx = d / "algo.so", d / "algo.ptx"
    if not so.exists():
        # Infrastructure, not an algorithm failure: say so plainly rather than letting
        # tig-runtime's exit code be classified as "panic".
        raise FileNotFoundError(f"artifact {artifact_id} missing on volume")
    t0 = clock()
    order = {(t["track"], t["nonce"]): i for i, t in enumerate(tasks)}
    todo = [(challenge_id, t["track"], t["rand_hash"], t["nonce"], str(so), t["fuel"],
             min(t["timeout_s"], NONCE_TIMEOUT_S), str(ptx) if ptx.exists() else None,
             str(MONOREPO), t["hyperparameters"]) for t in tasks]
    rows = list(inside.run_nonces(todo, workers, pool_factory=pool_factory))
    rows.sort(key=lambda r: order[(r["track"], r["nonce"])])
    return {"rows": rows, "seconds": clock() - t0}


def register(app, image=_image, probe_image=_probe_image) -> None:
    """Registers every function on `app`. CPU challenges get `compile_<name>` and
    `score_batch_<name>`. GPU challenges get one such pair per GPU in MODAL_GPUS, suffixed
    `_<gpu_slug>`, each pinned to a single GPU: a job is frozen to one of them at start
    (`talos/bench.py::ModalBench.select_hardware`), and the baseline and every candidate run on it.
    Modal's own `gpu=[...]` fallback list is deliberately not used, because it picks a GPU per
    container, which is a per-call fallback AGENTS.md invariant 1 forbids. One `probe_<slug>`
    per GPU, on a stock image, is what select_hardware spawns to see whether that GPU has
    capacity. A batch holds `talos/challenges.py::modal_workers` nonces scored at once, so the
    score function's timeout still covers one nonce plus slack."""
    for name, spec in CHALLENGES.items():
        variants = [(f"_{gpu_slug(g)}", {"gpu": g}) for g in MODAL_GPUS] if spec.is_gpu \
            else [("", {"cpu": spec.cpu, "memory": spec.memory_mib})]
        for suffix, resources in variants:
            kw = dict(image=image(name), volumes={ARTIFACTS: volume}, serialized=True,
                      **resources)
            app.function(name=f"compile_{name}{suffix}", timeout=3600,
                         **kw)(_mk_compile(name))
            app.function(name=f"score_batch_{name}{suffix}", timeout=NONCE_TIMEOUT_S + 120,
                         **kw)(_mk_score_batch(name, spec.id, modal_workers(spec)))
    for gpu in MODAL_GPUS:
        app.function(name=f"probe_{gpu_slug(gpu)}", image=probe_image(), gpu=gpu, timeout=60,
                     serialized=True)(_mk_probe(gpu))


def _mk_compile(n):
    def compile_fn(files: dict) -> dict:
        return _compile_impl(n, files)
    return compile_fn


def _mk_score_batch(n, cid, workers):
    def score_batch(artifact_id: str, tasks: list[dict]) -> dict:
        return _score_batch_impl(n, cid, artifact_id, tasks, workers)
    return score_batch


def _mk_probe(gpu):
    def probe() -> str:
        return gpu  # reaching here at all is the measurement: the GPU had capacity
    return probe


register(app)
