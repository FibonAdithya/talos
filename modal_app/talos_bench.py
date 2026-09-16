"""Modal app `talos-bench`. Deployed once into the user's Modal account by `talos setup`.
Every function runs inside the official TIG dev image for its challenge plus a pinned
monorepo checkout at /app. Artifacts live on the `talos-artifacts` Volume keyed by the
content hash of the submitted files."""
from __future__ import annotations

import shutil
from pathlib import Path

import modal

from talos import inside
from talos.challenges import CHALLENGES, DEV_IMAGE_TAG, MONOREPO_REF, dev_image
from talos.inside import NONCE_TIMEOUT_S

APP_NAME = "talos-bench"
ARTIFACTS = "/artifacts"
MONOREPO = Path("/app")

app = modal.App(APP_NAME)
volume = modal.Volume.from_name("talos-artifacts", create_if_missing=True)


def _image(name: str) -> modal.Image:
    return (
        modal.Image.from_registry(dev_image(name), add_python="3.11")
        .apt_install("git")
        .run_commands(
            "git clone https://github.com/tig-foundation/tig-monorepo.git /app",
            f"cd /app && git checkout {MONOREPO_REF}",
        )
        .env({"CHALLENGE": name})
        .add_local_python_source("modal_app", "talos")
    )


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


def _score_impl(name: str, challenge_id: str, artifact_id: str, track: str, rand_hash: str,
                nonce: int, fuel: int, timeout_s: int = NONCE_TIMEOUT_S) -> dict:
    volume.reload()
    d = Path(ARTIFACTS) / name / artifact_id
    so, ptx = d / "algo.so", d / "algo.ptx"
    if not so.exists():
        # Infrastructure, not an algorithm failure: say so plainly rather than letting
        # tig-runtime's exit code be classified as "panic".
        raise FileNotFoundError(f"artifact {artifact_id} missing on volume")
    return inside.run_nonce(challenge_id, track, rand_hash, nonce, so, fuel,
                            min(timeout_s, NONCE_TIMEOUT_S), ptx if ptx.exists() else None,
                            workdir=MONOREPO)


for _name, _spec in CHALLENGES.items():
    _kw = dict(image=_image(_name), volumes={ARTIFACTS: volume}, serialized=True)
    if _spec.is_gpu:
        _kw["gpu"] = _spec.gpu
    else:
        _kw["cpu"] = _spec.cpu
        _kw["memory"] = _spec.memory_mib

    def _mk_compile(n=_name):
        def compile_fn(files: dict) -> dict:
            return _compile_impl(n, files)
        return compile_fn

    def _mk_score(n=_name, cid=_spec.id):
        def score_nonce(artifact_id: str, track: str, rand_hash: str, nonce: int, fuel: int,
                        timeout_s: int = NONCE_TIMEOUT_S) -> dict:
            return _score_impl(n, cid, artifact_id, track, rand_hash, nonce, fuel, timeout_s)
        return score_nonce

    app.function(name=f"compile_{_name}", timeout=3600, **_kw)(_mk_compile())
    app.function(name=f"score_nonce_{_name}", timeout=NONCE_TIMEOUT_S + 120, **_kw)(_mk_score())
