"""Docker transport for the local backend: one detached container per evaluate call, driven
through the `docker` CLI. It satisfies talos.c3_transport.C3Transport, so C3Bench runs a local
job exactly as it runs a C3 one. Every subprocess call goes through the injected runner.

The flags in run_args are the sandbox for LLM-authored code on the user's machine. Loosening
them is a human decision (AGENTS.md, "What requires a human")."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from talos.c3_bench import C3CommandError
from talos.challenges import DEV_IMAGE_TAG, MONOREPO_REF
from talos.executables import argv0

APP = "/app"
WORK = "/work"
ARTIFACTS = "/artifacts"
PIDS_LIMIT = 4096
LABEL_LIMIT = "talos.time_limit_s"
LABEL_RUN = "talos.run"
LABEL_CARGO = "talos.cargo_home"
LABEL_RUSTUP = "talos.rustup_home"


def host_uid() -> str | None:
    """`uid:gid` for `docker run --user`, so files on the bind mounts belong to the user. None
    on Windows: there is no getuid, and Docker Desktop maps bind-mount ownership itself."""
    if os.name == "nt":
        return None
    return f"{os.getuid()}:{os.getgid()}"


def volume_key() -> str:
    """Both pins, so a pin bump gets fresh volumes and never reuses a target directory built
    against another monorepo (AGENTS.md invariant 3)."""
    return hashlib.sha256(f"{MONOREPO_REF}\0{DEV_IMAGE_TAG}".encode()).hexdigest()[:12]


def volume_names(challenge: str) -> tuple[str, str]:
    key = volume_key()
    return f"talos-app-{challenge}-{key}", f"talos-cargo-{challenge}-{key}"


def run_key(run_dir: Path) -> str:
    return hashlib.sha256(str(Path(run_dir).resolve()).encode()).hexdigest()[:8]


def container_name(run_dir: Path, purpose: str, request_hash: str) -> str:
    """Also the job id C3Bench records, so it must match c3_bench._JOB_ID_RE. The request hash
    is derived from a hash of the rand hash, never the rand hash itself (c3_jobdir.request_hash),
    so the name is safe to show in `docker ps`."""
    return f"talos-{run_key(run_dir)}-{purpose}-{request_hash}"


def parse_time(s: str) -> datetime:
    """Docker's RFC 3339 with nanoseconds; Python's fromisoformat takes at most microseconds."""
    s = s.rstrip("Z")
    if "." in s:
        head, frac = s.split(".", 1)
        s = f"{head}.{frac[:6]}"
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def run_args(local: dict, job_dir: Path, artifacts: Path, name: str, key: str, cargo_home: str,
             rustup_home: str, uid: str | None) -> list[str]:
    """The job container's argv. Each `-v` value is one element, so a path with a space stays
    whole. The cargo and rustup homes are passed explicitly because HOME is overridden."""
    app_vol, cargo_vol = volume_names(local["challenge"])
    args = ["docker", "run", "-d", "--name", name,  # [0] is dropped by DockerTransport.docker
            "--label", f"{LABEL_LIMIT}={local['time_limit_s']}", "--label", f"{LABEL_RUN}={key}",
            "-v", f"{job_dir}:{WORK}:ro", "-v", f"{artifacts}:{ARTIFACTS}",
            "-v", f"{app_vol}:{APP}", "-v", f"{cargo_vol}:{cargo_home}",
            "-e", f"C3_JOB_WORKDIR={WORK}", "-e", f"C3_ARTIFACTS_DIR={ARTIFACTS}",
            "-e", "HOME=/tmp", "-e", f"CARGO_HOME={cargo_home}", "-e", f"RUSTUP_HOME={rustup_home}",
            "--cpus", str(local["cpus"]), "--memory", f"{local['memory_gib']}g",
            "--network", "none", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--pids-limit", str(PIDS_LIMIT)]
    if uid:
        args += ["--user", uid]
    if local["gpu"]:
        args += ["--gpus", "all"]
    return args + [local["image"], "bash", f"{WORK}/job.sh"]


class DockerTransport:
    name = "docker"
    label = "local Docker"

    def __init__(self, run: Callable = subprocess.run,
                 now: Callable[[], datetime] | None = None, uid: str | None = None):
        self._run = run
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._uid = uid

    def docker(self, *args: str, timeout: int = 600, ok: tuple[int, ...] = (0,)) -> str:
        """One `docker` call. A missing binary, a hang, or an exit code outside `ok` is a
        C3CommandError, which C3Bench treats as a poll failure or a failed deploy, never a
        traceback. Never echoes argv: a bind-mount path is fine, but the habit is what keeps
        the rand hash out of messages elsewhere."""
        try:
            r = self._run([argv0("docker"), *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)
        except OSError as e:
            raise C3CommandError(f"docker {args[0]} could not be run: {str(e)[:200]}") from None
        except subprocess.TimeoutExpired:
            raise C3CommandError(f"docker {args[0]} timed out after {timeout}s") from None
        if r.returncode not in ok:
            raise C3CommandError(f"docker {args[0]} failed ({r.returncode}): "
                                 f"{(r.stderr or r.stdout)[-500:]}")
        return r.stdout

    # ── C3Transport ────────────────────────────────────────────────────
    def whoami(self) -> dict:
        raise NotImplementedError("the local backend has no account")

    def balance_gbp(self) -> float | None:
        raise NotImplementedError("the local backend has no balance")

    def deploy(self, job_dir: Path) -> str:
        job_dir = Path(job_dir).resolve()
        local = json.loads((job_dir / "local.json").read_text(encoding="utf-8"))
        run_dir = job_dir.parent.parent  # runs/<job_id>/local/<purpose>
        key = run_key(run_dir)
        name = container_name(run_dir, job_dir.name, local["request_hash"])
        _, cargo_vol = volume_names(local["challenge"])
        cargo_home = self.volume_label(cargo_vol, LABEL_CARGO)
        rustup_home = self.volume_label(cargo_vol, LABEL_RUSTUP)
        artifacts = job_dir / name / ARTIFACTS.strip("/")
        artifacts.mkdir(parents=True, exist_ok=True)  # else Docker creates it root-owned
        self._prune(key)
        self.docker("rm", "-f", name, ok=(0, 1))  # a leftover with this exact name
        self.docker(*run_args(local, job_dir, artifacts, name, key, cargo_home, rustup_home,
                              self._uid)[1:])
        return name

    def _prune(self, key: str) -> None:
        """Exited containers of earlier iterations of this run. Only exited ones: a resumed
        process may still be about to reattach to the newest, and only this run's."""
        ids = self.docker("ps", "-aq", "--filter", f"label={LABEL_RUN}={key}",
                          "--filter", "status=exited").split()
        if ids:
            self.docker("rm", *ids, ok=(0, 1))

    def _inspect(self, job_id: str) -> dict:
        try:
            return json.loads(self.docker("inspect", job_id))[0]
        except (ValueError, IndexError, TypeError) as e:
            raise C3CommandError(f"docker inspect returned no document: {e}") from None

    def status(self, job_id: str) -> str:
        doc = self._inspect(job_id)
        st = doc["State"]
        limit = int(doc["Config"]["Labels"][LABEL_LIMIT])
        status = st.get("Status")
        if status == "created":
            return "PENDING"
        started = parse_time(st["StartedAt"])
        if status == "running":
            if (self._now() - started).total_seconds() >= limit:
                self.docker("kill", job_id, ok=(0, 1))
                return "TIMED_OUT"
            return "RUNNING"
        if status in ("exited", "dead"):
            if (parse_time(st["FinishedAt"]) - started).total_seconds() >= limit:
                return "TIMED_OUT"  # the one we, or a previous process, killed
            return "SUCCEEDED" if st.get("ExitCode") == 0 else "FAILED"
        raise C3CommandError(f"unexpected container status {status!r}")

    def cancel(self, job_id: str) -> None:
        try:
            self.docker("rm", "-f", job_id)
        except C3CommandError:
            pass  # best effort: it may already be gone

    def fetch(self, job_id: str, name: str, dest: Path) -> bool:
        return Path(dest).exists()  # the artifacts directory is a bind mount

    # ── helpers for prepare ────────────────────────────────────────────
    def image_present(self, image: str) -> bool:
        try:
            self.docker("image", "inspect", image)
            return True
        except C3CommandError:
            return False

    def volume_exists(self, volume: str) -> bool:
        try:
            self.docker("volume", "inspect", volume)
            return True
        except C3CommandError:
            return False

    def volume_label(self, volume: str, key: str) -> str:
        out = self.docker("volume", "inspect", "--format", f'{{{{index .Labels "{key}"}}}}',
                          volume).strip()
        if not out:
            raise C3CommandError(f"volume {volume} has no {key} label; run `talos run` again "
                                 f"so prepare can recreate it")
        return out
