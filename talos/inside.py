"""Container-side logic for the Modal bench. Pure functions that take the subprocess runner
as a parameter, so they are unit-tested on any machine. Runs inside the TIG dev image,
where `build_algorithm`, `tig-runtime` and `tig-verifier` are on PATH and the monorepo
checkout is the working directory."""
from __future__ import annotations

import hashlib
import json
import multiprocessing
import re
import platform
import subprocess
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path

ALGO_NAME = "talos_cand"
# The challenge crate is compiled with the candidate as its only module (see stage_algorithm);
# part of every artifact id and baseline key, so a .so or a baseline measured beside the
# shipped algorithms is never mistaken for one measured in the pruned crate. Bump on any
# change to what the crate contains at build time.
CRATE_LAYOUT = "pruned-1"
PRISTINE_MOD_RS = "mod.rs.talos-pristine"
PRUNED_MARKER = ("// talos: crate pruned to the candidate; the pinned mod.rs is beside this "
                  "file as " + PRISTINE_MOD_RS)
_QUALITY_RE = re.compile(r"quality:\s*(-?\d+)")
# Exit codes, from tig-runtime/src/main.rs at MONOREPO_REF.
RUNTIME_ERROR_RC = 84  # compute_solution returned Err: the algorithm gave up / no solution
OUT_OF_FUEL_RC = 87    # the algorithm library exits 87 when fuel runs out
RUST_PANIC_RC = 101    # a Rust panic that unwinds to main


def _algo_root(monorepo: Path, challenge: str) -> Path:
    return monorepo / "tig-algorithms" / "src" / challenge


def _check_rel(target: Path, rel: str) -> None:
    """Reject anything that is not a plain relative file path strictly inside `target`.
    An LLM-authored file map is untrusted input: `""`, `"."`, `"/etc/evil.rs"` and
    `"sub/../../evil.rs"` all have to fail here rather than at write_text."""
    bad = f"file path escapes algorithm dir: {rel!r}"
    if not rel or rel == "." or Path(rel).is_absolute():
        raise ValueError(bad)
    if any(seg in ("", ".", "..") for seg in rel.split("/")):
        raise ValueError(bad)
    root = target.resolve()
    p = (target / rel).resolve()
    if p == root or not p.is_relative_to(root):  # a "/" prefix test fails on Windows paths
        raise ValueError(bad)


def stage_algorithm(monorepo: Path, challenge: str, files: dict[str, str], name: str) -> None:
    root = _algo_root(monorepo, challenge)
    target = root / name
    for rel in files:
        _check_rel(target, rel)
    target.mkdir(parents=True, exist_ok=True)
    for rel, text in files.items():
        p = target / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8", newline="\n")
    _prune_crate(root, name)


def _prune_crate(root: Path, name: str) -> None:
    """Make `name` the challenge crate's only module, keeping the pinned mod.rs beside it.

    The build compiles every module the crate lists, on one codegen unit (build_so passes
    `-C codegen-units=1`), so the crate's size sets the build cost whatever the candidate is:
    job_scheduling's 14 shipped algorithms are 345k lines and the full crate ran past 2 h and
    23 GB of rustc before it was stopped, against 629 s and 3.4 GB for the candidate alone
    (MEASURED 2026-09-25). TIG's own CI builds an algorithm from its branch, whose mod.rs
    lists only that algorithm, so the pruned crate is the closer match to mainnet, not the
    further one. The pinned file is kept byte for byte so unstage can put it back; it is not
    re-saved when mod.rs already carries the marker, because a job that died before
    unstaging (the local backend's /app volume outlives it) leaves the pruned file in place."""
    mod_rs = root / "mod.rs"
    pristine = root / PRISTINE_MOD_RS
    current = mod_rs.read_bytes()
    if not pristine.exists() and not current.startswith(PRUNED_MARKER.encode()):
        pristine.write_bytes(current)
    mod_rs.write_text(f"{PRUNED_MARKER}\npub mod {name};\n", encoding="utf-8", newline="\n")


def unstage_algorithm(monorepo: Path, challenge: str, name: str) -> None:
    import shutil
    root = _algo_root(monorepo, challenge)
    shutil.rmtree(root / name, ignore_errors=True)
    mod_rs = root / "mod.rs"
    pristine = root / PRISTINE_MOD_RS
    if pristine.exists():
        mod_rs.write_bytes(pristine.read_bytes())
        pristine.unlink()
        return
    # A crate that was never pruned (or whose pristine copy is gone): drop the line alone.
    line = f"pub mod {name};"
    kept = [ln for ln in mod_rs.read_text(encoding="utf-8").splitlines() if ln != line]
    mod_rs.write_text("\n".join(kept) + "\n", encoding="utf-8", newline="\n")


# A cap on what leaves the container, not a window on the diagnostics: every consumer filters
# with diagnostics.relevant first and truncates after. A 20000-byte window here kept 6 of the
# 8 errors of run 20260916-095103 iteration 5, cut off before anything could drop the other
# algorithms' warnings.
BUILD_OUTPUT_CAP = 1_000_000


def build(monorepo: Path, challenge: str, name: str, run=subprocess.run) -> tuple[bool, str]:
    r = run(["build_algorithm", name], cwd=monorepo, capture_output=True, text=True,
            encoding="utf-8", errors="replace")
    out = (r.stdout or "") + (r.stderr or "")
    return r.returncode == 0, out[-BUILD_OUTPUT_CAP:]


# The TIG build writes lib/<challenge>/<arch>/ using Docker's architecture names,
# not uname's: amd64 on x86_64 hosts, arm64 on aarch64 hosts.
_ARCH_DIR = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64", "arm64": "arm64"}


def artifact_paths(monorepo: Path, challenge: str, name: str,
                   machine: str | None = None) -> tuple[Path, Path | None]:
    machine = machine or platform.machine()
    arch = _ARCH_DIR.get(machine, machine)
    lib = monorepo / "tig-algorithms" / "lib" / challenge
    so = lib / arch / f"{name}.so"
    ptx = lib / "ptx" / f"{name}.ptx"
    return so, (ptx if ptx.exists() else None)


def classify(runtime_rc: int, verifier_rc: int, quality: int | None,
             timed_out: bool) -> tuple[bool, str | None]:
    """A run that saved a solution before running out of fuel still counts if it verifies.
    tig-verifier exits 1 for every rejection, so the runtime exit code carries the reason."""
    if timed_out:
        return False, "timeout"
    if verifier_rc == 0 and quality is not None:
        return True, None
    if runtime_rc == OUT_OF_FUEL_RC:
        return False, "out_of_fuel"
    if runtime_rc < 0 or runtime_rc == RUST_PANIC_RC:
        return False, "panic"
    if runtime_rc == RUNTIME_ERROR_RC:
        return False, "no_solution"
    if runtime_rc != 0:
        return False, "panic"
    return False, "invalid"


def run_nonce(challenge_id: str, track: str, rand_hash: str, nonce: int, so: Path, fuel: int,
              timeout_s: int, ptx: Path | None, run=subprocess.run,
              workdir: Path | None = None, clock=time.monotonic,
              hyperparameters: dict | None = None) -> dict:
    """Mirrors scripts/test_algorithm in the monorepo:
    `tig-runtime SETTINGS RAND_HASH NONCE SO --fuel F --output DIR [--hyperparameters JSON]
    [--ptx P --gpu 0]` writes DIR/<nonce>.json, then
    `tig-verifier SETTINGS RAND_HASH NONCE DIR/<nonce>.json [--ptx P --gpu 0]`
    prints `quality: N` and exits 0 on a valid solution. `{}` is passed as `{}`: the algorithm
    receives Some(empty map), not None."""
    settings = json.dumps({"algorithm_id": "", "challenge_id": challenge_id, "track_id": track,
                           "block_id": "", "player_id": ""}, separators=(",", ":"))
    gpu_args = ["--ptx", str(ptx), "--gpu", "0"] if ptx else []
    with tempfile.TemporaryDirectory(dir=workdir) as td:
        out_file = Path(td) / f"{nonce}.json"
        hp_args = ([] if hyperparameters is None else
                   ["--hyperparameters",
                    json.dumps(hyperparameters, separators=(",", ":"))])
        cmd = ["tig-runtime", settings, rand_hash, str(nonce), str(so),
               "--fuel", str(fuel), "--output", td] + hp_args + gpu_args
        t0 = clock()
        timed_out = False
        try:
            r1 = run(cmd, capture_output=True, text=True, timeout=timeout_s,
                     encoding="utf-8", errors="replace")
            rt_rc = r1.returncode
        except subprocess.TimeoutExpired:
            timed_out, rt_rc = True, -1
        elapsed = clock() - t0
        runtime_ms = int(elapsed * 1000)
        quality = None
        ver_rc = 1
        if not timed_out and out_file.exists():
            # The verifier gets only the time the runtime left. Giving it a fresh timeout_s put
            # the worst case at 2 x timeout_s, past the Modal function timeout, which kills the
            # container and turns a slow nonce into an infrastructure error instead of a result.
            # A verifier timeout must not raise either: TimeoutExpired's str carries the whole
            # argv, rand_hash included, and it would surface in a client-side error message.
            try:
                r2 = run(["tig-verifier", settings, rand_hash, str(nonce), str(out_file)] + gpu_args,
                         capture_output=True, text=True,
                         timeout=max(1, int(timeout_s - elapsed)),
                         encoding="utf-8", errors="replace")
                ver_rc = r2.returncode
                m = _QUALITY_RE.search(r2.stdout or "")
                quality = int(m.group(1)) if m else None
            except subprocess.TimeoutExpired:
                timed_out = True
    ok, err = classify(rt_rc, ver_rc, quality, timed_out)
    return {"track": track, "nonce": nonce, "ok": ok, "quality": quality if ok else None,
            "runtime_ms": runtime_ms, "error": err}


NONCE_TIMEOUT_S = 600

# One scoring task, as a positional tuple so it pickles into a worker process unchanged:
# (challenge_id, track, rand_hash, nonce, so, fuel, timeout_s, ptx, workdir, hyperparameters),
# with `so`, `ptx` and `workdir` as strings.
NonceTask = tuple


def run_task(task: NonceTask, run=subprocess.run) -> dict:
    challenge_id, track, rand_hash, nonce, so, fuel, timeout_s, ptx, workdir, hp = task
    return run_nonce(challenge_id, track, rand_hash, nonce, Path(so), fuel, timeout_s,
                     Path(ptx) if ptx else None, run=run, workdir=Path(workdir),
                     hyperparameters=hp)


def run_nonces(tasks: list[NonceTask], workers: int, run=subprocess.run,
               pool_factory=None) -> Iterator[dict]:
    """Scores `tasks` on `workers` processes at once and yields each row as it finishes, in no
    particular order. This is the one place a container's cores are spread over: the C3 and
    local jobs and every Modal batch go through it. `run_task` builds its own subprocess calls
    and cannot carry an injected runner into a worker process, so the pool is only used with
    the real subprocess, or with a `pool_factory` a test supplied to drive that branch."""
    if workers > 1 and (pool_factory is not None or run is subprocess.run):
        with (pool_factory or multiprocessing.Pool)(workers) as pool:
            yield from pool.imap_unordered(run_task, tasks)
    else:
        for t in tasks:
            yield run_task(t, run=run)


def content_hash(files: dict[str, str], monorepo_ref: str, dev_image_tag: str) -> str:
    """Artifact cache key. The monorepo pin, the dev image tag and the crate layout are part
    of it: the same sources built against a different monorepo, or beside a different set of
    modules, are a different .so."""
    h = hashlib.sha256()
    h.update(monorepo_ref.encode())
    h.update(b"\0")
    h.update(dev_image_tag.encode())
    h.update(b"\0")
    h.update(CRATE_LAYOUT.encode())
    h.update(b"\0")
    for k in sorted(files):
        h.update(k.encode())
        h.update(b"\0")
        h.update(files[k].encode())
        h.update(b"\0")
    return h.hexdigest()[:32]
