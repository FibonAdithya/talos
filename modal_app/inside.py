"""Container-side logic for the Modal bench. Pure functions that take the subprocess runner
as a parameter, so they are unit-tested on any machine. Runs inside the TIG dev image,
where `build_algorithm`, `tig-runtime` and `tig-verifier` are on PATH and the monorepo
checkout is the working directory."""
from __future__ import annotations

import json
import re
import subprocess
import tempfile
import time
from pathlib import Path

ALGO_NAME = "talos_cand"
_QUALITY_RE = re.compile(r"quality:\s*(-?\d+)")
# Exit codes, from tig-runtime/src/main.rs at MONOREPO_REF.
RUNTIME_ERROR_RC = 84  # compute_solution returned Err: the algorithm gave up / no solution
OUT_OF_FUEL_RC = 87    # the algorithm library exits 87 when fuel runs out
RUST_PANIC_RC = 101    # a Rust panic that unwinds to main


def _algo_root(monorepo: Path, challenge: str) -> Path:
    return monorepo / "tig-algorithms" / "src" / challenge


def stage_algorithm(monorepo: Path, challenge: str, files: dict[str, str], name: str) -> None:
    root = _algo_root(monorepo, challenge)
    target = root / name
    for rel in files:
        p = (target / rel).resolve()
        if not str(p).startswith(str(target.resolve()) + "/") and p != target.resolve():
            raise ValueError(f"file path escapes algorithm dir: {rel}")
    target.mkdir(parents=True, exist_ok=True)
    for rel, text in files.items():
        p = target / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    mod_rs = root / "mod.rs"
    line = f"pub mod {name};"
    existing = mod_rs.read_text()
    if line not in existing.splitlines():
        if not existing.endswith("\n"):
            existing += "\n"
        mod_rs.write_text(existing + line + "\n")


def unstage_algorithm(monorepo: Path, challenge: str, name: str) -> None:
    import shutil
    root = _algo_root(monorepo, challenge)
    shutil.rmtree(root / name, ignore_errors=True)
    mod_rs = root / "mod.rs"
    line = f"pub mod {name};"
    kept = [ln for ln in mod_rs.read_text().splitlines() if ln != line]
    mod_rs.write_text("\n".join(kept) + "\n")


def build(monorepo: Path, challenge: str, name: str, run=subprocess.run) -> tuple[bool, str]:
    r = run(["build_algorithm", name], cwd=monorepo, capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    return r.returncode == 0, out[-20000:]


def artifact_paths(monorepo: Path, challenge: str, name: str) -> tuple[Path, Path | None]:
    lib = monorepo / "tig-algorithms" / "lib" / challenge
    so = lib / "amd64" / f"{name}.so"
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
              workdir: Path | None = None) -> dict:
    """Mirrors scripts/test_algorithm in the monorepo:
    `tig-runtime SETTINGS RAND_HASH NONCE SO --fuel F --output DIR [--ptx P --gpu 0]` writes
    DIR/<nonce>.json, then `tig-verifier SETTINGS RAND_HASH NONCE DIR/<nonce>.json [--ptx P --gpu 0]`
    prints `quality: N` and exits 0 on a valid solution."""
    settings = json.dumps({"algorithm_id": "", "challenge_id": challenge_id, "track_id": track,
                           "block_id": "", "player_id": ""}, separators=(",", ":"))
    gpu_args = ["--ptx", str(ptx), "--gpu", "0"] if ptx else []
    with tempfile.TemporaryDirectory(dir=workdir) as td:
        out_file = Path(td) / f"{nonce}.json"
        cmd = ["tig-runtime", settings, rand_hash, str(nonce), str(so),
               "--fuel", str(fuel), "--output", td] + gpu_args
        t0 = time.time()
        timed_out = False
        try:
            r1 = run(cmd, capture_output=True, text=True, timeout=timeout_s)
            rt_rc = r1.returncode
        except subprocess.TimeoutExpired:
            timed_out, rt_rc = True, -1
        runtime_ms = int((time.time() - t0) * 1000)
        quality = None
        ver_rc = 1
        if not timed_out and out_file.exists():
            r2 = run(["tig-verifier", settings, rand_hash, str(nonce), str(out_file)] + gpu_args,
                     capture_output=True, text=True, timeout=timeout_s)
            ver_rc = r2.returncode
            m = _QUALITY_RE.search(r2.stdout or "")
            quality = int(m.group(1)) if m else None
    ok, err = classify(rt_rc, ver_rc, quality, timed_out)
    return {"track": track, "nonce": nonce, "ok": ok, "quality": quality if ok else None,
            "runtime_ms": runtime_ms, "error": err}
