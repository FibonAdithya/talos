"""Compare native compiler parallelism in an isolated, offline TIG dev container.

Run from the Talos checkout:
    python3 scripts/benchmark_native_build.py --monorepo ../tig-monorepo

The source comes from MONOREPO_REF, not the working tree. Each trial uses a fresh
Cargo target directory, then recompiles after a comment edit to the candidate.
Dependency downloads and source staging are outside the timed region. The
existing Cargo cache, if supplied, is mounted read-only and copied before use.
Raw logs, Cargo timing reports and results.json remain in the output directory.
No production build settings, research jobs or baseline caches are changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from talos import inside, native_runner  # noqa: E402
from talos.challenges import CHALLENGES, MONOREPO_REF, dev_image  # noqa: E402


def capture(cmd, **kwargs):
    return subprocess.check_output(cmd, text=True, **kwargs).strip()


def variants(cpus):
    # "current" is the parent branch's recipe: parallel codegen, one frontend thread.
    # Keep it explicit so rerunning after enabling frontend threads retains the control.
    return [
        {"name": "serial", "jobs": 1, "codegen_units": 1, "frontend_threads": 1},
        {"name": "current", "jobs": cpus, "codegen_units": 16, "frontend_threads": 1},
        {"name": "frontend_parallel", "jobs": cpus, "codegen_units": 16,
         "frontend_threads": cpus},
    ]


def summaries(rows):
    result = {}
    for name in sorted({r["variant"] for r in rows}):
        result[name] = {}
        for phase in ("clean", "edited"):
            times = [r["seconds"] for r in rows
                     if r["variant"] == name and r["phase"] == phase and r["returncode"] == 0]
            if times:
                result[name][phase] = {"n": len(times), "median_seconds": statistics.median(times),
                                       "min_seconds": min(times), "max_seconds": max(times)}
    return result


def copy_cache_file(source, destination):
    # Git pack files are read-only. Replace duplicates from the image cache when
    # overlaying the newer cache; opening the existing file for writing would fail.
    dest = Path(destination)
    if dest.exists():
        dest.unlink()
    return shutil.copy2(source, destination)


def run_inside(args):
    os.umask(0o002)
    bench = Path("/bench")
    mono = bench / "source"
    mono.mkdir()
    subprocess.run(["tar", "xf", str(bench / "source.tar"), "--no-same-owner", "-C", str(mono)],
                   check=True)
    old_cargo = Path(os.environ.get("CARGO_HOME", "/root/.cargo"))
    cargo_home = bench / "cargo"
    # Keep rustup's executable/toolchain in the image. Only dependency caches are copied.
    for seed in (old_cargo, Path("/seed-cargo")):
        for name in ("registry", "git"):
            if (seed / name).is_dir():
                shutil.copytree(seed / name, cargo_home / name, dirs_exist_ok=True,
                                copy_function=copy_cache_file)
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("CARGO_PROFILE_", "CARGO_TARGET_", "RUSTFLAGS"))
           and k not in ("CARGO_ENCODED_RUSTFLAGS", "CARGO_INCREMENTAL", "RUSTC_WRAPPER",
                         "RUSTC_WORKSPACE_WRAPPER", "CARGO_BUILD_RUSTFLAGS")}
    env.update(native_runner.NATIVE_ENV)
    env.update(CARGO_HOME=str(cargo_home), CARGO_NET_OFFLINE="true", CARGO_INCREMENTAL="0",
               CARGO_PROFILE_RELEASE_LTO="false", CARGO_PROFILE_RELEASE_OPT_LEVEL="3")
    candidate = mono / "tig-algorithms" / "src" / args.challenge / args.algorithm
    files = {p.relative_to(candidate).as_posix(): p.read_text()
             for p in candidate.rglob("*.rs")}
    if not files:
        raise RuntimeError(f"no Rust sources for {args.challenge}/{args.algorithm} at the pin")
    inside.stage_algorithm(mono, args.challenge, files, inside.ALGO_NAME)
    native_runner.stage(mono, args.challenge, inside.ALGO_NAME, is_gpu=False)
    edit_file = mono / "tig-algorithms" / "src" / args.challenge / inside.ALGO_NAME / "mod.rs"
    original = edit_file.read_text()
    logs = bench / "logs"
    logs.mkdir()
    metadata = json.loads((bench / "metadata.json").read_text())
    metadata.update(rustc=capture(["rustc", native_runner.TOOLCHAIN, "-vV"]),
                    cargo=capture(["cargo", native_runner.TOOLCHAIN, "-V"]),
                    machine=platform.machine(), cpuinfo=Path("/proc/cpuinfo").read_text(),
                    cpu_max=Path("/sys/fs/cgroup/cpu.max").read_text().strip(),
                    native_env=native_runner.NATIVE_ENV,
                    incremental=False, lto="false", opt_level=3,
                    variants=variants(args.cpus))
    rows = []

    def save():
        doc = {"metadata": metadata, "trials": rows, "summary": summaries(rows)}
        (bench / "results.json").write_text(json.dumps(doc, indent=2) + "\n")

    save()
    configs = variants(args.cpus)
    for repeat in range(args.repeats):
        # Rotate order to distribute filesystem warming and thermal/order effects.
        for config in configs[repeat % len(configs):] + configs[:repeat % len(configs)]:
            target = bench / "targets" / f"r{repeat + 1}-{config['name']}"
            build_env = dict(env, CARGO_TARGET_DIR=str(target),
                             CARGO_PROFILE_RELEASE_CODEGEN_UNITS=str(config["codegen_units"]),
                             RUSTFLAGS=f"-Z threads={config['frontend_threads']}")
            cmd = ["cargo", native_runner.TOOLCHAIN, "build", "--release", "-p",
                   native_runner.PACKAGE, "--offline", "--timings", "-j", str(config["jobs"])]
            for phase in ("clean", "edited"):
                edit_file.write_text(original + (f"\n// benchmark edit {repeat}\n"
                                                if phase == "edited" else ""))
                label = f"r{repeat + 1}-{config['name']}-{phase}"
                print(f"START {label}", flush=True)
                started = time.perf_counter()
                with (logs / f"{label}.log").open("w") as log:
                    completed = subprocess.run(cmd, cwd=mono, env=build_env,
                                               stdout=log, stderr=subprocess.STDOUT)
                seconds = time.perf_counter() - started
                log_text = (logs / f"{label}.log").read_text()
                rebuilt = "Compiling tig-algorithms" in log_text
                row = {"repeat": repeat + 1, "variant": config["name"], "phase": phase,
                       "seconds": seconds, "returncode": completed.returncode,
                       "candidate_recompiled": rebuilt, "command": cmd,
                       "rustflags": build_env["RUSTFLAGS"], "log": f"logs/{label}.log"}
                binary = target / "release" / native_runner.PACKAGE
                if completed.returncode == 0:
                    row["binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
                    row["binary_bytes"] = binary.stat().st_size
                rows.append(row)
                save()
                print(f"END {label}: {seconds:.3f}s rc={completed.returncode}", flush=True)
                if completed.returncode or not rebuilt:
                    print(log_text[-8000:], flush=True)
                    raise RuntimeError("build failed or Cargo did not recompile the candidate")
            # Preserve timing reports; remove only this trial's disposable target directory.
            shutil.copytree(target / "cargo-timings", logs / f"r{repeat + 1}-{config['name']}")
            shutil.rmtree(target)
    print(json.dumps(summaries(rows), indent=2), flush=True)


def run_host(args):
    if CHALLENGES[args.challenge].is_gpu:
        raise ValueError("this compiler benchmark currently supports CPU challenges only")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    # Root with all capabilities dropped cannot bypass the host directory mode.
    # Share this disposable directory through the host group, without DAC_OVERRIDE.
    output.chmod(0o775)
    with (output / "source.tar").open("wb") as archive:
        subprocess.run(["git", "archive", MONOREPO_REF], cwd=args.monorepo,
                       stdout=archive, check=True)
    image = args.image or dev_image(args.challenge)
    image_id = capture(["docker", "image", "inspect", "--format", "{{.Id}}", image])
    metadata = {"monorepo_ref": MONOREPO_REF, "talos_commit": capture(
        ["git", "rev-parse", "HEAD"], cwd=ROOT), "image": image, "image_id": image_id,
        "challenge": args.challenge, "algorithm": args.algorithm, "cpus": args.cpus,
        "memory_gib": args.memory_gib, "repeats": args.repeats,
        "source_tar_sha256": hashlib.sha256((output / "source.tar").read_bytes()).hexdigest(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    cmd = ["docker", "run", "--rm", "--name", f"talos-compiler-bench-{os.getpid()}",
           "--user", f"0:{os.getgid()}",
           "--network", "none", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
           "--pids-limit", "4096", "--cpus", str(args.cpus),
           "--memory", f"{args.memory_gib}g", "--memory-swap", f"{args.memory_gib}g",
           "-v", f"{ROOT}:/talos:ro", "-v", f"{output}:/bench", "-e", "PYTHONDONTWRITEBYTECODE=1"]
    if args.cargo_cache:
        # Inspect first so a misspelled name cannot silently create an empty volume.
        capture(["docker", "volume", "inspect", args.cargo_cache])
        cmd += ["-v", f"{args.cargo_cache}:/seed-cargo:ro"]
    cmd += [image_id, "python3", "/talos/scripts/benchmark_native_build.py", "--inside",
            "--challenge", args.challenge, "--algorithm", args.algorithm,
            "--cpus", str(args.cpus), "--repeats", str(args.repeats)]
    print(f"Results: {output}", flush=True)
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monorepo", type=Path, default=ROOT.parent / "tig-monorepo")
    parser.add_argument("--challenge", choices=CHALLENGES, default="knapsack")
    parser.add_argument("--algorithm", default="knap_lean")
    parser.add_argument("--cpus", type=int, default=8)
    parser.add_argument("--memory-gib", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--image", help="existing local dev image; default is the challenge pin")
    parser.add_argument("--cargo-cache", help="existing Docker Cargo cache volume, mounted read-only")
    parser.add_argument("--output", type=Path, default=ROOT / ".talos" / "compiler-benchmark"
                        / time.strftime("%Y%m%d-%H%M%S", time.gmtime()))
    parser.add_argument("--inside", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.cpus, args.memory_gib, args.repeats) < 1:
        parser.error("CPUs, memory and repeats must be positive")
    (run_inside if args.inside else run_host)(args)


if __name__ == "__main__":
    main()
