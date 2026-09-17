"""Runs inside the C3 container: build the candidate, score training nonces, decide on the
held-out set with the shipped baseline, score it on a win. Writes results.json after every
nonce so a job cut off at its time limit still reports what it scored. Log lines carry
timings and nonce numbers only, never the rand hash or a command line."""
from __future__ import annotations

import json
import multiprocessing
import os
import subprocess
import sys
import time
from pathlib import Path

from talos import inside
from talos.challenges import BeatRule
from talos.diagnostics import dead_new_functions, relevant
from talos.scoring import holdout_decision
from talos.types import NonceResult


def _atomic(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def _run_one(task: tuple) -> dict:
    challenge_id, track, rand_hash, nonce, so, fuel, timeout_s, ptx, workdir, hp = task
    return inside.run_nonce(challenge_id, track, rand_hash, nonce, Path(so), fuel, timeout_s,
                            Path(ptx) if ptx else None, workdir=Path(workdir),
                            hyperparameters=hp)


def _score(payload: dict, sets: list[dict], so: Path, ptx: Path | None, monorepo: Path,
           run, log, clock, t0: float, on_row, pool_factory=None) -> None:
    timeouts = payload.get("timeouts") or {}
    # talos.bench is not shipped to the container, so the per-track lookup is inlined here: a
    # track mapped to None, or absent, runs without the flag; {} is passed as {}.
    hyperparameters = payload.get("hyperparameters") or {}
    tasks = [(payload["challenge_id"], ns["track"], ns["rand_hash"], n, str(so), payload["fuel"],
              timeouts.get(ns["track"], payload["nonce_timeout_s"]),
              str(ptx) if ptx else None, str(monorepo), hyperparameters.get(ns["track"]))
             for ns in sets for n in range(ns["start"], ns["start"] + ns["count"])]
    workers = int(payload.get("workers", 1))
    # `_run_one` builds its own subprocess calls and ignores the injected runner, so the real
    # pool is only safe on the real subprocess; an injected pool_factory is a test driving
    # this branch on purpose.
    if workers > 1 and (pool_factory is not None or run is subprocess.run):
        with (pool_factory or multiprocessing.Pool)(workers) as pool:
            rows = pool.imap_unordered(_run_one, tasks)
            for r in rows:
                on_row(r)
                log(f"[{int(clock() - t0)}s] nonce {r['nonce']} ok={r['ok']} q={r['quality']} "
                    f"ms={r['runtime_ms']} err={r['error']}")
    else:
        for t in tasks:
            r = inside.run_nonce(t[0], t[1], t[2], t[3], so, t[5], t[6], ptx, run=run,
                                 workdir=monorepo, hyperparameters=t[9])
            on_row(r)
            log(f"[{int(clock() - t0)}s] nonce {r['nonce']} ok={r['ok']} q={r['quality']} "
                f"ms={r['runtime_ms']} err={r['error']}")


def main(workdir: Path | None = None, artifacts_dir: Path | None = None, run=subprocess.run,
         monorepo: Path = Path("/app"), log=print, clock=time.monotonic, pool_factory=None) -> int:
    t0 = clock()
    workdir = Path(workdir or os.environ["C3_JOB_WORKDIR"])
    art = Path(artifacts_dir or os.environ["C3_ARTIFACTS_DIR"])
    # C3 need not have created the artifacts dir. The staging-error branch writes build.log
    # from inside an `except`, where a second raise would lose the compile-only result.
    art.mkdir(parents=True, exist_ok=True)
    payload = json.loads((workdir / "payload.json").read_text(encoding="utf-8"))
    challenge = payload["challenge"]
    out: dict = {"compile": None, "training": [], "holdout": None,
                 "holdout_reason": "not_compiled",
                 "started": {"training": False, "holdout": False}}
    results = art / "results.json"

    log(f"[0s] challenge={challenge} files={sorted(payload['files'])} "
        f"workers={payload['workers']}")
    try:
        inside.stage_algorithm(monorepo, challenge, payload["files"], inside.ALGO_NAME)
        log(f"[{int(clock() - t0)}s] build start")
        ok, build_out = inside.build(monorepo, challenge, inside.ALGO_NAME, run=run)
    except Exception as e:
        # An application-level failure (e.g. a malformed relative path from an LLM-authored
        # file map) is a result, not a crash: raising here would exit non-zero and lose every
        # result, including the compile-only one this job could still report.
        build_out = f"staging failed: {e}"
        (art / "build.log").write_text(build_out, encoding="utf-8", newline="\n")
        out["compile"] = {"ok": False, "artifact_id": None, "output": build_out}
        _atomic(results, out)
        log(f"[{int(clock() - t0)}s] staging/build raised: {e}")
        return 0
    (art / "build.log").write_text(build_out, encoding="utf-8", newline="\n")
    so, ptx = inside.artifact_paths(monorepo, challenge, inside.ALGO_NAME)
    log(f"[{int(clock() - t0)}s] build ok={ok} so={so.exists()}")
    if not ok or not so.exists():
        out["compile"] = {"ok": False, "artifact_id": None,
                          "output": build_out
                          + ("" if so.exists() else f"\nbuild produced no .so at {so}")}
        _atomic(results, out)
        return 0  # a compile error is a result, not a job failure
    out["compile"] = {"ok": True, "output": relevant(build_out)[-20000:],
                      "artifact_id": inside.content_hash(payload["files"],
                                                         payload["monorepo_ref"],
                                                         payload["dev_image_tag"])}
    prior = payload.get("prior_functions")
    dead = dead_new_functions(build_out, prior) if prior is not None else []
    if dead:
        # The change is not on the solve path; scoring it would repeat the prior result
        # nonce for nonce. The client reads the names back out of the compile output.
        out["holdout_reason"] = "dead_code"
        _atomic(results, out)
        log(f"[{int(clock() - t0)}s] dead new code, not scored: {dead}")
        return 0
    # from here on the candidate has compiled: a job cut off mid-training must not still say
    # "not_compiled". The real decision overwrites this once training finishes.
    out["holdout_reason"] = "timeout"
    _atomic(results, out)

    def scored(key):
        def on_row(r):
            out[key].append(r)
            out[key].sort(key=lambda x: (x["track"], x["nonce"]))
            _atomic(results, out)
        return on_row

    out["started"]["training"] = True
    _atomic(results, out)
    _score(payload, payload["training"], so, ptx, monorepo, run, log, clock, t0,
           scored("training"), pool_factory)
    tr = [NonceResult.from_dict(r) for r in out["training"]]
    base = ([NonceResult.from_dict(r) for r in payload["baseline_training"]]
            if payload["baseline_training"] is not None else None)
    go, reason = holdout_decision(base, tr, BeatRule(**payload["rule"]))
    out["holdout_reason"] = reason
    log(f"[{int(clock() - t0)}s] training done: holdout={go} ({reason})")
    if go:
        out["holdout"] = []
        out["started"]["holdout"] = True
        _atomic(results, out)
        _score(payload, payload["holdout"], so, ptx, monorepo, run, log, clock, t0,
               scored("holdout"), pool_factory)
    _atomic(results, out)
    log(f"[{int(clock() - t0)}s] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
