import json
from pathlib import Path

from talos import c3_job
from talos.bench import EvalRequest
from talos.c3_jobdir import write_job_dir
from talos.challenges import CHALLENGES
from talos.types import NonceResult, NonceSet

HASH = "ab" * 32


class Result:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


def setup(tmp_path, n=2, baseline_q=100, baseline=True):
    mono = tmp_path / "mono"
    (mono / "tig-algorithms" / "src" / "knapsack").mkdir(parents=True)
    (mono / "tig-algorithms" / "src" / "knapsack" / "mod.rs").write_text("// c003\n")
    base = [NonceResult("t", i, True, baseline_q, 1) for i in range(n)] if baseline else None
    req = EvalRequest("knapsack", {"mod.rs": "fn x(){}"}, [NonceSet("t", HASH, 0, n)],
                      [NonceSet("t", HASH, 1_000_000, n)], 7, base, CHALLENGES["knapsack"].beat)
    work = write_job_dir(tmp_path / "work", req, "1", hardware="cpu-d3-4vcpu-16gb")
    art = tmp_path / "art"
    art.mkdir()
    return mono, work, art


def fake_run(quality=120, build_rc=0):
    """A subprocess stand-in that builds a .so and answers every nonce with `quality`."""
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        if cmd[0] == "build_algorithm":
            mono = Path(kw["cwd"])
            so = mono / "tig-algorithms" / "lib" / "knapsack" / "amd64" / "talos_cand.so"
            so.parent.mkdir(parents=True, exist_ok=True)
            so.write_bytes(b"\x7fELF")
            err = "" if build_rc == 0 else "error[E0308]"
            return Result(build_rc, "Linking talos_cand.so\n", err)
        if cmd[0] == "tig-runtime":
            out_dir = Path(cmd[cmd.index("--output") + 1])
            (out_dir / f"{cmd[3]}.json").write_text("{}")
            return Result(0)
        if cmd[0] == "tig-verifier":
            return Result(0, f"quality: {quality}\n")
        raise AssertionError(cmd)
    run.calls = calls
    return run


def test_compile_failure_writes_compile_only_results_and_exits_zero(tmp_path):
    mono, work, art = setup(tmp_path)
    rc = c3_job.main(workdir=work, artifacts_dir=art, run=fake_run(build_rc=1), monorepo=mono,
                     log=lambda *a: None)
    # mutation: exit 1 turns a compile error into a FAILED job, and the client resubmits it
    assert rc == 0
    r = json.loads((art / "results.json").read_text())
    assert r["compile"]["ok"] is False and "error[E0308]" in r["compile"]["output"]
    assert r["training"] == [] and r["holdout"] is None and r["holdout_reason"] == "not_compiled"
    assert r["started"] == {"training": False, "holdout": False}
    assert "error[E0308]" in (art / "build.log").read_text()


def test_win_scores_holdout_and_records_started_flags(tmp_path):
    mono, work, art = setup(tmp_path, baseline_q=100)
    rc = c3_job.main(workdir=work, artifacts_dir=art, run=fake_run(quality=120), monorepo=mono,
                     log=lambda *a: None)
    assert rc == 0
    r = json.loads((art / "results.json").read_text())
    assert r["compile"]["ok"] and len(r["compile"]["artifact_id"]) == 32
    assert [x["nonce"] for x in r["training"]] == [0, 1]
    assert [x["quality"] for x in r["training"]] == [120, 120]
    # mutation: skipping the held-out set on a win leaves the loop with nothing to confirm
    assert r["holdout_reason"] == "won"
    assert [x["nonce"] for x in r["holdout"]] == [1_000_000, 1_000_001]
    assert r["started"] == {"training": True, "holdout": True}


def test_loss_skips_holdout(tmp_path):
    mono, work, art = setup(tmp_path, baseline_q=200)
    c3_job.main(workdir=work, artifacts_dir=art, run=fake_run(quality=120), monorepo=mono,
                log=lambda *a: None)
    r = json.loads((art / "results.json").read_text())
    # mutation: unconditional held-out scoring doubles every losing job's runtime
    assert r["holdout"] is None and r["holdout_reason"] == "not_won"
    assert r["started"] == {"training": True, "holdout": False}


def test_no_baseline_forces_holdout(tmp_path):
    mono, work, art = setup(tmp_path, baseline=False)
    c3_job.main(workdir=work, artifacts_dir=art, run=fake_run(quality=5), monorepo=mono,
                log=lambda *a: None)
    r = json.loads((art / "results.json").read_text())
    # mutation: treating a null baseline as a loss (skipping held-out) leaves the baseline
    # measurement with nothing to record
    assert r["holdout_reason"] == "forced"
    assert [x["nonce"] for x in r["holdout"]] == [1_000_000, 1_000_001]
    assert [x["quality"] for x in r["holdout"]] == [5, 5]
    assert r["started"] == {"training": True, "holdout": True}


def test_staging_error_is_a_compile_failure_not_a_crash(tmp_path):
    mono = tmp_path / "mono"
    (mono / "tig-algorithms" / "src" / "knapsack").mkdir(parents=True)
    (mono / "tig-algorithms" / "src" / "knapsack" / "mod.rs").write_text("// c003\n")
    base = [NonceResult("t", i, True, 100, 1) for i in range(2)]
    req = EvalRequest("knapsack", {"../evil.rs": "x"}, [NonceSet("t", HASH, 0, 2)],
                      [NonceSet("t", HASH, 1_000_000, 2)], 7, base, CHALLENGES["knapsack"].beat)
    work = write_job_dir(tmp_path / "work", req, "1", hardware="cpu-d3-4vcpu-16gb")
    art = tmp_path / "art"
    art.mkdir()
    # mutation: an uncaught ValueError from stage_algorithm's path check crashes the job and
    # loses even the compile-only result instead of reporting a deterministic failure
    rc = c3_job.main(workdir=work, artifacts_dir=art, run=fake_run(), monorepo=mono,
                     log=lambda *a: None)
    assert rc == 0
    r = json.loads((art / "results.json").read_text())
    assert r["compile"]["ok"] is False
    assert "escapes algorithm dir" in r["compile"]["output"]
    assert r["training"] == [] and r["holdout"] is None
    assert r["holdout_reason"] == "not_compiled"
    assert r["started"] == {"training": False, "holdout": False}
    assert (art / "build.log").exists()


def test_results_are_written_after_every_nonce(tmp_path):
    # mutation: writing results.json only at the end loses every scored nonce when C3 kills the
    # job at its time limit
    mono, work, art = setup(tmp_path, n=3, baseline_q=200)
    snapshots = []
    inner = fake_run(quality=120)

    def run(cmd, **kw):
        if cmd[0] == "tig-runtime" and (art / "results.json").exists():
            snapshots.append(len(json.loads((art / "results.json").read_text())["training"]))
        return inner(cmd, **kw)
    c3_job.main(workdir=work, artifacts_dir=art, run=run, monorepo=mono, log=lambda *a: None)
    assert snapshots == [0, 1, 2]


def test_log_lines_never_carry_the_rand_hash_or_argv(tmp_path):
    mono, work, art = setup(tmp_path)
    lines = []
    c3_job.main(workdir=work, artifacts_dir=art, run=fake_run(), monorepo=mono, log=lines.append)
    # mutation: logging the runtime command line prints the rand hash into `c3 logs`
    assert lines and all(HASH not in ln and "tig-runtime" not in ln for ln in lines)
    assert any("nonce" in ln for ln in lines)


def test_a_job_cut_off_mid_training_does_not_still_claim_not_compiled(tmp_path):
    mono, work, art = setup(tmp_path, n=2, baseline_q=200)
    snaps = []
    inner = fake_run(quality=120)

    def run(cmd, **kw):
        if cmd[0] == "tig-runtime" and not snaps:
            snaps.append(json.loads((art / "results.json").read_text()))
        return inner(cmd, **kw)
    c3_job.main(workdir=work, artifacts_dir=art, run=run, monorepo=mono, log=lambda *a: None)
    # mutation: leaving holdout_reason at "not_compiled" once the compile has succeeded makes
    # a job killed mid-training ship compile.ok=True with "held-out not scored (not_compiled)"
    assert snaps[0]["compile"]["ok"] is True and snaps[0]["holdout_reason"] == "timeout"
    # mutation: setting it and never overwriting it turns every finished job into a timeout
    assert json.loads((art / "results.json").read_text())["holdout_reason"] == "not_won"


def test_the_artifacts_dir_is_created_when_c3_did_not_make_it(tmp_path):
    mono, work, _ = setup(tmp_path)
    art = tmp_path / "fresh" / "art"
    # mutation: dropping the mkdir raises FileNotFoundError on the first write_text, and in
    # the staging-error branch it raises inside the `except`, losing the compile-only result
    c3_job.main(workdir=work, artifacts_dir=art, run=fake_run(), monorepo=mono,
                log=lambda *a: None)
    assert (art / "results.json").exists() and (art / "build.log").exists()


class FakePool:
    """Stands in for `multiprocessing.Pool`; yields the rows back to front so the sort in
    `main` is what puts results.json in nonce order, not the order the workers finished."""

    def __init__(self, workers):
        self.workers = workers
        FakePool.sizes.append(workers)

    sizes: list[int] = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def imap_unordered(self, fn, tasks):
        return [fn(t) for t in reversed(list(tasks))]


def test_the_pool_path_passes_run_one_positional_tuples_and_sorts_the_rows(tmp_path,
                                                                          monkeypatch):
    from talos import inside
    mono, work, art = setup(tmp_path, n=2, baseline_q=200)
    seen = []

    def stub(task):
        seen.append(task)
        return {"track": task[1], "nonce": task[3], "ok": True, "quality": 120,
                "runtime_ms": 5, "error": None}
    monkeypatch.setattr(c3_job, "_run_one", stub)
    FakePool.sizes = []
    c3_job.main(workdir=work, artifacts_dir=art, run=fake_run(), monorepo=mono,
                log=lambda *a: None, pool_factory=FakePool)
    assert FakePool.sizes == [4]  # payload["workers"] for a CPU challenge
    so, _ptx = inside.artifact_paths(mono, "knapsack", inside.ALGO_NAME)
    # mutation: swapping two positions in the task tuple (say nonce and fuel) sends the wrong
    # nonce or the wrong fuel to run_nonce, and _run_one unpacks it without noticing
    assert {t[3]: t for t in seen}[0] == ("c003", "t", HASH, 0, str(so), 7,
                                          inside.NONCE_TIMEOUT_S, None, str(mono), None)
    assert [t[3] for t in seen] == [1, 0]  # the pool really did finish out of order
    r = json.loads((art / "results.json").read_text())
    # mutation: dropping the sort in `scored` writes the rows in completion order, and the
    # loop's nonce-by-nonce comparison against the baseline then lines up the wrong pairs
    assert [(x["track"], x["nonce"]) for x in r["training"]] == [("t", 0), ("t", 1)]


DEAD_WARNING = ("warning: function `polish` is never used\n"
                "   --> tig-algorithms/src/knapsack/talos_cand/mod.rs:3:4\n")


def test_a_dead_new_function_ends_the_job_after_the_build(tmp_path):
    # iteration 7 of run 20260916-095103 compiled, its new function was never called, and
    # 25 minutes of scoring reproduced the baseline exactly.
    # mutation: scoring anyway, or reporting the reason as not_won, hides the no-op
    mono, work, art = setup(tmp_path)
    payload = json.loads((work / "payload.json").read_text())
    payload["prior_functions"] = {"mod.rs": ["x"]}
    (work / "payload.json").write_text(json.dumps(payload))
    run = fake_run()
    real = run

    def with_warning(cmd, **kw):
        r = real(cmd, **kw)
        if cmd[0] == "build_algorithm":
            r.stderr = DEAD_WARNING
        return r
    rc = c3_job.main(workdir=work, artifacts_dir=art, run=with_warning, monorepo=mono,
                     log=lambda *a: None)
    assert rc == 0
    out = json.loads((art / "results.json").read_text())
    assert out["compile"]["ok"] is True
    assert out["holdout_reason"] == "dead_code" and out["training"] == []
    assert out["started"]["training"] is False
    assert "polish" in out["compile"]["output"]
    assert not any(c[0] == "tig-runtime" for c in run.calls)


def test_a_dead_function_the_prior_code_already_had_is_scored(tmp_path):
    # mutation: ignoring prior_functions fails every candidate edited from a baseline that
    # carries its own dead code
    mono, work, art = setup(tmp_path)
    payload = json.loads((work / "payload.json").read_text())
    payload["prior_functions"] = {"mod.rs": ["x", "polish"]}
    (work / "payload.json").write_text(json.dumps(payload))
    real = fake_run()

    def with_warning(cmd, **kw):
        r = real(cmd, **kw)
        if cmd[0] == "build_algorithm":
            r.stderr = DEAD_WARNING
        return r
    c3_job.main(workdir=work, artifacts_dir=art, run=with_warning, monorepo=mono,
                log=lambda *a: None)
    out = json.loads((art / "results.json").read_text())
    assert out["started"]["training"] is True and len(out["training"]) == 2


def test_per_track_timeouts_reach_each_nonce_task(tmp_path, monkeypatch):
    from talos import inside
    mono, work, art = setup(tmp_path, n=2, baseline_q=200)
    payload = json.loads((work / "payload.json").read_text())
    payload["timeouts"] = {"t": 42}
    (work / "payload.json").write_text(json.dumps(payload))
    seen = []

    def stub(task):
        seen.append(task)
        return {"track": task[1], "nonce": task[3], "ok": True, "quality": 120,
                "runtime_ms": 5, "error": None}
    monkeypatch.setattr(c3_job, "_run_one", stub)
    c3_job.main(workdir=work, artifacts_dir=art, run=fake_run(), monorepo=mono,
                log=lambda *a: None, pool_factory=FakePool)
    # mutation: reading nonce_timeout_s alone ignores the per-track cap the loop computed
    assert {t[6] for t in seen} == {42}
    assert inside.NONCE_TIMEOUT_S != 42


def test_each_track_gets_its_own_hyperparameters_in_the_pool_tasks(tmp_path, monkeypatch):
    mono, work, art = setup(tmp_path, n=1, baseline_q=200)
    payload = json.loads((work / "payload.json").read_text())
    payload["training"].append({"track": "u", "rand_hash": HASH, "start": 0, "count": 1})
    payload["hyperparameters"] = {"t": {"x": 1}, "u": None}
    (work / "payload.json").write_text(json.dumps(payload))
    seen = []

    def stub(task):
        seen.append(task)
        return {"track": task[1], "nonce": task[3], "ok": True, "quality": 120,
                "runtime_ms": 5, "error": None}
    monkeypatch.setattr(c3_job, "_run_one", stub)
    c3_job.main(workdir=work, artifacts_dir=art, run=fake_run(), monorepo=mono,
                log=lambda *a: None, pool_factory=FakePool)
    # mutation: putting the whole map in the task hands tig-runtime {"t": ..., "u": ...}
    assert sorted((t[1], json.dumps(t[9])) for t in seen) == [("t", '{"x": 1}'), ("u", "null")]


def test_the_serial_path_passes_the_hyperparameters_to_tig_runtime(tmp_path):
    mono, work, art = setup(tmp_path, n=1, baseline_q=200)
    payload = json.loads((work / "payload.json").read_text())
    payload["hyperparameters"] = {"t": {"x": 1}}
    (work / "payload.json").write_text(json.dumps(payload))
    run = fake_run()
    c3_job.main(workdir=work, artifacts_dir=art, run=run, monorepo=mono, log=lambda *a: None)
    runtime = [c for c in run.calls if c[0] == "tig-runtime"]
    # mutation: the serial branch not passing t[9] runs local and CPU jobs without the map
    assert runtime and all(c[c.index("--hyperparameters") + 1] == '{"x":1}' for c in runtime)


def test_a_leftover_candidate_from_a_previous_job_is_removed_before_staging(tmp_path):
    mono, work, art = setup(tmp_path)
    left = mono / "tig-algorithms" / "src" / "knapsack" / "talos_cand"
    left.mkdir()
    (left / "extra.rs").write_text("fn stale() {}\n")
    mod_rs = mono / "tig-algorithms" / "src" / "knapsack" / "mod.rs"
    mod_rs.write_text(mod_rs.read_text() + "pub mod talos_cand;\n")
    c3_job.main(workdir=work, artifacts_dir=art, run=fake_run(), monorepo=mono,
                log=lambda *a: None)
    # mutation: on a persistent /app volume the previous candidate's extra file would be
    # compiled into this candidate
    assert not (left / "extra.rs").exists() and (left / "mod.rs").exists()
    assert mod_rs.read_text().count("pub mod talos_cand;") == 1
