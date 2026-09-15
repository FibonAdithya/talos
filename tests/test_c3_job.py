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
    work = write_job_dir(tmp_path / "work", req, "1")
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
    assert r["holdout_reason"] == "forced" and len(r["holdout"]) == 2


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
