import json
import subprocess
from pathlib import Path

import pytest

from talos import inside


def make_monorepo(tmp_path: Path) -> Path:
    mono = tmp_path / "mono"
    (mono / "tig-algorithms" / "src" / "knapsack").mkdir(parents=True)
    (mono / "tig-algorithms" / "src" / "knapsack" / "mod.rs").write_text("// c003_a001\n")
    return mono


class Result:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


def test_stage_writes_files_and_registers_module_once(tmp_path):
    mono = make_monorepo(tmp_path)
    inside.stage_algorithm(mono, "knapsack", {"mod.rs": "fn x(){}", "ls.rs": "fn y(){}"}, "talos_cand")
    d = mono / "tig-algorithms" / "src" / "knapsack" / "talos_cand"
    assert (d / "mod.rs").read_text() == "fn x(){}" and (d / "ls.rs").exists()
    modrs = (mono / "tig-algorithms" / "src" / "knapsack" / "mod.rs").read_text()
    assert modrs.count("pub mod talos_cand;") == 1
    # mutation: unstage that leaves the line behind breaks the next compile
    inside.unstage_algorithm(mono, "knapsack", "talos_cand")
    assert not d.exists()
    assert (mono / "tig-algorithms" / "src" / "knapsack" / "mod.rs").read_text() == "// c003_a001\n"


@pytest.mark.parametrize("rel", ["../evil.rs", "", ".", "sub/../../evil.rs", "/etc/evil.rs"])
def test_stage_rejects_path_escape(tmp_path, rel):
    mono = make_monorepo(tmp_path)
    # mutation: a containment check that only compares resolved prefixes lets "", "." and
    # "sub/../../evil.rs" through, and they then raise IsADirectoryError deep inside the
    # container instead of reporting a failed compile
    with pytest.raises(ValueError):
        inside.stage_algorithm(mono, "knapsack", {rel: "x"}, "talos_cand")
    assert not list(tmp_path.rglob("evil.rs"))


def test_build_invokes_build_algorithm_and_reports(tmp_path):
    calls = []

    def run(cmd, **kw):
        calls.append((cmd, kw.get("cwd")))
        return Result(0, "ok", "")

    # mutation: running build_algorithm without cwd=monorepo makes every compile fail
    ok, out = inside.build(tmp_path, "knapsack", "talos_cand", run)
    assert ok and calls[0][0] == ["build_algorithm", "talos_cand"] and calls[0][1] == tmp_path


def test_classify():
    # exit codes from tig-runtime/src/main.rs and tig-verifier/src/main.rs at MONOREPO_REF
    # mutation: keying ok off runtime rc instead of verifier rc + quality
    assert inside.classify(87, 0, 500, False) == (True, None)        # out of fuel but solved
    assert inside.classify(87, 1, None, False) == (False, "out_of_fuel")
    assert inside.classify(84, 1, None, False) == (False, "no_solution")  # "Runtime Error" exit
    assert inside.classify(0, 1, None, False) == (False, "invalid")       # verifier rejected it
    assert inside.classify(-11, 1, None, False) == (False, "panic")       # killed by a signal
    assert inside.classify(101, 1, None, False) == (False, "panic")       # rust panic exit code
    assert inside.classify(0, 0, None, True) == (False, "timeout")
    assert inside.classify(0, 0, 7, False) == (True, None)


def test_run_nonce_builds_commands_and_parses_quality(tmp_path):
    seen = []

    def run(cmd, **kw):
        seen.append(cmd)
        if cmd[0] == "tig-runtime":
            # tig-runtime's --output names a FOLDER and it writes <nonce>.json inside it.
            # mutation: passing a file path there makes tig-runtime mkdir a folder of that name
            out_dir = Path(cmd[cmd.index("--output") + 1])
            assert out_dir.is_dir(), "--output must name an existing folder"
            (out_dir / f"{cmd[3]}.json").write_text('{"solution": "e30="}')
            return Result(0, "", "")
        return Result(0, "quality: 4242\n", "")

    row = inside.run_nonce("c003", "n=1", "ab" * 32, 7, Path("/lib/x.so"), 10, 600, None, run, tmp_path)
    assert row["ok"] and row["quality"] == 4242 and row["nonce"] == 7 and row["track"] == "n=1"
    rt, ver = seen
    settings = json.loads(rt[1])
    assert settings == {"algorithm_id": "", "challenge_id": "c003", "track_id": "n=1",
                        "block_id": "", "player_id": ""}
    assert rt[2] == "ab" * 32 and rt[3] == "7" and rt[4] == str(Path("/lib/x.so"))
    assert "--fuel" in rt and rt[rt.index("--fuel") + 1] == "10"
    # tig-verifier takes positional SETTINGS RAND_HASH NONCE SOLUTION_FILE and has no subcommand
    # mutation: inserting a "verify_solution" argv[1] makes clap reject every call
    assert ver[:4] == ["tig-verifier", rt[1], "ab" * 32, "7"] and Path(ver[4]).name == "7.json"
    assert len(ver) == 5
    assert "--ptx" not in rt  # CPU challenge


def test_run_nonce_gpu_passes_ptx_and_gpu_to_both_binaries(tmp_path):
    seen = []

    def run(cmd, **kw):
        seen.append(cmd)
        if cmd[0] == "tig-runtime":
            (Path(cmd[cmd.index("--output") + 1]) / f"{cmd[3]}.json").write_text("{}")
        return Result(0, "quality: 1\n", "")

    inside.run_nonce("c005", "k=1", "ab" * 32, 3, Path("/a.so"), 10, 600, Path("/a.ptx"), run, tmp_path)
    rt, ver = seen
    for cmd in (rt, ver):  # mutation: dropping --gpu from the verifier call breaks GPU challenges
        assert cmd[cmd.index("--ptx") + 1] == str(Path("/a.ptx")) and cmd[cmd.index("--gpu") + 1] == "0"


def test_run_nonce_classifies_no_solution(tmp_path):
    def run(cmd, **kw):
        if cmd[0] == "tig-runtime":
            # tig-runtime writes an empty solution and exits 84 when the algorithm returns Err
            (Path(cmd[cmd.index("--output") + 1]) / f"{cmd[3]}.json").write_text('{"solution": ""}')
            return Result(84, "", "Runtime Error: no solution")
        return Result(1, "", "Verification error: Invalid solution")

    # mutation: reporting ok on a non-zero verifier rc would let junk solutions score
    row = inside.run_nonce("c003", "n=1", "ab" * 32, 1, Path("/x.so"), 10, 600, None, run, tmp_path)
    assert not row["ok"] and row["error"] == "no_solution" and row["quality"] is None


def test_verifier_gets_only_the_time_the_runtime_left(tmp_path):
    # The Modal function timeout is NONCE_TIMEOUT_S + 120, so a runtime and a verifier that each
    # get the full per-nonce timeout can exceed it and have the container killed.
    # mutation: passing timeout_s to the verifier makes the pair's worst case 2 x timeout_s
    seen = []
    now = [0.0]

    def run(cmd, **kw):
        seen.append((cmd[0], kw.get("timeout")))
        if cmd[0] == "tig-runtime":
            now[0] += 500.0  # the runtime took 500s of the 600s budget
            (Path(cmd[cmd.index("--output") + 1]) / f"{cmd[3]}.json").write_text("{}")
            return Result(0, "", "")
        return Result(0, "quality: 1\n", "")

    row = inside.run_nonce("c003", "n=1", "ab" * 32, 1, Path("/x.so"), 10, 600, None, run,
                           tmp_path, clock=lambda: now[0])
    assert seen[0] == ("tig-runtime", 600)
    assert seen[1] == ("tig-verifier", 100)
    assert row["runtime_ms"] == 500_000
    # mutation: `timeout_s - elapsed` without the floor passes 0 or a negative timeout, which
    # subprocess treats as "expire immediately"
    now[0] = 0.0
    seen.clear()

    def slow(cmd, **kw):
        seen.append((cmd[0], kw.get("timeout")))
        if cmd[0] == "tig-runtime":
            now[0] += 599.9
            (Path(cmd[cmd.index("--output") + 1]) / f"{cmd[3]}.json").write_text("{}")
        return Result(0, "quality: 1\n", "")

    inside.run_nonce("c003", "n=1", "ab" * 32, 1, Path("/x.so"), 10, 600, None, slow, tmp_path,
                     clock=lambda: now[0])
    assert seen[1] == ("tig-verifier", 1)


def test_run_nonce_handles_a_verifier_timeout(tmp_path):
    def run(cmd, **kw):
        if cmd[0] == "tig-runtime":
            (Path(cmd[cmd.index("--output") + 1]) / f"{cmd[3]}.json").write_text("{}")
            return Result(0, "", "")
        raise subprocess.TimeoutExpired(cmd, 1)

    # mutation: an unhandled verifier timeout raises through Modal and puts the argv
    # (with rand_hash) in the client error
    row = inside.run_nonce("c003", "n=1", "ab" * 32, 1, Path("/x.so"), 10, 600, None, run, tmp_path)
    assert row["error"] == "timeout" and not row["ok"] and row["quality"] is None


def test_build_keeps_the_diagnostics_past_20k_of_other_algorithms_warnings(tmp_path):
    # In run 20260916-095103 every successful build log came back exactly 20000 bytes, and
    # iteration 5's kept 6 of its 8 errors: the raw cap was applied before anything filtered
    # the other algorithms' warnings out. mutation: restoring `out[-20000:]` drops the error
    # and the candidate's own dead-code warning, which sit before the foreign spam
    from talos.diagnostics import dead_new_functions, relevant
    own_error = ("error[E0308]: mismatched types\n"
                 "   --> tig-algorithms/src/knapsack/talos_cand/mod.rs:9:17\n\n")
    own_dead = ("warning: function `polish` is never used\n"
                "   --> tig-algorithms/src/knapsack/talos_cand/mod.rs:3:4\n\n")
    foreign = ("warning: unused variable: `snap`\n"
               "    --> tig-algorithms/src/knapsack/superfast_knap_v1/track5.rs:2069:17\n"
               "     |\n2069 |             let snap = state.clone_solution();\n"
               "     |                 ^^^^ help: prefix it with an underscore\n\n")
    stderr = own_error + own_dead + foreign * 100
    assert len(stderr) > 20000
    _, out = inside.build(tmp_path, "knapsack", "talos_cand",
                          lambda cmd, **kw: Result(1, "", stderr))
    assert own_error in relevant(out)
    assert dead_new_functions(out, {"mod.rs": ["solve"]}) == ["mod.rs: polish"]


def _capture_run(seen):
    def run(cmd, **kw):
        seen.append(cmd)
        if cmd[0] == "tig-runtime":
            (Path(cmd[cmd.index("--output") + 1]) / f"{cmd[3]}.json").write_text("{}")
        return Result(0, "quality: 1\n", "")
    return run


@pytest.mark.parametrize("hp, flag", [
    ({"b": [2, 3], "a": 1}, '{"b":[2,3],"a":1}'),
    ({}, "{}"),
])
def test_run_nonce_passes_hyperparameters_to_the_runtime_only(tmp_path, hp, flag):
    seen = []
    inside.run_nonce("c003", "n=1", "ab" * 32, 7, Path("/lib/x.so"), 10, 600, None,
                     _capture_run(seen), tmp_path, hyperparameters=hp)
    rt, ver = seen
    # mutation: `if hyperparameters:` skips the flag for {}, which the benchmark ran with
    # mutation: json.dumps without compact separators still parses, but pins a different argv
    assert rt[rt.index("--hyperparameters") + 1] == flag
    # mutation: appending the flag to the verifier call makes clap reject every verification
    assert "--hyperparameters" not in ver


def test_run_nonce_without_hyperparameters_passes_no_flag(tmp_path):
    seen = []
    inside.run_nonce("c003", "n=1", "ab" * 32, 7, Path("/lib/x.so"), 10, 600, None,
                     _capture_run(seen), tmp_path)
    # mutation: always passing the flag sends "null", which tig-runtime rejects as not an object
    assert all("--hyperparameters" not in cmd for cmd in seen)


class FakePool:
    """Stands in for `multiprocessing.Pool`: records the worker count and yields the rows back
    to front, so a caller that needs nonce order has to sort."""
    sizes: list[int] = []

    def __init__(self, workers):
        FakePool.sizes.append(workers)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def imap_unordered(self, fn, tasks):
        return [fn(t) for t in reversed(list(tasks))]


def _task(nonce, workdir, so="/lib/x.so", timeout_s=600, hp=None):
    return ("c003", "t", "ab" * 32, nonce, so, 10, timeout_s, None, str(workdir), hp)


def test_run_nonces_spreads_the_tasks_over_a_pool_of_the_given_workers(tmp_path, monkeypatch):
    seen = []

    def stub(task):
        seen.append(task[3])
        return {"track": task[1], "nonce": task[3], "ok": True, "quality": 1, "runtime_ms": 1,
                "error": None}
    monkeypatch.setattr(inside, "run_task", stub)
    FakePool.sizes = []
    rows = list(inside.run_nonces([_task(n, tmp_path) for n in range(6)], 4,
                                  pool_factory=FakePool))
    # mutation: a pool sized to the task count, or to os.cpu_count(), oversubscribes the
    # container's CPU quota; a pool of 1 idles the other cores
    assert FakePool.sizes == [4]
    assert sorted(seen) == list(range(6)) and [r["nonce"] for r in rows] == list(range(5, -1, -1))


def test_run_nonces_with_one_worker_runs_each_task_on_the_injected_runner(tmp_path):
    seen = []
    FakePool.sizes = []
    rows = list(inside.run_nonces([_task(n, tmp_path, hp={"x": n}) for n in range(3)], 1,
                                  run=_capture_run(seen), pool_factory=FakePool))
    # mutation: building a pool for one worker forks a process per nonce for nothing
    assert FakePool.sizes == []
    runtimes = [cmd for cmd in seen if cmd[0] == "tig-runtime"]
    assert [int(cmd[3]) for cmd in runtimes] == [0, 1, 2]
    # mutation: the serial branch dropping a tuple field (here the hyperparameters) runs the
    # nonce with different settings from the pool branch
    assert [cmd[cmd.index("--hyperparameters") + 1] for cmd in runtimes] == [
        '{"x":0}', '{"x":1}', '{"x":2}']
    assert all(r["ok"] and r["quality"] == 1 for r in rows)


def test_run_nonces_never_pools_an_injected_runner_without_an_injected_pool(tmp_path):
    """`run_task` builds its own subprocess calls, so a real pool would ignore a test's fake
    runner and exec a real tig-runtime. Workers above one only take the pool path with the
    real subprocess or a pool the caller supplied."""
    seen = []
    rows = list(inside.run_nonces([_task(n, tmp_path) for n in range(2)], 4,
                                  run=_capture_run(seen)))
    assert len(rows) == 2 and len([c for c in seen if c[0] == "tig-runtime"]) == 2
def test_artifact_paths_follow_the_container_architecture(tmp_path):
    """The TIG build writes lib/<challenge>/<arch>/, amd64 on x86 and arm64 on
    aarch64 hosts; the path must follow the machine the build ran on."""
    from talos import inside

    lib = tmp_path / "tig-algorithms" / "lib" / "knapsack"
    (lib / "arm64").mkdir(parents=True)
    (lib / "amd64").mkdir()
    so, ptx = inside.artifact_paths(tmp_path, "knapsack", "talos_cand", machine="aarch64")
    assert so == lib / "arm64" / "talos_cand.so"
    so, _ = inside.artifact_paths(tmp_path, "knapsack", "talos_cand", machine="x86_64")
    assert so == lib / "amd64" / "talos_cand.so"
    assert ptx is None
