import json
import subprocess
from pathlib import Path

import pytest

from modal_app import inside


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
    assert rt[2] == "ab" * 32 and rt[3] == "7" and rt[4] == "/lib/x.so"
    assert "--fuel" in rt and rt[rt.index("--fuel") + 1] == "10"
    # tig-verifier takes positional SETTINGS RAND_HASH NONCE SOLUTION_FILE and has no subcommand
    # mutation: inserting a "verify_solution" argv[1] makes clap reject every call
    assert ver[:4] == ["tig-verifier", rt[1], "ab" * 32, "7"] and ver[4].endswith("/7.json")
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
        assert cmd[cmd.index("--ptx") + 1] == "/a.ptx" and cmd[cmd.index("--gpu") + 1] == "0"


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
