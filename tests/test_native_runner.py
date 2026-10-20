import ast
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from talos import native_runner

WORKSPACE = """[workspace]
members = [
    "tig-algorithms",
    "tig-binary",
]
exclude = []
resolver = "2"
"""


def test_cpu_render_substitutes_the_challenge_and_calls_the_three_argument_solve():
    files = native_runner.render("knapsack", "talos_cand", is_gpu=False)
    main, cargo = files["talos-native/src/main.rs"], files["talos-native/Cargo.toml"]
    # mutation: a placeholder left unfilled fails the build with an unresolved import
    for text in (main, cargo):
        assert "{CHALLENGE}" not in text and "{ALGORITHM}" not in text
    assert "use tig_algorithms::knapsack::talos_cand as algorithm;" in main
    assert "use tig_challenges::knapsack::*;" in main
    # mutation: the GPU variant chosen for a CPU challenge passes module/stream/prop
    assert "algorithm::solve_challenge(&challenge, &save, &hyperparameters)" in main
    assert "cudarc" not in main and "cudarc" not in cargo
    assert 'features = ["knapsack"]' in cargo


def test_gpu_render_passes_module_stream_and_prop_and_launches_initialize_kernel():
    files = native_runner.render("hypergraph", "talos_cand", is_gpu=True)
    main, cargo = files["talos-native/src/main.rs"], files["talos-native/Cargo.toml"]
    assert "use tig_algorithms::hypergraph::talos_cand as algorithm;" in main
    # mutation: the CPU variant chosen for a GPU challenge does not type-check against
    # hypergraph's six-argument solve_challenge
    assert ("algorithm::solve_challenge(&challenge, &save, &hyperparameters,\n"
            "                                   module.clone(), stream.clone(), &prop)") in main
    assert 'load_function("initialize_kernel")' in main and "seed[8..16]" in main
    # the fuel-check stream synchronises after every launch; the native path must not use it
    assert "default_stream()" in main and "fuel_check_stream" not in main
    # the cudarc source must match tig-challenges' character for character, or cargo resolves
    # two cudarc packages and the CudaModule types stop matching
    assert ('cudarc = { git = "https://github.com/tig-foundation/cudarc.git", '
            'branch = "runtime-fuel/cudnn-cublas", features = '
            '["cuda-version-from-build-system"] }') in cargo


@pytest.mark.parametrize("is_gpu", [False, True])
def test_both_variants_exit_87_at_the_budget_and_84_on_error(is_gpu):
    main = native_runner.render("knapsack", "talos_cand", is_gpu)["talos-native/src/main.rs"]
    # talos/inside.py::classify reads these two codes exactly as it reads tig-runtime's
    assert "const OUT_OF_FUEL: i32 = 87;" in main and "const RUNTIME_ERROR: i32 = 84;" in main
    assert "std::process::exit(OUT_OF_FUEL)" in main
    # mutation: writing the file in place lets the watchdog's exit leave half a JSON document
    assert "std::fs::rename(&tmp, path)" in main
    assert '"solve_us"' in main and "catch_unwind" in main


@pytest.mark.parametrize("is_gpu", [False, True])
def test_both_variants_take_the_fuel_cap_and_start_the_fuel_clock_with_it(is_gpu):
    main = native_runner.render("knapsack", "talos_cand", is_gpu)["talos-native/src/main.rs"]
    # talos/inside.py::run_nonce_native passes `--fuel F`; an unknown flag fails every nonce
    assert '"--fuel" => fuel = Some(value.parse()?)' in main
    # mutation: the watchdog started without the cap leaves __fuel_remaining at 0, and an
    # algorithm that budgets by it degrades to its fallback on every nonce
    assert main.count("watchdog(args.fuel, args.budget_us);") == 1
    assert native_runner._FUEL in main


# An algorithm crate reading the fuel counter exactly as mainnet's energy_arbitrage gridflow
# (track_t3.rs) and job_scheduling ember_stromboli do at MONOREPO_REF.
_ALGO = """
extern "C" {
    #[allow(non_upper_case_globals)]
    static __fuel_remaining: u64;
}

pub fn fuel_remaining() -> u64 {
    unsafe { core::ptr::read_volatile(core::ptr::addr_of!(__fuel_remaining)) }
}
"""

_HARNESS = """
use std::time::{Duration, Instant};
const OUT_OF_FUEL: i32 = 87;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let n = |i: usize| a[i].parse::<u64>().unwrap();
    match a[1].as_str() {
        "left" => println!("{}", fuel_left(n(2), n(3), n(4))),
        "unbudgeted" => {
            watchdog(5_000_000_000_000, None);
            std::thread::sleep(Duration::from_millis(50));
            println!("{}", algo::fuel_remaining());
        }
        "budgeted" => {
            let start = Instant::now();
            watchdog(1_000_000, Some(2_000_000));
            std::thread::sleep(Duration::from_millis(100));
            println!("{} {}", algo::fuel_remaining(), start.elapsed().as_micros());
            std::thread::sleep(Duration::from_secs(10));
            println!("survived");
        }
        _ => unreachable!(),
    }
}
"""


@pytest.fixture(scope="module")
def fuel_bin(tmp_path_factory):
    """The runner's fuel stand-in linked against an algorithm that declares the counter
    extern, built with debug overflow checks on. Without a definition in the runner this link
    fails with `undefined reference to __fuel_remaining`, as the native build of gridflow did
    on 2026-10-20."""
    if shutil.which("rustc") is None:
        pytest.skip("rustc not on PATH")
    d = tmp_path_factory.mktemp("fuel")
    (d / "algo.rs").write_text(_ALGO, encoding="utf-8")
    (d / "main.rs").write_text(native_runner._FUEL + _HARNESS, encoding="utf-8")
    common = ["rustc", "--edition", "2021", "-C", "overflow-checks=on"]
    subprocess.run(common + ["--crate-type", "rlib", "--crate-name", "algo", "algo.rs"],
                   cwd=d, check=True, capture_output=True)
    exe = d / ("fuel.exe" if os.name == "nt" else "fuel")
    r = subprocess.run(common + ["--crate-name", "fuel", "--extern", "algo=libalgo.rlib",
                                 "-L", ".", "-o", str(exe), "main.rs"],
                       cwd=d, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return exe


def _fuel(exe, *args, timeout=30):
    return subprocess.run([str(exe), *map(str, args)], capture_output=True, text=True,
                          timeout=timeout)


@pytest.mark.parametrize("spent, expected", [
    (0, 5_000_000_000_000),
    (3_000_000_000, 2_500_000_000_000),
    # mutation: dropping the `spent >= budget` guard underflows (budget - spent)
    (6_000_000_000, 0),
    (9_000_000_000, 0),
])
def test_fuel_left_falls_linearly_to_zero_at_the_budget(fuel_bin, spent, expected):
    # 5e12 is energy_arbitrage's fuel at MONOREPO_REF; 6e9 us is a 100-minute budget.
    # mutation: u64 arithmetic overflows 5e12 x 3e9 and panics under overflow checks
    r = _fuel(fuel_bin, "left", 5_000_000_000_000, spent, 6_000_000_000)
    assert r.returncode == 0, r.stderr
    assert int(r.stdout) == expected


def test_fuel_left_with_a_zero_budget_is_zero(fuel_bin):
    # mutation: dividing by the budget before checking it panics on a zero budget
    r = _fuel(fuel_bin, "left", 1000, 0, 0)
    assert r.returncode == 0, r.stderr
    assert int(r.stdout) == 0


def test_without_a_budget_the_algorithm_reads_the_full_cap(fuel_bin):
    # calibration runs the baseline with no budget: the algorithm must see all of its fuel,
    # not 0 (its degrade-to-zeros fallback) and not a value that falls (nothing to fall to)
    r = _fuel(fuel_bin, "unbudgeted")
    assert r.returncode == 0, r.stderr
    assert int(r.stdout) == 5_000_000_000_000


def test_with_a_budget_the_algorithm_reads_falling_fuel_then_the_watchdog_exits_87(fuel_bin):
    r = _fuel(fuel_bin, "budgeted")
    assert r.returncode == 87, (r.returncode, r.stdout, r.stderr)
    assert "survived" not in r.stdout
    left, at_us = map(int, r.stdout.split())
    # mutation: no clock thread leaves the cap in place for the whole budget
    assert 0 < left < 1_000_000
    # linear in elapsed time: 1 ms ticks, with 50 ms of slack for a loaded CI runner
    expected = 1_000_000 * (2_000_000 - at_us) // 2_000_000
    assert abs(left - expected) <= 1_000_000 * 50_000 // 2_000_000


def test_add_workspace_member_inserts_once_inside_the_members_list():
    once = native_runner.add_workspace_member(WORKSPACE)
    assert once.count('"talos-native"') == 1
    members = once.split("members = [", 1)[1].split("]", 1)[0]
    # mutation: appending after the list instead of inside it leaves the package outside
    # the workspace, and `cargo build -p talos-native` reports no such package
    assert '"talos-native"' in members
    # mutation: a second insert on the local backend's persistent /app duplicates the member,
    # which cargo rejects
    assert native_runner.add_workspace_member(once) == once


def test_add_workspace_member_rejects_a_manifest_without_members():
    with pytest.raises(ValueError):
        native_runner.add_workspace_member("[package]\nname = \"x\"\n")


def test_stage_writes_the_package_and_registers_it(tmp_path):
    (tmp_path / "Cargo.toml").write_text(WORKSPACE, encoding="utf-8")
    native_runner.stage(tmp_path, "knapsack", "talos_cand", is_gpu=False)
    native_runner.stage(tmp_path, "knapsack", "talos_cand", is_gpu=False)
    assert (tmp_path / "talos-native" / "src" / "main.rs").exists()
    assert (tmp_path / "Cargo.toml").read_text(encoding="utf-8").count('"talos-native"') == 1


def test_native_runner_imports_only_the_standard_library():
    # it is shipped into the C3 job directory, where talos is not installed
    tree = ast.parse(Path(native_runner.__file__).read_text(encoding="utf-8"))
    mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    mods |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not any(m and m.split(".")[0] == "talos" for m in mods)
