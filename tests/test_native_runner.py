import ast
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
