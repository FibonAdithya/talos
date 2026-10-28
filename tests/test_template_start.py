import pytest

from talos.template_start import read_template_root, root_names, template_root


def test_cpu_root_is_mod_rs_only():
    # mutation: adding a .cu for CPU challenges
    assert template_root("knapsack", "RS", None) == {"mod.rs": "RS"}
    assert root_names("knapsack") == ["mod.rs"]


def test_gpu_root_carries_the_cu():
    # mutation: a root of mod.rs only makes every GPU candidate fail to build
    assert template_root("hypergraph", "RS", "CU") == {"mod.rs": "RS", "algorithm.cu": "CU"}
    assert root_names("hypergraph") == ["algorithm.cu", "mod.rs"]


def test_gpu_root_without_cu_raises():
    # mutation: silently returning a root without the .cu
    with pytest.raises(ValueError, match="template.cu"):
        template_root("vector_search", "RS", None)


def test_read_template_root_needs_the_cu_on_gpu(tmp_path):
    (tmp_path / "template.rs").write_text("RS")
    # mutation: reading template.rs alone for a GPU challenge
    assert read_template_root(tmp_path, "hypergraph") is None
    (tmp_path / "template.cu").write_text("CU")
    assert read_template_root(tmp_path, "hypergraph") == {"mod.rs": "RS", "algorithm.cu": "CU"}
    assert read_template_root(tmp_path, "knapsack") == {"mod.rs": "RS"}
    # mutation: a missing template.rs read as an empty root instead of None
    assert read_template_root(tmp_path / "nope", "knapsack") is None
