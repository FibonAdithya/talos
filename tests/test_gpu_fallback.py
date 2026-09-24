"""The GPU preference lists and the per-job GPU choice they feed. The choice is frozen per
job (AGENTS.md invariant 1): every helper here takes the chosen GPU explicitly."""
import pytest

from talos.challenges import (C3_GPU_CLASSES, CHALLENGES, MODAL_GPUS, c3_gpu_options,
                              c3_hardware_class, c3_profile, gpu_options, gpu_slug,
                              hardware_class)


def test_preference_lists_start_with_the_l40s_and_hold_only_48gb_or_larger_gpus():
    # mutation: reordering puts a pricier GPU first on every run; adding an A10/L4/T4 (24 GB
    # or less) can OOM a candidate the baseline fit
    assert MODAL_GPUS == ("L40S", "A100-80GB", "H100")
    assert C3_GPU_CLASSES == ("l40", "a100", "h100")
    assert len(MODAL_GPUS) == len(C3_GPU_CLASSES)


def test_only_gpu_challenges_have_options():
    assert gpu_options(CHALLENGES["hypergraph"]) == MODAL_GPUS
    assert c3_gpu_options(CHALLENGES["hypergraph"]) == C3_GPU_CLASSES
    # mutation: giving CPU challenges a list makes execute_job probe (and pay) for a GPU
    assert gpu_options(CHALLENGES["knapsack"]) == ()
    assert c3_gpu_options(CHALLENGES["knapsack"]) == ()


def test_hardware_class_carries_the_chosen_gpu():
    gpu = CHALLENGES["hypergraph"]
    assert hardware_class(gpu, "L40S") == "gpu-L40S"
    # mutation: ignoring the argument keys an A100 baseline as an L40S one
    assert hardware_class(gpu, "A100-80GB") == "gpu-A100-80GB"
    assert c3_hardware_class(gpu, "l40") == "c3-l40"
    assert c3_hardware_class(gpu, "h100") == "c3-h100"
    assert c3_profile(gpu, "a100") == "a100"
    # the CPU class never takes a GPU, and passing one is ignored rather than fatal
    assert hardware_class(CHALLENGES["knapsack"]) == "cpu4-mem8192"
    assert c3_profile(CHALLENGES["knapsack"]) == "cpu-d3-4vcpu-16gb"


def test_a_gpu_challenge_without_a_chosen_gpu_or_with_an_unknown_one_is_an_error():
    gpu = CHALLENGES["hypergraph"]
    # mutation: defaulting to the first option lets a job frozen to an A100 look up (and hit)
    # an L40S baseline after the state field is lost
    with pytest.raises(ValueError):
        hardware_class(gpu)
    with pytest.raises(ValueError):
        c3_profile(gpu)
    with pytest.raises(ValueError):
        hardware_class(gpu, "T4")
    with pytest.raises(ValueError):
        c3_hardware_class(gpu, "L40S")  # a Modal name is not a C3 class


def test_gpu_slug_is_a_valid_function_name_suffix():
    # mutation: keeping the dash makes "compile_hypergraph_a100-80gb" an invalid Modal name
    assert gpu_slug("A100-80GB") == "a100_80gb"
    assert gpu_slug("L40S") == "l40s"
    assert len({gpu_slug(g) for g in MODAL_GPUS}) == len(MODAL_GPUS)
