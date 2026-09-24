"""The hardware preference lists and the per-job choice they feed. The choice is frozen per
job (AGENTS.md invariant 1): every helper here takes the chosen hardware explicitly."""
import pytest

from talos.c3_bench import GBP_PER_HOUR
from talos.challenges import (C3_CPU_PROFILES, C3_GPU_CLASSES, CHALLENGES, MODAL_GPUS,
                              c3_hardware_class, c3_hardware_options, c3_profile, gpu_slug,
                              hardware_class, hardware_options)


def test_gpu_preference_lists_start_with_the_l40s_and_hold_only_48gb_or_larger_gpus():
    # mutation: reordering puts a pricier GPU first on every run; adding an A10/L4/T4 (24 GB
    # or less) can OOM a candidate the baseline fit
    assert MODAL_GPUS == ("L40S", "A100-80GB", "H100")
    assert C3_GPU_CLASSES == ("l40", "a100", "h100")
    assert len(MODAL_GPUS) == len(C3_GPU_CLASSES)


def test_c3_cpu_profiles_are_the_4vcpu_16gb_ones_with_the_established_profile_first():
    # mutation: putting the e2 profile first moves every new CPU job off the profile the
    # existing baselines were measured on; adding cpu-n1-4vcpu-4gb (4 GB, under the 8 GB the
    # challenge spec declares) can OOM a candidate the baseline fit; a 48/96 vCPU profile
    # costs 10-20x for the same 4 workers
    assert C3_CPU_PROFILES == ("cpu-d3-4vcpu-16gb", "cpu-e2-4vcpu-16gb")


def test_every_c3_option_is_priced():
    # mutation: an option without a rate is a KeyError in the first cost estimate, after the
    # job has already run
    for option in C3_GPU_CLASSES + C3_CPU_PROFILES:
        assert option in GBP_PER_HOUR


def test_modal_offers_gpus_to_gpu_challenges_only_and_c3_offers_both_lists():
    assert hardware_options(CHALLENGES["hypergraph"]) == MODAL_GPUS
    assert c3_hardware_options(CHALLENGES["hypergraph"]) == C3_GPU_CLASSES
    # mutation: giving a CPU challenge a Modal list makes execute_job probe (and pay) for a
    # GPU; Modal's CPU containers have one class and nothing to fall back to
    assert hardware_options(CHALLENGES["knapsack"]) == ()
    assert c3_hardware_options(CHALLENGES["knapsack"]) == C3_CPU_PROFILES


def test_hardware_class_carries_the_chosen_hardware():
    gpu, cpu = CHALLENGES["hypergraph"], CHALLENGES["knapsack"]
    assert hardware_class(gpu, "L40S") == "gpu-L40S"
    # mutation: ignoring the argument keys an A100 baseline as an L40S one
    assert hardware_class(gpu, "A100-80GB") == "gpu-A100-80GB"
    assert c3_hardware_class(gpu, "l40") == "c3-l40"
    assert c3_hardware_class(gpu, "h100") == "c3-h100"
    assert c3_profile(gpu, "a100") == "a100"
    # ...and an Ice Lake baseline as a Genoa one
    assert c3_profile(cpu, "cpu-e2-4vcpu-16gb") == "cpu-e2-4vcpu-16gb"
    assert c3_hardware_class(cpu, "cpu-d3-4vcpu-16gb") == "c3-cpu-d3-4vcpu-16gb"
    assert c3_hardware_class(cpu, "cpu-e2-4vcpu-16gb") == "c3-cpu-e2-4vcpu-16gb"
    # the Modal CPU class has nothing to choose, and passing something is ignored, not fatal
    assert hardware_class(cpu) == "cpu4-mem8192-x4"
    assert hardware_class(cpu, "cpu-e2-4vcpu-16gb") == "cpu4-mem8192-x4"


def test_a_challenge_with_options_but_no_chosen_hardware_or_an_unknown_one_is_an_error():
    gpu, cpu = CHALLENGES["hypergraph"], CHALLENGES["knapsack"]
    # mutation: defaulting to the first option lets a job frozen to an A100 (or the e2
    # profile) look up (and hit) an L40S (or d3) baseline after the state field is lost
    with pytest.raises(ValueError):
        hardware_class(gpu)
    with pytest.raises(ValueError):
        c3_profile(gpu)
    with pytest.raises(ValueError):
        c3_profile(cpu)
    with pytest.raises(ValueError):
        hardware_class(gpu, "T4")
    with pytest.raises(ValueError):
        c3_hardware_class(gpu, "L40S")  # a Modal name is not a C3 class
    with pytest.raises(ValueError):
        c3_profile(cpu, "l40")  # a GPU class is not a CPU profile
    with pytest.raises(ValueError):
        c3_profile(cpu, "cpu-n1-4vcpu-4gb")  # a real profile, but not one on the list


def test_gpu_slug_is_a_valid_function_name_suffix():
    # mutation: keeping the dash makes "compile_hypergraph_a100-80gb" an invalid Modal name
    assert gpu_slug("A100-80GB") == "a100_80gb"
    assert gpu_slug("L40S") == "l40s"
    assert len({gpu_slug(g) for g in MODAL_GPUS}) == len(MODAL_GPUS)
