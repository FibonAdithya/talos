import json
import os
import stat

import pytest

from talos import c3_jobdir
from talos.bench import EvalRequest
from talos.c3_jobdir import LocalSettings
from talos.challenges import (CHALLENGES, DEV_IMAGE_TAG, MONOREPO_REF, c3_hardware_class, c3_image,
                              c3_profile, c3_workers, dev_image)
from talos.types import NonceResult, NonceSet

HASH = "ab" * 32


def req(challenge="knapsack", n=4, baseline=True):
    tr = [NonceSet("t", HASH, 0, n)]
    ho = [NonceSet("t", HASH, 1_000_000, n)]
    base = [NonceResult("t", i, True, 100, 1) for i in range(n)] if baseline else None
    return EvalRequest(
        challenge, {"mod.rs": "fn x(){}"}, tr, ho, 7, base, CHALLENGES[challenge].beat)


def test_c3_pulls_the_official_ghcr_dev_image():
    # C3 pulls straight from GHCR; the reference is literal so a namespace or tag drift on
    # either function is caught, and equal to Modal's so the two backends build in one image
    assert c3_image("knapsack") == f"ghcr.io/tig-foundation/tig-monorepo/knapsack/dev:{DEV_IMAGE_TAG}"
    assert c3_image("hypergraph") == dev_image("hypergraph")


def test_profile_workers_and_hardware_class():
    cpu, gpu = CHALLENGES["knapsack"], CHALLENGES["hypergraph"]
    assert c3_profile(cpu) == "cpu-d3-4vcpu-16gb" and c3_profile(gpu, "l40") == "l40"
    # mutation: 4 workers on one GPU would serialise on the device and time out the job
    assert c3_workers(cpu) == 4 and c3_workers(gpu) == 1
    # mutation: reusing the Modal hardware class string lets a Modal baseline serve a C3 run
    assert c3_hardware_class(cpu) == "c3-cpu-d3-4vcpu-16gb"
    assert c3_hardware_class(gpu, "l40") == "c3-l40"


def test_time_limit_formula_and_cap():
    # 1 nonce, 4 workers: 1200 + ceil(600/4) = 1350 -> rounded up to 1380 (23 min)
    assert c3_jobdir.time_limit_s(1, 4) == 1380
    # mutation: forgetting the cap makes a 3-track 64-nonce GPU job ask for 10+ hours
    assert c3_jobdir.time_limit_s(192, 1) == 21600
    assert c3_jobdir.hhmmss(1380) == "00:23:00" and c3_jobdir.hhmmss(21600) == "06:00:00"


def test_job_settings_and_the_c3_file_agree():
    s = c3_jobdir.job_settings("knapsack", "3", 1380)
    text = c3_jobdir.c3_config_text("knapsack", "3", 1380)
    assert set(s) == {"project", "job_name", "script", "hardware", "walltime_seconds",
                      "docker_image", "docker_requires_accelerator"}
    # mutation: a hardware profile hard-coded on either path scores the baseline and the
    # candidate on different hardware (AGENTS.md invariant 1)
    assert f"hardware: {s['hardware']}\n" in text
    assert f"  image: {s['docker_image']}\n" in text
    assert f"  requires_accelerator: {s['docker_requires_accelerator']}\n" in text
    assert f"job_name: {s['job_name']}\n" in text and f"script: {s['script']}\n" in text
    assert f"project: {s['project']}\n" in text
    assert s["walltime_seconds"] == 1380 and 'time: "00:23:00"' in text
    # literal expectations (not derived from job_settings/c3_image), so a hard-coded value on
    # either path that happens to still agree with itself does not slip past this test
    assert s["project"] == "talos" and s["job_name"] == "talos-knapsack-3"
    assert s["script"] == "job.sh"
    assert s["hardware"] == "cpu-d3-4vcpu-16gb"
    assert s["docker_requires_accelerator"] == "none"
    assert s["docker_image"] == f"ghcr.io/tig-foundation/tig-monorepo/knapsack/dev:{DEV_IMAGE_TAG}"
    g = c3_jobdir.job_settings("hypergraph", "1", 60, gpu="l40")
    assert g["docker_requires_accelerator"] == "cuda" and g["hardware"] == "l40"
    # mutation: a hard-coded "l40" runs a job frozen to the a100 class on the wrong hardware
    assert c3_jobdir.job_settings("hypergraph", "1", 60, gpu="a100")["hardware"] == "a100"


def test_c3_config_text_is_exact():
    text = c3_jobdir.c3_config_text("knapsack", "3", 1380)
    assert text == (
        "project: talos\n"
        "job_name: talos-knapsack-3\n"
        "script: job.sh\n"
        "hardware: cpu-d3-4vcpu-16gb\n"
        'time: "00:23:00"\n'
        "docker:\n"
        f"  image: ghcr.io/tig-foundation/tig-monorepo/knapsack/dev:{DEV_IMAGE_TAG}\n"
        "  requires_accelerator: none\n")
    # mutation: a GPU image on a CPU-flagged job is rejected only at run time, after the pull
    assert "requires_accelerator: cuda" in c3_jobdir.c3_config_text("hypergraph", "1", 60, "l40")
    assert "hardware: h100\n" in c3_jobdir.c3_config_text("hypergraph", "1", 60, "h100")


def test_write_job_dir_contents_and_secrecy(tmp_path):
    d = c3_jobdir.write_job_dir(tmp_path / "job", req(), "3")
    names = sorted(p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file())
    assert names == [".c3", "job.sh", "payload.json", "talos/__init__.py", "talos/c3_job.py",
                     "talos/challenges.py", "talos/diagnostics.py", "talos/inside.py",
                     "talos/scoring.py", "talos/types.py"]
    if os.name != "nt":  # Windows has no execute bit to set
        assert stat.S_IMODE((d / "job.sh").stat().st_mode) & stat.S_IXUSR
    # mutation: text-mode writes on Windows end lines in CRLF, and bash in the Linux container
    # then fails on `/bin/bash\r`
    assert b"\r" not in (d / "job.sh").read_bytes() and b"\r" not in (d / ".c3").read_bytes()
    assert MONOREPO_REF in (d / "job.sh").read_text()
    p = json.loads((d / "payload.json").read_text())
    assert p["challenge"] == "knapsack" and p["challenge_id"] == "c003" and p["fuel"] == 7
    assert p["training"][0]["rand_hash"] == HASH and p["rule"]["margin"] == 0.005
    assert p["monorepo_ref"] == MONOREPO_REF and p["workers"] == 4
    # mutation: the rand hash in any file C3 shows in its dashboard (the .c3, the job name,
    # the script) is a leak; only payload.json may carry it
    for name in (".c3", "job.sh"):
        assert HASH not in (d / name).read_text()
    # mutation: a job dir that is not wiped first ships the previous iteration's files
    (d / "stale.txt").write_text("x")
    c3_jobdir.write_job_dir(tmp_path / "job", req(), "3")
    assert not (d / "stale.txt").exists()


def test_request_hash_tracks_files_and_nonces_only():
    a = c3_jobdir.request_hash(req())
    assert a == c3_jobdir.request_hash(req())
    # mutation: hashing only the files lets a resumed job with different nonce sets reattach
    assert a != c3_jobdir.request_hash(req(n=5))
    other = req()
    other.files = {"mod.rs": "fn y(){}"}
    assert a != c3_jobdir.request_hash(other)
    assert len(a) == 16 and HASH not in a


def test_a_compile_only_request_gets_the_one_nonce_floor_and_empty_sets(tmp_path):
    r = EvalRequest("knapsack", {"mod.rs": "fn x(){}"}, [], [], 7, None,
                    CHALLENGES["knapsack"].beat)
    d = c3_jobdir.write_job_dir(tmp_path / "job", r, "compile")
    floor = c3_jobdir.hhmmss(c3_jobdir.time_limit_s(1, c3_workers(CHALLENGES["knapsack"])))
    # mutation: dropping the max(nonces, 1) floor asks C3 for time_limit_s(0, 4) = 00:20:00,
    # the bare build allowance with no slack at all
    assert f'time: "{floor}"' in (d / ".c3").read_text() and floor == "00:23:00"
    p = json.loads((d / "payload.json").read_text())
    assert p["training"] == [] and p["holdout"] == [] and p["baseline_training"] is None


def test_payload_carries_the_prior_function_names(tmp_path):
    # mutation: dropping prior_functions from the payload leaves the job unable to tell a
    # function the candidate added from one the baseline already had
    r = req()
    r.prior_functions = {"mod.rs": ["solve"]}
    assert c3_jobdir.payload(r)["prior_functions"] == {"mod.rs": ["solve"]}
    assert c3_jobdir.payload(req())["prior_functions"] is None


def test_payload_carries_per_track_timeouts(tmp_path):
    # mutation: dropping timeouts from the payload leaves the job on the flat 600 s cap
    r = req()
    r.timeouts = {"t": 42}
    assert c3_jobdir.payload(r)["timeouts"] == {"t": 42}
    assert c3_jobdir.payload(req())["timeouts"] is None


def test_payload_carries_the_hyperparameters(tmp_path):
    # mutation: dropping the map from the payload runs every C3 nonce without hyperparameters
    r = req()
    r.hyperparameters = {"t": {"x": 1}}
    assert c3_jobdir.payload(r)["hyperparameters"] == {"t": {"x": 1}}
    # mutation: always writing the key (even None) changes the hash of a pre-upgrade request
    assert "hyperparameters" not in c3_jobdir.payload(req())


def test_request_hash_is_unchanged_from_before_hyperparameters_existed():
    # mutation: always writing the "hyperparameters" key changes this request's hash, so a
    # resume against a job that was submitted before the upgrade orphans it instead of
    # reattaching, leaving it to keep billing
    assert c3_jobdir.request_hash(req()) == "8360658c78306a4f"


def test_request_hash_changes_with_the_hyperparameters():
    # mutation: a hash that ignores the map lets a resume reattach to a job run without it
    with_hp = req()
    with_hp.hyperparameters = {"t": {"x": 1}}
    other_hp = req()
    other_hp.hyperparameters = {"t": {"x": 2}}
    hashes = {c3_jobdir.request_hash(r) for r in (req(), with_hp, other_hp)}
    assert len(hashes) == 3


def test_local_flavour_writes_local_json_and_a_job_sh_without_a_download(tmp_path):
    d = c3_jobdir.write_job_dir(tmp_path / "job", req(n=4), "3", local=LocalSettings(8, 12))
    names = sorted(p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file())
    # mutation: shipping .c3 on the local path, or forgetting local.json
    assert ".c3" not in names and "local.json" in names and "payload.json" in names
    sh = (d / "job.sh").read_text()
    # mutation: the C3 job.sh downloads the monorepo; locally it is on the /app volume
    assert "curl" not in sh and "codeload" not in sh and "python3 -m talos.c3_job" in sh
    assert b"\r" not in (d / "job.sh").read_bytes()
    if os.name != "nt":
        assert stat.S_IMODE((d / "job.sh").stat().st_mode) & stat.S_IXUSR
    doc = json.loads((d / "local.json").read_text())
    assert doc["challenge"] == "knapsack" and doc["image"] == dev_image("knapsack")
    assert doc["cpus"] == 8 and doc["memory_gib"] == 12 and doc["gpu"] is False
    # mutation: workers left at C3's 4 on a 8-cpu container idles half the machine
    assert doc["workers"] == 8
    assert json.loads((d / "payload.json").read_text())["workers"] == 8
    # 8 nonces, 8 workers: the local build allowance 3600 + ceil(8*600/8) = 4200 s. The C3
    # allowance of 1200 s is too tight locally: MEASURED 2026-09-23, a candidate build took
    # 14m44s on 16 cores, and fewer cores take longer.
    assert doc["time_limit_s"] == 4200
    assert c3_jobdir.LOCAL_BUILD_ALLOWANCE_S == 3600
    assert doc["request_hash"] == c3_jobdir.request_hash(req(n=4))
    # mutation: the rand hash in local.json would land in the container name and `docker ps`
    assert HASH not in (d / "local.json").read_text() and HASH not in sh


def test_local_job_sh_runs_the_runner_under_the_app_volume_lock(tmp_path):
    d = c3_jobdir.write_job_dir(tmp_path / "job", req(n=4), "3", local=LocalSettings(8, 12))
    sh = (d / "job.sh").read_text()
    # mutation: without the lock, two jobs of one challenge on this machine (two runs, or a
    # `talos compile` beside a run) restage talos_cand in the shared /app checkout under each
    # other's build, and a candidate is compiled from the other job's files
    assert "exec flock /app/.talos-lock python3 -m talos.c3_job" in sh
    assert "flock" not in c3_jobdir.job_sh_text(MONOREPO_REF)  # C3: one checkout per job


def test_local_flavour_gpu_challenge_runs_one_worker(tmp_path):
    d = c3_jobdir.write_job_dir(tmp_path / "job", req(challenge="hypergraph"), "1",
                                local=LocalSettings(8, 12))
    doc = json.loads((d / "local.json").read_text())
    assert doc["gpu"] is True and doc["workers"] == 1


def test_request_hash_is_the_same_for_both_flavours():
    # mutation: hashing the workers count makes a local request never match a C3 one, and worse,
    # never match itself after a setup that changed the CPU count
    r = req()
    assert c3_jobdir.request_hash(r) == c3_jobdir.request_hash(r)
    assert c3_jobdir.payload(r, workers=8)["workers"] == 8
    assert c3_jobdir.payload(r)["workers"] == 4


def test_parse_c3_inverts_render_c3():
    from talos.c3_jobdir import parse_c3, render_c3
    s = c3_jobdir.job_settings("hypergraph", "7", 1380, gpu="a100")
    # mutation: a regex that drops the minutes term, or reads hardware from the wrong line,
    # sends the MCP path a different job from the one the CLI path reads out of .c3
    assert parse_c3(render_c3(s)) == s
    assert parse_c3(c3_jobdir.c3_config_text("knapsack", "3", 1380)) == \
        c3_jobdir.job_settings("knapsack", "3", 1380)
    with pytest.raises(ValueError):
        parse_c3("project: talos\n")


def test_probe_dir_is_a_tiny_job_on_the_named_class(tmp_path):
    from talos.c3_jobdir import (PROBE_IMAGE, PROBE_WALLTIME_S, parse_c3, probe_settings,
                                 write_probe_dir)
    d = write_probe_dir(tmp_path / "probe", "h100")
    s = parse_c3((d / ".c3").read_text(encoding="utf-8"))
    assert s == probe_settings("h100")
    # mutation: the dev image (13 GB) makes every probe a minutes-long pull; the challenge's
    # walltime makes a probe that is never cancelled bill for hours
    assert s["hardware"] == "h100" and s["docker_image"] == PROBE_IMAGE
    assert s["walltime_seconds"] == PROBE_WALLTIME_S <= 300
    assert s["docker_requires_accelerator"] == "cuda"
    assert s["job_name"] == "talos-probe-h100" and s["script"] == "job.sh"
    sh = d / "job.sh"
    assert sh.read_text(encoding="utf-8").startswith("#!/bin/bash")
    if os.name != "nt":  # Windows has no execute bit to set
        assert stat.S_IMODE(sh.stat().st_mode) & stat.S_IXUSR
    # nothing of a real job: no payload (no rand_hash) and no talos modules
    assert sorted(p.name for p in d.iterdir()) == [".c3", "job.sh"]
    write_probe_dir(tmp_path / "probe", "l40")  # rewriting is fine
    assert parse_c3((d / ".c3").read_text(encoding="utf-8"))["hardware"] == "l40"


def test_write_job_dir_takes_the_frozen_gpu(tmp_path):
    from talos.c3_jobdir import parse_c3
    d = c3_jobdir.write_job_dir(tmp_path / "job", req(challenge="hypergraph"), "3", gpu="a100")
    # mutation: dropping the argument submits every GPU job on the first class
    assert parse_c3((d / ".c3").read_text(encoding="utf-8"))["hardware"] == "a100"
    with pytest.raises(ValueError):  # a GPU job with no chosen GPU is a programming error
        c3_jobdir.write_job_dir(tmp_path / "job2", req(challenge="hypergraph"), "3")
