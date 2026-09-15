import json
import stat

from talos import c3_jobdir
from talos.bench import EvalRequest
from talos.challenges import (CHALLENGES, DEV_IMAGE_TAG, MONOREPO_REF, c3_hardware_class, c3_image,
                              c3_profile, c3_workers, image_namespace)
from talos.types import NonceResult, NonceSet

HASH = "ab" * 32


def req(challenge="knapsack", n=4, baseline=True):
    tr = [NonceSet("t", HASH, 0, n)]
    ho = [NonceSet("t", HASH, 1_000_000, n)]
    base = [NonceResult("t", i, True, 100, 1) for i in range(n)] if baseline else None
    return EvalRequest(
        challenge, {"mod.rs": "fn x(){}"}, tr, ho, 7, base, CHALLENGES[challenge].beat)


def test_image_name_and_namespace_override(monkeypatch):
    assert c3_image("knapsack") == f"docker.io/fibonadithya/tig-knapsack-dev:{DEV_IMAGE_TAG}"
    monkeypatch.setenv("TALOS_IMAGE_NAMESPACE", "other")
    # mutation: reading the namespace at import time makes the maintainer override dead
    assert image_namespace() == "other" and c3_image("knapsack").startswith("docker.io/other/")


def test_profile_workers_and_hardware_class():
    cpu, gpu = CHALLENGES["knapsack"], CHALLENGES["hypergraph"]
    assert c3_profile(cpu) == "cpu-d3-4vcpu-16gb" and c3_profile(gpu) == "l40"
    # mutation: 4 workers on one GPU would serialise on the device and time out the job
    assert c3_workers(cpu) == 4 and c3_workers(gpu) == 1
    # mutation: reusing the Modal hardware class string lets a Modal baseline serve a C3 run
    assert c3_hardware_class(cpu) == "c3-cpu-d3-4vcpu-16gb" and c3_hardware_class(gpu) == "c3-l40"


def test_time_limit_formula_and_cap():
    # 1 nonce, 4 workers: 1200 + ceil(600/4) = 1350 -> rounded up to 1380 (23 min)
    assert c3_jobdir.time_limit_s(1, 4) == 1380
    # mutation: forgetting the cap makes a 3-track 64-nonce GPU job ask for 10+ hours
    assert c3_jobdir.time_limit_s(192, 1) == 21600
    assert c3_jobdir.hhmmss(1380) == "00:23:00" and c3_jobdir.hhmmss(21600) == "06:00:00"


def test_c3_config_text_is_exact():
    text = c3_jobdir.c3_config_text("knapsack", "3", 1380)
    assert text == (
        "project: talos\n"
        "job_name: talos-knapsack-3\n"
        "script: job.sh\n"
        "hardware: cpu-d3-4vcpu-16gb\n"
        'time: "00:23:00"\n'
        "docker:\n"
        f"  image: docker.io/fibonadithya/tig-knapsack-dev:{DEV_IMAGE_TAG}\n"
        "  requires_accelerator: none\n")
    # mutation: a GPU image on a CPU-flagged job is rejected only at run time, after the pull
    assert "requires_accelerator: cuda" in c3_jobdir.c3_config_text("hypergraph", "1", 60)
    assert "hardware: l40\n" in c3_jobdir.c3_config_text("hypergraph", "1", 60)


def test_write_job_dir_contents_and_secrecy(tmp_path):
    d = c3_jobdir.write_job_dir(tmp_path / "job", req(), "3")
    names = sorted(p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file())
    assert names == [".c3", "job.sh", "payload.json", "talos/__init__.py", "talos/c3_job.py",
                     "talos/challenges.py", "talos/inside.py", "talos/scoring.py", "talos/types.py"]
    assert stat.S_IMODE((d / "job.sh").stat().st_mode) & stat.S_IXUSR
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
