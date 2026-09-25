import json
import os
import shutil
import subprocess
import types
from datetime import datetime, timedelta, timezone

import pytest

from talos.c3_bench import C3CommandError
from talos.challenges import DEV_IMAGE_TAG, MONOREPO_REF, dev_image
from talos.local_transport import (APP, DockerTransport, PIDS_LIMIT, READY_MARKER, WARM_MARKER,
                                   clone_script, container_name, docker_info, docker_runtimes,
                                   has_gpu_runtime, parse_time, prepare, run_args, run_key,
                                   volume_key, volume_names, warm_script)

T0 = datetime(2026, 9, 23, 10, 0, 0, tzinfo=timezone.utc)


def local_doc(gpu=False, seconds=1800):
    return {"challenge": "hypergraph" if gpu else "knapsack",
            "image": dev_image("hypergraph" if gpu else "knapsack"), "cpus": 8, "memory_gib": 12,
            "gpu": gpu, "workers": 1 if gpu else 8, "time_limit_s": seconds,
            "request_hash": "ab12" * 4}


def job_dir(tmp_path, gpu=False, seconds=1800):
    d = tmp_path / "runs" / "job1" / "local" / "3"
    d.mkdir(parents=True)
    (d / "local.json").write_text(json.dumps(local_doc(gpu, seconds)), encoding="utf-8")
    return d


class FakeDocker:
    """Scripted `docker` CLI. `inspect` is the State document returned; None = no container."""

    def __init__(self, inspect=None, run_rc=0, volume_labels=None, ps_ids="", files=None,
                 daemon_down=False):
        self.inspect = inspect
        self.run_rc = run_rc
        self.volume_labels = volume_labels or {}
        self.ps_ids = ps_ids
        self.files = files or {}  # name -> text, what the container has under /artifacts
        self.daemon_down = daemon_down
        self.calls = []

    def __call__(self, cmd, **kw):
        self.calls.append(cmd)
        rc, out = 0, ""
        word = cmd[1]
        if word == "run":
            rc, out = self.run_rc, "0123456789abcdef" * 4 + "\n"
        elif word == "inspect":
            if self.inspect is None:
                rc, out = 1, "Error: No such object"
            else:
                out = json.dumps([{"State": self.inspect,
                                   "Config": {"Labels": {"talos.time_limit_s": "1800"},
                                              "Image": "img:1"},
                                   "Mounts": [{"Source": "/host/art",
                                               "Destination": "/artifacts"}]}])
        elif word == "ps":
            out = self.ps_ids
        elif word == "volume" and cmd[2] == "inspect":
            label = cmd[-1]
            fmt = [a for a in cmd if a.startswith("{{")]
            key = fmt[0].split('"')[1] if fmt else ""
            out = self.volume_labels.get((label, key), "")
            if label not in {v for v, _ in self.volume_labels}:
                rc, out = 1, "Error: No such volume"
        elif word == "image" and cmd[2] == "inspect":
            rc = 0 if "present" in cmd[-1] else 1
        elif word in ("rm", "kill"):
            rc = 0 if self.inspect is not None else 1
        elif word == "cp":
            if self.daemon_down:
                rc, out = 1, "Cannot connect to the Docker daemon at unix:///var/run/docker.sock"
            else:
                src, dest = cmd[2], cmd[3]
                name = src.split("/")[-1]
                if name in self.files:
                    from pathlib import Path
                    Path(dest).write_text(self.files[name], encoding="utf-8")
                else:
                    rc, out = 1, (f"Error response from daemon: Could not find the file "
                                  f"/artifacts/{name} in container {src.split(':')[0]}")
        return types.SimpleNamespace(returncode=rc, stdout=out, stderr="" if rc == 0 else out)


def state(status, exit_code=0, started=T0, finished=None):
    zero = "0001-01-01T00:00:00Z"
    return {"Status": status, "Running": status == "running", "ExitCode": exit_code,
            "StartedAt": started.isoformat().replace("+00:00", "Z"),
            "FinishedAt": zero if finished is None else
            finished.isoformat().replace("+00:00", "Z")}


def transport(fake, now=None):
    return DockerTransport(run=fake, now=now or (lambda: T0 + timedelta(seconds=60)))


def test_parse_time_handles_nanoseconds_and_the_zero_sentinel():
    assert parse_time("2026-09-23T10:00:00.123456789Z") == T0 + timedelta(microseconds=123456)
    assert parse_time("2026-09-23T10:00:00Z") == T0
    assert parse_time("0001-01-01T00:00:00Z").year == 1


def test_volume_and_container_names_are_keyed_on_the_pins_and_the_run(tmp_path):
    key = volume_key()
    assert len(key) == 12 and volume_names("knapsack") == (f"talos-app-knapsack-{key}",
                                                           f"talos-cargo-knapsack-{key}")
    # mutation: a key that ignores either pin reuses a target dir built against another monorepo
    assert MONOREPO_REF[:12] != key and DEV_IMAGE_TAG not in key
    a, b = tmp_path / "runs" / "a", tmp_path / "runs" / "b"
    assert container_name(a, "3", "ab12" * 4) != container_name(b, "3", "ab12" * 4)
    assert container_name(a, "3", "ab12" * 4) == f"talos-{run_key(a)}-3-{'ab12' * 4}"


def test_run_args_carry_every_hardening_flag_and_the_mounts(tmp_path):
    jd = tmp_path / "job dir"
    args = run_args(local_doc(), jd, "n", "runkey12", "/usr/local/cargo", "/usr/local/rustup")
    assert args[:4] == ["docker", "run", "-d", "--name"] and args[4] == "n"
    joined = " ".join(args)
    for flag in ("--network none", "--cap-drop ALL", "--security-opt no-new-privileges",
                 f"--pids-limit {PIDS_LIMIT}", "--cpus 8", "--memory 12g",
                 "-e C3_JOB_WORKDIR=/work", "-e C3_ARTIFACTS_DIR=/artifacts",
                 "-e CARGO_HOME=/usr/local/cargo", "-e RUSTUP_HOME=/usr/local/rustup",
                 "--label talos.time_limit_s=1800", "--label talos.run=runkey12"):
        # mutation: any one hardening flag dropped
        assert flag in joined, flag
    app, cargo = volume_names("knapsack")
    assert f"{app}:/app" in args and f"{cargo}:/usr/local/cargo" in args
    assert f"{jd}:/work:ro" in args
    # mutation: bind-mounting /artifacts from the host (see test_deploy_runs_the_container...)
    assert not any(a.endswith(":/artifacts") for a in args)
    assert args[-3:] == [dev_image("knapsack"), "bash", "/work/job.sh"]
    assert "--gpus" not in args
    gpu = run_args(local_doc(gpu=True), jd, "n", "k", "/c", "/r")
    assert "--gpus" in gpu and gpu[gpu.index("--gpus") + 1] == "all"
    # The image keeps cargo and rustup under /root (mode 700), so the job runs as root; the
    # artifacts are copied out of the stopped container, never written to the host by it.
    assert "--user" not in args and "HOME=/tmp" not in joined


def test_run_args_keeps_a_path_with_a_space_as_one_argument(tmp_path):
    jd = tmp_path / "my runs" / "j"
    args = run_args(local_doc(), jd, "n", "k", "/c", "/r")
    assert args[args.index("-v") + 1] == f"{jd}:/work:ro"


def test_deploy_runs_the_container_named_from_the_job_dir_and_returns_the_name(tmp_path):
    fake = FakeDocker(inspect=None, volume_labels={
        (volume_names("knapsack")[1], "talos.cargo_home"): "/usr/local/cargo\n",
        (volume_names("knapsack")[1], "talos.rustup_home"): "/usr/local/rustup\n"})
    jd = job_dir(tmp_path)
    name = transport(fake).deploy(jd)
    run_dir = tmp_path / "runs" / "job1"
    assert name == container_name(run_dir, "3", "ab12" * 4)
    run_cmd = [c for c in fake.calls if c[1] == "run"][0]
    assert run_cmd[run_cmd.index("--name") + 1] == name
    # mutation: a host bind mount for /artifacts lets root inside the container plant a
    # setuid-root file on the host (review, 2026-09-23); the files are copied out instead
    assert not any(v.endswith(":/artifacts") for v in run_cmd)
    assert not (jd / name).exists()
    assert "-e" in run_cmd and "CARGO_HOME=/usr/local/cargo" in run_cmd
    # mutation: a stale container with this name makes `docker run` fail with a name clash
    assert ["rm", "-f", name] in [c[1:] for c in fake.calls]


def test_deploy_prunes_exited_containers_of_the_same_run_first(tmp_path):
    fake = FakeDocker(inspect=None, ps_ids="aaa\nbbb\n", volume_labels={
        (volume_names("knapsack")[1], "talos.cargo_home"): "/c",
        (volume_names("knapsack")[1], "talos.rustup_home"): "/r"})
    jd = job_dir(tmp_path)
    transport(fake).deploy(jd)
    ps = [c for c in fake.calls if c[1] == "ps"][0]
    key = run_key(tmp_path / "runs" / "job1")
    # mutation: pruning every exited container on the machine, or none
    assert f"label={'talos.run'}={key}" in ps and "status=exited" in ps
    assert ["rm", "aaa", "bbb"] in [c[1:] for c in fake.calls]


def test_deploy_without_the_cargo_volume_is_a_command_error_not_a_crash(tmp_path):
    fake = FakeDocker(inspect=None)
    with pytest.raises(C3CommandError):
        transport(fake).deploy(job_dir(tmp_path))


@pytest.mark.parametrize("st, expect", [
    (state("created"), "PENDING"),
    (state("running"), "RUNNING"),
    (state("exited", 0, finished=T0 + timedelta(seconds=30)), "SUCCEEDED"),
    (state("exited", 1, finished=T0 + timedelta(seconds=30)), "FAILED"),
    (state("exited", 137, finished=T0 + timedelta(seconds=1800)), "TIMED_OUT"),
    (state("exited", 0, finished=T0 + timedelta(seconds=1801)), "TIMED_OUT"),
])
def test_status_maps_docker_state_onto_the_c3_vocabulary(st, expect):
    fake = FakeDocker(inspect=st)
    assert transport(fake).status("n") == expect
    # mutation: killing a container that has already exited
    assert not any(c[1] == "kill" for c in fake.calls)


def test_status_kills_a_running_container_past_its_limit_and_reports_timed_out():
    fake = FakeDocker(inspect=state("running"))
    t = transport(fake, now=lambda: T0 + timedelta(seconds=1800))
    assert t.status("n") == "TIMED_OUT"
    assert ["kill", "n"] in [c[1:] for c in fake.calls]
    # mutation: a limit read from a hard-coded constant instead of the container's label
    fake2 = FakeDocker(inspect=state("running"))
    assert transport(fake2, now=lambda: T0 + timedelta(seconds=1799)).status("n") == "RUNNING"


def test_status_on_a_missing_container_is_a_command_error():
    with pytest.raises(C3CommandError):
        transport(FakeDocker(inspect=None)).status("n")


def test_cancel_removes_the_container_and_tolerates_a_missing_one():
    fake = FakeDocker(inspect=state("running"))
    transport(fake).cancel("n")
    assert ["rm", "-f", "n"] in [c[1:] for c in fake.calls]
    transport(FakeDocker(inspect=None)).cancel("n")  # no raise


def test_fetch_copies_the_named_file_out_of_the_container_and_reports_absence(tmp_path):
    fake = FakeDocker(files={"results.json": "{}"})
    dest = tmp_path / "n" / "artifacts" / "results.json"
    assert transport(fake).fetch("n", "results.json", dest) is True
    assert dest.read_text() == "{}"
    assert ["cp", "n:/artifacts/results.json", str(dest)] in [c[1:] for c in fake.calls]
    # mutation: returning True for a file the job never wrote makes _collect read a stale path
    assert transport(fake).fetch("n", "build.log", tmp_path / "n" / "artifacts" / "build.log") is False


def test_fetch_tells_a_missing_file_from_a_dead_daemon(tmp_path):
    # mutation: every cp failure reported as "no results" resubmits the job (twice, then
    # BenchUnavailable "failed twice") when Docker itself is what is broken
    with pytest.raises(C3CommandError):
        transport(FakeDocker(daemon_down=True)).fetch("n", "results.json", tmp_path / "r.json")


def test_whoami_and_balance_are_not_part_of_the_local_path():
    t = transport(FakeDocker())
    with pytest.raises(NotImplementedError):
        t.whoami()
    with pytest.raises(NotImplementedError):
        t.balance_gbp()


def test_a_missing_docker_binary_or_a_hang_is_a_command_error():
    def missing(cmd, **kw):
        raise FileNotFoundError("docker")
    with pytest.raises(C3CommandError):
        DockerTransport(run=missing).docker("info")
    import subprocess

    def hung(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, kw.get("timeout"))
    with pytest.raises(C3CommandError):
        DockerTransport(run=hung).docker("info")


class FakePrepareDocker:
    """`docker` for prepare: which image and volumes exist, which markers the /app volume has."""

    def __init__(self, image=False, volumes=(), markers=(), runtimes=("runc",), gpu_name="L40S",
                 ncpu=16, mem_bytes=30 * 2 ** 30 + 12345):
        self.image, self.volumes, self.markers = image, set(volumes), set(markers)
        self.runtimes, self.gpu_name = runtimes, gpu_name
        self.ncpu, self.mem_bytes = ncpu, mem_bytes
        self.calls = []

    def __call__(self, cmd, **kw):
        self.calls.append(cmd)
        rc, out = 0, ""
        if cmd[1:3] == ["image", "inspect"]:
            rc = 0 if self.image else 1
        elif cmd[1] == "pull":
            self.image = True
        elif cmd[1:3] == ["volume", "inspect"]:
            vol = cmd[-1]
            if vol not in self.volumes:
                rc = 1
            elif "--format" in cmd:
                out = "/usr/local/cargo\n" if "cargo_home" in cmd[-2] else "/usr/local/rustup\n"
        elif cmd[1:3] == ["volume", "create"]:
            self.volumes.add(cmd[-1])
        elif cmd[1] == "info":
            # `docker info --format {{json .}}`: the whole document, of which three keys matter
            out = json.dumps({"Runtimes": {r: {"path": r} for r in self.runtimes},
                              "NCPU": self.ncpu, "MemTotal": self.mem_bytes, "Images": 3})
        elif cmd[1] == "run":
            script = cmd[-1]
            if cmd[-2] == "-c" and script.startswith("test -e"):
                rc = 0 if script.split("/")[-1] in self.markers else 1
            elif "nvidia-smi" in cmd:
                out = f"{self.gpu_name}\n"
            elif "echo ${CARGO_HOME" in script:
                out = "/usr/local/cargo\n"
            elif "echo ${RUSTUP_HOME" in script:
                out = "/usr/local/rustup\n"
            elif "codeload" in script:
                self.markers.add(READY_MARKER)
            elif "build_algorithm" in script:
                self.markers.add(WARM_MARKER)
        return types.SimpleNamespace(returncode=rc, stdout=out, stderr="" if rc == 0 else out)


def runs(fake):
    return [c for c in fake.calls if c[1] == "run"]


def test_prepare_from_nothing_pulls_creates_volumes_clones_and_warms(tmp_path):
    fake = FakePrepareDocker()
    lines = []
    assert prepare("knapsack", run=fake, log=lines.append) is None
    app, cargo = volume_names("knapsack")
    assert ["pull", dev_image("knapsack")] in [c[1:] for c in fake.calls]
    create = [c for c in fake.calls if c[1:3] == ["volume", "create"]]
    # mutation: the cargo volume created without its labels leaves deploy unable to mount it
    assert any(c[-1] == cargo and "--label" in c
               and any(a.startswith("talos.cargo_home=/usr/local/cargo") for a in c)
               and any(a.startswith("talos.rustup_home=/usr/local/rustup") for a in c)
               for c in create)
    assert any(c[-1] == app for c in create)
    scripts = [c[-1] for c in runs(fake) if c[-2] == "-c"]
    clone = [s for s in scripts if "codeload" in s][0]
    warm = [s for s in scripts if "build_algorithm" in s][0]
    assert MONOREPO_REF in clone and "--strip-components=1" in clone
    # mutation: a warm build that builds nothing leaves the registry cold and the first job
    # fails with --network none
    assert "tig-algorithms/src/knapsack" in warm and "talos_cand" in warm
    assert f"touch /app/{READY_MARKER}" in clone and f"touch /app/{WARM_MARKER}" in warm
    # the job runs as root (the image keeps cargo under /root, mode 700): nothing to chown
    assert "chown" not in clone and "chown" not in warm
    # mutation: prepare's containers with --network none cannot download anything
    for c in runs(fake):
        assert "--network" not in c
    assert "--gpus" not in " ".join(" ".join(c) for c in fake.calls)
    assert any("pulling" in ln for ln in lines) and any("warm" in ln.lower() for ln in lines)


@pytest.mark.parametrize("have, absent_step", [
    ("image", "pull"), ("volumes", "create"), ("ready", "codeload"), ("warm", "build_algorithm")])
def test_prepare_skips_each_step_whose_marker_is_present(tmp_path, have, absent_step):
    app, cargo = volume_names("knapsack")
    fake = FakePrepareDocker(
        image=have in ("image", "volumes", "ready", "warm"),
        volumes=(app, cargo) if have in ("volumes", "ready", "warm") else (),
        markers={"ready": (READY_MARKER,), "warm": (READY_MARKER, WARM_MARKER)}.get(have, ()))
    prepare("knapsack", run=fake, log=lambda *a: None)
    # mutation: a step that runs unconditionally re-clones (or re-pulls 13 GB) on every run
    assert not any(absent_step in " ".join(c) for c in fake.calls), absent_step


def test_prepare_for_a_gpu_challenge_warms_with_the_gpu_and_returns_its_name(tmp_path):
    fake = FakePrepareDocker(runtimes=("runc", "nvidia"), gpu_name="NVIDIA L40S")
    assert prepare("hypergraph", run=fake, log=lambda *a: None) == "NVIDIA L40S"
    warm = [c for c in runs(fake) if c[-2] == "-c" and "build_algorithm" in c[-1]][0]
    assert "--gpus" in warm
    smi = [c for c in runs(fake) if "nvidia-smi" in c][0]
    assert "--gpus" in smi and "--query-gpu=name" in smi


def test_docker_runtimes_and_the_gpu_check():
    assert docker_runtimes(FakePrepareDocker(runtimes=("runc", "nvidia"))) == ["nvidia", "runc"]
    assert has_gpu_runtime(FakePrepareDocker(runtimes=("runc", "nvidia"))) is True
    assert has_gpu_runtime(FakePrepareDocker(runtimes=("runc",))) is False

    def down(cmd, **kw):
        return types.SimpleNamespace(returncode=1, stdout="",
                                     stderr="Cannot connect to the Docker daemon")
    # mutation: a daemon that is down reported as "no GPU" instead of "no Docker"
    with pytest.raises(C3CommandError):
        docker_runtimes(down)


def test_docker_info_reports_the_daemons_cpus_and_memory_in_gib():
    info = docker_info(FakePrepareDocker(runtimes=("runc", "nvidia"), ncpu=6,
                                         mem_bytes=8 * 2 ** 30 + 12345))
    # mutation: MemTotal left in bytes, or rounded up, over-reports what the container may take
    assert (info.ncpu, info.mem_total_gib, info.runtimes) == (6, 8, ["nvidia", "runc"])
    # a daemon that reports nothing for either is "unknown", never 0 CPUs or 0 GiB
    unknown = docker_info(FakePrepareDocker(ncpu=0, mem_bytes=0))
    assert unknown.ncpu is None and unknown.mem_total_gib is None

    def no_doc(cmd, **kw):
        return types.SimpleNamespace(returncode=0, stdout="not json", stderr="")
    with pytest.raises(C3CommandError):
        docker_info(no_doc)


def test_prepare_scripts_take_the_app_volume_lock():
    # mutation: no lock lets a second run of the same challenge re-clone or rebuild in /app
    # while this one's build is reading it (the volume is shared by every job of the challenge)
    for script in (clone_script(), warm_script("knapsack")):
        assert "flock" in script and f"{APP}/.talos-lock" in script, script
        assert script.index("flock") < script.index("touch")



def test_warm_script_prunes_the_crate_before_building():
    # mutation: a warm-up that builds the shipped algorithm in the full crate hits the same
    # 2 h, 23 GB wall on job_scheduling as an unpruned job (MEASURED 2026-09-25), before the
    # first job can run
    from talos.inside import PRISTINE_MOD_RS, PRUNED_MARKER
    from talos.local_transport import prune_lines
    script = warm_script("job_scheduling")
    assert prune_lines("job_scheduling") in script
    assert script.index(PRUNED_MARKER) < script.index("build_algorithm")
    assert "tig-algorithms/src/job_scheduling/" + PRISTINE_MOD_RS in script


@pytest.mark.skipif(shutil.which("bash") is None, reason="the prune lines are bash")
def test_prune_lines_build_only_the_named_module_then_restore_the_pinned_mod_rs(tmp_path):
    # Runs the bash for real, with build_algorithm stubbed to record what it was handed.
    # mutation: a trap that does not put the pinned file back leaves the crate pruned for the
    # jobs, whose stage then saves the pruned file as pristine
    from talos.inside import PRISTINE_MOD_RS
    from talos.local_transport import prune_lines
    pinned = "// c003_a001\npub mod knapsplatt;\npub use knapsplatt as c003_a001;\npub mod other;\n"
    crate = tmp_path / "tig-algorithms" / "src" / "knapsack"
    crate.mkdir(parents=True)
    (crate / "mod.rs").write_text(pinned)
    stub = tmp_path / "bin"
    stub.mkdir()
    seen = tmp_path / "seen.rs"
    (stub / "build_algorithm").write_text(f"#!/bin/bash\ncp tig-algorithms/src/knapsack/mod.rs {seen}\n")
    (stub / "build_algorithm").chmod(0o755)
    script = ("set -euo pipefail\nname=knapsplatt\n" + prune_lines("knapsack")
              + 'build_algorithm "$name"\n')
    env = dict(os.environ, PATH=f"{stub}{os.pathsep}{os.environ['PATH']}")
    r = subprocess.run(["bash", "-c", script], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    items = [ln for ln in seen.read_text().splitlines() if ln and not ln.startswith("//")]
    assert items == ["pub mod knapsplatt;"]
    assert (crate / "mod.rs").read_text() == pinned
    assert not (crate / PRISTINE_MOD_RS).exists()


@pytest.mark.skipif(shutil.which("bash") is None, reason="the prune lines are bash")
def test_prune_lines_keep_an_existing_pristine_copy(tmp_path):
    # A job that died before unstaging leaves the pruned mod.rs and the pristine copy behind.
    # mutation: an unconditional cp captures the pruned file, and the restore installs it
    from talos.inside import PRISTINE_MOD_RS, PRUNED_MARKER
    from talos.local_transport import prune_lines
    pinned = "// c003_a001\npub mod knapsplatt;\n"
    crate = tmp_path / "tig-algorithms" / "src" / "knapsack"
    crate.mkdir(parents=True)
    (crate / "mod.rs").write_text(f"{PRUNED_MARKER}\npub mod talos_cand;\n")
    (crate / PRISTINE_MOD_RS).write_text(pinned)
    script = "set -euo pipefail\nname=knapsplatt\n" + prune_lines("knapsack") + "true\n"
    r = subprocess.run(["bash", "-c", script], cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (crate / "mod.rs").read_text() == pinned
