import json
import os
import types
from datetime import datetime, timedelta, timezone

import pytest

from talos.c3_bench import C3CommandError
from talos.challenges import DEV_IMAGE_TAG, MONOREPO_REF, dev_image
from talos.local_transport import (DockerTransport, PIDS_LIMIT, container_name, host_uid,
                                   parse_time, run_args, run_key, volume_key, volume_names)

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

    def __init__(self, inspect=None, run_rc=0, volume_labels=None, ps_ids=""):
        self.inspect = inspect
        self.run_rc = run_rc
        self.volume_labels = volume_labels or {}
        self.ps_ids = ps_ids
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
                                   "Config": {"Labels": {"talos.time_limit_s": "1800"}}}])
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
        return types.SimpleNamespace(returncode=rc, stdout=out, stderr="" if rc == 0 else out)


def state(status, exit_code=0, started=T0, finished=None):
    zero = "0001-01-01T00:00:00Z"
    return {"Status": status, "Running": status == "running", "ExitCode": exit_code,
            "StartedAt": started.isoformat().replace("+00:00", "Z"),
            "FinishedAt": zero if finished is None else
            finished.isoformat().replace("+00:00", "Z")}


def transport(fake, now=None, uid="1000:1000"):
    return DockerTransport(run=fake, now=now or (lambda: T0 + timedelta(seconds=60)), uid=uid)


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
    jd, art = tmp_path / "job dir", tmp_path / "job dir" / "n" / "artifacts"
    args = run_args(local_doc(), jd, art, "n", "runkey12", "/usr/local/cargo",
                    "/usr/local/rustup", "1000:1000")
    assert args[:4] == ["docker", "run", "-d", "--name"] and args[4] == "n"
    joined = " ".join(args)
    for flag in ("--network none", "--cap-drop ALL", "--security-opt no-new-privileges",
                 f"--pids-limit {PIDS_LIMIT}", "--cpus 8", "--memory 12g", "--user 1000:1000",
                 "-e C3_JOB_WORKDIR=/work", "-e C3_ARTIFACTS_DIR=/artifacts", "-e HOME=/tmp",
                 "-e CARGO_HOME=/usr/local/cargo", "-e RUSTUP_HOME=/usr/local/rustup",
                 "--label talos.time_limit_s=1800", "--label talos.run=runkey12"):
        # mutation: any one hardening flag dropped
        assert flag in joined, flag
    app, cargo = volume_names("knapsack")
    assert f"{app}:/app" in args and f"{cargo}:/usr/local/cargo" in args
    assert f"{jd}:/work:ro" in args and f"{art}:/artifacts" in args
    assert args[-3:] == [dev_image("knapsack"), "bash", "/work/job.sh"]
    assert "--gpus" not in args
    gpu = run_args(local_doc(gpu=True), jd, art, "n", "k", "/c", "/r", None)
    assert "--gpus" in gpu and gpu[gpu.index("--gpus") + 1] == "all"
    # mutation: `--user None` on Windows
    assert "--user" not in gpu and "None" not in gpu


def test_run_args_keeps_a_path_with_a_space_as_one_argument(tmp_path):
    jd = tmp_path / "my runs" / "j"
    args = run_args(local_doc(), jd, jd / "n" / "artifacts", "n", "k", "/c", "/r", None)
    assert args[args.index("-v") + 1] == f"{jd}:/work:ro"


def test_host_uid_is_uid_gid_on_posix_and_none_on_windows():
    if os.name == "nt":
        assert host_uid() is None
    else:
        assert host_uid() == f"{os.getuid()}:{os.getgid()}"


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
    # mutation: the artifacts dir left for Docker to create is root-owned on Linux
    assert (jd / name / "artifacts").is_dir()
    assert f"{jd / name / 'artifacts'}:/artifacts" in run_cmd
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


def test_fetch_reports_the_file_on_the_bind_mount_and_runs_nothing(tmp_path):
    fake = FakeDocker()
    dest = tmp_path / "n" / "artifacts" / "results.json"
    assert transport(fake).fetch("n", "results.json", dest) is False
    dest.parent.mkdir(parents=True)
    dest.write_text("{}")
    assert transport(fake).fetch("n", "results.json", dest) is True
    assert fake.calls == []


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
