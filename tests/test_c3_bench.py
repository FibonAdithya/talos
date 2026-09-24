import json
from pathlib import Path

import pytest

from talos.bench import BenchCancelled, BenchUnavailable, EvalRequest, PendingJobStore
from talos.c3_bench import C3Bench, C3CommandError, fill_timeouts, parse_json_stdout
from talos.c3_jobdir import LocalSettings
from talos.challenges import CHALLENGES
from talos.types import NonceResult, NonceSet

HASH = "ab" * 32


def req(n=2, baseline=True):
    base = [NonceResult("t", i, True, 100, 1) for i in range(n)] if baseline else None
    return EvalRequest("knapsack", {"mod.rs": "fn x(){}"}, [NonceSet("t", HASH, 0, n)],
                       [NonceSet("t", HASH, 1_000_000, n)], 7, base, CHALLENGES["knapsack"].beat)


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def results_doc(n=2, holdout=True, compile_ok=True):
    tr = [{"track": "t", "nonce": i, "ok": True, "quality": 120, "runtime_ms": 5, "error": None}
          for i in range(n)]
    ho = [{"track": "t", "nonce": 1_000_000 + i, "ok": True, "quality": 120, "runtime_ms": 5,
           "error": None} for i in range(n)]
    if not compile_ok:
        return {"compile": {"ok": False, "artifact_id": None, "output": "error[E0308]"},
                "training": [], "holdout": None, "holdout_reason": "not_compiled",
                "started": {"training": False, "holdout": False}}
    return {"compile": {"ok": True, "artifact_id": "a" * 32, "output": "ok"}, "training": tr,
            "holdout": ho if holdout else None, "holdout_reason": "won" if holdout else "not_won",
            "started": {"training": True, "holdout": holdout}}


class FakeC3:
    """Scripted `c3` CLI. `statuses` is the sequence squeue reports; `results` is what pull
    writes (None = no results.json); `deploy_rc` non-zero fails the deploy."""

    def __init__(self, statuses, results=None, deploy_rc=0, job_id="job_1", squeue_fail=0,
                 pull_fail=0):
        self.statuses = list(statuses)
        self.results = results
        self.deploy_rc = deploy_rc
        self.job_id = job_id
        self.squeue_fail = squeue_fail
        self.pull_fail = pull_fail
        self.calls = []
        self.envs = []

    def __call__(self, cmd, **kw):
        self.calls.append((cmd, kw.get("cwd")))
        self.envs.append(kw.get("env"))
        rc, out = 0, ""
        if cmd[1] == "deploy":
            rc = self.deploy_rc
            out = ("Warning: CPU availability is experimental.\n"
                   + json.dumps({"id": self.job_id, "status": "PENDING"}))
        elif cmd[1] == "squeue":
            if self.squeue_fail:
                self.squeue_fail -= 1
                rc, out = 1, "network error"
            else:
                st = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
                out = json.dumps([{"job_id": self.job_id, "status": st}])
        elif cmd[1] == "pull":
            if self.pull_fail:
                self.pull_fail -= 1
                rc, out = 1, "pull error"
            else:
                d = Path(kw["cwd"]) / self.job_id / "artifacts"
                d.mkdir(parents=True, exist_ok=True)
                (d / "build.log").write_text("built")
                if self.results is not None:
                    text = (self.results if isinstance(self.results, str)
                           else json.dumps(self.results))
                    (d / "results.json").write_text(text)
                out = json.dumps({"jobs": [{"job_id": self.job_id, "directory": str(d.parent),
                                            "files": [], "downloaded_count": 1}]})
        elif cmd[1] == "cancel":
            out = "cancelled"
        import types
        return types.SimpleNamespace(returncode=rc, stdout=out, stderr="" if rc == 0 else out)


def bench(tmp_path, c3, pending=None, clock=None):
    clock = clock or Clock()

    def sleep(s):
        clock.t += s
    return C3Bench(tmp_path, pending=pending or PendingJobStore.memory(), run=c3, clock=clock,
                   sleep=sleep, poll_s=20.0, pending_timeout_s=1800,
                   hardware="cpu-d3-4vcpu-16gb"), clock


def test_parse_json_stdout_skips_the_warning_line():
    # mutation: json.loads on the raw stdout fails on the "Warning:" line C3 prints first
    assert parse_json_stdout("Warning: x\n{\"id\": \"j\"}\n") == {"id": "j"}
    assert parse_json_stdout("[{\"a\": 1}]") == [{"a": 1}]
    with pytest.raises(ValueError):
        parse_json_stdout("no json here")


def test_successful_job_maps_to_eval_result_and_charges_running_time_only(tmp_path):
    c3 = FakeC3(["PENDING", "PENDING", "RUNNING", "RUNNING", "SUCCEEDED"], results_doc())
    b, clock = bench(tmp_path, c3)
    r = b.evaluate(req())
    assert r.compile.ok and [x.quality for x in r.training] == [120, 120]
    assert r.holdout_reason == "won" and len(r.holdout) == 2
    # deploy was run from the job dir and the .c3 was generated there
    deploy_cwd = [cwd for (cmd, cwd) in c3.calls if cmd[1] == "deploy"][0]
    assert (Path(deploy_cwd) / ".c3").exists() and Path(deploy_cwd).name == "adhoc"
    # RUNNING was first seen at t=40 (two PENDING polls), terminal at t=80: 40 s billed
    # mutation: charging from submission counts the queue, which C3 does not bill
    expected = 40 / 3600 * 0.11 * 1.35
    assert abs(b.cost_mark() - expected) < 1e-9


def test_compile_failure_comes_back_as_a_result(tmp_path):
    c3 = FakeC3(["RUNNING", "SUCCEEDED"], results_doc(compile_ok=False))
    b, _ = bench(tmp_path, c3)
    r = b.evaluate(req())
    # mutation: raising on compile.ok False burns a resubmission on a deterministic error
    assert not r.compile.ok and "E0308" in r.compile.output and r.training == []
    assert r.holdout_reason == "not_compiled"


def test_timed_out_job_fills_missing_nonces_as_timeouts(tmp_path):
    partial = results_doc(n=2, holdout=False)
    partial["training"] = partial["training"][:1]  # one nonce scored before the limit
    c3 = FakeC3(["RUNNING", "TIMED_OUT"], partial)
    b, _ = bench(tmp_path, c3)
    r = b.evaluate(req(n=2))
    assert [x.nonce for x in r.training] == [0, 1]
    # mutation: dropping the missing nonce makes bundle_delta raise "nonce mismatch" every time
    assert r.training[1].error == "timeout" and not r.training[1].ok
    assert r.holdout is None and r.holdout_reason == "not_won"


def test_timed_out_during_holdout_marks_holdout_reason_timeout(tmp_path):
    doc = results_doc(n=2, holdout=True)
    doc["holdout"] = []  # started, nothing finished
    c3 = FakeC3(["RUNNING", "TIMED_OUT"], doc)
    b, _ = bench(tmp_path, c3)
    r = b.evaluate(req(n=2))
    assert r.holdout is not None and all(x.error == "timeout" for x in r.holdout)
    # the reason stays what the job wrote ("won"); turning an all-timeout held-out set into
    # a non-win is the loop's confirmation step, not this client's job
    assert r.holdout_reason == "won"
    doc2 = results_doc(n=2, holdout=False)
    doc2["holdout_reason"] = "won"  # decided, never started
    doc2["started"]["holdout"] = False
    r2 = bench(tmp_path / "b", FakeC3(["RUNNING", "TIMED_OUT"], doc2))[0].evaluate(req(n=2))
    # mutation: trusting a "won" reason the job never got to act on reports a win that was
    # never actually scored
    assert r2.holdout is None and r2.holdout_reason == "timeout"


def test_failed_job_without_results_is_resubmitted_once_then_unavailable(tmp_path):
    c3 = FakeC3(["RUNNING", "FAILED"], results=None)
    b, _ = bench(tmp_path, c3)
    with pytest.raises(BenchUnavailable):
        b.evaluate(req())
    deploys = [c for (c, _) in c3.calls if c[1] == "deploy"]
    # mutation: retrying for ever on a deterministic infrastructure failure never pauses the run
    assert len(deploys) == 2


def test_timed_out_without_results_is_not_resubmitted(tmp_path):
    c3 = FakeC3(["RUNNING", "TIMED_OUT"], results=None)
    b, _ = bench(tmp_path, c3)
    with pytest.raises(BenchUnavailable) as ei:
        b.evaluate(req())
    assert "timed out" in str(ei.value)
    deploys = [c for (c, _) in c3.calls if c[1] == "deploy"]
    # mutation: resubmitting on a job that burned its whole time budget doubles the most
    # expensive failure mode instead of surfacing it
    assert len(deploys) == 1


def test_unknown_status_is_a_poll_failure_bounded_by_the_limit(tmp_path):
    c3 = FakeC3(["SUSPENDED"], results_doc())
    b, _ = bench(tmp_path, c3)
    with pytest.raises(BenchUnavailable) as ei:
        b.evaluate(req())
    # mutation: leaving ACTIVE unused and falling through on any unrecognised status polls
    # for ever instead of bounding it like any other CLI failure
    assert "SUSPENDED" in str(ei.value)


def test_pull_failure_is_retried_once_against_the_same_job(tmp_path):
    c3 = FakeC3(["RUNNING", "SUCCEEDED"], results_doc(), pull_fail=1)
    b, _ = bench(tmp_path, c3)
    r = b.evaluate(req())
    assert r.compile.ok
    pulls = [c for (c, _) in c3.calls if c[1] == "pull"]
    # mutation: leaving the pull call outside a try means a transient pull failure raises
    # straight through evaluate instead of being retried against the same job id
    assert len(pulls) == 2 and all(p[2] == "job_1" for p in pulls)


def test_unknown_error_kind_in_results_becomes_unavailable_not_a_crash(tmp_path):
    doc = results_doc()
    doc["training"][0]["error"] = "oom"  # NonceResult validates ERROR_KINDS; this is not one
    c3 = FakeC3(["RUNNING", "SUCCEEDED"], doc)
    b, _ = bench(tmp_path, c3)
    with pytest.raises(BenchUnavailable) as ei:
        b.evaluate(req())
    # mutation: leaving _result_from outside a try lets a malformed results.json crash
    # evaluate with a raw ValueError instead of pausing the run
    assert HASH not in str(ei.value)


def test_non_json_results_becomes_unavailable_not_a_crash(tmp_path):
    c3 = FakeC3(["RUNNING", "SUCCEEDED"], results="not valid json {")
    b, _ = bench(tmp_path, c3)
    with pytest.raises(BenchUnavailable) as ei:
        b.evaluate(req())
    # mutation: leaving json.loads outside a try lets a truncated results.json crash evaluate
    # with a raw JSONDecodeError instead of pausing the run
    assert HASH not in str(ei.value)


def test_request_stop_before_evaluate_raises_cancelled_without_deploying(tmp_path):
    c3 = FakeC3(["RUNNING"], results_doc())
    b, _ = bench(tmp_path, c3)
    b.request_stop()
    with pytest.raises(BenchCancelled):
        b.evaluate(req())
    # mutation: checking _stop only inside _wait deploys a fresh job before noticing the stop,
    # then immediately cancels the job it just paid to submit
    assert not any(c[1] == "deploy" for (c, _) in c3.calls)


def test_request_stop_before_evaluate_cancels_a_reattachable_job(tmp_path):
    from talos.c3_jobdir import request_hash
    c3 = FakeC3(["RUNNING"], results_doc(), job_id="job_old")
    pending = PendingJobStore.memory()
    pending.set({"purpose": 3, "job_id": "job_old", "request_hash": request_hash(req()),
                 "job_dir": str(tmp_path / "c3" / "3")})
    b, _ = bench(tmp_path, c3, pending=pending)
    b.request_stop()
    with pytest.raises(BenchCancelled):
        b.evaluate(req())
    # mutation: clearing the record without cancelling leaves a billing job nobody can find
    assert any(c[1] == "cancel" and c[2] == "job_old" for (c, _) in c3.calls)
    assert not any(c[1] == "deploy" for (c, _) in c3.calls)
    assert pending.get() is None


def test_pending_too_long_cancels_and_pauses(tmp_path):
    c3 = FakeC3(["PENDING"], results_doc())
    pending = PendingJobStore.memory()
    b, clock = bench(tmp_path, c3, pending=pending)
    with pytest.raises(BenchUnavailable) as ei:
        b.evaluate(req())
    # mutation: dropping the profile from the message loses the "which hardware is scarce"
    # signal a human reading the pause needs
    assert "capacity" in str(ei.value) and "cpu-d3-4vcpu-16gb" in str(ei.value)
    # mutation: forgetting the cancel leaves a job that will start and bill while nobody waits
    assert any(c[1] == "cancel" for (c, _) in c3.calls)
    assert clock.t >= 1800
    # mutation: leaving the pending record makes the next resume reattach to a cancelled job
    assert pending.get() is None


def test_request_stop_cancels_the_job_and_raises_cancelled(tmp_path):
    c3 = FakeC3(["RUNNING"], results_doc())
    pending = PendingJobStore.memory()
    calls = 0
    real_c3 = c3.__call__
    holder = {}  # `b` does not exist until `bench()` returns; a one-slot holder closes the loop

    def counting(cmd, **kw):
        nonlocal calls
        calls += 1
        if calls == 3:
            holder["b"].request_stop()
        return real_c3(cmd, **kw)
    b, _ = bench(tmp_path, counting, pending=pending)
    holder["b"] = b
    with pytest.raises(BenchCancelled):
        b.evaluate(req())
    assert any(c[1] == "cancel" for (c, _) in c3.calls)
    # mutation: leaving the pending record makes the next resume reattach to a cancelled job
    assert pending.get() is None


def test_reattaches_to_a_pending_job_with_the_same_request_hash(tmp_path):
    from talos.c3_jobdir import request_hash
    c3 = FakeC3(["SUCCEEDED"], results_doc(), job_id="job_old")
    pending = PendingJobStore.memory()
    pending.set({"purpose": 3, "job_id": "job_old", "request_hash": request_hash(req()),
                 "job_dir": str(tmp_path / "c3" / "3")})
    b, _ = bench(tmp_path, c3, pending=pending)
    r = b.evaluate(req())
    assert r.compile.ok
    # mutation: ignoring the pending record submits a second job for work already paid for
    assert not any(c[1] == "deploy" for (c, _) in c3.calls)


def test_pending_job_with_a_different_hash_is_not_reattached(tmp_path):
    c3 = FakeC3(["SUCCEEDED"], results_doc())
    pending = PendingJobStore.memory()
    pending.set({"purpose": 3, "job_id": "job_old", "request_hash": "0" * 16})
    b, _ = bench(tmp_path, c3, pending=pending)
    b.evaluate(req())
    # mutation: reattaching on a hash mismatch would return stale results for a rewritten request
    assert any(c[1] == "deploy" for (c, _) in c3.calls)
    assert pending.get()["job_id"] == "job_1" and pending.get()["purpose"] == 3


def test_squeue_failures_are_tolerated_up_to_the_limit(tmp_path):
    c3 = FakeC3(["RUNNING", "SUCCEEDED"], results_doc(), squeue_fail=3)
    b, _ = bench(tmp_path, c3)
    assert b.evaluate(req()).compile.ok
    c3b = FakeC3(["RUNNING"], results_doc(), squeue_fail=99)
    b2, _ = bench(tmp_path / "b", c3b)
    with pytest.raises(BenchUnavailable) as ei:
        b2.evaluate(req())
    # mutation: treating one squeue blip as an outage pauses the run on every network hiccup
    assert "unreachable" in str(ei.value)


def test_error_messages_never_carry_the_rand_hash(tmp_path):
    def run(cmd, **kw):
        import types
        return types.SimpleNamespace(returncode=1, stdout="",
                                     stderr=f"bad request {HASH} in payload")
    b, _ = bench(tmp_path, run)
    with pytest.raises(BenchUnavailable) as ei:
        b.evaluate(req())
    # mutation: raising with the raw stderr instead of the _redact'ed text leaks the rand hash
    assert HASH not in str(ei.value) and "<hash>" in str(ei.value)


def test_fill_timeouts_orders_by_track_and_nonce():
    rows = [NonceResult("t", 1, True, 5, 1)]
    out = fill_timeouts(rows, [NonceSet("t", HASH, 0, 3)])
    # mutation: appending real rows before filled timeouts (instead of walking the nonce
    # range in order) would put nonce 1 first instead of in its slot
    assert [(r.nonce, r.error) for r in out] == [(0, "timeout"), (1, None), (2, "timeout")]


def _terminal_without_results(tmp_path, terminal, total_deploys):
    """Two evaluates against a bench whose job reaches `terminal` with no results.json."""
    c3 = FakeC3(["RUNNING", terminal], results=None)
    pending = PendingJobStore.memory()
    pending.set({"purpose": "3", "hypothesis": {"idea": "swap the pivot"}})
    b, _ = bench(tmp_path, c3, pending=pending)
    with pytest.raises(BenchUnavailable):
        b.evaluate(req())
    rec = pending.get()
    # mutation: keeping the reattach keys makes every `talos run --resume` reattach to a job
    # that is already terminal, find no artifacts and pause again without deploying anything
    assert not {"job_id", "request_hash", "job_dir"} & set(rec)
    # mutation: clearing the whole record loses the iteration, spending a fresh hypothesis
    assert rec["purpose"] == "3" and rec["hypothesis"] == {"idea": "swap the pivot"}
    with pytest.raises(BenchUnavailable):
        b.evaluate(req())
    assert len([c for (c, _) in c3.calls if c[1] == "deploy"]) == total_deploys
    return rec


def test_a_timed_out_job_without_results_is_forgotten_so_a_resume_redeploys(tmp_path):
    _terminal_without_results(tmp_path, "TIMED_OUT", total_deploys=2)


def test_a_succeeded_job_without_results_is_forgotten_so_a_resume_redeploys(tmp_path):
    _terminal_without_results(tmp_path, "SUCCEEDED", total_deploys=2)


def test_a_job_that_failed_twice_leaves_no_job_id_on_record(tmp_path):
    # FAILED is resubmitted once inside each evaluate, so two evaluates deploy four times
    rec = _terminal_without_results(tmp_path, "FAILED", total_deploys=4)
    assert "job_id" not in rec


def test_a_missing_c3_binary_pauses_the_run_instead_of_tracebacking(tmp_path):
    def run(cmd, **kw):
        raise FileNotFoundError(2, "No such file or directory", "c3")
    b, _ = bench(tmp_path, run)
    with pytest.raises(BenchUnavailable) as ei:
        b.evaluate(req())
    # mutation: letting OSError escape _c3 crashes evaluate into execute_job's blanket handler,
    # which marks the whole job failed instead of pausing it for a resume
    assert "deploy" in str(ei.value)


def test_a_hung_c3_call_pauses_the_run_instead_of_tracebacking(tmp_path):
    import subprocess

    def run(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, kw.get("timeout"))
    b, _ = bench(tmp_path, run)
    with pytest.raises(BenchUnavailable) as ei:
        b.evaluate(req())
    # mutation: TimeoutExpired is not an OSError, so catching OSError alone in _c3 lets a
    # `c3 deploy` that hangs past its timeout escape evaluate into execute_job's blanket
    # handler, which marks the whole job failed instead of pausing it for a resume
    assert "deploy" in str(ei.value) and "timed out" in str(ei.value)


def test_every_c3_call_carries_the_api_key_in_its_environment(tmp_path):
    c3 = FakeC3(["RUNNING", "SUCCEEDED"], results_doc())
    clock = Clock()

    def sleep(s):
        clock.t += s
    b = C3Bench(tmp_path, run=c3, clock=clock, sleep=sleep, api_key="c3_key_secret",
                hardware="cpu-d3-4vcpu-16gb")
    b.evaluate(req())
    assert {cmd[1] for cmd, _ in c3.calls} >= {"deploy", "squeue", "pull"}
    # mutation: passing env= on deploy only authenticates the submit and not the polling
    assert all(env and env["C3_API_KEY"] == "c3_key_secret" and "PATH" in env for env in c3.envs)
    assert all("c3_key_secret" not in " ".join(cmd) for cmd, _ in c3.calls)


def test_without_an_api_key_c3_calls_inherit_the_environment(tmp_path):
    c3 = FakeC3(["RUNNING", "SUCCEEDED"], results_doc())
    b, _ = bench(tmp_path, c3)
    b.evaluate(req())
    # mutation: forcing C3_API_KEY into the env for a login-session user
    assert c3.envs and all(env is None for env in c3.envs)


def _no_cli(cmd, **kw):
    # Safety: if C3Bench ever ignored `transport=`, this stops it reaching a logged-in `c3`.
    raise AssertionError(f"a transport was injected; C3Bench must not run {cmd[:2]}")


class FakeTransport:
    """A C3Transport with no CLI and no HTTP behind it."""

    name = "fake"

    def __init__(self, statuses, results=None):
        self.statuses, self.results = list(statuses), results
        self.job_ids = ["job_1", "job_2", "job_3"]
        self.calls = []

    def deploy(self, job_dir):
        self.calls.append(("deploy", str(job_dir)))
        return self.job_ids.pop(0)

    def status(self, job_id):
        self.calls.append(("status", job_id))
        return self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]

    def cancel(self, job_id):
        self.calls.append(("cancel", job_id))

    def fetch(self, job_id, name, dest):
        self.calls.append(("fetch", job_id, name))
        if name != "results.json" or self.results is None:
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(self.results), encoding="utf-8", newline="\n")
        return True


def tbench(tmp_path, t):
    clock = Clock()

    def sleep(s):
        clock.t += s
    return C3Bench(tmp_path, pending=PendingJobStore.memory(), run=_no_cli, transport=t,
                   clock=clock, sleep=sleep, poll_s=20.0, pending_timeout_s=1800,
                   hardware="cpu-d3-4vcpu-16gb")


def test_an_injected_transport_carries_the_whole_job(tmp_path):
    t = FakeTransport(["PENDING", "RUNNING", "SUCCEEDED"], results_doc())
    r = tbench(tmp_path, t).evaluate(req())
    # mutation: C3Bench ignoring `transport=` and building its own CliTransport
    assert r.compile.ok and ("fetch", "job_1", "results.json") in t.calls
    assert [c[0] for c in t.calls].count("deploy") == 1


def test_an_unusable_job_id_is_rejected_before_any_status_or_fetch_call(tmp_path):
    t = FakeTransport(["SUCCEEDED"], results_doc())
    t.job_ids = ["../../x"]
    with pytest.raises(BenchUnavailable) as ei:
        tbench(tmp_path, t).evaluate(req())
    # mutation: dropping the job id check lets it become job_dir/../../x/artifacts on disk
    assert "unusable job id" in str(ei.value)
    assert not any(c[0] in ("status", "fetch") for c in t.calls)


def test_pending_timeout_cancels_through_the_transport(tmp_path):
    t = FakeTransport(["PENDING"])
    with pytest.raises(BenchUnavailable) as ei:
        tbench(tmp_path, t).evaluate(req())
    # mutation: a cancel left on the deleted CLI helper leaves a queued job billing later
    assert "no C3 capacity" in str(ei.value) and ("cancel", "job_1") in t.calls


def test_a_failed_job_is_resubmitted_once_through_the_transport(tmp_path):
    t = FakeTransport(["FAILED"], results=None)
    with pytest.raises(BenchUnavailable) as ei:
        tbench(tmp_path, t).evaluate(req())
    # mutation: treating fetch() == False as a pull error skips the one resubmission
    assert "failed twice" in str(ei.value)
    assert [c[0] for c in t.calls].count("deploy") == 2


def test_success_without_results_is_unavailable_not_a_resubmission(tmp_path):
    t = FakeTransport(["SUCCEEDED"], results=None)
    with pytest.raises(BenchUnavailable) as ei:
        tbench(tmp_path, t).evaluate(req())
    # mutation: ignoring fetch's return value reads a results.json that is not there
    assert "without results.json" in str(ei.value)
    assert [c[0] for c in t.calls].count("deploy") == 1


class FakeTransportWithStaleResults(FakeTransport):
    """fetch writes a results.json to `dest` (as a real pull sometimes leaves one behind from an
    earlier, unrelated attempt) but still reports the fetch as having found nothing."""

    def fetch(self, job_id, name, dest):
        self.calls.append(("fetch", job_id, name))
        if name == "results.json":
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(results_doc()), encoding="utf-8", newline="\n")
        return False


def test_a_stale_results_file_on_disk_is_not_mistaken_for_a_successful_fetch(tmp_path):
    t = FakeTransportWithStaleResults(["FAILED"])
    with pytest.raises(BenchUnavailable) as ei:
        tbench(tmp_path, t).evaluate(req())
    # mutation: checking results.exists() instead of trusting fetch's return value would read
    # this stale file left on disk and report a result instead of failing twice
    assert "failed twice" in str(ei.value)
    assert [c[0] for c in t.calls].count("deploy") == 2


def test_local_settings_select_the_local_subdir_flavour_and_a_zero_rate(tmp_path):
    t = FakeTransport(["PENDING", "RUNNING", "RUNNING", "SUCCEEDED"], results_doc())
    clock = Clock()

    def sleep(s):
        clock.t += s  # the job runs for 40 s of billable time; at $0/h that is still $0
    b = C3Bench(tmp_path, pending=PendingJobStore.memory(), run=_no_cli, transport=t,
                clock=clock, sleep=sleep, poll_s=20.0, local=LocalSettings(8, 12),
                usd_per_hour=0.0)
    r = b.evaluate(req())
    assert r.compile.ok
    job_dir = Path([c[1] for c in t.calls if c[0] == "deploy"][0])
    # mutation: writing the C3 flavour under runs/<id>/c3 on the local backend
    assert job_dir.parent.name == "local" and (job_dir / "local.json").exists()
    assert not (job_dir / ".c3").exists()
    assert json.loads((job_dir / "local.json").read_text())["cpus"] == 8
    # mutation: `usd_per_hour or GBP_RATE` treats 0.0 as "not given" and bills local time
    assert b.cost_mark() == 0.0


def test_a_given_rate_is_charged_and_none_keeps_the_c3_rate(tmp_path):
    t = FakeTransport(["RUNNING", "RUNNING", "SUCCEEDED"], results_doc())
    clock = Clock()

    def sleep(s):
        clock.t += s
    b = C3Bench(tmp_path, pending=PendingJobStore.memory(), run=_no_cli, transport=t,
                clock=clock, sleep=sleep, poll_s=20.0, usd_per_hour=3.6,
                hardware="cpu-d3-4vcpu-16gb")
    b.evaluate(req())
    # RUNNING first seen at t=0, terminal at t=40: 40 s at $3.6/h = $0.04
    assert abs(b.cost_mark() - 0.04) < 1e-9


def test_messages_name_the_transport_label_when_it_has_one(tmp_path):
    t = FakeTransport(["PENDING"])
    t.label = "local Docker"
    t.statuses = ["FAILED"]
    with pytest.raises(BenchUnavailable) as ei:
        tbench(tmp_path, t).evaluate(req())
    # mutation: a local failure that tells the user to check C3
    assert "local Docker job failed twice" in str(ei.value) and "C3" not in str(ei.value)


# ── Hardware fallback ──────────────────────────────────────────────────────
class ProbeTransport(FakeTransport):
    """Statuses per deployed job, in deploy order; a job's list is replayed like FakeTransport's."""

    def __init__(self, per_job: list[list[str]], results=None):
        super().__init__(["PENDING"], results)
        self.per_job = [list(s) for s in per_job]
        self.job_ids = [f"job_{i + 1}" for i in range(len(per_job) + 1)]
        self.dirs = []

    def deploy(self, job_dir):
        job_dir = Path(job_dir)
        self.dirs.append(job_dir)
        self.statuses = self.per_job.pop(0) if self.per_job else ["PENDING", "RUNNING", "SUCCEEDED"]
        return super().deploy(job_dir)


def _hw(job_dir):
    from talos.c3_jobdir import parse_c3
    return parse_c3((Path(job_dir) / ".c3").read_text(encoding="utf-8"))["hardware"]


def test_select_hardware_probes_each_class_in_turn_until_one_leaves_the_queue(tmp_path):
    t = ProbeTransport([["PENDING"], ["SCHEDULING", "RUNNING"]])
    b = tbench(tmp_path, t)
    assert b.select_hardware("hypergraph") == "a100"
    assert [_hw(d) for d in t.dirs] == ["l40", "a100"]
    # mutation: not cancelling the stuck probe leaves it billing when an L40 frees up; not
    # cancelling the running one bills its whole walltime
    assert ("cancel", "job_1") in t.calls and ("cancel", "job_2") in t.calls
    assert all(d.name.startswith("probe-") for d in t.dirs)
    # mutation: the chosen class must reach the real job's .c3, or invariant 1 breaks
    t.results = results_doc()
    b.evaluate(EvalRequest("hypergraph", {"mod.rs": "x"}, [NonceSet("t", HASH, 0, 2)],
                           [NonceSet("t", HASH, 1_000_000, 2)], 7, None,
                           CHALLENGES["hypergraph"].beat))
    assert _hw(t.dirs[-1]) == "a100"
    from talos.c3_bench import GBP_PER_HOUR, USD_PER_GBP
    assert b.cost_mark() == pytest.approx(20 / 3600 * GBP_PER_HOUR["a100"] * USD_PER_GBP)


def test_select_hardware_reports_all_classes_unavailable(tmp_path):
    t = ProbeTransport([["PENDING"], ["PENDING"], ["SCHEDULING"]])
    with pytest.raises(BenchUnavailable) as ei:
        tbench(tmp_path, t).select_hardware("hypergraph")
    assert [_hw(d) for d in t.dirs] == ["l40", "a100", "h100"]
    assert [c for c in t.calls if c[0] == "cancel"] == [("cancel", "job_1"), ("cancel", "job_2"),
                                                        ("cancel", "job_3")]
    assert "l40" in str(ei.value) and "h100" in str(ei.value) and "1800" in str(ei.value)


def test_select_hardware_treats_a_probe_that_already_finished_as_capacity(tmp_path):
    # a 20 s poll can miss RUNNING on a job whose script is `true`
    t = ProbeTransport([["PENDING", "SUCCEEDED"]])
    assert tbench(tmp_path, t).select_hardware("hypergraph") == "l40"
    assert ("cancel", "job_1") not in t.calls


def test_select_hardware_honours_a_frozen_choice_and_skips_local(tmp_path):
    t = ProbeTransport([])
    b = tbench(tmp_path, t)
    assert b.select_hardware("hypergraph", chosen="h100") == "h100" and t.dirs == []
    with pytest.raises(ValueError):
        b.select_hardware("hypergraph", chosen="L40S")
    assert b.select_hardware("knapsack", chosen="cpu-e2-4vcpu-16gb") == "cpu-e2-4vcpu-16gb"
    assert t.dirs == []
    with pytest.raises(ValueError):
        b.select_hardware("knapsack", chosen="l40")
    local = C3Bench(tmp_path, pending=PendingJobStore.memory(), run=_no_cli, transport=t,
                    local=LocalSettings(4, 8), usd_per_hour=0.0)
    # mutation: probing on the local backend submits a C3-shaped job to Docker
    assert local.select_hardware("hypergraph") is None and t.dirs == []


def test_select_hardware_probes_the_cpu_profiles_for_a_cpu_challenge(tmp_path):
    t = ProbeTransport([["PENDING"], ["SCHEDULING", "RUNNING"]])
    b = tbench(tmp_path, t)
    # mutation: a CPU challenge with no options stays on d3 when d3 is out of stock, which is
    # the shortage that was seen live on 2026-09-23
    assert b.select_hardware("knapsack") == "cpu-e2-4vcpu-16gb"
    assert [_hw(d) for d in t.dirs] == ["cpu-d3-4vcpu-16gb", "cpu-e2-4vcpu-16gb"]
    assert [d.name for d in t.dirs] == ["probe-cpu-d3-4vcpu-16gb", "probe-cpu-e2-4vcpu-16gb"]
    assert ("cancel", "job_1") in t.calls and ("cancel", "job_2") in t.calls
    # the chosen profile reaches the real job's .c3 and its rate
    t.results = results_doc()
    b.evaluate(req())
    assert _hw(t.dirs[-1]) == "cpu-e2-4vcpu-16gb"
    from talos.c3_bench import GBP_PER_HOUR, USD_PER_GBP
    assert b.cost_mark() == pytest.approx(20 / 3600 * GBP_PER_HOUR["cpu-e2-4vcpu-16gb"]
                                          * USD_PER_GBP)


class RefusingProbeTransport(ProbeTransport):
    """deploy raises the given C3CommandError text for a job (None accepts it), then behaves
    like ProbeTransport for the accepted ones."""

    def __init__(self, refusals, per_job):
        super().__init__(per_job)
        self.refusals = list(refusals)

    def deploy(self, job_dir):
        msg = self.refusals.pop(0) if self.refusals else None
        if msg is not None:
            self.dirs.append(Path(job_dir))
            self.calls.append(("deploy", str(job_dir)))
            raise C3CommandError(msg)
        return super().deploy(job_dir)


OUT_OF_STOCK = ("c3 deploy failed (1): 409 GPU_OUT_OF_STOCK: cpu-d3-4vcpu-16gb does not "
                "currently have capacity")


def test_select_hardware_moves_on_when_a_profile_is_refused_at_deploy_as_out_of_stock(tmp_path):
    # C3 can refuse the deploy itself (409, seen live 2026-09-23) rather than queue the job
    t = RefusingProbeTransport([OUT_OF_STOCK, None], [["RUNNING"]])
    b = tbench(tmp_path, t)
    # mutation: treating the refusal as a deploy failure pauses the run without trying e2
    assert b.select_hardware("knapsack") == "cpu-e2-4vcpu-16gb"
    assert [_hw(d) for d in t.dirs] == ["cpu-d3-4vcpu-16gb", "cpu-e2-4vcpu-16gb"]
    # nothing was queued for d3, so there is nothing to cancel there
    assert [c for c in t.calls if c[0] == "cancel"] == [("cancel", "job_1")]


def test_select_hardware_reports_every_profile_refused_as_out_of_stock(tmp_path):
    t = RefusingProbeTransport([OUT_OF_STOCK, OUT_OF_STOCK.replace("d3", "e2")], [])
    with pytest.raises(BenchUnavailable) as ei:
        tbench(tmp_path, t).select_hardware("knapsack")
    assert "cpu-d3-4vcpu-16gb" in str(ei.value) and "cpu-e2-4vcpu-16gb" in str(ei.value)
    assert not any(c[0] == "cancel" for c in t.calls)


def test_select_hardware_still_fails_fast_on_a_deploy_error_that_is_not_a_shortage(tmp_path):
    t = RefusingProbeTransport(["c3 deploy failed (1): 401 unauthorized"], [["RUNNING"]])
    with pytest.raises(BenchUnavailable) as ei:
        tbench(tmp_path, t).select_hardware("knapsack")
    # mutation: moving on from every deploy error retries a bad key on each option and then
    # reports "no capacity", hiding the real cause
    assert "probe deploy failed" in str(ei.value) and "401" in str(ei.value)
    assert len(t.dirs) == 1


def test_select_hardware_stop_request_cancels_the_probe(tmp_path):
    t = ProbeTransport([["PENDING"]])
    b = tbench(tmp_path, t)
    b.request_stop()
    with pytest.raises(BenchCancelled):
        b.select_hardware("hypergraph")
    assert ("cancel", "job_1") in t.calls


def test_a_probe_never_touches_the_pending_record(tmp_path):
    # a resumed pre-change job can hold a real pending job while it is being frozen; a probe
    # that times out (or is stopped) must not erase the record that reattaches to it
    keep = {"purpose": "baseline", "job_id": "job_real", "request_hash": "x", "job_dir": "d"}
    pending = PendingJobStore.memory()
    pending.set(dict(keep))
    t = ProbeTransport([["PENDING"], ["RUNNING"]])
    clock = Clock()
    b = C3Bench(tmp_path, pending=pending, run=_no_cli, transport=t, clock=clock,
                sleep=lambda s: setattr(clock, "t", clock.t + s), poll_s=20.0,
                pending_timeout_s=1800)
    assert b.select_hardware("hypergraph") == "a100"
    assert pending.get() == keep
