import json
from pathlib import Path

import pytest

from talos.bench import BenchCancelled, BenchUnavailable, EvalRequest, PendingJobStore
from talos.c3_bench import C3Bench, fill_timeouts, parse_json_stdout
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
                   sleep=sleep, poll_s=20.0, pending_timeout_s=1800), clock


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
    b, _ = bench(tmp_path, c3, pending=pending)
    calls = 0
    real_c3 = c3.__call__

    def counting(cmd, **kw):
        nonlocal calls
        calls += 1
        if calls == 3:
            b.request_stop()
        return real_c3(cmd, **kw)
    b._run = counting
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
    b = C3Bench(tmp_path, run=c3, clock=clock, sleep=sleep, api_key="c3_key_secret")
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
