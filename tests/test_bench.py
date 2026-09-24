import sys
import types

import pytest

from talos.bench import (BenchUnavailable, EvalRequest, FakeBench, ModalBench, PendingJobStore,
                         _redact)
from talos.challenges import BeatRule
from talos.scoring import holdout_decision
from talos.types import NonceResult, NonceSet

TR = [NonceSet("t", "ab" * 32, 0, 3)]
HO = [NonceSet("t", "ab" * 32, 1_000_000, 3)]


def req(files=None, baseline=None, training=TR, holdout=HO):
    return EvalRequest(challenge="knapsack", files=files or {"mod.rs": "x"}, training=training,
                       holdout=holdout, fuel=1, baseline_training=baseline, rule=BeatRule())


def base(q=100):
    return [NonceResult("t", n, True, q, 1) for n in range(3)]


def batch_rows(args, quality=100, runtime_ms=1, seconds=1.0):
    """What a `score_batch` starmap returns: one {"rows", "seconds"} per (artifact_id, tasks)
    argument, the rows in task order."""
    return [{"rows": [{"track": t["track"], "nonce": t["nonce"], "ok": True, "quality": quality,
                       "runtime_ms": runtime_ms, "error": None} for t in tasks],
             "seconds": seconds} for (_a, tasks) in args]


def batch_tasks(args):
    """Every task dict across a starmap's batches, in call order."""
    return [t for (_a, tasks) in args for t in tasks]


def test_holdout_decision_forced_won_not_won():
    tr = base(110)
    # mutation: treating a None baseline as "never score held-out" makes the baseline
    # measurement return no held-out results and resolve_baseline fails every job
    assert holdout_decision(None, tr, BeatRule()) == (True, "forced")
    assert holdout_decision(base(100), tr, BeatRule()) == (True, "won")
    assert holdout_decision(base(100), base(100), BeatRule()) == (False, "not_won")
    # mutation: letting ScoringError escape here crashes the C3 job instead of skipping held-out
    short = [NonceResult("t", 0, True, 100, 1)]
    assert holdout_decision(short, tr, BeatRule()) == (False, "not_won")


def test_fake_bench_scores_training_and_tracks_cost():
    fb = FakeBench(lambda ch, files, ns: [100 + n for n in ns.nonces()])
    mark = fb.cost_mark()
    r = fb.evaluate(req(baseline=base(200)))
    assert r.compile.ok and r.compile.artifact_id
    assert [x.quality for x in r.training] == [100, 101, 102]
    assert all(x.track == "t" for x in r.training)
    # mutation: scoring held-out on a loss wastes a run per iteration and hides losses from tests
    assert r.holdout is None and r.holdout_reason == "not_won" and fb.holdout_runs == 0
    assert fb.cost_usd_since(mark) > 0  # mutation: not charging makes budget tests vacuous
    assert fb.calls == [req(baseline=base(200))]


def test_fake_bench_scores_holdout_on_a_win_and_when_forced():
    fb = FakeBench(lambda ch, files, ns: [110 for _ in ns.nonces()])
    won = fb.evaluate(req(baseline=base(100)))
    # mutation: scoring held-out only on "won" and never on "forced" (or the reverse) leaves
    # the baseline job with no held-out results
    assert (won.holdout_reason == "won"
            and [x.nonce for x in won.holdout] == [1_000_000, 1_000_001, 1_000_002])
    forced = fb.evaluate(req(baseline=None))
    assert forced.holdout_reason == "forced" and len(forced.holdout) == 3
    assert fb.holdout_runs == 2


def test_fake_bench_none_means_error():
    fb = FakeBench(lambda ch, files, ns: [None, 5, 5])
    r = fb.evaluate(req(baseline=base(1)))
    # mutation: mapping a None quality to ok=True would feed phantom solutions to scoring
    assert (not r.training[0].ok and r.training[0].error == "no_solution"
            and r.training[1].quality == 5)


def test_fake_bench_compile_failure_scores_nothing():
    fb = FakeBench(lambda ch, files, ns: [1, 1, 1],
                   compile_ok=lambda files: "BUG" not in files["mod.rs"])
    r = fb.evaluate(req(files={"mod.rs": "BUG"}))
    # mutation: ignoring compile_ok makes every compile-failure path in the loop untestable;
    # scoring a failed compile would hand the loop qualities for code that never built
    assert not r.compile.ok and r.training == [] and r.holdout is None
    assert r.holdout_reason == "not_compiled"


def test_fake_bench_rejects_a_short_scores_callback():
    fb = FakeBench(lambda ch, files, ns: [1, 2])
    # mutation: zip() silently truncates, so a 2-quality callback would score a 3-nonce set
    # as 2 results and every count-based assertion downstream would quietly pass
    with pytest.raises(ValueError):
        fb.evaluate(req())


def test_pending_job_store_in_memory_roundtrip():
    # mutation: a memory store whose set() is a no-op makes every reattach test pass vacuously
    p = PendingJobStore.memory()
    assert p.get() is None
    p.set({"job_id": "j"})
    assert p.get() == {"job_id": "j"}


def test_redact_hides_a_rand_hash():
    # mutation: dropping the redaction leaks the seed into the pause message
    assert _redact("x " + "ab" * 32 + " y") == "x <hash> y"


def test_redact_hides_a_c3_api_key():
    out = _redact("sending Bearer c3_key_Ab-9_z to the server")
    # mutation: a key echoed by C3 or by an exception reaches the pause message and state.json
    assert "c3_key_" not in out and "Ab-9_z" not in out and "to the server" in out


def test_redact_hides_a_long_key_even_when_the_hash_pattern_matches_inside_it():
    # a 64-char key body contains a 64-hex-char run, which the hash regex alone would consume,
    # leaving the tail of the key exposed in the output
    out = _redact("Bearer c3_key_" + "ab" * 32 + "ZZTOPSECRET")
    assert "ZZTOPSECRET" not in out and "c3_key_" not in out


class FakeClock:
    """Wall clock the tests advance by hand, so retry behaviour is exercised without sleeping."""

    def __init__(self, t: float = 0.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


def recording_sleep(clock: FakeClock, sleeps: list):
    def sleep(d):
        sleeps.append(d)
        clock.t += d
    return sleep


def test_with_retry_returns_after_transient_failures():
    clock = FakeClock()
    sleeps = []
    b = ModalBench(clock=clock, sleep=recording_sleep(clock, sleeps))
    calls = []

    def call():
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError("boom")
        return 42

    assert b._with_retry(call) == 42
    # mutation: a fixed delay, or no backoff at all, hammers Modal during an outage
    assert sleeps == [5, 10]


def test_with_retry_window_is_anchored_at_the_first_failure():
    clock = FakeClock()
    sleeps = []
    b = ModalBench(retry_window_s=900, clock=clock, sleep=recording_sleep(clock, sleeps))

    def call():
        if not sleeps:
            clock.t += 1000  # the first call itself outlasts the whole retry window
        raise ConnectionError("boom")

    with pytest.raises(BenchUnavailable):
        b._with_retry(call)
    # mutation: anchoring the deadline at call start makes a long first call raise with
    # zero sleeps, so a 64-nonce starmap gets no retry at all
    assert sleeps and sum(sleeps) <= 900


def _fake_modal(raiser):
    """A stand-in `modal` module. `raiser(NotFoundError)` returns the exception to raise from
    Function.from_name, so a test can pick the real NotFoundError class or anything else."""
    class NotFoundError(Exception):
        pass

    def from_name(app_name, name):
        raise raiser(NotFoundError)

    mod = types.ModuleType("modal")
    mod.exception = types.SimpleNamespace(NotFoundError=NotFoundError)
    mod.Function = types.SimpleNamespace(from_name=from_name)
    return mod


def test_missing_deployment_says_run_talos_setup(monkeypatch):
    mod = _fake_modal(lambda nf: nf("app has no function score_batch_knapsack"))
    monkeypatch.setitem(sys.modules, "modal", mod)
    clock = FakeClock()
    sleeps = []
    b = ModalBench(clock=clock, sleep=recording_sleep(clock, sleeps))
    with pytest.raises(BenchUnavailable) as ei:
        b.evaluate(req())
    assert "talos setup" in str(ei.value)
    # mutation: retrying a NotFoundError wastes the whole window on a deterministic error
    assert sleeps == []


def test_transient_lookup_failure_is_retried_not_blamed_on_setup(monkeypatch):
    mod = _fake_modal(lambda nf: ConnectionError("connection reset"))
    monkeypatch.setitem(sys.modules, "modal", mod)
    clock = FakeClock()
    sleeps = []
    b = ModalBench(retry_window_s=20, clock=clock, sleep=recording_sleep(clock, sleeps))
    with pytest.raises(BenchUnavailable) as ei:
        b.evaluate(req())
    # mutation: converting every lookup error to "not deployed" pauses the run on a network
    # blip and prints the wrong advice
    assert "talos setup" not in str(ei.value)
    assert sleeps


def test_lookup_is_cached_after_a_successful_hydrate(monkeypatch):
    lookups = []

    class Fn:
        def hydrate(self):
            pass

        def remote(self, files):
            return {"ok": True, "artifact_id": "art", "output": "ok"}

        def starmap(self, args):
            return batch_rows(args, quality=1)

    def from_name(app_name, name):
        lookups.append(name)
        return Fn()

    mod = types.ModuleType("modal")
    mod.exception = types.SimpleNamespace(NotFoundError=type("NotFoundError", (Exception,), {}))
    mod.Function = types.SimpleNamespace(from_name=from_name)
    monkeypatch.setitem(sys.modules, "modal", mod)

    b = ModalBench()
    for _ in range(3):
        b.evaluate(req())
    # mutation: re-hydrating on every call adds a round trip per nonce batch
    assert lookups == ["compile_knapsack", "score_batch_knapsack"]


def test_modal_evaluate_skips_holdout_on_a_loss_and_charges_each_call(monkeypatch):
    starmaps = []

    class Fn:
        def hydrate(self):
            pass

        def remote(self, files):
            return {"ok": True, "artifact_id": "art", "output": "ok"}

        def starmap(self, args):
            starmaps.append(len(batch_tasks(args)))
            return batch_rows(args, runtime_ms=1000)

    mod = types.ModuleType("modal")
    mod.exception = types.SimpleNamespace(NotFoundError=type("NotFoundError", (Exception,), {}))
    mod.Function = types.SimpleNamespace(from_name=lambda app, name: Fn())
    monkeypatch.setitem(sys.modules, "modal", mod)
    b = ModalBench()
    r = b.evaluate(req(baseline=base(100)))  # candidate 100 vs baseline 100: not a win
    # mutation: unconditional held-out scoring doubles Modal spend on every losing iteration
    assert starmaps == [3] and r.holdout is None and r.holdout_reason == "not_won"
    assert b.cost_mark() > 0
    r2 = b.evaluate(req(baseline=base(50)))
    assert starmaps == [3, 3, 3] and r2.holdout_reason == "won" and len(r2.holdout) == 3


def test_modal_score_never_starmaps_an_empty_nonce_list(monkeypatch):
    """`draw_nonce_sets(..., holdout_count=0)` returns one NonceSet per track with count 0, so
    `request.holdout` is a non-empty list that expands to no starmap args. Measured against
    modal 1.5.5: `Function.starmap([])` never returns, so the client hung after a compile and a
    training score that had both succeeded."""
    starmaps = []

    class Fn:
        def hydrate(self):
            pass

        def remote(self, files):
            return {"ok": True, "artifact_id": "art", "output": "ok"}

        def starmap(self, args):
            starmaps.append(len(batch_tasks(args)))
            return batch_rows(args, runtime_ms=1000)

    mod = types.ModuleType("modal")
    mod.exception = types.SimpleNamespace(NotFoundError=type("NotFoundError", (Exception,), {}))
    mod.Function = types.SimpleNamespace(from_name=lambda app, name: Fn())
    monkeypatch.setitem(sys.modules, "modal", mod)
    empty = [NonceSet(track="t", rand_hash="ab" * 32, start=1_000_000, count=0)]
    r = ModalBench().evaluate(req(baseline=None, holdout=empty))
    # mutation: starmapping the empty holdout hangs the real client forever
    assert starmaps == [3]
    assert r.holdout == [] and r.holdout_reason == "forced" and len(r.training) == 3


DEAD_WARNING = ("warning: function `polish` is never used\n"
                "   --> tig-algorithms/src/knapsack/talos_cand/mod.rs:3:4\n")


def test_fake_bench_reports_dead_new_code_without_scoring():
    scored = []
    fb = FakeBench(lambda ch, files, ns: scored.append(ns) or [100] * ns.count,
                   compile_output=lambda files: DEAD_WARNING)
    r = req(baseline=base(100))
    r.prior_functions = {"mod.rs": ["x"]}
    out = fb.evaluate(r)
    # mutation: scoring the dead candidate anyway costs the loop a full C3 job for a no-op
    assert out.compile.ok and out.holdout_reason == "dead_code" and out.training == []
    assert scored == []
    # a request without prior_functions (the baseline measurement) is never checked
    assert fb.evaluate(req(baseline=base(100))).holdout_reason == "not_won"


def test_modal_evaluate_stops_after_a_build_with_dead_new_code(monkeypatch):
    starmaps = []

    class Fn:
        def hydrate(self):
            pass

        def remote(self, files):
            return {"ok": True, "artifact_id": "art", "output": DEAD_WARNING}

        def starmap(self, args):
            starmaps.append(len(args))
            return []

    mod = types.ModuleType("modal")
    mod.exception = types.SimpleNamespace(NotFoundError=type("NotFoundError", (Exception,), {}))
    mod.Function = types.SimpleNamespace(from_name=lambda app, name: Fn())
    monkeypatch.setitem(sys.modules, "modal", mod)
    r = req(baseline=base(100))
    r.prior_functions = {"mod.rs": ["x"]}
    out = ModalBench().evaluate(r)
    # mutation: starmapping the nonces before the check spends Modal time on a no-op
    assert out.holdout_reason == "dead_code" and out.training == [] and starmaps == []


def test_modal_starmap_carries_the_per_track_timeout(monkeypatch):
    from talos.inside import NONCE_TIMEOUT_S
    timeouts = []

    class Fn:
        def hydrate(self):
            pass

        def remote(self, files):
            return {"ok": True, "artifact_id": "art", "output": "ok"}

        def starmap(self, args):
            timeouts.extend(t["timeout_s"] for t in batch_tasks(args))
            return batch_rows(args)

    mod = types.ModuleType("modal")
    mod.exception = types.SimpleNamespace(NotFoundError=type("NotFoundError", (Exception,), {}))
    mod.Function = types.SimpleNamespace(from_name=lambda app, name: Fn())
    monkeypatch.setitem(sys.modules, "modal", mod)
    r = req(baseline=base(100))
    r.timeouts = {"t": 42}
    ModalBench().evaluate(r)
    # mutation: a flat NONCE_TIMEOUT_S in the starmap args ignores the loop's per-track cap
    assert timeouts == [42, 42, 42]
    timeouts.clear()
    ModalBench().evaluate(req(baseline=base(100)))
    assert timeouts == [NONCE_TIMEOUT_S] * 3


def test_a_stale_deploy_is_reported_at_once_not_retried_as_an_outage(monkeypatch):
    # The PR that added timeout_s to the score function needs `talos setup` re-run on Modal. Without
    # this check a client talking to the old deploy retried the TypeError for the whole
    # 900 s window and then reported "Modal unreachable". mutation: treating the TypeError
    # like a transport error brings the 17 retries and the wrong diagnosis back
    class Fn:
        def hydrate(self):
            pass

        def remote(self, files):
            return {"ok": True, "artifact_id": "art", "output": "ok"}

        def starmap(self, args):
            raise TypeError("score_batch() takes 2 positional arguments but 3 were given")

    mod = types.ModuleType("modal")
    mod.exception = types.SimpleNamespace(NotFoundError=type("NotFoundError", (Exception,), {}))
    mod.Function = types.SimpleNamespace(from_name=lambda app, name: Fn())
    monkeypatch.setitem(sys.modules, "modal", mod)
    clock, sleeps = FakeClock(), []
    b = ModalBench(clock=clock, sleep=recording_sleep(clock, sleeps))
    with pytest.raises(BenchUnavailable, match="talos setup"):
        b.evaluate(req(baseline=base(100)))
    assert sleeps == []


def test_hyperparameters_for_a_track():
    from talos.bench import hyperparameters_for
    hp = {"t": {"x": 1}, "u": None, "e": {}}
    assert hyperparameters_for(hp, "t") == {"x": 1}
    # mutation: `or None` would turn a track's {} into None
    assert hyperparameters_for(hp, "e") == {}
    assert hyperparameters_for(hp, "u") is None and hyperparameters_for(hp, "missing") is None
    assert hyperparameters_for(None, "t") is None


def test_modal_starmap_carries_each_tracks_hyperparameters(monkeypatch):
    seen = []

    class Fn:
        def hydrate(self):
            pass

        def remote(self, files):
            return {"ok": True, "artifact_id": "art", "output": "ok"}

        def starmap(self, args):
            seen.extend((t["track"], t["hyperparameters"]) for t in batch_tasks(args))
            return batch_rows(args)

    mod = types.ModuleType("modal")
    mod.exception = types.SimpleNamespace(NotFoundError=type("NotFoundError", (Exception,), {}))
    mod.Function = types.SimpleNamespace(from_name=lambda app, name: Fn())
    monkeypatch.setitem(sys.modules, "modal", mod)
    two = [NonceSet("t", "ab" * 32, 0, 1), NonceSet("u", "ab" * 32, 0, 1)]
    r = req(training=two, holdout=[])
    r.hyperparameters = {"t": {"x": 1}, "u": None}
    ModalBench().evaluate(r)
    # mutation: dropping the element from the args runs every nonce without hyperparameters
    # mutation: passing the whole map instead of the track's entry hands tig-runtime {"t": ...}
    assert seen == [("t", {"x": 1}), ("u", None)]
    seen.clear()
    ModalBench().evaluate(req(training=two, holdout=[]))
    assert seen == [("t", None), ("u", None)]
    seen.clear()
    # forced held-out (no baseline_training): the holdout starmap call must carry the map too
    r_ho = req(training=[], holdout=two, baseline=None)
    r_ho.hyperparameters = {"t": {"x": 1}, "u": None}
    ModalBench().evaluate(r_ho)
    # mutation: passing None instead of request.hyperparameters to the holdout _score call
    assert seen == [("t", {"x": 1}), ("u", None)]


def _batching_modal(monkeypatch, batches: list, seconds: float = 2.0):
    class Fn:
        def hydrate(self):
            pass

        def spawn(self):
            raise AssertionError("no probe expected")

        def remote(self, files):
            return {"ok": True, "artifact_id": "art", "output": "ok"}

        def starmap(self, args):
            batches.extend(args)
            return batch_rows(args, runtime_ms=50, seconds=seconds)

    mod = types.ModuleType("modal")
    mod.exception = types.SimpleNamespace(NotFoundError=type("NotFoundError", (Exception,), {}))
    mod.Function = types.SimpleNamespace(from_name=lambda app, name: Fn())
    monkeypatch.setitem(sys.modules, "modal", mod)


def test_modal_scores_nonces_in_batches_of_the_containers_worker_count(monkeypatch):
    from talos.challenges import CHALLENGES
    batches = []
    _batching_modal(monkeypatch, batches)
    ten = [NonceSet("t", "ab" * 32, 0, 7), NonceSet("u", "ab" * 32, 0, 3)]
    r = ModalBench().evaluate(req(training=ten, holdout=[], baseline=None))
    cpu = CHALLENGES["knapsack"].cpu
    # mutation: one nonce per call idles three of the container's four billed cores;
    # a batch larger than the worker count queues nonces behind each other inside one
    # container and can outlive the function timeout
    assert [len(tasks) for (_a, tasks) in batches] == [cpu, cpu, 2]
    assert all(a == "art" for (a, _tasks) in batches)
    # mutation: chunking by track leaves the short track's batch under-filled
    assert [(t["track"], t["nonce"]) for t in batch_tasks(batches)] == (
        [("t", n) for n in range(7)] + [("u", n) for n in range(3)])
    # the rows come back flattened, in the order the tasks went out
    assert [(x.track, x.nonce) for x in r.training] == [(t["track"], t["nonce"])
                                                         for t in batch_tasks(batches)]


def test_modal_charges_each_batch_by_its_container_seconds(monkeypatch):
    from talos.bench import _seconds_cost
    batches = []
    _batching_modal(monkeypatch, batches, seconds=2.0)
    b = ModalBench(clock=FakeClock())  # a frozen clock: the compile call charges nothing
    b.evaluate(req(training=TR, holdout=[], baseline=None))  # 3 nonces: one batch of 2.0 s
    # mutation: charging each nonce's runtime_ms (3 x 0.05 s) undercounts the container by
    # more than ten times here; the container bills its wall time, verifier included
    assert b.cost_mark() == pytest.approx(_seconds_cost("knapsack", 2.0, None), rel=1e-9)
    assert b.cost_mark() > _seconds_cost("knapsack", 3 * 0.05, None)


def test_a_gpu_challenge_scores_one_nonce_per_batch(monkeypatch):
    batches = []
    _batching_modal(monkeypatch, batches)
    b = ModalBench()
    b.select_gpu("hypergraph", chosen="L40S")
    b.evaluate(gpu_req())  # 3 training nonces
    # mutation: packing a GPU batch like a CPU one serialises the nonces on the device
    assert [len(tasks) for (_a, tasks) in batches] == [1, 1, 1]


# ── GPU fallback ──────────────────────────────────────────────────────
def _fake_modal_gpu(monkeypatch, probes: dict, lookups: list, cancelled: list, clock=None):
    """`probes` maps a probe function name to True (starts at once) or False (never starts).
    A successful probe advances `clock` by 3 s, so its charge is checkable."""
    class Call:
        def __init__(self, name):
            self.name = name

        def get(self, timeout=None):
            if probes[self.name]:
                if clock is not None:
                    clock.t += 3
                return self.name
            raise TimeoutError()  # modal 1.5.5 raises the builtin from FunctionCall.get

        def cancel(self):
            cancelled.append(self.name)

    class Fn:
        def __init__(self, name):
            self.name = name

        def hydrate(self):
            pass

        def spawn(self):
            return Call(self.name)

        def remote(self, files):
            return {"ok": True, "artifact_id": "art", "output": f"built on {self.name}"}

        def starmap(self, args):
            return batch_rows(args, quality=1, runtime_ms=1000, seconds=1.0)

    def from_name(app_name, name):
        lookups.append(name)
        return Fn(name)

    mod = types.ModuleType("modal")
    mod.exception = types.SimpleNamespace(NotFoundError=type("NotFoundError", (Exception,), {}))
    mod.Function = types.SimpleNamespace(from_name=from_name)
    monkeypatch.setitem(sys.modules, "modal", mod)


def gpu_req():
    return EvalRequest(challenge="hypergraph", files={"mod.rs": "x"}, training=TR, holdout=[],
                       fuel=1, baseline_training=None, rule=BeatRule())


def test_select_gpu_takes_the_first_gpu_whose_probe_starts_and_cancels_the_rest(monkeypatch):
    lookups, cancelled = [], []
    clock = FakeClock()
    _fake_modal_gpu(monkeypatch, {"probe_l40s": False, "probe_a100_80gb": True, "probe_h100": True},
                lookups, cancelled, clock=clock)
    b = ModalBench(probe_window_s=30, clock=clock)  # the clock moves only inside a probe
    assert b.select_gpu("hypergraph") == "A100-80GB"
    from talos.bench import GPU_USD_PER_SECOND
    # mutation: an unchanged probe cost hides a real GPU call from the compute cap; the
    # timed-out L40S probe never ran and must charge nothing
    assert b.cost_mark() == pytest.approx(3 * GPU_USD_PER_SECOND["A100-80GB"], abs=1e-9)
    # mutation: not cancelling the timed-out probe leaves an input queued that runs (and
    # bills) whenever an L40S frees up; probing past the first success pays for an H100 start
    assert cancelled == ["probe_l40s"] and lookups == ["probe_l40s", "probe_a100_80gb"]
    # the choice routes every later call to that GPU's function set
    r = b.evaluate(gpu_req())
    assert r.compile.output == "built on compile_hypergraph_a100_80gb"
    assert "score_batch_hypergraph_a100_80gb" in lookups
    # ...and prices it as that GPU: the probe's 3 s plus 3 one-nonce batches of 1 s each at
    # the A100-80GB rate
    assert b.cost_mark() == pytest.approx(6 * GPU_USD_PER_SECOND["A100-80GB"], abs=1e-9)


def test_select_gpu_reports_all_unavailable_after_trying_every_gpu(monkeypatch):
    lookups, cancelled = [], []
    _fake_modal_gpu(monkeypatch, {"probe_l40s": False, "probe_a100_80gb": False,
                                  "probe_h100": False}, lookups, cancelled)
    b = ModalBench(probe_window_s=30)
    with pytest.raises(BenchUnavailable) as ei:
        b.select_gpu("hypergraph")
    # mutation: stopping at the first miss never reaches the fallbacks
    assert cancelled == ["probe_l40s", "probe_a100_80gb", "probe_h100"]
    assert b.cost_mark() == 0.0  # nothing ran
    assert "L40S" in str(ei.value) and "H100" in str(ei.value) and "30" in str(ei.value)


def test_select_gpu_honours_a_frozen_choice_without_probing(monkeypatch):
    lookups, cancelled = [], []
    _fake_modal_gpu(monkeypatch, {"probe_l40s": True}, lookups, cancelled)
    b = ModalBench()
    # mutation: probing again on a resume can move the candidates to another GPU than the
    # one the baseline was measured on
    assert b.select_gpu("hypergraph", chosen="H100") == "H100"
    assert lookups == []
    b.evaluate(gpu_req())
    assert lookups[0] == "compile_hypergraph_h100"
    with pytest.raises(ValueError):
        b.select_gpu("hypergraph", chosen="T4")


def test_select_gpu_is_a_no_op_for_cpu_challenges_and_the_fake_bench(monkeypatch):
    lookups, cancelled = [], []
    _fake_modal_gpu(monkeypatch, {}, lookups, cancelled)
    b = ModalBench()
    assert b.select_gpu("knapsack") is None and lookups == []
    b.evaluate(req())
    assert lookups[0] == "compile_knapsack"  # CPU function names are unchanged
    fake = FakeBench(lambda c, f, ns: [1] * ns.count)
    assert fake.select_gpu("knapsack") is None
    # in-process, so no probe, but a GPU challenge still needs a GPU for its hardware class
    assert fake.select_gpu("hypergraph") == "L40S"
    assert fake.select_gpu("hypergraph", chosen="H100") == "H100"


def test_a_gpu_call_before_select_gpu_is_refused_loudly(monkeypatch):
    lookups, cancelled = [], []
    _fake_modal_gpu(monkeypatch, {}, lookups, cancelled)
    with pytest.raises(ValueError):  # mutation: defaulting to the first GPU hides the bug
        ModalBench().evaluate(gpu_req())
    assert lookups == []
