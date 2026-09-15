import sys
import types

import pytest

from talos.bench import BenchUnavailable, FakeBench, ModalBench, _redact
from talos.types import NonceSet


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


def test_fake_bench_scores_per_nonce_and_tracks_cost():
    def scores(challenge, files, ns):
        return [100 + n for n in ns.nonces()]
    fb = FakeBench(scores)
    c = fb.compile("knapsack", {"mod.rs": "x"})
    assert c.ok and c.artifact_id
    mark = fb.cost_mark()
    res = fb.score("knapsack", c.artifact_id, [NonceSet("t", "ab", 0, 3)], fuel=1)
    assert [r.quality for r in res] == [100, 101, 102]
    assert all(r.track == "t" for r in res)
    assert fb.cost_usd_since(mark) > 0  # mutation: not charging makes budget tests vacuous


def test_fake_bench_none_means_error():
    fb = FakeBench(lambda ch, files, ns: [None, 5])
    c = fb.compile("knapsack", {"mod.rs": "x"})
    res = fb.score("knapsack", c.artifact_id, [NonceSet("t", "ab", 0, 2)], fuel=1)
    # mutation: mapping a None quality to ok=True would feed phantom solutions to scoring
    assert not res[0].ok and res[0].error == "no_solution" and res[1].quality == 5


def test_fake_bench_compile_failure():
    fb = FakeBench(lambda ch, files, ns: [], compile_ok=lambda files: "BUG" not in files["mod.rs"])
    # mutation: ignoring compile_ok makes every compile-failure path in the loop untestable
    assert not fb.compile("knapsack", {"mod.rs": "BUG"}).ok


def test_fake_bench_rejects_a_short_scores_callback():
    fb = FakeBench(lambda ch, files, ns: [1, 2])
    c = fb.compile("knapsack", {"mod.rs": "x"})
    # mutation: zip() silently truncates, so a 2-quality callback would score a 3-nonce set
    # as 2 results and every count-based assertion downstream would quietly pass
    with pytest.raises(ValueError):
        fb.score("knapsack", c.artifact_id, [NonceSet("t", "ab", 0, 3)], fuel=1)


def test_redact_hides_a_rand_hash():
    # mutation: dropping the redaction leaks the seed into the pause message
    assert _redact("x " + "ab" * 32 + " y") == "x <hash> y"


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
    mod = _fake_modal(lambda nf: nf("app has no function score_nonce_knapsack"))
    monkeypatch.setitem(sys.modules, "modal", mod)
    clock = FakeClock()
    sleeps = []
    b = ModalBench(clock=clock, sleep=recording_sleep(clock, sleeps))
    with pytest.raises(BenchUnavailable) as ei:
        b.score("knapsack", "art", [NonceSet("t", "ab", 0, 1)], fuel=1)
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
        b.score("knapsack", "art", [NonceSet("t", "ab", 0, 1)], fuel=1)
    # mutation: converting every lookup error to "not deployed" pauses the run on a network
    # blip and prints the wrong advice
    assert "talos setup" not in str(ei.value)
    assert sleeps


def test_lookup_is_cached_after_a_successful_hydrate(monkeypatch):
    lookups = []

    class Fn:
        def hydrate(self):
            pass

        def starmap(self, args):
            return [{"track": t, "nonce": n, "ok": True, "quality": 1, "runtime_ms": 1,
                     "error": None} for (_a, t, _h, n, _f) in args]

    def from_name(app_name, name):
        lookups.append(name)
        return Fn()

    mod = types.ModuleType("modal")
    mod.exception = types.SimpleNamespace(NotFoundError=type("NotFoundError", (Exception,), {}))
    mod.Function = types.SimpleNamespace(from_name=from_name)
    monkeypatch.setitem(sys.modules, "modal", mod)

    b = ModalBench()
    for _ in range(3):
        b.score("knapsack", "art", [NonceSet("t", "ab", 0, 1)], fuel=1)
    # mutation: re-hydrating on every call adds a round trip per nonce batch
    assert lookups == ["score_nonce_knapsack"]
