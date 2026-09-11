from talos.bench import FakeBench
from talos.types import NonceSet


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
