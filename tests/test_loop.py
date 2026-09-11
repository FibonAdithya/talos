from talos.budget import Budget, Spend
from talos.bench import FakeBench
from talos.loop import Loop, Thresholds
from talos.providers.fake import FakeProvider
from talos.state import BaselineRecord, JobSpec, JobState, JobStore
from talos.types import NonceResult, NonceSet

HASH = "ab" * 32
TR = [NonceSet("t", HASH, 0, 4)]
HO = [NonceSet("t", HASH, 1_000_000, 4)]
BASE_FILES = {"mod.rs": "fn solve() { let k = 1; }\n"}


def spec(budget=None):
    return JobSpec(job_id="j", challenge="knapsack", direction="go", provider="fake", model="m",
                   mode="single-shot",
                   budget=budget or Budget(usd=None, hours=None, iterations=20, modal_usd=None),
                   rand_hash=HASH, tracks=["t"], training=TR, holdout=HO, fuel=1, created_at=0.0,
                   monorepo_ref="r", challenge_id="c003")


def baseline(q=100):
    tr = [NonceResult("t", n, True, q, 1) for n in range(4)]
    ho = [NonceResult("t", 1_000_000 + n, True, q, 1) for n in range(4)]
    return BaselineRecord(name="base", adoption=1, artifact_id="base-art", files=BASE_FILES,
                          training=tr, holdout=ho)


def hyp(title, tag="local_search"):
    return f'{{"title": "{title}", "description": "d", "strategy_tag": "{tag}"}}'


def edit(k, frm=1):
    """An edit block turning `let k = {frm};` into `let k = {k};`. The SEARCH text must match the
    files the loop is currently editing (the best so far), not always the baseline."""
    return f"<<<<<<< SEARCH mod.rs\nlet k = {frm};\n=======\nlet k = {k};\n>>>>>>> REPLACE\n"


def quality_from_files(challenge, files, ns):
    # quality = 100 + k on every nonce; k parsed from the code. "let k = 7" -> 107
    import re
    k = int(re.search(r"let k = (\d+);", files["mod.rs"]).group(1))
    return [100 + k - 1 for _ in ns.nonces()]  # k=1 -> 100 (baseline parity)


def make(tmp_path, script, scores=quality_from_files, budget=None, thresholds=None):
    store = JobStore(tmp_path)
    sp = spec(budget)
    store.write_spec(sp)
    st = JobState.fresh(Spend(started_at=0.0))
    st.baseline = baseline()
    st.status = "researching"
    st.best = None
    fb = FakeBench(scores)
    fb.compile("knapsack", BASE_FILES)  # register the baseline files under an artifact id
    fp = FakeProvider(script)
    loop = Loop(sp, st, store, fp, fb, template_rs="pub fn solve_challenge(",
                clock=lambda: 0.0, sleep=lambda s: None, thresholds=thresholds or Thresholds())
    return loop, fp, fb, store


def test_win_requires_training_then_holdout_beat(tmp_path):
    # mutation: skipping the held-out confirmation declares a win after training alone
    loop, fp, fb, store = make(tmp_path, [hyp("bump k"), edit(5)])
    st = loop.run()
    assert st.status == "won" and st.best.iteration == 1
    assert st.best.holdout is not None and st.confirmed == [1]
    assert fb.score_calls == 1 + 1  # training, then held-out


def test_false_positive_returns_to_research(tmp_path):
    # training says win, held-out says no -> keep researching, record false positive
    # mutation: confirming on held-out but still setting status "won" (or dropping the
    # false_positives record) loses the distinction between a real win and a fluke
    def scores(challenge, files, ns):
        import re
        k = int(re.search(r"let k = (\d+);", files["mod.rs"]).group(1))
        if ns.start >= 1_000_000:
            return [100 for _ in ns.nonces()]  # held-out never improves
        return [100 + k - 1 for _ in ns.nonces()]
    b = Budget(usd=None, hours=None, iterations=2, modal_usd=None)
    # iteration 2 edits the iteration-1 best (k=5), so its SEARCH text is `let k = 5;`
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(5), hyp("b"), edit(6, frm=5)], scores, b)
    st = loop.run()
    assert st.status == "exhausted" and st.stop_reason == "iterations"
    assert st.false_positives == [1, 2] and st.best.iteration == 2


def test_compile_fix_rounds_then_skip(tmp_path):
    # mutation: unlimited fix rounds never terminates on a stubborn error
    bad = "<<<<<<< SEARCH mod.rs\nlet k = 1;\n=======\nlet k = BUG;\n>>>>>>> REPLACE\n"
    script = [hyp("a"), bad, bad, bad, bad, hyp("b"), edit(5)]

    def compile_ok(files):
        return "BUG" not in files["mod.rs"]

    loop, fp, fb, store = make(tmp_path, script)
    fb._compile_ok = compile_ok
    st = loop.run()
    assert st.status == "won"
    assert st.hypotheses[0]["outcome"] == "failed:compile"
    assert fb.compile_calls == 1 + 4 + 1  # baseline reg + 1 edit + 3 fixes + winning edit


def test_budget_stops_before_llm_call(tmp_path):
    # mutation: checking the budget only between iterations lets a half-iteration overspend
    # (the compile here must be refused, not run)
    b = Budget(usd=0.015, hours=None, iterations=None, modal_usd=None)  # one fake call = 0.01
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(2), hyp("b"), edit(3)], budget=b)
    st = loop.run()
    assert st.status == "exhausted" and st.stop_reason == "usd"
    assert len(fp.calls) == 2  # hypothesis + edit, then the compile's budget check refuses
    assert fb.compile_calls == 1  # only the baseline registration in make(); no candidate compile


def test_over_error_ceiling_is_failed_runtime(tmp_path):
    # spec §9: a candidate over the error ceiling is logged failed:runtime and never becomes best
    # mutation: dropping the ceiling check lets a half-crashing candidate become the best
    def scores(challenge, files, ns):
        import re
        k = int(re.search(r"let k = (\d+);", files["mod.rs"]).group(1))
        if k == 1:
            return [100 for _ in ns.nonces()]
        return [None if n % 2 else 100 + k for n in ns.nonces()]  # half the nonces error out

    b = Budget(usd=None, hours=None, iterations=1, modal_usd=None)
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(9)], scores, b)
    st = loop.run()
    assert st.hypotheses[0]["outcome"] == "failed:runtime" and st.best is None
    assert st.status == "exhausted"


def test_stagnation_recall_distill_reset(tmp_path):
    # non-improving edits (k stays 1) drive stagnation through recall -> distill -> reset
    # mutation: never resetting runs_since_improvement thresholds (or anchoring the recall list
    # on the wrong iteration) drops the recall block, the distillation and the forced tag
    script = []
    for i in range(6):
        script += [hyp(f"h{i}"), edit(1)]
    script.insert(6, "LESSON: constants are not the answer.")  # distill call after 3rd failure
    b = Budget(usd=None, hours=None, iterations=6, modal_usd=None)
    loop, fp, fb, store = make(tmp_path, script, budget=b,
                               thresholds=Thresholds(recall=2, distill=3, reset=5))
    st = loop.run()
    assert "constants are not the answer" in st.tacit
    users = [u for _, u in fp.calls]
    assert any("do not repeat" in u for u in users)          # recall block appeared
    assert any("strategy_tag MUST be" in u for u in users)   # reset forced a tag


def test_resume_discards_incomplete_iteration(tmp_path):
    # mutation: resuming without discarding the half-written iteration dir would leave the
    # stale hypothesis.json in place and re-number the next iteration
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(5)])
    loop.state.iteration = 3
    store.iteration_dir(4).joinpath("hypothesis.json").write_text("{}")  # started, never finished
    store.iteration_dir(4).joinpath("stale.rs").write_text("half-written")
    store.save(loop.state)
    loop2 = Loop(store.read_spec(), store.load(), store, fp, fb, template_rs="x",
                 clock=lambda: 0.0, sleep=lambda s: None)
    st = loop2.run()
    assert st.status == "won" and st.best.iteration == 4
    assert not store.iteration_dir(4).joinpath("hypothesis.json").read_text() == "{}"
    assert not store.iteration_dir(4).joinpath("stale.rs").exists()  # the dir was discarded whole


def test_request_stop_cancels_at_safe_point(tmp_path):
    # mutation: ignoring the stop flag (or checking it only after the LLM call) burns a
    # provider call and reports a status other than "cancelled"
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(1), hyp("b"), edit(5)])
    loop.request_stop()
    st = loop.run()
    assert st.status == "cancelled" and len(fp.calls) == 0
