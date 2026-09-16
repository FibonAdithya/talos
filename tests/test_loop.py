import json
import types

import pytest

from talos.budget import Budget, BudgetExhausted, Spend
from talos.bench import FakeBench
from talos.loop import Loop, Thresholds
from talos.providers import ProviderAuthError, ProviderRateLimited
from talos.providers.fake import FakeProvider
from talos.state import BaselineRecord, JobSpec, JobState, JobStore
from talos.types import NonceResult, NonceSet

HASH = "ab" * 32
TR = [NonceSet("t", HASH, 0, 4)]
HO = [NonceSet("t", HASH, 1_000_000, 4)]
TR2 = [NonceSet("t", HASH, 0, 4), NonceSet("u", HASH, 0, 4)]
HO2 = [NonceSet("t", HASH, 1_000_000, 4), NonceSet("u", HASH, 1_000_000, 4)]
BASE_FILES = {"mod.rs": "fn solve() { let k = 1; }\n"}


def spec(budget=None, track=None, two_tracks=False):
    tracks = ["t", "u"] if two_tracks else ["t"]
    return JobSpec(job_id="j", challenge="knapsack", direction="go", provider="fake", model="m",
                   mode="single-shot",
                   budget=budget or Budget(usd=None, hours=None, iterations=20, compute_usd=None),
                   rand_hash=HASH, tracks=tracks, training=TR2 if two_tracks else TR,
                   holdout=HO2 if two_tracks else HO, fuel=1, created_at=0.0,
                   monorepo_ref="r", challenge_id="c003", track=track)


def baseline(q=100, holdout_n=4, tracks=("t",)):
    tr = [NonceResult(t, n, True, q, 1) for t in tracks for n in range(4)]
    ho = [NonceResult(t, 1_000_000 + n, True, q, 1) for t in tracks for n in range(holdout_n)]
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


def make(tmp_path, script, scores=quality_from_files, budget=None, thresholds=None, holdout_n=4,
         track=None, two_tracks=False):
    store = JobStore(tmp_path)
    sp = spec(budget, track=track, two_tracks=two_tracks)
    store.write_spec(sp)
    st = JobState.fresh(Spend(started_at=0.0))
    st.baseline = baseline(holdout_n=holdout_n, tracks=("t", "u") if two_tracks else ("t",))
    st.status = "researching"
    st.best = None
    fb = FakeBench(scores)
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
    assert fb.holdout_runs == 1
    # mutation: stamping events with state.iteration labels iteration 1's events as iteration 0
    raw = (tmp_path / "timeline.jsonl").read_text()
    events = [json.loads(ln) for ln in raw.splitlines()]
    mid = [e for e in events if e["kind"] in ("hypothesis", "scored", "confirming", "won")]
    assert mid and all(e["iteration"] == 1 for e in mid)
    # spec §7.10: the rand_hash reaches neither the timeline nor a prompt
    assert HASH not in raw
    assert all(HASH not in system and HASH not in user for system, user in fp.calls)


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
    b = Budget(usd=None, hours=None, iterations=2, compute_usd=None)
    # iteration 2 edits the iteration-1 best (k=5), so its SEARCH text is `let k = 5;`
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(5), hyp("b"), edit(6, frm=5)], scores, b)
    st = loop.run()
    assert st.status == "exhausted" and st.stop_reason == "iterations"
    assert st.false_positives == [1, 2] and st.best.iteration == 2


def test_compile_fix_rounds_then_skip(tmp_path):
    # mutation: unlimited fix rounds never terminates on a stubborn error
    def swap(frm, to):
        return f"<<<<<<< SEARCH mod.rs\nlet k = {frm};\n=======\nlet k = {to};\n>>>>>>> REPLACE\n"

    # each fix APPLIES (so the loop would keep going) but still does not build
    script = [hyp("a"), swap(1, "BUG"), swap("BUG", "BUG2"), swap("BUG2", "BUG3"),
              swap("BUG3", "BUG4"), hyp("b"), edit(5)]

    def compile_ok(files):
        return "BUG" not in files["mod.rs"]

    loop, fp, fb, store = make(tmp_path, script)
    fb._compile_ok = compile_ok
    st = loop.run()
    assert st.status == "won"
    assert st.hypotheses[0]["outcome"] == "failed:compile"
    assert len(fb.calls) == 4 + 1  # 1 edit + 3 fixes + winning edit


def test_budget_stops_before_llm_call(tmp_path):
    # mutation: checking the budget only between iterations lets a half-iteration overspend
    # (the compile here must be refused, not run)
    b = Budget(usd=0.015, hours=None, iterations=None, compute_usd=None)  # one fake call = 0.01
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(2), hyp("b"), edit(3)], budget=b)
    st = loop.run()
    assert st.status == "exhausted" and st.stop_reason == "usd"
    assert len(fp.calls) == 2  # hypothesis + edit, then the compile's budget check refuses
    assert len(fb.calls) == 0  # the evaluate's budget check refuses before any bench call


def test_over_error_ceiling_is_failed_runtime(tmp_path):
    # spec §9: a candidate over the error ceiling is logged failed:runtime and never becomes best
    # mutation: dropping the ceiling check lets a half-crashing candidate become the best
    def scores(challenge, files, ns):
        import re
        k = int(re.search(r"let k = (\d+);", files["mod.rs"]).group(1))
        if k == 1:
            return [100 for _ in ns.nonces()]
        return [None if n % 2 else 100 + k for n in ns.nonces()]  # half the nonces error out

    b = Budget(usd=None, hours=None, iterations=1, compute_usd=None)
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
    b = Budget(usd=None, hours=None, iterations=6, compute_usd=None)
    loop, fp, fb, store = make(tmp_path, script, budget=b,
                               thresholds=Thresholds(recall=2, distill=3, reset=5))
    st = loop.run()
    assert "constants are not the answer" in st.tacit
    users = [u for _, u in fp.calls]
    assert any("do not repeat" in u for u in users)          # recall block appeared
    assert any("strategy_tag MUST be" in u for u in users)   # reset forced a tag
    # mutation: distilling on `>=` instead of `==` fires a distill call every stagnant iteration,
    # which eats the scripted hypothesis of the next iteration and derails it into failed:edit.
    # The call count alone does NOT catch this (each extra distill is offset by the provider call
    # a derailed iteration no longer makes), so pin the outcomes too.
    assert len(fp.calls) == 6 * 2 + 1  # 6 hypothesis + 6 edit + exactly one distill
    assert [h["outcome"] for h in st.hypotheses] == ["failed:score"] * 6


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
    # mutation: discarding before the TERMINAL early return deletes a finished job's winning
    # iteration directory (state.iteration lags by one when the kill lands mid-confirmation)
    loop2.state.iteration = 0
    store.iteration_dir(1).joinpath("mod.rs").write_text("winner")
    assert loop2.run().status == "won"
    assert store.iteration_dir(1).joinpath("mod.rs").read_text() == "winner"


def test_request_stop_cancels_at_safe_point(tmp_path):
    # mutation: ignoring the stop flag (or checking it only after the LLM call) burns a
    # provider call and reports a status other than "cancelled"
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(1), hyp("b"), edit(5)])
    loop.request_stop()
    st = loop.run()
    assert st.status == "cancelled" and len(fp.calls) == 0


def test_win_with_lower_delta_than_a_false_positive_best(tmp_path):
    # mutation: confirming a candidate without making it the best reports won with the wrong code
    # in state.best
    def scores(challenge, files, ns):
        import re
        k = int(re.search(r"let k = (\d+);", files["mod.rs"]).group(1))
        if ns.start >= 1_000_000:
            return [100 + (2 if k == 3 else 0) for _ in ns.nonces()]  # only k=3 holds up
        return [100 + k - 1 for _ in ns.nonces()]

    # k=5 wins training (+4%) but not held-out; k=3 wins less on training (+2%) yet confirms
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(5), hyp("b"), edit(3, frm=5)], scores)
    st = loop.run()
    assert st.status == "won" and st.best.iteration == 2
    assert "let k = 3;" in st.best.files["mod.rs"]
    assert st.hypotheses[1]["outcome"] == "won"
    assert st.confirmed == [2] and st.false_positives == [1]


def test_distill_auth_error_stops_the_run(tmp_path):
    # mutation: swallowing ProviderAuthError in _distill lets a dead key keep the loop running
    seen = {"n": 0}

    def script(system, user):
        if "distill one reusable lesson" in system:
            raise ProviderAuthError("key expired")
        seen["n"] += 1
        return hyp(f"h{seen['n']}") if seen["n"] % 2 else edit(1)

    loop, fp, fb, store = make(tmp_path, script)  # 3 non-improving iterations reach the distill
    st = loop.run()
    assert st.status == "failed"
    assert st.stop_reason.startswith("provider auth")


def test_holdout_scoring_error_is_a_false_positive(tmp_path):
    # mutation: letting ScoringError escape _confirm strands the job at "confirming"
    b = Budget(usd=None, hours=None, iterations=1, compute_usd=None)
    # the baseline holds 3 held-out nonces but the run scores 4 -> bundle_delta cannot pair them
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(5)], budget=b, holdout_n=3)
    st = loop.run()
    assert st.false_positives == [1] and st.confirmed == []
    assert st.status == "exhausted" and st.stop_reason == "iterations"


def test_compile_fix_stops_when_repair_applies_nothing(tmp_path):
    # mutation: recompiling after a repair that matched nothing burns a bench call per round on
    # byte-identical files
    broken = "<<<<<<< SEARCH mod.rs\nlet k = 1;\n=======\nlet k = BUG;\n>>>>>>> REPLACE\n"
    never_matches = "<<<<<<< SEARCH mod.rs\nlet zzz = 1;\n=======\nlet zzz = 2;\n>>>>>>> REPLACE\n"
    loop, fp, fb, store = make(tmp_path, [hyp("a"), broken, never_matches, hyp("b"), edit(5)])

    def compile_ok(files):
        return "BUG" not in files["mod.rs"]

    fb._compile_ok = compile_ok
    st = loop.run()
    assert st.status == "won"
    assert st.hypotheses[0]["outcome"] == "failed:compile"
    assert len(fb.calls) == 1 + 1  # the broken edit + the winning edit


def test_recall_starts_exactly_at_threshold(tmp_path):
    # mutation: `> recall` instead of `>= recall` delays the recall block by a whole iteration
    script = []
    for i in range(3):
        script += [hyp(f"h{i}"), edit(1)]
    b = Budget(usd=None, hours=None, iterations=3, compute_usd=None)
    loop, fp, fb, store = make(tmp_path, script, budget=b,
                               thresholds=Thresholds(recall=2, distill=99, reset=99))
    loop.run()
    users = [u for _, u in fp.calls]
    assert "do not repeat" not in users[2]  # 2nd hypothesis: runs_since_improvement == 1
    assert "do not repeat" in users[4]      # 3rd hypothesis: runs_since_improvement == 2


def test_rate_limit_wait_rechecks_the_budget(tmp_path):
    # mutation: not re-checking the budget after a rate-limit sleep ignores a stop request (or an
    # hours cap) for the whole retry storm
    seen = {"n": 0}

    def script(system, user):
        seen["n"] += 1
        if seen["n"] == 1:
            raise ProviderRateLimited("429")
        return hyp("a")

    loop, fp, fb, store = make(tmp_path, script)

    def stop_while_sleeping(seconds):
        loop.request_stop()

    loop.sleep = stop_while_sleeping
    st = loop.run()
    assert st.status == "cancelled"
    assert seen["n"] == 1  # the retry after the sleep never happened


def test_rejected_edit_paths_fail_the_iteration(tmp_path):
    # spec §9: an edit outside the algorithm files fails the iteration in both modes.
    # mutation: applying the in-scope blocks after a rejected one lets an out-of-scope edit go
    # unpunished — the response below would otherwise reach "won" on its second block.
    # mutation: dropping the edits_rejected event hides an LLM trying to write outside the
    # algorithm's own files
    stray = "<<<<<<< SEARCH Cargo.toml\nfoo\n=======\nbar\n>>>>>>> REPLACE\n" + edit(5)
    b = Budget(usd=None, hours=None, iterations=1, compute_usd=None)
    loop, fp, fb, store = make(tmp_path, [hyp("a"), stray], budget=b)
    st = loop.run()
    assert st.status == "exhausted" and st.best is None
    assert st.hypotheses[0]["outcome"] == "failed:edit"
    assert "Cargo.toml" in st.hypotheses[0]["error"]
    assert len(fb.calls) == 0  # the candidate never reached the bench
    events = [json.loads(ln) for ln in (tmp_path / "timeline.jsonl").read_text().splitlines()]
    rejected = [e for e in events if e["kind"] == "edits_rejected"]
    assert rejected and rejected[0]["paths"] == ["Cargo.toml"]


def test_baseline_is_budget_checked_before_the_first_bench_call(tmp_path):
    # mutation: an unchecked baseline spends the whole compute budget before the first check —
    # resolve_baseline compiles and scores both nonce sets in one evaluate of its own, and on a
    # cold cache that single call is the most expensive of the whole job
    b = Budget(usd=None, hours=None, iterations=20, compute_usd=0.0)
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(5)], budget=b)
    loop.state.baseline = None
    calls_before = len(fb.calls)
    mainnet = types.SimpleNamespace(
        top_algorithm=lambda ch: ("base", 1),
        fetch_algorithm_files=lambda ch, name: BASE_FILES,
        fetch_template=lambda ch: "pub fn solve_challenge(")
    with pytest.raises(BudgetExhausted) as ei:
        loop.measure_baseline(tmp_path / "cache", "cpu4-mem8192", mainnet=mainnet)
    assert ei.value.dimension == "compute_usd"
    assert len(fb.calls) == calls_before  # nothing was evaluated
    assert not (tmp_path / "cache").exists()


def test_baseline_compute_cost_is_charged_before_the_next_check(tmp_path):
    # The baseline is now a single evaluate, so "charged per call" and "charged once
    # resolve_baseline returns" cannot be told apart inside measure_baseline itself; what still
    # has to hold is that the charge is on spend, and on disk, before the NEXT budget check.
    # mutation: dropping the charge (or the save) in _BudgetedBench._metered lets the loop's
    # next check pass on a stale zero and burn an iteration past the compute cap
    b = Budget(usd=None, hours=None, iterations=20, compute_usd=0.015)  # one scored nonce = 0.01
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(5)], budget=b)
    loop.state.baseline = None
    mainnet = types.SimpleNamespace(
        top_algorithm=lambda ch: ("base", 1),
        fetch_algorithm_files=lambda ch, name: BASE_FILES,
        fetch_template=lambda ch: "pub fn solve_challenge(")
    loop.measure_baseline(tmp_path / "cache", "cpu4-mem8192", mainnet=mainnet)
    # the evaluate call charged before the next check
    assert loop.state.spend.compute_usd > 0.0
    assert json.loads((tmp_path / "state.json").read_text())["spend"]["compute_usd"] > 0.0
    st = loop.run()
    assert st.status == "exhausted" and st.stop_reason == "compute_usd"
    assert fp.calls == [] and len(fb.calls) == 1  # the baseline evaluate only


def test_baseline_and_best_dirs_are_written(tmp_path):
    # spec §5.4: runs/<job>/baseline/ and best/ hold the current sources on disk.
    # mutation: never writing them leaves two directories the spec promises permanently empty
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(5)])
    loop.state.baseline = None
    mainnet = types.SimpleNamespace(
        top_algorithm=lambda ch: ("base", 1),
        fetch_algorithm_files=lambda ch, name: BASE_FILES,
        fetch_template=lambda ch: "pub fn solve_challenge(")
    loop.measure_baseline(tmp_path / "cache", "cpu4-mem8192", mainnet=mainnet)
    assert (tmp_path / "baseline" / "mod.rs").read_text() == BASE_FILES["mod.rs"]
    results = json.loads((tmp_path / "baseline" / "results.json").read_text())
    assert len(results["training"]) == 4 and len(results["holdout"]) == 4
    st = loop.run()
    assert st.status == "won"
    assert "let k = 5;" in (tmp_path / "best" / "mod.rs").read_text()


NEVER_MATCHES = "<<<<<<< SEARCH mod.rs\nlet zzz = 1;\n=======\nlet zzz = 2;\n>>>>>>> REPLACE\n"
STRAY = "<<<<<<< SEARCH Cargo.toml\nfoo\n=======\nbar\n>>>>>>> REPLACE\n"


def test_repair_round_keeps_the_blocks_already_applied(tmp_path):
    # One good block and one miss: the good block is applied, the miss goes to a repair round,
    # and the repair misses again. search_replace's contract is "repair, then skip whatever still
    # doesn't match", so the iteration must go on with the block that DID apply.
    # mutation: replacing the outcome with the repair round's own outcome resets `applied` to 0
    # and fails the iteration as "no edit block applied" although one block was applied
    b = Budget(usd=None, hours=None, iterations=1, compute_usd=None)
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(5) + NEVER_MATCHES, NEVER_MATCHES],
                               budget=b)
    st = loop.run()
    assert st.status == "won" and "let k = 5;" in st.best.files["mod.rs"]
    assert len(fp.calls) == 3  # hypothesis, edit, exactly one repair round


def test_rejected_path_survives_a_repair_round(tmp_path):
    # spec §9: the out-of-scope block in the FIRST response must still fail the iteration when a
    # miss in the same response sends the loop through a repair round.
    # mutation: taking `rejected` from the repair round's outcome alone forgets the stray block
    b = Budget(usd=None, hours=None, iterations=1, compute_usd=None)
    loop, fp, fb, store = make(tmp_path, [hyp("a"), STRAY + edit(5) + NEVER_MATCHES, NEVER_MATCHES],
                               budget=b)
    st = loop.run()
    assert st.hypotheses[0]["outcome"] == "failed:edit"
    assert "Cargo.toml" in st.hypotheses[0]["error"]
    assert len(fb.calls) == 0  # the candidate never reached the bench


def test_compile_fix_rejects_out_of_scope_edits(tmp_path):
    # spec §9 applies to the compile-fix response too: a fix that touches a file outside the
    # algorithm fails the iteration instead of having its in-scope blocks applied.
    # mutation: ignoring `rejected` in the compile-fix loop lets the fix below reach "won"
    def swap(frm, to):
        return f"<<<<<<< SEARCH mod.rs\nlet k = {frm};\n=======\nlet k = {to};\n>>>>>>> REPLACE\n"

    b = Budget(usd=None, hours=None, iterations=1, compute_usd=None)
    loop, fp, fb, store = make(tmp_path, [hyp("a"), swap(1, "BUG"), STRAY + swap("BUG", 5)],
                               budget=b)
    fb._compile_ok = lambda files: "BUG" not in files["mod.rs"]
    st = loop.run()
    assert st.hypotheses[0]["outcome"] == "failed:edit"
    assert "Cargo.toml" in st.hypotheses[0]["error"]
    assert len(fb.calls) == 1  # the broken edit only; the fix never built
    events = [json.loads(ln) for ln in (tmp_path / "timeline.jsonl").read_text().splitlines()]
    assert [e["paths"] for e in events if e["kind"] == "edits_rejected"] == [["Cargo.toml"]]


def test_win_is_recorded_before_the_time_cap_bites(tmp_path):
    # mutation: checking the hours cap between the evaluate call and recording the win would
    # strand a proven winner as "exhausted" with best.holdout set but confirmed == []
    b = Budget(usd=None, hours=1.0, iterations=None, compute_usd=None)
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(5)], budget=b)
    loop.clock = lambda: 7200.0 if fb.calls else 0.0  # the cap passes during the evaluate
    st = loop.run()
    assert st.status == "won" and st.confirmed == [1]


def test_holdout_not_scored_is_a_false_positive(tmp_path):
    # mutation: treating holdout=None as "won" would confirm a candidate whose held-out run
    # timed out inside a C3 job
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(5)],
                               budget=Budget(usd=None, hours=None, iterations=1,
                                             compute_usd=None))
    real = fb.evaluate

    def evaluate(request):
        r = real(request)
        r.holdout, r.holdout_reason = None, "timeout"
        return r
    fb.evaluate = evaluate
    st = loop.run()
    assert st.false_positives == [1] and st.confirmed == [] and st.status == "exhausted"
    events = [json.loads(ln) for ln in (tmp_path / "timeline.jsonl").read_text().splitlines()]
    fp_ev = [e for e in events if e["kind"] == "false_positive"]
    assert fp_ev and "timeout" in fp_ev[0]["error"]


def test_pending_job_is_written_before_evaluate_and_cleared_after(tmp_path):
    # mutation: writing pending_job after evaluate returns means a kill during a 20-minute C3
    # job leaves nothing to reattach to; never clearing it makes every later resume "reattach"
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(5)],
                               budget=Budget(usd=None, hours=None, iterations=1,
                                             compute_usd=None))
    seen = {}
    real = fb.evaluate

    def evaluate(request):
        seen["pending"] = json.loads((tmp_path / "state.json").read_text())["pending_job"]
        return real(request)
    fb.evaluate = evaluate
    st = loop.run()
    assert seen["pending"]["purpose"] == 1
    assert seen["pending"]["files"] == {"mod.rs": "fn solve() { let k = 5; }\n"}
    assert seen["pending"]["hypothesis"]["title"] == "a"
    assert st.pending_job is None
    assert json.loads((tmp_path / "state.json").read_text())["pending_job"] is None


def test_pending_job_files_follow_a_compile_fix(tmp_path):
    # mutation: writing pending_job only in iterate() (and not in _bench_evaluate) leaves the
    # pre-fix sources in the record, so a resume during the fix round's evaluate reattaches to,
    # or rebuilds, a request for code the loop has already thrown away
    broken = "<<<<<<< SEARCH mod.rs\nlet k = 1;\n=======\nlet k = BUG;\n>>>>>>> REPLACE\n"
    fix = "<<<<<<< SEARCH mod.rs\nlet k = BUG;\n=======\nlet k = 5;\n>>>>>>> REPLACE\n"
    loop, fp, fb, store = make(tmp_path, [hyp("a"), broken, fix])
    fb._compile_ok = lambda files: "BUG" not in files["mod.rs"]
    seen = []
    real = fb.evaluate

    def evaluate(request):
        seen.append(json.loads((tmp_path / "state.json").read_text())["pending_job"]["files"])
        return real(request)
    fb.evaluate = evaluate
    st = loop.run()
    assert st.status == "won"
    assert seen == [{"mod.rs": "fn solve() { let k = BUG; }\n"},
                    {"mod.rs": "fn solve() { let k = 5; }\n"}]


def test_resume_with_a_pending_job_re_evaluates_without_a_new_hypothesis(tmp_path):
    # mutation: ignoring pending_job on resume proposes a fresh hypothesis (fp.calls != [])
    # and abandons the job C3 is still billing for
    store = JobStore(tmp_path)
    store.write_spec(spec(Budget(usd=None, hours=None, iterations=1, compute_usd=None)))
    st = JobState.fresh(Spend(started_at=0.0))
    st.baseline = baseline()
    st.status = "researching"
    files = {"mod.rs": "fn solve() { let k = 5; }\n"}
    st.pending_job = {"purpose": 1, "files": files,
                      "hypothesis": {"title": "a", "description": "d",
                                     "strategy_tag": "local_search"}}
    store.save(st)
    fb = FakeBench(quality_from_files)
    fp = FakeProvider([])  # an empty script raises if the loop asks for a completion
    loop = Loop(store.read_spec(), store.load(), store, fp, fb, template_rs="x",
                clock=lambda: 0.0, sleep=lambda s: None)
    result = loop.run()
    assert fp.calls == [] and len(fb.calls) == 1 and fb.calls[0].files == files
    assert result.status == "won" and result.confirmed == [1] and result.iteration == 1
    assert result.hypotheses[0]["title"] == "a" and result.hypotheses[0]["outcome"] == "won"
    assert result.strategy_counts == {"local_search": 1}  # counted once, not on both runs
    assert result.pending_job is None


def test_resume_does_not_discard_the_pending_iteration_dir(tmp_path):
    # mutation: _discard_incomplete_iteration wiping iterations/0001 on a pending resume deletes
    # the hypothesis.json the package later reads
    store = JobStore(tmp_path)
    store.write_spec(spec(Budget(usd=None, hours=None, iterations=1, compute_usd=None)))
    st = JobState.fresh(Spend(started_at=0.0))
    st.baseline = baseline()
    st.status = "researching"
    st.pending_job = {"purpose": 1, "files": {"mod.rs": "fn solve() { let k = 5; }\n"},
                      "hypothesis": {"title": "a", "description": "d",
                                     "strategy_tag": "local_search"}}
    store.save(st)
    (store.iteration_dir(1) / "hypothesis.json").write_text("{}")
    loop = Loop(store.read_spec(), store.load(), store, FakeProvider([]),
                FakeBench(quality_from_files), template_rs="x",
                clock=lambda: 0.0, sleep=lambda s: None)
    loop.run()
    assert (tmp_path / "iterations" / "0001" / "hypothesis.json").exists()


def test_stale_baseline_pending_job_is_cleared(tmp_path):
    # mutation: dropping run()'s `purpose == "baseline"` branch leaves the stale record on disk
    # until iterate() happens to overwrite it — so a run that exhausts its budget (or is stopped)
    # before the first iteration keeps a baseline job_id a C3 backend would reattach to. The
    # end-state assertions alone cannot see that, so snapshot the record before iterating.
    # mutation: a leftover {"purpose": "baseline"} record must also never be taken for a pending
    # iteration (an int purpose), which would send _resume_pending looking for a hypothesis
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(5)],
                               budget=Budget(usd=None, hours=None, iterations=1,
                                             compute_usd=None))
    loop.state.pending_job = {"purpose": "baseline", "job_id": "job_old"}
    seen = {}
    real = fp.complete

    def complete(system, user):
        seen.setdefault("pending",
                        json.loads((tmp_path / "state.json").read_text())["pending_job"])
        return real(system, user)
    fp.complete = complete
    st = loop.run()
    assert seen["pending"] is None  # cleared before the first iteration, not by it
    assert st.status == "won" and st.pending_job is None


def test_measure_baseline_keeps_a_stored_baseline_job_record(tmp_path):
    # mutation: overwriting pending_job with a fresh {"purpose": "baseline"} drops the job_id a
    # killed baseline run stored, so the resume submits (and pays for) a second baseline job
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(5)])
    loop.state.baseline = None
    loop.state.pending_job = {"purpose": "baseline", "job_id": "job_b", "request_hash": "h"}
    seen = {}
    real = fb.evaluate

    def evaluate(request):
        seen["pending"] = dict(loop.state.pending_job)
        return real(request)
    fb.evaluate = evaluate
    mainnet = types.SimpleNamespace(
        top_algorithm=lambda ch: ("base", 1),
        fetch_algorithm_files=lambda ch, name: BASE_FILES,
        fetch_template=lambda ch: "pub fn solve_challenge(")
    loop.measure_baseline(tmp_path / "cache", "cpu4-mem8192", mainnet=mainnet)
    assert seen["pending"]["job_id"] == "job_b" and loop.state.pending_job is None


def test_bench_cancelled_stops_the_run_as_cancelled(tmp_path):
    # mutation: letting BenchCancelled escape run() tracebacks out of the CLI instead of
    # packaging the best so far
    from talos.bench import BenchCancelled
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(5)])

    def evaluate(request):
        raise BenchCancelled("job_x")
    fb.evaluate = evaluate
    st = loop.run()
    assert st.status == "cancelled" and "job_x" in st.stop_reason


def test_focused_request_narrows_training_and_guards_holdout(tmp_path):
    # mutation: forgetting the guard sends only t's held-out set; sending the whole baseline
    # makes the in-job holdout decision raise a nonce mismatch on every iteration
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(5)], track="t", two_tracks=True)
    loop.run()
    req = fb.calls[0]
    assert req.training == [TR2[0]]
    assert req.holdout == [HO2[0], TR2[1]]
    assert {(r.track, r.nonce) for r in req.baseline_training} == {("t", n) for n in range(4)}


def test_focused_win_is_confirmed_on_the_track_and_the_guard(tmp_path):
    # mutation: comparing the guard rows against the baseline's held-out rows (wrong nonces)
    # raises a ScoringError and records a false positive instead of the win
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(5)], track="t", two_tracks=True)
    st = loop.run()
    assert st.status == "won" and st.confirmed == [1]
    assert {r.track for r in st.best.holdout} == {"t", "u"}


def test_focused_guard_regression_is_a_false_positive(tmp_path):
    # mutation: confirming on the focus track alone declares a win that broke track u
    def scores(challenge, files, ns):
        import re
        k = int(re.search(r"let k = (\d+);", files["mod.rs"]).group(1))
        if ns.track == "u" and k != 1:
            return [90 for _ in ns.nonces()]  # the edit regresses the other track
        return [100 + k - 1 for _ in ns.nonces()]
    b = Budget(usd=None, hours=None, iterations=1, compute_usd=None)
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(5)], scores, b, track="t",
                               two_tracks=True)
    st = loop.run()
    assert st.status == "exhausted" and st.false_positives == [1] and st.confirmed == []
    events = [json.loads(ln) for ln in (tmp_path / "timeline.jsonl").read_text().splitlines()]
    fp_event = next(e for e in events if e["kind"] == "false_positive")
    assert fp_event["holdout"]["worst_rel_delta"] < 0


def test_unfocused_request_is_unchanged(tmp_path):
    # mutation: the focus path leaking into unfocused jobs changes every existing request
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(5)], two_tracks=True)
    loop.run()
    req = fb.calls[0]
    assert req.training == TR2 and req.holdout == HO2
    assert len(req.baseline_training) == 8


def test_focused_context_names_track_and_guards(tmp_path):
    # mutation: dropping either field leaves the model aiming at every track
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(5)], track="t", two_tracks=True)
    ctx = loop._context()
    assert ctx.track == "t" and ctx.guard_tracks == ["u"]
    loop2, *_ = make(tmp_path / "b", [hyp("a"), edit(5)], two_tracks=True)
    ctx2 = loop2._context()
    assert ctx2.track is None and ctx2.guard_tracks == []


DEAD_WARNING = ("warning: function `polish` is never used\n"
                "   --> tig-algorithms/src/knapsack/talos_cand/mod.rs:3:4\n")


def add_polish(k, call):
    body = f"let k = {k};" + (" polish();" if call else "")
    return ("<<<<<<< SEARCH mod.rs\nfn solve() { let k = 1; }\n=======\n"
            f"fn polish() {{}}\nfn solve() {{ {body} }}\n>>>>>>> REPLACE\n")


def dead_if_uncalled(files):
    src = files["mod.rs"]
    return DEAD_WARNING if "fn polish" in src and "polish();" not in src else "ok"


def test_dead_new_code_gets_a_fix_round_before_scoring(tmp_path):
    # mutation: treating a dead_code result like a scored one records failed:score for a
    # candidate whose change was never on the solve path
    wire = "<<<<<<< SEARCH mod.rs\nlet k = 2; }\n=======\nlet k = 2; polish(); }\n>>>>>>> REPLACE\n"
    b = Budget(usd=None, hours=None, iterations=1, compute_usd=None)
    loop, fp, fb, store = make(tmp_path, [hyp("a"), add_polish(2, call=False), wire], budget=b)
    fb._compile_output = dead_if_uncalled
    st = loop.run()
    assert st.status == "won"
    assert len(fb.calls) == 2
    events = [json.loads(ln) for ln in (tmp_path / "timeline.jsonl").read_text().splitlines()]
    dead = [e for e in events if e["kind"] == "dead_code"]
    assert [e["names"] for e in dead] == [["mod.rs: polish"]]
    # the request tells the bench which functions the edited code already had
    assert fb.calls[0].prior_functions == {"mod.rs": ["solve"]}


def test_dead_new_code_after_the_fix_rounds_fails_the_iteration(tmp_path):
    # mutation: falling through to scoring after the rounds run out scores the no-op after all
    still_dead = "<<<<<<< SEARCH mod.rs\nlet k = 2;\n=======\nlet k = 3;\n>>>>>>> REPLACE\n"
    b = Budget(usd=None, hours=None, iterations=1, compute_usd=None)
    loop, fp, fb, store = make(tmp_path, [hyp("a"), add_polish(2, call=False), still_dead],
                               budget=b, thresholds=Thresholds(compile_fix_rounds=1))
    fb._compile_output = dead_if_uncalled
    st = loop.run()
    assert st.hypotheses[0]["outcome"] == "failed:dead_code"
    assert "polish" in st.hypotheses[0]["error"]
    assert len(fb.calls) == 2


def test_a_scored_record_carries_the_numbers_the_next_prompt_needs(tmp_path):
    # mutation: recording only the outcome leaves the recall block with titles alone
    b = Budget(usd=None, hours=None, iterations=1, compute_usd=None)
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(2)], budget=b,
                               scores=lambda ch, files, ns: [100] * ns.count)
    fb._runtime_ms = 3
    st = loop.run()
    rec = st.hypotheses[0]
    assert rec["outcome"] == "failed:score"
    assert rec["mean_rel_delta"] == pytest.approx(0.0) and rec["worst_track"] == "t"
    assert rec["worst_rel_delta"] == pytest.approx(0.0)
    assert rec["runtime_ratio"] == pytest.approx(3.0)  # baseline rows run 1 ms


def test_candidate_timeouts_follow_the_baseline_runtime_per_track(tmp_path):
    # iteration 2 of run 20260916-095103 ran 395 s per nonce against a 28 s baseline under a
    # flat 600 s timeout and ate a third of the hours budget. mutation: a flat timeout, or a
    # ratio without the floor, sends 600 or 3 seconds for a 1 ms baseline track
    b = Budget(usd=None, hours=None, iterations=1, compute_usd=None)
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(2)], budget=b, two_tracks=True)
    loop.state.baseline.training[0].runtime_ms = 100_000  # track "t"; track "u" stays at 1 ms
    loop.run()
    assert fb.calls[0].timeouts == {"t": 300, "u": 60}
    # the ceiling never exceeds the flat timeout the baseline itself ran under
    loop2, fp2, fb2, store2 = make(tmp_path / "b", [hyp("a"), edit(2)], budget=b)
    loop2.state.baseline.training[0].runtime_ms = 10_000_000
    loop2.run()
    assert fb2.calls[0].timeouts == {"t": 600}
    # a zero ceiling disables the guard
    loop3, fp3, fb3, store3 = make(tmp_path / "c", [hyp("a"), edit(2)], budget=b,
                                   thresholds=Thresholds(runtime_ceiling=0.0))
    loop3.run()
    assert fb3.calls[0].timeouts is None
