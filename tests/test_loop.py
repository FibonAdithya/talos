import json
import types

import pytest

from talos.budget import Budget, BudgetExhausted, Spend
from talos.bench import FakeBench
from talos.loop import Loop, Thresholds
from talos.providers import ProviderAuthError, ProviderRateLimited
from talos.providers.fake import FakeProvider
from talos.state import BaselineRecord, Candidate, JobSpec, JobState, JobStore
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


def baseline(q=100, holdout_n=4):
    tr = [NonceResult("t", n, True, q, 1) for n in range(4)]
    ho = [NonceResult("t", 1_000_000 + n, True, q, 1) for n in range(holdout_n)]
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


def make(tmp_path, script, scores=quality_from_files, budget=None, thresholds=None, holdout_n=4):
    store = JobStore(tmp_path)
    sp = spec(budget)
    store.write_spec(sp)
    st = JobState.fresh(Spend(started_at=0.0))
    st.baseline = baseline(holdout_n=holdout_n)
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
    b = Budget(usd=None, hours=None, iterations=2, modal_usd=None)
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


def test_resume_finishes_a_pending_confirmation(tmp_path):
    # mutation: forcing status back to researching on resume drops a pending held-out confirmation
    store = JobStore(tmp_path)
    store.write_spec(spec())
    fb = FakeBench(quality_from_files)
    files = {"mod.rs": "fn solve() { let k = 5; }\n"}
    art = fb.compile("knapsack", files).artifact_id
    st = JobState.fresh(Spend(started_at=0.0))
    st.baseline = baseline()
    st.status = "confirming"          # killed between the training score and the held-out run
    st.iteration = 1
    st.spend.iterations = 1
    st.best = Candidate(iteration=1, files=files, artifact_id=art,
                        training=[NonceResult("t", n, True, 104, 1) for n in range(4)],
                        delta={"tracks": [], "mean_rel_delta": 0.04, "worst_rel_delta": 0.04,
                               "error_rate": 0.0},
                        hypothesis={"title": "a", "description": "d",
                                    "strategy_tag": "local_search"})
    st.hypotheses = [{"iteration": 1, "against": 0, "title": "a", "description": "d",
                      "strategy_tag": "local_search", "outcome": "improved"}]
    store.save(st)
    fp = FakeProvider([])  # an empty script raises if the loop asks for a completion
    loop = Loop(store.read_spec(), store.load(), store, fp, fb, template_rs="x",
                clock=lambda: 0.0, sleep=lambda s: None)
    result = loop.run()
    assert result.status == "won" and result.confirmed == [1]
    assert result.hypotheses[0]["outcome"] == "won"
    assert fb.score_calls == 1  # the held-out run only; training is not repeated
    assert fp.calls == []       # no new hypothesis was proposed


def test_holdout_scoring_error_is_a_false_positive(tmp_path):
    # mutation: letting ScoringError escape _confirm strands the job at "confirming"
    b = Budget(usd=None, hours=None, iterations=1, modal_usd=None)
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
    assert fb.compile_calls == 1 + 1 + 1  # baseline reg + the broken edit + the winning edit


def test_recall_starts_exactly_at_threshold(tmp_path):
    # mutation: `> recall` instead of `>= recall` delays the recall block by a whole iteration
    script = []
    for i in range(3):
        script += [hyp(f"h{i}"), edit(1)]
    b = Budget(usd=None, hours=None, iterations=3, modal_usd=None)
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


def test_confirm_still_honours_the_time_cap(tmp_path):
    # the held-out confirmation is exempt from the ITERATIONS cap only (its iteration is already
    # counted); every spend dimension still bites.
    # mutation: exempting confirmation from every budget dimension lets it run past a hard cap
    b = Budget(usd=None, hours=1.0, iterations=None, modal_usd=None)
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(5)], budget=b)
    loop.clock = lambda: 7200.0 if fb.score_calls >= 1 else 0.0  # cap passes after training
    st = loop.run()
    assert st.status == "exhausted" and st.stop_reason == "hours"
    assert st.best is not None and st.best.holdout is None  # refused, not waved through


def test_rejected_edit_paths_fail_the_iteration(tmp_path):
    # spec §9: an edit outside the algorithm files fails the iteration in both modes.
    # mutation: applying the in-scope blocks after a rejected one lets an out-of-scope edit go
    # unpunished — the response below would otherwise reach "won" on its second block.
    # mutation: dropping the edits_rejected event hides an LLM trying to write outside the
    # algorithm's own files
    stray = "<<<<<<< SEARCH Cargo.toml\nfoo\n=======\nbar\n>>>>>>> REPLACE\n" + edit(5)
    b = Budget(usd=None, hours=None, iterations=1, modal_usd=None)
    loop, fp, fb, store = make(tmp_path, [hyp("a"), stray], budget=b)
    st = loop.run()
    assert st.status == "exhausted" and st.best is None
    assert st.hypotheses[0]["outcome"] == "failed:edit"
    assert "Cargo.toml" in st.hypotheses[0]["error"]
    assert fb.compile_calls == 1  # baseline registration only: the candidate never reached compile
    events = [json.loads(ln) for ln in (tmp_path / "timeline.jsonl").read_text().splitlines()]
    rejected = [e for e in events if e["kind"] == "edits_rejected"]
    assert rejected and rejected[0]["paths"] == ["Cargo.toml"]


def test_baseline_is_budget_checked_before_the_first_modal_call(tmp_path):
    # mutation: an unchecked baseline spends the whole Modal budget before the first check —
    # resolve_baseline makes one compile and two scoring runs of its own
    b = Budget(usd=None, hours=None, iterations=20, modal_usd=0.0)
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(5)], budget=b)
    loop.state.baseline = None
    calls_before = fb.compile_calls
    mainnet = types.SimpleNamespace(
        top_algorithm=lambda ch: ("base", 1),
        fetch_algorithm_files=lambda ch, name: BASE_FILES,
        fetch_template=lambda ch: "pub fn solve_challenge(")
    with pytest.raises(BudgetExhausted) as ei:
        loop.measure_baseline(tmp_path / "cache", "cpu4-mem8192", mainnet=mainnet)
    assert ei.value.dimension == "modal_usd"
    assert fb.compile_calls == calls_before  # nothing was compiled
    assert not (tmp_path / "cache").exists()


def test_baseline_modal_cost_is_charged_per_call(tmp_path):
    # mutation: accounting for the baseline's Modal cost only after resolve_baseline returns
    # lets a cold baseline run past the cap and charges it when it is too late to matter
    b = Budget(usd=None, hours=None, iterations=20, modal_usd=0.015)  # one scored nonce = 0.01
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(5)], budget=b)
    loop.state.baseline = None
    mainnet = types.SimpleNamespace(
        top_algorithm=lambda ch: ("base", 1),
        fetch_algorithm_files=lambda ch, name: BASE_FILES,
        fetch_template=lambda ch: "pub fn solve_challenge(")
    with pytest.raises(BudgetExhausted):
        loop.measure_baseline(tmp_path / "cache", "cpu4-mem8192", mainnet=mainnet)
    # the compile and the training scoring run charged before the next call was refused
    assert loop.state.spend.modal_usd > 0.0
    assert json.loads((tmp_path / "state.json").read_text())["spend"]["modal_usd"] > 0.0


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
    b = Budget(usd=None, hours=None, iterations=1, modal_usd=None)
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(5) + NEVER_MATCHES, NEVER_MATCHES],
                               budget=b)
    st = loop.run()
    assert st.status == "won" and "let k = 5;" in st.best.files["mod.rs"]
    assert len(fp.calls) == 3  # hypothesis, edit, exactly one repair round


def test_rejected_path_survives_a_repair_round(tmp_path):
    # spec §9: the out-of-scope block in the FIRST response must still fail the iteration when a
    # miss in the same response sends the loop through a repair round.
    # mutation: taking `rejected` from the repair round's outcome alone forgets the stray block
    b = Budget(usd=None, hours=None, iterations=1, modal_usd=None)
    loop, fp, fb, store = make(tmp_path, [hyp("a"), STRAY + edit(5) + NEVER_MATCHES, NEVER_MATCHES],
                               budget=b)
    st = loop.run()
    assert st.hypotheses[0]["outcome"] == "failed:edit"
    assert "Cargo.toml" in st.hypotheses[0]["error"]
    assert fb.compile_calls == 1  # baseline registration only


def test_compile_fix_rejects_out_of_scope_edits(tmp_path):
    # spec §9 applies to the compile-fix response too: a fix that touches a file outside the
    # algorithm fails the iteration instead of having its in-scope blocks applied.
    # mutation: ignoring `rejected` in the compile-fix loop lets the fix below reach "won"
    def swap(frm, to):
        return f"<<<<<<< SEARCH mod.rs\nlet k = {frm};\n=======\nlet k = {to};\n>>>>>>> REPLACE\n"

    b = Budget(usd=None, hours=None, iterations=1, modal_usd=None)
    loop, fp, fb, store = make(tmp_path, [hyp("a"), swap(1, "BUG"), STRAY + swap("BUG", 5)],
                               budget=b)
    fb._compile_ok = lambda files: "BUG" not in files["mod.rs"]
    st = loop.run()
    assert st.hypotheses[0]["outcome"] == "failed:edit"
    assert "Cargo.toml" in st.hypotheses[0]["error"]
    assert fb.compile_calls == 1 + 1  # baseline registration + the broken edit; the fix never built
    events = [json.loads(ln) for ln in (tmp_path / "timeline.jsonl").read_text().splitlines()]
    assert [e["paths"] for e in events if e["kind"] == "edits_rejected"] == [["Cargo.toml"]]


def _pending_confirmation(tmp_path, status, training_quality=104):
    store = JobStore(tmp_path)
    store.write_spec(spec(Budget(usd=None, hours=None, iterations=1, modal_usd=None)))
    fb = FakeBench(quality_from_files)
    files = {"mod.rs": "fn solve() { let k = 5; }\n"}
    art = fb.compile("knapsack", files).artifact_id
    st = JobState.fresh(Spend(started_at=0.0))
    st.baseline = baseline()
    st.status = status
    st.iteration = 1
    st.spend.iterations = 1
    st.best = Candidate(iteration=1, files=files, artifact_id=art,
                        training=[NonceResult("t", n, True, training_quality, 1) for n in range(4)],
                        delta={"tracks": [], "mean_rel_delta": training_quality / 100 - 1,
                               "worst_rel_delta": 0.0, "error_rate": 0.0},
                        hypothesis={"title": "a", "description": "d",
                                    "strategy_tag": "local_search"})
    st.hypotheses = [{"iteration": 1, "against": 0, "title": "a", "description": "d",
                      "strategy_tag": "local_search", "outcome": "improved"}]
    store.save(st)
    return store, fb


def test_run_confirms_a_training_winner_after_the_status_was_reset(tmp_path):
    # A stop request or a bench outage landing during the held-out run leaves the job
    # "cancelled"/"paused", and the CLI resets that to "researching" on resume. The training
    # winner still has no held-out result, so the confirmation must run before anything else.
    # mutation: keying the resumed confirmation on status == "confirming" alone skips it, and
    # the loop proposes a fresh hypothesis instead — the winner is never held-out scored
    store, fb = _pending_confirmation(tmp_path, "researching")
    fp = FakeProvider([])  # an empty script raises if the loop asks for a completion
    loop = Loop(store.read_spec(), store.load(), store, fp, fb, template_rs="x",
                clock=lambda: 0.0, sleep=lambda s: None)
    result = loop.run()
    assert result.status == "won" and result.confirmed == [1]
    assert fb.score_calls == 1 and fp.calls == []


def test_run_does_not_confirm_a_best_that_never_won_on_training(tmp_path):
    # mutation: confirming every best with an empty holdout spends a held-out run on a candidate
    # that only improved on the baseline without beating it
    store, fb = _pending_confirmation(tmp_path, "researching", training_quality=100)
    fp = FakeProvider([])
    loop = Loop(store.read_spec(), store.load(), store, fp, fb, template_rs="x",
                clock=lambda: 0.0, sleep=lambda s: None)
    result = loop.run()  # the iterations cap (1) is already spent: exhausted at once
    assert result.status == "exhausted" and fb.score_calls == 0 and result.confirmed == []
