"""The options X3 uses: no confirmation step, a nonce policy, a recorded score table."""
import json
import re

import pytest

from talos.bench import BenchUnavailable, FakeBench
from talos.budget import Budget, Spend
from talos.loop import Loop
from talos.policy import FreshTraining
from talos.providers.fake import FakeProvider
from talos.state import JobState, JobStore

from tests.test_loop import HASH, HO, baseline, edit, hyp, quality_from_files, spec
from tests.test_loop import make as base_make

THREE = [hyp("a"), edit(5), hyp("b"), edit(6, frm=5), hyp("c"), edit(7, frm=6)]


def make(tmp_path, script, scores=quality_from_files, iterations=3, **loop_kwargs):
    store = JobStore(tmp_path)
    sp = spec(Budget(usd=None, hours=None, iterations=iterations, compute_usd=None))
    store.write_spec(sp)
    st = JobState.fresh(Spend(started_at=0.0))
    st.baseline = baseline()
    st.status = "researching"
    fb, fp = FakeBench(scores), FakeProvider(script)
    loop = Loop(sp, st, store, fp, fb, template_rs="pub fn solve_challenge(",
                clock=lambda: 0.0, sleep=lambda s: None, **loop_kwargs)
    return loop, fp, fb


def test_default_still_stops_at_the_first_confirmed_win(tmp_path):
    # mutation: making confirm_wins default to False changes Talos for every other caller
    loop, fp, fb = make(tmp_path, THREE)
    st = loop.run()
    assert st.status == "won" and st.iteration == 1 and st.confirmed == [1]
    assert fb.holdout_runs == 1


def test_without_confirmation_every_iteration_runs(tmp_path):
    # mutation: leaving the `_confirm` call unconditional records every training win as a
    # false positive (the request carries no held-out sets, so there is nothing to confirm on)
    loop, fp, fb = make(tmp_path, THREE, confirm_wins=False)
    st = loop.run()
    assert (st.status, st.stop_reason, st.iteration) == ("exhausted", "iterations", 3)
    assert st.confirmed == [] and st.false_positives == []
    assert st.best.iteration == 3
    # mutation: still passing the held-out sets lets the bench score them on every training win
    assert all(call.holdout == [] for call in fb.calls)


def test_without_confirmation_a_weaker_winner_does_not_replace_the_best(tmp_path):
    # k=9 scores 108 (+8%); k=5 scores 104 (+4%), which is above the baseline and so is a win.
    # mutation: keeping `or wins` in the acceptance rule replaces the +8% best with the +4% one
    script = [hyp("a"), edit(9), hyp("b"), edit(5, frm=9)]
    loop, fp, fb = make(tmp_path, script, iterations=2, confirm_wins=False)
    st = loop.run()
    assert st.best.iteration == 1
    assert st.hypotheses[1]["outcome"] == "failed:score"
    assert re.search(r"let k = 9;", st.best.files["mod.rs"])


# ── nonce policy ──────────────────────────────────────────────────────

def table_scores(table):
    """quality by (k in the code, first nonce of the set); every nonce of a set scores the same"""
    def scores(challenge, files, ns):
        k = int(re.search(r"let k = (\d+);", files["mod.rs"]).group(1))
        return [table[(k, ns.start)]] * ns.count
    return scores


def ks(calls):
    return [int(re.search(r"let k = (\d+);", c.files["mod.rs"]).group(1)) for c in calls]


TWO = [hyp("a"), edit(5), hyp("b"), edit(6, frm=5)]


def test_fixed_policy_scores_the_same_nonces_every_iteration(tmp_path):
    # mutation: a fixed policy that moves its sets by 4 nonces per iteration gives [0, 4]
    loop, fp, fb = make(tmp_path, THREE, confirm_wins=False)
    loop.run()
    assert [c.training[0].start for c in fb.calls] == [0, 0, 0]


def test_fresh_policy_solves_baseline_and_best_on_each_new_window(tmp_path):
    # Iteration 1: baseline, candidate (the best is still the baseline, solved once).
    # Iteration 2: baseline, best, candidate.
    # mutation: an off-by-one in the window index (`count * iteration`) puts the windows at
    # [4, 4, 8, 8, 8]; the score table has no entry for nonce 8, so the run raises KeyError
    scores = table_scores({(1, 0): 1000, (5, 0): 1020, (1, 4): 1000, (5, 4): 1010, (6, 4): 1015})
    loop, fp, fb = make(tmp_path, TWO, scores, iterations=2, confirm_wins=False,
                        nonce_policy=FreshTraining(4))
    loop.run()
    assert [c.training[0].start for c in fb.calls] == [0, 0, 4, 4, 4]
    assert ks(fb.calls) == [1, 5, 1, 5, 6]


def test_fresh_policy_accepts_on_the_current_window_not_the_recorded_delta(tmp_path):
    # The best scored +2% on its own window and scores +1% on the new one. The candidate
    # scores +1.5% on the new one, so it is better on the same instances.
    # mutation: comparing with the best's recorded +2% rejects it
    scores = table_scores({(1, 0): 1000, (5, 0): 1020, (1, 4): 1000, (5, 4): 1010, (6, 4): 1015})
    loop, fp, fb = make(tmp_path, TWO, scores, iterations=2, confirm_wins=False,
                        nonce_policy=FreshTraining(4))
    st = loop.run()
    assert st.best.iteration == 2 and st.hypotheses[1]["outcome"] == "improved"


def test_fresh_policy_rejects_a_candidate_the_best_beats_on_the_current_window(tmp_path):
    # The best scored +1% on its own window and scores +2% on the new one. The candidate
    # scores +1.5% there.
    # mutation: comparing with the best's recorded +1% accepts it
    scores = table_scores({(1, 0): 1000, (5, 0): 1010, (1, 4): 1000, (5, 4): 1020, (6, 4): 1015})
    loop, fp, fb = make(tmp_path, TWO, scores, iterations=2, confirm_wins=False,
                        nonce_policy=FreshTraining(4))
    st = loop.run()
    assert st.best.iteration == 1 and st.hypotheses[1]["outcome"] == "failed:score"


def test_fresh_policy_compares_the_candidate_with_the_baseline_on_the_new_window(tmp_path):
    # The baseline scores 2000 on the second window. Against its stored 100 on nonces 0 to 3
    # the loop would raise a nonce mismatch; against the first window's 1000 it would report +101%.
    # mutation: comparing with the stored baseline rows ends iteration 2 as failed:score with no
    # mean_rel_delta in its record (KeyError here)
    scores = table_scores({(1, 0): 1000, (5, 0): 1020, (1, 4): 2000, (5, 4): 2000, (6, 4): 2020})
    loop, fp, fb = make(tmp_path, TWO, scores, iterations=2, confirm_wins=False,
                        nonce_policy=FreshTraining(4))
    st = loop.run()
    assert abs(st.hypotheses[1]["mean_rel_delta"] - 0.01) < 1e-12


def test_default_focused_job_keeps_a_limit_for_every_track(tmp_path):
    # A job with no X3 option set, focused on track "t" with "u" as its guard track. The stored
    # baseline ran 1 ms per nonce, so each track's limit is the 60 s floor.
    # mutation: deriving the limits from the baseline rows of the sets being scored drops "u",
    # whose guard solves would then run under the flat 600 s
    loop, fp, fb, store = base_make(tmp_path, [hyp("a"), edit(5)], track="t", two_tracks=True)
    loop.run()
    assert fb.calls[0].timeouts == {"t": 60, "u": 60}


SCORES_4 = {(1, 0): 1000, (5, 0): 1020, (1, 4): 1000, (5, 4): 1010, (6, 4): 1015}


def test_fresh_policy_limits_come_from_the_start_on_the_same_window(tmp_path):
    # The stored baseline ran 1 ms per nonce, which would give a limit of 60 s. The bench now
    # solves in 50 s per nonce, so the start's fresh results give 3 * 50 = 150 s. The start and
    # the best run under the flat limit (None); only the candidates get a derived one.
    # mutation: passing the stored baseline rows to the candidate's evaluate gives {"t": 60};
    # adding `timeouts` to `_plain_evaluate` gives a dict for the start and best calls
    loop, fp, fb = make(tmp_path, TWO, table_scores(SCORES_4), iterations=2, confirm_wins=False,
                        nonce_policy=FreshTraining(4))
    fb._runtime_ms = 50_000
    loop.run()
    assert [c.timeouts for c in fb.calls] == [None, {"t": 150}, None, None, {"t": 150}]


def test_fresh_policy_resume_finishes_the_same_iteration_on_the_same_window(tmp_path):
    # The bench goes down on the fifth call, iteration 2's candidate. The resumed loop must
    # score that candidate on the window of iteration 2 (nonces 4 to 7), not a later one.
    # mutation: taking the resumed loop's window from a counter other than `n`, such as
    # state.iteration + 2, makes the resumed calls start at 8 and the score table has no entry
    # for it (KeyError)
    loop, fp, fb = make(tmp_path, TWO, table_scores(SCORES_4), iterations=2, confirm_wins=False,
                        nonce_policy=FreshTraining(4))
    real = fb.evaluate

    def flaky(req):
        if len(fb.calls) == 4:
            fb.calls.append(req)
            raise BenchUnavailable("down")
        return real(req)

    fb.evaluate = flaky
    st = loop.run()
    assert (st.status, st.iteration) == ("paused", 1)
    assert st.pending_job["purpose"] == 2
    resumed = Loop(loop.spec, st, loop.store, FakeProvider([]), fb,
                   template_rs="pub fn solve_challenge(", clock=lambda: 0.0,
                   sleep=lambda s: None, confirm_wins=False, nonce_policy=FreshTraining(4))
    fb.evaluate = real
    fb.calls.clear()
    st = resumed.run()
    assert st.iteration == 2 and st.spend.iterations == 2
    assert [h["outcome"] for h in st.hypotheses] == ["improved", "improved"]
    assert fb.calls and all(c.training[0].start == 4 for c in fb.calls)


def test_fresh_policy_reference_requests_are_metered_and_plain(tmp_path):
    # Confirmation is on here, so a candidate's request carries the held-out sets and the
    # reference requests can be told apart from it. Both candidates win on training and lose on
    # the held-out nonces (50 against the stored baseline's 100), so the job reaches iteration 2.
    # mutation: passing the nonce sets positionally to `_bench_evaluate` makes them the scoring
    # mode and the baseline's rows the `validation` flag, so the candidate's request is built as
    # a validation request: no dead-code check, and the job's fixed sets instead of the window.
    # The table has an entry for k=6 on nonce 0 so that such a run gets as far as the assertions.
    held_out = HO[0].start
    scores = table_scores({**SCORES_4, (6, 0): 1015, (5, held_out): 50, (6, held_out): 50})
    loop, fp, fb = make(tmp_path, TWO, scores, iterations=2, nonce_policy=FreshTraining(4))
    st = loop.run()
    assert ks(fb.calls) == [1, 5, 1, 5, 6]
    references = [fb.calls[i] for i in (0, 2, 3)]
    candidates = [fb.calls[i] for i in (1, 4)]
    assert all(c.mode == "metered" and c.fuel_budgets_us is None for c in fb.calls)
    assert all(c.baseline_training is None and c.holdout == [] for c in references)
    # A validation request is the one upstream builds with the dead-code check off and the
    # job's own sets. No candidate request is one, and nothing was validated.
    assert all(c.prior_functions is not None and c.holdout == HO for c in candidates)
    assert [[r.nonce for r in c.baseline_training] for c in candidates] == [[0, 1, 2, 3],
                                                                           [4, 5, 6, 7]]
    assert st.validations == [] and st.false_positives == [1, 2]


def test_fresh_policy_compile_fix_round_stays_on_the_iterations_window(tmp_path):
    # Iteration 2's candidate (k=6) does not build; the compile-fix reply makes it k=7, which
    # is evaluated again. The bench solves in 50 s per nonce, so limits taken from the
    # baseline's rows on the window are 150 s and limits from the stored baseline are 60 s.
    # The table has an entry for k=7 on nonce 0 so that a run that re-evaluates on the job's
    # fixed sets gets as far as the assertions.
    # mutation: dropping `training=sets, base=bench_base` at the compile-fix re-evaluate scores
    # k=7 on nonces 0 to 3 against the stored baseline;
    # passing them positionally there builds that request as a validation request, on the
    # job's fixed sets with no dead-code check
    script = [hyp("a"), edit(5), hyp("b"), edit(6, frm=5), edit(7, frm=6)]
    scores = table_scores({(1, 0): 1000, (5, 0): 1020, (1, 4): 1000, (5, 4): 1010,
                           (7, 4): 1015, (7, 0): 1015})
    loop, fp, fb = make(tmp_path, script, scores, iterations=2, confirm_wins=False,
                        nonce_policy=FreshTraining(4))
    fb._compile_ok = lambda files: "let k = 6;" not in files["mod.rs"]
    fb._runtime_ms = 50_000
    st = loop.run()
    assert ks(fb.calls) == [1, 5, 1, 5, 6, 7]
    second = fb.calls[2:]
    assert [c.training[0].start for c in second] == [4, 4, 4, 4]
    fixed = fb.calls[5]
    assert [r.nonce for r in fixed.baseline_training] == [4, 5, 6, 7]
    assert fixed.timeouts == {"t": 150} and fixed.prior_functions is not None
    # scored against the baseline on the window: 1015 against 1000, above the best's +1% there
    assert st.hypotheses[1]["outcome"] == "improved"
    assert abs(st.hypotheses[1]["mean_rel_delta"] - 0.015) < 1e-12


# ── recorded scores ───────────────────────────────────────────────────

def distinct_scores(challenge, files, ns):
    """k=5 on nonce 2 scores 5027: values that appear nowhere else in a prompt"""
    k = int(re.search(r"let k = (\d+);", files["mod.rs"]).group(1))
    return [1000 * k + 7 * n + 13 for n in ns.nonces()]


def test_recorded_scores_reach_the_next_prompt_and_older_rows_do_not(tmp_path):
    loop, fp, fb = make(tmp_path, THREE, distinct_scores, confirm_wins=False, record_scores=True)
    loop.run()
    hypothesis_prompts = [user for system, user in fp.calls[0::2]]
    # iteration 1 has nothing scored before it
    assert "Per-instance" not in hypothesis_prompts[0]
    # iteration 2 sees iteration 1: the stored baseline (100 on every nonce, from
    # tests.test_loop.baseline), and the candidate, which was accepted and so is also the best
    assert "t | 0 | 100 | 5013 | 5013" in hypothesis_prompts[1]
    assert "t | 3 | 100 | 5034 | 5034" in hypothesis_prompts[1]
    # iteration 3 sees iteration 2 only
    # mutation: accumulating tables instead of replacing keeps 5013 in the third prompt
    assert "t | 0 | 100 | 6013 | 6013" in hypothesis_prompts[2]
    assert "5013" not in hypothesis_prompts[2]
    # mutation: recording NonceSet objects whole would put the rand_hash in scores.json
    saved = (tmp_path / "iterations" / "0002" / "scores.json").read_text()
    assert HASH not in saved and all(HASH not in p for p in hypothesis_prompts)
    assert json.loads(saved)["sets"] == [["t", 0, 4]]


def test_a_rejected_candidate_is_shown_beside_the_best_it_failed_to_beat(tmp_path):
    # mutation: recording a rejected candidate as accepted shows it in the best's column too
    script = [hyp("a"), edit(5), hyp("b"), edit(3, frm=5), hyp("c"), edit(7, frm=5)]
    loop, fp, fb = make(tmp_path, script, distinct_scores, confirm_wins=False, record_scores=True)
    st = loop.run()
    assert st.hypotheses[1]["outcome"] == "failed:score"
    third = fp.calls[4][1]
    assert "t | 0 | 100 | 5013 | 3013" in third
    assert json.loads((tmp_path / "iterations" / "0002" / "scores.json").read_text())["accepted"] is False


def test_scores_are_not_recorded_unless_asked(tmp_path):
    # mutation: recording by default adds a file and a prompt block to every existing job
    loop, fp, fb = make(tmp_path, THREE, distinct_scores, confirm_wins=False)
    st = loop.run()
    assert st.last_scores is None
    assert not (tmp_path / "iterations" / "0001" / "scores.json").exists()
    assert all("Per-instance" not in user for system, user in fp.calls)


def test_a_stored_table_is_not_shown_by_a_loop_that_does_not_record(tmp_path):
    # The state carries last_scores from an earlier run that recorded them; this loop was built
    # without record_scores.
    # mutation: removing `self.record_scores and` in `_context` puts the stored table (7781)
    # in every prompt
    def row(nonce):
        return {"track": "t", "nonce": nonce, "ok": True, "quality": 7781, "runtime_ms": 1,
                "error": None}
    stored = {"iteration": 9, "accepted": True, "sets": [["t", 0, 1]], "start": [row(0)],
              "best_before": [row(0)], "candidate": [row(0)]}
    loop, fp, fb = make(tmp_path, TWO, iterations=2, confirm_wins=False)
    loop.state.last_scores = stored
    st = loop.run()
    assert len(fp.calls) == 4
    assert all("Per-instance" not in user and "7781" not in user for system, user in fp.calls)
    assert st.last_scores == stored


def failing_at_6(challenge, files, ns):
    """k=6 solves nothing; any other k scores 1000 + k on every nonce"""
    k = int(re.search(r"let k = (\d+);", files["mod.rs"]).group(1))
    return [None if k == 6 else 1000 + k] * ns.count


def test_a_candidate_over_the_error_ceiling_is_recorded_on_each_policy(tmp_path):
    # Iteration 1 makes k=5 the best. Iteration 2's candidate (k=6) fails every solve.
    # Fixed policy: nonces 0 to 3, the stored baseline's 100. Fresh policy: nonces 4 to 7, the
    # baseline solved there (1001).
    # mutation: deleting the `_record_scores` call on the error-ceiling exit leaves iteration 2
    # without a scores.json (FileNotFoundError here)
    script = [hyp("a"), edit(5), hyp("b"), edit(6, frm=5), hyp("c"), edit(7, frm=5)]
    for name, policy, first, base in (("fixed", None, 0, 100),
                                      ("fresh", FreshTraining(4), 4, 1001)):
        run_dir = tmp_path / name
        loop, fp, fb = make(run_dir, script, failing_at_6, confirm_wins=False,
                            record_scores=True, nonce_policy=policy)
        st = loop.run()
        assert st.hypotheses[1]["outcome"] == "failed:runtime" and st.best.iteration != 2
        saved = json.loads((run_dir / "iterations" / "0002" / "scores.json").read_text())
        assert saved["accepted"] is False and saved["sets"] == [["t", first, 4]]
        assert [(r["nonce"], r["quality"]) for r in saved["best_before"]] == [
            (first + i, 1005) for i in range(4)]
        assert all(r["ok"] is False for r in saved["candidate"])
        assert f"t | {first} | {base} | 1005 | failed" in fp.calls[4][1]


# ── native scoring ────────────────────────────────────────────────────

# Each option set away from its default. The loop refuses every one on a job whose spec asks
# for native scoring: the native path (metered validation of a new best) does not know them.
NON_DEFAULT = {"confirm_wins": {"confirm_wins": False},
               "nonce_policy": {"nonce_policy": FreshTraining(4)},
               "record_scores": {"record_scores": True}}


def native_loop(tmp_path, spec_scoring="native", state_scoring=None, **loop_kwargs):
    """A loop on a spec that asks for native scoring, as it is constructed before calibrate()
    has run: the spec field is the only thing that says native at that point. `state_scoring`
    is what calibrate() would have left in the state."""
    store = JobStore(tmp_path)
    sp = spec(scoring=spec_scoring)
    store.write_spec(sp)
    st = JobState.fresh(Spend(started_at=0.0))
    st.baseline = baseline()
    st.scoring = state_scoring
    return Loop(sp, st, store, FakeProvider([]), FakeBench(quality_from_files),
                template_rs="pub fn solve_challenge(", clock=lambda: 0.0, sleep=lambda s: None,
                **loop_kwargs)


def test_native_job_refuses_each_x3_option(tmp_path):
    # mutation: dropping one option from the check constructs that loop without an error
    for name, kwargs in NON_DEFAULT.items():
        with pytest.raises(ValueError) as refused:
            native_loop(tmp_path / name, **kwargs)
        message = str(refused.value)
        assert [option for option in NON_DEFAULT if option in message] == [name]
        assert "metered path only" in message


def test_native_job_with_default_options_is_accepted(tmp_path):
    # mutation: refusing every native job raises here
    loop = native_loop(tmp_path)
    assert loop.spec.scoring == "native" and loop.confirm_wins is True


def test_a_state_that_says_native_refuses_each_x3_option(tmp_path):
    # The spec says metered and the state says native. Such a state scores natively
    # (Loop._scoring reads the state), so the options are refused there as well.
    # mutation: checking the spec field alone constructs these loops without an error
    for name, kwargs in NON_DEFAULT.items():
        with pytest.raises(ValueError) as refused:
            native_loop(tmp_path / name, spec_scoring="metered", state_scoring="native",
                        **kwargs)
        message = str(refused.value)
        assert [option for option in NON_DEFAULT if option in message] == [name]
        assert "metered path only" in message
    # with the options at their defaults the same state is accepted
    loop = native_loop(tmp_path / "defaults", spec_scoring="metered", state_scoring="native")
    assert loop.state.scoring == "native" and loop.record_scores is False


class FalsyFresh(FreshTraining):
    """A policy object that is false in a boolean test, as one with `__len__` or `__bool__` is."""

    def __bool__(self):
        return False


def test_a_policy_object_that_is_falsy_is_still_the_policy(tmp_path):
    # mutation: `nonce_policy or FixedTraining()` replaces this policy with the fixed one, so
    # iteration 2 scores nonces 0 to 3 again
    policy = FalsyFresh(4)
    loop, fp, fb = make(tmp_path, TWO, table_scores(SCORES_4), iterations=2, confirm_wins=False,
                        nonce_policy=policy)
    assert loop.policy is policy
    loop.run()
    assert [c.training[0].start for c in fb.calls] == [0, 0, 4, 4, 4]
