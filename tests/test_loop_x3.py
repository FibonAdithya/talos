"""The options X3 uses: no confirmation step, a nonce policy, a recorded score table."""
import re

import pytest

from talos.bench import BenchUnavailable, FakeBench
from talos.budget import Budget, Spend
from talos.loop import Loop
from talos.policy import FreshTraining
from talos.providers.fake import FakeProvider
from talos.state import JobState, JobStore

from tests.test_loop import HO, baseline, edit, hyp, quality_from_files, spec
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


# ── native scoring ────────────────────────────────────────────────────

# Each option set away from its default. The loop refuses every one on a job whose spec asks
# for native scoring: the native path (metered validation of a new best) does not know them.
NON_DEFAULT = {"confirm_wins": {"confirm_wins": False},
               "nonce_policy": {"nonce_policy": FreshTraining(4)}}


def native_loop(tmp_path, **loop_kwargs):
    """A loop on a spec that asks for native scoring, as it is constructed before calibrate()
    has run: the spec field is the only thing that says native at that point."""
    store = JobStore(tmp_path)
    sp = spec(scoring="native")
    store.write_spec(sp)
    st = JobState.fresh(Spend(started_at=0.0))
    st.baseline = baseline()
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
