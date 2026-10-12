"""The options X3 uses: no confirmation step, a nonce policy, a recorded score table."""
import re

import pytest

from talos.bench import FakeBench
from talos.budget import Budget, Spend
from talos.loop import Loop
from talos.providers.fake import FakeProvider
from talos.state import JobState, JobStore

from tests.test_loop import baseline, edit, hyp, quality_from_files, spec

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


# ── native scoring ────────────────────────────────────────────────────

# Each option set away from its default. The loop refuses every one on a job whose spec asks
# for native scoring: the native path (metered validation of a new best) does not know them.
NON_DEFAULT = {"confirm_wins": {"confirm_wins": False}}


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
