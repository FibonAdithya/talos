import json
from dataclasses import replace

import pytest

from talos.budget import Budget, Spend
from talos.state import JobSpec, JobState, JobStore, TERMINAL
from talos.types import NonceSet


def spec():
    return JobSpec(job_id="j1", challenge="knapsack", direction="try tabu", provider="fake",
                   model="m", mode="single-shot",
                   budget=Budget(usd=1.0, hours=None, iterations=None, compute_usd=None),
                   rand_hash="ab" * 32, tracks=["n=1"],
                   training=[NonceSet("n=1", "ab" * 32, 0, 2)],
                   holdout=[NonceSet("n=1", "ab" * 32, 1_000_000, 2)],
                   fuel=10, created_at=1.0, monorepo_ref="deadbeef", challenge_id="c003")


def test_spec_roundtrip_and_redaction(tmp_path):
    store = JobStore(tmp_path)
    store.write_spec(spec())
    back = store.read_spec()
    assert back == spec()
    red = spec().redacted()
    # mutation: forgetting to strip nonce-set hashes leaks the seed via `training`
    assert "ab" * 32 not in json.dumps(red)


def test_write_spec_refuses_to_overwrite(tmp_path):
    # mutation: dropping the exists() guard lets a resume or a retry silently
    # rewrite the job's inputs
    store = JobStore(tmp_path)
    store.write_spec(spec())
    before = (tmp_path / "job.json").read_text()
    with pytest.raises(FileExistsError):
        store.write_spec(spec())
    assert (tmp_path / "job.json").read_text() == before


def test_state_roundtrip_atomic(tmp_path):
    store = JobStore(tmp_path)
    st = JobState.fresh(Spend(started_at=1.0))
    st.status = "researching"
    st.iteration = 3
    store.save(st)
    assert not list(tmp_path.glob("*.tmp"))  # mutation: non-atomic write leaves temp
    assert store.load().iteration == 3 and store.load().status == "researching"


def test_timeline_appends_json_lines(tmp_path):
    # mutation: opening the timeline file in "w" mode instead of "a" would
    # truncate on every event, losing all but the last
    store = JobStore(tmp_path)
    store.event("hypothesis", iteration=1, title="x")
    store.event("score", iteration=1, delta=0.1)
    lines = (tmp_path / "timeline.jsonl").read_text().splitlines()
    assert [json.loads(line)["kind"] for line in lines] == ["hypothesis", "score"]
    assert "ts" in json.loads(lines[0])


def test_terminal_set():
    # mutation: dropping "won" or "exhausted" from TERMINAL would let a
    # finished job keep being polled as if still researching
    assert TERMINAL == {"won", "exhausted", "failed", "cancelled"}


def test_pending_job_roundtrips_and_defaults_to_none(tmp_path):
    store = JobStore(tmp_path)
    st = JobState.fresh(Spend(started_at=0.0))
    assert st.pending_job is None
    st.pending_job = {"purpose": 3, "job_id": "job_x", "files": {"mod.rs": "x"}}
    store.save(st)
    # mutation: dropping pending_job from to_dict loses the in-flight job on every resume
    assert store.load().pending_job == st.pending_job
    (tmp_path / "state.json").write_text(json.dumps({**st.to_dict(), "pending_job": None}))
    assert store.load().pending_job is None


def test_spec_track_round_trips_and_defaults_to_none(tmp_path):
    # mutation: dropping the `.get` default in from_dict makes every job.json written before
    # the field existed unresumable; dropping the field loses the focus on resume
    focused = replace(spec(), track="n=1")
    store = JobStore(tmp_path)
    store.write_spec(focused)
    assert store.read_spec().track == "n=1"
    old = spec().to_dict()
    del old["track"]
    assert JobSpec.from_dict(old).track is None
    assert spec().track is None


def test_spec_hyperparameter_fields_round_trip_default_to_none_and_survive_redaction(tmp_path):
    pinned = replace(spec(), baseline_algorithm={"name": "a", "id": "c003_a1", "adoption": 9},
                     hyperparameters={"n=1": {"x": 1}},
                     hyperparameters_source={"n=1": {"benchmark_id": "b", "player_id": "0xp",
                                                     "mean_quality": 1.0}})
    store = JobStore(tmp_path)
    store.write_spec(pinned)
    # mutation: dropping a field loses the pinned algorithm or map on resume
    assert store.read_spec() == pinned
    old = spec().to_dict()
    for key in ("baseline_algorithm", "hyperparameters", "hyperparameters_source"):
        del old[key]
    # mutation: a field without a default makes every job.json written before it unresumable
    loaded = JobSpec.from_dict(old)
    assert (loaded.baseline_algorithm, loaded.hyperparameters,
            loaded.hyperparameters_source) == (None, None, None)
    red = pinned.redacted()
    # mutation: stripping them from the redacted spec hides from the agent what every run passes
    assert red["hyperparameters"] == {"n=1": {"x": 1}}
    assert red["baseline_algorithm"]["id"] == "c003_a1"
    assert "ab" * 32 not in json.dumps(red)
