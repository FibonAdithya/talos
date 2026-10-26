import json
from dataclasses import replace

import pytest

from talos.budget import Budget, Spend
from talos.lineage import AnnealSettings, Chain, Node
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


def test_spec_image_pin_roundtrips_and_legacy_jobs_remain_readable(tmp_path):
    pinned = replace(spec(), dev_image_tag="0.0.8")
    store = JobStore(tmp_path)
    store.write_spec(pinned)
    assert store.read_spec() == pinned
    old = pinned.to_dict()
    del old["dev_image_tag"]
    assert JobSpec.from_dict(old).dev_image_tag is None


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


def test_state_hardware_round_trips_and_defaults_to_none(tmp_path):
    # mutation: dropping the field from to_dict loses the frozen GPU on the first save, so a
    # resume probes again and can land the candidates on a different GPU from the baseline
    store = JobStore(tmp_path)
    st = JobState.fresh(Spend(started_at=0.0))
    assert st.hardware is None
    st.hardware = "A100-80GB"
    store.save(st)
    assert store.load().hardware == "A100-80GB"
    old = st.to_dict()
    del old["hardware"]
    assert JobState.from_dict(old).hardware is None  # a state.json written before the field
    # mutation: dropping the old key re-freezes a job whose state.json still says `gpu` to
    # the first option, which need not be the GPU its baseline ran on
    assert JobState.from_dict({**old, "gpu": "H100"}).hardware == "H100"


def test_scoring_fields_default_to_metered_and_round_trip():
    from talos.budget import Spend
    from talos.state import JobState
    st = JobState.fresh(Spend(started_at=0.0))
    # mutation: a default of "native" would switch every existing job's scoring on resume
    assert st.scoring is None and st.fuel_budgets_us is None and st.validations == []
    st.scoring, st.fuel_budgets_us = "native", {"t": 1_200_000}
    st.validations.append({"iteration": 1, "outcome": "validated", "reason": None})
    back = JobState.from_dict(st.to_dict())
    assert back.scoring == "native" and back.fuel_budgets_us == {"t": 1_200_000}
    assert back.validations == [{"iteration": 1, "outcome": "validated", "reason": None}]
    # a state.json written before this change loads with the defaults
    d = st.to_dict()
    for k in ("scoring", "fuel_budgets_us", "validations"):
        d.pop(k)
    old = JobState.from_dict(d)
    assert old.scoring is None and old.fuel_budgets_us is None and old.validations == []


def test_job_spec_scoring_defaults_to_metered_for_an_old_job_json():
    from talos.state import JobSpec
    from talos.budget import Budget
    from talos.types import NonceSet
    sp = JobSpec(job_id="j", challenge="knapsack", direction="d", provider="fake", model="m",
                 mode="single-shot", budget=Budget(usd=None, hours=None, iterations=1,
                                                   compute_usd=None),
                 rand_hash="ab" * 32, tracks=["t"], training=[NonceSet("t", "ab" * 32, 0, 1)],
                 holdout=[NonceSet("t", "ab" * 32, 1_000_000, 1)], fuel=1, created_at=0.0,
                 monorepo_ref="r", challenge_id="c003")
    d = sp.to_dict()
    assert d["scoring"] == "metered"
    d.pop("scoring")
    # mutation: a required field would make every job.json written before this unreadable
    assert JobSpec.from_dict(d).scoring == "metered"


def test_new_spec_fields_round_trip():
    # mutation: a field missing from the dataclass (TypeError) or from_dict dropping one
    s = replace(spec(), start="template", search="anneal", chains=3,
                anneal=AnnealSettings().to_dict())
    back = JobSpec.from_dict(json.loads(json.dumps(s.to_dict())))
    assert back == s
    assert (back.start, back.search, back.chains) == ("template", "anneal", 3)
    assert back.anneal == AnnealSettings().to_dict()


def test_tree_round_trips_through_state(tmp_path):
    # mutation: to_dict omitting "nodes" (or "chains")
    store = JobStore(tmp_path)
    state = JobState.fresh(Spend(started_at=1.0))
    state.nodes = [Node(id=1, chain=0, parent=0, family="tabu", title="t", outcome="improved",
                        score=1.5, tracks={"n=1": 1.5}, accepted=True, probability=1.0)]
    state.chains = [Chain(index=0, current=1, best=1, best_score=1.5),
                    Chain(index=1, stall=2, reheated_at=0.4)]
    store.save(state)
    loaded = store.load()
    assert loaded.nodes == state.nodes and loaded.chains == state.chains
    assert loaded.nodes and loaded.chains


def test_job_from_before_lineage_loads_with_defaults():
    # mutation: d["nodes"] instead of d.get("nodes", []), or a required spec field
    sd = spec().to_dict()
    for key in ("start", "search", "chains", "anneal"):
        del sd[key]
    td = JobState.fresh(Spend(started_at=1.0)).to_dict()
    for key in ("nodes", "chains"):
        del td[key]
    loaded_spec, loaded_state = JobSpec.from_dict(sd), JobState.from_dict(td)
    assert (loaded_spec.start, loaded_spec.search, loaded_spec.chains,
            loaded_spec.anneal) == ("mainnet", "greedy", 1, None)
    assert loaded_state.nodes == [] and loaded_state.chains == []


def test_budget_extensions_default_and_round_trip():
    # mutation: from_dict reading d["budget_extensions"] fails on every state.json written
    # before this field; to_dict omitting it loses the extension on the next save
    d = JobState.fresh(Spend(started_at=0.0)).to_dict()
    d.pop("budget_extensions", None)
    assert JobState.from_dict(d).budget_extensions == []
    entry = {"at": 1.0, "iteration": 2, "previous_status": "exhausted",
             "add": {"iterations": 3, "hours": None, "usd": None, "compute_usd": None}}
    st = JobState.fresh(Spend(started_at=0.0))
    st.budget_extensions = [entry]
    assert JobState.from_dict(st.to_dict()).budget_extensions == [entry]
