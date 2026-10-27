"""Builds a run directory in tmp_path for the report tests: job.json, state.json, timeline.jsonl,
iterations/NNNN/ and, for a template start, challenge/template.rs."""
from __future__ import annotations

import json
from pathlib import Path

from talos.budget import Budget, Spend
from talos.lineage import Node
from talos.state import BaselineRecord, Candidate, JobSpec, JobState, JobStore
from talos.types import NonceResult, NonceSet

HASH = "ab" * 32
T0 = 1_000_000_000.0  # September 2001, so a date stamped with today's date is detectable
BASELINE_FILES = {"mod.rs": "line a\nline b\nline c\n"}
_DEFAULT = object()


def _baseline() -> BaselineRecord:
    return BaselineRecord(
        "base", 1, "a", dict(BASELINE_FILES),
        [NonceResult("t", 0, True, 100, 1), NonceResult("t", 1, True, 100, 1)],
        [NonceResult("t", 1_000_000, True, 100, 1), NonceResult("t", 1_000_001, True, 100, 1)])


def make_run(run_dir, hyps, nodes=(), best=None, start="mainnet", false_positives=(),
             validations=(), confirmed=(), timeline=(), iteration_files=None,
             baseline=_DEFAULT, pending_job=None, template=None, status="exhausted",
             provider="fake", model="m", stop_reason="budget exhausted: iterations"):
    """`hyps` are hypothesis records; title, description, strategy_tag and outcome get
    defaults, `against` does not. `nodes` are Node objects or dicts. `iteration_files` maps an
    iteration to the files written to iterations/NNNN/ (an empty dict makes an empty
    directory). `timeline=None` writes no timeline.jsonl. `template` is the text of
    challenge/template.rs."""
    run_dir = Path(run_dir)
    iteration_files = dict(iteration_files or {})
    spec = JobSpec(job_id="J", challenge="knapsack", direction="go", provider=provider, model=model,
                   mode="single-shot",
                   budget=Budget(usd=1.0, hours=2.0, iterations=10, compute_usd=None),
                   rand_hash=HASH, tracks=["t"], training=[NonceSet("t", HASH, 0, 2)],
                   holdout=[NonceSet("t", HASH, 1_000_000, 2)], fuel=1, created_at=T0,
                   monorepo_ref="r", challenge_id="c003", start=start)
    st = JobState.fresh(Spend(started_at=T0))
    st.status = status
    st.stop_reason = stop_reason
    st.baseline = _baseline() if baseline is _DEFAULT else baseline
    records = []
    for h in hyps:
        r = dict(h)
        i = r["iteration"]
        r.setdefault("title", f"title {i}")
        r.setdefault("description", f"description {i}")
        r.setdefault("strategy_tag", "tag")
        r.setdefault("outcome", "failed:score")
        records.append(r)
    st.hypotheses = records
    st.iteration = max((r["iteration"] for r in records), default=0)
    st.spend.iterations = len(records)
    st.nodes = [n if isinstance(n, Node) else Node.from_dict(n) for n in nodes]
    st.false_positives = list(false_positives)
    st.validations = list(validations)
    st.confirmed = list(confirmed)
    st.pending_job = pending_job
    if best is not None:
        files = iteration_files.setdefault(
            best, {"mod.rs": f"line a\nline b\nline c best {best}\n"})
        hyp = next((r for r in records if r["iteration"] == best), {})
        st.best = Candidate(
            best, dict(files), "b",
            [NonceResult("t", 0, True, 110, 1), NonceResult("t", 1, True, 111, 1)],
            {"mean_rel_delta": 0.105},
            {k: hyp.get(k, "") for k in ("title", "description", "strategy_tag")},
            holdout=[NonceResult("t", 1_000_000, True, 108, 1),
                     NonceResult("t", 1_000_001, True, 109, 1)])
    store = JobStore(run_dir)
    store.write_spec(spec)
    store.save(st)
    if timeline is not None:
        with (run_dir / "timeline.jsonl").open("w", encoding="utf-8", newline="\n") as f:
            for k, row in enumerate(timeline):
                f.write(json.dumps({"ts": T0 + 60 * k, **row}) + "\n")
    for i, files in iteration_files.items():
        d = run_dir / "iterations" / f"{i:04d}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "hypothesis.json").write_text("{}", encoding="utf-8")
        for name, text in files.items():
            (d / name).write_text(text, encoding="utf-8", newline="\n")
    if template is not None:
        (run_dir / "challenge").mkdir(exist_ok=True)
        (run_dir / "challenge" / "template.rs").write_text(template, encoding="utf-8",
                                                           newline="\n")
    return spec, st, store
