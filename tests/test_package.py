import zipfile

from talos.budget import Budget, Spend
from talos.package import build_package, evidence_draft, scores_markdown
from talos.state import BaselineRecord, Candidate, JobSpec, JobState, JobStore
from talos.types import NonceResult, NonceSet

HASH = "cd" * 32


def make(tmp_path, status="won"):
    spec = JobSpec(job_id="j", challenge="knapsack", direction="go", provider="fake", model="m",
                   mode="single-shot", budget=Budget(usd=1.0, hours=None, iterations=None, modal_usd=None),
                   rand_hash=HASH, tracks=["t"], training=[NonceSet("t", HASH, 0, 2)],
                   holdout=[NonceSet("t", HASH, 1_000_000, 2)], fuel=1, created_at=0.0,
                   monorepo_ref="r", challenge_id="c003")
    base = BaselineRecord("base", 1, "a", {"mod.rs": "fn a() {}\nlet k = 1;\n"},
                          [NonceResult("t", 0, True, 100, 1), NonceResult("t", 1, True, 100, 1)],
                          [NonceResult("t", 1_000_000, True, 100, 1), NonceResult("t", 1_000_001, True, 100, 1)])
    cand = Candidate(3, {"mod.rs": "fn a() {}\nlet k = 9;\n"}, "b",
                     [NonceResult("t", 0, True, 110, 1), NonceResult("t", 1, True, 111, 1)],
                     {"mean_rel_delta": 0.105}, {"title": "Bump k", "description": "d", "strategy_tag": "hybrid"},
                     holdout=[NonceResult("t", 1_000_000, True, 108, 1), NonceResult("t", 1_000_001, True, 109, 1)])
    st = JobState.fresh(Spend(started_at=0.0))
    st.status, st.best, st.baseline = status, cand, base
    st.hypotheses = [{"iteration": 3, "against": 0, "title": "Bump k", "description": "d",
                      "strategy_tag": "hybrid", "outcome": "improved"}]
    store = JobStore(tmp_path)
    store.write_spec(spec)
    store.save(st)
    return spec, st, store


def test_package_contents_and_no_hash(tmp_path):
    spec, st, store = make(tmp_path)
    pkg = build_package(spec, st, store)
    names = {p.name for p in pkg.iterdir()}
    assert {"mod.rs", "diff_vs_baseline.patch", "scores.md", "hypotheses.md",
            "evidence_draft.md", "README.md"} <= names
    assert "let k = 9" in (pkg / "diff_vs_baseline.patch").read_text()
    # mutation: dumping job.json into the package leaks rand_hash
    for p in pkg.rglob("*"):
        if p.is_file():
            assert HASH not in p.read_text(errors="ignore")
    z = zipfile.ZipFile(tmp_path / "package.zip")
    assert "mod.rs" in {n.rsplit("/", 1)[-1] for n in z.namelist()}


def test_scores_markdown_has_per_nonce_rows_and_delta():
    # mutation: swapping baseline/candidate columns or dropping the error note
    base = [NonceResult("t", 0, True, 100, 1), NonceResult("t", 1, False, None, 1, "panic")]
    cand = [NonceResult("t", 0, True, 120, 1), NonceResult("t", 1, True, 130, 1)]
    md = scores_markdown(base, cand)
    assert "| t | 0 | 100 | 120 |" in md and "panic" in md and "mean_rel_delta" in md


def test_evidence_draft_prefills_challenge_and_method(tmp_path):
    # mutation: leaving "YOUR RESPONSE HERE" unreplaced, or not reading the shipped template
    spec, st, store = make(tmp_path)
    text = evidence_draft(spec, st)
    assert "knapsack" in text and "Bump k" in text and "UNIQUE ALGORITHM IDENTIFIER" in text


def test_package_for_exhausted_job_still_builds(tmp_path):
    # mutation: README always claiming confirmation regardless of state.confirmed
    spec, st, store = make(tmp_path, status="exhausted")
    pkg = build_package(spec, st, store)
    assert "not confirmed" in (pkg / "README.md").read_text()
