import zipfile
from dataclasses import replace

from talos.budget import Budget, Spend
from talos.package import build_package, evidence_draft, scores_markdown
from talos.state import BaselineRecord, Candidate, JobSpec, JobState, JobStore
from talos.types import NonceResult, NonceSet

HASH = "cd" * 32


def make(tmp_path, status="won"):
    spec = JobSpec(job_id="j", challenge="knapsack", direction="go", provider="fake", model="m",
                   mode="single-shot", budget=Budget(usd=1.0, hours=None, iterations=None, compute_usd=None),
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
    st.hypotheses.append({"iteration": 2, "against": 0, "title": "Bad edit", "description": "x",
                          "strategy_tag": "hybrid", "outcome": "failed:edit",
                          "error": "no edit block applied"})
    pkg = build_package(spec, st, store)
    names = {p.name for p in pkg.iterdir()}
    assert {"mod.rs", "diff_vs_baseline.patch", "scores.md", "hypotheses.md",
            "evidence_draft.md", "README.md"} <= names
    assert "let k = 9" in (pkg / "diff_vs_baseline.patch").read_text()
    # mutation: dropping the error string from the log hides why an iteration failed
    hyps_text = (pkg / "hypotheses.md").read_text()
    assert "vs best #0" in hyps_text
    assert "no edit block applied" in hyps_text
    # mutation: dumping job.json into the package leaks rand_hash
    for p in pkg.rglob("*"):
        if p.is_file():
            assert HASH not in p.read_text(errors="ignore")
    z = zipfile.ZipFile(tmp_path / "package.zip")
    assert "mod.rs" in {n.rsplit("/", 1)[-1] for n in z.namelist()}
    # mutation: building the zip from run_dir would ship job.json
    for name in z.namelist():
        assert HASH not in z.read(name).decode(errors="ignore")


def test_scores_markdown_has_per_nonce_rows_and_delta():
    # mutation: swapping baseline/candidate columns or dropping the error note
    base = [NonceResult("t", 0, True, 100, 1), NonceResult("t", 1, False, None, 1, "panic")]
    cand = [NonceResult("t", 0, True, 120, 1), NonceResult("t", 1, True, 130, 1)]
    md = scores_markdown(base, cand)
    assert "| t | 0 | 100 | 120 |" in md and "panic" in md and "mean_rel_delta" in md
    assert "| panic | 130 | err |" in md


def test_evidence_draft_prefills_challenge_and_method(tmp_path):
    # mutation: leaving "YOUR RESPONSE HERE" unreplaced, or not reading the shipped template
    spec, st, store = make(tmp_path)
    text = evidence_draft(spec, st)
    assert "knapsack" in text and "Bump k" in text and "UNIQUE ALGORITHM IDENTIFIER" in text


def test_package_for_exhausted_job_still_builds(tmp_path):
    # mutation: README always claiming confirmation regardless of state.confirmed
    spec, st, store = make(tmp_path, status="exhausted")
    pkg = build_package(spec, st, store)
    assert "never scored" in (pkg / "README.md").read_text()


def test_package_without_candidate_says_so(tmp_path):
    # mutation: rendering the candidate README with best=None tells the user to submit
    # files that do not exist
    spec, st, store = make(tmp_path, status="exhausted")
    st.best = None
    pkg = build_package(spec, st, store)
    names = {p.name for p in pkg.iterdir()}
    assert names == {"README.md"}
    text = (pkg / "README.md").read_text()
    assert "No candidate" in text
    assert "scores.md" not in text
    assert "Copy the algorithm files" not in text
    assert (tmp_path / "package.zip").exists()


def test_readme_distinguishes_false_positive_from_untested(tmp_path):
    # mutation: collapsing the two unconfirmed cases hides a measured held-out miss from
    # the submitter
    spec_fp, st_fp, store_fp = make(tmp_path / "fp", status="exhausted")
    st_fp.false_positives = [st_fp.best.iteration]
    pkg_fp = build_package(spec_fp, st_fp, store_fp)
    fp_text = (pkg_fp / "README.md").read_text()

    spec_u, st_u, store_u = make(tmp_path / "untested", status="exhausted")
    pkg_u = build_package(spec_u, st_u, store_u)
    untested_text = (pkg_u / "README.md").read_text()

    assert "false positive" in fp_text
    assert "never scored" in untested_text
    assert "false positive" not in untested_text


def test_readme_does_not_deny_a_training_win_that_was_never_confirmed(tmp_path):
    # A run can stop (budget, Ctrl-C) between the training win and the held-out run. The README
    # must say the candidate beat the baseline on training and was never confirmed, not that it
    # "did not beat the baseline on training".
    # mutation: collapsing every unconfirmed candidate into "did not beat the baseline" misreports
    # a measured training win to the submitter
    spec_w, st_w, store_w = make(tmp_path / "winner", status="exhausted")
    st_w.best.holdout = None  # +10.5% on training, confirmation never ran
    winner_text = (build_package(spec_w, st_w, store_w) / "README.md").read_text()
    assert "beat the baseline on the training nonces" in winner_text
    assert "never" in winner_text and "did not beat the baseline on training" not in winner_text

    spec_l, st_l, store_l = make(tmp_path / "loser", status="exhausted")
    st_l.best.holdout = None
    st_l.best.training = [NonceResult("t", 0, True, 100, 1), NonceResult("t", 1, True, 100, 1)]
    loser_text = (build_package(spec_l, st_l, store_l) / "README.md").read_text()
    assert "did not beat the baseline on training" in loser_text
    assert "beat the baseline on the training nonces" not in loser_text


def make_focused(tmp_path):
    """Two tracks t and u; the job focuses t; the candidate's held-out carries t's held-out rows
    and u's training rows (the guard), exactly as the loop stores them."""
    spec, st, store = make(tmp_path)
    spec = replace(spec, tracks=["t", "u"], track="t",
                   training=[NonceSet("t", HASH, 0, 2), NonceSet("u", HASH, 0, 2)],
                   holdout=[NonceSet("t", HASH, 1_000_000, 2), NonceSet("u", HASH, 1_000_000, 2)])
    st.baseline.training += [NonceResult("u", 0, True, 50, 1), NonceResult("u", 1, True, 50, 1)]
    st.baseline.holdout += [NonceResult("u", 1_000_000, True, 50, 1),
                            NonceResult("u", 1_000_001, True, 50, 1)]
    st.best.holdout += [NonceResult("u", 0, True, 50, 1), NonceResult("u", 1, True, 49, 1)]
    (tmp_path / "job.json").unlink()
    store.write_spec(spec)
    return spec, st, store


def test_focused_package_splits_scores_into_track_and_guard_sections(tmp_path):
    # mutation: comparing the whole baseline against the single-track candidate raises a
    # ScoringError and prints "(no bundle delta" instead of the delta; dropping the guard
    # section hides the u regression the confirmation saw
    spec, st, store = make_focused(tmp_path)
    pkg = build_package(spec, st, store)
    scores = (pkg / "scores.md").read_text()
    assert "# Training nonces (track t)" in scores
    assert "# Held-out nonces (track t)" in scores
    assert "# Regression guard (other tracks, training nonces)" in scores
    assert "(no bundle delta" not in scores
    guard = scores.split("# Regression guard")[1]
    assert "| u | 1 | 50 | 49 | - |" in guard
    readme = (pkg / "README.md").read_text()
    assert "Optimised for track t" in readme and "regression guard" in readme
    evidence = (pkg / "evidence_draft.md").read_text()
    assert "Optimised for track t" in evidence and "Regression guard" in evidence


def test_unfocused_package_keeps_todays_headings(tmp_path):
    # mutation: the focus headings leaking into unfocused packages
    spec, st, store = make(tmp_path)
    scores = (build_package(spec, st, store) / "scores.md").read_text()
    assert "# Training nonces\n" in scores and "(track" not in scores
    assert "Regression guard" not in scores


def test_package_records_the_hyperparameters_and_their_source(tmp_path):
    spec, st, store = make(tmp_path)
    spec = replace(spec, hyperparameters={"t": {"x": 1}},
                   hyperparameters_source={"t": {"benchmark_id": "bm1", "player_id": "0xp",
                                                 "mean_quality": 150.0}})
    pkg = build_package(spec, st, store)
    readme = (pkg / "README.md").read_text()
    # mutation: omitting the section leaves the user submitting without the values they beat with
    assert "## Hyperparameters" in readme
    assert '- t: `{"x": 1}` (mainnet benchmark bm1 by 0xp, mean quality 150)' in readme
    # mutation: not writing the file leaves nothing to paste into a benchmarker config
    import json
    assert json.loads((pkg / "hyperparameters.json").read_text()) == {"t": {"x": 1}}


def test_package_without_hyperparameters_has_no_section_or_file(tmp_path):
    spec, st, store = make(tmp_path)
    pkg = build_package(spec, st, store)
    # mutation: writing the section for None claims values were used when none were
    assert "## Hyperparameters" not in (pkg / "README.md").read_text()
    assert not (pkg / "hyperparameters.json").exists()
