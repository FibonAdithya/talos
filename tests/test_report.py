import json

from talos.lineage import Node
from talos.report import (
    axis_for,
    holdout_points,
    node_diffs,
    read_timeline,
    report_nodes,
    run_sources,
    truncate,
)
from tests.run_fixture import make_run


def _node(i, parent=0, **kw):
    d = {"id": i, "chain": 0, "parent": parent, "family": "", "title": f"title {i}",
         "outcome": "accepted", "score": 0.01, "accepted": True}
    d.update(kw)
    return Node.from_dict(d)


def _diffs(spec, st, store, **kw):
    nodes = report_nodes(spec, st)
    return node_diffs(nodes, run_sources(store.run_dir, spec, st), spec, **kw)


# ── nodes ────────────────────────────────────────────────────────────


def test_old_run_nodes_follow_against(tmp_path):
    hyps = [{"iteration": 1, "against": 0}, {"iteration": 2, "against": 0},
            {"iteration": 3, "against": 1}]
    spec, st, _ = make_run(tmp_path, hyps)
    assert st.nodes == []
    assert [(n.id, n.parent) for n in report_nodes(spec, st)] == [(1, 0), (2, 0), (3, 1)]


def test_missing_against_defaults_to_root(tmp_path):
    spec, st, _ = make_run(tmp_path, [{"iteration": 1}])
    assert report_nodes(spec, st)[0].parent == 0


def test_resumed_old_run_merges_by_iteration(tmp_path):
    hyps = [{"iteration": i, "against": 0, "strategy_tag": "old"} for i in range(1, 7)]
    nodes = [_node(4), _node(5, family="csr", score=0.03), _node(6)]
    spec, st, _ = make_run(tmp_path, hyps, nodes=nodes)
    ns = report_nodes(spec, st)
    assert [n.id for n in ns] == [1, 2, 3, 4, 5, 6]
    assert ns[4].family == "csr" and ns[4].score == 0.03
    assert ns[0].family == "old"


def test_one_star_from_state_best(tmp_path):
    hyps = [{"iteration": i, "against": 0, "outcome": "improved"} for i in range(1, 6)]
    nodes = [_node(i, outcome="improved", best=i in (2, 5)) for i in range(1, 6)]
    spec, st, _ = make_run(tmp_path, hyps, nodes=nodes, best=5)
    ks = {n.id: (n.kind, n.former_best) for n in report_nodes(spec, st)}
    assert ks[5] == ("best", False) and ks[2][0] != "best" and ks[2][1] is True


def test_best_that_failed_holdout_keeps_false_positive_flag(tmp_path):
    hyps = [{"iteration": i, "against": 0, "outcome": "improved", "mean_rel_delta": 0.01}
            for i in range(1, 6)]
    spec, st, _ = make_run(tmp_path, hyps, best=5, false_positives=[5])
    n5 = {n.id: n for n in report_nodes(spec, st)}[5]
    assert n5.kind == "best" and n5.false_positive is True


def test_demoted_and_false_positive_kinds(tmp_path):
    hyps = [{"iteration": 1, "against": 0, "outcome": "failed:score", "mean_rel_delta": -0.01},
            {"iteration": 2, "against": 0, "outcome": "accepted", "mean_rel_delta": 0.001,
             "accepted": True},
            {"iteration": 3, "against": 0, "outcome": "failed:validation",
             "mean_rel_delta": 0.02},
            {"iteration": 4, "against": 0, "outcome": "improved", "mean_rel_delta": 0.03},
            {"iteration": 5, "against": 0, "outcome": "failed:compile"},
            {"iteration": 6, "against": 4, "outcome": "improved", "mean_rel_delta": 0.04}]
    spec, st, _ = make_run(tmp_path, hyps, best=6, false_positives=[4],
                           validations=[{"iteration": 3, "outcome": "demoted", "reason": "x"}])
    ks = {n.id: n.kind for n in report_nodes(spec, st)}
    assert ks == {1: "rejected", 2: "accepted", 3: "demoted", 4: "false_positive",
                  5: "failed", 6: "best"}


def test_started_iteration_is_a_failure(tmp_path):
    hyps = [{"iteration": 1, "against": 0, "mean_rel_delta": 0.01},
            {"iteration": 2, "against": 0, "outcome": "started"}]
    spec, st, _ = make_run(tmp_path, hyps, pending_job={"purpose": 2})
    assert report_nodes(spec, st)[-1].kind == "failed"


def test_in_flight_iteration_has_no_node(tmp_path):
    hyps = [{"iteration": 1, "against": 0}, {"iteration": 2, "against": 0}]
    spec, st, _ = make_run(tmp_path, hyps, pending_job={"purpose": 3},
                           iteration_files={3: {}})
    assert [n.id for n in report_nodes(spec, st)] == [1, 2]


# ── sources and diffs ────────────────────────────────────────────────


def test_diff_is_against_parent_not_baseline(tmp_path):
    hyps = [{"iteration": 1, "against": 0}, {"iteration": 2, "against": 1},
            {"iteration": 3, "against": 0}]
    files = {1: {"mod.rs": "line a\none b\nline c\n"}, 2: {"mod.rs": "line a\none b\ntwo c\n"},
             3: {"mod.rs": "three a\nline b\nline c\n"}}
    spec, st, store = make_run(tmp_path, hyps, iteration_files=files)
    d = _diffs(spec, st, store)
    assert "+two c" in d[2] and "-line b" not in d[2] and "three a" not in d[2]
    assert "#1/mod.rs" in d[2] and "baseline/mod.rs" in d[1]


def test_failed_compile_has_no_diff(tmp_path):
    hyps = [{"iteration": 1, "against": 0, "mean_rel_delta": 0.01},
            {"iteration": 2, "against": 0, "outcome": "failed:compile"}]
    files = {1: {"mod.rs": "line a\nline b\nline z\n"}, 2: {}}
    spec, st, store = make_run(tmp_path, hyps, iteration_files=files)
    assert _diffs(spec, st, store)[2] == "no files saved"


def test_parent_files_not_saved(tmp_path):
    hyps = [{"iteration": 2, "against": 0, "outcome": "failed:compile"},
            {"iteration": 3, "against": 2, "mean_rel_delta": 0.01}]
    files = {2: {}, 3: {"mod.rs": "line a\nline b\nline z\n"}}
    spec, st, store = make_run(tmp_path, hyps, iteration_files=files)
    assert _diffs(spec, st, store)[3] == "parent's files not saved"


def test_best_sources_come_from_state_best(tmp_path):
    hyps = [{"iteration": 1, "against": 0, "outcome": "improved", "mean_rel_delta": 0.1}]
    spec, st, store = make_run(tmp_path, hyps, best=1)
    st.best.files = {"mod.rs": "line a\nline b\nfrom state\n"}
    assert run_sources(store.run_dir, spec, st)(1) == {"mod.rs": "line a\nline b\nfrom state\n"}


def test_template_root_is_template_not_baseline(tmp_path):
    hyps = [{"iteration": 1, "against": 0, "mean_rel_delta": -0.4}]
    files = {1: {"mod.rs": "fn solve() { 1 }\n"}}
    spec, st, store = make_run(tmp_path / "a", hyps, start="template", iteration_files=files,
                               template="fn solve() {}\n")
    d = _diffs(spec, st, store)
    assert "-fn solve() {}" in d[1] and "template/mod.rs" in d[1] and "line a" not in d[1]
    spec, st, store = make_run(tmp_path / "b", hyps, start="template", iteration_files=files)
    assert _diffs(spec, st, store)[1] == "parent's files not saved"


def test_identical_files_say_no_change(tmp_path):
    hyps = [{"iteration": 1, "against": 0}]
    spec, st, store = make_run(tmp_path, hyps,
                               iteration_files={1: {"mod.rs": "line a\nline b\nline c\n"}})
    assert _diffs(spec, st, store)[1] == "no change"


def test_truncation_shares_budget_and_names_dirs(tmp_path):
    base = "".join(f"line {k}\n" for k in range(200))
    hyps = [{"iteration": i, "against": 0} for i in (1, 2, 3, 4)]
    files = {i: {"mod.rs": "".join(f"node {i} line {k}\n" for k in range(200))}
             for i in (1, 2, 3, 4)}
    spec, st, store = make_run(tmp_path, hyps, iteration_files=files)
    st.baseline.files = {"mod.rs": base}
    d = _diffs(spec, st, store, total=1000, per_node=600)
    for i in (1, 2, 3, 4):
        assert len(d[i]) <= 250 + 200   # cap 250, plus the note line
        assert d[i].rstrip().endswith(f"diff -u baseline iterations/{i:04d}")
        assert "truncated (" in d[i]


def test_template_truncation_note_names_the_file(tmp_path):
    hyps = [{"iteration": 1, "against": 0}]
    files = {1: {"mod.rs": "".join(f"x {k}\n" for k in range(100))}}
    spec, st, store = make_run(tmp_path, hyps, start="template", iteration_files=files,
                               template="fn solve() {}\n")
    d = _diffs(spec, st, store, total=100, per_node=100)
    assert d[1].endswith("diff -u challenge/template.rs iterations/0001/mod.rs")


def test_truncate_cuts_inside_a_long_line():
    out = truncate("x" * 10_000 + "\n", 100, "NOTE")
    assert len(out) <= 100 + len("NOTE") + 2 and out.endswith("NOTE")
    assert truncate("ab\n", 100, "NOTE") == "ab\n"


def test_truncate_counts_lines_left_out():
    assert truncate("a\nb\nc\nd\n", 4, "({n})") == "a\nb\n(2)"


# ── axis ─────────────────────────────────────────────────────────────


def test_axis_cuts_outlier():
    a = axis_for([0.02 + i * 0.001 for i in range(10)] + [-1.0])
    assert a.cut_lo is not None and a.cut_lo > -1.0 and a.lo > -1.0 and a.hi >= 0.029
    assert a.cut_hi is None


def test_axis_all_negative_template_scores_uncut():
    a = axis_for([-0.6 + 0.03 * i for i in range(11)])   # -0.6 .. -0.3
    assert a.cut_lo is None and a.cut_hi is None and a.lo <= -0.6 and a.hi >= 0.0


def test_axis_few_points_never_cut():
    assert axis_for([0.01, -5.0]).cut_lo is None


def test_axis_cut_needs_eight_scores():
    assert axis_for([0.02 + i * 0.001 for i in range(6)] + [-1.0]).cut_lo is None
    assert axis_for([0.02 + i * 0.001 for i in range(7)] + [-1.0]).cut_lo is not None


def test_axis_never_empty():
    a = axis_for([])
    assert a.hi > a.lo
    a = axis_for([0.0])
    assert a.hi > a.lo


# ── timeline ─────────────────────────────────────────────────────────


def test_holdout_points_from_timeline():
    rows = [{"kind": "scored", "iteration": 2, "holdout": {"mean_rel_delta": 0.5}},
            {"kind": "false_positive", "iteration": 4, "holdout": {"mean_rel_delta": -0.01}},
            {"kind": "false_positive", "iteration": 5, "error": "held-out not scored (x)"},
            {"kind": "won", "iteration": 7, "holdout": {"mean_rel_delta": 0.02}}]
    assert holdout_points(rows) == {4: -0.01, 7: 0.02}


def test_read_timeline_skips_bad_lines(tmp_path):
    p = tmp_path / "timeline.jsonl"
    p.write_text(json.dumps({"kind": "a"}) + "\n{not json\n" + json.dumps({"kind": "b"}) + "\n",
                 encoding="utf-8")
    assert len(read_timeline(p)) == 2 and read_timeline(tmp_path / "absent") == []
