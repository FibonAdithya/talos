import json
import re
from datetime import datetime, timezone

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
from tests.run_fixture import BASELINE_FILES, HASH, make_run


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


# ── render_report ────────────────────────────────────────────────────


def _render(spec, st, store):
    from talos.report import render_report
    return render_report(spec, st, read_timeline(store.run_dir / "timeline.jsonl"),
                         run_sources(store.run_dir, spec, st))


def rendered(tmp_path, hyps, **kw):
    return _render(*make_run(tmp_path, hyps, **kw))


def _scored(n, against=0, **kw):
    return [{"iteration": i, "against": against, "mean_rel_delta": 0.01 * i, **kw}
            for i in range(1, n + 1)]


def _classes(html):
    return dict(re.findall(r'<g data-node="(\d+)" class="([^"]*)"', html))


def test_every_node_drawn_once(tmp_path):
    hyps = _scored(6) + [{"iteration": 7, "against": 0, "outcome": "failed:compile"}]
    html = rendered(tmp_path, hyps, best=5)
    ids = re.findall(r'data-node="(\d+)"', html)
    assert sorted(map(int, ids)) == [1, 2, 3, 4, 5, 6, 7]


def test_no_external_urls(tmp_path):
    html = rendered(tmp_path, _scored(3), best=2,
                    timeline=[{"kind": "started"}, {"kind": "false_positive", "iteration": 2,
                                                    "holdout": {"mean_rel_delta": -0.01}}])
    assert not re.search(r"https?://|@import|(?:src|href)=\"(?!#)", html)


def test_no_rand_hash(tmp_path):
    assert HASH not in rendered(tmp_path, _scored(3), best=2)


def test_no_rand_hash_from_timeline_rows(tmp_path):
    # The loop never writes the hash to the timeline (tests/test_loop.py); this guards the
    # report against ever rendering a row's text, should one carry it.
    timeline = [{"kind": "started", "note": HASH},
                {"kind": "won", "iteration": 2, "error": HASH,
                 "holdout": {"mean_rel_delta": 0.02, "tracks": HASH}}]
    assert HASH not in rendered(tmp_path, _scored(3), best=2, confirmed=[2], timeline=timeline)


def test_llm_text_is_escaped(tmp_path):
    evil = "</script><script>alert(1)</script>"
    hyps = [{"iteration": 1, "against": 0, "outcome": "improved", "mean_rel_delta": 0.1,
             "title": evil, "description": evil, "strategy_tag": evil, "error": evil}]
    files = {1: {"mod.rs": f"line a\n{evil}\nline c\n"}}
    html = rendered(tmp_path, hyps, best=1, iteration_files=files)
    assert evil not in html and "<script>alert(1)" not in html
    assert html.count("<script") == 2   # the JSON block and the page script


def test_holdout_diamond_at_iteration(tmp_path):
    timeline = [{"kind": "started"},
                {"kind": "false_positive", "iteration": 4, "holdout": {"mean_rel_delta": -0.01}}]
    html = rendered(tmp_path, _scored(5), best=5, false_positives=[4], timeline=timeline)
    assert 'data-holdout="4"' in html and 'data-holdout="5"' not in html


def test_failed_node_on_strip_and_outlier_cut(tmp_path):
    hyps = [{"iteration": i, "against": 0, "mean_rel_delta": 0.02 + 0.001 * i}
            for i in range(1, 11)]
    hyps += [{"iteration": 11, "against": 0, "mean_rel_delta": -1.0},
             {"iteration": 12, "against": 0, "outcome": "failed:compile"}]
    cls = _classes(rendered(tmp_path, hyps))
    assert "cut" in cls["11"].split() and "cut" not in cls["5"].split()
    assert {"k-failed", "strip"} <= set(cls["12"].split()) and "strip" not in cls["5"].split()


def test_one_star_rendered(tmp_path):
    hyps = _scored(5, outcome="improved")
    nodes = [_node(i, outcome="improved", score=0.01 * i, best=i in (2, 5))
             for i in range(1, 6)]
    cls = _classes(rendered(tmp_path, hyps, nodes=nodes, best=5))
    assert [i for i, c in cls.items() if "k-best" in c.split()] == ["5"]


def test_unrecorded_parent_has_no_edge(tmp_path):
    hyps = [{"iteration": 1, "against": 0, "mean_rel_delta": 0.01},
            {"iteration": 2, "against": 1, "mean_rel_delta": 0.02},
            {"iteration": 3, "against": 99, "mean_rel_delta": 0.03}]
    html = rendered(tmp_path, hyps)
    assert "parent #99 not recorded" in html
    assert 'data-edge="0-1"' in html and 'data-edge="1-2"' in html
    assert 'data-edge="99-3"' not in html


def test_deterministic(tmp_path):
    run = make_run(tmp_path, _scored(4), best=3,
                   timeline=[{"kind": "started"}, {"kind": "won", "iteration": 3,
                                                   "holdout": {"mean_rel_delta": 0.02}}])
    a, b = _render(*run), _render(*run)
    assert a == b
    assert datetime.now(timezone.utc).strftime("%Y-%m-%d") not in a
    assert "2001-09-09" in a   # the fixture's first timeline ts


def test_no_best_and_no_baseline_branches(tmp_path):
    no_best = rendered(tmp_path / "a", _scored(3))
    assert "No candidate beat the baseline" in no_best and 'data-node="1"' in no_best
    no_base = rendered(tmp_path / "b", _scored(3), baseline=None)
    assert "data-node=" not in no_base and "knapsack" in no_base


def test_per_track_tables(tmp_path):
    html = rendered(tmp_path, _scored(3), best=2)
    # make_run's best: baseline 100/100, candidate 110/111 on training -> +10.500%
    assert "Training nonces" in html and "Held-out nonces" in html
    assert "<td>t</td>" in html and "+10.500%" in html


def test_best_diff_count_is_of_the_whole_diff(tmp_path):
    # 5000 added lines of 100 characters: the diff is over BEST_DIFF_CAP, so it is cut
    added = "".join(f"{k:06d}" + "x" * 93 + "\n" for k in range(5000))
    html = rendered(tmp_path, _scored(1), best=1,
                    iteration_files={1: {"mod.rs": BASELINE_FILES["mod.rs"] + added}})
    assert "… truncated (" in html
    assert "5000 lines added or removed" in html


def test_200_node_report_under_2mb(tmp_path):
    def body(i, lines):
        return "".join(f"    let value_{i}_{k:05d} = compute(x, y, z);\n" for k in range(lines))
    hyps = [{"iteration": i, "against": i - 1, "outcome": "improved",
             "mean_rel_delta": 0.0001 * i} for i in range(1, 201)]
    files = {i: {"mod.rs": body(i, 1300)} for i in range(1, 200)}
    files[200] = {"mod.rs": body(200, 11000)}
    html = rendered(tmp_path, hyps, best=200, iteration_files=files)
    assert len(html.encode("utf-8")) < 2_000_000
    assert html.count("truncated (") == 201


# ── write_report ─────────────────────────────────────────────────────


def test_write_report_with_no_timeline(tmp_path):
    from talos.report import write_report
    _, _, store = make_run(tmp_path, _scored(3), best=2, timeline=None)
    path = write_report(store)
    assert path == tmp_path / "package" / "report.html"
    text = path.read_text(encoding="utf-8")
    assert "<th>Started</th><td>n/a</td>" in text and "<th>Elapsed</th><td>n/a</td>" in text
    assert "data-holdout" not in text and 'data-node="3"' in text


def test_write_report_reads_the_timeline(tmp_path):
    from talos.report import write_report
    _, _, store = make_run(tmp_path, _scored(3), best=2,
                           timeline=[{"kind": "started"}, {"kind": "x"}, {"kind": "y"}])
    text = write_report(store).read_text(encoding="utf-8")
    assert "<th>Elapsed</th><td>0h 02m</td>" in text


OK = {"mean_rel_se": 0.002, "noise_ratio": 2.5, "noise_band": "clear", "noise_state": "ok"}


def test_whisker_for_ok_nodes_only(tmp_path):
    # spec Tests 13. mutation: drawing on "SE is not null" draws one on the broken node 2
    hyps = [{**_scored(1)[0], **OK},
            {"iteration": 2, "against": 0, "mean_rel_delta": -0.3, "mean_rel_se": 0.01,
             "noise_ratio": None, "noise_band": None, "noise_state": "over_ceiling"}]
    html = rendered(tmp_path, hyps)
    assert 'data-whisker="1"' in html and 'data-whisker="2"' not in html
    assert '"noise_band": "clear"' in html  # the panel's node data


def test_old_records_draw_no_whisker(tmp_path):
    # Review Focus 5 and spec Tests 7
    html = rendered(tmp_path, _scored(3))
    assert "data-whisker" not in html


def test_whisker_is_clipped_to_the_plot(tmp_path):
    # Review Focus 4. An SE of 50 percentage points runs far past the axis.
    # mutation: no clipping puts a y coordinate outside the plot area
    from talos.report import _PLOT_BOTTOM, _TOP
    hyps = [{**h, **OK, "mean_rel_se": 0.5} for h in _scored(3)]
    html = rendered(tmp_path, hyps)
    pairs = re.findall(r'<line class="whisker"[^>]* y1="([\d.]+)"[^>]* y2="([\d.]+)"', html)
    assert len(pairs) == 3
    ys = [float(v) for pair in pairs for v in pair]
    assert all(_TOP <= y <= _PLOT_BOTTOM for y in ys)


def test_track_table_mean_row_colour_follows_the_sign(tmp_path):
    # spec Tests 9. The fixture's best scores [110, 111]. Against a baseline of [200, 202] the
    # delta is about -45% at a clear ratio. mutation: colouring by band alone makes it green
    from talos.state import BaselineRecord
    from talos.types import NonceResult
    low = BaselineRecord("base", 1, "a", dict(BASELINE_FILES),
                         [NonceResult("t", 0, True, 200, 1), NonceResult("t", 1, True, 202, 1)],
                         [NonceResult("t", 1_000_000, True, 100, 1),
                          NonceResult("t", 1_000_001, True, 100, 1)])
    html = rendered(tmp_path, _scored(1), best=1, baseline=low)
    row = re.search(r"<tr class='total[^']*'>", html).group(0)
    assert "noise-muted" in row and "noise-clear" not in row
    up = rendered(tmp_path / "up", _scored(1), best=1)  # the default baseline: +10.5%, clear
    assert "noise-clear" in re.search(r"<tr class='total[^']*'>", up).group(0)


def test_track_table_zero_se_is_not_n1(tmp_path):
    # mutation: `if t.rel_se:` prints "n=1" for a zero-spread track. A best of [110, 110]
    # against [100, 100] has differences [10, 10]: SE 0.0
    from talos.state import Candidate
    from talos.types import NonceResult
    spec, st, store = make_run(tmp_path, _scored(1), best=1)
    st.best = Candidate(1, st.best.files, "b",
                        [NonceResult("t", 0, True, 110, 1), NonceResult("t", 1, True, 110, 1)],
                        {"mean_rel_delta": 0.1}, st.best.hypothesis)
    html = _render(spec, st, store)
    assert "±0.000%" in html and "<td>n=1</td>" not in html


def test_track_table_over_the_ceiling_shows_no_se(tmp_path):
    # spec decision 3. mutation: printing the SE cells whatever the noise state
    from talos.state import Candidate
    from talos.types import NonceResult
    spec, st, store = make_run(tmp_path, _scored(1), best=1)
    st.best = Candidate(1, st.best.files, "b",
                        [NonceResult("t", 0, True, 110, 1),
                         NonceResult("t", 1, False, None, 1, "panic")],
                        {"mean_rel_delta": 0.1}, st.best.hypothesis)
    html = _render(spec, st, store)
    tables = html[html.index("<h3>"):html.index("</table>", html.index("<h3>"))]
    assert "<th>SE</th>" in tables and "±" not in tables and "n=1" not in tables


def test_tooltip_uses_the_panel_noise_text(tmp_path):
    # spec: the tooltip shows the same noise text as the panel. mutation: the tooltip
    # appending only the ratio again
    html = rendered(tmp_path, [{**_scored(1)[0], **OK}])
    handler = html[html.index('addEventListener("mousemove"'):]
    handler = handler[:handler.index('addEventListener("mouseleave"')]
    assert "noiseText(n)" in handler and "noise_ratio" not in handler


def test_report_script_carries_the_shared_state_words(tmp_path):
    # the JS keeps a literal copy of the words in talos.scoring.NOISE_WORDS
    from talos.scoring import NOISE_WORDS
    html = rendered(tmp_path, _scored(1))
    for word in NOISE_WORDS.values():
        assert word.replace("±", "\\u00b1") in html
