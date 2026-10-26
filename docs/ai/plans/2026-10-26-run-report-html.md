# Run report.html Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Write a self-contained `runs/<job_id>/package/report.html` (header, score-plot lineage figure, node panel with diffs, per-track table, best diff, hypotheses) from an existing run directory, on every `build_package` call and via `talos report JOB_ID`.

**Architecture:** New `talos/report.py`. The data layer is pure functions over `JobSpec`, `JobState`, timeline rows and a sources callable: nodes, diffs, axis and held-out points. The renderer is a pure `render_report` that assembles HTML with inline CSS, a Python-built SVG, a small inline JS, and node data in a `<script type="application/json">` block. A thin I/O layer (`write_report`) reads the run directory. `build_package` and a new CLI subcommand call it.

**Tech Stack:** Python 3.11 standard library only (`html`, `json`, `difflib`, `statistics`). pytest, ruff. No new dependencies.

**Spec:** `docs/ai/specs/2026-10-26-run-report-html-design.md`. Read it first; this plan does not repeat its page layout.

## Global Constraints

- The report contains no `http://`, `https://`, `@import`, or `src=`/`href=` to anything off-page. No CDN, no fonts.
- The report never contains `spec.rand_hash`. Shown settings come from `spec.redacted()` only.
- Every piece of LLM-written text (title, description, error, family, tag, diff) is `html.escape`d in HTML. In the JSON block, every `<` is replaced with the six-character JSON escape `\u003c` (backslash, u, 0, 0, 3, c): `json.dumps(data).replace("<", "\\u003c")`.
- Under 2,000,000 bytes for a 200-node run. Node diffs share a 1,000,000-character budget, each capped at 64,000 characters. The best diff is capped at 400,000 characters.
- `render_report` does no I/O and reads no clock or randomness. The same inputs give byte-identical output.
- Files are written with `encoding="utf-8", newline="\n"` (Windows CI; see `talos/package.py`).
- The inline `<svg>` has no `xmlns` attribute (HTML5 does not need it, and its `http://www.w3.org/...` value would fail the no-URL test).
- A report failure never fails `build_package`. The zip is still written.
- Verify with `make check PYTHON=.venv/bin/python` (ruff, pytest `-m "not live"`, agentify).

## Review Focus

1. **In-progress run:** the last hypothesis has `outcome: "started"` and `state.pending_job` is set. The node is drawn as a failure (no score) and nothing crashes. Test in Task 1.
2. **Very old hypothesis records with no `against` key:** the parent defaults to 0 (the root). Test in Task 1.
3. **One diff line longer than the cap** (minified or generated source): truncation must also cut inside a line, so the cap holds. Test in Task 1.
4. **A node whose parent id has no node** (a record lost, or a parent outside the recorded list): the edge is not drawn, and the panel says `parent #N not recorded`. Test in Task 2.
5. **Run directory with no or empty `timeline.jsonl`:** the header shows `n/a` for times and there are no held-out points. Test in Task 3.

---

### Task 1: Report data layer (nodes, sources, diffs, axis, held-out points)

**Files:**
- Create: `talos/report.py`
- Modify: `talos/package.py:52-75`. Split `scores_sections` into `score_pairs` plus the markdown wrapper.
- Test: `tests/test_report.py`

**Interfaces:**
- Produces (all in `talos/report.py`):
  - `@dataclass(frozen=True) class ReportNode` with fields `id: int, parent: int, chain: int, family: str, title: str, description: str, strategy_tag: str, outcome: str, error: str | None, score: float | None, tracks: dict[str, float] | None, metered_score: float | None, accepted: bool, probability: float | None, draw: float | None, temperature: float | None, kind: str, former_best: bool, confirmed: bool, false_positive: bool`.
  - `KINDS = ("best", "false_positive", "demoted", "improved", "accepted", "rejected", "failed")`.
  - `report_nodes(spec: JobSpec, state: JobState) -> list[ReportNode]`. One node per hypothesis record, sorted by id. The root is not included.
  - `run_sources(run_dir: Path, spec: JobSpec, state: JobState) -> Callable[[int], dict[str, str] | None]`
  - `root_location(spec: JobSpec) -> str`. Returns `"challenge/template.rs"` for a template start and `"baseline"` otherwise. It is used in truncation notes.
  - `node_diffs(nodes: list[ReportNode], sources: Callable[[int], dict[str, str] | None], spec: JobSpec, total: int = 1_000_000, per_node: int = 64_000) -> dict[int, str]`. The value is the diff text or a message: `"no files saved"`, `"parent's files not saved"`, `"no change"`.
  - `truncate(text: str, cap: int, note: str) -> str`
  - `@dataclass(frozen=True) class Axis` with fields `lo: float, hi: float, cut_lo: float | None, cut_hi: float | None`, plus `axis_for(scores: list[float]) -> Axis`.
  - `read_timeline(path: Path) -> list[dict]` and `holdout_points(timeline: list[dict]) -> dict[int, float]`.
- Produces in `talos/package.py`: `score_pairs(spec, state) -> list[tuple[str, list[NonceResult], list[NonceResult]]]` (heading, baseline rows, candidate rows). `scores_sections` becomes `[(h, scores_markdown(b, c)) for h, b, c in score_pairs(...)]` and its output is unchanged.

Rules the code must follow (from the spec, sections "Nodes" and "Node sources and diffs"):
- Merge by iteration. A stored `state.nodes` entry with a matching id supplies `parent, chain, family, score=rank_score(), tracks, metered_score, accepted, probability, draw, temperature`. Otherwise these come from the record: `parent = h.get("against", 0)`, `chain = h.get("chain", 0)`, `family = h.get("design_family") or h.get("strategy_tag", "")`, `score = h.get("mean_rel_delta")`, `accepted = h.get("accepted", h["outcome"] in ("improved", "won", "accepted"))`. Title, description, strategy tag, outcome and error always come from the record.
- `kind`, first match wins: current best (`state.best.iteration`) → `best`; in `state.false_positives` → `false_positive`; iteration with a `state.validations` entry whose outcome is `"demoted"` → `demoted`; `score is None` → `failed`; outcome in `("improved", "won")` → `improved`; `accepted` → `accepted`; else `rejected`.
- `former_best = stored node.best and kind != "best"`. `confirmed = id in state.confirmed`. `false_positive = id in state.false_positives`. This flag is separate from `kind`, because the current best is often also a false positive (an "improved but not won" run keeps it as best), and `kind == "best"` would hide that.
- The sources lookup: id 0 gives `state.baseline.files`, or for a template start `{"mod.rs": <run_dir>/challenge/template.rs text}` (None if that file is absent). The current best gives `state.best.files`. Any other id gives the files present in `iterations/NNNN/` among the root's file names, decoded with `errors="replace"`, and None if none are present. With no baseline and no template, the root is None.
- The diff text is `talos.package._diff(parent_files, node_files, label)` with `label = "template"` if the parent is the template root, `"baseline"` if it is the baseline root, else `f"#{parent}"`.
- The per-node cap is `min(per_node, total // max(1, count of nodes whose diff is a real diff))`. The truncation note is `f"… truncated ({n} more lines); full diff: diff -u {parent_loc} {node_loc}"`. Here a location is `root_location(spec)` for id 0 and `f"iterations/{id:04d}"` otherwise.
- `axis_for`: with fewer than 8 scores there is no cut. Otherwise, with `q1, _, q3 = statistics.quantiles(scores, n=4, method="inclusive")`, `w = max(q3 - q1, 0.01)`, `cut_lo = q1 - 3 * w`, `cut_hi = q3 + 3 * w`. Each cut is kept only if some score lies beyond it, else it is None. `lo = min(0, min of scores within cuts)` and `hi = max(0, max of scores within cuts)`. If `lo == hi`, then `hi = lo + 0.01`.
- `read_timeline` skips lines that fail `json.loads` and returns `[]` for a missing file. `holdout_points` takes rows with `kind in ("won", "false_positive")` and a dict `holdout` carrying `mean_rel_delta`, keyed by `row["iteration"]`.

- [ ] **Step 0: Record the starting test count**

  Run: `.venv/bin/python -m pytest -q -m "not live" > /tmp/claude-1000/-home-fibonadithya-TIG-Talos/2a5dd1b3-6d32-4d01-9a6b-64a5dc5d0075/scratchpad/pytest-before.log 2>&1; tail -3 /tmp/claude-1000/-home-fibonadithya-TIG-Talos/2a5dd1b3-6d32-4d01-9a6b-64a5dc5d0075/scratchpad/pytest-before.log`. Run it in the background with a generous timeout, and note `N passed`.

- [ ] **Step 1: Write the fixture helper and failing tests**

  In a new `tests/run_fixture.py` (a helper module, not a test module), write `make_run(run_dir, hyps, nodes=(), best=None, start="mainnet", false_positives=(), validations=(), confirmed=(), timeline=(), iteration_files=None)`. It writes `job.json` via `JobStore.write_spec` (copy the `make()` pattern in `tests/test_package.py`, with `HASH = "ab" * 32`), then `state.json`, `timeline.jsonl` and `iterations/NNNN/<name>`. It returns `(spec, state, store)`. The baseline files are `{"mod.rs": "line a\nline b\nline c\n"}`. Every timeline `ts` in the fixture is around `1_000_000_000` (September 2001), so a date stamped with today's date is detectable. `tests/test_report.py` and `tests/test_cli.py` import it with `from tests.run_fixture import make_run, HASH`.

  Tests, each with the mutation it catches:

  ```python
  def test_old_run_nodes_follow_against(tmp_path):
      # no state.nodes; 1 and 2 hang off 0, 3 hangs off 1
      # mutation: ignoring `against`, or requiring state.nodes
      assert [(n.id, n.parent) for n in report_nodes(spec, st)] == [(1, 0), (2, 0), (3, 1)]

  def test_missing_against_defaults_to_root(tmp_path):        # Review Focus 2
      assert report_nodes(spec, st)[0].parent == 0

  def test_resumed_old_run_merges_by_iteration(tmp_path):
      # hypotheses 1-6, stored nodes for 4-6 only, node 5 stored with family "csr" and score 0.03
      # mutation: using state.nodes alone when non-empty
      ns = report_nodes(spec, st)
      assert [n.id for n in ns] == [1, 2, 3, 4, 5, 6]
      assert ns[4].family == "csr" and ns[4].score == 0.03

  def test_one_star_from_state_best(tmp_path):
      # stored nodes 2 and 5 both best=True; state.best.iteration == 5
      # mutation: trusting node.best
      ks = {n.id: (n.kind, n.former_best) for n in report_nodes(spec, st)}
      assert ks[5] == ("best", False) and ks[2][0] != "best" and ks[2][1] is True

  def test_best_that_failed_holdout_keeps_false_positive_flag(tmp_path):
      # best is 5 and 5 is in state.false_positives
      # mutation: deriving the flag from kind (first match wins)
      n5 = {n.id: n for n in report_nodes(spec, st)}[5]
      assert n5.kind == "best" and n5.false_positive is True

  def test_demoted_and_false_positive_kinds(tmp_path):
      # 3: outcome failed:validation, validations [{"iteration": 3, "outcome": "demoted"}], score 0.02
      # 4: outcome improved, in false_positives; best is 6
      # mutation: colouring by outcome only
      ks = {n.id: n.kind for n in report_nodes(spec, st)}
      assert ks[3] == "demoted" and ks[4] == "false_positive"

  def test_started_iteration_is_a_failure(tmp_path):           # Review Focus 1
      # last record outcome "started", no mean_rel_delta, pending_job set
      assert report_nodes(spec, st)[-1].kind == "failed"

  def test_diff_is_against_parent_not_baseline(tmp_path):
      # iterations/0001/mod.rs = "line a\none b\nline c\n"; 0002 (parent 1) = "line a\none b\ntwo c\n";
      # 0003 (a sibling, parent 0) = "three a\nline b\nline c\n"
      # mutation: diffing against baseline or best
      d = node_diffs(nodes, run_sources(store.run_dir, spec, st), spec)
      assert "+two c" in d[2] and "-line b" not in d[2] and "three a" not in d[2]

  def test_failed_compile_has_no_diff(tmp_path):
      # iterations/0002/ holds hypothesis.json only
      assert node_diffs(...)[2] == "no files saved"

  def test_truncation_shares_budget_and_names_dirs(tmp_path):
      # call node_diffs with total=1000, per_node=600 over 4 nodes each changing 200 lines
      # mutation: removing the budget
      for i in (1, 2, 3, 4):
          assert len(d[i]) <= 250 + 200   # cap 250, plus the note line
          assert d[i].rstrip().endswith(f"diff -u baseline iterations/{i:04d}")

  def test_truncate_cuts_inside_a_long_line():                  # Review Focus 3
      out = truncate("x" * 10_000 + "\n", 100, "NOTE")
      assert len(out) <= 100 + len("NOTE") + 2 and out.endswith("NOTE")
      assert truncate("ab\n", 100, "NOTE") == "ab\n"   # mutation: always appending the note

  def test_axis_cuts_outlier():
      # mutation: removing the cut
      a = axis_for([0.02 + i * 0.001 for i in range(10)] + [-1.0])
      assert a.cut_lo is not None and a.cut_lo > -1.0 and a.lo > -1.0 and a.hi >= 0.029

  def test_axis_all_negative_template_scores_uncut():
      a = axis_for([-0.6 + 0.03 * i for i in range(11)])   # -0.6 .. -0.3
      assert a.cut_lo is None and a.cut_hi is None and a.lo <= -0.6 and a.hi >= 0.0

  def test_axis_few_points_never_cut():
      assert axis_for([0.01, -5.0]).cut_lo is None

  def test_holdout_points_from_timeline():
      rows = [{"kind": "false_positive", "iteration": 4, "holdout": {"mean_rel_delta": -0.01}},
              {"kind": "false_positive", "iteration": 5, "error": "held-out not scored (x)"},
              {"kind": "won", "iteration": 7, "holdout": {"mean_rel_delta": 0.02}}]
      assert holdout_points(rows) == {4: -0.01, 7: 0.02}

  def test_read_timeline_skips_bad_lines(tmp_path):
      # file with one valid line, one "{not json", one valid line
      assert len(read_timeline(p)) == 2 and read_timeline(tmp_path / "absent") == []
  ```

  In `tests/test_package.py`, add `test_score_pairs_match_scores_sections`. It asserts that, for the `make()` fixture, `[h for h, _, _ in score_pairs(spec, st)] == [h for h, _ in scores_sections(spec, st)]`, which catches a refactor that drops the held-out section.

- [ ] **Step 2: Run the tests to verify they fail**

  Run: `.venv/bin/python -m pytest tests/test_report.py tests/test_package.py -q`
  Expected: collection error `ModuleNotFoundError: No module named 'talos.report'` and `ImportError` for `score_pairs`.

- [ ] **Step 3: Implement the Interfaces above in `talos/report.py`, and `score_pairs` in `talos/package.py`**

  The module docstring says what the module does and that the data functions are pure.

- [ ] **Step 4: Run the tests to verify they pass**

  Run: `.venv/bin/python -m pytest tests/test_report.py tests/test_package.py -q`
  Expected: all pass, and the existing `test_package.py` tests still pass unchanged.

- [ ] **Step 5: Mutation-check**

  Apply each of the following, run `tests/test_report.py` with `PYTHONDONTWRITEBYTECODE=1`, confirm a named test fails, then restore. (Memory: stale `.pyc` gives false MISSED; isolate the bytecode cache.)
  1. `parent = 0` always.
  2. Use `state.nodes` only when it is non-empty.
  3. `kind = "best" if node.best`.
  4. Diff every node against `sources(0)`.
  5. `per_node` ignores `total`.
  6. `axis_for` never cuts.

- [ ] **Step 6: Commit**

  ```bash
  git add talos/report.py talos/package.py tests/test_report.py tests/test_package.py
  git commit -m "report: node, diff, axis and held-out data for report.html"
  ```

### Task 2: `render_report` (the HTML page)

**Files:**
- Modify: `talos/report.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes the Task 1 functions and types, `talos.package.score_pairs`, `talos.package._diff`, `talos.scoring.bundle_delta` / `ScoringError`, `talos.budget.effective_budget`, and `talos.cli._llm_spend_text(provider, model, llm_usd, why=True)`, imported inside the function to keep `talos.report` light to import.
- Produces: `render_report(spec: JobSpec, state: JobState, timeline: list[dict], sources: Callable[[int], dict[str, str] | None]) -> str`.

Decisions the tests depend on:
- Each node is written as exactly `<g data-node="<id>" class="node k-<kind>[ cut][ strip]">`, with the attributes in that order. `cut` marks a point beyond a cut. `strip` marks a node on the failure strip, which holds every node with `kind == "failed"`. The root is `<g data-root class="root">` with no `data-node`. A held-out point carries `data-holdout="<id>"`. An edge is `<line data-edge="<parent>-<child>" …>`. Hypothesis-list rows use `data-goto="<id>"`, never `data-node`.
- Edges are drawn only when the parent is 0 or a node in the list. Otherwise that node's JSON object has `parent_note: "parent #N not recorded"`, which the panel shows.
- Node data for the panel and toggle: one `<script type="application/json" id="nodes">` holding a list of objects (the `ReportNode` fields plus `diff`). The `\u003c` escape is applied to the whole serialized string.
- Sections follow the spec's order. With `state.baseline is None` only the header is rendered. With `state.best is None` sections 4 and 5 are replaced by the text `No candidate beat the baseline`.
- The per-track tables use `bundle_delta(b, c).tracks` for each `score_pairs` entry, giving the columns track, baseline mean, best mean, rel delta, errors, n. A `ScoringError` gives one row with the error text.
- Times are formatted with `datetime.fromtimestamp(ts, timezone.utc)`. That reads no clock: the timestamps come from the inputs. With no timeline the times show `n/a`. Elapsed is last ts minus first ts. Working hours is last ts minus `state.spend.started_at`, as the spec's header section says.
- The JS (one inline `<script>`) handles the toggle (outcome, chain or family, shown only when the run has more than one value), the tooltip, the click → panel, and the hypothesis-list click. No `eval`, no `innerHTML` with node text: use `textContent`.

- [ ] **Step 1: Write the failing tests**

  Use a `rendered(tmp_path, **make_run_kwargs) -> str` helper that calls `render_report(spec, st, read_timeline(...), run_sources(...))`.

  ```python
  def test_every_node_drawn_once(tmp_path):
      # 7 hypotheses incl. a failed compile; mutation: dropping a node, off-by-one
      html = rendered(...)
      ids = re.findall(r'data-node="(\d+)"', html)
      assert sorted(map(int, ids)) == [1, 2, 3, 4, 5, 6, 7]

  def test_no_external_urls(tmp_path):
      # mutation: adding a CDN script or font
      html = rendered(...)
      assert not re.search(r"https?://|@import|(?:src|href)=\"(?!#)", html)

  def test_no_rand_hash(tmp_path):
      # mutation: spec.to_dict() instead of redacted()
      assert HASH not in rendered(...)

  def test_llm_text_is_escaped(tmp_path):
      evil = "</script><script>alert(1)</script>"
      # used as a title, a description, and a line in iterations/0001/mod.rs
      html = rendered(...)
      assert evil not in html and "<script>alert(1)" not in html

  def test_holdout_diamond_at_iteration(tmp_path):
      # timeline has a false_positive event for iteration 4 with holdout mean -0.01
      assert 'data-holdout="4"' in rendered(...)

  def test_failed_node_on_strip_and_outlier_cut(tmp_path):
      # 10 scores near +0.02, node 11 at -1.0, node 12 failed:compile
      html = rendered(...)
      cls = dict(re.findall(r'<g data-node="(\d+)" class="([^"]*)"', html))
      assert "cut" in cls["11"].split() and "cut" not in cls["5"].split()
      assert {"k-failed", "strip"} <= set(cls["12"].split()) and "strip" not in cls["5"].split()

  def test_unrecorded_parent_has_no_edge(tmp_path):               # Review Focus 4
      # nodes 1 (against 0), 2 (against 1), 3 (against 99, no node 99)
      # mutation: drawing edges to unrecorded parents, or drawing none at all
      html = rendered(...)
      assert "parent #99 not recorded" in html
      assert 'data-edge="0-1"' in html and 'data-edge="1-2"' in html
      assert 'data-edge="99-3"' not in html

  def test_deterministic(tmp_path):
      a = rendered(...); b = rendered(...)   # same run dir
      assert a == b                          # catches random ids
      # catches a "generated at <now>" stamp: fixture timestamps are from 2001
      assert datetime.now(timezone.utc).strftime("%Y-%m-%d") not in a

  def test_no_best_and_no_baseline_branches(tmp_path):
      no_best = rendered(..., best=None)
      assert "No candidate beat the baseline" in no_best and 'data-node="1"' in no_best
      no_base = rendered(..., baseline=None, best=None)
      assert 'data-node=' not in no_base and "knapsack" in no_base

  def test_200_node_report_under_2mb(tmp_path):
      # 200 hypotheses, iterations/NNNN/mod.rs with ~50 KB of changes vs parent each,
      # best = 200 with 300 KB of changes vs baseline
      # mutation: removing either cap
      html = rendered(...)
      assert len(html.encode("utf-8")) < 2_000_000
      # 200 node diffs plus the best-vs-baseline diff; removing the 400,000 best cap drops this to 200
      assert html.count("truncated (") == 201
  ```

  For `test_no_best_and_no_baseline_branches`, extend `make_run` with a `baseline=` argument (default: the Task 1 baseline).

- [ ] **Step 2: Run the tests to verify they fail**

  Run: `.venv/bin/python -m pytest tests/test_report.py -q -k "drawn or urls or hash or escaped or diamond or strip or unrecorded or deterministic or branches or 2mb"`
  Expected: `ImportError: cannot import name 'render_report'`.

- [ ] **Step 3: Implement `render_report` in `talos/report.py`**

  Build the SVG as strings from `axis_for` over the plotted scores. Map x from iteration `0..max id` and y from `[axis.lo, axis.hi]` into a `viewBox` with the failure strip below the x-axis. Draw the best-so-far step line from the first scored node: the running max of `score` in id order, over nodes whose kind is not `demoted` or `false_positive`. Colours come from CSS custom properties per `k-<kind>`, with a `prefers-color-scheme: dark` block.

- [ ] **Step 4: Run all report tests to verify they pass**

  Run: `.venv/bin/python -m pytest tests/test_report.py -q`
  Expected: all pass.

- [ ] **Step 5: Mutation-check**

  Same procedure as Task 1. The mutations:
  1. Skip the last node in the SVG loop.
  2. Show `spec.to_dict()`.
  3. Drop `html.escape` on the title.
  4. Drop the `\u003c` replacement.
  5. Ignore `holdout_points`.
  6. Remove the per-node cap.
  7. Draw edges to parents that are not recorded.

- [ ] **Step 6: Commit**

  ```bash
  git add talos/report.py tests/test_report.py
  git commit -m "report: render report.html with search figure, node panel, tables and diffs"
  ```

### Task 3: `write_report`, `build_package` hook, `talos report` CLI, docs

**Files:**
- Modify: `talos/report.py` (add `write_report`)
- Modify: `talos/package.py:283-330` (`build_package`)
- Modify: `talos/cli.py` (new `cmd_report` and subparser beside `status`, around lines 1265 and 1272)
- Modify: `README.md`. Add a row for `talos report` to the command table near line 472, a `### talos report` section after `### talos status` near line 581, and a `report.html` row in the package files table near line 646.
- Modify: `docs/architecture.md` only if it lists the package files. Check with `grep -n "lineage.md\|scores.md" docs/architecture.md`, and add `report.html` where they are listed.
- Test: `tests/test_report.py`, `tests/test_package.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes `render_report`, `read_timeline` and `run_sources` from Tasks 1 and 2.
- Produces:
  - `write_report(store: JobStore, spec: JobSpec | None = None, state: JobState | None = None) -> Path`. It writes `store.run_dir / "package" / "report.html"` (`mkdir(exist_ok=True)`) and returns that path.
  - `cmd_report(args, ask) -> int` in `talos/cli.py`. `args.job_id` is resolved under `Path.cwd() / "runs"`, as `cmd_status` does.

Rules:
- `build_package` calls the private `_write_report_safely(store, spec, state)` immediately before each of its three `shutil.make_archive` calls. That function wraps `write_report` in `try/except Exception as e`, printing `report.html not written: {e}` to stderr.
- `cmd_report` checks `(Path.cwd() / "runs" / job_id / "state.json").is_file()` **before** constructing `JobStore`, because `JobStore.__init__` creates the directory. On failure it prints `no job {job_id} under runs/` (or `job {job_id} has no state.json`) to stderr and returns 1. On success it prints `Report: <path>` and returns 0.
- The subparser is `rep = sub.add_parser("report", help="write runs/<job_id>/package/report.html from the run's files")` with `rep.add_argument("job_id")`.

- [ ] **Step 1: Write the failing tests**

  ```python
  # tests/test_package.py
  def test_package_zip_holds_report_with_best(tmp_path):
      spec, st, store = make(tmp_path)
      build_package(spec, st, store)
      assert (tmp_path / "package" / "report.html").is_file()
      assert "report.html" in zipfile.ZipFile(tmp_path / "package.zip").namelist()

  def test_package_zip_holds_report_with_no_candidate(tmp_path):
      spec, st, store = make(tmp_path); st.best = None
      build_package(spec, st, store)
      assert "report.html" in zipfile.ZipFile(tmp_path / "package.zip").namelist()

  def test_report_failure_does_not_stop_package(tmp_path, monkeypatch, capsys):
      spec, st, store = make(tmp_path)
      monkeypatch.setattr("talos.report.render_report", lambda *a: 1 / 0)
      build_package(spec, st, store)
      assert (tmp_path / "package.zip").is_file()
      assert "report.html not written" in capsys.readouterr().err

  # tests/test_report.py
  def test_write_report_with_no_timeline(tmp_path):                # Review Focus 5
      spec, st, store = make_run(tmp_path, ..., timeline=None)   # no timeline.jsonl at all
      text = write_report(store).read_text(encoding="utf-8")
      assert "<th>Started</th><td>n/a</td>" in text and "<th>Elapsed</th><td>n/a</td>" in text
      assert "data-holdout" not in text

  # tests/test_cli.py  (monkeypatch.chdir(tmp_path); make_run(tmp_path / "runs" / "J", ...)
  # from tests.run_fixture; main is talos.cli.main)
  def test_talos_report_writes_and_prints_path(tmp_path, monkeypatch, capsys):
      assert main(["report", "J"]) == 0
      assert (tmp_path / "runs/J/package/report.html").is_file()
      assert "report.html" in capsys.readouterr().out

  def test_talos_report_unknown_job_leaves_no_dir(tmp_path, monkeypatch):
      monkeypatch.chdir(tmp_path)
      assert main(["report", "NOPE"]) == 1
      assert not (tmp_path / "runs" / "NOPE").exists()
  ```

  Check the mock-only test (`test_report_failure_does_not_stop_package`): it must patch the name `build_package` actually reaches. Patch `talos.report.render_report`, and have `write_report` call `render_report` through the module global, not an alias bound at import. Otherwise the test passes against a missing try/except.

- [ ] **Step 2: Run the tests to verify they fail**

  Run: `.venv/bin/python -m pytest tests/test_report.py tests/test_package.py tests/test_cli.py -q -k "report"`
  Expected: failures, either `report.html` missing or `invalid choice: 'report'`.

- [ ] **Step 3: Implement `write_report`, the `build_package` hook, `cmd_report` and the subparser, then the README and architecture doc rows**

- [ ] **Step 4: Run the full check**

  Run: `make check PYTHON=.venv/bin/python > /tmp/claude-1000/-home-fibonadithya-TIG-Talos/2a5dd1b3-6d32-4d01-9a6b-64a5dc5d0075/scratchpad/check.log 2>&1; tail -20 /tmp/claude-1000/-home-fibonadithya-TIG-Talos/2a5dd1b3-6d32-4d01-9a6b-64a5dc5d0075/scratchpad/check.log`. This is a full suite, so run it in the background with a generous timeout.
  Expected: ruff clean, pytest `N passed` with no failures, agentify check clean. Report the change in test count from Task 1 Step 0.

- [ ] **Step 5: Mutation-check**

  1. Remove the try/except.
  2. Call the writer only in the best branch.
  3. Construct `JobStore` before the existence check.

  For each, confirm a named test fails, then restore.

- [ ] **Step 6: Commit**

  ```bash
  git add talos/report.py talos/package.py talos/cli.py README.md tests/test_report.py tests/test_package.py tests/test_cli.py
  # plus docs/architecture.md if changed
  git commit -m "report: write report.html with every package and add talos report"
  ```

### Task 4: Check the report on the local runs

This task is for the orchestrator, not a subagent. Nothing is committed unless a bug turns up; a fix goes back through TDD in the owning task.

- [ ] **Step 1: Generate a report for each local run**

  Run `for j in $(ls runs); do .venv/bin/python -m talos.cli report "$j"; done`. All 7 should print `Report:`. Note that `20260925-103339-hypergraph` has no `package/` until now.

- [ ] **Step 2: Render screenshots**

  Run `firefox --headless --no-remote --profile "$(mktemp -d)" --screenshot <scratchpad>/<job>.png --window-size=1400,2400 file://$PWD/runs/<job>/package/report.html` for the 8-iteration knapsack run (20260923-092930) and one hypergraph run (no baseline). Then do the same for a synthetic 200-node anneal fixture with two chains, a template start and outliers, written to the scratchpad by the Task 2 test helper.

- [ ] **Step 3: Look at each screenshot**

  Check that the figure renders, the axes are scaled with the points inside, the 0 line is visible, failures sit on the strip, and there is one star. Check the per-track table, and the file size against 2 MB with `ls -l`.

- [ ] **Step 4: Check interactivity**

  Open the knapsack report in a real browser and click a node to see its panel and diff. Report the results to the user with the screenshots.
