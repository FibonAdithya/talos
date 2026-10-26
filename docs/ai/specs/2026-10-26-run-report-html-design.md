# Run report: one self-contained report.html per job (issue #67)

## Intent

After a job, the user reads `timeline.jsonl`, `state.json`, `scores.md`, `lineage.md` and
`hypotheses.md` one by one to see what the search did. No single view shows the search.
AIDE (`tree_plot.html`) and AI Scientist v2 (`unified_tree_viz.html`) each write one
self-contained HTML file per run. This adds the same to Talos: `runs/<job_id>/package/report.html`,
built only from files already in the run directory, so it also works on old runs.

## Goals

- One HTML file with inline CSS, JS and SVG. No CDN, no fonts, no network. It opens offline.
- Written whenever `talos/package.py::build_package` runs, on every branch of it (no candidate,
  no baseline, a best), and on demand by a new `talos report JOB_ID` subcommand.
- Works on runs from before the lineage tree existed (no `state.nodes`) and on runs still in
  progress.
- Under 2 MB for a 200-iteration job.

## Non-goals

- Not rewritten after every iteration. The user chose end of run plus `talos report`; the loop is
  not touched. (A running job can be inspected with `talos report` at any time.)
- No noise band. Issue #69 adds the per-candidate standard error; once it is in the run data,
  the report draws it. This issue leaves a place for it and nothing more.
- No `talos status/show/diff --json` (issue #68). Only the `report` subcommand from that issue is
  built here.
- No change to scoring, the loop, the beat rule, or any file the package already writes.

## Anti-goals

- The report must never contain the job's `rand_hash` (AGENTS.md invariant 2). Settings come from
  `JobSpec.redacted()`, never `to_dict()`. A test greps the file for the hash.
- LLM-written text (titles, descriptions, errors, diffs) must never be interpreted as HTML or
  script. All of it is escaped. Embedded JSON is placed in `<script type="application/json">`
  with every `<` replaced by the six-character JSON escape (backslash, `u`, `0`, `0`, `3`, `c`),
  so a `</script>` in a diff cannot end the block. MEASURED by the builder check (2026-10-26):
  on run 20260923's 63,408-character best-vs-baseline diff this adds 3.4%.
- A failure to write the report must not fail the package. `build_package` catches the
  exception, prints one line to stderr, and carries on; the zip is still written.

## Inputs

All read from `runs/<job_id>/`:

| source | used for |
|---|---|
| `job.json` via `JobStore.read_spec()` then `.redacted()` | header settings, challenge, tracks, budget |
| `state.json` via `JobStore.load()` | status, stop reason, spend, best, baseline, nodes, hypotheses, confirmed, false_positives, validations, budget extensions |
| `timeline.jsonl` | start and end time (first and last `ts`); held-out deltas from `won` and `false_positive` events; tolerate unreadable lines by skipping them |
| `iterations/NNNN/<file>` | each node's sources, for the diff against its parent |
| `challenge/template.rs` | the root's source in a `--start template` job |

A run with no `state.json` (killed before the first save) gets no report; `talos report` says so
and returns 1.

### Nodes, including old and partly old runs

Nodes are merged by iteration (my call). For each hypothesis record, use the stored
`state.nodes` entry with the same id if there is one. Otherwise build one from the record:
`id = iteration`, `parent = against`, `outcome = outcome`, `score = mean_rel_delta` (None if
absent), chain 0, and family from `design_family` or `strategy_tag`. This covers runs from before
the lineage tree, which have no nodes, and runs started before it and resumed under current code,
which have nodes only for the later iterations (`talos/loop.py::Loop._chain` continues chain 0
from the old best). Titles and descriptions always come from the hypothesis record.

Display status is derived from the job's lists, not from `node.best` or `node.outcome` alone:

- **Best (the star):** only `state.best.iteration`. `node.best` is set when a node is added and
  never cleared, so every former best keeps `best=True` (`talos/loop.py::Loop._add_node`).
  Former bests can get a smaller ring; the star is only for the current best.
- **Demoted:** iterations with `outcome == "demoted"` in `state.validations`. Their node outcome
  is `failed:validation`, and they keep a native score, so they are plotted at that score.
- **False positive:** iterations in `state.false_positives`. Their outcome stays `improved`.
- **Confirmed:** iterations in `state.confirmed`.

### Node sources and diffs

The report has its own lookup, which tolerates missing files; `Loop._node_files` raises when a
file is missing, so it can't be reused. The root (node 0) is `state.baseline.files`, or
`challenge/template.rs` as `mod.rs` for a template start. The current best is
`state.best.files`. Any other node is whatever files from the root's file names exist in
`iterations/NNNN/`. A node with none of them has no diff: edit and compile failures write only
`hypothesis.json`, or nothing at all. Its panel says "no files saved" and gives the outcome. A
node's parent with no saved files gives "parent's files not saved" in place of the diff.

Diffs are `difflib.unified_diff` per file, as `talos/package.py::_diff` already produces; reuse it.

## Size limits (builder's call, set here)

- Per-node diffs share a 1,000,000-character budget, split evenly across the nodes that have one,
  each capped at 64,000 characters. An 8-node run gets 64 KB per node; a 200-node run about
  5 KB. A cut diff ends with a line `… truncated (N more lines); full diff: diff -u <parent dir>
  <node dir>`, giving the two iteration directories.
- The best-vs-baseline (or vs-template) diff is capped at 400,000 characters the same way.
- The test builds a 200-node fixture with large diffs and asserts the file is under 2,000,000
  bytes.

## Page contents, top to bottom

1. **Header.** Challenge and id, status and stop reason, best training delta and which
   iteration, held-out result (confirmed / false positive / not scored), spend (LLM USD, with
   "not tracked" for codex as the run summary already prints it, `talos/cli.py` near line 490;
   compute USD; iterations), two time figures, both labelled (my call), the started and
   last-event time (UTC), and settings. The time figures are "elapsed": last `ts` minus first
   `ts` in the timeline. And "working hours": last `ts` minus `state.spend.started_at`, which is
   the measure the hours budget uses, since a resume shifts `started_at` forward by the pause.
   The hours budget is shown next to "working hours". The settings are provider, model, mode, search, chains,
   start, scoring (requested and actual from `state.scoring`), track focus, fuel, tracks,
   starting budget and the effective budget after extensions.

2. **Search figure (score plot and lineage tree in one SVG).** The user chose this over a separate
   node-link tree.
   - x is the iteration number, y is the node's training delta vs baseline (`rank_score()`:
     metered if validated, else research score). A horizontal reference line at y = 0 marks the
     mainnet baseline, in template jobs too, since their scores are also measured against it.
     On a mainnet start the root sits at (0, 0). In a template job the root is unscored and is
     drawn at x = 0 on the failure strip, labelled "template", with its edges drawn from there.
   - A line joins each node to its parent. Lines into accepted or improved nodes are solid;
     lines into rejected ones are faint.
   - Point marker and colour by outcome: improved / won (green), accepted (blue), failed:score
     (grey), demoted (orange), false positive (orange ring). Nodes with no score
     (failed:edit, failed:compile, failed:dead_code, failed:runtime with no delta, started)
     are drawn as red crosses on a strip below the x-axis.
   - The current best gets a star (see "Display status" above). A step line shows the
     best-so-far over iterations. It starts at the first scored node, never at minus infinity.
   - Held-out deltas from `won` / `false_positive` events that carry a `holdout` dict are drawn
     as hollow diamonds at the same x. A `false_positive` event with only `error` (held-out not
     scored) adds no diamond; its node is still marked as a false positive.
   - y range: from the lowest to the highest score shown, with 0 always included. The range
     ignores scores beyond a cut (my call; it works the same whether the scores are all
     negative, all positive or mixed). With at least 8 scored nodes, let q1 and q3 be their
     quartiles and IQR = q3 - q1. The cut is at `q1 - 3 * max(IQR, 0.01)` and
     `q3 + 3 * max(IQR, 0.01)`. Points beyond it are drawn as triangles on the top or bottom
     edge, pointing outward, with the value in the tooltip. With fewer than 8 scored nodes there
     is no cut. Without the cut, one score of -100% would squash every other point onto one
     line.
   - A toggle above the figure recolours points by outcome (default), chain, or design family.
     Chain and family are offered only when the run has more than one value.
   - Hover shows `#id title (delta)`; click opens the node panel.
   - Width is the page width, so 200 nodes do not scroll sideways. The SVG has a `viewBox` and
     scales.

3. **Node panel.** Opens on click, beside the figure on wide screens and below it on narrow ones.
   It shows the id, parent, chain, family, strategy tag, outcome, error (if any), training delta,
   per-track deltas when `node.tracks` is present, metered score if validated, acceptance
   probability, draw and temperature when annealing, the hypothesis title and description, and
   the diff against the parent, coloured by line (+ / - / @@) in a scrolling `<pre>`.

4. **Per-track table.** Baseline vs best, one row per track: baseline mean quality, best mean
   quality, relative delta, errors, n. One table for training and one for held-out when the best
   has held-out results. With `--track`, the same split as `scores_sections`: focus track
   training, focus track held-out, regression guard. Compute the rows with `bundle_delta` and the
   `focus_sets` / `select` helpers already in `talos/scoring.py`.

5. **Best diff.** The best vs baseline (or vs template), with the same colouring, in a collapsed
   `<details>`.

6. **Hypotheses list.** Every record in iteration order, one line each, as in `hypotheses.md`;
   clicking a line opens its node panel.

A run with no best shows the header, the figure, and the hypotheses list, with "No candidate beat
the baseline" in place of sections 4 and 5. A run with no baseline shows the header only.

## Code layout (builder's call)

- New module `talos/report.py`:
  - `render_report(spec, state, timeline, sources) -> str`. Pure: takes the parsed timeline rows
    and a `sources(node_id) -> dict[str, str] | None` callable, does no I/O, and reads no clock.
    Same inputs give byte-identical output.
  - `write_report(store: JobStore, spec: JobSpec | None = None, state: JobState | None = None)
    -> Path`. Reads `job.json` / `state.json` only when `spec` / `state` are not given, reads the
    timeline and sources, calls `render_report`, writes `run_dir/package/report.html` (creating
    `package/` if absent), returns the path. `render_report` takes the full `JobSpec`, because the
    scoring helpers need its `NonceSet`s, but shows only fields from `spec.redacted()`.
- `build_package` calls `write_report(store, spec, state)` just before each
  `shutil.make_archive` call, so the zip holds it. The try/except described under anti-goals
  wraps the call.
- `talos report JOB_ID` in `talos/cli.py`, with help text: loads the job, calls `write_report`,
  prints the path. It needs no credentials or config. It checks that `runs/JOB_ID/state.json`
  exists **before** building a `JobStore`, because `JobStore.__init__` creates the directory. An
  unknown id must not leave a `runs/JOB_ID/` behind. Unknown job id or missing `state.json`:
  message to stderr, exit 1. Apart from `package/report.html`, it writes nothing to the run.
- The SVG is built in Python (string building, no plotting library). The JS is limited to the
  toggle, tooltip and node panel, and reads node data from the embedded JSON block.
- Styling: one inline `<style>`; light and dark through `prefers-color-scheme`; system fonts.

## Tests

In `tests/test_report.py`, against fixture run directories built in `tmp_path` by a helper that
writes `job.json`, `state.json`, `timeline.jsonl` and iteration directories:

| test | the mutation it catches |
|---|---|
| every node id in the fixture appears in the HTML as a node element (`data-node="N"`), and the count of such elements equals the node count | dropping a node from the figure, off-by-one in the node loop |
| the HTML contains no `http://`, `https://`, `src=` pointing off-page, or `@import` (fixture text has no URLs) | adding a CDN script or font |
| the job's `rand_hash` (a distinctive value set in the fixture) appears nowhere in the file | using `spec.to_dict()` instead of `redacted()` |
| a hypothesis title of `</script><script>alert(1)</script>` appears only escaped | missing escaping in the JSON block or in the HTML |
| an old-format run (no `nodes`) yields nodes from hypotheses with the right parents | ignoring `against`, or requiring `state.nodes` |
| a resumed old run (nodes for iterations 4-6 only, hypotheses 1-6) shows all six nodes, with stored fields for 4-6 | using `state.nodes` alone when it is non-empty |
| a run where nodes 2 and 5 both have `best=True` and `state.best.iteration == 5` draws exactly one star, on 5 | trusting `node.best` |
| a demoted iteration (in `state.validations`, outcome `failed:validation`) and a false positive (in `state.false_positives`, outcome `improved`) get their own markers | colouring by `outcome` only |
| a template-start fixture with every score between -0.6 and -0.3 gives a y range that includes those points and 0, with no point cut | the cut rule breaking on all-negative scores |
| `talos report NOPE` exits 1 and `runs/NOPE` does not exist afterwards | building a `JobStore` before the existence check |
| a node with a failed compile has no diff and is drawn on the failure strip | reading files that are not there |
| a node's diff contains a line changed between its parent's and its own iteration directory, and not a line changed only in a sibling | diffing against the baseline or the best instead of the parent |
| 200-node fixture with 50 KB changes per node is under 2,000,000 bytes and each truncated diff ends with the truncation line | removing the budget |
| a score of -1.0 among 10 scores near +0.02 produces a cut axis and a down-triangle for that node | removing the cut |
| a held-out delta from a `false_positive` event shows up as a diamond at its iteration | ignoring timeline held-out data |
| `render_report` called twice on the same input gives identical strings | wall-clock or random use in rendering |
| `build_package` writes `package/report.html` and the zip contains it, on the no-candidate branch and on the best branch | calling the writer on only one branch |
| a writer that raises does not stop `build_package` from writing the zip | removing the try/except |
| `talos report JOB_ID` exits 0 and prints the path; an unknown id exits 1 | CLI wiring |

Expected values in these tests come from the fixture (ids, titles, hash, file lines), not from
`render_report` itself.

## Done when

- The tests above pass under `make check`.
- The builder has generated the report on the local runs in `runs/` (knapsack, hypergraph, all
  without lineage nodes) and opened each in a browser to check that the figure renders and the
  axes are scaled.
- A maintainer has opened it on a real long run (the 200-iteration energy_arbitrage job
  20261022-125434 is not in this clone; it has to be checked where that run lives).

## Left to the builder

Exact colours, marker sizes, page CSS, tooltip behaviour, the wording of headings, and how the
SVG strings are assembled.
