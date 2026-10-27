# Inspecting runs from the CLI: `talos status`, `show`, `diff` (issue #68)

## Intent

`talos status` prints one line per job. To see a job's best candidate, an iteration's
hypothesis or a diff, the user (or a coding agent driving Talos) opens files under
`runs/<job_id>/` and has to know that layout. Issue #68 asks for commands that answer those
questions directly, with a `--json` form whose schema is documented and versioned, so an agent
can rely on it. Weco's `weco run status|show|diff --step best` is the prior art the issue cites.

`talos report JOB_ID` (the fourth command in the issue) already exists (PR #80) and is not
changed here.

## Goals

- `talos status` without an id: one table row per job under `runs/`.
- `talos status JOB_ID`: that job's detail, including budget left and current activity.
- `talos show JOB_ID [--iteration N|best]`: one iteration's hypothesis, outcome, scores and
  file paths.
- `talos diff JOB_ID [--iteration N|best] [--against parent|baseline]`: a unified diff on
  stdout.
- `--json` on `status` (both forms) and `show`, with a `schema_version` field and a schema
  documented in `docs/json-output.md`.
- All three commands are read-only and offline. They read the run directory and nothing
  else: no config, no credentials, no network, no Docker.

## Non-goals

- No change to the research loop or to what a run writes. Per-track training deltas are
  already saved for every iteration that scored (`state.nodes[*].tracks`, written by
  `Loop._add_node` for greedy and anneal jobs alike). Runs from before the lineage tree
  (September 2026) have no nodes; for those, `show` reports what the hypothesis record holds
  (mean delta, worst track and its delta) and `null` per-track scores.
- No `--json` on `diff`: its output is a patch.
- No liveness check. Talos writes no pid or lock file, so a job whose status is `researching`
  may belong to a process that died. The commands report the status and the time of the last
  timeline event; they do not claim the process is running.
- No change to `talos report`.

## Anti-goals

- A command that writes anything. `JobStore(run_dir)` creates the directory in its
  constructor, so these commands must not construct one; they read `job.json`, `state.json`
  and `timeline.jsonl` with plain file reads (the way `cmd_report` checks the directory before
  touching `JobStore`).
- One unreadable run directory stopping `talos status` from listing the others.
- A JSON key that disappears or changes meaning without `schema_version` changing.

## Data sources (all already on disk)

| What | Where |
|---|---|
| job settings, original budget | `runs/<id>/job.json` (`JobSpec`) |
| status, iteration, spend, best, baseline, hypotheses, nodes, validations, budget extensions, pending job | `runs/<id>/state.json` (`JobState`) |
| events with timestamps | `runs/<id>/timeline.jsonl` (`talos.report.read_timeline` already tolerates bad lines and a missing file) |
| an iteration's files | `runs/<id>/iterations/NNNN/` (only for iterations that compiled; `talos.report.run_sources` already resolves files per node id, root included) |
| one node per iteration, with parent, outcome, score, per-track deltas, kind | `talos.report.report_nodes(spec, state)` |
| best vs baseline per-track tables, training and held-out | `talos.package.score_pairs` + `talos.scoring.bundle_delta` (as `report._track_tables` uses them) |
| held-out mean delta per confirmed iteration | `talos.report.holdout_points(timeline)` |
| effective budget | `talos.budget.effective_budget(spec.budget, state.budget_extensions)` |
| LLM spend is a measurement or not | `talos.cli.unpriced_reason(provider, model)` |

The new code reuses these rather than re-deriving the run layout.

## Commands

### `talos status` (list)

A fixed-width table, one row per directory under `runs/` that holds a `job.json` or a
`state.json`, sorted by job id:

```
JOB                              CHALLENGE        STATUS       IT  BEST      LLM       COMPUTE  LAST EVENT
20261026-120733-job_scheduling   job_scheduling   exhausted     6  +0.797%   $1.84     $12.40   2026-10-26 21:14 UTC
20260925-103339-hypergraph       hypergraph       failed        0  n/a       unpriced  $0.31    2026-09-25 10:52 UTC
```

(values above are illustrative, not measured.)

- BEST is `state.best.delta["mean_rel_delta"]` or `n/a`.
- LLM uses `_llm_spend_text`, so codex-cli shows `not tracked` and an unpriced model shows
  `unpriced`, as today.
- LAST EVENT is the `ts` of the last readable timeline row, in UTC; `-` when there is none.
- A directory whose `state.json` or `job.json` is missing, or cannot be read or parsed,
  gets a row with status `unreadable` and `-` in the other columns; the reason goes to stderr
  (and to the row's `error` in JSON). A job that crashed before its first `state.json` save
  is therefore listed, not hidden.
- Exit code is always 0. With no jobs (an empty or missing `runs/`) the human form prints
  `no jobs under runs/`; `--json` prints `{"schema_version": 1, "jobs": []}`.

This replaces today's `<id>: <status> it=N llm=... compute=...` line. README's
`### talos status` section is updated to match.

### `talos status JOB_ID`

Human form: a key/value block.

- job: id, challenge, track focus, provider/model, mode, search, scoring (requested and
  actual), start, created time
- status, stop reason, iteration, iterations since improvement
- best: iteration, title, mean training delta, held-out status and delta (below); or `none`
- budget: for each of iterations, hours, LLM USD, compute USD — cap (effective, after
  extensions), spent, left; `∞` for a cap that is None; LLM shows unpriced as in status
- current activity: the pending job's purpose (`baseline`, an iteration number, …) when
  `state.pending_job` is set; the last timeline event's kind, time and one-line text (from
  `_event_line` where it returns text, else the kind)
- hardware

**Held-out status** of a candidate, as `report._header` already decides it:
`confirmed` (iteration in `state.confirmed`), `false_positive` (in `state.false_positives`;
`state.best` is not cleared when its held-out run fails, so the best can be one),
`not_scored` otherwise. The held-out mean delta is the value from the timeline's `won` /
`false_positive` event (`holdout_points`) whether it passed or failed, and null when there is
none (including a false positive whose held-out run could not be scored).

**Hours spent** (my call): for a job whose status is terminal (`won`, `exhausted`, `failed`,
`cancelled`) or `paused`, the last timeline event's `ts` minus `spend.started_at`, as
`report._header` measures working hours; for any other status, now minus `started_at`, as
the live status line does. Both forms also give `last_event_ts`, so a reader can see when a
`researching` job has gone quiet. No timeline rows: spent is null.

**Unpriced LLM spend** (my call, from `unpriced_reason`'s rule that it is never reported as
a measured $0.00): when the provider/model is unpriced, `llm_usd`, `budget.usd.spent` and
`budget.usd.left` are null in JSON and the human form says `unpriced` / `not tracked`.

Unknown id: `no job <id> under runs/` on stderr, exit 1 (same text as `talos report`).

### `talos show JOB_ID [--iteration N|best]`

`--iteration` defaults to `best`. Resolves to one node from `report_nodes`.

Human form:

- iteration, parent (`baseline`/`template` or `#N`), chain, kind (`best`, `improved`,
  `rejected`, `failed`, …), outcome, error
- hypothesis: title, strategy tag, design family, description (full, wrapped)
- scores: mean training delta, worst track and its delta, runtime ratio when recorded;
  per-track training deltas from the node (`null`/omitted for pre-lineage runs); metered
  score when a native job validated it; held-out status and mean delta (see `status
  JOB_ID`). The training mean comes from the hypothesis record's `mean_rel_delta` (or
  `Node.score`), not `ReportNode.score`, which is the metered score when one exists
- for the best only: the per-track tables from `score_pairs` (baseline mean, best mean,
  rel delta, errors, n), training and held-out
- files: paths relative to the current directory of each file in `iterations/NNNN/` (or
  `best/` for the best when that directory exists), plus `hypothesis.json`

Errors, exit 1 with a message on stderr:

- `--iteration best` (or the default) on a job with no best:
  `job <id> has no best candidate; use --iteration N` (with `--json` too; the message is not
  JSON).
- `--iteration N` with no hypothesis record N: `job <id> has no iteration N (it has 1..M)`.
  An iteration still in flight has no record yet, so this is also what it gets.

### `talos diff JOB_ID [--iteration N|best] [--against parent|baseline]`

`--iteration` defaults to `best`; `--against` defaults to `baseline` (the root: the mainnet
baseline's files, or the template for a `--start template` job). `parent` is the node's
parent from the lineage; for a node whose parent is the root the two are the same.

Prints the unified diff from `talos.package._diff` to stdout, untruncated, labelled with
the base (`baseline/`, `template/`, `#N/`) and `candidate/`. An empty diff prints nothing and
exits 0.

Errors, exit 1 with a message on stderr: no best (as `show`); no iteration N; the
iteration's files were not saved (edit or compile failure:
`iteration N saved no files (outcome failed:compile)`); the base's files were not saved.

## JSON output

`--json` prints one JSON object to stdout and nothing else on stdout. Every object has
`"schema_version": 1` as a top-level key. Floats are raw (a delta of +0.797% is `0.00797`),
times are Unix seconds (float), absent values are `null` rather than missing keys.

- `status --json`: `{"schema_version", "jobs": [ {job_id, challenge, status, iteration,
  best_mean_rel_delta, llm_usd, llm_priced, llm_unpriced_reason, compute_usd,
  last_event_ts, error} ... ]}` — `error` is null unless the row is `unreadable`;
  `llm_usd` is null when unpriced.
- `status JOB_ID --json`: `{"schema_version", "job": {...spec fields minus rand_hash and
  nonce seeds, as JobSpec.redacted()...}, "status", "stop_reason", "iteration",
  "runs_since_improvement", "best": {iteration, title, mean_rel_delta, holdout_status,
  holdout_mean_rel_delta} | null, "budget": {iterations|hours|usd|compute_usd: {cap, spent,
  left}}, "llm_priced", "llm_unpriced_reason", "last_event_ts", "activity": {pending: {purpose,
  title} | null, last_event: {kind, ts, iteration, text} | null}, "hardware", "scoring":
  {requested, actual}}`.
- `show --json`: `{"schema_version", "job_id", "iteration", "parent", "chain", "kind",
  "outcome", "error", "hypothesis": {title, description, strategy_tag, design_family},
  "scores": {mean_rel_delta, worst_track, worst_rel_delta, runtime_ratio, metered_rel_delta,
  tracks: {track: rel_delta} | null, holdout_status, holdout_mean_rel_delta,
  tables: [...] | null},
  "files": [paths]}`.

The exact key list per command goes in `docs/json-output.md`, one table per command (key
path, type, meaning). A test parses those tables and fails if any documented key path is
missing from the command's output on the fixture runs. A new key is a compatible addition;
removing or renaming a key, or changing a value's meaning, bumps `schema_version`.

`docs/json-output.md` is a new authoritative doc (the issue suggested `docs/usage.md`; the
README is the usage doc in this repo, and a separate file keeps the schema tables parseable).
AGENTS.md's routing table gets a row for it, and README's command table links to it.

## Tests

Fixture runs come from `tests/run_fixture.make_run` (already used by the report tests),
extended where needed. Required cases, per the issue:

- a finished run with a best (greedy, nodes with per-track deltas)
- a run with no candidate (`best=None`)
- a run in progress (`status="researching"`, `pending_job` set, timeline ending mid-iteration)
- a pre-lineage run (no nodes) for `show`'s fallback
- a template-start run for `diff --against baseline` (root is the template)
- an unreadable run directory for `status` (corrupt `state.json`), and one with
  `job.json` but no `state.json`
- a best that is a held-out false positive
- an unpriced (codex-cli) job, for the null `llm_usd`
- an empty and a missing `runs/`

Plus: none of the commands creates a file or directory under `runs/` (compare the tree
before and after); the documented-keys test above; CLI parsing for the new flags.

## Left to the builder

- Module layout (a new `talos/inspect.py` for the pure data-gathering, with thin `cmd_*`
  wrappers in `cli.py`, is the suggestion, not a requirement).
- Column widths and wrapping in the human output.
- Exact wording of messages not quoted above.
