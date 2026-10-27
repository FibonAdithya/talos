# Inspecting runs (`talos status` / `show` / `diff`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `talos status [JOB_ID] [--json]`, `talos show JOB_ID [--iteration N|best] [--json]` and `talos diff JOB_ID [--iteration N|best] [--against parent|baseline]`, read-only over `runs/<job_id>/`, with a versioned, documented JSON schema (issue #68).

**Architecture:** A new module `talos/inspect_run.py` loads a run directory with plain file reads (never `JobStore`, whose constructor creates directories) and builds plain dicts, which are the `--json` output. Its human formatters render those same dicts, so the two forms cannot disagree. `talos/cli.py` gets thin `cmd_status` / `cmd_show` / `cmd_diff` wrappers. Node, file, diff and score logic is reused from `talos/report.py`, `talos/package.py` and `talos/scoring.py`.

**Tech Stack:** Python 3.11, argparse, pytest, ruff. No new dependencies.

**Spec:** `docs/ai/specs/2026-10-27-inspect-runs-design.md`

**Worktree:** `/home/fibonadithya/TIG/Talos/.claude/worktrees/issue-68-inspect-runs`, branch `inspect-runs` (fast-forwarded on 2026-10-27 to `origin/main` 9b07825, which includes PR #82's `talos status` changes). Run every command from there. Python: `/home/fibonadithya/TIG/Talos/.venv/bin/python` (written `$PY` below). From the worktree root it imports the worktree's `talos` (checked: `talos.__file__` resolves inside the worktree).

## Global Constraints

- Read-only and offline: no config load, no credentials, no network, no Docker, no `JobStore(...)`. Nothing is created or modified under `runs/`.
- `--json` writes exactly one JSON object to stdout and nothing else to stdout; every object has `"schema_version": 1` as a top-level key. Errors go to stderr as plain text, exit 1.
- JSON values: deltas are raw fractions (`0.00797`, not `0.797`), times are Unix seconds (float), an absent value is `null` and never a missing key.
- Unpriced LLM spend (`talos.cli.unpriced_reason(provider, model)` is not None) is `null` in JSON (`llm_usd`, `budget.usd.spent`, `budget.usd.left`) and reads `unpriced` / `not tracked` in human output via `talos.cli._llm_spend_text`. Never `0.0`.
- Messages fixed by the spec: `no job <id> under runs/`; `job <id> has no best candidate; use --iteration N`; `job <id> has no iteration N (it has 1..M)`; `iteration N saved no files (outcome <outcome>)`; `no jobs under runs/`.
- `talos diff` defaults: `--iteration best`, `--against baseline`. `talos show` default: `--iteration best`.
- `talos report` is not changed.
- A job's id is its directory name (`run.run_dir.name`), never `spec.job_id`. Every message and every `job_id` key uses it. (`tests/run_fixture.make_run` always writes `job_id="J"` into job.json whatever the directory is called, so tests under `runs/A`, `runs/K` would expose the wrong choice.)
- Line length and lint: `$PY -m ruff check .` must pass (the repo's pinned rule set).
- Git: stage explicit paths only (never `git add -A` / `.`); commit message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Tests that run CLI code must not reach any real backend: these commands never construct a bench or provider, and a test asserting that is in Task 3.

## Review Focus

1. A job id containing `/`, `\` or `..` (`talos show ../x`): expect `no job <id> under runs/`, and nothing outside `runs/` is read. Test: `test_load_run_rejects_path_like_ids` (Task 1).
2. A job that failed while measuring its baseline (`state.baseline is None`, no best, no nodes): `status JOB_ID` and `--json` must work, `show` gives the no-best error, not a traceback. Test: `test_detail_of_a_job_with_no_baseline` (Task 1).
3. A truncated or non-JSON last line in `timeline.jsonl` (a job killed mid-write): the last *readable* event is reported. Test: `test_last_event_skips_a_torn_line` (Task 1).
4. `--json` stdout stays parseable when the command fails: stdout is empty, the message is on stderr, exit 1. Test: `test_json_error_leaves_stdout_empty` (Task 3).
5. A best candidate whose `iterations/NNNN/` lost its source files: `diff` still works from `state.best.files` (that is what `report.run_sources` returns for the best). Test: `test_diff_best_uses_state_files_when_the_iteration_dir_is_empty` (Task 2).

---

## File structure

| File | Responsibility |
|---|---|
| `talos/inspect_run.py` (new) | `Run`, `InspectError`, `load_run`, status rows, job detail, iteration resolution, iteration detail, iteration diff, human formatters. Pure except the reads in `load_run` / `status_rows` / file listing. |
| `talos/cli.py` | `cmd_status` (rewritten), `cmd_show`, `cmd_diff`, parser entries, `parse_which`. |
| `tests/run_fixture.py` | `make_run` gains `provider`, `model`, `stop_reason` keyword arguments. |
| `tests/test_inspect_run.py` (new) | Tasks 1 and 2 unit tests. |
| `tests/test_inspect_cli.py` (new) | Task 3 CLI tests, and Task 4's documented-keys test. |
| `tests/test_cli.py` | Delete `test_status_lists_every_run` (superseded by Task 3 tests). |
| `docs/json-output.md` (new) | The JSON schema, one table per command. |
| `README.md`, `AGENTS.md` | Command table rows, `talos status`/`show`/`diff` sections, routing row. |

---

### Task 1: Loading runs, status rows, job detail

**Files:**
- Create: `talos/inspect_run.py`
- Modify: `tests/run_fixture.py` (the `make_run` signature and the `JobSpec(...)` / `st.stop_reason` lines)
- Test: `tests/test_inspect_run.py`

**Interfaces:**
- Consumes (existing): `talos.state.JobSpec.from_dict`, `JobState.from_dict`, `TERMINAL`; `talos.report.read_timeline(path) -> list[dict]`, `holdout_points(timeline) -> dict[int, float]`; `talos.budget.effective_budget(budget, extensions) -> Budget`; `talos.cli.unpriced_reason`, `_llm_spend_text`, `_event_line(kind, data, width=100) -> str | None`. Import the `talos.cli` names inside functions (as `report._header` does), so importing `talos.inspect_run` does not import the CLI.
- Produces:
  - `SCHEMA_VERSION: int = 1`
  - `class InspectError(Exception)`: its `str()` is the user-facing message.
  - `@dataclass(frozen=True) class Run: run_dir: Path; spec: JobSpec; state: JobState; timeline: list[dict]`
  - `load_run(runs_root: Path, job_id: str) -> Run`
  - `holdout_status(state: JobState, iteration: int) -> str` returns `"confirmed" | "false_positive" | "not_scored"`
  - `hours_spent(run: Run, now: float) -> float | None`
  - `last_event(timeline: list[dict]) -> dict | None` returns `{"kind", "ts", "iteration", "text"}`
  - `status_rows(runs_root: Path) -> list[dict]`
  - `job_detail(run: Run, now: float) -> dict`
  - `format_status_table(rows: list[dict]) -> str`, `format_job_detail(d: dict) -> str`

- [ ] **Step 1: Extend the fixture.** `make_run(..., provider="fake", model="m", stop_reason="budget exhausted: iterations")`, passing `provider`/`model` to `JobSpec` and `stop_reason` to `st.stop_reason`. Defaults keep every existing caller unchanged. Run `$PY -m pytest -q tests/test_report.py`: 41 passed.

- [ ] **Step 2: Write the failing tests** in `tests/test_inspect_run.py`. The fixture's timeline rows are stamped `T0 + 60*k` and `spend.started_at = T0` (see `tests/run_fixture.py`).

```python
from talos.inspect_run import (InspectError, SCHEMA_VERSION, holdout_status, hours_spent,
                               job_detail, last_event, load_run, status_rows)
from tests.run_fixture import T0, make_run

def _run(tmp_path, name="J", **kw):
    make_run(tmp_path / "runs" / name, kw.pop("hyps", [{"iteration": 1}]), **kw)
    return load_run(tmp_path / "runs", name)

def test_load_run_reads_without_creating(tmp_path):
    # mutation: load_run constructing JobStore(run_dir) creates runs/nope
    with pytest.raises(InspectError, match=r"^no job nope under runs/$"):
        load_run(tmp_path / "runs", "nope")
    assert not (tmp_path / "runs").exists()

@pytest.mark.parametrize("bad", ["../J", "a/b", "a\\b", ".."])
def test_load_run_rejects_path_like_ids(tmp_path, bad):
    # mutation: dropping the separator check lets "../J" resolve to a sibling directory
    make_run(tmp_path / "J", [{"iteration": 1}])          # a real run OUTSIDE runs/
    (tmp_path / "runs").mkdir()
    with pytest.raises(InspectError, match="under runs/"):
        load_run(tmp_path / "runs", bad)

def test_load_run_names_the_missing_file(tmp_path):
    # mutation: treating a missing state.json as "no job" hides a job that crashed early
    d = tmp_path / "runs" / "J"; make_run(d, [{"iteration": 1}]); (d / "state.json").unlink()
    with pytest.raises(InspectError, match=r"^job J has no state.json$"):
        load_run(tmp_path / "runs", "J")
    (d / "state.json").write_text("{not json")
    with pytest.raises(InspectError, match=r"^job J: cannot read state.json"):
        load_run(tmp_path / "runs", "J")

def test_holdout_status_three_ways(tmp_path):
    run = _run(tmp_path, hyps=[{"iteration": i} for i in (1, 2, 3)], best=3,
               confirmed=[1], false_positives=[3])
    assert [holdout_status(run.state, i) for i in (1, 3, 2)] == \
        ["confirmed", "false_positive", "not_scored"]

@pytest.mark.parametrize("status", ["exhausted", "paused"])
def test_hours_spent_stopped_job_uses_last_event(tmp_path, status):
    # mutation: using `now` for a stopped job makes a finished run's hours grow forever;
    # the "paused" case catches dropping `or status == "paused"`
    run = _run(tmp_path, status=status, timeline=[{"kind": "a"}] * 4)   # last ts T0+180
    assert hours_spent(run, now=T0 + 10**6) == pytest.approx(180 / 3600)

def test_hours_spent_running_job_uses_now(tmp_path):
    # mutation: using the last event for a running job under-reports time in a long iteration
    run = _run(tmp_path, status="researching", timeline=[{"kind": "a"}] * 4)
    assert hours_spent(run, now=T0 + 7200) == pytest.approx(2.0)

def test_hours_spent_stopped_job_without_timeline_is_none(tmp_path):
    run = _run(tmp_path, status="failed", timeline=None)
    assert hours_spent(run, now=T0 + 7200) is None

def test_last_event_skips_a_torn_line(tmp_path):
    run_dir = tmp_path / "runs" / "J"
    make_run(run_dir, [{"iteration": 1}],
             timeline=[{"kind": "scored", "iteration": 1, "mean_rel_delta": 0.01,
                        "worst_rel_delta": 0.0, "error_rate": 0.0}])
    with (run_dir / "timeline.jsonl").open("a") as f:
        f.write('{"ts": 1, "kind": "hypo')                 # torn write, no newline
    ev = last_event(load_run(tmp_path / "runs", "J").timeline)
    assert ev["kind"] == "scored" and ev["ts"] == T0 and ev["iteration"] == 1
    assert ev["text"].startswith("scored +1.000% vs baseline")

def test_status_rows_one_per_job_including_unreadable(tmp_path):
    # mutation: a bare `except` that skips the bad dir, or iterating only */state.json
    runs = tmp_path / "runs"
    make_run(runs / "A", [{"iteration": 1}], best=1, timeline=[{"kind": "x"}])
    make_run(runs / "B", [{"iteration": 1}]); (runs / "B" / "state.json").unlink()
    make_run(runs / "C", [{"iteration": 1}]); (runs / "C" / "state.json").write_text("{")
    (runs / "D").mkdir()                                   # nothing saved yet: no row
    # replaces PR #82's test_status_without_a_readable_job_json_still_prints_the_job:
    # an incomplete job.json must give a row, not a traceback
    make_run(runs / "E", [{"iteration": 1}]); (runs / "E" / "job.json").write_text('{"provider": "x"}')
    rows = status_rows(runs)
    assert [r["job_id"] for r in rows] == ["A", "B", "C", "E"]
    a, b, c, e = rows
    assert e["status"] == "unreadable" and e["error"].startswith("job E: cannot read job.json")
    assert a["status"] == "exhausted" and a["best_mean_rel_delta"] == 0.105
    assert a["last_event_ts"] == T0 and a["error"] is None and a["llm_usd"] == 0.0
    assert b["status"] == "unreadable" and b["error"] == "job B has no state.json"
    assert c["status"] == "unreadable" and c["error"].startswith("job C: cannot read state.json")
    assert set(b) == set(a)                                # same keys, nulls not missing keys

def test_status_rows_missing_runs_dir_is_empty(tmp_path):
    assert status_rows(tmp_path / "runs") == []

def test_unpriced_spend_is_null_not_zero(tmp_path):
    # mutation: reporting state.spend.llm_usd (0.0) for codex-cli
    make_run(tmp_path / "runs" / "J", [{"iteration": 1}], provider="codex-cli", model="x")
    row, = status_rows(tmp_path / "runs")
    assert row["llm_usd"] is None and row["llm_priced"] is False
    assert row["llm_unpriced_reason"] == "codex-cli does not report what its calls cost"
    d = job_detail(load_run(tmp_path / "runs", "J"), now=T0)
    assert d["budget"]["usd"] == {"cap": 1.0, "spent": None, "left": None}

def test_job_detail_budget_and_best(tmp_path):
    # Fixture budget: usd 1.0, hours 2.0, iterations 10, compute_usd None; 1 iteration spent.
    run = _run(tmp_path, hyps=[{"iteration": 1}, {"iteration": 2}], best=2, confirmed=[2],
               status="won", timeline=[{"kind": "x"}, {"kind": "won", "iteration": 2,
               "holdout": {"mean_rel_delta": 0.04, "worst_rel_delta": 0.01}}])
    d = job_detail(run, now=T0 + 10**6)
    assert d["schema_version"] == SCHEMA_VERSION == 1
    assert d["budget"]["iterations"] == {"cap": 10, "spent": 2, "left": 8}
    assert d["budget"]["hours"] == {"cap": 2.0, "spent": pytest.approx(60 / 3600),
                                    "left": pytest.approx(2.0 - 60 / 3600)}
    assert d["budget"]["compute_usd"]["cap"] is None and d["budget"]["compute_usd"]["left"] is None
    assert d["best"] == {"iteration": 2, "title": "title 2", "mean_rel_delta": 0.105,
                         "holdout_status": "confirmed", "holdout_mean_rel_delta": 0.04}
    assert "rand_hash" not in d["job"]
    assert d["activity"]["pending"] is None and d["activity"]["last_event"]["kind"] == "won"

def test_budget_left_never_negative(tmp_path):
    # mutation: cap - spent without max(0, ...) prints "-3 iterations left"
    run = _run(tmp_path, hyps=[{"iteration": i} for i in range(1, 14)])   # 13 > cap 10
    assert job_detail(run, now=T0)["budget"]["iterations"]["left"] == 0

def test_budget_counts_extensions(tmp_path):
    # mutation: reading spec.budget instead of effective_budget(...)
    run_dir = tmp_path / "runs" / "J"; make_run(run_dir, [{"iteration": 1}])
    st = json.loads((run_dir / "state.json").read_text())
    st["budget_extensions"] = [{"add": {"iterations": 5}}]
    (run_dir / "state.json").write_text(json.dumps(st))
    d = job_detail(load_run(tmp_path / "runs", "J"), now=T0)
    assert d["budget"]["iterations"]["cap"] == 15

def test_detail_in_progress_names_the_pending_iteration(tmp_path):
    run = _run(tmp_path, status="researching", timeline=[{"kind": "hypothesis", "iteration": 2,
               "title": "try X"}], pending_job={"purpose": 2, "hypothesis": {"title": "try X"},
               "files": {}})
    assert job_detail(run, now=T0)["activity"]["pending"] == {"purpose": 2, "title": "try X"}

def test_detail_of_a_job_with_no_baseline(tmp_path):
    run = _run(tmp_path, hyps=[], baseline=None, status="failed", timeline=None)
    d = job_detail(run, now=T0)
    assert d["best"] is None and d["activity"]["last_event"] is None
    assert d["last_event_ts"] is None
```

- [ ] **Step 3: Run them.** `$PY -m pytest -q tests/test_inspect_run.py`. Expected: collection error `ModuleNotFoundError: No module named 'talos.inspect_run'`.

- [ ] **Step 4: Implement `talos/inspect_run.py`.**
  - `load_run`: reject an id that is empty, is `.` or `..`, or contains `/` or `\`, with `no job <id> under runs/`. Then: not a directory, same message. Missing `state.json` gives `job <id> has no state.json`; missing `job.json` gives `job <id> has no job.json` (state is checked first). A read, JSON or `from_dict` failure (`OSError, ValueError, KeyError, TypeError`) gives `job <id>: cannot read <name>: <exc>`. The timeline comes from `read_timeline`.
  - `hours_spent`: if `state.status in TERMINAL or state.status == "paused"`, the last timeline `ts` (rows whose `ts` is a number) minus `spend.started_at`, or None with no such row. Otherwise `now - started_at`.
  - `last_event`: the last row with a numeric `ts` and a string `kind`. `text` is `_event_line(kind, data-without-ts-kind)`, falling back to `kind` when that returns None or raises `KeyError`/`TypeError`/`ValueError` (old rows may lack fields).
  - `status_rows`: for each sorted subdirectory of `runs_root` holding `job.json` or `state.json`, either a row from `load_run` or an `unreadable` row. Row keys, in this order: `job_id, challenge, provider, model, status, iteration, best_mean_rel_delta, llm_usd, llm_priced, llm_unpriced_reason, compute_usd, last_event_ts, error`. (`provider`/`model` are an addition to the spec's key list, so the table can call `_llm_spend_text` instead of re-deriving its label rule.) An unreadable row has `status="unreadable"`, its `error` message, and None everywhere else. A missing `runs_root` gives `[]`.
  - `job_detail` keys: `schema_version, job_id, job` (`spec.redacted()`), `status, stop_reason, iteration, runs_since_improvement, hardware, scoring` (`{"requested": spec.scoring, "actual": state.scoring}`), `best` (`{iteration, title, mean_rel_delta, holdout_status, holdout_mean_rel_delta}` or None; the title is from `best.hypothesis["title"]`, the held-out value from `holdout_points`), `budget` (keys `iterations, hours, usd, compute_usd`, each `{cap, spent, left}`; `left = max(0, cap - spent)`, None when the cap or the spent value is None; spent is `spend.iterations`, `hours_spent`, `spend.llm_usd` (None when unpriced), `spend.compute_usd`), `llm_priced, llm_unpriced_reason, last_event_ts, activity` (`{"pending": {"purpose", "title"} | None, "last_event": last_event(...)}`, where `title` is `pending_job["hypothesis"]["title"]` when present, else None).

- [ ] **Step 5: Add the formatters and their tests.** `format_status_table(rows)` gives a header line `JOB CHALLENGE STATUS IT BEST LLM COMPUTE LAST EVENT`, padded to column widths, then one line per row. BEST is `+0.105%`-style (`f"{x:+.3%}"`) or `n/a`. LLM is `_llm_spend_text(row["provider"], row["model"], row["llm_usd"] or 0.0)`, or `-` on an unreadable row. LAST EVENT is `%Y-%m-%d %H:%M UTC` or `-`. `format_job_detail(d)` gives `key: value` lines covering every top-level key; a None cap prints `∞`. Tests:

```python
def test_status_table_shows_unpriced_and_missing_values(tmp_path):
    runs = tmp_path / "runs"
    make_run(runs / "A", [{"iteration": 1}], best=1, timeline=[{"kind": "x"}])
    make_run(runs / "B", [{"iteration": 1}], provider="codex-cli", model="x", timeline=None)
    lines = format_status_table(status_rows(runs)).splitlines()
    assert lines[0].split()[:4] == ["JOB", "CHALLENGE", "STATUS", "IT"]
    assert "+10.500%" in lines[1] and "2001-09-09 01:46 UTC" in lines[1]   # T0 in UTC
    # COMPUTE is a measured $0.00 on this row; the LLM cell must not be a second one
    assert "not tracked" in lines[2] and lines[2].count("$0.00") == 1
    assert lines[2].rstrip().endswith("-")
    # replaces PR #82's test_status_labels_untracked_and_unpriced_llm_spend (openai half)
    make_run(runs / "C", [{"iteration": 1}], provider="openai", model="gpt-5", timeline=None)
    assert "unpriced" in format_status_table(status_rows(runs)).splitlines()[3]

def test_job_detail_text_prints_infinite_cap(tmp_path):
    text = format_job_detail(job_detail(_run(tmp_path), now=T0))
    assert "∞" in text and "compute" in text
```

- [ ] **Step 6: Run the tests.** `$PY -m pytest -q tests/test_inspect_run.py tests/test_report.py`: all pass. `$PY -m ruff check talos/inspect_run.py tests/test_inspect_run.py tests/run_fixture.py`: clean.

- [ ] **Step 7: Commit.**
```bash
git add talos/inspect_run.py tests/test_inspect_run.py tests/run_fixture.py
git commit -m "inspect: load runs read-only, status rows and job detail (#68)"
```

---

### Task 2: Iteration detail and diff

**Files:**
- Modify: `talos/inspect_run.py`
- Test: `tests/test_inspect_run.py`

**Interfaces:**
- Consumes: Task 1's `Run`, `InspectError`, `holdout_status`, `SCHEMA_VERSION`; existing `talos.report.report_nodes(spec, state) -> list[ReportNode]`, `run_sources(run_dir, spec, state) -> Callable[[int], dict | None]`, `holdout_points`; `talos.package._diff(base, cand, label) -> str`, `score_pairs(spec, state)`; `talos.scoring.bundle_delta`, `ScoringError`.
- Produces:
  - `resolve_node(run: Run, which: str | int) -> ReportNode`: `which` is `"best"` or a positive int.
  - `iteration_detail(run: Run, which: str | int) -> dict`
  - `iteration_diff(run: Run, which: str | int, against: str) -> str`: `against` is `"baseline"` or `"parent"`.
  - `format_iteration(d: dict) -> str`

- [ ] **Step 1: Write the failing tests.** The fixture's `make_run(best=N)` writes `iterations/NNNN/mod.rs` = `"line a\nline b\nline c best N\n"` and baseline files `{"mod.rs": "line a\nline b\nline c\n"}`.

```python
def test_resolve_errors(tmp_path):
    run = _run(tmp_path, hyps=[{"iteration": 1}, {"iteration": 2}])
    with pytest.raises(InspectError, match=r"^job J has no best candidate; use --iteration N$"):
        resolve_node(run, "best")
    with pytest.raises(InspectError, match=r"^job J has no iteration 3 \(it has 1\.\.2\)$"):
        resolve_node(run, 3)
    empty = _run(tmp_path, name="K", hyps=[])
    with pytest.raises(InspectError, match=r"^job K has no iteration 1 \(it has none\)$"):
        resolve_node(empty, 1)

def test_show_uses_record_mean_not_metered_rank_score(tmp_path):
    # mutation: reading ReportNode.score (the metered score when present) as the training mean
    nodes = [{"id": 1, "chain": 0, "parent": 0, "family": "f", "title": "t", "outcome": "improved",
              "score": 0.02, "tracks": {"t": 0.02}, "metered_score": 0.015}]
    run = _run(tmp_path, hyps=[{"iteration": 1, "mean_rel_delta": 0.02, "worst_track": "t",
               "worst_rel_delta": 0.02, "design_family": "f"}], nodes=nodes,
               iteration_files={1: {"mod.rs": "x\n"}})
    s = iteration_detail(run, 1)["scores"]
    assert s["mean_rel_delta"] == 0.02 and s["metered_rel_delta"] == 0.015
    assert s["tracks"] == {"t": 0.02} and s["tables"] is None

def test_show_pre_lineage_record_has_null_tracks(tmp_path):
    run = _run(tmp_path, hyps=[{"iteration": 1, "mean_rel_delta": -0.01, "worst_track": "t",
               "worst_rel_delta": -0.01, "against": 0}])
    d = iteration_detail(run, 1)
    assert d["scores"]["tracks"] is None and d["scores"]["worst_track"] == "t"
    assert d["parent"] == 0 and d["schema_version"] == 1

def test_show_best_has_tables_holdout_and_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)                       # file paths are relative to the cwd
    run = _run(tmp_path, hyps=[{"iteration": 1}, {"iteration": 2}], best=2, false_positives=[2],
               timeline=[{"kind": "false_positive", "iteration": 2,
                          "holdout": {"mean_rel_delta": -0.03, "worst_rel_delta": -0.05}}])
    d = iteration_detail(run, "best")
    assert d["iteration"] == 2 and d["kind"] == "best"
    assert d["scores"]["holdout_status"] == "false_positive"
    assert d["scores"]["holdout_mean_rel_delta"] == -0.03       # the failing value, not null
    headings = [t["heading"] for t in d["scores"]["tables"]]
    assert headings == ["Training nonces", "Held-out nonces"]
    assert d["scores"]["tables"][0]["tracks"][0]["track"] == "t"
    assert "runs/J/iterations/0002/mod.rs" in d["files"]

def test_show_best_lists_the_best_dir(tmp_path, monkeypatch):
    # mutation: listing only iterations/NNNN/ (the spec names best/ for the best)
    monkeypatch.chdir(tmp_path)
    run = _run(tmp_path, hyps=[{"iteration": 1}, {"iteration": 2}], best=2,
               iteration_files={1: {"mod.rs": "one\n"}})
    (run.run_dir / "best").mkdir(); (run.run_dir / "best" / "mod.rs").write_text("x\n")
    assert "runs/J/best/mod.rs" in iteration_detail(run, "best")["files"]
    # mutation: listing best/ for every iteration
    assert "runs/J/best/mod.rs" not in iteration_detail(run, 1)["files"]

def test_diff_best_against_baseline_and_parent(tmp_path):
    run = _run(tmp_path, hyps=[{"iteration": 1}, {"iteration": 2, "against": 1}], best=2,
               iteration_files={1: {"mod.rs": "line a\nline b\nline c one\n"}})
    base = iteration_diff(run, "best", "baseline")
    assert base.startswith("--- baseline/mod.rs\n+++ candidate/mod.rs\n")
    assert "+line c best 2" in base and "-line c\n" in base
    par = iteration_diff(run, "best", "parent")
    assert par.startswith("--- #1/mod.rs") and "-line c one" in par

def test_diff_template_start_labels_template(tmp_path):
    run = _run(tmp_path, start="template", template="fn t() {}\n", best=1)
    assert iteration_diff(run, "best", "baseline").startswith("--- template/mod.rs")

def test_diff_against_a_parent_with_no_files(tmp_path):
    # mutation: diffing against {} when the base is None prints a whole-file "addition"
    run = _run(tmp_path, hyps=[{"iteration": 1, "outcome": "failed:compile"},
                               {"iteration": 2, "against": 1}], best=2, iteration_files={1: {}})
    with pytest.raises(InspectError, match=r"^the files of #1 were not saved$"):
        iteration_diff(run, "best", "parent")

def test_diff_of_an_iteration_with_no_files(tmp_path):
    run = _run(tmp_path, hyps=[{"iteration": 1, "outcome": "failed:compile"}],
               iteration_files={1: {}})
    with pytest.raises(InspectError,
                       match=r"^iteration 1 saved no files \(outcome failed:compile\)$"):
        iteration_diff(run, 1, "baseline")

def test_diff_best_uses_state_files_when_the_iteration_dir_is_empty(tmp_path):
    # make_run copies iteration_files[1] into state.best.files, so set those afterwards
    run = _run(tmp_path, best=1, iteration_files={1: {}})   # iterations/0001 has no sources
    st = json.loads((run.run_dir / "state.json").read_text())
    st["best"]["files"] = {"mod.rs": "line a\nline b\nfrom state\n"}
    (run.run_dir / "state.json").write_text(json.dumps(st))
    run = load_run(run.run_dir.parent, "J")
    assert "+from state" in iteration_diff(run, "best", "baseline")

def test_diff_is_not_truncated(tmp_path):
    # mutation: reusing report.node_diffs / truncate caps the patch
    big = "".join(f"line {i}\n" for i in range(200_000))
    run = _run(tmp_path, best=1, iteration_files={1: {"mod.rs": big}})
    assert iteration_diff(run, "best", "baseline").count("\n+line ") == 200_000

def test_format_iteration_names_parent(tmp_path):
    run = _run(tmp_path, hyps=[{"iteration": 1}, {"iteration": 2, "against": 1}], best=2)
    assert "parent: #1" in format_iteration(iteration_detail(run, 2))
    assert "parent: baseline" in format_iteration(iteration_detail(run, 1))
```

- [ ] **Step 2: Run them.** `$PY -m pytest -q tests/test_inspect_run.py -k "resolve or show or diff or format_iteration"`. Expected: ImportError on the new names.

- [ ] **Step 3: Implement.**
  - `resolve_node`: `"best"` means `state.best.iteration`; with no best, raise the no-best message. A number means the `report_nodes` entry with that id. A missing id raises `has no iteration N (it has 1..M)`, where M is the highest record id, or `(it has none)` when there are no records.
  - `iteration_detail` keys: `schema_version, job_id, iteration, parent` (int, 0 = root), `chain, kind, outcome, error, hypothesis` (`{title, description, strategy_tag, design_family}`, where `design_family` is the record's `design_family` or the node family, `""` if neither), `scores` (`{mean_rel_delta, worst_track, worst_rel_delta, runtime_ratio, metered_rel_delta, tracks, holdout_status, holdout_mean_rel_delta, tables}`), `files`.
    - `mean_rel_delta`: the record's `mean_rel_delta`, else the stored `Node.score`, else None.
    - `metered_rel_delta` and `tracks`: from the stored `Node` (`state.nodes` by id), else None.
    - `tables`: only when the node is the best and `state.baseline` is set; else None. Each table is `{heading, tracks: [{track, base_mean, cand_mean, rel_delta, cand_errors, n}], mean_rel_delta, error_rate, error}`, built from `score_pairs` + `bundle_delta`. On `ScoringError`, that table has `tracks: []`, the numbers None, and `error` set to the message.
    - `files`: sorted POSIX paths, relative to the current directory, of every file under `runs_root/<id>/iterations/NNNN/`, plus, for the best node only, every file under `runs_root/<id>/best/` when that directory exists (the loop writes it: `talos/loop.py`, `self.store.run_dir / "best"`). Use `run.run_dir.relative_to(Path.cwd())` when possible, else the absolute path.
  - `iteration_diff`: the candidate's files are `run_sources(...)(node.id)`; None or an empty dict (`if not cand`: an empty file map means nothing was saved, `run_sources` returns `dict(state.best.files)` even when that is `{}`) raises the saved-no-files message. The same rule for the base files. The base id is 0 for `baseline` and `node.parent` for `parent`. Base files None raises `the files of <label> were not saved`. The label is `template` (template start, base 0), `baseline` (base 0), or `#N`. Return `_diff(base, cand, label)` untruncated.
  - `format_iteration(d)`: `key: value` lines. `parent: baseline` / `parent: template` / `parent: #N`. The description is printed whole. Per-track lines are `  <track> +x.xxx%`. Tables are printed as aligned rows. The files are listed last.

- [ ] **Step 4: Run.** `$PY -m pytest -q tests/test_inspect_run.py tests/test_report.py` passes; ruff is clean.

- [ ] **Step 5: Commit.**
```bash
git add talos/inspect_run.py tests/test_inspect_run.py
git commit -m "inspect: iteration detail and untruncated diff (#68)"
```

---

### Task 3: CLI commands

**Files:**
- Modify: `talos/cli.py` (`cmd_status`; new `cmd_show`, `cmd_diff`, `parse_which`; `build_parser` around the `status` / `report` sub-parsers; the `COMMANDS` dict that `main` dispatches through)
- Modify: `tests/test_cli.py` (delete `test_status_lists_every_run`, `_status_job`, `test_status_labels_untracked_and_unpriced_llm_spend` and `test_status_without_a_readable_job_json_still_prints_the_job`; the last two came with PR #82 and assert the old one-line format)
- Test: `tests/test_inspect_cli.py` (new)

**Interfaces:**
- Consumes: everything Task 1 and Task 2 produce.
- Produces: CLI surface `talos status [JOB_ID] [--json]`, `talos show JOB_ID [--iteration N|best] [--json]`, `talos diff JOB_ID [--iteration N|best] [--against {parent,baseline}]`; `parse_which(text: str) -> str | int` (argparse `type=`).

- [ ] **Step 1: Write the failing tests** (`monkeypatch.chdir(tmp_path)`, runs built under `tmp_path / "runs"` with `make_run`):

```python
def test_status_table_and_json(tmp_path, monkeypatch, capsys):
    ...two jobs...
    assert cli.main(["status"]) == 0
    assert capsys.readouterr().out.splitlines()[0].startswith("JOB")
    assert cli.main(["status", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["schema_version"] == 1 and [j["job_id"] for j in out["jobs"]] == ["A", "B"]

def test_status_with_no_jobs(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert cli.main(["status"]) == 0 and "no jobs under runs/" in capsys.readouterr().out
    assert cli.main(["status", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {"schema_version": 1, "jobs": []}

def test_status_job_detail_json(...):     # keys budget, activity, best present; exit 0
def test_show_and_diff_defaults_to_best(...):
    assert cli.main(["show", "J", "--json"]) == 0 and json.loads(out)["iteration"] == 2
    assert cli.main(["diff", "J"]) == 0 and out.startswith("--- baseline/")

def test_json_error_leaves_stdout_empty(tmp_path, monkeypatch, capsys):
    # mutation: printing the error (or a JSON error object) to stdout
    make_run(tmp_path / "runs" / "J", [{"iteration": 1}]); monkeypatch.chdir(tmp_path)
    assert cli.main(["show", "J", "--json"]) == 1
    cap = capsys.readouterr()
    assert cap.out == "" and "has no best candidate" in cap.err

@pytest.mark.parametrize("bad", ["0", "-1", "abc", "1.5", "\u00b2"])   # "²".isdigit() is True
def test_iteration_flag_rejects_bad_values(bad, ...):
    with pytest.raises(SystemExit) as e: cli.main(["show", "J", "--iteration", bad])
    assert e.value.code == 2

def test_inspect_commands_write_nothing(tmp_path, monkeypatch):
    # mutation: any of the commands constructing JobStore, or writing report.html
    ...run with best, a no-candidate run, an in-progress run...
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    mtimes = {p: p.stat().st_mtime_ns for p in tmp_path.rglob("*") if p.is_file()}
    for argv in (["status"], ["status", "--json"], ["status", "J"], ["status", "nope"],
                 ["show", "J"], ["show", "J", "--iteration", "1", "--json"], ["diff", "J"],
                 ["diff", "J", "--against", "parent"], ["show", "N"], ["status", "P", "--json"]):
        cli.main(argv)
    assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*")) == before
    assert all(p.stat().st_mtime_ns == m for p, m in mtimes.items())

def test_inspect_commands_need_no_config_or_network(tmp_path, monkeypatch):
    # mutation: cmd_status calling config.load() or make_bench
    monkeypatch.setattr(cli, "load", lambda *a, **k: pytest.fail("config loaded"))
    monkeypatch.setattr(cli, "make_bench", lambda *a, **k: pytest.fail("bench built"))
    monkeypatch.setattr(cli.urllib.request, "urlopen", lambda *a, **k: pytest.fail("network"))
    ...each command returns 0 on a run with a best...
```

Fill in the `...` fixture setup with `make_run` calls: J has a best (2), N has no candidate, P is `status="researching"` with a `pending_job`. They are written out in full in the test file. They are elided here only because they repeat Task 1.

- [ ] **Step 2: Run.** `$PY -m pytest -q tests/test_inspect_cli.py`. Expected: failures (`invalid choice: 'show'`, the old status format).

- [ ] **Step 3: Implement in `talos/cli.py`.**
  - `parse_which`: `"best"` returns `"best"`. A string with `text.isascii() and text.isdigit()` whose value is at least 1 returns an int (`isdigit` alone accepts `"²"`, on which `int()` raises). Anything else raises `argparse.ArgumentTypeError("expected a positive iteration number or 'best'")`.
  - `cmd_status(args, ask)`: `root = Path.cwd() / "runs"`. With no `job_id`: `status_rows(root)`. Empty rows print `no jobs under runs/` (human) or `{"schema_version": 1, "jobs": []}` (JSON). Otherwise print the table, or the JSON object `{"schema_version": SCHEMA_VERSION, "jobs": rows}`. For each unreadable row in human mode, also print `<job_id>: <error>` to stderr. With a `job_id`: `job_detail(load_run(root, id), time.time())`.
  - `cmd_show`: `iteration_detail`, printed as `format_iteration` or JSON.
  - `cmd_diff`: `sys.stdout.write(iteration_diff(...))`.
  - All three catch `InspectError`, print it to stderr and return 1. JSON uses `json.dumps(obj, indent=1)` plus a newline.
  - Register `"show": cmd_show, "diff": cmd_diff` in `COMMANDS`.
  - Parser: `status` gets `job_id` (`nargs="?"`) and `--json`. `show` gets `job_id`, `--iteration` (`type=parse_which`, default `"best"`) and `--json`. `diff` gets `job_id`, `--iteration` (same) and `--against` (`choices=("parent", "baseline")`, default `"baseline"`). Help strings say they read `runs/` only. Import `talos.inspect_run` inside the three `cmd_*` functions.
  - Delete from `tests/test_cli.py`: `test_status_lists_every_run`, the `_status_job` helper, `test_status_labels_untracked_and_unpriced_llm_spend` and `test_status_without_a_readable_job_json_still_prints_the_job`. Their mutations are carried by Task 1's `test_status_rows_one_per_job_including_unreadable` (dir with nothing saved; incomplete job.json gives a row, not a traceback) and `test_status_table_shows_unpriced_and_missing_values` (codex `not tracked`, openai/gpt-5 `unpriced`, no measured-looking LLM `$0.00`). Behaviour change from PR #82: a job whose job.json cannot be read was shown with `llm=$x.xx (provider unknown)`; it is now an `unreadable` row, as the spec decides.

- [ ] **Step 4: Run.** `$PY -m pytest -q tests/test_inspect_cli.py tests/test_cli.py tests/test_inspect_run.py`: all pass. Ruff is clean.

- [ ] **Step 5: Commit.**
```bash
git add talos/cli.py tests/test_cli.py tests/test_inspect_cli.py
git commit -m "cli: talos status JOB_ID, show and diff with --json (#68)"
```

---

### Task 4: JSON schema doc, documented-keys test, README and AGENTS.md

**Files:**
- Create: `docs/json-output.md`
- Modify: `README.md` (command table near the `talos status` row; replace the `### talos status` section; add `### talos show` and `### talos diff` sections after it), `AGENTS.md` (routing table near the `docs/compute-backends.md` row)
- Test: `tests/test_inspect_cli.py`

**Interfaces:**
- Consumes: the key lists produced by Tasks 1–3 (`status_rows` row keys, `job_detail` keys, `iteration_detail` keys).
- Produces: `docs/json-output.md` with three sections whose headings are exactly `## talos status --json`, `## talos status JOB_ID --json` and `## talos show --json`. Each section has one table with header `| key | type | meaning |`. A key cell is a backticked dotted path: `jobs[].job_id`, `budget.hours.left`, `scores.tables[].tracks[].rel_delta`. `[]` means "each element of this list". Maps keyed by track name are documented down to the map (`scores.tracks`), not inside it.

- [ ] **Step 1: Write the failing test.**

```python
SECTIONS = {"## talos status --json": ["status", "--json"],
            "## talos status JOB_ID --json": ["status", "J", "--json"],
            "## talos show --json": ["show", "J", "--json"]}

def documented_keys(text: str) -> dict[str, list[str]]:
    """heading -> key paths from the first column of that section's table."""

def has_path(obj, path: str) -> bool:
    """True if every segment exists as a key; a `[]` segment needs a non-empty list and is
    checked on its first element. A null value at the end of the path counts as present."""

def test_every_documented_key_is_in_the_output(tmp_path, monkeypatch, capsys):
    # mutation: renaming a key in inspect_run (e.g. best_mean_rel_delta) fails this
    doc = (REPO / "docs" / "json-output.md").read_text(encoding="utf-8")
    keys = documented_keys(doc)
    assert set(keys) == set(SECTIONS)
    assert all(len(v) >= 8 for v in keys.values())   # guards a parser that finds nothing
    # Fixture J: best=2 (confirmed, won event with holdout), nodes with tracks, a pending job
    # and status "researching", so every documented path has a non-null parent.
    ...
    for heading, argv in SECTIONS.items():
        assert cli.main(argv) == 0
        out = json.loads(capsys.readouterr().out)
        missing = [k for k in keys[heading] if not has_path(out, k)]
        assert not missing, (heading, missing)
        # and the reverse at the top level: a new top-level key cannot ship undocumented
        tops = {k.split(".")[0].split("[")[0] for k in keys[heading]}
        assert set(out) <= tops, (heading, set(out) - tops)

def test_has_path_is_strict():
    # guards the checker itself: it must not accept everything
    assert has_path({"a": {"b": None}}, "a.b")
    assert not has_path({"a": {}}, "a.b")
    assert not has_path({"a": []}, "a[].b")
    assert has_path({"a": [{"b": 1}]}, "a[].b")
```

- [ ] **Step 2: Run.** It fails with `FileNotFoundError` for `docs/json-output.md`.

- [ ] **Step 3: Write `docs/json-output.md`.** It opens with a stability paragraph: `schema_version` is 1; a new key is a compatible addition; removing or renaming a key, or changing what a value means, bumps it; deltas are fractions; times are Unix seconds; absent values are `null`; errors are not JSON (stderr, exit 1). Then the three tables, listing every key from Tasks 1–3, including each `budget.<dim>.{cap,spent,left}` for all four dimensions, `activity.pending.purpose`/`title`, `activity.last_event.{kind,ts,iteration,text}`, and `scores.tables[].tracks[].{track,base_mean,cand_mean,rel_delta,cand_errors,n}`. For `job`, document only `job` itself as "`JobSpec.redacted()`: the job's settings without the nonce seed". Its fields follow `JobSpec`, and the doc says they are outside the versioned contract.

- [ ] **Step 4: README and AGENTS.md.**
  - README command table: change the `talos status` row to "Lists every job under `runs/`, or one job's detail with `JOB_ID`; `--json` for scripts." Add rows for `talos show JOB_ID` and `talos diff JOB_ID`.
  - Replace the `### talos status` body (including PR #82's paragraph on `llm=not tracked` / `(provider unknown)`, which describes the old one-line format): the table example (an illustrative row, labelled as illustrative), the `unreadable` row, the `JOB_ID` detail, how hours spent is measured, and `--json` with a link to `docs/json-output.md`. Keep the existing paragraph listing the statuses.
  - New `### talos show` and `### talos diff` sections: usage line, flags, defaults, and exit codes (0, 1 with a stderr message, 2 for a bad flag).
  - AGENTS.md routing table: the row `| JSON output of talos status / show (the versioned schema) | docs/json-output.md |`. If AGENTS.md lists the authoritative docs (item 3 of its reading order), add `docs/json-output.md` there too.

- [ ] **Step 5: Run the full check.** `make check PYTHON=$PY`. Expected: ruff clean, all non-live tests pass (`tests/test_docs_references.py` included, which resolves backticked references in the docs), agentify check passes. Record the pass count from the output.

- [ ] **Step 6: Commit.**
```bash
git add docs/json-output.md README.md AGENTS.md tests/test_inspect_cli.py
git commit -m "docs: JSON schema for status/show, README sections, keys test (#68)"
```

---

## After the tasks

- Mutation-check the new tests (CLAUDE.md "Plans and specs"): each test above names its mutation. Apply each mutation, run the test under a fresh bytecode cache (`PYTHONDONTWRITEBYTECODE=1 $PY -m pytest -p no:cacheprovider ...`; see the stale-pyc memory), confirm it fails, then restore.
- Smoke on real runs (read-only): `$PY -m talos.cli status`, `status 20261026-120733-job_scheduling --json`, `show 20261026-120733-job_scheduling`, and `diff 20261026-120733-job_scheduling | head`, run from `/home/fibonadithya/TIG/Talos` with `PYTHONPATH` pointing at the worktree. Check the September runs (pre-lineage) for `show --iteration 1`.
- PR to `main` that closes #68, then the `review-pr` loop.
