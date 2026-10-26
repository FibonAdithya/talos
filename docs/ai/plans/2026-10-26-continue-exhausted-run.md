# Continue an Exhausted Run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `talos run --resume JOB --add-iterations N` (and `--add-hours`, `--add-usd`,
`--add-compute-usd`) adds budget to a job, so an exhausted job continues where it stopped.

**Architecture:** `job.json` stays write-once. Extensions are appended to a new
`JobState.budget_extensions` list. `talos/budget.py::effective_budget` sums them onto the spec's
budget, and every budget reader uses that sum. A new `talos/extend.py::extend` validates one
extension, rescales annealing reheat points, and appends the entry. `execute_job` calls it on
resume, before anything is saved.

**Tech Stack:** Python 3.11 (`.venv`), pytest, ruff.

**Spec:** `docs/ai/specs/2026-10-26-continue-exhausted-run-design.md` (committed `cfe7e8a`).

## Global Constraints

- `job.json` is never rewritten (`talos/state.py::JobStore.write_spec` is write-once).
- An extension never re-measures the baseline or re-calibrates.
- A refused extension leaves `state.json` and `timeline.jsonl` byte-identical.
- Refusals exit 2. A `won` job is refused as today (exit 1, package rebuilt).
- Lines ≤ 100 characters. ruff's pinned rule set does not check E501. Check with
  `.venv/bin/python -c "import sys;[print(f,i+1) for f in sys.argv[1:] for i,l in enumerate(open(f,encoding='utf-8')) if len(l.rstrip('\n'))>100]" <files>`.
- Tests: `.venv/bin/python -m pytest -q -m "not live" <file>`. Full gate:
  `make check PYTHON=.venv/bin/python`. At `cfe7e8a` this gives `863 passed, 7 deselected`
  (MEASURED 2026-10-26). Never run pytest without `-m "not live"`.
- Dimension names are exactly `iterations`, `hours`, `usd`, `compute_usd`, matching
  `talos/budget.py::Budget` fields.
- Every new test carries a `# mutation:` comment naming the bug it catches, as the existing
  tests do.

## Review Focus

1. Second extension of an already extended job: the rescale must use the already extended
   budget as "old", not `spec.budget`. Test in Task 1:
   `test_second_extension_rescales_against_the_extended_budget`.
2. `--add-*` on a `won` job: refused as today, and nothing recorded. Test in Task 3:
   `test_add_flags_on_a_won_job_record_nothing`.
3. A plain `--resume` of an `exhausted` job whose `stop_reason` names no dimension (an old job,
   or a hand-edited one): exits 1 with the old message and no traceback. Test in Task 3:
   `test_exhausted_resume_with_an_unknown_reason_does_not_crash`.
4. A paused job with a compute job in flight: the extension leaves `pending_job` untouched, so
   the resume reattaches. Test in Task 1: `test_extend_leaves_the_pending_job_alone`.
5. A job exhausted on hours, given `--add-hours`: the clock shift and the addition together let
   it continue. Without the shift it would be refused. Test in Task 3:
   `test_hours_exhausted_job_continues_with_added_hours`.

---

### Task 1: Extension core (budget sum, state field, `extend`)

**Files:**
- Modify: `talos/budget.py` (add `ADD_DIMENSIONS`, `effective_budget`)
- Modify: `talos/state.py` (`JobState.budget_extensions`, `to_dict`, `from_dict`)
- Create: `talos/extend.py`
- Test: `tests/test_budget.py`, `tests/test_state.py`, `tests/test_extend.py` (new)

**Interfaces:**
- Produces:
  - `talos.budget.ADD_DIMENSIONS: tuple[str, ...] = ("iterations", "hours", "usd", "compute_usd")`
  - `talos.budget.effective_budget(budget: Budget, extensions: list[dict]) -> Budget`: each
    extension's `e["add"]` maps dimension to a number or `None`. The result adds every
    non-`None` addition to a non-`None` cap. A `None` cap stays `None`.
  - `JobState.budget_extensions: list[dict]`, default `[]`. `from_dict` uses
    `d.get("budget_extensions", [])`.
  - `talos.extend.ADD_FLAGS: dict[str, str]`: `{"iterations": "--add-iterations", "hours":
    "--add-hours", "usd": "--add-usd", "compute_usd": "--add-compute-usd"}`
  - `talos.extend.BUDGET_FLAGS: dict[str, str]`: the same keys mapped to `--budget-iterations`,
    `--budget-hours`, `--budget-usd`, `--budget-compute-usd`.
  - `class talos.extend.ExtensionRefused(ValueError)`: its `str()` is the user-facing message.
  - `talos.extend.extend(budget: Budget, state: JobState, add: dict[str, float], now: float,
    previous_status: str) -> dict`. `add` holds only the dimensions the user gave. It raises
    `ExtensionRefused` before any mutation, in this order:
    1. any value `<= 0`: message contains `"{ADD_FLAGS[d]} must be greater than 0"`
    2. a dimension whose `budget` cap is `None`: message contains `BUDGET_FLAGS[d]`, saying
       the job was started without it
    3. `exhausted(new_effective, state.spend, now)` returns a dimension `d2`: message contains
       `"still out of budget on {d2}"` and `ADD_FLAGS[d2]`

    Otherwise it builds the entry `{"at": now, "iteration": state.iteration,
    "previous_status": previous_status, "add": {each of ADD_DIMENSIONS: add.get(d)}}` and
    rescales every `state.chains[i].reheated_at` (algorithm below). It appends the entry to
    `state.budget_extensions` and returns it. It does not save.

Rescale algorithm (the spec fixes it; the signature does not determine it):

```python
old = effective_budget(budget, state.budget_extensions)
new = effective_budget(budget, state.budget_extensions + [entry])
s_old, s_new = spent_fraction(old, state.spend, now), spent_fraction(new, state.spend, now)
if s_old > 0:
    for c in state.chains:
        c.reheated_at *= s_new / s_old
```

- [ ] **Step 1: Write the failing tests**

`tests/test_budget.py`:

```python
def test_effective_budget_adds_every_extension_to_set_caps_only():
    # mutation: using only the last extension gives iterations 7
    # mutation: `(cap or 0) + add` turns the unset compute cap into 1.0
    b = Budget(usd=1.0, hours=2.0, iterations=5, compute_usd=None)
    ext = [{"add": {"iterations": 3, "hours": None, "usd": 0.5, "compute_usd": None}},
           {"add": {"iterations": 2, "hours": 1.5, "usd": None, "compute_usd": 1.0}}]
    assert effective_budget(b, ext) == Budget(usd=1.5, hours=3.5, iterations=10,
                                              compute_usd=None)
    assert effective_budget(b, []) == b
```

`tests/test_state.py`:

```python
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
```

`tests/test_extend.py` (new). Use a helper `job(iterations_cap, spent, hours=None,
started_at=0.0)` returning `(Budget(usd=None, hours=hours, iterations=iterations_cap,
compute_usd=None), JobState)`, where `state.spend = Spend(started_at=started_at,
iterations=spent)` and `state.iteration = spent`.

```python
def test_extend_appends_an_entry_and_lifts_the_cap():
    # mutation: not appending leaves the job exhausted
    b, st = job(5, 5)
    entry = extend(b, st, {"iterations": 2}, now=100.0, previous_status="exhausted")
    assert entry == {"at": 100.0, "iteration": 5, "previous_status": "exhausted",
                     "add": {"iterations": 2, "hours": None, "usd": None, "compute_usd": None}}
    assert st.budget_extensions == [entry]
    assert exhausted(effective_budget(b, st.budget_extensions), st.spend, 100.0) is None

def test_extend_refuses_when_another_cap_is_still_spent():
    # mutation: checking only the dimensions being added lets an hours-exhausted job through
    # mutation: appending before the check leaves a refused entry in state
    b, st = job(5, 5, hours=1.0)
    with pytest.raises(ExtensionRefused) as e:
        extend(b, st, {"iterations": 2}, now=3600.0, previous_status="exhausted")
    assert "still out of budget on hours" in str(e.value) and "--add-hours" in str(e.value)
    assert st.budget_extensions == []

def test_extend_refuses_a_cap_the_job_never_had():
    # mutation: adding to a None cap silently creates an hours limit the user never set
    b, st = job(5, 5)
    with pytest.raises(ExtensionRefused) as e:
        extend(b, st, {"hours": 1.0, "iterations": 1}, now=0.0, previous_status="exhausted")
    assert "--budget-hours" in str(e.value)
    assert st.budget_extensions == []

@pytest.mark.parametrize("value", [0, -1])
def test_extend_refuses_a_non_positive_addition(value):
    # mutation: `if v < 0` lets a zero extension through; `if not v` lets -1 through
    b, st = job(5, 4)
    with pytest.raises(ExtensionRefused) as e:
        extend(b, st, {"iterations": value}, now=0.0, previous_status="cancelled")
    assert "--add-iterations must be greater than 0" in str(e.value)

def test_extend_rescales_reheat_points_onto_the_new_budget():
    # 20 of 20 iterations spent (s_old = 1.0); +20 gives s_new = 0.5. By hand:
    # 0.8 * 0.5 / 1.0 = 0.4; progress (0.5 - 0.4) / 0.6 = 1/6; T = 0.05 * (1 - 0.9 / 6) = 0.0425
    # mutation: no rescale leaves 0.8 > s_new, progress clamps to 0 and T = t0 = 0.05
    # mutation: s_old / s_new instead gives 1.6
    b, st = job(20, 20)
    st.chains = [Chain(0, reheated_at=0.8), Chain(1, reheated_at=0.0)]
    extend(b, st, {"iterations": 20}, now=0.0, previous_status="exhausted")
    assert st.chains[0].reheated_at == pytest.approx(0.4)
    assert st.chains[1].reheated_at == 0.0
    assert temperature(AnnealSettings(), 0.5, st.chains[0].reheated_at) == pytest.approx(0.0425)

def test_extend_on_an_unspent_job_does_not_divide_by_zero():
    # mutation: dividing without the s_old > 0 guard raises ZeroDivisionError
    b, st = job(5, 0)
    st.chains = [Chain(0)]
    extend(b, st, {"iterations": 5}, now=0.0, previous_status="cancelled")
    assert st.chains[0].reheated_at == 0.0

def test_second_extension_rescales_against_the_extended_budget():
    # Review Focus 1. 10/10 spent, +10: s 1.0 -> 0.5, 0.5 -> 0.25. Then 15/20 spent
    # (s_old 0.75), +10: s_new 15/30 = 0.5, 0.25 * 0.5 / 0.75 = 1/6.
    # mutation: computing s_old from spec.budget alone (15/10, clamped to 1.0) gives 0.125
    b, st = job(10, 10)
    st.chains = [Chain(0, reheated_at=0.5)]
    extend(b, st, {"iterations": 10}, now=0.0, previous_status="exhausted")
    assert st.chains[0].reheated_at == pytest.approx(0.25)
    st.spend.iterations = st.iteration = 15
    extend(b, st, {"iterations": 10}, now=0.0, previous_status="cancelled")
    assert st.chains[0].reheated_at == pytest.approx(1 / 6)
    assert effective_budget(b, st.budget_extensions).iterations == 30

def test_extend_leaves_the_pending_job_alone():
    # Review Focus 4. mutation: clearing pending_job on extension orphans a job C3 still bills
    b, st = job(5, 5)
    st.pending_job = {"purpose": 6, "job_id": "c3-123"}
    extend(b, st, {"iterations": 1}, now=0.0, previous_status="paused")
    assert st.pending_job == {"purpose": 6, "job_id": "c3-123"}
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest -q -m "not live" tests/test_budget.py tests/test_state.py tests/test_extend.py`
Expected: FAIL with an `ImportError` for `effective_budget` / `talos.extend`, and an
`AttributeError` or `KeyError` for `budget_extensions`.

- [ ] **Step 3: Implement `effective_budget`, `ADD_DIMENSIONS`, the state field, and
  `talos/extend.py` as specified under Interfaces.**

`talos/extend.py` imports from `talos.budget`, `talos.lineage` (`spent_fraction`) and
`talos.state`. `talos/budget.py` must not import `talos.lineage`, because lineage already imports
budget.

- [ ] **Step 4: Run them to verify they pass**

Same command. Expected: all pass. Then run the line-length check on the touched files.

- [ ] **Step 5: Commit**

```bash
git add talos/budget.py talos/state.py talos/extend.py tests/test_budget.py tests/test_state.py tests/test_extend.py
git commit -m "budget extensions: effective budget, state field, extend()"
```

---

### Task 2: Every budget reader uses the effective budget

**Files:**
- Modify: `talos/loop.py` (`Loop._check_budget`, `Loop._spent`)
- Modify: `talos/cli.py` (`freeze_hardware` pre-probe check, `_status_line`)
- Test: `tests/test_loop.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `effective_budget(budget, extensions)` and `extend(...)` from Task 1.
- Produces: no new names. After this task, `grep -n "spec.budget" talos/loop.py talos/cli.py`
  shows only `effective_budget(...)` calls. Every other budget check (`_BudgetedBench`,
  `measure_baseline`, `talos/agentic.py`) goes through `Loop._check_budget` and needs no change.

- [ ] **Step 1: Write the failing tests**

`tests/test_loop.py` (uses the file's `make`, `hyp`, `edit`, `heldout_flat`, and the `annealing`
pattern for a second `Loop`):

```python
def test_loop_checks_and_anneals_against_the_extended_budget(tmp_path):
    # mutation: _spent reading spec.budget gives 1.0; _check_budget reading it raises
    b = Budget(usd=None, hours=None, iterations=4, compute_usd=None)
    loop, *_ = make(tmp_path, [], budget=b)
    loop.state.spend.iterations = 4
    loop.state.budget_extensions = [{"add": {"iterations": 4, "hours": None, "usd": None,
                                             "compute_usd": None}}]
    assert loop._spent() == pytest.approx(0.5)
    loop._check_budget()  # does not raise

def test_an_extended_exhausted_loop_runs_the_added_iterations(tmp_path):
    # mutation: any reader left on spec.budget stops the second run at once with 2 iterations
    # failed edits: each iteration is counted, nothing ever wins, two provider calls each
    script = [r for t in "abcd" for r in (hyp(t), "no edit blocks here")]
    b = Budget(usd=None, hours=None, iterations=2, compute_usd=None)
    th = Thresholds(distill=99)  # two provider calls per iteration, no distill call
    loop, _, _, store = make(tmp_path, script[:4], scores=heldout_flat, budget=b,
                             thresholds=th)
    assert loop.run().status == "exhausted"
    st = store.load()
    extend(b, st, {"iterations": 2}, now=0.0, previous_status="exhausted")
    st.status = "researching"  # as execute_job does
    again = Loop(store.read_spec(), st, store, FakeProvider(script[4:]),
                 FakeBench(heldout_flat), template_rs="x", clock=lambda: 0.0,
                 sleep=lambda s: None, thresholds=th)
    final = again.run()
    assert final.status == "exhausted" and final.stop_reason == "iterations"
    assert final.spend.iterations == 4
```


`tests/test_cli.py`:

```python
def test_status_line_counts_added_hours(tmp_path):
    # mutation: _status_line reading spec.budget.hours prints "0.0h left" (1h cap, 1.5h used)
    # (spec built as in test_status_line_reports_hours_left, with hours=1.0)
    state = JobState.fresh(Spend(started_at=0.0))
    state.budget_extensions = [{"add": {"iterations": None, "hours": 2.0, "usd": None,
                                        "compute_usd": None}}]
    assert cli._status_line(spec, state, 5400.0).endswith("| 1.5h left")

def test_freeze_hardware_checks_the_extended_budget(tmp_path, monkeypatch):
    # mutation: the pre-probe check reading spec.budget raises BudgetExhausted on a job whose
    # extension gave it room
    # spec: challenge "hypergraph" (a GPU challenge, so Modal has options to probe), budget
    # iterations=1; state: fresh, spend.iterations=1, baseline None, hardware None, one
    # extension adding 1 iteration
    monkeypatch.delenv("TALOS_HARDWARE", raising=False)  # freeze_hardware sets it; restore
    bench = types.SimpleNamespace(select_hardware=lambda ch, chosen=None: "L40",
                                  cost_mark=lambda: 0.0, cost_usd_since=lambda m: 0.0)
    cli.freeze_hardware(bench, spec, state, JobStore(tmp_path), "modal", log=lambda s: None,
                        clock=lambda: 0.0)
    assert state.hardware == "L40"
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest -q -m "not live" tests/test_loop.py tests/test_cli.py -k "extended or added_hours or freeze_hardware_checks"`
Expected: 4 FAIL. The two loop tests fail on the 1.0 spent fraction or on `BudgetExhausted`,
the status-line test reads `0.0h left`, and `freeze_hardware` raises `BudgetExhausted`.

- [ ] **Step 3: Replace `spec.budget` / `self.spec.budget` with
  `effective_budget(<spec>.budget, <state>.budget_extensions)` in the four readers.**

- [ ] **Step 4: Run the four tests, then the whole of both files**

Run: `.venv/bin/python -m pytest -q -m "not live" tests/test_loop.py tests/test_cli.py`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add talos/loop.py talos/cli.py tests/test_loop.py tests/test_cli.py
git commit -m "budget readers use the job's extended budget"
```

---

### Task 3: `--add-*` flags and the resume flow

**Files:**
- Modify: `talos/cli.py` (argparse for `run`, `cmd_run`, `execute_job`)
- Test: `tests/test_cli.py`. Also, any existing `monkeypatch.setattr(cli, "execute_job", ...)`
  stand-in that a `--resume` reaches gains an `add=None` parameter.

**Interfaces:**
- Consumes: `extend`, `ExtensionRefused`, `ADD_FLAGS`, `effective_budget` from Task 1.
- Produces: `execute_job(spec, store, cfg, resume: bool, add: dict[str, float] | None = None)
  -> int`. `cmd_run` passes `add=` only on the resume branch.

Behaviour, in order (spec §CLI flow):

1. At the top of `cmd_run`, before any config or network work: an `--add-*` flag whose value
   `is not None` (so `--add-iterations 0` counts as given), without `--resume`, exits 2 with
   `"{flag} only applies with --resume; use {BUDGET_FLAGS[d]} for a new job"` on stderr.
2. `add = {d: v for each dimension whose --add-* value is not None}`, with argparse types
   `int` for iterations and `float` for the rest.
3. In `execute_job` on resume, after the API-key check and the `won` refusal (unchanged):
   - `exhausted` with no `add`: print the existing "already exhausted; nothing to resume" line.
     When `ADD_FLAGS.get(state.stop_reason)` exists, also print
     `"to continue it, add budget with {flag}"`. Then rebuild the package and exit 1, as today.
   - Run the monorepo/dev-image check, unchanged.
   - For `cancelled`, `failed`, `paused`, `exhausted`: compute `paused_s` and shift
     `started_at` in memory, without saving yet.
   - If `add`: call `extend(spec.budget, state, add, time.time(), previous_status)`. On
     `ExtensionRefused`, print it to stderr and return 2. Nothing has been saved, so the files
     are untouched.
   - For the four statuses above, set `researching`, clear `stop_reason`, save, and write the
     `resumed` event as today. For any other status, save only if `add` was given.
   - If `add`: write the event `store.event("budget_extended", add=entry["add"],
     budget=<effective>.to_dict())` and print
     `"Budget extended: {additions}; budget now {effective.to_dict()}"`.

Test helper (in `tests/test_cli.py`):

```python
def stopped_job(tmp_path, monkeypatch, status="exhausted", stop_reason="iterations",
                extra=(), **state_kw):
    """A fake job whose 5-iteration cap is spent, stopped before any candidate; once resumed
    the fake provider wins on its first iteration."""
    # fake_config + fake_run(monkeypatch, extra); then rewrite state.json with
    # status, stop_reason, best=None, iteration=5, confirmed=[], hypotheses=[], nodes=[],
    # chains=[], spend.iterations=5, then state_kw; return run_dir
```

- [ ] **Step 1: Write the failing tests**

```python
def test_resume_continues_an_exhausted_job_with_added_iterations(tmp_path, monkeypatch, capsys):
    # mutation: the old refusal of exhausted jobs exits 1 "nothing to resume"
    # mutation: rewriting job.json to hold the new budget breaks invariant 1's write-once file
    run_dir = stopped_job(tmp_path, monkeypatch)
    spec_bytes = (run_dir / "job.json").read_bytes()
    capsys.readouterr()
    assert cli.main(["run", "--resume", run_dir.name, "--add-iterations", "2"]) == 0
    assert "Status: won" in capsys.readouterr().out
    st = json.loads((run_dir / "state.json").read_text())
    assert st["spend"]["iterations"] == 6
    [entry] = st["budget_extensions"]
    assert entry["previous_status"] == "exhausted" and entry["add"]["iterations"] == 2
    ev = [json.loads(ln) for ln in (run_dir / "timeline.jsonl").read_text().splitlines()]
    assert [e["budget"]["iterations"] for e in ev if e["kind"] == "budget_extended"] == [7]
    assert "resumed" in [e["kind"] for e in ev]
    assert (run_dir / "job.json").read_bytes() == spec_bytes

def test_exhausted_resume_without_added_budget_names_the_flag(tmp_path, monkeypatch, capsys):
    # mutation: dropping the hint leaves the user with "nothing to resume" and no way forward
    run_dir = stopped_job(tmp_path, monkeypatch)
    capsys.readouterr()
    assert cli.main(["run", "--resume", run_dir.name]) == 1
    out = capsys.readouterr().out
    assert "already exhausted; nothing to resume" in out and "--add-iterations" in out

def test_exhausted_resume_with_an_unknown_reason_does_not_crash(tmp_path, monkeypatch, capsys):
    # Review Focus 3. mutation: ADD_FLAGS[state.stop_reason] raises KeyError
    run_dir = stopped_job(tmp_path, monkeypatch, stop_reason="budget")
    assert cli.main(["run", "--resume", run_dir.name]) == 1
    assert "already exhausted; nothing to resume" in capsys.readouterr().out

def test_extension_still_out_of_budget_is_refused_untouched(tmp_path, monkeypatch, capsys):
    # mutation: saving the shift or the entry before extend() checks changes state.json
    # 4 h cap, started 5 h ago, saved just now (no pause to shift): still out on hours
    run_dir = stopped_job(tmp_path, monkeypatch, stop_reason="hours",
                          extra=("--budget-hours", "4"))
    st = json.loads((run_dir / "state.json").read_text())
    st["spend"]["started_at"] = time.time() - 5 * 3600
    (run_dir / "state.json").write_text(json.dumps(st))
    before = {n: (run_dir / n).read_bytes() for n in ("state.json", "timeline.jsonl")}
    assert cli.main(["run", "--resume", run_dir.name, "--add-iterations", "2"]) == 2
    err = capsys.readouterr().err
    assert "still out of budget on hours" in err and "--add-hours" in err
    assert before == {n: (run_dir / n).read_bytes() for n in before}

def test_hours_exhausted_job_continues_with_added_hours(tmp_path, monkeypatch, capsys):
    # Review Focus 5. 4 h cap, started 5 h ago, last saved 1 h ago: the shift gives 4 h used,
    # and +1 h leaves room.
    # mutation: not shifting started_at for an exhausted job counts 5 h, refused with exit 2
    run_dir = stopped_job(tmp_path, monkeypatch, stop_reason="hours",
                          extra=("--budget-hours", "4"))
    st = json.loads((run_dir / "state.json").read_text())
    st["spend"]["iterations"] = 0  # only hours is spent
    st["spend"]["started_at"] = time.time() - 5 * 3600
    (run_dir / "state.json").write_text(json.dumps(st))
    saved = time.time() - 3600
    os.utime(run_dir / "state.json", (saved, saved))
    assert cli.main(["run", "--resume", run_dir.name, "--add-hours", "1"]) == 0

def test_add_flag_for_a_cap_the_job_never_had_is_refused(tmp_path, monkeypatch, capsys):
    # mutation: creating a new hours cap the user never set
    run_dir = stopped_job(tmp_path, monkeypatch)
    before = (run_dir / "state.json").read_bytes()
    assert cli.main(["run", "--resume", run_dir.name, "--add-hours", "1"]) == 2
    assert "--budget-hours" in capsys.readouterr().err
    assert (run_dir / "state.json").read_bytes() == before

def test_add_flags_need_resume(tmp_path, monkeypatch, capsys):
    # mutation: ignoring --add-iterations on a new job silently drops the user's budget
    monkeypatch.chdir(tmp_path)
    fake_config(tmp_path)
    assert fake_run(monkeypatch, extra=("--add-iterations", "2")) == 2
    assert "only applies with --resume" in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()

def test_extension_raises_the_cap_of_a_cancelled_job(tmp_path, monkeypatch, capsys):
    # mutation: applying extensions to exhausted jobs only resumes this one at 5/5, exit 1
    run_dir = stopped_job(tmp_path, monkeypatch, status="cancelled",
                          stop_reason="user requested stop")
    assert cli.main(["run", "--resume", run_dir.name, "--add-iterations", "1"]) == 0
    st = json.loads((run_dir / "state.json").read_text())
    assert st["budget_extensions"][0]["previous_status"] == "cancelled"

def test_hard_killed_job_is_extended_without_a_clock_shift(tmp_path, monkeypatch, capsys):
    # mutation: shifting started_at for every status moves a hard-killed job's clock by the
    # hour since its last save
    run_dir = stopped_job(tmp_path, monkeypatch, status="researching", stop_reason=None)
    st = json.loads((run_dir / "state.json").read_text())
    started = st["spend"]["started_at"]
    saved = time.time() - 3600
    os.utime(run_dir / "state.json", (saved, saved))
    assert cli.main(["run", "--resume", run_dir.name, "--add-iterations", "1"]) == 0
    st = json.loads((run_dir / "state.json").read_text())
    assert st["spend"]["started_at"] == started
    assert st["budget_extensions"][0]["previous_status"] == "researching"

def test_job_exhausted_before_its_baseline_measures_it_once(tmp_path, monkeypatch, capsys):
    # mutation: an extension path that skips measure_baseline when the status was exhausted
    run_dir = stopped_job(tmp_path, monkeypatch, baseline=None)
    n_before = len((run_dir / "timeline.jsonl").read_text().splitlines())
    assert cli.main(["run", "--resume", run_dir.name, "--add-iterations", "1"]) == 0
    new = [json.loads(ln)["kind"]
           for ln in (run_dir / "timeline.jsonl").read_text().splitlines()[n_before:]]
    assert new.count("baseline_ready") == 1

def test_add_flags_on_a_won_job_record_nothing(tmp_path, monkeypatch, capsys):
    # Review Focus 2. mutation: extending before the won refusal records an entry on a job
    # that never runs again
    monkeypatch.chdir(tmp_path)
    fake_config(tmp_path)
    assert fake_run(monkeypatch) == 0
    run_dir = next((tmp_path / "runs").glob("*/job.json")).parent
    assert cli.main(["run", "--resume", run_dir.name, "--add-iterations", "1"]) == 1
    assert "already won" in capsys.readouterr().out
    assert json.loads((run_dir / "state.json").read_text()).get("budget_extensions", []) == []
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest -q -m "not live" tests/test_cli.py -k "added or extension or add_flag or exhausted_resume or hard_killed or before_its_baseline or need_resume"`
Expected: every test fails or errors. Until the flags exist, argparse raises `SystemExit: 2`
on `--add-*` (`cli.main` does not catch it), and the plain-resume tests fail on the missing
hint.

- [ ] **Step 3: Add the four argparse flags, the `cmd_run` guard, the `add` argument to
  `execute_job`, and the resume flow above.** Add `add=None` to every resume-reachable
  `execute_job` stand-in in the existing tests.

- [ ] **Step 4: Run the new tests, then the full gate**

Run: `make check PYTHON=.venv/bin/python`
Expected: ruff clean, `889 passed, 7 deselected` (ESTIMATE (unverified): 863 + 11 + 4 + 11 new
test items from Tasks 1–3), agentify `level 1: PASS`.

- [ ] **Step 5: Commit**

```bash
git add talos/cli.py tests/test_cli.py
git commit -m "talos run --resume --add-*: continue an exhausted job with more budget"
```

---

### Task 4: Package README and docs

**Files:**
- Modify: `talos/package.py` (`_readme`, `_readme_no_candidate`)
- Modify: `README.md`, `docs/architecture.md`
- Test: `tests/test_package.py`

**Interfaces:**
- Consumes: `effective_budget`, `ADD_DIMENSIONS` from Task 1.
- Produces: `talos.package._budget_section(spec: JobSpec, state: JobState) -> str`. It returns
  `""` with no extensions. Otherwise, exactly:

```
## Budget extensions

- Started with: {render(spec.budget)}
- Extended at iteration {e["iteration"]} (was {e["previous_status"]}): {additions}
- Now: {render(effective_budget(spec.budget, state.budget_extensions))}

```

  Here `render(b)` is `", ".join(f"{d} {getattr(b, d)}" for d in ("usd", "hours",
  "iterations", "compute_usd") if getattr(b, d) is not None)`, and `additions` is
  `", ".join(f"+{v} {d}" ...)` over the non-`None` entries of `e["add"]` in `ADD_DIMENSIONS`
  order. There is one "Extended" line per entry. Both READMEs include the section right after
  their `Status: ... Reason: ...` line.

- [ ] **Step 1: Write the failing tests**

```python
def test_readme_reports_budget_extensions(tmp_path):
    # mutation: omitting the section hides that the job ran on more budget than job.json says
    spec, st, store = make(tmp_path, status="exhausted")  # budget usd=1.0
    st.budget_extensions = [{"at": 0.0, "iteration": 3, "previous_status": "exhausted",
                             "add": {"iterations": None, "hours": None, "usd": 0.5,
                                     "compute_usd": None}}]
    text = (build_package(spec, st, store) / "README.md").read_text(encoding="utf-8")
    assert "- Started with: usd 1.0\n" in text
    assert "- Extended at iteration 3 (was exhausted): +0.5 usd\n" in text
    assert "- Now: usd 1.5\n" in text

def test_readme_without_extensions_has_no_budget_section(tmp_path):
    # mutation: always printing the section
    spec, st, store = make(tmp_path, status="exhausted")
    assert "Budget extensions" not in (build_package(spec, st, store) / "README.md").read_text(
        encoding="utf-8")

def test_no_candidate_readme_reports_budget_extensions(tmp_path):
    # mutation: adding the section to the candidate README only
    spec, st, store = make(tmp_path, status="exhausted")
    st.best = None
    st.budget_extensions = [{"at": 0.0, "iteration": 3, "previous_status": "exhausted",
                             "add": {"iterations": None, "hours": None, "usd": 0.5,
                                     "compute_usd": None}}]
    assert "- Now: usd 1.5\n" in (build_package(spec, st, store) / "README.md").read_text(
        encoding="utf-8")
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest -q -m "not live" tests/test_package.py`
Expected: 2 FAIL (the two "reports" tests). The "has no budget section" test passes now; it
guards against the mutation.

- [ ] **Step 3: Implement `_budget_section` and include it in both READMEs. Update the docs:**
  - `README.md`, table of `talos run` flags: the `--resume JOB_ID` row says it also continues
    an `exhausted` job given an `--add-*` flag. Add one row each for `--add-iterations N`,
    `--add-hours N`, `--add-usd N`, `--add-compute-usd N` ("Adds N to the job's cap. Only
    with `--resume`; refused for a cap the job was started without.").
  - `README.md#resume`: one paragraph with the example
    `talos run --resume 20260917-101006-knapsack --add-iterations 20`. Say that nothing is
    re-measured, and that annealing continues at the temperature for the new total.
  - `README.md#budget`: one bullet saying extensions add to the caps, and that the time between
    a stop and the continuation is not counted against `--budget-hours`.
  - `README.md` run-directory table: the `state.json` row mentions `budget_extensions`, and
    the `timeline.jsonl` row mentions `budget_extended`.
  - `README.md` statuses paragraph (`exhausted` (a budget ran out)): add "resumable with
    `--add-*`".
  - `docs/architecture.md`: beside the stop diagram (the line with
    `Status: exhausted, cancelled (Ctrl-C), failed or paused`), say that an exhausted job
    resumes with added budget via `talos/extend.py::extend`.

- [ ] **Step 4: Run the full gate**

Run: `make check PYTHON=.venv/bin/python`
Expected: ruff clean, `892 passed, 7 deselected` (ESTIMATE (unverified): 889 + 3), and agentify
`level 1: PASS`, which
includes every doc reference resolving. Run the line-length check on the touched files.

- [ ] **Step 5: Commit**

```bash
git add talos/package.py tests/test_package.py README.md docs/architecture.md
git commit -m "package and docs: report budget extensions"
```
