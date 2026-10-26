# Continue an exhausted run — design

Date: 2026-10-26. Non-authoritative (see `docs/ai/README.md`).

## Intent

A job that stops `exhausted` has a measured baseline, frozen nonces, a frozen hardware class,
a calibration, a best candidate and a lineage tree. Today `talos run --resume` refuses it
(`talos/cli.py::execute_job`, "already exhausted; nothing to resume"), so getting more
iterations means starting a new job and measuring the baseline again. This feature lets the
user give an exhausted job more budget and continue it where it stopped.

## Goals

- `talos run --resume JOB_ID` accepts four new flags, each adding to one cap of the job's
  budget:
  - `--add-iterations N` (int)
  - `--add-hours N` (float)
  - `--add-usd N` (float, the LLM cap)
  - `--add-compute-usd N` (float)
- An `exhausted` job given enough added budget resumes and runs more iterations with
  everything else it already had: baseline, nonces, fuel, track, hyperparameters, hardware,
  scoring and calibration, best candidate, hypotheses, lineage nodes and chains.
- The same flags work on any job `--resume` already accepts, so a job can be given more budget
  without first running out. That covers `cancelled`, `failed` and `paused`, and also a job a
  hard kill left in a non-terminal status (`queued`, `measuring_baseline`, `researching`,
  `confirming`). *Our call:* a hard-killed job keeps its current resume behaviour, with no
  `started_at` shift and no `resumed` event. Its extension entry records that status as
  `previous_status`, and the annealing rescale runs without a shift.
- An `exhausted` job that stopped before its baseline was measured can also be extended. This
  happens when the budget ran out before the hardware probe (`talos/cli.py::freeze_hardware`)
  or during the baseline. *Our call:* the continuation runs the probe, the baseline and the
  calibration for the first time, exactly as a fresh start would. Each is measured once.
- Every extension is recorded and visible.

## Non-goals

- Changing anything other than the budget on resume. The existing refusals (mode, track,
  hyperparameters, nonces, scoring, start, search, chains) stay as they are.
- Continuing a `won` job. It stays refused, as today.
- Lowering a cap, or setting an absolute new total. The user chose increments.
- Adding a cap the job never had. If the job was started without `--budget-hours`, then
  `--add-hours` is refused: the job has no hours cap to add to.

## Anti-goals

- `job.json` must stay write-once (`talos/state.py::JobStore.write_spec`). The extension must
  not rewrite it.
- A resume must never re-measure the baseline or re-calibrate because of an extension
  (AGENTS.md invariant 1).
- An extension must not, by itself, reheat an annealing job to full temperature (see below).

## Decisions made with the user

1. **Increments, not totals.** `--add-iterations 20` means 20 more than the current cap.
2. **Annealing continues from the new total.** The temperature after an extension is the one
   the job would have at its current spend if the larger budget had been its budget from the
   start. Example: a job exhausted at 10 of 10 iterations, extended by 10, continues at the
   temperature for a spent fraction of 0.5.

## Design

### Where the extension lives

`JobState` gets a new field `budget_extensions: list[dict]`, default `[]`. Old `state.json`
files load with `[]`. Each entry is:

```json
{"at": 1761480000.0, "iteration": 8, "previous_status": "exhausted",
 "add": {"iterations": 20, "hours": null, "usd": null, "compute_usd": null}}
```

A pure function `effective_budget(spec.budget, state.budget_extensions) -> Budget` in
`talos/budget.py` returns the job's cap plus every extension, dimension by dimension. A
dimension whose cap is `None` stays `None`. The CLI refuses an addition to such a cap before
an entry is written, so this case never appears in stored data.

Every place that reads `spec.budget` to check or show the budget switches to the effective
budget:

- `talos/loop.py::Loop._check_budget`
- `talos/loop.py::Loop._spent` (the annealing schedule)
- `talos/cli.py::freeze_hardware` (the pre-probe check)
- `talos/cli.py::_status_line` (the "left" figure)

The `Job ...: budget {...}` line printed at job start reports the original budget, because it
is printed only for new jobs.

### CLI flow

In `talos/cli.py` (the `--resume` branch and `execute_job`):

1. Any `--add-*` flag without `--resume` is refused, exit 2:
   "`--add-iterations` only applies with `--resume`; use `--budget-iterations` for a new job".
2. Each `--add-*` value must be > 0, or the command is refused with exit 2.
3. An `--add-*` flag for a dimension whose `spec.budget` cap is `None` is refused with exit 2,
   naming the dimension.
4. A `won` job is refused as today.
5. Whenever `--add-*` flags are given, whatever the status: compute the effective budget with
   the new additions. If
   `exhausted(new_budget, state.spend, now_after_shift)` still returns a dimension, refuse with
   exit 2 and name it ("job is still out of budget on hours; add --add-hours"). For an
   `exhausted` job with no `--add-*` flags at all, keep the current message and add a hint
   naming the exhausted dimension's flag. A non-exhausted resume with no flags behaves as
   today.
6. The monorepo/dev-image check runs before anything is written, as it does for other
   resumes.
7. Treat `exhausted` like the other resumable statuses: shift `spend.started_at` forward by
   the time since the last save. The time between exhaustion and continuation does not count
   against `--budget-hours`, exactly as for a paused job. Append the extension entry, set the
   status to `researching`, clear `stop_reason`, save, and write the `resumed` event as today.
8. If any `--add-*` flag was given, write a `budget_extended` timeline event with the
   additions and the resulting effective budget, and print one line saying the same.

Step 5 needs the shifted `started_at` for the hours check. Compute the shift first, check
against it, and save only if the check passes. A refused extension leaves `state.json`
untouched.

### Annealing: rescaling `reheated_at`

`Chain.reheated_at` is the spent fraction at the chain's last reheat, measured against the
budget in force at that moment. `temperature()` measures progress from `reheated_at` to 1.
After an extension the current spent fraction drops, so it can fall below `reheated_at`.
`progress` then clamps to 0 and the chain runs at `t0`, which is a full reheat and contradicts
decision 2.

At the moment of extension, for each chain: compute `s_old = spent_fraction(old_budget, ...)`
and `s_new = spent_fraction(new_budget, ...)` (both after the `started_at` shift), and set
`chain.reheated_at *= s_new / s_old` when `s_old > 0`. This maps the reheat point onto the new
budget's scale. It is exact when one dimension sets the spent fraction both before and after
the extension. Otherwise (a different dimension takes over the max) it is an approximation.
*Builder's call within that: any rule that keeps `reheated_at <= s_new` and preserves the
ratio in the single-dimension case is acceptable.* The rescale does not apply to a greedy job.
It is a no-op for a chain that never reheated (`reheated_at == 0`).

### Package and status

`talos/package.py` builds its report from `state`. When `budget_extensions` is non-empty, the
report states the original budget, each extension, and the effective budget. The package is
rebuilt when the job stops again, as it is for any resume. `talos status` needs no change
beyond the status value it already shows.

### Docs

- README: the `--resume JOB_ID` row says it also continues an exhausted job when given
  `--add-*`. Add one row per new flag, and a short paragraph under "Resume".
- README "Budget": add a line saying extensions add to the cap and that time between
  exhaustion and continuation is not counted.
- README run-directory table: `state.json` mentions `budget_extensions`, and `timeline.jsonl`
  mentions `budget_extended`.
- `docs/architecture.md`: wherever it lists terminal statuses or describes resume, say that
  `exhausted` is resumable with added budget.

## Constraints

- Python 3.10 venv. Lint per `make check`.
- Tests run with `-m "not live" --ignore=tests/test_live.py` (an unguarded live test bills
  Modal). CLI tests use the fake provider and bench; no real backend is touched.
- `job.json` is not rewritten. A test asserts its bytes are unchanged after an extended resume.

## What is left to the builder

- Exact message wording, within the content given above.
- Whether `effective_budget` takes the extension list or a pre-summed dict.
- Test layout, beyond these behaviours, each of which needs a test that fails without the
  feature:
  - Exhausted on iterations, given `--add-iterations 2`: two more iterations run.
  - Still exhausted after the additions: refused, and `state.json` unchanged.
  - `--add-hours` on a job with no hours cap: refused.
  - `--add-*` without `--resume`: refused.
  - Extension on a cancelled job raises the cap.
  - `job.json` bytes are unchanged.
  - Annealing: after an extension, the temperature equals `temperature(settings, s_new,
    rescaled reheated_at)` and is below `t0` for a chain that had cooled.
  - An old `state.json` without `budget_extensions` loads.
  - A job exhausted before its baseline, once extended, measures the baseline exactly once
    (fake bench call count) and then iterates.
  - A hard-killed `researching` job given `--add-iterations` gets the extension with no
    `started_at` shift.
