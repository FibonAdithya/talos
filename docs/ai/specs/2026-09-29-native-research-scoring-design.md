# Native research scoring — design

Date: 2026-09-29. Status: design approved section by section in conversation; not yet
implemented. Branch: to be cut from `main` once PR #29 (agentic permission fix) has merged.

This spec changes how the research loop scores candidates. Today every candidate is built with
TIG's fuel-instrumented toolchain and run under `tig-runtime` with a fuel limit. After this
change, research candidates are built and run natively, with fuel approximated by wall time
calibrated against the baseline. The TIG metered path is kept for validating each new best
before it counts.

Spec B (parallel edits and scoring, including validation that runs beside research instead of
blocking it) builds on this one and is tracked separately.

## 1. Problem

The metered path is slow, and about half of it goes on metering, not searching.

Where the time went in run `20260928-093637-hypergraph` (tig-adi, 20 iterations, MEASURED from
its `timeline.jsonl`):

| phase | iterations | total | mean |
|---|---|---|---|
| agent editing, successful | 14 | 258 min | 18.4 min |
| agent editing, failed | 6 | 71 min | 11.8 min |
| scoring on C3 | 14 | 301 min | 21.5 min |

A probe on C3 (job `job_1790678981167_ip26xh`, L40, TIG hypergraph dev image 0.0.7, monorepo
`84a5787`) compared the two paths on the same baseline (`sigma_freud_opt`) and the same five
training nonces (one per track). All MEASURED in that job unless marked:

| step | TIG metered | native |
|---|---|---|
| build | 286.6 s (`build_algorithm`) | 43.9 s cold, 27.7 s warm after touching one file; PTX 3.7 s |
| 10k nonce | 67.9 s | 41.3 s |
| 20k nonce | 72.2 s | 30.4 s |
| 50k nonce | 106.0 s | 36.0 s |
| 100k nonce | 153.8 s | 54.2 s |
| 200k nonce | 209.2 s | 72.4 s |
| 5 nonces total | 609 s | 234 s |

Quality was identical on every track across four runs: the morning's Talos baseline job
(metered), the probe's metered run, and two native runs. Scoring is deterministic per instance,
and the native path returns the metered path's quality exactly.

Why it is faster:

- **Build.** `tig-binary/scripts/build_so` recompiles the standard library, `tig_challenges` and
  the algorithm to LLVM IR on one codegen unit (`-Z build-std`, `-C codegen-units=1`) and runs a
  fuel-instrumentation pass over every basic block. A native build is plain `cargo build`, with
  LTO off and 16 codegen units, on all cores.
- **Run.** `tig-binary/scripts/build_ptx` rewrites every PTX basic block to count fuel, and the
  CPU code decrements a fuel counter per block. Native code does neither.

## 2. Decisions taken during design

| # | decision | chosen | alternatives rejected |
|---|---|---|---|
| D1 | What runs research scoring | a native harness inside the existing TIG dev image | a warm long-lived TIG container (saves only per-job overhead); a separate lean image (the pull was not the bottleneck: 28 s from submission to script start on a node with the image cached) |
| D2 | Where it runs | the existing per-candidate jobs on C3, Modal and local docker | a long-lived C3 job: idles while the agent edits, and must work on Modal too |
| D3 | Stand-in for the fuel limit | native wall time, calibrated against the baseline's metered `fuel_consumed` and native runtime | none; fuel cannot be counted without instrumentation |
| D4 | How calibration follows the machine | once per (challenge, track, hardware type, …), reused by every later job on that hardware | re-running the baseline in every job (+234 s per hypergraph job); per type plus a drift check |
| D5 | When the TIG docker validates | every new best; the loop builds on the last validated best | end of run only; top-k at the end |
| D6 | Validation blocks research or not | blocks, in this spec; spec B moves it into the background | — |

## 3. Architecture

The container image and backends stay as they are. What runs inside the container gains a
second mode, selected by the payload every backend already ships:

| | `metered` (today) | `native` (new) |
|---|---|---|
| build | `build_algorithm` | `cargo build --release -p talos-native` (LTO off, 16 codegen units) + PTX without fuel injection |
| run one nonce | `tig-runtime … --fuel F` | the `talos-native` binary, killed at the track's time budget |
| quality | `tig-verifier` | `tig-verifier` (unchanged, reading the same solution file format) |
| records | quality, runtime, and `fuel_consumed` (new) | quality, runtime |

The loop scores research candidates in `native` mode. A candidate that becomes the new best is
scored again in `metered` mode, on training and held-out nonces, before it is accepted.

## 4. Components

### 4.1 Native runner (`talos/native_runner/`)

Templates for a Cargo package added to the monorepo workspace inside the container, filled in
the way TIG's `build_so` fills in `tig-binary/src/entry_point.rs`: `{CHALLENGE}` and
`{ALGORITHM}` substituted. Two variants, because `solve_challenge` takes different arguments:

- **GPU** (`hypergraph`, `vector_search`, `neuralnet_optimizer`): creates the CUDA context,
  loads the unmetered PTX, calls `Challenge::generate_instance(seed, track, module, stream,
  prop)`, launches `initialize_kernel` with seed bytes 8..16 (as `tig-runtime` does), then calls
  `<algorithm>::solve_challenge(challenge, save_solution, hyperparameters, module, stream, prop)`.
- **CPU** (`knapsack`, `job_scheduling`, …): `Challenge::generate_instance(seed, track)`, then
  `solve_challenge(challenge, save_solution, hyperparameters)`.

Both derive the seed with `BenchmarkSettings::calc_seed(rand_hash, nonce)`, keep the last saved
solution, and write `{"nonce": n, "solution": "<solution json>"}`, the shape `tig-verifier`
reads. The runner cannot compute quality itself: `tig-algorithms` enables
`tig-challenges/hide_verification`, which makes `evaluate_solution` private. Dependencies are
only crates already in the pinned `Cargo.lock` (`anyhow`, `serde_json`, `cudarc`,
`tig-algorithms`, `tig-challenges`, `tig-structs`, `tig-utils`).

The probe's GPU runner compiled and produced identical quality. The CPU variant is written but
not yet measured.

### 4.2 In-container functions (`talos/inside.py`)

- `build_native(monorepo, challenge, name) -> (ok, output)`: stage the runner, add it to the
  workspace members, build the PTX with the fuel-injection step removed (for GPU challenges),
  and `cargo build --release -p talos-native` with `CARGO_PROFILE_RELEASE_LTO=false` and
  `CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16`.
- `run_nonce_native(...)`: run the binary under the track's time budget, then `tig-verifier`
  with the same PTX the metered path uses. Returns the same row shape as `run_nonce`.
- `run_nonce` (metered) also reads `fuel_consumed` from `tig-runtime`'s output file and returns
  it. `NonceResult` gains an optional `fuel_consumed: int | None`.

### 4.3 Request and payload

`EvalRequest` gains `mode: "metered" | "native"` (default `"metered"`) and
`budgets: dict[str, float] | None` (seconds per track, native mode only). They are written into
the C3 payload and the Modal call, and `c3_job.main`, `_compile_impl` and `_score_batch_impl`
branch on `mode`. The Modal artifact volume keys native artifacts separately from metered ones
(the content hash includes the mode).

### 4.4 Calibration (`talos/calibration.py`)

A record per key
`(challenge, track, hardware, monorepo_ref, dev_image_tag, baseline_content_hash)`, stored as
JSON under `~/.talos/calibration/` beside the existing `~/.talos/baselines/` cache
(`talos/cli.py::BASELINE_CACHE`). The baseline's hash is part of the key, because the ratio of
fuel to time depends on the code that produced it.

For each track, over that track's training nonces, where `fuel_consumed` and native runtime
come from the same nonces:

```
ratio      = median(native_ms_i / fuel_consumed_i)      over nonces with fuel_consumed_i > 0
fuel_ms    = ratio × max_fuel × margin                   # native time equal to the fuel cap
ceiling_ms = runtime_ceiling × max(native_ms_i)          # today's "fail slow candidates fast"
budget_s   = min(NONCE_TIMEOUT_S,
                 max(runtime_floor_s, min(fuel_ms, ceiling_ms) / 1000))
```

`runtime_ceiling` and `runtime_floor_s` are the loop's existing settings (`talos/loop.py`,
`Loop._timeouts`, today applied to metered runtimes). The outer `NONCE_TIMEOUT_S` cap stays
because `talos/c3_jobdir.py` sizes a C3 job's time limit from it; a budget above it could
outlive the job.
`margin` starts at 0.8; that value is an ESTIMATE with no measurement behind it. The record also
keeps which bound was binding, so a native nonce killed at the budget is reported as
`out_of_fuel` when `fuel_ms` was the binding bound and as `timeout` when `ceiling_ms` or the
`NONCE_TIMEOUT_S` cap was, the same two errors the metered path reports.

A track whose calibration has no nonce with `fuel_consumed > 0` gets no budget (no division by
zero), which triggers the metered fallback in §6.

### 4.5 Loop (`talos/loop.py`, `talos/state.py`)

- `JobState` gains `validated_best: Candidate | None` and `validations: list[dict]`, one entry
  per validation attempt with the iteration, the outcome and the reason.
- `_current_files()` returns `validated_best` when it is set, else the baseline. Research
  builds only on validated code.
- A native score that the loop's existing improvement rule (`talos/loop.py`, the `improved`
  test in `_score_candidate`) counts as better than `validated_best` triggers validation (§5,
  step 3). It becomes `best` only as `validated_best`. That rule currently compares means only;
  changing it to respect the worst track is a separate request, and this spec uses whatever
  rule is in place.
- Held-out confirmation (`_confirm`) runs on the metered validation result, so `won` keeps
  meaning "passed TIG's own runtime on training and held-out".

### 4.6 `talos compile`

The agent's check that its edit compiles (`talos/cli.py::cmd_compile`) uses native mode by
default: 44 s instead of 287 s (MEASURED, hypergraph, cold). `--metered` keeps the old path. A
candidate that compiles natively but fails the metered build is caught at validation (§6).

## 5. Data flow for one job

1. **Calibrate.** Look up the record for this job's key. If there is none, score the baseline
   on the training nonces in both modes and store the record. This is the only extra cost, paid
   once per hardware type and baseline.
2. **Research.** Each candidate is built and scored natively with the per-track budgets.
3. **New best.** A candidate that the improvement rule (§4.5) ranks above `validated_best` is
   scored in metered mode on training and held-out nonces. It passes only if all of these hold:
   - no training nonce errors with `out_of_fuel`;
   - metered quality equals native quality on every training nonce that is ok in both modes;
   - the existing `beats()` rule passes on training.
4. **Accept or demote.** On a pass it becomes `validated_best`, and the held-out decision runs as
   today. On a fail it is demoted, the reason is recorded in `validations` and in the
   calibration record, and the next iteration builds on the previous `validated_best`.

## 6. Error handling

| case | handling |
|---|---|
| native build fails | a compile failure for the iteration; the output goes back to the agent as today |
| native build passes, metered build fails | validation fails with reason `native_metered_build_mismatch` |
| native and metered quality differ on an ok nonce | demote with reason `nondeterministic`, both qualities recorded |
| native nonce exceeds its budget | `out_of_fuel` or `timeout` depending on the binding bound (§4.4); counts toward the error ceiling |
| validation reports `out_of_fuel` | demote with reason `fuel_proxy_miss`; after 3 misses on a track (an ESTIMATE of a sensible threshold), that track's `margin` drops by 0.1 in the stored record |
| no calibration record and none can be made (baseline unscoreable, no `fuel_consumed`) | fall back to metered research scoring for this job and say so in the run log |
| the CPU runner variant fails to build for a challenge | that challenge stays metered-only until the runner is fixed |

## 7. Testing

Unit tests run in CI with fakes only. For each, the mutation it must catch:

| test | mutation caught |
|---|---|
| budget formula over fixed per-nonce inputs, with literal expected seconds | mean instead of median; margin dropped; ratio inverted; `min` replaced by `max` |
| a nonce with `fuel_consumed = 0` gets no budget, and the track falls back | division by zero; a zero budget that fails every candidate |
| `fuel_consumed` parsed from a sample `tig-runtime` output file | left as `None`, which would leave calibration without data |
| native nonce over budget → `out_of_fuel` when the fuel bound binds, `timeout` when the ceiling binds | budget kill reported as `panic` or as success; the two bounds swapped |
| runner templates render for `knapsack` (CPU) and `hypergraph` (GPU) with the right `solve_challenge` arguments | wrong variant chosen; placeholder left unfilled |
| payload `mode` routes `c3_job.main` and the Modal functions | `mode` ignored, so every job is metered |
| Modal artifact hash includes the mode | a native artifact served for a metered request |
| loop: native best → validation pass → `validated_best`; each failure reason demotes, and the next iteration's files are the previous `validated_best` | building on an unvalidated best; a demotion that leaves `best` pointing at the rejected candidate |
| calibration key includes hardware, monorepo ref, image tag and baseline hash | an L40 budget reused on another GPU or after a baseline change |
| no calibration → metered fallback, logged | a native run with no budget |

Live tests, opt-in and run by hand because they spend credit:

- C3 hypergraph: calibrate and score the baseline natively, and assert native quality equals
  metered quality on every nonce (today's probe, through Talos's own code).
- C3 knapsack: the first measurement of a CPU challenge.
- Modal hypergraph: the same parity check on Modal's GPU types.

New tests are mutation-checked after implementation with the bytecode cache isolated.

## 8. Rollout

1. Land behind `talos run --scoring native|metered`, default `metered`.
2. Run the three live tests. A challenge whose native quality differs from metered stays
   metered.
3. Run one hypergraph night on tig-adi with `--scoring native`, with the same direction and
   budget as a recent metered night. Compare iteration wall time, validation pass rate and
   fuel-proxy misses.
4. Switch the default to `native` for each challenge that passed.

## 9. Out of scope

- Parallel edits and scoring, and background validation (spec B).
- Carrying knowledge across runs, and a knowledge base of what worked (separate request).
- More nonces per track. Native speed makes this affordable, and determinism (§1) makes it the
  way to reduce uncertainty, but the choice of counts is a separate decision.

## 10. Claims

| claim | value | MEASURED / ESTIMATED | source | when |
|---|---|---|---|---|
| time split of run 20260928-093637-hypergraph | 258 / 71 / 301 min | MEASURED | its `timeline.jsonl` on tig-adi | 2026-09-29 |
| metered build, hypergraph | 286.6 s | MEASURED | probe job_1790678981167_ip26xh `probe.json` | 2026-09-29 |
| native build cold / warm / PTX | 43.9 / 27.7 / 3.7 s | MEASURED | same | 2026-09-29 |
| 5-nonce scoring metered / native | 609 / 234 s | MEASURED | same | 2026-09-29 |
| identical quality across 4 runs, all tracks | yes | MEASURED | same, plus run 20260929-073341-hypergraph baseline `results.json` | 2026-09-29 |
| submission to script start | 28 s | MEASURED | same job's C3 log | 2026-09-29 |
| scoring one candidate ≈ 4.6 min native vs ≈ 15 min metered | — | ESTIMATED | sum of the build and scoring rows above | — |
| `margin = 0.8`, 3-miss threshold, 0.1 step | — | ESTIMATED | no data yet; tuned from recorded misses | — |
| CPU challenges behave like hypergraph | — | ESTIMATED (unmeasured) | knapsack live test in §7 | — |
