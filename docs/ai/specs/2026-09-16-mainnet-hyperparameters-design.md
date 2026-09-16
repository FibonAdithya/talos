# Mainnet hyperparameters — design

Date: 2026-09-16. Branch: `mainnet-hyperparameters`, cut from
`run-feedback-fixes` at `a4811d1` because it builds on `--track` (per-track
prompt and package sections), which is not yet on `main`.

This spec makes Talos run the baseline algorithm, and every candidate, with the
hyperparameters of that algorithm's best-quality mainnet benchmark on each
track, instead of with no hyperparameters.

## 1. Why

`tig-runtime` takes an optional `--hyperparameters <json>` and hands it to the
algorithm's `solve_challenge` as `Option<Map<String, Value>>`. Talos never
passes it (`talos/inside.py::run_nonce`), so every algorithm runs on the
defaults in its code. Mainnet benchmarkers do pass it, per track, and the
values differ by track. A baseline measured on defaults can be weaker than the
same algorithm as it actually runs on mainnet, and a candidate that "beats" it
may not beat what benchmarkers already get.

### Facts this design rests on

| claim | value | MEASURED / ESTIMATED | source | when |
|---|---|---|---|---|
| hyperparameters are per benchmark, one track per benchmark | `PrecommitDetails.hyperparameters`, `settings.track_id` | MEASURED | `tig-structs/src/core.rs:460-469` (monorepo HEAD `66f0e150`); live `/get-benchmarks` | 2026-09-16 |
| pinned `tig-runtime` accepts the flag | `--hyperparameters` at `main.rs:37` | MEASURED | raw GitHub fetch at `84a5787` | 2026-09-16 |
| players listed by `/get-opow` | 13 | MEASURED | `/get-opow` at block 1345242 | 2026-09-16 |
| precommits in the ~120-block window | 2862 | MEASURED | `/get-benchmarks` for all 13 players | 2026-09-16 |
| share with hyperparameters set | c004 0/327, c003 106/332, c006 338/345 | MEASURED | same | 2026-09-16 |
| best-quality benchmark runs a different algorithm than top adoption | c003 3/5 tracks (c003_a148 vs c003_a144), c006 `n_hidden=4`, c008 `s=congested` | MEASURED | same, plus `/get-algorithms` | 2026-09-16 |
| bundles per benchmark | 2 in the sampled benchmark | MEASURED (one sample) | same | 2026-09-16 |
| unknown/missing keys are tolerated | `hgs_prometheus` merges the map into its defaults | MEASURED for one algorithm | `params.rs:393-427` on `vehicle_routing/hgs_prometheus` | 2026-09-16 |

Consequence of row 6: hyperparameters are keyed to one algorithm's code and
cannot be transplanted. Consequence of row 7: "best quality" is a noisy
ranking; this design accepts that noise (the user chose best-quality over
most-used).

## 2. Decisions

| question | chosen | rejected |
|---|---|---|
| which algorithm | keep top-adoption; take the best benchmark **of that algorithm** per track | follow the top benchmark's algorithm per track (one job holds one file set; much larger change) |
| which benchmark | highest mean of `average_quality_by_bundle` | most-used hyperparameter set (less noisy, but not what was asked) |
| candidates | same per-track hyperparameters as the baseline; values shown in the prompt | hidden from the LLM (it may rename a key without knowing it matters) |
| default | on; `--hyperparameters none` disables | opt-in |

## 3. Selection — `talos/mainnet.py`

`top_algorithm` returns the algorithm id alongside name and adoption:
`(name, algorithm_id, adoption)`. Callers are updated.

New:

```python
@dataclass(frozen=True)
class TrackHyperparameters:
    hyperparameters: dict | None   # None = run without the flag
    benchmark_id: str | None       # None = no matching benchmark
    player_id: str | None
    mean_quality: float | None

def top_hyperparameters(algorithm_id: str, tracks: list[str], fuel: int,
                        get_json=_get_json) -> dict[str, TrackHyperparameters]:
```

1. `block_id = _block_id(get_json)`; players = `player_id` of every row of
   `/get-opow?block_id=`.
2. For each player, `/get-benchmarks?block_id=&player_id=`. Build
   `precommits` by `benchmark_id`, `benchmarks` by `id`, and the set of fraud
   `benchmark_id`s.
3. A benchmark is eligible when all hold:
   - it has a precommit with `settings.algorithm_id == algorithm_id`;
   - `details.fuel_budget == fuel` (hyperparameters are tuned to a fuel budget);
   - `settings.track_id` is in `tracks`;
   - its id is not in the fraud set;
   - `details.average_quality_by_bundle` is a non-empty list.
4. Per track, pick the eligible benchmark with the highest
   `mean(average_quality_by_bundle)`. Ties break on `benchmark_id` ascending
   so the choice is deterministic.
5. Every track in `tracks` appears in the result. A track with no eligible
   benchmark gets all-`None`. A chosen benchmark whose hyperparameters are
   `null` gets `hyperparameters=None`; `{}` stays `{}` (it is passed as `{}`,
   which is what that benchmarker ran).

A player whose `/get-benchmarks` call fails raises `MainnetError`; the job
does not start on a partial view. `--hyperparameters none` is the escape
hatch.

## 4. Storage — `talos/state.py`

`JobSpec` gains, after `track`:

```python
baseline_algorithm: str | None = None                   # algorithm name the map belongs to
hyperparameters: dict[str, dict | None] | None = None   # track -> map; None = feature off
hyperparameters_source: dict[str, dict] | None = None   # track -> {benchmark_id, player_id, mean_quality}
```

- All three are frozen in the write-once `job.json`.
- `baseline_algorithm` is set whenever `hyperparameters` is. Today
  `resolve_baseline` (called from `talos/loop.py`, after `job.json` is written)
  asks mainnet for the top algorithm itself. If adoption changes between job
  start and the baseline run, or before a resume, the frozen map would belong
  to a different algorithm. When `spec.baseline_algorithm` is set,
  `resolve_baseline` uses it instead of calling `top_algorithm`, and errors if
  that algorithm is no longer on mainnet rather than silently switching.
- A `job.json` without the keys loads with all three `None`, which behaves
  exactly as today (top algorithm resolved at baseline time, no flag on any run).
- `redacted()` keeps both: they are public mainnet data and contain no
  `rand_hash`. The mainnet benchmark's own `rand_hash` is never stored.

## 5. Runtime

- `inside.run_nonce(..., hyperparameters: dict | None = None)`: when not
  `None`, append `--hyperparameters <json.dumps(hp, separators=(",", ":"))>`
  to the `tig-runtime` argv only. `tig-verifier` is unchanged. `{}` is passed
  (not skipped), because `Some({})` and `None` are different inputs to the
  algorithm.
- `EvalRequest` gains `hyperparameters: dict[str, dict | None] | None`. Each
  nonce looks up its own track: `(request.hyperparameters or {}).get(track)`.
- Modal (`modal_app/talos_bench.py`): `score_nonce` / `_score_impl` take a
  `hyperparameters` argument; `talos/bench.py` passes the track's map in the
  starmap args. This changes the deployed function signature: `talos setup`
  must be re-run. README says so.
- C3 (`talos/c3_jobdir.py::payload`, `talos/c3_job.py::_score`): the map goes
  into `payload.json`; `_run_one` and the serial branch pass the track's map.
  `c3_job` ships its modules per job, so no image rebuild. The payload is part
  of `request_hash`, so the C3 result cache distinguishes the two.
- The loop builds every request (baseline, candidate, confirmation) from
  `spec.hyperparameters`. There is no per-request override, which is what
  keeps invariant 1.

## 6. Baseline cache key — `talos/baseline.py`

`cache_key` gains `hyperparameters` and includes it in the hashed payload as
`json.dumps(..., sort_keys=True)` input. `None` and a map of all-`None`
values must hash **differently from each other only if they run differently**;
they run identically (no flag on any nonce), so both normalise to `None`
before hashing. `{}` for a track does run differently and hashes differently.

`resolve_baseline` passes the spec's map into the bench request. When
`_require_scoreable` fails and hyperparameters were in use, the message adds:
`the baseline ran with mainnet hyperparameters; retry with --hyperparameters none to rule them out`.

## 7. CLI — `talos/cli.py`

- `talos run --hyperparameters {mainnet,none}`, default `mainnet`.
- The wizard asks `Hyperparameters (mainnet or none)` with default `mainnet`.
- On `mainnet`, after `top_algorithm` and before `write_spec`, call
  `top_hyperparameters(algorithm_id, info.tracks, info.max_fuel)`.
  If every track comes back all-`None`, store `hyperparameters=None` (feature
  effectively off) and say so.
- The job summary line gains: `hyperparameters: 4/5 tracks from mainnet` or
  `hyperparameters: none`.

The CLI calls `top_algorithm` once, uses its id for `top_hyperparameters`, and
writes its name into `spec.baseline_algorithm` (section 4). The loop's
`resolve_baseline` then reads the name from the spec instead of calling
mainnet a second time; two calls can straddle a block and disagree.

## 8. Prompt — `talos/prompts.py`

`PromptContext` gains `hyperparameters: dict[str, dict | None] | None`. When
not `None`, the prompt gets a section:

```
## Hyperparameters

Every run of your code, on the baseline and on your candidate alike, passes these
values to solve_challenge as `hyperparameters`, per track. They came from the best
mainnet benchmark of this algorithm. Keep every key readable: you may add keys, but
do not rename or remove existing ones.

track n_nodes=600: {"allow_swap3":true,...}
track n_nodes=1000: none (solve_challenge receives None)
```

With `--track`, only the focus track's line is shown plus a one-line note that
guard tracks also run with their own values.

## 9. Package — `talos/package.py`

A `Hyperparameters` section after the score tables: per track, the JSON and
its source (`benchmark_id`, `player_id`, mean quality), or `none`. The user
submits benchmarks with these values; the package states that the measured
improvement holds only with them.

## 10. AGENTS.md

- Invariant 1: add "and identical hyperparameters (`JobSpec.hyperparameters`,
  applied per track in `inside.run_nonce`)".
- Invariant 3: add that the hyperparameter map is part of the baseline cache key.

## 11. Tests and the mutation each catches

| test | mutation it catches |
|---|---|
| `top_hyperparameters` ignores a higher-quality benchmark on another algorithm | drop the `algorithm_id` filter |
| ... ignores a higher-quality benchmark at a different fuel | drop the `fuel_budget` filter |
| ... ignores a higher-quality fraud benchmark | drop the fraud filter |
| ... picks the higher mean, not the higher first bundle (`[10, 100]` vs `[60, 60]`) | rank by `q[0]` or `max(q)` |
| ... tie on mean resolves to the lower `benchmark_id` in both input orders | nondeterministic tie-break |
| ... a track with no eligible benchmark is present with `hyperparameters is None` | omit the track / `KeyError` downstream |
| ... chosen `{}` stays `{}`; chosen `null` becomes `None` | `if hp:` falsy guard collapsing `{}` to `None` |
| ... a failing player call raises `MainnetError` | swallowing the error |
| `run_nonce` argv has `--hyperparameters` + exact compact JSON when given a map, also for `{}` | falsy guard on `{}`; flag sent to verifier |
| `run_nonce` argv has no `--hyperparameters` when `None` | always passing the flag |
| `run_nonce` sends the flag to `tig-runtime` and not `tig-verifier` | flag appended to both |
| C3 `_score` gives each track its own map (two tracks, different maps) | using one map for all tracks |
| Modal bench starmap args carry the track's map | argument dropped in `bench.py` |
| `cache_key` differs between `None` and `{"t": {}}` and between two different maps; equal for `None` and `{"t": None}` | key ignores the map; normalisation missing |
| old `job.json` without the keys loads with `None` | missing dataclass default |
| `resolve_baseline` with `spec.baseline_algorithm="a"` uses `a` when mainnet's top is now `b` | still calling `top_algorithm` |
| `resolve_baseline` with `baseline_algorithm=None` still calls `top_algorithm` | breaking old jobs |
| loop: baseline request and candidate request carry the same map as the spec | a code path that builds a request without it |
| prompt contains each track's JSON and `none` for a missing track; still no `rand_hash` | section omitted; leak |
| package has the section with `benchmark_id` | section omitted |
| cli `--hyperparameters none` never calls `top_hyperparameters`; default does | flag ignored |

Fixtures are captured `/get-opow` and `/get-benchmarks` JSON cut down to a few
rows, with ids kept and nothing else from the live window. cli tests stub the
subprocess and the C3 client (memory: 2026-09-15 stray job).

## 12. Out of scope

- Following another algorithm's benchmark (decision above).
- Searching or tuning hyperparameters ourselves.
- Refreshing the map during a job; it is frozen at job start.
- Checking that a candidate still reads the keys. A candidate that ignores
  them is scored as it runs, like any other change.
