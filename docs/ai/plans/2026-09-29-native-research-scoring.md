# Native Research Scoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Behind `talos run --scoring native`, research candidates build with plain `cargo build` and run without fuel metering, under a per-track time budget calibrated against the baseline's metered `fuel_consumed`. Every candidate that would become the new best is first re-scored on TIG's metered runtime, and only a validated candidate becomes `best`.

**Architecture:** The in-container runner (`talos/inside.py`, driven by `talos/c3_job.py` on C3 and local Docker and by `modal_app/talos_bench.py` on Modal) gains a second mode. `native` renders a small Cargo binary (`talos-native`) into the monorepo workspace and builds it. For GPU challenges it first compiles the PTX without fuel injection. It then runs one nonce per process, and a watchdog thread exits 87 when the fuel-equivalent time is used up, the same exit code a metered run gets when its fuel runs out. `tig-verifier` still computes quality. A new pure module `talos/calibration.py` turns the paired metered fuel and native solve times into per-track budgets and caches them under `~/.talos/calibration/`. The loop (`talos/loop.py`) calibrates once per job, scores research natively, and validates each would-be best on the metered path with a pure rule (`talos/scoring.py::validation_failure`). Held-out confirmation still runs only on metered results.

**Tech Stack:** Python 3.10+, pytest with injected subprocess runners and `FakeBench` (no unit test touches Docker, Modal, C3 or the network), Rust (edition 2021, the monorepo's pinned `Cargo.lock` crates only), ruff.

**Spec:** `docs/ai/specs/2026-09-29-native-research-scoring-design.md`

**Branch:** rebase `native-scoring-spec` onto `main` once PR #29 has merged, then implement on it. One PR carries the spec, this plan and the code.

## Deviations from the spec

Each item below was either decided with the user on 2026-09-29 or is forced by the code as it stands at `289b655`. The spec is not edited. This list is the record.

1. **Validation criterion (user decision).** Spec §5.3 requires `beats()` on training. The plan instead requires the metered result to still count as *improved* under the loop's existing rule (`mean_rel_delta > best` or `beats`). Otherwise a stepping stone could never become `best` and research could never build on one. `beats()` still decides whether held-out is scored, exactly as in today's metered loop.
2. **What happens at the fuel budget (user decision).** Spec §4.4 kills the process and reports `out_of_fuel`. The plan's runner times its own solve and exits 87 when the budget is used up. The last saved solution is then verified, as `tig-runtime` does with a metered out-of-fuel exit, and `inside.classify` needs no change. So the "binding bound" in the record is not needed. Exit 87 is the fuel bound, and the outer subprocess timeout is the `timeout` bound.
3. **Budget floor (user decision).** The spec floors the budget at `runtime_floor_s` (60 s). Hypergraph baseline nonces run 30–72 s natively (MEASURED in the probe), so that floor would bind on most tracks. The plan uses a separate `BUDGET_FLOOR_US = 1_000_000` (1 s, ESTIMATE).
4. **The runtime ceiling stays per job and metered-derived.** The spec's `ceiling_ms` is `runtime_ceiling × max(native_ms_i)` over the calibrating job's own nonces, and a cached record is reused by jobs with different nonces. So the plan keeps today's `Loop._timeouts(baseline.training)` (metered runtimes, per job) as the outer subprocess timeout for native nonces too. Metered runtimes run 1.6–2.9× the native ones (MEASURED in the probe), so this ceiling is looser than the spec's, never tighter.
5. **Calibration measures time at the last save.** The runner records `solve_us`, the microseconds from the start of `solve_challenge` to its last `save_solution` call. `fuel_consumed` in `tig-runtime`'s output file is also read at the last save. The ratio pairs two quantities taken at the same point. This matters because an algorithm that keeps running after its last save would otherwise have more native time than fuel in the pair. The ratio is `solve_us / fuel_consumed`, in microseconds per fuel unit.
6. **No separate `validated_best` field.** In native mode `state.best` is only ever set after validation passes, so `best` *is* the validated best. `_current_files()` needs no change, and `best` can never point at a demoted candidate. `JobState` gains `validations`, `scoring` and `fuel_budgets_us`.
7. **Templates live in a Python module, not a directory.** Modal's `add_local_python_source` ships only `.py` files, and the C3 job directory copies `JOB_MODULES` by name. So the Rust templates are string constants in `talos/native_runner.py`.
8. **The verifier uses the native PTX.** `build_ptx` leaves every kernel in `framework.cu` and the challenge's `.cu` files uninstrumented (`kernels_to_ignore`), and those are the only kernels `tig-verifier` launches. The metered PTX would verify identically, so it is not built in native mode. The live parity test (Task 11) checks this.
9. **Toolchain.** The native build uses `cargo +nightly-2025-02-10`, the toolchain `build_so` uses at `MONOREPO_REF`, so both paths compile the algorithm with the same rustc and LLVM. The probe's toolchain was not recorded.
10. **GPU stream.** The runner uses `ctx.default_stream()`, not `tig-runtime`'s `fuel_check_stream()`. The latter launches `finalize_kernel` and reads back the error flag after every kernel launch, a sync that only metering needs (`cudarc` `b3fccf5`, `src/driver/safe/launch.rs::perform_fuel_check`).
11. **Local backend: offline cargo.** The local job container has no network. The warm-up (`talos/local_transport.py::warm_script`) gains `cargo fetch`, so every crate in `Cargo.lock` is present, including `tig-structs` and `tig-utils`, which the metered warm build never compiles. Its marker is bumped so existing volumes re-warm. The local `job.sh` exports `CARGO_NET_OFFLINE=true`, which turns a network wait into an immediate error, and a network access there fails anyway.
12. **`limit_hit` on every row (audit, 2026-09-29).** A run that exits 87 after saving a verified solution is `ok` with `error` None, as TIG counts it, so neither path's `error` shows that the fuel or the budget ran out. Without that, a candidate whose native run stopped at its budget (by design: margin 0.8 < 1) and whose metered run finished inside its fuel would be demoted as `nondeterministic`, and a metered fuel exhaustion with a solution would never count as a `fuel_proxy_miss`. Rows carry `limit_hit`; a limit-cut nonce is never a quality mismatch; a miss is a metered fuel limit where the native run still had budget.
13. **`talos compile` stays metered by default (audit, 2026-09-29).** See Task 10.
14. **Resume keeps the job's budgets (audit, 2026-09-29).** `Loop.calibrate` never re-derives or re-measures once `state.fuel_budgets_us` is set. See Task 9.

## Global Constraints

- Python floor is 3.10 (`pyproject.toml`): no `match`, no `tomllib`, no `datetime.UTC`.
- Lines are at most 100 characters. Ruff's pinned rule set (`E4,E7,E9,F`) does not check this. Check with `python3 -c "import sys; [print(f, i+1) for f in sys.argv[1:] for i, l in enumerate(open(f, encoding='utf-8')) if len(l.rstrip('\n')) > 100]" <files>`.
- Tests run with `.venv/bin/python -m pytest <path> -q`. The full gate is `make check` from a 3.11+ venv: `uv venv --python 3.11 $S/venv311 && uv pip install --python $S/venv311/bin/python -r requirements-dev.txt -e . && make check PYTHON=$S/venv311/bin/python`, where `$S` is the session scratchpad. Never `.venv/bin/pip`.
- CI runs Linux, macOS and Windows. No unit test may run `docker`, `cargo`, `nvcc`, `c3` or `modal`, or reach the network. Every subprocess goes through an injected `run`.
- `talos/c3_job.py` and the other `JOB_MODULES` are copied into the C3 job directory. They may import only each other and the standard library. `talos/native_runner.py` joins them.
- The job's `rand_hash` never reaches a log line, an exception message, a timeline event or a calibration record. The calibration key hashes the baseline *files*, never the nonce sets.
- A metered request must stay byte-for-byte what it is today: same C3 payload keys (so `request_hash` is unchanged and a resumed job reattaches), same Modal call arguments, same `content_hash`. Native-only keys are added only when `mode == "native"`.
- Budget rule (AGENTS.md invariant 5): every compute call, including calibration, goes through `_BudgetedBench` or `Loop._bench_evaluate`, and the budget is checked before the call.
- `margin = 0.8`, the 0.1 step, the 0.3 margin floor, the 3-miss limit and the 1 s budget floor are ESTIMATES and are labelled so wherever they appear.
- Commit after every task with explicit paths (`git add <paths>`, never `-A`/`.`). Run `git status --short` first.
- Every number that reaches a doc, commit message or PR body is labelled MEASURED (run in this session, output shown) or ESTIMATE.
- New tests carry a `# mutation:` comment naming the defect they catch, as the suite already does.

## Review Focus

These are inputs the spec implies but names no test for. Each one's test is in the task that owns the code.

1. **A baseline cached before this change has no `fuel_consumed`.** Every existing `~/.talos/baselines` record is like this. Expected: calibration runs one metered training pass to get fuel, rather than falling back to metered research. Test in Task 9 (`test_calibrate_rescores_metered_fuel_when_the_cached_baseline_has_none`).
2. **A job killed during validation is resumed.** Expected: the resume re-enters validation with the native results it already had, without scoring natively again and without a new hypothesis. Test in Task 9 (`test_resume_mid_validation_goes_straight_to_the_metered_run`).
3. **`talos run --resume <id> --scoring native` on a metered job.** Expected: refused with a message, like `--mode`. Test in Task 10 (`test_resume_refuses_a_different_scoring_mode`).
4. **The deployed Modal app predates this client.** Expected: a metered job's Modal calls are unchanged (so an old deploy keeps working), and a native job's extra `mode=` argument is refused by an old deploy as a stale deploy ("run `talos setup`"), not retried for 15 minutes. Test in Task 6 (`test_metered_modal_calls_are_unchanged_and_native_ones_add_mode`).
5. **A corrupt or truncated calibration file.** Expected: treated as missing and re-measured, never a traceback. Test in Task 7 (`test_load_treats_a_corrupt_record_as_missing`).

---

### Task 1: Value types and request fields

**Files:**
- Modify: `talos/types.py` (`NonceResult`)
- Modify: `talos/bench.py` (`EvalRequest`)
- Modify: `talos/state.py` (`JobSpec`, `JobState`)
- Test: `tests/test_types.py`, `tests/test_state.py`

**Interfaces:**
- Produces: `NonceResult.fuel_consumed: int | None = None`, `NonceResult.solve_us: int | None = None`, `NonceResult.limit_hit: bool = False`; `EvalRequest.mode: str = "metered"`, `EvalRequest.fuel_budgets_us: dict[str, int] | None = None`; `SCORING_MODES = ("metered", "native")` in `talos/types.py`; `JobSpec.scoring: str = "metered"`; `JobState.scoring: str | None = None`, `JobState.fuel_budgets_us: dict[str, int] | None = None`, `JobState.validations: list[dict]` (default `[]`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_types.py`:

```python
def test_nonce_result_carries_fuel_and_solve_time_and_reads_old_rows():
    from talos.types import NonceResult
    r = NonceResult("t", 1, True, 5, 10, None, fuel_consumed=1234, solve_us=987,
                    limit_hit=True)
    d = r.to_dict()
    # mutation: a field missing from to_dict never reaches results.json or the baseline cache
    assert d["fuel_consumed"] == 1234 and d["solve_us"] == 987 and d["limit_hit"] is True
    assert NonceResult.from_dict(d) == r
    # rows written before this change (every cached baseline) have none of the keys
    old = NonceResult.from_dict({"track": "t", "nonce": 1, "ok": True, "quality": 5,
                                 "runtime_ms": 10, "error": None})
    assert old.fuel_consumed is None and old.solve_us is None and old.limit_hit is False
```

Append to `tests/test_state.py`:

```python
def test_scoring_fields_default_to_metered_and_round_trip():
    from talos.budget import Spend
    from talos.state import JobState
    st = JobState.fresh(Spend(started_at=0.0))
    # mutation: a default of "native" would switch every existing job's scoring on resume
    assert st.scoring is None and st.fuel_budgets_us is None and st.validations == []
    st.scoring, st.fuel_budgets_us = "native", {"t": 1_200_000}
    st.validations.append({"iteration": 1, "outcome": "validated", "reason": None})
    back = JobState.from_dict(st.to_dict())
    assert back.scoring == "native" and back.fuel_budgets_us == {"t": 1_200_000}
    assert back.validations == [{"iteration": 1, "outcome": "validated", "reason": None}]
    # a state.json written before this change loads with the defaults
    d = st.to_dict()
    for k in ("scoring", "fuel_budgets_us", "validations"):
        d.pop(k)
    old = JobState.from_dict(d)
    assert old.scoring is None and old.fuel_budgets_us is None and old.validations == []


def test_job_spec_scoring_defaults_to_metered_for_an_old_job_json():
    from talos.state import JobSpec
    from talos.budget import Budget
    from talos.types import NonceSet
    sp = JobSpec(job_id="j", challenge="knapsack", direction="d", provider="fake", model="m",
                 mode="single-shot", budget=Budget(usd=None, hours=None, iterations=1,
                                                   compute_usd=None),
                 rand_hash="ab" * 32, tracks=["t"], training=[NonceSet("t", "ab" * 32, 0, 1)],
                 holdout=[NonceSet("t", "ab" * 32, 1_000_000, 1)], fuel=1, created_at=0.0,
                 monorepo_ref="r", challenge_id="c003")
    d = sp.to_dict()
    assert d["scoring"] == "metered"
    d.pop("scoring")
    # mutation: a required field would make every job.json written before this unreadable
    assert JobSpec.from_dict(d).scoring == "metered"
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_types.py tests/test_state.py -q`
Expected: FAIL with `TypeError: ... unexpected keyword argument 'fuel_consumed'` and `AttributeError: 'JobState' object has no attribute 'scoring'`.

- [ ] **Step 3: Implement**

In `talos/types.py`, add after `ERROR_KINDS`:

```python
SCORING_MODES = ("metered", "native")
```

and extend `NonceResult` (the new fields go last, with defaults, so positional construction everywhere keeps working):

```python
@dataclass
class NonceResult:
    track: str
    nonce: int
    ok: bool
    quality: int | None
    runtime_ms: int
    error: str | None = None
    # Metered runs: the fuel tig-runtime recorded at the last save_solution call. Native runs:
    # microseconds from the start of solve_challenge to its last save. Calibration pairs the
    # two (talos/calibration.py::track_ratios). None when the run did not record it.
    fuel_consumed: int | None = None
    solve_us: int | None = None
    # The solver exited 87: tig-runtime's fuel ran out (metered) or the native budget did.
    # A run that saved a verified solution first is still ok, as TIG counts it, so `error`
    # cannot carry this; validation needs it to tell a truncated run from nondeterminism
    # (talos/scoring.py::validation_failure).
    limit_hit: bool = False
```

In `talos/bench.py`, add to `EvalRequest` after `hyperparameters`:

```python
    # "metered" builds with TIG's build_algorithm and runs tig-runtime under `fuel`; "native"
    # builds the talos-native runner and runs it under the per-track budget in
    # fuel_budgets_us (microseconds of solve time; a track absent from it runs without one).
    mode: str = "metered"
    fuel_budgets_us: dict[str, int] | None = None

    def __post_init__(self) -> None:
        if self.mode not in SCORING_MODES:
            raise ValueError(f"unknown scoring mode {self.mode!r}")
```

and import `SCORING_MODES` from `talos.types`.

In `talos/state.py`, add to `JobSpec` after `hyperparameters_source`:

```python
    scoring: str = "metered"  # research scoring, `talos run --scoring`; frozen with the job
```

Add to `JobState` after `hardware`:

```python
    # What research scoring actually runs: None until Loop.calibrate has run, "native" once a
    # calibration record gave every research track a budget, "metered" when the job asked for
    # native and calibration could not give one (the fallback is kept for the job's life).
    scoring: str | None = None
    fuel_budgets_us: dict[str, int] | None = None  # track -> native solve budget, microseconds
    # One entry per metered validation of a would-be best: iteration, outcome, reason.
    validations: list[dict] = field(default_factory=list)
```

Add `"scoring": self.scoring, "fuel_budgets_us": self.fuel_budgets_us, "validations": self.validations` to `JobState.to_dict`. Pass `scoring=d.get("scoring"), fuel_budgets_us=d.get("fuel_budgets_us"), validations=d.get("validations", [])` in `JobState.from_dict`.

- [ ] **Step 4: Run to verify they pass, then the whole suite**

Run: `.venv/bin/python -m pytest tests/test_types.py tests/test_state.py -q && .venv/bin/python -m pytest -q -m "not live"`
Expected: PASS. The whole suite stays green because every new field has a default.

- [ ] **Step 5: Commit**

```bash
git status --short
git add talos/types.py talos/bench.py talos/state.py tests/test_types.py tests/test_state.py
git commit -m "types: fuel_consumed and solve_us on results, scoring mode on requests and jobs"
```

---

### Task 2: The native runner templates

**Files:**
- Create: `talos/native_runner.py`
- Test: `tests/test_native_runner.py`

**Interfaces:**
- Consumes: nothing from earlier tasks. This module is shipped into the container, so it imports only the standard library.
- Produces: `PACKAGE = "talos-native"`, `TOOLCHAIN = "+nightly-2025-02-10"`, `render(challenge: str, algorithm: str, is_gpu: bool) -> dict[str, str]` (relative path under the monorepo → text: `talos-native/Cargo.toml`, `talos-native/src/main.rs`), `add_workspace_member(cargo_toml: str) -> str`, `stage(monorepo: Path, challenge: str, algorithm: str, is_gpu: bool) -> None`.

The Rust below follows `tig-runtime/src/main.rs` and `tig-binary/src/entry_point_template.rs` at `MONOREPO_REF` (`84a5787`), which were read while writing this plan. Seed: `BenchmarkSettings::calc_seed`. Track: parsed from the quoted `track_id`. Hyperparameters: a JSON object string, parsed into `Option<Map<String, Value>>` as `entry_point` does. Panics are caught and become exit 84, as `entry_point`'s `catch_unwind` turns them into `Err`. When no solution was saved, an empty `Solution::new()` is written, as `tig-runtime` does.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_native_runner.py`:

```python
import ast
from pathlib import Path

import pytest

from talos import native_runner

WORKSPACE = """[workspace]
members = [
    "tig-algorithms",
    "tig-binary",
]
exclude = []
resolver = "2"
"""


def test_cpu_render_substitutes_the_challenge_and_calls_the_three_argument_solve():
    files = native_runner.render("knapsack", "talos_cand", is_gpu=False)
    main, cargo = files["talos-native/src/main.rs"], files["talos-native/Cargo.toml"]
    # mutation: a placeholder left unfilled fails the build with an unresolved import
    for text in (main, cargo):
        assert "{CHALLENGE}" not in text and "{ALGORITHM}" not in text
    assert "use tig_algorithms::knapsack::talos_cand as algorithm;" in main
    assert "use tig_challenges::knapsack::*;" in main
    # mutation: the GPU variant chosen for a CPU challenge passes module/stream/prop
    assert "algorithm::solve_challenge(&challenge, &save, &hyperparameters)" in main
    assert "cudarc" not in main and "cudarc" not in cargo
    assert 'features = ["knapsack"]' in cargo


def test_gpu_render_passes_module_stream_and_prop_and_launches_initialize_kernel():
    files = native_runner.render("hypergraph", "talos_cand", is_gpu=True)
    main, cargo = files["talos-native/src/main.rs"], files["talos-native/Cargo.toml"]
    assert "use tig_algorithms::hypergraph::talos_cand as algorithm;" in main
    # mutation: the CPU variant chosen for a GPU challenge does not type-check against
    # hypergraph's six-argument solve_challenge
    assert ("algorithm::solve_challenge(&challenge, &save, &hyperparameters,\n"
            "                                   module.clone(), stream.clone(), &prop)") in main
    assert 'load_function("initialize_kernel")' in main and "seed[8..16]" in main
    # the fuel-check stream synchronises after every launch; the native path must not use it
    assert "default_stream()" in main and "fuel_check_stream" not in main
    # the cudarc source must match tig-challenges' character for character, or cargo resolves
    # two cudarc packages and the CudaModule types stop matching
    assert ('cudarc = { git = "https://github.com/tig-foundation/cudarc.git", '
            'branch = "runtime-fuel/cudnn-cublas", features = '
            '["cuda-version-from-build-system"] }') in cargo


@pytest.mark.parametrize("is_gpu", [False, True])
def test_both_variants_exit_87_at_the_budget_and_84_on_error(is_gpu):
    main = native_runner.render("knapsack", "talos_cand", is_gpu)["talos-native/src/main.rs"]
    # talos/inside.py::classify reads these two codes exactly as it reads tig-runtime's
    assert "const OUT_OF_FUEL: i32 = 87;" in main and "const RUNTIME_ERROR: i32 = 84;" in main
    assert "std::process::exit(OUT_OF_FUEL)" in main
    # mutation: writing the file in place lets the watchdog's exit leave half a JSON document
    assert "std::fs::rename(&tmp, path)" in main
    assert '"solve_us"' in main and "catch_unwind" in main


def test_add_workspace_member_inserts_once_inside_the_members_list():
    once = native_runner.add_workspace_member(WORKSPACE)
    assert once.count('"talos-native"') == 1
    members = once.split("members = [", 1)[1].split("]", 1)[0]
    # mutation: appending after the list instead of inside it leaves the package outside
    # the workspace, and `cargo build -p talos-native` reports no such package
    assert '"talos-native"' in members
    # mutation: a second insert on the local backend's persistent /app duplicates the member,
    # which cargo rejects
    assert native_runner.add_workspace_member(once) == once


def test_add_workspace_member_rejects_a_manifest_without_members():
    with pytest.raises(ValueError):
        native_runner.add_workspace_member("[package]\nname = \"x\"\n")


def test_stage_writes_the_package_and_registers_it(tmp_path):
    (tmp_path / "Cargo.toml").write_text(WORKSPACE, encoding="utf-8")
    native_runner.stage(tmp_path, "knapsack", "talos_cand", is_gpu=False)
    native_runner.stage(tmp_path, "knapsack", "talos_cand", is_gpu=False)
    assert (tmp_path / "talos-native" / "src" / "main.rs").exists()
    assert (tmp_path / "Cargo.toml").read_text(encoding="utf-8").count('"talos-native"') == 1


def test_native_runner_imports_only_the_standard_library():
    # it is shipped into the C3 job directory, where talos is not installed
    tree = ast.parse(Path(native_runner.__file__).read_text(encoding="utf-8"))
    mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    mods |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not any(m and m.split(".")[0] == "talos" for m in mods)
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_native_runner.py -q`
Expected: FAIL with `ImportError: cannot import name 'native_runner'`.

- [ ] **Step 3: Implement `talos/native_runner.py`**

```python
"""The talos-native runner: a small Cargo binary that runs one nonce of the staged algorithm
without TIG's fuel instrumentation. Rendered into the monorepo workspace the way
tig-binary/scripts/build_so fills in entry_point_template.rs, then built with plain
`cargo build`. Shipped into the C3 job directory, so standard library only.

Everything here follows tig-runtime/src/main.rs and tig-binary/src/entry_point_template.rs at
MONOREPO_REF: the seed, the track parse, the hyperparameter map, the exit codes, and the empty
solution written when the algorithm saved none. Two deliberate differences: no fuel counter
(a watchdog thread exits 87 at the per-track budget instead), and on GPU the default stream
rather than tig-runtime's fuel-check stream, which synchronises after every kernel launch."""
from __future__ import annotations

from pathlib import Path

PACKAGE = "talos-native"
# The toolchain build_so uses at MONOREPO_REF, so native and metered builds share rustc/LLVM.
TOOLCHAIN = "+nightly-2025-02-10"

_CUDARC = ('cudarc = { git = "https://github.com/tig-foundation/cudarc.git", '
           'branch = "runtime-fuel/cudnn-cublas", features = '
           '["cuda-version-from-build-system"] }')

CARGO_TOML = """[package]
name = "talos-native"
version = "0.1.0"
edition = "2021"
publish = false

[[bin]]
name = "talos-native"
path = "src/main.rs"

[dependencies]
anyhow = "1.0.81"
serde_json = { version = "1.0.113" }
tig-algorithms = { path = "../tig-algorithms", features = ["{CHALLENGE}"] }
tig-challenges = { path = "../tig-challenges", features = ["{CHALLENGE}"] }
tig-structs = { path = "../tig-structs" }
tig-utils = { path = "../tig-utils" }
{CUDARC}
"""

_COMMON = r"""use anyhow::{anyhow, Result};
use serde_json::{Map, Value};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tig_algorithms::{CHALLENGE}::{ALGORITHM} as algorithm;
use tig_challenges::{CHALLENGE}::*;
use tig_structs::core::BenchmarkSettings;
use tig_utils::dejsonify;

// tig-runtime's exit codes at MONOREPO_REF: 84 the algorithm returned Err (a panic inside
// solve_challenge becomes Err in entry_point), 87 the fuel ran out.
const RUNTIME_ERROR: i32 = 84;
const OUT_OF_FUEL: i32 = 87;

struct Args {
    settings: String,
    rand_hash: String,
    nonce: u64,
    output: PathBuf,
    budget_us: Option<u64>,
    hyperparameters: Option<String>,
    ptx: Option<PathBuf>,
}

fn parse_args() -> Result<Args> {
    let mut it = std::env::args().skip(1);
    let mut next = |what: &str| it.next().ok_or_else(|| anyhow!("missing {}", what));
    let settings = next("SETTINGS")?;
    let rand_hash = next("RAND_HASH")?;
    let nonce = next("NONCE")?.parse()?;
    let output = PathBuf::from(next("OUTPUT")?);
    let (mut budget_us, mut hyperparameters, mut ptx) = (None, None, None);
    while let Ok(flag) = next("flag") {
        let value = next(flag.as_str())?;
        match flag.as_str() {
            "--budget-us" => budget_us = Some(value.parse()?),
            "--hyperparameters" => hyperparameters = Some(value),
            "--ptx" => ptx = Some(PathBuf::from(value)),
            _ => return Err(anyhow!("unknown flag {}", flag)),
        }
    }
    Ok(Args { settings, rand_hash, nonce, output, budget_us, hyperparameters, ptx })
}

fn seed_and_track(args: &Args) -> Result<([u8; 32], Track)> {
    let settings: BenchmarkSettings = dejsonify(&args.settings)?;
    let seed = settings.calc_seed(&args.rand_hash, args.nonce);
    let track_id = if settings.track_id.starts_with('"') && settings.track_id.ends_with('"') {
        settings.track_id.clone()
    } else {
        format!(r#""{}""#, settings.track_id)
    };
    let track = serde_json::from_str(&track_id)
        .map_err(|_| anyhow!("Failed to parse track_id '{}'", settings.track_id))?;
    Ok((seed, track))
}

fn hyperparameters(args: &Args) -> Result<Option<Map<String, Value>>> {
    match &args.hyperparameters {
        Some(s) => Ok(Some(serde_json::from_str(s)?)),
        None => Ok(None),
    }
}

// The shape tig-verifier reads (a "solution" string field), plus the solve time at this save.
fn write_output(path: &Path, nonce: u64, solution: &Solution, solve_us: u64) -> Result<()> {
    let doc = serde_json::json!({
        "nonce": nonce,
        "solution": serde_json::to_string(solution)?,
        "solve_us": solve_us,
    });
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, doc.to_string())?;
    // The watchdog may exit mid-save: a rename never leaves half a document behind.
    std::fs::rename(&tmp, path)?;
    Ok(())
}

// The fuel stand-in: exits 87 once the solve has run for the budget, leaving the last saved
// solution for tig-verifier, as a metered run that runs out of fuel does.
fn watchdog(budget_us: Option<u64>) {
    if let Some(us) = budget_us {
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_micros(us));
            std::process::exit(OUT_OF_FUEL);
        });
    }
}

fn main() {
    let result = parse_args().and_then(run);
    if let Err(e) = result {
        eprintln!("Runtime Error: {}", e);
        std::process::exit(RUNTIME_ERROR);
    }
}
"""

_CPU_RUN = r"""
fn run(args: Args) -> Result<()> {
    let (seed, track) = seed_and_track(&args)?;
    let hyperparameters = hyperparameters(&args)?;
    let challenge = Challenge::generate_instance(&seed, &track)?;
    let (output, nonce) = (args.output.clone(), args.nonce);
    let start = Instant::now();
    let save = |solution: &Solution| -> Result<()> {
        write_output(&output, nonce, solution, start.elapsed().as_micros() as u64)
    };
    watchdog(args.budget_us);
    let result = catch_unwind(AssertUnwindSafe(|| {
        algorithm::solve_challenge(&challenge, &save, &hyperparameters)
    }))
    .unwrap_or_else(|_| Err(anyhow!("Panic occurred calling solve_challenge")));
    if !output.exists() {
        save(&Solution::new())?;
    }
    result
}
"""

_GPU_RUN = r"""
use cudarc::{
    driver::{CudaContext, LaunchConfig, PushKernelArg},
    nvrtc::Ptx,
    runtime::result::device::get_device_prop,
};

fn run(args: Args) -> Result<()> {
    let (seed, track) = seed_and_track(&args)?;
    let hyperparameters = hyperparameters(&args)?;
    let ptx_path = args.ptx.clone().ok_or_else(|| anyhow!("--ptx is required on GPU"))?;
    let ptx = Ptx::from_src(std::fs::read_to_string(&ptx_path)?);
    let ctx = CudaContext::new(0)?;
    ctx.set_blocking_synchronize()?;
    let module = ctx.load_module(ptx)?;
    let stream = ctx.default_stream();
    let prop = get_device_prop(0)?;
    let challenge =
        Challenge::generate_instance(&seed, &track, module.clone(), stream.clone(), &prop)?;
    let initialize_kernel = module.load_function("initialize_kernel")?;
    let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1, 1, 1), shared_mem_bytes: 0 };
    unsafe {
        stream
            .launch_builder(&initialize_kernel)
            .arg(&(u64::from_be_bytes(seed[8..16].try_into().unwrap())))
            .launch(cfg)?;
    }
    let (output, nonce) = (args.output.clone(), args.nonce);
    let start = Instant::now();
    let save = |solution: &Solution| -> Result<()> {
        stream.synchronize()?; // as tig-runtime's save: the time covers finished kernels
        write_output(&output, nonce, solution, start.elapsed().as_micros() as u64)
    };
    watchdog(args.budget_us);
    let result = catch_unwind(AssertUnwindSafe(|| {
        algorithm::solve_challenge(&challenge, &save, &hyperparameters,
                                   module.clone(), stream.clone(), &prop)
    }))
    .unwrap_or_else(|_| Err(anyhow!("Panic occurred calling solve_challenge")));
    if !output.exists() {
        save(&Solution::new())?;
    }
    result
}
"""


def _fill(text: str, challenge: str, algorithm: str) -> str:
    return text.replace("{CHALLENGE}", challenge).replace("{ALGORITHM}", algorithm)


def render(challenge: str, algorithm: str, is_gpu: bool) -> dict[str, str]:
    """The runner's files, keyed by their path under the monorepo root."""
    main = _fill(_COMMON + (_GPU_RUN if is_gpu else _CPU_RUN), challenge, algorithm)
    cargo = _fill(CARGO_TOML, challenge, algorithm).replace("{CUDARC}",
                                                            _CUDARC if is_gpu else "")
    return {f"{PACKAGE}/Cargo.toml": cargo, f"{PACKAGE}/src/main.rs": main}


def add_workspace_member(cargo_toml: str) -> str:
    """The workspace manifest with the runner listed once in `members`. Idempotent, because the
    local backend's /app volume keeps the manifest from one job to the next."""
    if f'"{PACKAGE}"' in cargo_toml:
        return cargo_toml
    marker = "members = ["
    i = cargo_toml.find(marker)
    if i < 0:
        raise ValueError("workspace Cargo.toml has no `members = [` list")
    i += len(marker)
    return cargo_toml[:i] + f'\n    "{PACKAGE}",' + cargo_toml[i:]


def stage(monorepo: Path, challenge: str, algorithm: str, is_gpu: bool) -> None:
    for rel, text in render(challenge, algorithm, is_gpu).items():
        p = Path(monorepo) / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8", newline="\n")
    manifest = Path(monorepo) / "Cargo.toml"
    manifest.write_text(add_workspace_member(manifest.read_text(encoding="utf-8")),
                        encoding="utf-8", newline="\n")
```

- [ ] **Step 4: Run to verify they pass, and check line lengths**

Run: `.venv/bin/python -m pytest tests/test_native_runner.py -q` and the line-length one-liner from Global Constraints on `talos/native_runner.py tests/test_native_runner.py`.
Expected: PASS, and no line reported.

- [ ] **Step 5: Commit**

```bash
git status --short
git add talos/native_runner.py tests/test_native_runner.py
git commit -m "native_runner: render the talos-native Cargo binary for CPU and GPU challenges"
```

---

### Task 3: In-container native build and nonce run

**Files:**
- Modify: `talos/inside.py`
- Test: `tests/test_inside.py`

**Interfaces:**
- Consumes: `native_runner.stage`, `native_runner.PACKAGE`, `native_runner.TOOLCHAIN` (Task 2); `CHALLENGES` from `talos/challenges.py`, already a job module.
- Produces:
  - `content_hash(files, monorepo_ref, dev_image_tag, mode="metered") -> str`. For `"metered"` the bytes hashed are unchanged.
  - `build_native_ptx(monorepo, challenge, name, run=subprocess.run) -> tuple[bool, str]`
  - `build_native(monorepo, challenge, name, run=subprocess.run) -> tuple[bool, str]`
  - `native_artifact_paths(monorepo, challenge, name) -> tuple[Path, Path | None]`
  - `run_nonce(...)` rows gain `"fuel_consumed"` and `"limit_hit"` (tig-runtime exited 87).
  - `run_nonce_native(challenge_id, track, rand_hash, nonce, binary, budget_us, timeout_s, ptx, run=subprocess.run, workdir=None, clock=time.monotonic, hyperparameters=None) -> dict` (rows carry `"solve_us"` and `"limit_hit"`, the runner exited 87)
  - `NonceTask` becomes a 12-tuple `(challenge_id, track, rand_hash, nonce, so, fuel, timeout_s, ptx, workdir, hyperparameters, mode, budget_us)`, and `run_task` dispatches on `mode`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_inside.py`:

```python
import hashlib


def test_metered_content_hash_is_unchanged_and_native_differs():
    files = {"mod.rs": "fn x(){}"}
    # recomputed independently from the pre-change definition, not by calling the function
    h = hashlib.sha256()
    for part in ("r", "t", inside.CRATE_LAYOUT, "mod.rs", "fn x(){}"):
        h.update(part.encode())
        h.update(b"\0")
    # mutation: hashing the mode for metered too changes every cached artifact id and every
    # C3 request_hash, so a resumed job orphans its still-billing job
    assert inside.content_hash(files, "r", "t") == h.hexdigest()[:32]
    assert inside.content_hash(files, "r", "t", "metered") == h.hexdigest()[:32]
    # mutation: a native artifact served for a metered request
    assert inside.content_hash(files, "r", "t", "native") != h.hexdigest()[:32]


def test_run_nonce_records_the_fuel_tig_runtime_wrote(tmp_path):
    def run(cmd, **kw):
        if cmd[0] == "tig-runtime":
            (Path(cmd[cmd.index("--output") + 1]) / f"{cmd[3]}.json").write_text(
                '{"nonce": 7, "runtime_signature": 1, "fuel_consumed": 12345, '
                '"solution": "e30=", "cpu_arch": "AMD64"}')
            return Result(0)
        return Result(0, "quality: 5\n")
    row = inside.run_nonce("c003", "n=1", "ab" * 32, 7, Path("/x.so"), 10, 600, None, run,
                           tmp_path)
    # mutation: leaving fuel_consumed None starves calibration of data
    assert row["fuel_consumed"] == 12345 and row["ok"]
    assert row["limit_hit"] is False


def test_run_nonce_fuel_is_none_when_the_runtime_wrote_nothing(tmp_path):
    def run(cmd, **kw):
        return Result(87 if cmd[0] == "tig-runtime" else 1)
    row = inside.run_nonce("c003", "n=1", "ab" * 32, 7, Path("/x.so"), 10, 600, None, run,
                           tmp_path)
    assert row["fuel_consumed"] is None and row["error"] == "out_of_fuel"
    assert row["limit_hit"] is True


def test_run_nonce_marks_a_verified_out_of_fuel_run_as_limit_hit(tmp_path):
    def run(cmd, **kw):
        if cmd[0] == "tig-runtime":
            (Path(cmd[cmd.index("--output") + 1]) / f"{cmd[3]}.json").write_text(
                '{"nonce": 7, "fuel_consumed": 9, "solution": "e30="}')
            return Result(87)
        return Result(0, "quality: 5\n")
    row = inside.run_nonce("c003", "n=1", "ab" * 32, 7, Path("/x.so"), 10, 600, None, run,
                           tmp_path)
    # ok, as tig-runtime counts it, so `error` is None and cannot say the fuel ran out
    # mutation: limit_hit taken from `error == "out_of_fuel"` is False here, and validation
    # then calls a fuel-truncated metered run "nondeterministic"
    assert row["ok"] and row["error"] is None and row["limit_hit"] is True


def _native_run(seen, rc=0, write=True, quality=4242):
    def run(cmd, **kw):
        seen.append(cmd)
        if cmd[0].endswith("talos-native"):
            if write:
                Path(cmd[4]).write_text('{"nonce": 7, "solution": "e30=", "solve_us": 31000}')
            return Result(rc)
        return Result(0, f"quality: {quality}\n") if write else Result(1)
    return run


def test_run_nonce_native_builds_the_runner_and_verifier_commands(tmp_path):
    seen = []
    row = inside.run_nonce_native("c003", "n=1", "ab" * 32, 7, Path("/b/talos-native"),
                                  1_500_000, 600, None, _native_run(seen), tmp_path,
                                  hyperparameters={"a": 1})
    nat, ver = seen
    settings = json.loads(nat[1])
    assert settings["challenge_id"] == "c003" and settings["track_id"] == "n=1"
    assert nat[0] == str(Path("/b/talos-native")) and nat[2:4] == ["ab" * 32, "7"]
    # the runner takes an output FILE (not tig-runtime's folder) and the verifier reads it
    assert Path(nat[4]).name == "7.json" and ver[4] == nat[4]
    # mutation: the fuel limit passed to the runner instead of the time budget
    assert nat[nat.index("--budget-us") + 1] == "1500000" and "--fuel" not in nat
    assert nat[nat.index("--hyperparameters") + 1] == '{"a":1}'
    assert ver[:4] == ["tig-verifier", nat[1], "ab" * 32, "7"] and len(ver) == 5
    assert row["ok"] and row["quality"] == 4242 and row["solve_us"] == 31000
    assert row["fuel_consumed"] is None and row["limit_hit"] is False


def test_run_nonce_native_without_a_budget_passes_no_flag(tmp_path):
    seen = []
    inside.run_nonce_native("c003", "n=1", "ab" * 32, 7, Path("/b/talos-native"), None, 600,
                            None, _native_run(seen), tmp_path)
    # mutation: `--budget-us None` makes the runner's parse fail every calibration nonce
    assert "--budget-us" not in seen[0] and "--hyperparameters" not in seen[0]


def test_run_nonce_native_gpu_passes_the_ptx_to_both_and_the_gpu_to_the_verifier(tmp_path):
    seen = []
    inside.run_nonce_native("c005", "k=1", "ab" * 32, 3, Path("/b/talos-native"), None, 600,
                            Path("/a.native.ptx"), _native_run(seen), tmp_path)
    nat, ver = seen
    assert nat[nat.index("--ptx") + 1] == str(Path("/a.native.ptx")) and "--gpu" not in nat
    assert ver[ver.index("--ptx") + 1] == str(Path("/a.native.ptx"))
    assert ver[ver.index("--gpu") + 1] == "0"


def test_a_budget_exit_with_a_saved_solution_counts_like_metered_out_of_fuel(tmp_path):
    row = inside.run_nonce_native("c003", "n=1", "ab" * 32, 7, Path("/b/talos-native"),
                                  1000, 600, None, _native_run([], rc=87), tmp_path)
    # mutation: reporting every budget exit as an error is stricter than tig-runtime
    assert row["ok"] and row["error"] is None
    # mutation: limit_hit dropped makes validation compare this truncated quality with the
    # metered one and demote a valid candidate as "nondeterministic"
    assert row["limit_hit"] is True


def test_a_budget_exit_with_nothing_saved_is_out_of_fuel(tmp_path):
    row = inside.run_nonce_native("c003", "n=1", "ab" * 32, 7, Path("/b/talos-native"),
                                  1000, 600, None, _native_run([], rc=87, write=False),
                                  tmp_path)
    # mutation: a budget exit classified as panic or success
    assert not row["ok"] and row["error"] == "out_of_fuel"


def test_the_outer_timeout_on_a_native_nonce_is_timeout(tmp_path):
    def run(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, kw["timeout"])
    row = inside.run_nonce_native("c003", "n=1", "ab" * 32, 7, Path("/b/talos-native"),
                                  1000, 5, None, run, tmp_path)
    assert not row["ok"] and row["error"] == "timeout"


def test_run_task_dispatches_on_the_mode(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(inside, "run_nonce", lambda *a, **k: calls.append(("metered", a)) or {})
    monkeypatch.setattr(inside, "run_nonce_native",
                        lambda *a, **k: calls.append(("native", a)) or {})
    base = ("c003", "t", "ab" * 32, 1, "/x", 10, 60, None, str(tmp_path), None)
    inside.run_task(base + ("metered", None))
    inside.run_task(base + ("native", 777))
    # mutation: ignoring the mode runs every native task through tig-runtime
    assert [c[0] for c in calls] == ["metered", "native"]
    assert calls[0][1][5] == 10          # metered gets the fuel
    assert calls[1][1][5] == 777         # native gets the budget in the same slot


GPU_WORKSPACE = "[workspace]\nmembers = [\n    \"tig-algorithms\",\n]\n"


def _gpu_monorepo(tmp_path):
    mono = tmp_path / "mono"
    (mono / "tig-binary" / "src").mkdir(parents=True)
    (mono / "tig-binary" / "src" / "framework.cu").write_text("// framework\n")
    (mono / "tig-challenges" / "src" / "hypergraph").mkdir(parents=True)
    (mono / "tig-challenges" / "src" / "hypergraph" / "kernels.cu").write_text("// challenge\n")
    algo = mono / "tig-algorithms" / "src" / "hypergraph" / "talos_cand"
    algo.mkdir(parents=True)
    (algo / "kernels.cu").write_text("// algorithm\n")
    (mono / "Cargo.toml").write_text(GPU_WORKSPACE)
    return mono


def test_build_native_ptx_concatenates_like_build_ptx_and_skips_fuel_injection(tmp_path):
    mono = _gpu_monorepo(tmp_path)
    seen = []

    def run(cmd, **kw):
        seen.append(cmd)
        seen.append(Path(cmd[2]).read_text())  # the temp .cu, read before it is deleted
        Path(cmd[cmd.index("-o") + 1]).write_text("// ptx\n")
        return Result(0, "", "")
    ok, _ = inside.build_native_ptx(mono, "hypergraph", "talos_cand", run)
    cmd, code = seen
    assert ok
    # mutation: a different order or missing file compiles a different PTX than TIG's
    assert code.index("// framework") < code.index("// challenge") < code.index("// algorithm")
    # the flags build_ptx passes at MONOREPO_REF
    assert cmd[0] == "nvcc" and cmd[1] == "-ptx"
    assert cmd[cmd.index("-arch") + 1] == "compute_70" and cmd[cmd.index("-code") + 1] == "sm_70"
    assert "--use_fast_math" in cmd and "-dopt=on" in cmd
    dest = Path(cmd[cmd.index("-o") + 1])
    # mutation: writing over the metered <name>.ptx lets a native PTX reach tig-runtime
    assert dest.name == "talos_cand.native.ptx"


def test_build_native_runs_cargo_with_the_pinned_toolchain_and_fast_profile(tmp_path):
    mono = _gpu_monorepo(tmp_path)
    seen = []

    def run(cmd, **kw):
        seen.append((cmd, kw))
        if cmd[0] == "nvcc":
            Path(cmd[cmd.index("-o") + 1]).write_text("// ptx\n")
        return Result(0, "Compiling talos-native\n", "")
    ok, out = inside.build_native(mono, "hypergraph", "talos_cand", run)
    assert ok and "Compiling talos-native" in out
    (nvcc, _), (cargo, kw) = seen
    assert nvcc[0] == "nvcc"  # GPU: the PTX first, the runner loads it at run time
    assert cargo == ["cargo", "+nightly-2025-02-10", "build", "--release", "-p", "talos-native"]
    assert kw["cwd"] == mono
    # mutation: the workspace profile (lto = true, codegen-units = 1) makes the native build
    # as slow as the metered one it replaces
    assert kw["env"]["CARGO_PROFILE_RELEASE_LTO"] == "false"
    assert kw["env"]["CARGO_PROFILE_RELEASE_CODEGEN_UNITS"] == "16"
    assert (mono / "talos-native" / "src" / "main.rs").exists()


def test_build_native_stops_at_a_failed_ptx_build(tmp_path):
    mono = _gpu_monorepo(tmp_path)
    seen = []

    def run(cmd, **kw):
        seen.append(cmd[0])
        return Result(1, "", "kernels.cu(3): error: identifier undefined")
    ok, out = inside.build_native(mono, "hypergraph", "talos_cand", run)
    # mutation: carrying on to cargo reports the Rust build and hides the CUDA error
    assert not ok and seen == ["nvcc"] and "identifier undefined" in out


def test_build_native_cpu_builds_no_ptx(tmp_path):
    mono = make_monorepo(tmp_path)
    (mono / "Cargo.toml").write_text(GPU_WORKSPACE)
    seen = []
    inside.build_native(mono, "knapsack", "talos_cand",
                        lambda cmd, **kw: seen.append(cmd[0]) or Result(0))
    assert seen == ["cargo"]


def test_native_artifact_paths(tmp_path):
    binary, ptx = inside.native_artifact_paths(tmp_path, "hypergraph", "talos_cand")
    assert binary == tmp_path / "target" / "release" / "talos-native" and ptx is None
    p = tmp_path / "tig-algorithms" / "lib" / "hypergraph" / "ptx" / "talos_cand.native.ptx"
    p.parent.mkdir(parents=True)
    p.write_text("x")
    assert inside.native_artifact_paths(tmp_path, "hypergraph", "talos_cand")[1] == p
```

In the same file, update the two existing 10-tuple helpers and assertions to the 12-tuple form. In `_task`, return `("c003", "t", "ab" * 32, nonce, so, 10, timeout_s, None, str(workdir), hp, "metered", None)`.

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_inside.py -q`
Expected: FAIL (`AttributeError: module 'talos.inside' has no attribute 'run_nonce_native'`, `TypeError` on `content_hash`'s fourth argument, and the tuple-unpack error in `run_task`).

- [ ] **Step 3: Implement in `talos/inside.py`**

Add imports: `import glob`, `import os`, `from talos import native_runner` and `from talos.challenges import CHALLENGES`.

Replace `content_hash`:

```python
def content_hash(files: dict[str, str], monorepo_ref: str, dev_image_tag: str,
                 mode: str = "metered") -> str:
    """Artifact cache key. The monorepo pin, the dev image tag and the crate layout are part
    of it: the same sources built against a different monorepo, or beside a different set of
    modules, are a different .so. A native build is a different artifact from a metered one;
    the mode enters the hash only when it is not "metered", so every metered id is unchanged."""
    h = hashlib.sha256()
    h.update(monorepo_ref.encode())
    h.update(b"\0")
    h.update(dev_image_tag.encode())
    h.update(b"\0")
    h.update(CRATE_LAYOUT.encode())
    h.update(b"\0")
    if mode != "metered":
        h.update(f"mode={mode}".encode())
        h.update(b"\0")
    for k in sorted(files):
        h.update(k.encode())
        h.update(b"\0")
        h.update(files[k].encode())
        h.update(b"\0")
    return h.hexdigest()[:32]
```

Add after `build`:

```python
# Profile overrides for the native build: the workspace's release profile (lto = true,
# codegen-units = 1) is what makes TIG's build slow, and the native binary is never submitted.
NATIVE_ENV = {"CARGO_PROFILE_RELEASE_LTO": "false", "CARGO_PROFILE_RELEASE_CODEGEN_UNITS": "16"}
NATIVE_PTX_SUFFIX = ".native.ptx"


def _native_ptx(monorepo: Path, challenge: str, name: str) -> Path:
    return monorepo / "tig-algorithms" / "lib" / challenge / "ptx" / f"{name}{NATIVE_PTX_SUFFIX}"


def build_native_ptx(monorepo: Path, challenge: str, name: str,
                     run=subprocess.run) -> tuple[bool, str]:
    """tig-binary/scripts/build_ptx at MONOREPO_REF without inject_fuel_and_runtime_sig: the
    same files in the same order (framework, the challenge's .cu by the same recursive glob,
    the algorithm's), the same nvcc flags, written beside the metered PTX, never over it."""
    framework = monorepo / "tig-binary" / "src" / "framework.cu"
    challenge_cus = glob.glob(str(monorepo / "tig-challenges" / "src" / challenge / "**" / "*.cu"),
                              recursive=True)
    algo_cus = glob.glob(str(_algo_root(monorepo, challenge) / name / "*.cu"))
    if not algo_cus:
        return False, f"no .cu files in the {name} algorithm; a GPU algorithm needs its kernels"
    code = framework.read_text(encoding="utf-8") + "\n"
    for p in challenge_cus:
        code += Path(p).read_text(encoding="utf-8") + "\n"
    for p in algo_cus:
        code += Path(p).read_text(encoding="utf-8") + "\n\n"
    dest = _native_ptx(monorepo, challenge, name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        cu = Path(td) / "temp.cu"
        cu.write_text(code, encoding="utf-8", newline="\n")
        r = run(["nvcc", "-ptx", str(cu), "-o", str(dest), "-arch", "compute_70",
                 "-code", "sm_70", "--use_fast_math", "-dopt=on"],
                capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.returncode == 0, (r.stdout or "") + (r.stderr or "")


def build_native(monorepo: Path, challenge: str, name: str,
                 run=subprocess.run) -> tuple[bool, str]:
    """Stage the talos-native runner, build the unmetered PTX for a GPU challenge, then
    `cargo build` the runner. The algorithm must already be staged (stage_algorithm)."""
    is_gpu = CHALLENGES[challenge].is_gpu
    native_runner.stage(monorepo, challenge, name, is_gpu)
    out = ""
    if is_gpu:
        ok, ptx_out = build_native_ptx(monorepo, challenge, name, run)
        out += ptx_out
        if not ok:
            return False, out[-BUILD_OUTPUT_CAP:]
    r = run(["cargo", native_runner.TOOLCHAIN, "build", "--release", "-p", native_runner.PACKAGE],
            cwd=monorepo, capture_output=True, text=True, encoding="utf-8", errors="replace",
            env={**os.environ, **NATIVE_ENV})
    out += (r.stdout or "") + (r.stderr or "")
    return r.returncode == 0, out[-BUILD_OUTPUT_CAP:]


def native_artifact_paths(monorepo: Path, challenge: str, name: str) -> tuple[Path, Path | None]:
    binary = monorepo / "target" / "release" / native_runner.PACKAGE
    ptx = _native_ptx(monorepo, challenge, name)
    return binary, (ptx if ptx.exists() else None)
```

Refactor `run_nonce` so that the runtime call, verifier call and output-file read are shared with the native path. Replace `run_nonce` with:

```python
def _settings(challenge_id: str, track: str) -> str:
    return json.dumps({"algorithm_id": "", "challenge_id": challenge_id, "track_id": track,
                       "block_id": "", "player_id": ""}, separators=(",", ":"))


def _hp_args(hyperparameters: dict | None) -> list[str]:
    return ([] if hyperparameters is None else
            ["--hyperparameters", json.dumps(hyperparameters, separators=(",", ":"))])


def _run_and_verify(cmd: list[str], out_file: Path, settings: str, rand_hash: str, nonce: int,
                    gpu_args: list[str], timeout_s: int, run, clock) -> dict:
    """Runs the solver (`cmd`), then tig-verifier on what it saved. Returns the classified row
    fields plus the saved JSON document (None when nothing was saved)."""
    t0 = clock()
    timed_out = False
    try:
        r1 = run(cmd, capture_output=True, text=True, timeout=timeout_s,
                 encoding="utf-8", errors="replace")
        rt_rc = r1.returncode
    except subprocess.TimeoutExpired:
        timed_out, rt_rc = True, -1
    elapsed = clock() - t0
    quality = None
    ver_rc = 1
    doc = None
    if not timed_out and out_file.exists():
        try:
            doc = json.loads(out_file.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            doc = None
        # The verifier gets only the time the runtime left. Giving it a fresh timeout_s put
        # the worst case at 2 x timeout_s, past the Modal function timeout, which kills the
        # container and turns a slow nonce into an infrastructure error instead of a result.
        # A verifier timeout must not raise either: TimeoutExpired's str carries the whole
        # argv, rand_hash included, and it would surface in a client-side error message.
        try:
            r2 = run(["tig-verifier", settings, rand_hash, str(nonce), str(out_file)] + gpu_args,
                     capture_output=True, text=True,
                     timeout=max(1, int(timeout_s - elapsed)),
                     encoding="utf-8", errors="replace")
            ver_rc = r2.returncode
            m = _QUALITY_RE.search(r2.stdout or "")
            quality = int(m.group(1)) if m else None
        except subprocess.TimeoutExpired:
            timed_out = True
    ok, err = classify(rt_rc, ver_rc, quality, timed_out)
    return {"ok": ok, "quality": quality if ok else None, "runtime_ms": int(elapsed * 1000),
            "error": err, "doc": doc if isinstance(doc, dict) else None,
            "limit_hit": rt_rc == OUT_OF_FUEL_RC}


def _int_or_none(v) -> int | None:
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def run_nonce(challenge_id: str, track: str, rand_hash: str, nonce: int, so: Path, fuel: int,
              timeout_s: int, ptx: Path | None, run=subprocess.run,
              workdir: Path | None = None, clock=time.monotonic,
              hyperparameters: dict | None = None) -> dict:
    """Mirrors scripts/test_algorithm in the monorepo:
    `tig-runtime SETTINGS RAND_HASH NONCE SO --fuel F --output DIR [--hyperparameters JSON]
    [--ptx P --gpu 0]` writes DIR/<nonce>.json, then
    `tig-verifier SETTINGS RAND_HASH NONCE DIR/<nonce>.json [--ptx P --gpu 0]`
    prints `quality: N` and exits 0 on a valid solution. `{}` is passed as `{}`: the algorithm
    receives Some(empty map), not None. The row carries the fuel_consumed tig-runtime wrote at
    the last save, which calibration pairs with the native solve time."""
    settings = _settings(challenge_id, track)
    gpu_args = ["--ptx", str(ptx), "--gpu", "0"] if ptx else []
    with tempfile.TemporaryDirectory(dir=workdir) as td:
        out_file = Path(td) / f"{nonce}.json"
        cmd = ["tig-runtime", settings, rand_hash, str(nonce), str(so),
               "--fuel", str(fuel), "--output", td] + _hp_args(hyperparameters) + gpu_args
        v = _run_and_verify(cmd, out_file, settings, rand_hash, nonce, gpu_args, timeout_s,
                            run, clock)
    fuel_consumed = _int_or_none((v["doc"] or {}).get("fuel_consumed"))
    return {"track": track, "nonce": nonce, "ok": v["ok"], "quality": v["quality"],
            "runtime_ms": v["runtime_ms"], "error": v["error"], "fuel_consumed": fuel_consumed,
            "limit_hit": v["limit_hit"]}


def run_nonce_native(challenge_id: str, track: str, rand_hash: str, nonce: int, binary: Path,
                     budget_us: int | None, timeout_s: int, ptx: Path | None,
                     run=subprocess.run, workdir: Path | None = None, clock=time.monotonic,
                     hyperparameters: dict | None = None) -> dict:
    """`talos-native SETTINGS RAND_HASH NONCE OUT_FILE [--budget-us B] [--hyperparameters JSON]
    [--ptx P]`, then tig-verifier exactly as run_nonce calls it. The runner exits 87 when its
    solve has run for budget_us, so a budget exit is classified as a metered out-of-fuel exit
    is: a saved solution that verifies still counts. `timeout_s` is the outer cap and reports
    `timeout`, as on the metered path."""
    settings = _settings(challenge_id, track)
    gpu_args = ["--ptx", str(ptx), "--gpu", "0"] if ptx else []
    with tempfile.TemporaryDirectory(dir=workdir) as td:
        out_file = Path(td) / f"{nonce}.json"
        cmd = [str(binary), settings, rand_hash, str(nonce), str(out_file)]
        if budget_us is not None:
            cmd += ["--budget-us", str(budget_us)]
        cmd += _hp_args(hyperparameters) + (["--ptx", str(ptx)] if ptx else [])
        v = _run_and_verify(cmd, out_file, settings, rand_hash, nonce, gpu_args, timeout_s,
                            run, clock)
    solve_us = _int_or_none((v["doc"] or {}).get("solve_us"))
    return {"track": track, "nonce": nonce, "ok": v["ok"], "quality": v["quality"],
            "runtime_ms": v["runtime_ms"], "error": v["error"], "solve_us": solve_us,
            "limit_hit": v["limit_hit"]}
```

Replace the `NonceTask` comment and `run_task`:

```python
# One scoring task, as a positional tuple so it pickles into a worker process unchanged:
# (challenge_id, track, rand_hash, nonce, so, fuel, timeout_s, ptx, workdir, hyperparameters,
#  mode, budget_us), with `so`, `ptx` and `workdir` as strings. For mode "native", `so` is the
# talos-native binary and `fuel` is unused; budget_us is the track's solve budget or None.
NonceTask = tuple


def run_task(task: NonceTask, run=subprocess.run) -> dict:
    (challenge_id, track, rand_hash, nonce, so, fuel, timeout_s, ptx, workdir, hp, mode,
     budget_us) = task
    if mode == "native":
        return run_nonce_native(challenge_id, track, rand_hash, nonce, Path(so), budget_us,
                                timeout_s, Path(ptx) if ptx else None, run=run,
                                workdir=Path(workdir), hyperparameters=hp)
    return run_nonce(challenge_id, track, rand_hash, nonce, Path(so), fuel, timeout_s,
                     Path(ptx) if ptx else None, run=run, workdir=Path(workdir),
                     hyperparameters=hp)
```

`test_run_task_dispatches_on_the_mode` reads positional slot 5 of each stubbed call. That holds for the signatures above: slot 5 is `fuel` for `run_nonce` and `budget_us` for `run_nonce_native`.

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_inside.py -q`
Expected: PASS. `tests/test_c3_job.py` and `tests/test_talos_bench.py` still fail on the 10-tuple asserts; Tasks 4 and 6 fix them.

- [ ] **Step 5: Commit**

```bash
git status --short
git add talos/inside.py tests/test_inside.py
git commit -m "inside: native build and nonce run beside the metered ones; fuel_consumed on metered rows"
```

---

### Task 4: The C3/local job runs native payloads; local warm-up fetches every crate

**Files:**
- Modify: `talos/c3_job.py`, `talos/c3_jobdir.py`, `talos/local_transport.py`
- Test: `tests/test_c3_job.py`, `tests/test_c3_jobdir.py`, `tests/test_local_transport.py`

**Interfaces:**
- Consumes: `EvalRequest.mode`, `EvalRequest.fuel_budgets_us` (Task 1); `inside.build_native`, `inside.native_artifact_paths`, `content_hash(..., mode)`, the 12-field `NonceTask` (Task 3).
- Produces: payload keys `"mode"` and `"fuel_budgets_us"`, present only for native requests. `JOB_MODULES` includes `"native_runner"`. The local `job.sh` exports `CARGO_NET_OFFLINE=true`. `WARM_MARKER = ".talos-warm-2"`, and `warm_script` ends with `cargo +nightly-2025-02-10 fetch`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_c3_jobdir.py`:

```python
def _native_req(budgets=None):
    return EvalRequest("knapsack", {"mod.rs": "fn x(){}"}, [NonceSet("t", "ab" * 32, 0, 2)], [],
                       7, None, CHALLENGES["knapsack"].beat, mode="native",
                       fuel_budgets_us=budgets)


def test_native_payload_carries_mode_and_budgets_and_metered_carries_neither():
    p = c3_jobdir.payload(_native_req({"t": 1_500_000}))
    assert p["mode"] == "native" and p["fuel_budgets_us"] == {"t": 1_500_000}
    metered = EvalRequest("knapsack", {"mod.rs": "fn x(){}"}, [NonceSet("t", "ab" * 32, 0, 2)],
                          [], 7, None, CHALLENGES["knapsack"].beat)
    # mutation: always writing the keys changes request_hash for every metered request, so a
    # resumed job cannot find the C3 job it is still paying for
    assert "mode" not in c3_jobdir.payload(metered)
    assert "fuel_budgets_us" not in c3_jobdir.payload(metered)
    assert c3_jobdir.request_hash(_native_req()) != c3_jobdir.request_hash(metered)


def test_the_local_job_runs_cargo_offline_and_the_c3_job_does_not():
    assert "export CARGO_NET_OFFLINE=true\n" in c3_jobdir.local_job_sh_text()
    # C3 containers have a network; offline there would fail a cold registry
    assert "CARGO_NET_OFFLINE" not in c3_jobdir.job_sh_text("abc")


def test_every_job_module_imports_only_job_modules(tmp_path):
    import ast
    from pathlib import Path
    root = Path(c3_jobdir.__file__).resolve().parent
    for mod in c3_jobdir.JOB_MODULES:
        tree = ast.parse((root / f"{mod}.py").read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = ([node.module] if isinstance(node, ast.ImportFrom) and node.module else
                     [a.name for a in node.names] if isinstance(node, ast.Import) else [])
            if isinstance(node, ast.ImportFrom) and node.module == "talos":
                names = [f"talos.{a.name}" for a in node.names]
            for name in names:
                if name.startswith("talos."):
                    # mutation: inside.py importing a module the job dir does not ship fails
                    # every C3 and local job at import, before it can report anything
                    assert name.split(".")[1] in c3_jobdir.JOB_MODULES, (mod, name)
    job = c3_jobdir.write_job_dir(tmp_path / "j", _native_req(), "1",
                                  hardware="cpu-d3-4vcpu-16gb")
    assert (job / "talos" / "native_runner.py").exists()
```

Append to `tests/test_c3_job.py`:

```python
def native_setup(tmp_path, budgets=None, n=2):
    mono = tmp_path / "mono"
    (mono / "tig-algorithms" / "src" / "knapsack").mkdir(parents=True)
    (mono / "tig-algorithms" / "src" / "knapsack" / "mod.rs").write_text("// c003\n")
    (mono / "Cargo.toml").write_text("[workspace]\nmembers = [\n    \"tig-algorithms\",\n]\n")
    req = EvalRequest("knapsack", {"mod.rs": "fn x(){}"}, [NonceSet("t", HASH, 0, n)], [], 7,
                      None, CHALLENGES["knapsack"].beat, mode="native", fuel_budgets_us=budgets)
    work = write_job_dir(tmp_path / "work", req, "1", hardware="cpu-d3-4vcpu-16gb")
    art = tmp_path / "art"
    art.mkdir()
    return mono, work, art


def fake_native_run(quality=120, build_rc=0):
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        if cmd[0] == "cargo":
            binary, _ = inside.native_artifact_paths(Path(kw["cwd"]), "knapsack", "talos_cand")
            binary.parent.mkdir(parents=True, exist_ok=True)
            binary.write_bytes(b"\x7fELF")
            return Result(build_rc, "Compiling talos-native\n", "" if build_rc == 0 else "error")
        if cmd[0].endswith("talos-native"):
            Path(cmd[4]).write_text('{"nonce": 0, "solution": "e30=", "solve_us": 900}')
            return Result(0)
        if cmd[0] == "tig-verifier":
            return Result(0, f"quality: {quality}\n")
        raise AssertionError(cmd)
    run.calls = calls
    return run


def test_a_native_payload_builds_and_scores_with_the_runner(tmp_path):
    from talos.challenges import DEV_IMAGE_TAG, MONOREPO_REF
    mono, work, art = native_setup(tmp_path, budgets={"t": 1_500_000})
    run = fake_native_run()
    c3_job.main(workdir=work, artifacts_dir=art, run=run, monorepo=mono, log=lambda *a: None)
    r = json.loads((art / "results.json").read_text())
    # mutation: ignoring the mode builds with build_algorithm and scores with tig-runtime
    assert not any(c[0] in ("build_algorithm", "tig-runtime") for c in run.calls)
    assert r["compile"]["ok"]
    assert r["compile"]["artifact_id"] == inside.content_hash(
        {"mod.rs": "fn x(){}"}, MONOREPO_REF, DEV_IMAGE_TAG, "native")
    assert [x["solve_us"] for x in r["training"]] == [900, 900]
    runner = next(c for c in run.calls if c[0].endswith("talos-native"))
    assert runner[runner.index("--budget-us") + 1] == "1500000"


def test_a_native_build_failure_is_a_compile_result(tmp_path):
    mono, work, art = native_setup(tmp_path)
    c3_job.main(workdir=work, artifacts_dir=art, run=fake_native_run(build_rc=101),
                monorepo=mono, log=lambda *a: None)
    r = json.loads((art / "results.json").read_text())
    assert r["compile"]["ok"] is False and r["holdout_reason"] == "not_compiled"


def test_native_pool_tasks_carry_the_mode_and_each_tracks_budget(tmp_path, monkeypatch):
    mono, work, art = native_setup(tmp_path, budgets={"t": 1_500_000})
    seen = []
    monkeypatch.setattr(inside, "run_task", lambda task: seen.append(task) or {
        "track": task[1], "nonce": task[3], "ok": True, "quality": 1, "runtime_ms": 1,
        "error": None})
    FakePool.sizes = []
    c3_job.main(workdir=work, artifacts_dir=art, run=fake_native_run(), monorepo=mono,
                log=lambda *a: None, pool_factory=FakePool)
    binary, _ = inside.native_artifact_paths(mono, "knapsack", inside.ALGO_NAME)
    assert {t[3]: t for t in seen}[0] == ("c003", "t", HASH, 0, str(binary), 7,
                                          inside.NONCE_TIMEOUT_S, None, str(mono), None,
                                          "native", 1_500_000)
```

In `test_the_pool_path_passes_run_task_positional_tuples_and_sorts_the_rows`, extend the expected tuple with `"metered", None` at the end.

Append to `tests/test_local_transport.py`:

```python
def test_warm_script_fetches_every_locked_crate_after_the_build():
    script = warm_script("knapsack")
    build = script.index('build_algorithm "$name"')
    fetch = script.index("cargo +nightly-2025-02-10 fetch")
    # mutation: no fetch leaves tig-structs' and tig-utils' crates out of the registry, and
    # the first native build in the network-less job container fails to resolve them
    assert build < fetch < script.index(f"touch /app/{WARM_MARKER}")
    # mutation: keeping the old marker never re-warms a volume made before this change
    assert WARM_MARKER == ".talos-warm-2"
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_c3_jobdir.py tests/test_c3_job.py tests/test_local_transport.py -q`
Expected: FAIL on every new test.

- [ ] **Step 3: Implement**

`talos/c3_jobdir.py`:
- `JOB_MODULES = ("__init__", "inside", "scoring", "types", "challenges", "diagnostics", "native_runner", "c3_job")`.
- In `local_job_sh_text`, insert `"export CARGO_NET_OFFLINE=true\n"` after `set -euo pipefail\n`, and add a sentence to the docstring: "Cargo runs offline: the job container has no network, and offline turns a network wait into an immediate error."
- In `payload`, after the hyperparameters block:

```python
    if request.mode != "metered":
        # Only for native, so every metered request hashes exactly as before (see above).
        p["mode"] = request.mode
        p["fuel_budgets_us"] = request.fuel_budgets_us
```

`talos/c3_job.py`:
- In `_score`, add `mode = payload.get("mode", "metered")` and `budgets = payload.get("fuel_budgets_us") or {}`, and extend each task tuple with `mode, budgets.get(ns["track"])`.
- In `main`, replace the build call and artifact lookup:

```python
        mode = payload.get("mode", "metered")
        log(f"[{int(clock() - t0)}s] build start ({mode})")
        if mode == "native":
            ok, build_out = inside.build_native(monorepo, challenge, inside.ALGO_NAME, run=run)
        else:
            ok, build_out = inside.build(monorepo, challenge, inside.ALGO_NAME, run=run)
```

and after the `except` block:

```python
    if mode == "native":
        so, ptx = inside.native_artifact_paths(monorepo, challenge, inside.ALGO_NAME)
    else:
        so, ptx = inside.artifact_paths(monorepo, challenge, inside.ALGO_NAME)
```

Initialise `mode = payload.get("mode", "metered")` before the `try`, so the `except` path and the later code both see it. Change the "build produced no .so" message to `f"build produced no {'runner' if mode == 'native' else '.so'} at {so}"`. Pass `payload.get("mode", "metered")` as the fourth argument of `inside.content_hash`.

`talos/local_transport.py`:
- `WARM_MARKER = ".talos-warm-2"`. Add a comment: "-2: the warm-up also runs `cargo fetch`, which the native build needs offline; a volume warmed before that is warmed again."
- In `warm_script`, before `touch`, add: `"cargo +nightly-2025-02-10 fetch\n"`. Import the toolchain constant: `from talos.native_runner import TOOLCHAIN` and write `f"cargo {TOOLCHAIN} fetch\n"`. Extend the docstring: "`cargo fetch` then downloads every crate in Cargo.lock, including the ones the talos-native runner uses and the metered build does not (tig-structs, tig-utils), so a native build in the network-less job container resolves offline."

- [ ] **Step 4: Run to verify they pass, then the whole suite**

Run: `.venv/bin/python -m pytest tests/test_c3_jobdir.py tests/test_c3_job.py tests/test_local_transport.py -q && .venv/bin/python -m pytest -q -m "not live"`
Expected: PASS, except `tests/test_talos_bench.py`'s tuple asserts, which Task 6 fixes. Run `.venv/bin/python -m pytest -q -m "not live" --deselect tests/test_talos_bench.py` if that file is the only red one.

- [ ] **Step 5: Commit**

```bash
git status --short
git add talos/c3_job.py talos/c3_jobdir.py talos/local_transport.py tests/test_c3_job.py tests/test_c3_jobdir.py tests/test_local_transport.py
git commit -m "c3_job: native payloads build and score with the runner; local warm-up fetches every crate"
```

---

### Task 5: Local integration check on knapsack (manual, real Docker, no credit)

This check runs Talos's own local backend end to end. It is the first time the rendered Rust compiles. It spends no C3 or Modal credit, but it pulls the knapsack dev image (about 10–13 GB; 35 GB was free on 2026-09-29, MEASURED with `df -h /`) and runs for several minutes. Ask the user before starting it.

**Files:**
- Create: `<scratchpad>/native_local_check.py` (the session scratchpad, never the repo)
- Modify: `docs/ai/plans/2026-09-29-native-research-scoring.md` (append a results block at the end)

**Interfaces:**
- Consumes: Tasks 1–4.
- Produces: MEASURED native and metered qualities, `fuel_consumed`, `solve_us` and build times for 3 knapsack nonces on one track, and a yes/no on whether the native build works offline in the job container. Task 11's docs cite these numbers.

- [ ] **Step 1: Write the check script**

```python
"""Throwaway. Metered vs native on the local backend, knapsack, mainnet top algorithm, 3 nonces
of the first track. Prints a JSON summary; the rand hash is never printed."""
import json
import sys
import time
from pathlib import Path

from talos import mainnet
from talos.bench import EvalRequest, PendingJobStore
from talos.challenges import CHALLENGES
from talos.cli import local_settings, make_bench
from talos.local_transport import prepare
from talos.nonces import draw_nonce_sets, new_rand_hash

ch = "knapsack"
run_dir = Path(sys.argv[1])
prepare(ch)  # pulls the image, clones, warms (now with cargo fetch)
info = mainnet.fetch_challenge_info(ch)
name, _id, _adoption = mainnet.top_algorithm(ch)
files = mainnet.fetch_algorithm_files(ch, name)
tr, _ = draw_nonce_sets(info.tracks[:1], new_rand_hash(), training_count=3, holdout_count=0)
# the daemon's own limits: a hard-coded cpus=8 is refused by a Docker with fewer CPUs
bench = make_bench("local", run_dir, PendingJobStore.memory(), local=local_settings(None))
out = {"algorithm": name, "track": tr[0].track}
for mode in ("metered", "native"):
    t0 = time.time()
    r = bench.evaluate(EvalRequest(ch, files, tr, [], info.max_fuel, None,
                                   CHALLENGES[ch].beat, mode=mode))
    out[mode] = {"wall_s": round(time.time() - t0, 1), "compiled": r.compile.ok,
                 "compile_tail": None if r.compile.ok else r.compile.output[-3000:],
                 "rows": [x.to_dict() for x in r.training]}
m = {x["nonce"]: x for x in out["metered"]["rows"]}
n = {x["nonce"]: x for x in out["native"]["rows"]}
out["quality_equal"] = all(m[k]["quality"] == n[k]["quality"] for k in m)
print(json.dumps(out, indent=1))
```

- [ ] **Step 2: Ask the user, then run it in the background with a log file**

Run: `.venv/bin/python <scratchpad>/native_local_check.py <scratchpad>/localcheck > <scratchpad>/localcheck.log 2>&1` in the background (Bash `run_in_background`, timeout 7200000), and poll the log.
Expected: `"compiled": true` for both modes, `"quality_equal": true`, every metered row with an integer `fuel_consumed`, and every native row with an integer `solve_us`.

- [ ] **Step 3: If the native build fails, diagnose before changing anything**

Read `compile_tail`. The three likely causes and their fixes:
- *unresolved crate, or cargo trying the network*: the `cargo fetch` warm-up did not run. Check that the volume was re-warmed (`.talos-warm-2` exists on it).
- *Rust type error in `main.rs`*: fix the template in `talos/native_runner.py` and add a render test that pins the corrected line.
- *toolchain missing*: `nightly-2025-02-10` is not in the image; stop and report.

Any template fix goes back through Task 2's test cycle and is committed on its own.

- [ ] **Step 4 (optional, only if `df -h /` shows 30 GB or more free after Step 2): compile the GPU variant**

No GPU is needed to compile. Render the GPU runner into the scratchpad (`$S`), then build it in the hypergraph dev image against a shipped algorithm copied in as `talos_cand`:

```bash
.venv/bin/python -c "
import pathlib
from talos import native_runner
for rel, text in native_runner.render('hypergraph', 'talos_cand', True).items():
    p = pathlib.Path('$S/gpu-runner') / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
"
docker run --rm -v "$S/gpu-runner:/runner:ro" \
  ghcr.io/tig-foundation/tig-monorepo/hypergraph/dev:0.0.7 bash -c '
set -euo pipefail
mkdir -p /app && cd /app
curl -fsSL https://codeload.github.com/tig-foundation/tig-monorepo/tar.gz/84a5787f5b14a630bdf40f52bccf37887d3d8464 \
  | tar xz --strip-components=1
src=$(ls -d tig-algorithms/src/hypergraph/*/ | head -1)
cp -r "$src" tig-algorithms/src/hypergraph/talos_cand
printf "pub mod talos_cand;\n" > tig-algorithms/src/hypergraph/mod.rs
cp -r /runner/talos-native /app/talos-native
sed -i "s/^members = \[/members = [\n    \"talos-native\",/" Cargo.toml
CARGO_PROFILE_RELEASE_LTO=false CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 \
  cargo +nightly-2025-02-10 build --release -p talos-native' > "$S/gpu-compile.log" 2>&1
echo "rc=$?"
```

Expected: `rc=0`. On a Rust error, fix the GPU template and pin the corrected line in a Task 2 render test, as in Step 3. Record pass or fail. If this step is skipped, say so in the results block: the GPU template is then first compiled in the human-run live test (Task 11). Remove the image afterwards if disk is short (`docker image rm ghcr.io/tig-foundation/tig-monorepo/hypergraph/dev:0.0.7`), and ask the user first.

- [ ] **Step 5: Append the results block to this plan and commit it**

Record, each labelled MEASURED with the command that produced it: the algorithm, the per-nonce metered and native qualities, `fuel_consumed`, `solve_us`, the metered and native `wall_s`, and whether Step 4 ran.

```bash
git add docs/ai/plans/2026-09-29-native-research-scoring.md
git commit -m "docs(ai): local knapsack native vs metered results"
```

Stop here and report to the user if `quality_equal` is false. The design depends on it.

---

### Task 6: Modal app and client carry the mode

**Files:**
- Modify: `modal_app/talos_bench.py`, `talos/bench.py` (`ModalBench`)
- Test: `tests/test_talos_bench.py`, `tests/test_bench.py`

**Interfaces:**
- Consumes: Tasks 1 and 3.
- Produces: `content_hash(files, mode="metered")` in `modal_app/talos_bench.py`; `_compile_impl(name, files, mode="metered")`; `_score_batch_impl(name, challenge_id, artifact_id, tasks, workers, mode="metered", pool_factory=None, clock=...)`; the deployed functions `compile_fn(files, mode="metered")` and `score_batch(artifact_id, tasks, mode="metered")`. `ModalBench` passes `mode="native"` only for native requests, and each native task dict carries `"budget_us"`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_talos_bench.py`:

```python
def test_native_compile_stores_the_runner_under_its_own_hash(monkeypatch, tmp_path):
    _sandbox(monkeypatch, tmp_path)

    def build_native(mono, name, algo, run=None):
        binary, _ = talos_bench.inside.native_artifact_paths(mono, name, algo)
        binary.parent.mkdir(parents=True, exist_ok=True)
        binary.write_bytes(b"\x7fELF")
        return True, "Compiling talos-native"
    monkeypatch.setattr(talos_bench.inside, "build_native", build_native)
    monkeypatch.setattr(talos_bench.inside, "build",
                        lambda *a, **k: pytest.fail("metered build in native mode"))
    out = talos_bench._compile_impl("knapsack", {"mod.rs": "fn x(){}"}, mode="native")
    assert out["ok"] and out["artifact_id"] == talos_bench.content_hash({"mod.rs": "fn x(){}"},
                                                                        mode="native")
    # mutation: storing the runner under the metered id serves it to a metered score call
    assert out["artifact_id"] != talos_bench.content_hash({"mod.rs": "fn x(){}"})
    dest = tmp_path / "artifacts" / "knapsack" / out["artifact_id"]
    assert (dest / "talos-native").exists() and not (dest / "algo.so").exists()
    again = talos_bench._compile_impl("knapsack", {"mod.rs": "fn x(){}"}, mode="native")
    assert again["output"] == "cached"


def test_native_score_batch_runs_the_runner_with_each_tasks_budget(monkeypatch, tmp_path):
    _sandbox(monkeypatch, tmp_path)
    art = tmp_path / "artifacts" / "knapsack" / "nat1"
    art.mkdir(parents=True)
    (art / "talos-native").write_bytes(b"\x7fELF")
    seen = []
    monkeypatch.setattr(talos_bench.inside, "run_task", lambda t: seen.append(t) or {
        "track": t[1], "nonce": t[3], "ok": True, "quality": 1, "runtime_ms": 1, "error": None})
    task = {**_batch_task(0), "budget_us": 2_000_000}
    talos_bench._score_batch_impl("knapsack", "c003", "nat1", [task], workers=1,
                                  mode="native", pool_factory=FakePool)
    assert seen[0] == ("c003", "t", "ab" * 32, 0, str(art / "talos-native"), 5, 60, None,
                       str(tmp_path / "mono"), None, "native", 2_000_000)


def test_native_score_batch_reports_a_missing_runner_as_infrastructure(monkeypatch, tmp_path):
    _sandbox(monkeypatch, tmp_path)
    art = tmp_path / "artifacts" / "knapsack" / "nat1"
    art.mkdir(parents=True)
    # mutation: native mode looking for algo.so finds this one and runs it as the runner
    (art / "algo.so").write_bytes(b"\x7fELF")
    with pytest.raises(FileNotFoundError):
        talos_bench._score_batch_impl("knapsack", "c003", "nat1", [_batch_task(0)], workers=1,
                                      mode="native")
```

(add `import pytest` at the top of the file if it is missing). In `test_score_batch_runs_the_tasks_through_a_pool_and_returns_them_in_nonce_order`, extend the expected tuple with `"metered", None`.

Append to `tests/test_bench.py`:

```python
def test_metered_modal_calls_are_unchanged_and_native_ones_add_mode(monkeypatch):
    remotes, starmaps = [], []

    class Fn:
        def hydrate(self):
            pass

        def remote(self, *args, **kwargs):
            remotes.append((args, kwargs))
            return {"ok": True, "artifact_id": "art", "output": "ok"}

        def starmap(self, args):
            starmaps.append(list(args))
            return [{"rows": [{"track": t["track"], "nonce": t["nonce"], "ok": True,
                               "quality": 100, "runtime_ms": 1, "error": None}
                              for t in a[1]], "seconds": 1.0} for a in args]

    mod = types.ModuleType("modal")
    mod.exception = types.SimpleNamespace(NotFoundError=type("NotFoundError", (Exception,), {}))
    mod.Function = types.SimpleNamespace(from_name=lambda app, name: Fn())
    monkeypatch.setitem(sys.modules, "modal", mod)
    b = ModalBench()
    b.evaluate(req(holdout=[]))
    # mutation: always passing mode= breaks every metered job against an app deployed before
    # this change, which does not take the argument
    assert remotes[0] == (({"mod.rs": "x"},), {})
    assert all(len(a) == 2 for a in starmaps[0])
    b.evaluate(EvalRequest(challenge="knapsack", files={"mod.rs": "x"}, training=TR,
                           holdout=[], fuel=1, baseline_training=None, rule=BeatRule(),
                           mode="native", fuel_budgets_us={"t": 900}))
    assert remotes[1] == (({"mod.rs": "x"},), {"mode": "native"})
    assert all(len(a) == 3 and a[2] == "native" for a in starmaps[1])
    # mutation: the budget left off the task runs every native nonce unbounded
    assert {t["budget_us"] for a in starmaps[1] for t in a[1]} == {900}
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_talos_bench.py tests/test_bench.py -q`
Expected: FAIL (`TypeError: ... unexpected keyword argument 'mode'`).

- [ ] **Step 3: Implement**

`modal_app/talos_bench.py`:

```python
NATIVE_BINARY = "talos-native"


def content_hash(files: dict[str, str], mode: str = "metered") -> str:
    return inside.content_hash(files, MONOREPO_REF, DEV_IMAGE_TAG, mode)
```

In `_compile_impl(name, files, mode="metered")`: set `art_id = content_hash(files, mode)`. Take the cached-artifact filename from the mode, `artifact = NATIVE_BINARY if mode == "native" else "algo.so"`, and check `(dest / artifact).exists()`. Build with `inside.build_native(MONOREPO, name, inside.ALGO_NAME)` in native mode, else `inside.build`. Look up the artifacts with `inside.native_artifact_paths` or `inside.artifact_paths`. Copy the built file to `dest / artifact` and the PTX to `dest / "algo.ptx"`. The failure message names "runner" instead of ".so" in native mode.

In `_score_batch_impl(..., workers, mode="metered", pool_factory=None, clock=...)`: set `so = d / (NATIVE_BINARY if mode == "native" else "algo.so")` and keep the missing-file `FileNotFoundError`. Each task tuple gains `mode, t.get("budget_us")`.

`_mk_compile`: `def compile_fn(files: dict, mode: str = "metered") -> dict: return _compile_impl(n, files, mode)`.
`_mk_score_batch`: `def score_batch(artifact_id: str, tasks: list[dict], mode: str = "metered") -> dict: return _score_batch_impl(n, cid, artifact_id, tasks, workers, mode)`.

`talos/bench.py` `ModalBench`:
- `_compile(self, challenge, files, mode="metered")`: `kw = {} if mode == "metered" else {"mode": mode}` and `self._fn(name).remote(files, **kw)`.
- `_score(..., hyperparameters, mode="metered", budgets=None)`: when native, each task dict gains `"budget_us": (budgets or {}).get(ns.track)`. The starmap args are `(artifact_id, batch)` for metered and `(artifact_id, batch, mode)` for native.
- `evaluate`: pass `request.mode` and `request.fuel_budgets_us` through.

- [ ] **Step 4: Run to verify they pass, then the whole suite**

Run: `.venv/bin/python -m pytest tests/test_talos_bench.py tests/test_bench.py -q && .venv/bin/python -m pytest -q -m "not live"`
Expected: PASS, whole suite green.

- [ ] **Step 5: Commit**

```bash
git status --short
git add modal_app/talos_bench.py talos/bench.py tests/test_talos_bench.py tests/test_bench.py
git commit -m "modal: native compile and score; metered calls unchanged for an older deploy"
```

---

### Task 7: FakeBench modes and the calibration module

**Files:**
- Create: `talos/calibration.py`
- Modify: `talos/bench.py` (`FakeBench`)
- Test: `tests/test_calibration.py`, `tests/test_bench.py`

**Interfaces:**
- Consumes: `NonceResult.fuel_consumed/solve_us` (Task 1), `inside.content_hash` (Task 3), `baseline.effective_hyperparameters`, `state._atomic_write`.
- Produces (all in `talos/calibration.py`):
  - `MARGIN_START = 0.8`, `MARGIN_STEP = 0.1`, `MARGIN_FLOOR = 0.3`, `MISS_LIMIT = 3`, `BUDGET_FLOOR_US = 1_000_000` (all ESTIMATE)
  - `calibration_key(challenge: str, hardware_class: str, baseline_files: dict[str, str], hyperparameters: dict | None) -> str`
  - `track_ratios(metered: list[NonceResult], native: list[NonceResult]) -> dict[str, float]`
  - `new_record(ratios: dict[str, float]) -> dict`
  - `budgets_us(record: dict, fuel: int) -> dict[str, int]`
  - `record_miss(record: dict, track: str) -> bool` (True when the margin changed)
  - `note_demotion(record: dict, reason: str) -> None`
  - `load(cache_dir: Path, challenge: str, key: str) -> dict | None`
  - `save(cache_dir: Path, challenge: str, key: str, record: dict) -> None`
- Produces (`FakeBench`): new keyword args `metered_scores=None`, `metered_compile_ok=None`, `fuel_consumed: int | None = 1000`, `solve_us: int | None = 500`. A str returned by a scores callback is the error kind for that nonce. Metered rows carry `fuel_consumed`, and native rows carry `solve_us`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_calibration.py`:

```python
import pytest

from talos import calibration
from talos.types import NonceResult


def m(track, nonce, fuel):
    return NonceResult(track, nonce, True, 1, 1, None, fuel_consumed=fuel)


def n(track, nonce, us):
    return NonceResult(track, nonce, True, 1, 1, None, solve_us=us)


METERED = [m("a", 0, 1000), m("a", 1, 2000), m("a", 2, 4000)]
NATIVE = [n("a", 0, 10_000), n("a", 1, 30_000), n("a", 2, 400_000)]


def test_ratio_is_the_median_of_microseconds_per_fuel():
    # per nonce: 10.0, 15.0, 100.0 us per fuel. The median is 15.0; the mean would be 41.67
    # mutation: mean instead of median; ratio inverted (fuel per us: median 0.0667)
    assert calibration.track_ratios(METERED, NATIVE) == {"a": 15.0}


def test_budget_is_ratio_times_fuel_times_margin_in_microseconds():
    rec = calibration.new_record({"a": 15.0})
    # 15.0 us/fuel x 100_000 fuel x 0.8 = 1_200_000 us, worked by hand
    # mutation: margin dropped -> 1_500_000; ceil on a float product -> 1_200_001
    assert calibration.budgets_us(rec, 100_000) == {"a": 1_200_000}


def test_a_tiny_budget_is_raised_to_the_floor():
    rec = calibration.new_record({"a": 1.0})
    # 1.0 x 1000 x 0.8 = 800 us, under the 1 s floor
    # mutation: floor dropped leaves an 800 us budget that times out on timer noise alone
    assert calibration.budgets_us(rec, 1000) == {"a": 1_000_000}


def test_nonces_without_fuel_or_solve_time_are_skipped_and_a_track_with_none_left_is_absent():
    metered = [m("a", 0, 0), m("a", 1, None), m("b", 0, 1000)]
    native = [n("a", 0, 5000), n("a", 1, 5000), n("b", 0, None)]
    # mutation: dividing by a zero fuel_consumed raises ZeroDivisionError; keeping the track
    # with ratio 0 gives it the 1 s floor instead of the metered fallback
    assert calibration.track_ratios(metered, native) == {}


def test_a_nonce_pairs_only_with_its_own_track_and_nonce():
    metered = [m("a", 0, 1000), m("b", 0, 10)]
    native = [n("a", 0, 2000), n("b", 1, 99_999)]
    # mutation: pairing by position or by nonce alone puts b's solve time on a's fuel
    assert calibration.track_ratios(metered, native) == {"a": 2.0}


def test_three_misses_lower_the_margin_once_and_the_floor_holds():
    rec = calibration.new_record({"a": 1.0})
    # mutation: `>` instead of `>=` needs a fourth miss
    assert [calibration.record_miss(rec, "a") for _ in range(3)] == [False, False, True]
    assert rec["tracks"]["a"]["margin"] == pytest.approx(0.7)
    assert rec["tracks"]["a"]["misses"] == 0
    for _ in range(3 * 10):
        calibration.record_miss(rec, "a")
    # mutation: no floor drives the margin to zero or below, a budget every candidate fails
    assert rec["tracks"]["a"]["margin"] == pytest.approx(0.3)


def test_the_key_changes_with_hardware_baseline_and_hyperparameters():
    base = calibration.calibration_key("hypergraph", "c3-l40", {"mod.rs": "a"}, None)
    assert base == calibration.calibration_key("hypergraph", "c3-l40", {"mod.rs": "a"}, None)
    # mutation: dropping any one of these reuses an L40 budget on an A100, or after the
    # baseline or its hyperparameters changed
    assert base != calibration.calibration_key("hypergraph", "c3-a100", {"mod.rs": "a"}, None)
    assert base != calibration.calibration_key("hypergraph", "c3-l40", {"mod.rs": "b"}, None)
    assert base != calibration.calibration_key("hypergraph", "c3-l40", {"mod.rs": "a"},
                                               {"t": {"x": 1}})
    # a track mapped to None is the same run as no map (baseline.effective_hyperparameters)
    assert base == calibration.calibration_key("hypergraph", "c3-l40", {"mod.rs": "a"},
                                               {"t": None})


def test_the_key_depends_on_the_pins(monkeypatch):
    base = calibration.calibration_key("knapsack", "hw", {"mod.rs": "a"}, None)
    monkeypatch.setattr(calibration, "MONOREPO_REF", "other")
    assert calibration.calibration_key("knapsack", "hw", {"mod.rs": "a"}, None) != base


def test_save_then_load_round_trips(tmp_path):
    rec = calibration.new_record({"a": 15.0})
    calibration.save(tmp_path, "knapsack", "k1", rec)
    assert calibration.load(tmp_path, "knapsack", "k1") == rec
    assert calibration.load(tmp_path, "knapsack", "k2") is None


def test_load_treats_a_corrupt_record_as_missing(tmp_path):
    p = tmp_path / "knapsack" / "k1.json"
    p.parent.mkdir(parents=True)
    p.write_text('{"tracks": {"a": {"ratio": 1.0,')   # truncated mid-write
    assert calibration.load(tmp_path, "knapsack", "k1") is None
    p.write_text('{"tracks": {"a": {}}}')             # no ratio
    # mutation: a KeyError here tracebacks out of `talos run` instead of re-measuring
    assert calibration.load(tmp_path, "knapsack", "k1") is None


def test_note_demotion_counts_by_reason():
    rec = calibration.new_record({"a": 1.0})
    calibration.note_demotion(rec, "nondeterministic")
    calibration.note_demotion(rec, "nondeterministic")
    assert rec["demotions"] == {"nondeterministic": 2}
```

Append to `tests/test_bench.py`:

```python
def test_fake_bench_modes_carry_fuel_or_solve_time_and_separate_scores():
    fb = FakeBench(lambda ch, files, ns: [100] * ns.count,
                   metered_scores=lambda ch, files, ns: ["out_of_fuel"] + [100] * (ns.count - 1),
                   fuel_consumed=1234, solve_us=55)
    nat = fb.evaluate(EvalRequest(challenge="knapsack", files={"mod.rs": "x"}, training=TR,
                                  holdout=[], fuel=1, baseline_training=None, rule=BeatRule(),
                                  mode="native"))
    met = fb.evaluate(req(holdout=[]))
    assert [r.solve_us for r in nat.training] == [55, 55, 55]
    assert all(r.fuel_consumed is None for r in nat.training)
    assert [r.fuel_consumed for r in met.training] == [1234, 1234, 1234]
    # mutation: the metered callback ignored makes every validation pass in the loop tests
    assert met.training[0].error == "out_of_fuel" and not met.training[0].ok
    assert all(r.ok for r in nat.training)
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_calibration.py tests/test_bench.py -q`
Expected: FAIL (`ModuleNotFoundError: talos.calibration`, and `TypeError` on FakeBench's new keywords).

- [ ] **Step 3: Implement `talos/calibration.py`**

```python
"""Native research scoring has no fuel counter. This module turns the baseline's metered
fuel_consumed and its native solve time, measured on the same nonces, into a per-track
time budget that stands in for the fuel limit. One record per (challenge, hardware class,
monorepo pin, dev image tag, baseline code, hyperparameters), cached under
~/.talos/calibration/ and reused by every later job with the same key."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import median

from talos import inside
from talos.baseline import effective_hyperparameters
from talos.challenges import DEV_IMAGE_TAG, MONOREPO_REF
from talos.state import _atomic_write
from talos.types import NonceResult

# ESTIMATES with no measurement behind them (spec §4.4 and §6); tuned from recorded misses.
MARGIN_START = 0.8
MARGIN_STEP = 0.1
MARGIN_FLOOR = 0.3
MISS_LIMIT = 3
# ESTIMATE: a guard against timer noise on very fast nonces, not a research economy cap.
# runtime_floor_s (60 s) would bind on most hypergraph tracks (user decision 2026-09-29).
BUDGET_FLOOR_US = 1_000_000


def calibration_key(challenge: str, hardware_class: str, baseline_files: dict[str, str],
                    hyperparameters: dict | None) -> str:
    """The baseline enters by its metered artifact hash, which covers its files and both pins.
    The nonce sets do not enter: a ratio of time to fuel carries over between nonce draws,
    and the rand hash must never reach a file outside the job."""
    payload = {"challenge": challenge, "hw": hardware_class, "ref": MONOREPO_REF,
               "image": DEV_IMAGE_TAG,
               "baseline": inside.content_hash(baseline_files, MONOREPO_REF, DEV_IMAGE_TAG),
               "hp": effective_hyperparameters(hyperparameters)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]


def track_ratios(metered: list[NonceResult], native: list[NonceResult]) -> dict[str, float]:
    """Median microseconds of native solve per unit of fuel, per track, over nonces that have
    both numbers. Both were taken at the algorithm's last save_solution call. A zero or
    missing fuel_consumed is skipped: `if f` below is meant to drop 0 as well as None."""
    fuel = {(r.track, r.nonce): r.fuel_consumed for r in metered}
    per: dict[str, list[float]] = {}
    for r in native:
        f = fuel.get((r.track, r.nonce))
        if f and r.solve_us is not None:
            per.setdefault(r.track, []).append(r.solve_us / f)
    return {t: median(v) for t, v in per.items()}


def new_record(ratios: dict[str, float]) -> dict:
    return {"version": 1, "demotions": {},
            "tracks": {t: {"ratio": r, "margin": MARGIN_START, "misses": 0}
                       for t, r in ratios.items()}}


def budgets_us(record: dict, fuel: int) -> dict[str, int]:
    """round, not ceil: ratio x fuel x margin is a float product, and ceil turns
    1200000.0000000002 into 1200001."""
    return {t: max(BUDGET_FLOOR_US, int(round(v["ratio"] * fuel * v["margin"])))
            for t, v in record["tracks"].items()}


def record_miss(record: dict, track: str) -> bool:
    """A validation found a nonce of `track` out of fuel on the metered path after it passed
    natively. Every MISS_LIMIT misses the track's margin drops by MARGIN_STEP, never below
    MARGIN_FLOOR. True when the margin changed, so the caller recomputes its budgets."""
    t = record["tracks"][track]
    t["misses"] += 1
    if t["misses"] < MISS_LIMIT:
        return False
    t["misses"] = 0
    lowered = max(MARGIN_FLOOR, round(t["margin"] - MARGIN_STEP, 6))
    changed = lowered != t["margin"]
    t["margin"] = lowered
    return changed


def note_demotion(record: dict, reason: str) -> None:
    record.setdefault("demotions", {})
    record["demotions"][reason] = record["demotions"].get(reason, 0) + 1


def _path(cache_dir: Path, challenge: str, key: str) -> Path:
    return Path(cache_dir) / challenge / f"{key}.json"


def load(cache_dir: Path, challenge: str, key: str) -> dict | None:
    """None for a missing, unreadable or malformed record: it is then measured again."""
    p = _path(cache_dir, challenge, key)
    if not p.exists():
        return None
    try:
        rec = json.loads(p.read_text(encoding="utf-8"))
        for v in rec["tracks"].values():
            float(v["ratio"]), float(v["margin"]), int(v["misses"])
    except (ValueError, KeyError, TypeError, AttributeError):
        return None
    return rec


def save(cache_dir: Path, challenge: str, key: str, record: dict) -> None:
    p = _path(cache_dir, challenge, key)
    p.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(p, json.dumps(record, indent=1))
```

Check the import cycle: `talos.baseline` imports `talos.state` and `talos.inside`, and neither imports `calibration`. There is no cycle.

`record_miss` does `round(margin - step, 6)` so that 0.8 − 0.1 is stored as 0.7, not 0.7000000000000001. The test uses `pytest.approx` in any case.

`FakeBench` in `talos/bench.py`:

```python
    def __init__(self, scores, compile_ok=lambda files: True, usd_per_nonce: float = 0.01,
                 compile_output=lambda files: "ok",
                 metered_scores=None, metered_compile_ok=None,
                 fuel_consumed: int | None = 1000, solve_us: int | None = 500):
        ...
        # Native and metered scoring of the same files give the same qualities unless a test
        # says otherwise: metered_scores and metered_compile_ok stand in for the metered path
        # in validation tests. A str from a scores callback is that nonce's error kind.
        self._metered_scores = metered_scores
        self._metered_compile_ok = metered_compile_ok
        self._fuel_consumed = fuel_consumed
        self._solve_us = solve_us
```

In `evaluate`, pick `compile_ok = self._metered_compile_ok if request.mode == "metered" and self._metered_compile_ok else self._compile_ok`, and pick the scores callback the same way. Pass `request.mode` into `_score`, and build each row as:

```python
                extra = ({"fuel_consumed": self._fuel_consumed} if mode == "metered"
                         else {"solve_us": self._solve_us})
                if isinstance(q, str):
                    # an out_of_fuel row stopped at its limit, as run_nonce reports it
                    out.append(NonceResult(ns.track, n, False, None, self._runtime_ms, q,
                                           limit_hit=q == "out_of_fuel", **extra))
                elif q is None:
                    out.append(NonceResult(ns.track, n, False, None, self._runtime_ms,
                                           "no_solution", **extra))
                else:
                    out.append(NonceResult(ns.track, n, True, q, self._runtime_ms, None,
                                           **extra))
```

- [ ] **Step 4: Run to verify they pass, then the whole suite**

Run: `.venv/bin/python -m pytest tests/test_calibration.py tests/test_bench.py -q && .venv/bin/python -m pytest -q -m "not live"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git status --short
git add talos/calibration.py talos/bench.py tests/test_calibration.py tests/test_bench.py
git commit -m "calibration: per-track native budgets from the baseline's metered fuel; FakeBench modes"
```

---

### Task 8: The validation rule

**Files:**
- Modify: `talos/scoring.py`
- Test: `tests/test_scoring.py`

**Interfaces:**
- Consumes: `bundle_delta`, `beats`, `BeatRule` (existing).
- Produces: `VALIDATION_REASONS: tuple[str, ...]`, `fuel_proxy_misses(metered: list[NonceResult], native: list[NonceResult]) -> list[NonceResult]` (the metered rows that hit the fuel limit where the same nonce's native run did not hit its budget), `quality_mismatches(metered: list[NonceResult], native: list[NonceResult]) -> list[dict]` (each `{"track", "nonce", "native", "metered"}`; a nonce where either run hit its limit is never a mismatch) and `validation_failure(compiled: bool, metered: list[NonceResult], native: list[NonceResult], baseline: list[NonceResult], best_delta: float, rule: BeatRule) -> tuple[str | None, BundleDelta | None]`. The rule returns `(None, delta)` when validation passes, and `(reason, delta or None)` when it fails.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_scoring.py`:

```python
from talos.scoring import fuel_proxy_misses, quality_mismatches, validation_failure


# The file already defines BASE (two tracks, t1/t2) and RULE at module level, and the existing
# tests read them at call time. These names must not collide: a second `BASE = ...` appended
# below would silently replace the first and break every existing bundle_delta/beats test.
def vrows(qs, track="t", error=None, limit=()):
    return [NonceResult(track, i, q is not None, q, 1, None if q is not None else error,
                        limit_hit=i in limit)
            for i, q in enumerate(qs)]


VBASE = vrows([100, 100, 100, 100])
VRULE = BeatRule()  # margin 0.005, track tolerance 0.0, error ceiling 0.05


def test_validation_passes_a_stepping_stone_short_of_the_margin():
    # mean 100.25 vs 100: +0.25%, short of the 0.5% margin, above a best of 0.0
    # mutation: requiring beats() here (spec §5.3 as written) demotes every stepping stone
    step = vrows([100, 100, 100, 101])
    reason, d = validation_failure(True, step, step, VBASE, 0.0, VRULE)
    assert reason is None and d.mean_rel_delta == pytest.approx(0.0025)


def test_each_failure_reason_in_order():
    good = vrows([110, 110, 110, 110])
    assert validation_failure(False, [], good, VBASE, 0.0, VRULE)[0] == \
        "native_metered_build_mismatch"
    oof = vrows([None, 110, 110, 110], error="out_of_fuel", limit=(0,))
    # mutation: out_of_fuel read as an ordinary error lets a fuel-proxy miss through as long
    # as the error rate stays under the ceiling; checked before the quality comparison
    assert validation_failure(True, oof, good, VBASE, 0.0, VRULE)[0] == "fuel_proxy_miss"
    drift = vrows([110, 110, 110, 111])
    # mutation: comparing means instead of nonce by nonce hides a +1/-1 pair of drifts
    assert validation_failure(True, drift, good, VBASE, 0.0, VRULE)[0] == "nondeterministic"
    pair = vrows([111, 109, 110, 110])
    assert validation_failure(True, pair, good, VBASE, 0.0, VRULE)[0] == "nondeterministic"
    worse = vrows([100, 100, 100, 100])
    assert validation_failure(True, worse, worse, VBASE, 0.0, VRULE)[0] == "not_improved"
    timeouts = vrows([None, 110, 110, 110], error="timeout")
    # 1 error in 4 nonces is a rate of 0.25, above the 0.05 ceiling (by hand)
    assert validation_failure(True, timeouts, good, VBASE, 0.0, VRULE)[0] == "error_ceiling"
    short = vrows([110, 110])
    assert validation_failure(True, short, short, VBASE, 0.0, VRULE)[0] == "unscoreable"


def test_a_metered_run_out_of_fuel_with_a_verified_solution_is_a_proxy_miss():
    # tig-runtime exited 87 on nonce 0 after saving a solution that verified: ok, error None
    metered = vrows([108, 110, 110, 110], limit=(0,))
    native = vrows([110, 110, 110, 110])
    # mutation: detecting misses by `error == "out_of_fuel"` alone calls this
    # "nondeterministic" (108 vs 110), and the margin never adapts
    assert validation_failure(True, metered, native, VBASE, 0.0, VRULE)[0] == "fuel_proxy_miss"
    assert [(r.track, r.nonce) for r in fuel_proxy_misses(metered, native)] == [("t", 0)]


def test_a_nonce_where_both_runs_hit_their_limit_is_neither_a_miss_nor_a_mismatch():
    # the native budget stopped nonce 0 early (margin < 1 by design) and the metered run used
    # its whole fuel: consistent, and the two qualities are expected to differ
    metered = vrows([112, 110, 110, 110], limit=(0,))
    native = vrows([109, 110, 110, 110], limit=(0,))
    # mutation: comparing qualities on a limit-truncated nonce demotes a valid candidate
    assert validation_failure(True, metered, native, VBASE, 0.0, VRULE)[0] is None
    assert quality_mismatches(metered, native) == []


def test_a_native_budget_exit_is_not_a_quality_mismatch():
    # native stopped at its budget; metered finished inside its fuel with a better solution
    metered = vrows([112, 110, 110, 110])
    native = vrows([109, 110, 110, 110], limit=(0,))
    # mutation: ignoring native limit_hit demotes every candidate that uses 80-100% of its fuel
    assert validation_failure(True, metered, native, VBASE, 0.0, VRULE)[0] is None


def test_a_nonce_ok_on_only_one_path_is_not_a_quality_mismatch():
    native = vrows([110, None, 110, 110], error="no_solution")
    metered = vrows([110, 110, 110, 110])
    # mutation: comparing None to 110 calls a native error "nondeterministic"
    reason, _ = validation_failure(True, metered, native, VBASE, 0.0, VRULE)
    assert reason is None


def test_a_stepping_stone_must_also_beat_the_current_best():
    step = vrows([100, 100, 100, 101])  # +0.25%: not over a best of +0.3%, and no beats()
    # mutation: comparing against 0.0 instead of the current best keeps replacing the best
    # with worse stepping stones
    assert validation_failure(True, step, step, VBASE, 0.003, VRULE)[0] == "not_improved"
    # the loop's rule is mean-over-best OR beats: +2% beats the baseline and passes even
    # under a +3% best (tests/test_loop.py::test_win_with_lower_delta_than_a_false_positive_best)
    big = vrows([102, 102, 102, 102])
    assert validation_failure(True, big, big, VBASE, 0.03, VRULE)[0] is None


def test_quality_mismatches_lists_each_differing_nonce():
    native = vrows([110, 110, None, 110], error="no_solution")
    metered = vrows([110, 111, 105, 109])
    # nonce 2 is ok on one path only, so it is not a mismatch
    assert quality_mismatches(metered, native) == [
        {"track": "t", "nonce": 1, "native": 110, "metered": 111},
        {"track": "t", "nonce": 3, "native": 110, "metered": 109}]
```

(Use `pytest`, `NonceResult` and `BeatRule` from the file's existing imports. Do not reuse the names `BASE`, `RULE` or `R`, which the file already defines.)

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_scoring.py -q`
Expected: FAIL with `ImportError: cannot import name 'validation_failure'`.

- [ ] **Step 3: Implement in `talos/scoring.py`**

```python
VALIDATION_REASONS = ("native_metered_build_mismatch", "fuel_proxy_miss", "nondeterministic",
                      "unscoreable", "error_ceiling", "not_improved")


def fuel_proxy_misses(metered: list[NonceResult], native: list[NonceResult]) -> list[NonceResult]:
    """Metered rows whose fuel ran out (with or without a saved solution) where the same
    nonce's native run did not reach its budget: the budget stood in for more fuel than TIG
    gives. A nonce that hit its limit on both paths is consistent, not a miss."""
    nat = {(r.track, r.nonce): r for r in native}
    out = []
    for r in metered:
        if r.error == "out_of_fuel" or r.limit_hit:
            o = nat.get((r.track, r.nonce))
            if o is None or not o.limit_hit:
                out.append(r)
    return out


def quality_mismatches(metered: list[NonceResult], native: list[NonceResult]) -> list[dict]:
    """Nonces both paths solved, neither cut off at its limit, with different qualities, in
    metered order. Scoring is deterministic per instance (MEASURED, spec §1), so any entry
    means the two paths ran different code or the algorithm depends on something other than
    its seed. A run stopped at its fuel or budget saved an earlier solution, so its quality
    is expected to differ and says nothing about determinism."""
    nat = {(r.track, r.nonce): r for r in native}
    out = []
    for r in metered:
        o = nat.get((r.track, r.nonce))
        if (o is not None and r.ok and o.ok and not r.limit_hit and not o.limit_hit
                and r.quality != o.quality):
            out.append({"track": r.track, "nonce": r.nonce, "native": o.quality,
                        "metered": r.quality})
    return out


def validation_failure(compiled: bool, metered: list[NonceResult], native: list[NonceResult],
                       baseline: list[NonceResult], best_delta: float,
                       rule: BeatRule) -> tuple[str | None, BundleDelta | None]:
    """Whether a candidate that improved natively still stands on TIG's metered runtime, and
    why not. It must build metered, run out of fuel on no nonce whose native run had budget
    left, give the native quality on every nonce both paths solved within their limits, stay
    under the error ceiling, and still count as improved
    by the loop's own rule (mean over the current best, or beats). beats() alone is not
    required: that would demote every stepping stone (user decision 2026-09-29)."""
    if not compiled:
        return "native_metered_build_mismatch", None
    if fuel_proxy_misses(metered, native):
        return "fuel_proxy_miss", None
    if quality_mismatches(metered, native):
        return "nondeterministic", None
    try:
        d = bundle_delta(baseline, metered)
    except ScoringError:
        return "unscoreable", None
    if d.error_rate > rule.error_ceiling:
        return "error_ceiling", d
    if not (d.mean_rel_delta > best_delta or beats(baseline, metered, rule)):
        return "not_improved", d
    return None, d
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_scoring.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git status --short
git add talos/scoring.py tests/test_scoring.py
git commit -m "scoring: validation_failure, the metered check a native best must pass"
```

---

### Task 9: The loop calibrates, scores natively and validates

**Files:**
- Modify: `talos/loop.py`
- Test: `tests/test_loop.py`

**Interfaces:**
- Consumes: Tasks 1, 7 and 8.
- Produces:
  - `Loop.calibrate(cache_dir: Path, hardware_class: str) -> None`
  - `Loop._scoring() -> str` ("native" only when `state.scoring == "native"`)
  - `Loop._request(files, baseline_training, scoring="metered", validation=False)`
  - `Loop._bench_evaluate(files, scoring="metered", validation=False)`
  - `Loop._validate_and_finish(n, record, cand, base_tr) -> None`
  - Timeline events `calibrated`, `calibration_fallback`, `validating`, `validated`, `demoted`
  - The iteration outcome `failed:validation`
  - A `pending_job["native_training"]` key while a validation is in flight.

- [ ] **Step 1: Write the failing tests**

In `tests/test_loop.py`, give `spec()` a `scoring="metered"` and a `fuel=1` parameter and pass both to `JobSpec`. Give `make()` the keyword parameters `scoring="metered"`, `fuel=1`, `budgets=None` and `bench=None`. When `bench` is given, use it instead of `FakeBench(scores)`. When `scoring == "native"`, set `st.scoring = "native"` and `st.fuel_budgets_us = budgets or {"t": 1_000_000}`. Then append:

```python
def native(tmp_path, script, **kw):
    kw.setdefault("scoring", "native")
    return make(tmp_path, script, **kw)


def test_native_research_validates_a_new_best_on_the_metered_path(tmp_path):
    loop, fp, fb, store = native(tmp_path, [hyp("bump k"), edit(5)])
    st = loop.run()
    research, validation = fb.calls[0], fb.calls[1]
    assert research.mode == "native" and research.fuel_budgets_us == {"t": 1_000_000}
    # held-out is decided on the metered run, never on native results
    assert research.holdout == [] and research.baseline_training is None
    assert validation.mode == "metered" and validation.holdout == HO
    assert validation.baseline_training is not None and validation.prior_functions is None
    # mutation: best set from the native rows (they carry solve_us, not fuel_consumed)
    assert st.best.iteration == 1 and st.best.training[0].fuel_consumed == 1000
    assert st.validations == [{"iteration": 1, "outcome": "validated", "reason": None}]
    assert st.status == "won" and st.confirmed == [1]
    # mutation: confirming on the native call's (empty) held-out rows instead of the metered
    # ones; FakeBench counts a held-out run for the native call too, so count rows, not runs
    assert len(st.best.holdout) == 4 and all(r.fuel_consumed == 1000 for r in st.best.holdout)


def test_a_native_score_carries_no_runtime_ratio_against_the_metered_baseline(tmp_path):
    loop, fp, fb, store = native(tmp_path, [hyp("lower k"), edit(0)],
                                 budget=Budget(usd=None, hours=None, iterations=1,
                                               compute_usd=None))
    st = loop.run()
    # mutation: a native-over-metered ratio reaches the recall prompt as "runtime 0.4x
    # baseline" for code that is no faster at all
    assert "runtime_ratio" not in st.hypotheses[0]
    scored = [json.loads(x) for x in
              (store.run_dir / "timeline.jsonl").read_text().splitlines()
              if json.loads(x)["kind"] == "scored"]
    assert scored[0]["runtime_ratio"] is None


def test_a_native_loss_is_not_validated(tmp_path):
    # k = 0 scores 99 against the baseline's 100: worse, so nothing to validate
    loop, fp, fb, store = native(tmp_path, [hyp("lower k"), edit(0)],
                                 budget=Budget(usd=None, hours=None, iterations=1,
                                               compute_usd=None))
    loop.run()
    # mutation: validating every scored candidate doubles the compute of every iteration
    assert [c.mode for c in fb.calls] == ["native"]


def test_a_nondeterministic_candidate_is_demoted_and_research_stays_on_the_old_code(tmp_path):
    def metered(ch, files, ns):
        q = quality_from_files(ch, files, ns)
        return [x + 1 for x in q] if "let k = 5;" in files["mod.rs"] else q
    fb = FakeBench(quality_from_files, metered_scores=metered)
    loop, fp, _, store = native(tmp_path, [hyp("a"), edit(5), hyp("b"), edit(3)], bench=fb,
                                budget=Budget(usd=None, hours=None, iterations=2,
                                              compute_usd=None))
    st = loop.run()
    # spec §6: both qualities recorded. k = 5 is 104 natively and 105 metered on every nonce
    assert st.validations[0] == {
        "iteration": 1, "outcome": "demoted", "reason": "nondeterministic",
        "mismatches": [{"track": "t", "nonce": n, "native": 104, "metered": 105}
                       for n in range(4)]}
    assert st.hypotheses[0]["outcome"] == "failed:validation"
    # mutation: building on the demoted candidate; edit(3) searches for "let k = 1;", which
    # only the baseline still has, so iteration 2 would fail to apply its edit
    assert st.best.iteration == 2 and "let k = 3;" in st.best.files["mod.rs"]
    # mutation: a demotion that leaves best pointing at the rejected candidate
    assert not (store.run_dir / "best" / "mod.rs").read_text().count("let k = 5;")


def test_a_metered_build_failure_demotes_with_its_reason(tmp_path):
    fb = FakeBench(quality_from_files, metered_compile_ok=lambda files: False)
    loop, fp, _, store = native(tmp_path, [hyp("a"), edit(5)], bench=fb,
                                budget=Budget(usd=None, hours=None, iterations=1,
                                              compute_usd=None))
    st = loop.run()
    assert st.validations[0]["reason"] == "native_metered_build_mismatch" and st.best is None


def _calibrated(loop, tmp_path, ratio=1.0):
    """A fresh native job's first calibrate, from a stored record. make() pre-sets native
    budgets for the loop tests; a fresh job has none, so they are cleared first."""
    from talos import calibration
    loop.state.scoring, loop.state.fuel_budgets_us = None, None
    key = calibration.calibration_key("knapsack", "hw", loop.state.baseline.files,
                                      loop.spec.hyperparameters)
    calibration.save(tmp_path / "cal", "knapsack", key, calibration.new_record({"t": ratio}))
    loop.calibrate(tmp_path / "cal", "hw")
    return key


def test_resume_keeps_the_jobs_budgets_and_its_pending_job(tmp_path):
    from talos import calibration
    loop, fp, fb, store = native(tmp_path, [], fuel=10_000_000, budgets={"t": 1_234_567})
    key = calibration.calibration_key("knapsack", "hw", loop.state.baseline.files, None)
    # another job has since tightened the shared record: 0.25 x 1e7 x 0.8 = 2_000_000
    calibration.save(tmp_path / "cal", "knapsack", key, calibration.new_record({"t": 0.25}))
    pend = {"purpose": 1, "hypothesis": {}, "files": {}, "job_id": "job_1",
            "request_hash": "abc"}
    loop.state.pending_job = dict(pend)
    loop.calibrate(tmp_path / "cal", "hw")
    # mutation: re-deriving budgets from the record changes the in-flight request's hash, and
    # the resume submits a second job while the first still bills
    assert loop.state.fuel_budgets_us == {"t": 1_234_567}
    # mutation: re-measuring on resume overwrites pending_job and loses job_1's id
    assert loop.state.pending_job == pend and fb.calls == []
    # demotions still reach the record
    assert loop._calibration == (tmp_path / "cal", key)


def test_three_fuel_misses_lower_the_margin_and_the_budgets(tmp_path):
    from talos import calibration
    def metered(ch, files, ns):
        return ["out_of_fuel"] + quality_from_files(ch, files, ns)[1:]
    fb = FakeBench(quality_from_files, metered_scores=metered)
    script = [hyp("a"), edit(5), hyp("b"), edit(6), hyp("c"), edit(7)]
    loop, fp, _, store = native(tmp_path, script, bench=fb, fuel=10_000_000,
                                budget=Budget(usd=None, hours=None, iterations=3,
                                              compute_usd=None))
    key = _calibrated(loop, tmp_path)
    # 1.0 us/fuel x 10_000_000 fuel x 0.8 = 8_000_000 us, by hand
    assert loop.state.fuel_budgets_us == {"t": 8_000_000}
    st = loop.run()
    assert [v["reason"] for v in st.validations] == ["fuel_proxy_miss"] * 3
    rec = calibration.load(tmp_path / "cal", "knapsack", key)
    assert rec["tracks"]["t"]["margin"] == pytest.approx(0.7)
    assert rec["demotions"] == {"fuel_proxy_miss": 3}
    # mutation: lowering the stored margin without recomputing the job's budgets
    assert st.fuel_budgets_us == {"t": 7_000_000}
    assert fb.calls[-2].fuel_budgets_us == {"t": 8_000_000}  # iteration 3's research call


def test_resume_mid_validation_goes_straight_to_the_metered_run(tmp_path):
    loop, fp, fb, store = native(tmp_path, [])  # no LLM reply scripted: none may be asked for
    files = {"mod.rs": "fn solve() { let k = 5; }\n"}
    native_rows = [NonceResult("t", n, True, 104, 1, None, solve_us=500) for n in range(4)]
    loop.state.pending_job = {"purpose": 1, "hypothesis": {"title": "h", "description": "d",
                                                           "strategy_tag": "local_search"},
                              "files": files,
                              "native_training": [r.to_dict() for r in native_rows]}
    loop._save()
    st = loop.run()
    # mutation: re-entering at _score_candidate scores natively again before validating
    assert fb.calls[0].mode == "metered"
    assert st.best is not None and st.best.iteration == 1


def test_calibrate_rescores_metered_fuel_when_the_cached_baseline_has_none(tmp_path):
    from talos import calibration
    loop, fp, fb, store = native(tmp_path, [], fuel=10_000_000)
    loop.state.scoring = None  # as a fresh native job before calibration
    loop.calibrate(tmp_path / "cal", "hw")
    # the test baseline's rows have no fuel_consumed, like every pre-change cached baseline
    # mutation: falling back to metered research instead of measuring the fuel
    assert [c.mode for c in fb.calls] == ["metered", "native"]
    assert all(c.holdout == [] for c in fb.calls)
    # FakeBench: fuel 1000, solve 500 us -> 0.5 us/fuel; 0.5 x 10_000_000 x 0.8 = 4_000_000
    assert loop.state.scoring == "native" and loop.state.fuel_budgets_us == {"t": 4_000_000}
    key = calibration.calibration_key("knapsack", "hw", loop.state.baseline.files, None)
    assert calibration.load(tmp_path / "cal", "knapsack", key) is not None
    assert loop.state.pending_job is None


def test_calibrate_uses_the_baselines_own_fuel_when_it_has_it(tmp_path):
    loop, fp, fb, store = native(tmp_path, [], fuel=10_000_000)
    loop.state.scoring = None
    loop.state.baseline.training = [replace(r, fuel_consumed=2000)
                                    for r in loop.state.baseline.training]
    loop.calibrate(tmp_path / "cal", "hw")
    assert [c.mode for c in fb.calls] == ["native"]
    # 500 us / 2000 fuel = 0.25; 0.25 x 10_000_000 x 0.8 = 2_000_000
    assert loop.state.fuel_budgets_us == {"t": 2_000_000}


def test_calibrate_reuses_a_stored_record_without_a_bench_call(tmp_path):
    # _calibrated clears make()'s preset budgets, so this is a fresh job reading the record
    loop, fp, fb, store = native(tmp_path, [], fuel=10_000_000)
    _calibrated(loop, tmp_path, ratio=0.25)
    assert fb.calls == [] and loop.state.fuel_budgets_us == {"t": 2_000_000}


def test_calibration_without_solve_times_falls_back_to_metered_research(tmp_path):
    fb = FakeBench(quality_from_files, solve_us=None)
    loop, fp, _, store = native(tmp_path, [hyp("a"), edit(5)], bench=fb,
                                budget=Budget(usd=None, hours=None, iterations=1,
                                              compute_usd=None))
    loop.state.scoring = None
    loop.calibrate(tmp_path / "cal", "hw")
    assert loop.state.scoring == "metered"
    events = [json.loads(x)["kind"] for x in
              (store.run_dir / "timeline.jsonl").read_text().splitlines()]
    assert "calibration_fallback" in events
    n_before = len(fb.calls)
    loop.run()
    # mutation: a native run with no budget
    assert all(c.mode == "metered" for c in fb.calls[n_before:])


def test_a_metered_job_never_calibrates(tmp_path):
    loop, fp, fb, store = make(tmp_path, [])
    loop.calibrate(tmp_path / "cal", "hw")
    assert fb.calls == [] and loop.state.scoring is None and loop._scoring() == "metered"


def test_calibration_is_budget_checked_before_its_first_call(tmp_path):
    loop, fp, fb, store = native(tmp_path, [],
                                 budget=Budget(usd=None, hours=None, iterations=5,
                                               compute_usd=0.0))
    loop.state.scoring = None
    # mutation: `if budget:` lets a zero compute cap through (AGENTS.md invariant 5)
    with pytest.raises(BudgetExhausted):
        loop.calibrate(tmp_path / "cal", "hw")
    assert fb.calls == []


def test_a_focused_native_job_needs_a_budget_only_for_its_track(tmp_path):
    from talos import calibration
    loop, fp, fb, store = native(tmp_path, [], track="t", two_tracks=True)
    loop.state.scoring, loop.state.fuel_budgets_us = None, None
    key = calibration.calibration_key("knapsack", "hw", loop.state.baseline.files, None)
    calibration.save(tmp_path / "cal", "knapsack", key, calibration.new_record({"t": 1.0}))
    loop.calibrate(tmp_path / "cal", "hw")
    # the guard track u is scored only in the metered validation, so it needs no budget
    # mutation: requiring every track's ratio sends a focused job to the metered fallback
    assert loop.state.scoring == "native"
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_loop.py -q`
Expected: the new tests FAIL (`AttributeError: 'Loop' object has no attribute 'calibrate'`, plus mode assertions). The existing tests PASS.

- [ ] **Step 3: Implement in `talos/loop.py`**

Imports: `from pathlib import Path`, `from talos import calibration`, `from talos.scoring import fuel_proxy_misses, quality_mismatches, validation_failure`, `from talos.state import Candidate, ...` (already imported), `from talos.types import Completion, NonceResult`.

In `Loop.__init__`, add `self._calibration: tuple[Path, str] | None = None`, which holds the (cache dir, key) of the record in use.

Add after `_timeouts`:

```python
    def _scoring(self) -> str:
        return "native" if self.state.scoring == "native" else "metered"
```

Replace `_request` and `_bench_evaluate`:

```python
    def _request(self, files: dict[str, str], baseline_training, scoring: str = "metered",
                 validation: bool = False) -> EvalRequest:
        training, holdout = self._focus()
        base = select(baseline_training, training) if baseline_training is not None else None
        prior = ({name: defined_functions(text) for name, text in self._current_files().items()}
                 if not validation else None)  # the native build already passed the check
        req = EvalRequest(challenge=self.spec.challenge, files=files, training=training,
                          holdout=holdout, fuel=self.spec.fuel, baseline_training=base,
                          rule=self.rule, prior_functions=prior,
                          timeouts=self._timeouts(baseline_training),
                          hyperparameters=self.spec.hyperparameters)
        if scoring == "native":
            # Held-out is decided on the metered validation run, never on native results.
            req.holdout, req.baseline_training = [], None
            req.mode, req.fuel_budgets_us = "native", self.state.fuel_budgets_us
        return req

    def _bench_evaluate(self, files: dict[str, str], scoring: str = "metered",
                        validation: bool = False) -> EvalResult:
        """pending_job carries the files BEFORE the call: a C3 job outlives this process, and a
        resume must be able to rebuild the exact request and reattach to it."""
        self._check_budget()
        self.state.pending_job = {**(self.state.pending_job or {}), "files": files}
        self._save()
        mark = self.bench.cost_mark()
        try:
            return self.bench.evaluate(self._request(files, self.state.baseline.training,
                                                     scoring, validation))
        finally:
            self.state.spend.compute_usd += self.bench.cost_usd_since(mark)
            self._save()
```

Add a calibration section after `measure_baseline`:

```python
    # ── calibration (native scoring) ──────────────────────────────────

    def calibrate(self, cache_dir, hardware_class: str) -> None:
        """Native scoring only. Finds this job's per-track budgets: a stored record for its key,
        or one measured now from the baseline on the training nonces in both modes. A job the
        record cannot give every research track a budget runs metered for good, and says so.
        Called on every start and resume, after the baseline; a stored decision is kept."""
        if self.spec.scoring != "native" or self.state.scoring == "metered":
            return
        key = calibration.calibration_key(self.spec.challenge, hardware_class,
                                          self.state.baseline.files, self.spec.hyperparameters)
        if self.state.scoring == "native" and self.state.fuel_budgets_us is not None:
            # A resume keeps the job's own budgets. Re-deriving them from the record (which
            # another job may have tightened since) changes an in-flight native request, so the
            # C3 request_hash no longer matches and the still-billing job is orphaned; and a
            # re-measure would overwrite pending_job, losing that job's id altogether.
            self._calibration = (Path(cache_dir), key)
            return
        rec = calibration.load(cache_dir, self.spec.challenge, key)
        if rec is None:
            rec = self._measure_calibration()
            if rec is not None:
                calibration.save(cache_dir, self.spec.challenge, key, rec)
        needed = {s.track for s in self._focus()[0]}
        missing = sorted(needed - set((rec or {}).get("tracks", {})))
        if rec is None or missing:
            self.state.scoring, self.state.fuel_budgets_us = "metered", None
            self._save()
            self._event("calibration_fallback",
                        reason=f"no native budget for track(s) {missing or sorted(needed)}; "
                               f"scoring this job metered")
            return
        self._calibration = (Path(cache_dir), key)
        self.state.scoring = "native"
        self.state.fuel_budgets_us = calibration.budgets_us(rec, self.spec.fuel)
        self._save()
        self._event("calibrated", budgets_us=self.state.fuel_budgets_us)

    def _measure_calibration(self) -> dict | None:
        """The baseline on the training nonces: metered only if its stored rows carry no fuel
        (every baseline cached before fuel was recorded), then native with no budget. Both go
        through the budgeted bench. None when no track yields a ratio."""
        bench = _BudgetedBench(self)
        pend = self.state.pending_job
        self.state.pending_job = (pend if pend and pend.get("purpose") == "calibration"
                                  else {"purpose": "calibration"})
        self._save()

        def score(mode: str):
            return bench.evaluate(EvalRequest(
                challenge=self.spec.challenge, files=self.state.baseline.files,
                training=self.spec.training, holdout=[], fuel=self.spec.fuel,
                baseline_training=None, rule=self.rule,
                hyperparameters=self.spec.hyperparameters, mode=mode))

        metered = self.state.baseline.training
        if all(r.fuel_consumed is None for r in metered):
            res = score("metered")
            metered = res.training if res.compile.ok else []
        res = score("native")
        self.state.pending_job = None
        self._save()
        if not res.compile.ok:
            self._event("calibration_fallback", reason="the baseline failed the native build",
                        output=res.compile.output[-2000:])
            return None
        ratios = calibration.track_ratios(metered, res.training)
        return calibration.new_record(ratios) if ratios else None
```

In `_score_candidate`, change both `self._bench_evaluate(files)` calls to `self._bench_evaluate(files, self._scoring())`.

In the same method, compute `runtime_ratio` only when the research scoring is metered. `self.state.baseline.training` holds metered runtimes, and a native candidate runs 1.6–2.9× faster than its metered self (MEASURED in the probe, spec §1). A native-over-metered ratio would tell the LLM, through the recall block (`talos/prompts.py`, `runtime Nx baseline`), that an unchanged candidate is about 0.4× the baseline's runtime. Replace the `record.update(...)` and `self._event("scored", ...)` lines with:

```python
        record.update(mean_rel_delta=delta.mean_rel_delta, worst_track=worst.track,
                      worst_rel_delta=delta.worst_rel_delta)
        if self._scoring() == "metered":
            # Native runtimes are not comparable with the baseline's metered ones; the
            # validated record gets the metered ratio instead (_validate_and_finish).
            record["runtime_ratio"] = runtime_ratio(base_tr, results)
        self._event("scored", mean_rel_delta=delta.mean_rel_delta,
                    worst_rel_delta=delta.worst_rel_delta, error_rate=delta.error_rate,
                    runtime_ratio=record.get("runtime_ratio"))
```

`talos/cli.py::_event_line` already prints no runtime when `runtime_ratio` is None.

Replace the block from `wins = beats(...)` to the end of the method with:

```python
        wins = beats(base_tr, results, self.rule)
        improved = delta.mean_rel_delta > self._best_delta() or wins
        if improved and self._scoring() == "native":
            self._validate_and_finish(n, record, cand, base_tr)
            return
        if improved:
            self.state.best = cand
            self._write_files(self.store.run_dir / "best", cand.files)
            record["outcome"] = "improved"
        else:
            record["outcome"] = "failed:score"
        # The record and the iteration bump are persisted BEFORE the confirmation is read out of
        # the result, so a kill between the two never loses the iteration.
        self._finish_iteration(n, record, improved=improved)
        if wins:
            self._confirm(cand, res.holdout, res.holdout_reason)
```

Add `_validate_and_finish` after `_dead_code`:

```python
    def _validate_and_finish(self, n: int, record: dict, cand: Candidate, base_tr) -> None:
        """A native candidate that would become the best is scored again on TIG's metered
        runtime, on the job's own training and held-out nonces, before it counts. It becomes
        best only on a pass; held-out confirmation then runs on the metered results. On a fail
        research stays on the previous best. The native rows go into pending_job first, so a
        resume re-enters here rather than scoring natively again."""
        self.state.pending_job = {**(self.state.pending_job or {}), "files": cand.files,
                                  "native_training": [r.to_dict() for r in cand.training]}
        self._save()
        self._event("validating")
        res = self._bench_evaluate(cand.files, "metered", validation=True)
        reason, d = validation_failure(res.compile.ok, res.training, cand.training, base_tr,
                                       self._best_delta(), self.rule)
        entry = {"iteration": n, "outcome": "demoted" if reason else "validated",
                 "reason": reason}
        if reason == "nondeterministic":
            entry["mismatches"] = quality_mismatches(res.training, cand.training)
        self.state.validations.append(entry)
        if reason:
            self._note_demotion(reason, res.training, cand.training)
            record.update(outcome="failed:validation", error=reason)
            self._event("demoted", reason=reason)
            self._finish_iteration(n, record, improved=False)
            return
        best = Candidate(iteration=n, files=cand.files, artifact_id=res.compile.artifact_id,
                         training=res.training, delta=d.to_dict(), hypothesis=cand.hypothesis)
        self.state.best = best
        self._write_files(self.store.run_dir / "best", best.files)
        record.update(outcome="improved", mean_rel_delta=d.mean_rel_delta,
                      worst_rel_delta=d.worst_rel_delta,
                      runtime_ratio=runtime_ratio(base_tr, res.training))
        self._event("validated", mean_rel_delta=d.mean_rel_delta)
        self._finish_iteration(n, record, improved=True)
        if beats(base_tr, res.training, self.rule):
            self._confirm(best, res.holdout, res.holdout_reason)

    def _note_demotion(self, reason: str, metered, native) -> None:
        """Counts the demotion in the calibration record; a fuel-proxy miss also counts against
        each track with a miss (talos/scoring.py::fuel_proxy_misses), and a lowered margin
        tightens this job's budgets."""
        if self._calibration is None:
            return
        cache_dir, key = self._calibration
        rec = calibration.load(cache_dir, self.spec.challenge, key)
        if rec is None:
            return
        calibration.note_demotion(rec, reason)
        changed = False
        if reason == "fuel_proxy_miss":
            for track in sorted({r.track for r in fuel_proxy_misses(metered, native)}):
                if track in rec["tracks"]:
                    changed = calibration.record_miss(rec, track) or changed
        calibration.save(cache_dir, self.spec.challenge, key, rec)
        if changed:
            self.state.fuel_budgets_us = calibration.budgets_us(rec, self.spec.fuel)
            self._save()
```

In `_resume_pending`, before the `_score_candidate` call:

```python
        if pending.get("native_training"):
            files = pending["files"]
            native = [NonceResult.from_dict(r) for r in pending["native_training"]]
            base_tr = select(self.state.baseline.training, self._focus()[0])
            cand = Candidate(iteration=n, files=files, artifact_id="", training=native,
                             delta=bundle_delta(base_tr, native).to_dict(),
                             hypothesis=hypothesis)
            self._validate_and_finish(n, record, cand, base_tr)
            return
```

In `run()`, change `pending.get("purpose") == "baseline"` to `pending.get("purpose") in ("baseline", "calibration")`.

Behaviour for the resume test: the scripted provider has no replies, so any LLM call raises inside the fake provider. `fb.calls[0].mode == "metered"` passes only if validation is the first bench call.

- [ ] **Step 4: Run to verify they pass, then the whole suite**

Run: `.venv/bin/python -m pytest tests/test_loop.py -q && .venv/bin/python -m pytest -q -m "not live"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git status --short
git add talos/loop.py tests/test_loop.py
git commit -m "loop: calibrate, score research natively, validate each new best on the metered path"
```

---

### Task 10: CLI — `--scoring`, calibration at job start, `talos compile --native`

**Deviation from spec §4.6 (audit, 2026-09-29).** The spec makes `talos compile` native by default. That contradicts the spec's own rollout (§8: native lands behind a flag, default metered, until the live parity tests pass). A native default would put the never-compiled GPU template, and the native toolchain, under every agentic job's build check the day this merges, metered jobs included. On Modal it would also fail every `talos compile` against an app deployed before this change, until `talos setup` is run. So `talos compile` stays metered by default, `--native` builds natively, and a job whose research scoring is native tells its agent to use `--native` (the prompt carries it; the sandbox's environment allowlist is not touched). Rollout step 4 flips the default together with `--scoring`.

**Files:**
- Modify: `talos/cli.py`, `talos/prompts.py` (`PromptContext`), `talos/agentic.py` (`claude_md`), `talos/loop.py` (`Loop._context`)
- Test: `tests/test_cli.py`, `tests/test_agentic.py`

**Interfaces:**
- Consumes: `JobSpec.scoring` (Task 1), `Loop.calibrate` and `Loop._scoring` (Task 9), `EvalRequest.mode`.
- Produces: `CALIBRATION_CACHE = Path.home() / ".talos" / "calibration"`, `talos run --scoring {native,metered}` (default `metered`, flag only, not asked by the wizard), `talos compile --native`, `PromptContext.scoring: str = "metered"`, and `_event_line` texts for the new events and outcome.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cli.py`:

```python
def test_fake_native_run_validates_its_best_and_wins(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    fake_config(tmp_path)
    rc = fake_run(monkeypatch, ["--scoring", "native"])
    out = capsys.readouterr().out
    assert rc == 0 and "Status: won" in out and "scoring native" in out
    run_dir = next((tmp_path / "runs").glob("*/job.json")).parent
    assert json.loads((run_dir / "job.json").read_text())["scoring"] == "native"
    kinds = [json.loads(x)["kind"] for x in (run_dir / "timeline.jsonl").read_text().splitlines()]
    # mutation: execute_job never calling calibrate leaves state.scoring None, so the native
    # job silently runs metered
    assert "calibrated" in kinds and "validated" in kinds
    # the fake run keeps its calibration inside the run directory, never in ~/.talos
    assert list((run_dir / "calibration_cache").rglob("*.json"))


def test_run_defaults_to_metered_scoring(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fake_config(tmp_path)
    seen = {}
    monkeypatch.setattr(cli, "execute_job",
                        lambda spec, store, cfg, resume: seen.update(spec=spec) or 0)
    fake_run(monkeypatch)
    # mutation: a native default before the rollout's live checks
    assert seen["spec"].scoring == "metered"


def test_resume_refuses_a_different_scoring_mode(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    fake_config(tmp_path)
    monkeypatch.setattr(cli, "execute_job", lambda spec, store, cfg, resume: 0)
    fake_run(monkeypatch)
    job_id = next((tmp_path / "runs").glob("*/job.json")).parent.name
    run_dir = tmp_path / "runs" / job_id
    (run_dir / "state.json").write_text("{}")
    rc = cli.main(["run", "--resume", job_id, "--scoring", "native"])
    assert rc == 2
    assert "started with metered scoring; start a new job" in capsys.readouterr().err


def test_compile_is_metered_by_default_and_native_on_the_flag(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "algorithm").mkdir()
    (tmp_path / "algorithm" / "mod.rs").write_text("fn x(){}")
    modes = []

    class Bench:
        def select_hardware(self, challenge, chosen=None):
            return None

        def evaluate(self, request):
            modes.append(request.mode)
            from talos.bench import EvalResult
            from talos.types import CompileResult
            return EvalResult(CompileResult(True, "a", "ok"), [], [], "forced")
    monkeypatch.setattr(cli, "make_bench", lambda *a, **k: Bench())
    monkeypatch.setenv("TALOS_BACKEND", "modal")
    assert cli.main(["compile", "--challenge", "knapsack"]) == 0
    assert cli.main(["compile", "--challenge", "knapsack", "--native"]) == 0
    # mutation: a native default before the rollout's live checks (spec §8); the flag ignored
    # keeps a native job's agent on the slow path for good
    assert modes == ["metered", "native"]


def test_event_lines_for_validation():
    assert cli._event_line("validating", {}) == "validating on TIG's metered runtime"
    assert cli._event_line("validated", {"mean_rel_delta": 0.012}) == \
        "validated: +1.200% vs baseline on the metered runtime"
    assert cli._event_line("demoted", {"reason": "nondeterministic"}) == \
        "demoted by metered validation: nondeterministic"
    assert cli._event_line("iteration_done", {"outcome": "failed:validation",
                                               "runs_since_improvement": 2}) == \
        "rejected: failed metered validation (2 in a row)"
```

Append to `tests/test_agentic.py` (build the `PromptContext` the way the file's existing `claude_md` tests do):

```python
def test_a_native_job_tells_its_agent_to_compile_natively():
    from dataclasses import replace
    ctx = <the file's existing PromptContext fixture for knapsack>
    metered = claude_md(ctx)
    native = claude_md(replace(ctx, scoring="native"))
    assert "`talos compile --challenge knapsack --dir algorithm`" in metered
    # mutation: the flag left out puts a native job's agent on the 287 s metered build
    assert "`talos compile --challenge knapsack --dir algorithm --native`" in native
```

Before writing the resume test, check how the existing `--mode` resume test builds its job (grep for `was started in` in `tests/test_cli.py`) and use the same setup. The `state.json` stub above only needs to exist. `cmd_run` refuses before it loads state.

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_cli.py tests/test_agentic.py -q -k "scoring or native or compile_is or event_lines_for_validation"`
Expected: FAIL (`unrecognized arguments: --scoring`).

- [ ] **Step 3: Implement in `talos/cli.py`**

- Beside `BASELINE_CACHE`: `CALIBRATION_CACHE = Path.home() / ".talos" / "calibration"`.
- Parser (`main`): `r.add_argument("--scoring", choices=["metered", "native"], help="research scoring: metered (TIG's runtime, the default) or native (fast, validated on metered)")`, and on the compile subparser `c.add_argument("--native", action="store_true", help="build the unmetered talos-native runner, as a --scoring native job scores it")`.
- `talos/prompts.py`: `PromptContext` gains `scoring: str = "metered"` (last, with its default). `talos/loop.py::Loop._context` passes `scoring=self._scoring()`. `talos/agentic.py::claude_md` writes the compile line as ``f"`talos compile --challenge {ctx.challenge} --dir algorithm{' --native' if ctx.scoring == 'native' else ''}`"``. The existing `Bash(talos compile:*)` permission already covers the flag.
- In `cmd_run`'s resume branch, after the `--nonces` check:

```python
        if args.scoring is not None and args.scoring != spec.scoring:
            print(f"job {spec.job_id} was started with {spec.scoring} scoring; start a new job "
                  f"to change scoring", file=sys.stderr)
            return 2
```

- Pass `scoring=args.scoring or "metered"` to the `JobSpec(...)` constructor, and add `f", scoring {spec.scoring}"` to the `Job {job_id}: ...` summary line, before the hyperparameters text.
- In `execute_job`: set `calibration_dir = store.run_dir / "calibration_cache"` in the fake branch and `CALIBRATION_CACHE` otherwise, next to `cache_dir`. In the `try` block, after the baseline or template branch and before `final = loop.run()`, add `loop.calibrate(calibration_dir, hardware)`. It runs on every start and resume, and a stored record makes it a no-op read.
- In `cmd_compile`, build the request with `mode="native" if args.native else "metered"`:

```python
    r = bench.evaluate(EvalRequest(args.challenge, files, [], [], 0, None,
                                   CHALLENGES[args.challenge].beat,
                                   mode="native" if args.native else "metered")).compile
```

- `OUTCOME_TEXT["failed:validation"] = "rejected: failed metered validation"`. Add `_event_line` branches:

```python
    elif kind == "validating":
        text = "validating on TIG's metered runtime"
    elif kind == "validated":
        text = f"validated: {data['mean_rel_delta']:+.3%} vs baseline on the metered runtime"
    elif kind == "demoted":
        text = f"demoted by metered validation: {data['reason']}"
    elif kind == "calibrated":
        text = "native budgets: " + ", ".join(f"{t} {us / 1e6:.1f}s"
                                              for t, us in sorted(data["budgets_us"].items()))
    elif kind == "calibration_fallback":
        text = f"native scoring unavailable, scoring metered: {data['reason']}"
```

- [ ] **Step 4: Run to verify they pass, then the whole suite**

Run: `.venv/bin/python -m pytest tests/test_cli.py tests/test_agentic.py -q && .venv/bin/python -m pytest -q -m "not live"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git status --short
git add talos/cli.py talos/prompts.py talos/agentic.py talos/loop.py tests/test_cli.py tests/test_agentic.py
git commit -m "cli: talos run --scoring native, calibration at job start, talos compile --native"
```

---

### Task 11: Docs, invariants and the live parity tests

**Files:**
- Modify: `AGENTS.md`, `README.md`, `docs/architecture.md`, `docs/compute-backends.md`
- Modify: `tests/test_live.py`

**Interfaces:**
- Consumes: every symbol added in Tasks 1–10. `tests/test_docs_references.py` resolves every `path::symbol` these docs cite, so cite only symbols that exist.

- [ ] **Step 1: AGENTS.md**

- In invariant 1, after the paragraph on the per-nonce timeout asymmetry, add: "With `--scoring native` there is a second, bounded asymmetry. Research candidates run natively (`talos/inside.py::run_nonce_native`) under a per-track time budget calibrated from the baseline's metered fuel (`talos/calibration.py::budgets_us`), while the baseline ran metered. A native score never decides anything by itself: a candidate becomes best only after `talos/loop.py::Loop._validate_and_finish` scores it on TIG's metered runtime, on the same nonces, fuel and hardware as the baseline, and held-out confirmation runs only on those metered results. The calibration key (`talos/calibration.py::calibration_key`) carries the hardware class, both pins, the baseline's code and its hyperparameters."
- In invariant 3, add: "The calibration key carries both pins too (`talos/calibration.py::calibration_key`). The native runner's toolchain is `talos/native_runner.py::TOOLCHAIN`, the one `build_so` uses at the pin, and must move with `MONOREPO_REF`."
- Under "What requires a human", add: "Changing the calibration constants in `talos/calibration.py` (`MARGIN_START`, `MARGIN_STEP`, `MARGIN_FLOOR`, `MISS_LIMIT`, `BUDGET_FLOOR_US`), or switching the default of `talos run --scoring` to `native`. Together they decide how close a native budget is to TIG's fuel limit." Also: "Running the native parity live tests (`tests/test_live.py::test_native_parity`). They spend Modal budget or C3 credit."
- In the "Where to look" table add rows: "Native research scoring: runner, calibration, validation" → `talos/native_runner.py::render`, `talos/calibration.py::budgets_us`, `talos/scoring.py::validation_failure`, `talos/loop.py::Loop.calibrate`.

- [ ] **Step 2: README.md**

- Under "Command reference" → "`talos run`", document `--scoring metered|native`: what native does, that the default is metered until the live checks pass, that a job keeps its scoring on resume, and that calibration is cached in `~/.talos/calibration/`.
- Under "`talos compile`", document `--native` (the unmetered runner build a `--scoring native` job scores with; its agent is told to pass it), and that the default stays TIG's metered build until the rollout's live checks pass. Say that a candidate which compiles natively can still fail the metered build, and that validation catches this as `native_metered_build_mismatch`.
- Under "Where results land", add `state.json`'s `validations` list and the `calibrated` / `validating` / `validated` / `demoted` timeline events.

- [ ] **Step 3: docs/architecture.md**

Add a section "## Native research scoring" after "The research loop". Cover in plain prose: the two modes; calibration (the key, the ratio at the last save, the margin and the floors, all labelled ESTIMATE); validation (the order of the checks and the reasons); why the verifier uses the unmetered PTX; the fallback to metered; and the resume behaviour. Cite the MEASURED probe numbers from the spec's §10 table, each with its job id and date. Cite the Task 5 local knapsack numbers, each labelled MEASURED with the date.

- [ ] **Step 4: docs/compute-backends.md**

- Under "Scoring workers", describe the 12-field task tuple and the mode dispatch (`talos/inside.py::run_task`).
- Under "Volumes and the prepare step", describe the `cargo fetch` warm-up and the `.talos-warm-2` marker.
- Under "The job container", describe `CARGO_NET_OFFLINE=true`.
- Add a paragraph on the Modal artifact volume: native artifacts are stored under their own hash as `talos-native`, beside `algo.ptx`. An app deployed before this change keeps working for metered jobs and refuses native ones as a stale deploy.

- [ ] **Step 5: Live parity test (written here, run only by the user)**

Append to `tests/test_live.py`:

```python
def test_native_parity(tmp_path):
    """Manual: the baseline scored metered and native on the same nonces, one per track, must
    give the same quality on every nonce. Run:
    TALOS_LIVE_NATIVE=c3 TALOS_LIVE_CHALLENGE=hypergraph .venv/bin/pytest -m live \
        tests/test_live.py -k native_parity -s
    TALOS_LIVE_NATIVE is c3, modal or local. Spends real credit on c3 and modal."""
    backend = os.environ.get("TALOS_LIVE_NATIVE")
    if backend not in ("c3", "modal", "local"):
        pytest.skip("set TALOS_LIVE_NATIVE=c3|modal|local")
    from talos.bench import EvalRequest, PendingJobStore
    from talos.challenges import CHALLENGES
    from talos.cli import make_bench
    ch = os.environ.get("TALOS_LIVE_CHALLENGE", "knapsack")
    info = mainnet.fetch_challenge_info(ch)
    name, _algorithm_id, _adoption = mainnet.top_algorithm(ch)
    files = mainnet.fetch_algorithm_files(ch, name)
    tr, _ = draw_nonce_sets(info.tracks, new_rand_hash(), training_count=1, holdout_count=0)
    from talos.cli import local_settings
    bench = make_bench(backend, tmp_path, PendingJobStore.memory(),
                       c3_api_key=os.environ.get("C3_API_KEY"),
                       local=local_settings(None) if backend == "local" else None)
    bench.select_hardware(ch)
    out = {}
    for mode in ("metered", "native"):
        r = bench.evaluate(EvalRequest(ch, files, tr, [], info.max_fuel, None,
                                       CHALLENGES[ch].beat, mode=mode))
        assert r.compile.ok, r.compile.output[-3000:]
        out[mode] = {(x.track, x.nonce): x for x in r.training}
    for k, m in out["metered"].items():
        n = out["native"][k]
        assert m.quality == n.quality, (k, m.to_dict(), n.to_dict())
        assert m.fuel_consumed is not None and n.solve_us is not None
    print({"algorithm": name, "rows": {str(k): (v.to_dict(), out["native"][k].to_dict())
                                       for k, v in out["metered"].items()}})
```

Without `C3_API_KEY`, the C3 backend uses the `c3` CLI's login. On `local`, run `talos.local_transport.prepare(ch)` first, as `execute_job` does.

- [ ] **Step 6: Run the docs and contract tests and the whole suite**

Run: `.venv/bin/python -m pytest tests/test_docs_references.py tests/test_contract.py -q` (per memory, these need agentify under Python 3.11; use the 3.11 venv from Global Constraints), then `.venv/bin/python -m pytest -q -m "not live"`.
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git status --short
git add AGENTS.md README.md docs/architecture.md docs/compute-backends.md tests/test_live.py
git commit -m "docs: native research scoring; invariants; the live parity test"
```

---

### Task 12: Gate and mutation check

**Files:**
- Create: `<scratchpad>/mutate.sh` (never the repo)

- [ ] **Step 1: Full gate**

Run: `make check PYTHON=$S/venv311/bin/python > $S/check.log 2>&1; echo rc=$?`, with the venv set up as in Global Constraints. Read `$S/check.log`.
Expected: `rc=0`.

- [ ] **Step 2: Mutation-check the new tests**

For each mutation below: apply it with `sed` or a small Python edit, run only the named test file with `PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=$S/pyc-$N`, confirm that it FAILS, then `git checkout -- <file>`. Record a table of `| mutation | file | test that caught it | CAUGHT/MISSED |`. A MISSED mutation gets its test strengthened in the task's own file, and is then re-checked.

| # | mutation | file |
|---|---|---|
| 1 | `median(v)` → `sum(v) / len(v)` | `talos/calibration.py` |
| 2 | `r.solve_us / f` → `f / r.solve_us` | `talos/calibration.py` |
| 3 | `* v["margin"]` removed | `talos/calibration.py` |
| 4 | `max(BUDGET_FLOOR_US, ...)` → the bare product | `talos/calibration.py` |
| 5 | `if t["misses"] < MISS_LIMIT` → `<=` | `talos/calibration.py` |
| 6 | `max(MARGIN_FLOOR, ...)` removed | `talos/calibration.py` |
| 7 | `"hw": hardware_class` removed from the key | `talos/calibration.py` |
| 8 | `if mode != "metered":` → `if True:` | `talos/inside.py` (content_hash) |
| 9 | `if mode == "native":` → `if False:` in `run_task` | `talos/inside.py` |
| 10 | `fuel_consumed = ...` → `None` in `run_nonce` | `talos/inside.py` |
| 11 | `NATIVE_ENV` merged out of the cargo call | `talos/inside.py` |
| 12 | `p["mode"] = request.mode` block made unconditional | `talos/c3_jobdir.py` |
| 13 | `mode = payload.get("mode", "metered")` → `"metered"` | `talos/c3_job.py` |
| 14 | `kw = {} if mode == "metered" else ...` → always `{"mode": mode}` | `talos/bench.py` |
| 15 | `if improved and self._scoring() == "native":` → `if False:` | `talos/loop.py` |
| 16 | `self.state.best = best` moved before the `if reason:` check | `talos/loop.py` |
| 17 | `if pending.get("native_training"):` → `if False:` | `talos/loop.py` |
| 18 | `if all(r.fuel_consumed is None ...)` → `if False:` | `talos/loop.py` |
| 19 | the recompute of `fuel_budgets_us` in `_note_demotion` removed | `talos/loop.py` |
| 20 | `r.quality != o.quality` → `False` | `talos/scoring.py` |
| 21 | `d.mean_rel_delta > best_delta or` removed (spec's beats-only rule) | `talos/scoring.py` |
| 22 | `"export CARGO_NET_OFFLINE=true\n"` removed | `talos/c3_jobdir.py` |
| 23 | `loop.calibrate(...)` call removed | `talos/cli.py` |
| 24 | `"limit_hit": rt_rc == OUT_OF_FUEL_RC` → `err == "out_of_fuel"` | `talos/inside.py` |
| 25 | `and not r.limit_hit and not o.limit_hit` removed from `quality_mismatches` | `talos/scoring.py` |
| 26 | `or r.limit_hit` removed from `fuel_proxy_misses` | `talos/scoring.py` |
| 27 | `if o is None or not o.limit_hit:` → `if True:` | `talos/scoring.py` |
| 28 | the resume early-return in `Loop.calibrate` removed | `talos/loop.py` |
| 29 | `if self._scoring() == "metered":` before `runtime_ratio` → `if True:` | `talos/loop.py` |
| 30 | `' --native' if ctx.scoring == 'native' else ''` → `''` | `talos/agentic.py` |
| 31 | `mode="native" if args.native else "metered"` → always `"native"` | `talos/cli.py` |

Stub `subprocess.run` for anything under `tests/test_cli.py` as in the existing CLI tests. No mutation run may reach a real `c3`, `docker` or `modal` (memory: talos-c3-test-safety).

- [ ] **Step 3: Report**

Report the gate's final lines, the test count before (MEASURED at the rebased branch point) and after, and the mutation table. Any MISSED mutation is fixed before the PR is opened.

---

## Rollout after merge (human steps, spec §8)

1. Run `test_native_parity` on C3 for hypergraph (the first compile of the GPU template if Task 5 Step 4 was skipped), on C3 for knapsack, and on Modal for hypergraph. A challenge whose native quality differs from metered stays metered.
2. Run one hypergraph night on tig-adi with `--scoring native`, with the same direction and budget as a recent metered night. Compare iteration wall time, validation pass rate and fuel-proxy misses from `state.json` `validations` and the calibration record's `demotions`.
3. Switching the default to `native` needs a human (AGENTS.md, Task 11). It flips `talos run --scoring` and `talos compile` together (Task 10's deviation note).

---

## Task 5 results: local knapsack, native vs metered (MEASURED 2026-09-29)

Command: `$S/venv311/bin/python $S/native_local_check.py $S/localcheck` (the Task 5 script, with
`local_settings(None)`), at commit f4402b7 plus Tasks 6–7, on this machine's Docker. Image
`ghcr.io/tig-foundation/tig-monorepo/knapsack/dev:0.0.7`, already local; `prepare` re-warmed the
volume for the `.talos-warm-2` marker (`cargo fetch`) in 40 s. Container limits from the daemon:
16 CPUs, 26 GiB. Algorithm: mainnet top `knap_exact16`, track `n_items=1000,budget=10`,
`max_fuel` 5,000,000,000,000, 3 training nonces.

| nonce | metered quality | native quality | fuel_consumed | solve_us | metered runtime_ms | native runtime_ms |
|---|---|---|---|---|---|---|
| 0 | 233007 | 233007 | 377,194,024 | 65,360 | 943 | 657 |
| 1 | 309510 | 309510 | 410,580,013 | 71,273 | 960 | 703 |
| 2 | 232131 | 232131 | 366,507,720 | 63,202 | 881 | 680 |

- `quality_equal`: true on all 3 nonces.
- Wall time per evaluate, build included: metered 260.8 s, native 60.6 s.
- The CPU runner template compiled on the first attempt, offline, in the network-less local job
  container (`CARGO_NET_OFFLINE=true`), so the `cargo fetch` warm-up covers every crate it needs.
- Step 4 (GPU template compile) was not run, by the user's choice. The GPU template is first
  compiled in the human-run live parity test (Task 11).
- Derived, not measured: the per-nonce ratio is about 1.7e-4 us per fuel unit, so a knapsack
  budget at `max_fuel` and margin 0.8 is about 690 s (ESTIMATE). That is above `NONCE_TIMEOUT_S`
  (600 s), so on this track the outer per-nonce timeout, not the fuel budget, is the binding cap.
