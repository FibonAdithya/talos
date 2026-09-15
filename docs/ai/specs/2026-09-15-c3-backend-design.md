# C3 bench backend — design

Date: 2026-09-15. Branch: `c3-backend`, cut from the Modal implementation at
`50e1e14`. Implementation starts by rebasing onto the tip of
`talos-implementation` (PR #1), which is two commits ahead.

This spec adds a second compute backend to Talos: C3 (cthree.cloud) batch
jobs, chosen at `talos setup` as an alternative to Modal. It also changes the
bench protocol so that both backends serve one call per iteration.

## 1. Why one call per iteration

C3 runs a job per submission with no image cache on the assigned VM. Measured
from the 2026-09-14 spike job (`job_1789400132628_x95n8c`, knapsack, profile
`cpu-d3-4vcpu-16gb`), all MEASURED from `c3 logs`:

| step | time |
|---|---|
| queue, image pull, workspace setup, before the script starts | ~4 min |
| `build_algorithm` from clean | 466 s |
| one nonce at mainnet fuel, runtime + verifier | 1–2 s |
| second `build_algorithm` in the same job | 443 s (builds are not incremental) |
| job created → job succeeded, 2 nonces | 19 min |

Every job therefore costs about 12 minutes of fixed overhead. Money is not the
constraint (that job cost about £0.04 at £0.11/h); wall clock is. The loop
today makes three bench calls per iteration (compile, score training, score
held-out on a win). Mapped one-to-one onto jobs, a winning iteration would take
about 40 minutes and every iteration would build the same code twice. So the
protocol collapses to one call, and the C3 backend runs that call as one job
with the held-out decision made inside the job.

## 2. Bench protocol

`talos/bench.py` replaces `compile` and `score` with one method. The cost
methods are unchanged.

```
class Bench(Protocol):
    def evaluate(self, request: EvalRequest) -> EvalResult: ...
    def cost_mark(self) -> float: ...
    def cost_usd_since(self, mark: float) -> float: ...

@dataclass(frozen=True)
class EvalRequest:
    challenge: str
    files: dict[str, str]
    training: list[NonceSet]
    holdout: list[NonceSet]
    fuel: int
    baseline_training: list[NonceResult] | None   # None = score held-out unconditionally
    rule: BeatRule

@dataclass
class EvalResult:
    compile: CompileResult
    training: list[NonceResult]                    # empty when compile.ok is False
    holdout: list[NonceResult] | None              # None when not scored
    holdout_reason: str      # "won" | "not_won" | "forced" | "not_compiled" | "timeout"
```

Rules:

- Held-out is scored when `beats(baseline_training, training, rule)` is true,
  or unconditionally when `baseline_training` is None (`"forced"`). The
  baseline measurement passes None.
- A compile failure returns `compile.ok == False`, `compile.output` holding the
  build log tail, empty `training`, `holdout=None`, reason `"not_compiled"`.
  It never raises; the loop's compile-fix rounds depend on that.
- `BenchUnavailable` is raised only for infrastructure failures, as today.
- The rand hash never appears in an exception message, a log line, or a job
  name. Both backends redact it from anything derived from a command line.

`ModalBench.evaluate` is the existing compile call, a starmap over the
training nonces, `beats` locally, and a starmap over the held-out nonces on a
win or when forced. Its cost accounting is unchanged.

`FakeBench.evaluate` keeps the `scores` callback and applies the same
conditional, so every loop test exercises the new shape.

`talos/loop.py`: the iterate step makes one `evaluate` call and then does the
bookkeeping it does today. `_confirm(cand)` no longer calls the bench; it takes
the result's `holdout` list. `_BudgetedBench` wraps `evaluate` only.
`resolve_baseline` makes one `evaluate` call with `baseline_training=None` and
requires `holdout` to be present.

The resume path for a run killed while `confirming` is kept and generalised:
see §5.

## 3. C3 backend, client side

New module `talos/c3_bench.py`, class `C3Bench`. One `evaluate` call is one
C3 job.

### 3.1 Job directory

`runs/<job_id>/c3/<iteration>/` (the baseline uses `c3/baseline/`). Contents:

| file | content |
|---|---|
| `.c3` | generated YAML, see §3.2 |
| `job.sh` | fixed wrapper, see §4 |
| `c3_job.py`, `inside.py`, `scoring.py`, `types.py`, `challenges.py` | copies of the Talos modules the job imports, flattened so the job needs no package install |
| `payload.json` | `challenge`, `challenge_id`, `files`, `training`, `holdout`, `fuel`, `baseline_training`, `rule`, `monorepo_ref` |

The rand hash lives only in `payload.json`. The whole directory is uploaded to
C3's workspace store on deploy; that is accepted, as the same secret already
goes to Modal as a function argument.

### 3.2 Generated `.c3`

```yaml
project: talos
job_name: talos-<challenge>-<iteration>
script: job.sh
hardware: <profile>
time: "<HH:MM:SS>"
docker:
  image: docker.io/<namespace>/tig-<challenge>-dev:<DEV_IMAGE_TAG>
  requires_accelerator: <cuda|none>
```

- `profile`: `cpu-d3-4vcpu-16gb` for CPU challenges (matches the 4-core Modal
  spec); `l40` class for GPU challenges (closest to Modal's L40S). Both are
  constants in `talos/challenges.py` next to the Modal resource fields.
- `time`: 20 minutes build allowance + ceil(nonce_count × NONCE_TIMEOUT_S /
  workers), rounded up to the minute, capped at 6 hours. `workers` is the
  profile's core count (4 for the CPU profile, 1 for GPU profiles, since one
  GPU serialises the runs). Hitting the cap is handled by the TIMED_OUT path in
  §3.4, so the cap bounds cost without losing results.
- Every field is a constant or derived from the request; a unit test asserts
  the exact file text for a fixed request.

### 3.3 Submission and polling

All subprocess calls go through an injected `run` (as `deploy_bench` does now)
and time through an injected clock and sleep, so the client is unit-tested
with canned stdout.

1. `c3 deploy --json` from the job directory; parse `job_id` from stdout.
2. Write `pending_job` to state (§5) before anything else happens.
3. Poll `c3 squeue --json` every 20 s until the status is not one of
   `PENDING`, `SCHEDULING`, `RUNNING`. Record the first time `RUNNING` was
   observed and the terminal time for cost.
4. A job still `PENDING`/`SCHEDULING` 30 minutes after submission is
   cancelled with `c3 cancel <id>` and reported as `BenchUnavailable`
   ("no C3 capacity for <profile> in 30 min"). The loop pauses the run as it
   does for a Modal outage.
5. `c3 pull <id> --json` into the job directory; read `results.json` and
   `build.log` from the pulled artifacts.

### 3.4 Failure mapping

| observed | result |
|---|---|
| `SUCCEEDED`, complete `results.json` | normal `EvalResult` |
| `SUCCEEDED`, `results.json` has compile section only | compile failure; `compile.output` = tail of `build.log` |
| `TIMED_OUT`, or `FAILED` with a partial `results.json` | nonces present are used. A set the job marked as started has its missing nonces filled in as `timeout` errors. A held-out set never started stays `holdout=None` with the job's `holdout_reason` if it wrote one, else `"timeout"` |
| `FAILED` with no `results.json`, `CANCELED` by someone else, or a `c3` command that exits non-zero or prints unparseable JSON | retried with the anchored exponential backoff window the Modal client uses (`retry_window_s`, anchored at the first failure); then `BenchUnavailable` |

A retry after a failed job submits a new job; a retry after a failed `pull`
or `squeue` call repeats that call against the same job id.

### 3.5 Cost

`cost_usd_since` accrues, per job, `(terminal_time − first_running_time) ×
GBP_PER_HOUR[profile] / 3600 × USD_PER_GBP`. Both tables are constants in
`talos/c3_bench.py`, marked ESTIMATE in code and in the status line. Queue
time is not billed by C3 and is not counted. Values at 2026-09-15 from
`c3 list`: `cpu-d3-4vcpu-16gb` £0.11/h, `l40` class £0.95–1.49/h (use 1.49).

## 4. Inside the job

`job.sh` (fixed text, part of the package):

```sh
#!/bin/bash
set -euo pipefail
curl -fsSL "https://codeload.github.com/tig-foundation/tig-monorepo/tar.gz/$MONOREPO_REF" | tar xz -C /app --strip-components=1
cd "$C3_JOB_WORKDIR"
exec python3 c3_job.py
```

`$MONOREPO_REF` is substituted into `job.sh` when the job directory is
generated, from the pin in `talos/challenges.py`. The spike measured the tarball download at 2 s for
15 MB. The mirrored dev image ships Python 3.12 (MEASURED 2026-09-15 with
`docker run ... python3 --version` against the knapsack mirror).

`modal_app/inside.py` moves to `talos/inside.py` unchanged; the Modal app
imports it from there. `tests/test_inside.py` follows it.

New module `talos/c3_job.py`, `main()`:

1. Read `payload.json`. Stage and build with `inside.stage_algorithm` and
   `inside.build`. On failure write `results.json` with the `compile` section
   only, copy the build log to `$C3_ARTIFACTS_DIR/build.log`, exit 0. Exit 0
   is deliberate: a compile error is a result, not a job failure.
2. Score the training nonces with a `multiprocessing` pool of `workers`
   processes calling `inside.run_nonce`. Fuel metering makes each result
   independent of scheduling, so parallel scoring is identical to sequential.
   After every nonce, rewrite `results.json` atomically (write temp, rename)
   so a timed-out job still yields partial results.
3. `beats(baseline_training, training, rule)` with the shipped
   `scoring.py`; on a win, or when `baseline_training` is null, score the
   held-out set the same way. Record `holdout_reason`.
4. Write the final `results.json` and `build.log` to `$C3_ARTIFACTS_DIR`.

`results.json`:

```json
{"compile": {"ok": true, "artifact_id": "<content hash>", "output": "<tail>"},
 "training": [<NonceResult dicts>],
 "holdout": [<NonceResult dicts>] | null,
 "holdout_reason": "won",
 "started": {"training": true, "holdout": false}}
```

`started` lets the client tell "held-out never began" from "held-out began
and was cut off" (§3.4). `artifact_id` is `content_hash(files)` computed with
the same function the Modal app uses, moved to `talos/inside.py` so both
backends agree.

Log lines carry elapsed seconds, nonce numbers, exit codes and timings. They
never carry the rand hash or the runtime command line.

## 5. State, resume, cancel

The `Bench` protocol gains one method:

```
    def reattach(self, pending: dict) -> EvalResult | None: ...
```

`JobState` gains `pending_job: dict | None`: `backend`, `job_id`, `purpose`
(`"baseline"` or an iteration number), `job_dir`, and for an iteration the
`hypothesis` dict, which is otherwise only persisted after scoring. `C3Bench`
writes it through a callback the loop injects, right after `c3 deploy`
returns, and clears it after `pull` succeeds. `ModalBench` never sets it and
its `reattach` returns None.

On resume, if `pending_job` is set, the loop calls `bench.reattach(pending)`
before generating any hypothesis. `C3Bench.reattach` rebuilds the request from
the job directory's `payload.json`, polls the recorded job id instead of
submitting, and returns the result. The loop then feeds that result through
the same baseline or iteration bookkeeping as a fresh call, using the files
from the payload and the hypothesis from the record. A None return means the
backend cannot reattach; the loop clears `pending_job` and redoes the step,
which for Modal is what happens today. The existing "confirmation killed
mid-flight" path is subsumed: with one call per iteration, a run can only be
mid-flight with a pending job.

Cancelling a run with a pending C3 job cancels the job with `c3 cancel`, since
a running job bills. Modal has no pending job and needs nothing.

The baseline cache key includes the hardware class; the C3 backend's class
string is `c3-<profile>`, so a baseline measured on C3 is never reused on
Modal or the reverse. The cache directory is shared.

## 6. Budget rename

`modal_usd` becomes `compute_usd` in `Budget`, `Spend`, the state file, and
the loop's accounting. The flag becomes `--budget-compute-usd`; the wizard
prompt and the status line say "compute spend (estimated)". `DEFAULT_MODAL_USD`
becomes `DEFAULT_COMPUTE_USD` with the same value. No state files from real
runs exist yet, so no migration.

## 7. Setup and config

`Config` gains `backend: str`, `"modal"` or `"c3"`, defaulting to `"modal"`
when the key is absent from an existing `talos.config.json`.

`talos setup` asks "Compute backend (modal or c3)" before the credential
prompts. For c3 it runs, through the injected runner:

- `c3 whoami`: a non-zero exit fails setup with the CLI's own message, which
  already says to run `c3 login`.
- `c3 balance`: parsed for the credit line; below £1 prints a warning and
  continues.

It skips the Modal token prompts and the deploy. For modal the flow is
unchanged.

`talos run` builds `C3Bench` or `ModalBench` from `config.backend`. For c3,
before the baseline step, it checks the image tag exists with a GET to
`https://hub.docker.com/v2/repositories/<namespace>/tig-<challenge>-dev/tags/<tag>`
and fails with the mirror instruction from §8 if it does not. The check goes
through an injected fetch so it is unit-tested.

## 8. Maintainer image mirror

Talos users never touch GHCR. The maintainers mirror the dev images once per
`DEV_IMAGE_TAG`:

- `scripts/mirror_images.sh [challenge...]`: for each challenge, `docker pull`
  the GHCR dev image at `DEV_IMAGE_TAG`, tag it
  `docker.io/<namespace>/tig-<challenge>-dev:<DEV_IMAGE_TAG>`, push. The
  namespace and tag are read from `talos/challenges.py` so there is one source
  of truth.
- `make mirror-images` runs it for all eight challenges.
- README: which tags are mirrored, and that a `DEV_IMAGE_TAG` bump requires
  re-running it before the release.

The namespace is the constant `IMAGE_NAMESPACE = "fibonadithya"` in
`talos/challenges.py`, overridable by the environment variable
`TALOS_IMAGE_NAMESPACE` for maintainers testing a new mirror. It is not a
setup prompt.

Already mirrored: `docker.io/fibonadithya/tig-knapsack-dev:0.0.7` (public,
verified 2026-09-15 with the Hub tags endpoint).

## 9. Testing

Unit, on any machine:

- `test_bench.py`: `ModalBench.evaluate` with a fake function table: compile
  failure short-circuits; held-out scored on a win, on forced, not otherwise;
  cost accrues per call. Existing rand-hash redaction tests kept.
- `test_c3_bench.py`: generated `.c3` text for a fixed request; time-limit
  formula at the boundaries (1 nonce, cap); deploy/squeue/pull sequence with
  canned JSON; each row of the §3.4 table; pending timeout cancels and raises;
  retry window anchored at first failure; cost from RUNNING to terminal only;
  `reattach` skips deploy; rand hash absent from every generated file except
  `payload.json` and from every exception message.
- `test_c3_job.py`: fake `run`; compile failure writes compile-only results
  and exits 0; partial `results.json` after each nonce; conditional held-out
  for won / not won / forced; `started` flags; log lines never contain the
  rand hash.
- `test_loop.py`: all existing tests moved to `FakeBench.evaluate`; resume
  with `pending_job` reattaches instead of resubmitting; cancel calls the
  bench's cancel hook.
- `test_cli.py`: setup with backend c3 skips Modal prompts, fails on expired
  session, warns on low balance; config without `backend` loads as modal; run
  refuses a missing image tag with the mirror instruction.
- `test_budget.py`, `test_state.py`: the rename.

Each new test names the mutation it catches, and the implementation plan's
audit checks that no test is tautological.

Live, marked `live`, run by hand:

- `test_live.py::test_c3_knapsack_job`: a real knapsack evaluate with two
  nonces on one track, `baseline_training=None`, asserting both sets are
  scored, qualities are positive integers, and cost is under £0.10. This is
  the release gate for the backend, and its measured time and cost go in the
  README as MEASURED with the date.

## 10. Out of scope

More than one job in flight; artifact chaining between jobs (`/jobs/<id>`
mounts); C3 datasets; provider pinning; a user-facing mirror command; any
change to what a score or a win means.
