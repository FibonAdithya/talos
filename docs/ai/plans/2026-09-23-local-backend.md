# Local Docker Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A third compute backend, `local`, that compiles and scores candidates in a Docker container on the user's own machine, chosen at `talos setup` beside `modal` and `c3`.

**Architecture:** The local backend is the existing `C3Bench` driven through a new `DockerTransport` that satisfies the C3 transport protocol (deploy, status, cancel, fetch). Each evaluate call writes the C3 job directory in a `local` flavour, runs the challenge's GHCR dev image detached with that directory bind-mounted, polls the container like a C3 job, and reads `results.json` off the mount. Two named Docker volumes per challenge, keyed by the monorepo and image pins, hold the checkout and the cargo registry so the job container runs with networking off. (Retracted after measurement: they do not make builds incremental; see the spike results and the spec's retraction note.)

**Tech Stack:** Python 3.10+, the `docker` CLI over `subprocess`, pytest with injected runners (no test touches Docker), ruff.

**Spec:** `docs/ai/specs/2026-09-23-local-backend-design.md`. One deliberate deviation: the spec gives `C3Bench` a `flavour: str` parameter; this plan gives it `local: LocalSettings | None` instead, because the job directory writer needs the CPU and memory limits and a dataclass carries them without a second parameter. The subdirectory is `local` when `local` is set and `c3` otherwise, which is what the spec's `flavour` selected. A second small addition to the spec's §4.4 flag table: the job container also gets `-e CARGO_HOME=` and `-e RUSTUP_HOME=` set to the paths the image uses, because `HOME=/tmp` would otherwise make cargo look for its registry and toolchain under /tmp.

## Global Constraints

- Python floor is 3.10 (`pyproject.toml`); no `match`, no `tomllib`, no `datetime.UTC`.
- CI runs the suite on Linux, macOS and Windows. No unit test may run `docker`, reach the network, or depend on `os.getuid` existing. Every subprocess call goes through an injected `run`.
- `make check` (ruff, pytest without `live`, agentify contract) is the gate. `tests/test_docs_references.py` resolves every `path::symbol` cited in `AGENTS.md`, `README.md` and `docs/*.md`; a cited symbol must exist.
- The job's `rand_hash` never reaches a log line, an exception message, a container name or a label. Only `payload.json` carries it.
- `talos/c3_job.py` and the other `JOB_MODULES` are copied into the job directory; they may import only each other and the standard library.
- Every number that reaches docs or the PR body is labelled MEASURED (run in this session, output shown) or ESTIMATE.
- Commit after every task with explicit paths. Never `git add -A`.
- The spec's §4.4 flag table is the sandbox. Loosening it is a human decision; the plan implements it verbatim.

## Review Focus

Inputs the spec implies but names no test for. Each has its test in the task that owns the code.

1. **Docker daemon down at `talos run` after a successful setup.** Expected: the job is marked failed with a stop reason and one stderr line naming Docker; no traceback. Test in Task 9 (`test_run_local_reports_a_docker_failure_and_fails_the_job`).
2. **`docker inspect` timestamps.** They carry nanoseconds (`2026-09-23T10:00:00.123456789Z`) and the never-finished sentinel `0001-01-01T00:00:00Z`. Expected: both parse; a running container is never reported finished. Test in Task 6 (`test_parse_time_handles_nanoseconds_and_the_zero_sentinel`).
3. **A run directory path with a space or a Windows drive letter.** Expected: each `-v` value is one argv element, never split. Test in Task 6 (`test_run_args_keeps_a_path_with_a_space_as_one_argument`).
4. **A hostname with dots, upper case or an empty string.** Expected: one stable slug; empty becomes `host`. Test in Task 2 (`test_host_slug_is_stable_and_never_empty`).
5. **Memory default on a small machine or on Windows.** Expected: total minus 4 with a floor of 4; 8 where total memory is unknown. Test in Task 9 (`test_default_local_memory_floors_and_falls_back`).

---

### Task 1: Spike the two assumptions on the knapsack image

The spec's §6 names two assumptions only an experiment settles. This task is throwaway: nothing it builds is kept, and its output is a results block appended to this plan.

**Files:**
- Create: `<scratchpad>/spike.sh` (the session scratchpad directory, never the repo)
- Modify: `docs/ai/plans/2026-09-23-local-backend.md` (append the results block at the end)

**Interfaces:**
- Produces: the image's `CARGO_HOME` and `RUSTUP_HOME` paths, whether an offline incremental build works, whether the container can run as the host uid, and MEASURED clean and incremental build times. Task 7 reads `CARGO_HOME`/`RUSTUP_HOME` handling from this; Task 10 records the timings.

- [ ] **Step 1: Write the spike script**

```bash
#!/bin/bash
# Throwaway. Answers spec §6. Run from anywhere; writes nothing into the repo.
set -uo pipefail
IMG=ghcr.io/tig-foundation/tig-monorepo/knapsack/dev:0.0.7
REF=84a5787f5b14a630bdf40f52bccf37887d3d8464
UID_GID="$(id -u):$(id -g)"
ART="$(dirname "$0")/art"; mkdir -p "$ART"

echo "== pull"; docker pull "$IMG" || exit 1
echo "== image env"
docker run --rm "$IMG" sh -c 'echo CARGO_HOME=${CARGO_HOME:-unset} RUSTUP_HOME=${RUSTUP_HOME:-unset} HOME=$HOME; id; which build_algorithm cargo'
CH=$(docker run --rm "$IMG" sh -c 'echo ${CARGO_HOME:-$HOME/.cargo}')
RH=$(docker run --rm "$IMG" sh -c 'echo ${RUSTUP_HOME:-$HOME/.rustup}')
echo "CARGO_HOME=$CH RUSTUP_HOME=$RH"
docker volume create spike-app >/dev/null; docker volume create spike-cargo >/dev/null

echo "== clone + clean build (network on, root)"
docker run --rm -v spike-app:/app -v "spike-cargo:$CH" "$IMG" bash -c "
  set -e
  curl -fsSL https://codeload.github.com/tig-foundation/tig-monorepo/tar.gz/$REF | tar xz -C /app --strip-components=1
  cd /app && time build_algorithm fast_and_furious
  chown -R $UID_GID /app $CH"
echo "clean build rc=$?"

echo "== incremental build, network off, host uid, hardened"
docker run --rm --network none --user "$UID_GID" -e HOME=/tmp -e "CARGO_HOME=$CH" -e "RUSTUP_HOME=$RH" \
  --cap-drop ALL --security-opt no-new-privileges --pids-limit 4096 \
  -v spike-app:/app -v "spike-cargo:$CH" "$IMG" bash -c "cd /app && time build_algorithm knap_lean"
echo "offline build rc=$?"

echo "== bind mount ownership"
docker run --rm --user "$UID_GID" -e HOME=/tmp -v "$ART:/artifacts" "$IMG" sh -c 'echo hi > /artifacts/x && id -u'
ls -ln "$ART"
docker volume rm spike-app spike-cargo >/dev/null
echo "== done"
```

- [ ] **Step 2: Run it in the background with a log file**

Run: `bash <scratchpad>/spike.sh > <scratchpad>/spike.log 2>&1` with `run_in_background: true` and `timeout: 3600000`. The pull is about 13 GB and the clean build about 8 minutes; do not poll more often than every 2 minutes.

Expected in the log: `CARGO_HOME=...`, `clean build rc=0` with a `real` line, `offline build rc=0` with a `real` line, and the `ls -ln` line showing your uid on `x`.

- [ ] **Step 3: Record the results in this plan**

Append to the end of this file:

```markdown
## Spike results (Task 1, MEASURED <date>)

| Question | Result | Evidence |
|---|---|---|
| `CARGO_HOME` in the image | `<path>` | `spike.log` image env line |
| `RUSTUP_HOME` in the image | `<path>` | same |
| Offline incremental build as host uid | rc=<n> | `offline build rc=` line |
| Bind-mount file owned by host uid | yes/no | `ls -ln` line |
| Clean `build_algorithm` | <m>m<s>s | `time` under "clean build" |
| Incremental `build_algorithm` | <m>m<s>s | `time` under "incremental build" |
```

If the offline build fails, Task 7's `run_args` drops `--network none` and Task 10's docs say so (spec §6 fallback 1). If the uid run fails, `host_uid()` in Task 6 returns `None` and Task 10 documents root-owned artifacts (spec §6 fallback 2). Both fallbacks are decided here, once, and the later tasks are edited before they start.

- [ ] **Step 4: Commit the results block**

```bash
git add docs/ai/plans/2026-09-23-local-backend.md
git commit -m "plan: record the local backend spike results"
```

---

### Task 2: Local hardware class

**Files:**
- Modify: `talos/challenges.py` (after `c3_hardware_class`)
- Test: `tests/test_baseline.py` (after `test_hardware_class_separates_cpu_memory_and_gpu`)

**Interfaces:**
- Produces: `host_slug(host: str) -> str`; `local_workers(spec: ChallengeSpec, cpus: int) -> int`; `local_hardware_class(spec: ChallengeSpec, cpus: int, memory_gib: int, gpu_name: str | None, host: str) -> str`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_baseline.py` (extend the existing `from talos.challenges import ...` line with `host_slug, local_hardware_class, local_workers`):

```python
def test_host_slug_is_stable_and_never_empty():
    # mutation: keeping case or dots lets "Box.local" and "box-local" become two cache keys
    assert host_slug("Box.local") == "box-local" == host_slug("box-local")
    assert host_slug("  ") == "host" and host_slug("") == "host"
    assert host_slug("a__b--c") == "a-b-c"


def test_local_hardware_class_separates_limits_host_and_gpu():
    knapsack, hypergraph = CHALLENGES["knapsack"], CHALLENGES["hypergraph"]
    assert local_hardware_class(knapsack, 8, 12, None, "Box") == "local-box-cpu8-mem12"
    # mutation: dropping cpus or memory lets a baseline measured under other limits serve a run
    assert local_hardware_class(knapsack, 4, 12, None, "box") != "local-box-cpu8-mem12"
    assert local_hardware_class(knapsack, 8, 8, None, "box") != "local-box-cpu8-mem12"
    assert local_hardware_class(hypergraph, 8, 12, "NVIDIA L40S", "box") == "local-box-gpu-nvidia-l40s"
    with pytest.raises(ValueError):
        local_hardware_class(hypergraph, 8, 12, None, "box")
    # mutation: a local class that does not start with "local-" could equal a Modal or C3 one
    for cls in (local_hardware_class(knapsack, 4, 8, None, "x"),
                local_hardware_class(hypergraph, 1, 8, "L40S", "x")):
        assert cls.startswith("local-") and cls != hardware_class(knapsack)


def test_local_workers_use_every_cpu_but_one_gpu():
    # mutation: 8 workers on one GPU serialise on the device and time the job out
    assert local_workers(CHALLENGES["knapsack"], 8) == 8
    assert local_workers(CHALLENGES["hypergraph"], 8) == 1
```

- [ ] **Step 2: Run to verify they fail**

Run: `python3 -m pytest tests/test_baseline.py -k "host_slug or local_hardware or local_workers" -v`
Expected: FAIL with `ImportError: cannot import name 'host_slug'`

- [ ] **Step 3: Implement**

In `talos/challenges.py` add `import re` at the top and, after `c3_hardware_class`:

```python
def host_slug(host: str) -> str:
    """A hostname (or a GPU name) as a cache-key component: lower case, every run of characters
    outside [a-z0-9] becomes one dash, no dash at either end. Empty input becomes "host"."""
    return re.sub(r"[^a-z0-9]+", "-", host.lower()).strip("-") or "host"


def local_workers(spec: ChallengeSpec, cpus: int) -> int:
    return 1 if spec.is_gpu else cpus


def local_hardware_class(spec: ChallengeSpec, cpus: int, memory_gib: int,
                         gpu_name: str | None, host: str) -> str:
    """Baseline cache key component for the local backend. Prefixed so a local baseline never
    matches a Modal or C3 one. The limits are in it because they change the timings a baseline
    was measured under; the host guards a home directory synced between machines."""
    if spec.is_gpu:
        if not gpu_name:
            raise ValueError(f"{spec.name} is a GPU challenge; its local hardware class needs "
                             f"the GPU name")
        return f"local-{host_slug(host)}-gpu-{host_slug(gpu_name)}"
    return f"local-{host_slug(host)}-cpu{cpus}-mem{memory_gib}"
```

- [ ] **Step 4: Run to verify they pass**

Run: `python3 -m pytest tests/test_baseline.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add talos/challenges.py tests/test_baseline.py
git commit -m "challenges: local hardware class keyed on host, limits and GPU name"
```

---

### Task 3: Local flavour of the job directory

**Files:**
- Modify: `talos/c3_jobdir.py`
- Test: `tests/test_c3_jobdir.py`

**Interfaces:**
- Consumes: `local_workers` from Task 2.
- Produces: `LocalSettings(cpus: int, memory_gib: int)` frozen dataclass; `local_job_sh_text() -> str`; `local_settings_doc(request: EvalRequest, local: LocalSettings, seconds: int) -> dict`; `payload(request, workers: int | None = None) -> dict`; `write_job_dir(job_dir, request, purpose, local: LocalSettings | None = None) -> Path`. With `local` set the directory holds `local.json` instead of `.c3`, and `payload.json`'s `workers` is `local_workers(spec, local.cpus)`. `local.json` keys: `challenge, image, cpus, memory_gib, gpu, workers, time_limit_s, request_hash`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_c3_jobdir.py` (extend the imports with `from talos.c3_jobdir import LocalSettings` and `dev_image` is already imported):

```python
def test_local_flavour_writes_local_json_and_a_job_sh_without_a_download(tmp_path):
    d = c3_jobdir.write_job_dir(tmp_path / "job", req(n=4), "3", local=LocalSettings(8, 12))
    names = sorted(p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file())
    # mutation: shipping .c3 on the local path, or forgetting local.json
    assert ".c3" not in names and "local.json" in names and "payload.json" in names
    sh = (d / "job.sh").read_text()
    # mutation: the C3 job.sh downloads the monorepo; locally it is on the /app volume
    assert "curl" not in sh and "codeload" not in sh and "python3 -m talos.c3_job" in sh
    assert b"\r" not in (d / "job.sh").read_bytes()
    if os.name != "nt":
        assert stat.S_IMODE((d / "job.sh").stat().st_mode) & stat.S_IXUSR
    doc = json.loads((d / "local.json").read_text())
    assert doc["challenge"] == "knapsack" and doc["image"] == dev_image("knapsack")
    assert doc["cpus"] == 8 and doc["memory_gib"] == 12 and doc["gpu"] is False
    # mutation: workers left at C3's 4 on a 8-cpu container idles half the machine
    assert doc["workers"] == 8
    assert json.loads((d / "payload.json").read_text())["workers"] == 8
    # 8 nonces, 8 workers: 1200 + ceil(8*600/8) = 1800 s
    assert doc["time_limit_s"] == 1800
    assert doc["request_hash"] == c3_jobdir.request_hash(req(n=4))
    # mutation: the rand hash in local.json would land in the container name and `docker ps`
    assert HASH not in (d / "local.json").read_text() and HASH not in sh


def test_local_flavour_gpu_challenge_runs_one_worker(tmp_path):
    d = c3_jobdir.write_job_dir(tmp_path / "job", req(challenge="hypergraph"), "1",
                                local=LocalSettings(8, 12))
    doc = json.loads((d / "local.json").read_text())
    assert doc["gpu"] is True and doc["workers"] == 1


def test_request_hash_is_the_same_for_both_flavours():
    # mutation: hashing the workers count makes a local request never match a C3 one, and worse,
    # never match itself after a setup that changed the CPU count
    r = req()
    assert c3_jobdir.request_hash(r) == c3_jobdir.request_hash(r)
    assert c3_jobdir.payload(r, workers=8)["workers"] == 8
    assert c3_jobdir.payload(r)["workers"] == 4
```

- [ ] **Step 2: Run to verify they fail**

Run: `python3 -m pytest tests/test_c3_jobdir.py -k local_flavour -v`
Expected: FAIL with `ImportError: cannot import name 'LocalSettings'`

- [ ] **Step 3: Implement**

In `talos/c3_jobdir.py`:

Extend the challenges import to `from talos.challenges import (CHALLENGES, DEV_IMAGE_TAG, MONOREPO_REF, c3_image, c3_profile, c3_workers, dev_image, local_workers)` and add `from dataclasses import asdict, dataclass`.

After `JOB_MODULES`:

```python
@dataclass(frozen=True)
class LocalSettings:
    """What the local backend fixes at `talos setup`: the job container's CPU and memory limits.
    Both are in the local hardware class, so changing either invalidates local baselines."""
    cpus: int
    memory_gib: int
```

After `job_sh_text`:

```python
def local_job_sh_text() -> str:
    """The monorepo is already on the /app volume (talos/local_transport.py::prepare), so the
    local job only changes into the bind-mounted job dir and runs the same runner as C3."""
    return ("#!/bin/bash\nset -euo pipefail\ncd \"$C3_JOB_WORKDIR\"\n"
            "exec python3 -m talos.c3_job\n")


def local_settings_doc(request: EvalRequest, local: LocalSettings, seconds: int) -> dict:
    """local.json: everything DockerTransport.deploy needs and nothing else reads. The request
    hash names the container; the rand hash must not be here (the name shows in `docker ps`)."""
    spec = CHALLENGES[request.challenge]
    return {"challenge": request.challenge, "image": dev_image(request.challenge),
            "cpus": local.cpus, "memory_gib": local.memory_gib, "gpu": spec.is_gpu,
            "workers": local_workers(spec, local.cpus), "time_limit_s": seconds,
            "request_hash": request_hash(request)}
```

Change `payload`'s signature to `def payload(request: EvalRequest, workers: int | None = None) -> dict:` and its `"workers"` entry to `"workers": c3_workers(spec) if workers is None else workers,`. `request_hash` keeps calling `payload(request)` with no workers, so the hash is unchanged.

Replace `write_job_dir` with:

```python
def write_job_dir(job_dir: Path, request: EvalRequest, purpose: str,
                  local: LocalSettings | None = None) -> Path:
    """Writes the deploy directory, wiping `job_dir` first if it already exists. With `local`
    it is the local flavour: local.json instead of .c3, a job.sh that does not download the
    monorepo, and the local worker count in the payload."""
    job_dir = Path(job_dir)
    if job_dir.exists():
        shutil.rmtree(job_dir)
    (job_dir / "talos").mkdir(parents=True)
    spec = CHALLENGES[request.challenge]
    nonces = sum(n.count for n in request.training) + sum(n.count for n in request.holdout)
    if local is None:
        workers = c3_workers(spec)
        seconds = time_limit_s(max(nonces, 1), workers)
        (job_dir / ".c3").write_text(c3_config_text(request.challenge, purpose, seconds),
                                     encoding="utf-8", newline="\n")
        sh_text = job_sh_text(MONOREPO_REF)
    else:
        workers = local_workers(spec, local.cpus)
        seconds = time_limit_s(max(nonces, 1), workers)
        (job_dir / "local.json").write_text(
            json.dumps(local_settings_doc(request, local, seconds), indent=1),
            encoding="utf-8", newline="\n")
        sh_text = local_job_sh_text()
    sh = job_dir / "job.sh"
    sh.write_text(sh_text, encoding="utf-8", newline="\n")
    sh.chmod(sh.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    (job_dir / "payload.json").write_text(json.dumps(payload(request, workers), indent=1),
                                          encoding="utf-8", newline="\n")
    for mod in JOB_MODULES:
        shutil.copy2(_PKG / f"{mod}.py", job_dir / "talos" / f"{mod}.py")
    return job_dir
```

- [ ] **Step 4: Run to verify they pass**

Run: `python3 -m pytest tests/test_c3_jobdir.py tests/test_c3_job.py tests/test_c3_bench.py -q`
Expected: all PASS (the C3 flavour is byte-for-byte what it was)

- [ ] **Step 5: Commit**

```bash
git add talos/c3_jobdir.py tests/test_c3_jobdir.py
git commit -m "c3_jobdir: local flavour with local.json and a job.sh that skips the download"
```

---

### Task 4: The runner unstages a leftover candidate before staging

**Files:**
- Modify: `talos/c3_job.py:70-75` (the `try:` around `stage_algorithm`)
- Test: `tests/test_c3_job.py`

**Interfaces:**
- Produces: no new names. `c3_job.main` removes `tig-algorithms/src/<challenge>/talos_cand/` before staging.

- [ ] **Step 1: Write the failing test**

```python
def test_a_leftover_candidate_from_a_previous_job_is_removed_before_staging(tmp_path):
    mono, work, art = setup(tmp_path)
    left = mono / "tig-algorithms" / "src" / "knapsack" / "talos_cand"
    left.mkdir()
    (left / "extra.rs").write_text("fn stale() {}\n")
    mod_rs = mono / "tig-algorithms" / "src" / "knapsack" / "mod.rs"
    mod_rs.write_text(mod_rs.read_text() + "pub mod talos_cand;\n")
    c3_job.main(workdir=work, artifacts_dir=art, run=fake_run(), monorepo=mono,
                log=lambda *a: None)
    # mutation: on a persistent /app volume the previous candidate's extra file would be
    # compiled into this candidate
    assert not (left / "extra.rs").exists() and (left / "mod.rs").exists()
    assert mod_rs.read_text().count("pub mod talos_cand;") == 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_c3_job.py -k leftover -v`
Expected: FAIL on `assert not (left / "extra.rs").exists()`

- [ ] **Step 3: Implement**

In `talos/c3_job.py::main`, inside the existing `try:`, before `inside.stage_algorithm(...)`:

```python
        # A persistent checkout (the local backend's /app volume) still holds the previous
        # candidate; its files must not be compiled into this one. On C3 the dir never exists.
        inside.unstage_algorithm(monorepo, challenge, inside.ALGO_NAME)
```

- [ ] **Step 4: Run to verify it passes**

Run: `python3 -m pytest tests/test_c3_job.py -q`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add talos/c3_job.py tests/test_c3_job.py
git commit -m "c3_job: unstage a leftover candidate before staging, for persistent checkouts"
```

---

### Task 5: C3Bench takes local settings, a rate, and a transport label

**Files:**
- Modify: `talos/c3_bench.py` (constructor, `evaluate`, `_submit`, `_collect`, every message that says "C3")
- Test: `tests/test_c3_bench.py`

**Interfaces:**
- Consumes: `LocalSettings`, `write_job_dir(..., local=)` from Task 3.
- Produces: `C3Bench(run_dir, pending=None, run=subprocess.run, clock=time.time, sleep=time.sleep, poll_s=20.0, pending_timeout_s=1800, poll_failures_max=15, api_key=None, transport=None, local: LocalSettings | None = None, usd_per_hour: float | None = None)`. Attribute `subdir` is `"local"` when `local` is set, else `"c3"`. Messages use `transport.label` when the transport has one, else `"C3"`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_c3_bench.py` (extend imports with `from talos.c3_jobdir import LocalSettings`):

```python
def test_local_settings_select_the_local_subdir_flavour_and_a_zero_rate(tmp_path):
    t = FakeTransport(["PENDING", "RUNNING", "RUNNING", "SUCCEEDED"], results_doc())
    b = C3Bench(tmp_path, pending=PendingJobStore.memory(), run=_no_cli, transport=t,
                clock=Clock(), sleep=lambda s: None, local=LocalSettings(8, 12),
                usd_per_hour=0.0)
    r = b.evaluate(req())
    assert r.compile.ok
    job_dir = Path([c[1] for c in t.calls if c[0] == "deploy"][0])
    # mutation: writing the C3 flavour under runs/<id>/c3 on the local backend
    assert job_dir.parent.name == "local" and (job_dir / "local.json").exists()
    assert not (job_dir / ".c3").exists()
    assert json.loads((job_dir / "local.json").read_text())["cpus"] == 8
    # mutation: `usd_per_hour or GBP_RATE` treats 0.0 as "not given" and bills local time
    assert b.cost_mark() == 0.0


def test_a_given_rate_is_charged_and_none_keeps_the_c3_rate(tmp_path):
    t = FakeTransport(["RUNNING", "RUNNING", "SUCCEEDED"], results_doc())
    clock = Clock()

    def sleep(s):
        clock.t += s
    b = C3Bench(tmp_path, pending=PendingJobStore.memory(), run=_no_cli, transport=t,
                clock=clock, sleep=sleep, poll_s=20.0, usd_per_hour=3.6)
    b.evaluate(req())
    # RUNNING first seen at t=0, terminal at t=40: 40 s at $3.6/h = $0.04
    assert abs(b.cost_mark() - 0.04) < 1e-9


def test_messages_name_the_transport_label_when_it_has_one(tmp_path):
    t = FakeTransport(["PENDING"])
    t.label = "local Docker"
    t.statuses = ["FAILED"]
    with pytest.raises(BenchUnavailable) as ei:
        tbench(tmp_path, t).evaluate(req())
    # mutation: a local failure that tells the user to check C3
    assert "local Docker job failed twice" in str(ei.value) and "C3" not in str(ei.value)
```

- [ ] **Step 2: Run to verify they fail**

Run: `python3 -m pytest tests/test_c3_bench.py -k "local_settings or given_rate or transport_label" -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'local'`

- [ ] **Step 3: Implement**

In `talos/c3_bench.py`:

Import: `from talos.c3_jobdir import LocalSettings, request_hash, write_job_dir`.

Constructor: add the two parameters after `transport` and, after `self._t = ...`:

```python
        self.local = local
        self.subdir = "local" if local is not None else "c3"
        self.usd_per_hour = usd_per_hour
        # Error messages name the backend the user chose. Transports without a label are C3's.
        self._label = getattr(self._t, "label", "C3")
```

`evaluate`: `job_dir = self.run_dir / self.subdir / purpose`.

`_submit`: `write_job_dir(job_dir, request, purpose, local=self.local)` and the pending record's `"backend": self.subdir`.

`_collect`: replace the cost line with

```python
            rate = (GBP_PER_HOUR[profile] * USD_PER_GBP if self.usd_per_hour is None
                    else self.usd_per_hour)
            self._cost += (t_end - t_run) / 3600 * rate
```

Every f-string that begins `"C3 ` or contains `no C3 capacity` uses `{self._label}` instead: `f"{self._label} unreachable for ..."`, `f"no {self._label} capacity for ..."`, `f"{self._label} job failed twice: ..."`, `f"{self._label} deploy failed: ..."`, `f"{self._label} pull failed for ..."`, `f"{self._label} job {job_id} succeeded without results.json"`, `f"{self._label} job {job_id} timed out before writing results"`, `f"{self._label} job {job_id} wrote unreadable results: ..."`. The `_safe_job_id` message is a module function and stays as it is.

- [ ] **Step 4: Run to verify they pass**

Run: `python3 -m pytest tests/test_c3_bench.py -q`
Expected: all PASS, including the untouched `no C3 capacity` assertion (FakeTransport has no `label`).

- [ ] **Step 5: Commit**

```bash
git add talos/c3_bench.py tests/test_c3_bench.py
git commit -m "c3_bench: local settings, an injectable hourly rate, and transport-labelled messages"
```

---

### Task 6: DockerTransport

**Files:**
- Create: `talos/local_transport.py`
- Test: `tests/test_local_transport.py`

**Interfaces:**
- Consumes: `C3CommandError` from `talos/c3_bench.py`; `argv0` from `talos/executables.py`; `DEV_IMAGE_TAG`, `MONOREPO_REF` from `talos/challenges.py`.
- Produces (all in `talos/local_transport.py`):
  - `host_uid() -> str | None` — `"uid:gid"` on POSIX, `None` on Windows.
  - `volume_key() -> str`, `volume_names(challenge: str) -> tuple[str, str]` (app volume, cargo volume).
  - `container_name(run_dir: Path, purpose: str, request_hash: str) -> str` and `run_key(run_dir: Path) -> str`.
  - `parse_time(s: str) -> datetime` (UTC-aware).
  - `run_args(local: dict, job_dir: Path, artifacts: Path, name: str, key: str, cargo_home: str, rustup_home: str, uid: str | None) -> list[str]`.
  - `class DockerTransport(run=subprocess.run, now=None, uid=None)` with `name = "docker"`, `label = "local Docker"`, and methods `docker(*args, timeout=600, ok=(0,)) -> str`, `deploy`, `status`, `cancel`, `fetch`, `image_present(image) -> bool`, `volume_exists(vol) -> bool`, `volume_label(vol, key) -> str`. `whoami` and `balance_gbp` raise `NotImplementedError`.
  - Constants `PIDS_LIMIT = 4096`, `APP = "/app"`, `WORK = "/work"`, `ARTIFACTS = "/artifacts"`, `LABEL_LIMIT = "talos.time_limit_s"`, `LABEL_RUN = "talos.run"`, `LABEL_CARGO = "talos.cargo_home"`, `LABEL_RUSTUP = "talos.rustup_home"`.

- [ ] **Step 1: Write the failing tests**

`tests/test_local_transport.py`:

```python
import json
import os
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from talos.c3_bench import C3CommandError
from talos.challenges import DEV_IMAGE_TAG, MONOREPO_REF, dev_image
from talos.local_transport import (DockerTransport, PIDS_LIMIT, container_name, host_uid,
                                   parse_time, run_args, run_key, volume_key, volume_names)

T0 = datetime(2026, 9, 23, 10, 0, 0, tzinfo=timezone.utc)


def local_doc(gpu=False, seconds=1800):
    return {"challenge": "hypergraph" if gpu else "knapsack",
            "image": dev_image("hypergraph" if gpu else "knapsack"), "cpus": 8, "memory_gib": 12,
            "gpu": gpu, "workers": 1 if gpu else 8, "time_limit_s": seconds,
            "request_hash": "ab12" * 4}


def job_dir(tmp_path, gpu=False, seconds=1800):
    d = tmp_path / "runs" / "job1" / "local" / "3"
    d.mkdir(parents=True)
    (d / "local.json").write_text(json.dumps(local_doc(gpu, seconds)), encoding="utf-8")
    return d


class FakeDocker:
    """Scripted `docker` CLI. `inspect` is the State document returned; None = no container."""

    def __init__(self, inspect=None, run_rc=0, volume_labels=None, ps_ids=""):
        self.inspect = inspect
        self.run_rc = run_rc
        self.volume_labels = volume_labels or {}
        self.ps_ids = ps_ids
        self.calls = []

    def __call__(self, cmd, **kw):
        self.calls.append(cmd)
        rc, out = 0, ""
        word = cmd[1]
        if word == "run":
            rc, out = self.run_rc, "0123456789abcdef" * 4 + "\n"
        elif word == "inspect":
            if self.inspect is None:
                rc, out = 1, "Error: No such object"
            else:
                out = json.dumps([{"State": self.inspect,
                                   "Config": {"Labels": {"talos.time_limit_s": "1800"}}}])
        elif word == "ps":
            out = self.ps_ids
        elif word == "volume" and cmd[2] == "inspect":
            label = cmd[-1]
            fmt = [a for a in cmd if a.startswith("{{")]
            key = fmt[0].split('"')[1] if fmt else ""
            out = self.volume_labels.get((label, key), "")
            if label not in {v for v, _ in self.volume_labels}:
                rc, out = 1, "Error: No such volume"
        elif word == "image" and cmd[2] == "inspect":
            rc = 0 if "present" in cmd[-1] else 1
        elif word in ("rm", "kill"):
            rc = 0 if self.inspect is not None else 1
        return types.SimpleNamespace(returncode=rc, stdout=out, stderr="" if rc == 0 else out)


def state(status, exit_code=0, started=T0, finished=None):
    zero = "0001-01-01T00:00:00Z"
    return {"Status": status, "Running": status == "running", "ExitCode": exit_code,
            "StartedAt": started.isoformat().replace("+00:00", "Z"),
            "FinishedAt": zero if finished is None else
            finished.isoformat().replace("+00:00", "Z")}


def transport(fake, now=None, uid="1000:1000"):
    return DockerTransport(run=fake, now=now or (lambda: T0 + timedelta(seconds=60)), uid=uid)


def test_parse_time_handles_nanoseconds_and_the_zero_sentinel():
    assert parse_time("2026-09-23T10:00:00.123456789Z") == T0 + timedelta(microseconds=123456)
    assert parse_time("2026-09-23T10:00:00Z") == T0
    assert parse_time("0001-01-01T00:00:00Z").year == 1


def test_volume_and_container_names_are_keyed_on_the_pins_and_the_run(tmp_path):
    key = volume_key()
    assert len(key) == 12 and volume_names("knapsack") == (f"talos-app-knapsack-{key}",
                                                           f"talos-cargo-knapsack-{key}")
    # mutation: a key that ignores either pin reuses a target dir built against another monorepo
    assert MONOREPO_REF[:12] != key and DEV_IMAGE_TAG not in key
    a, b = tmp_path / "runs" / "a", tmp_path / "runs" / "b"
    assert container_name(a, "3", "ab12" * 4) != container_name(b, "3", "ab12" * 4)
    assert container_name(a, "3", "ab12" * 4) == f"talos-{run_key(a)}-3-{'ab12' * 4}"


def test_run_args_carry_every_hardening_flag_and_the_mounts(tmp_path):
    jd, art = tmp_path / "job dir", tmp_path / "job dir" / "n" / "artifacts"
    args = run_args(local_doc(), jd, art, "n", "runkey12", "/usr/local/cargo",
                    "/usr/local/rustup", "1000:1000")
    assert args[:4] == ["docker", "run", "-d", "--name"] and args[4] == "n"
    joined = " ".join(args)
    for flag in ("--network none", "--cap-drop ALL", "--security-opt no-new-privileges",
                 f"--pids-limit {PIDS_LIMIT}", "--cpus 8", "--memory 12g", "--user 1000:1000",
                 "-e C3_JOB_WORKDIR=/work", "-e C3_ARTIFACTS_DIR=/artifacts", "-e HOME=/tmp",
                 "-e CARGO_HOME=/usr/local/cargo", "-e RUSTUP_HOME=/usr/local/rustup",
                 "--label talos.time_limit_s=1800", "--label talos.run=runkey12"):
        # mutation: any one hardening flag dropped
        assert flag in joined, flag
    app, cargo = volume_names("knapsack")
    assert f"{app}:/app" in args and f"{cargo}:/usr/local/cargo" in args
    assert f"{jd}:/work:ro" in args and f"{art}:/artifacts" in args
    assert args[-3:] == [dev_image("knapsack"), "bash", "/work/job.sh"]
    assert "--gpus" not in args
    gpu = run_args(local_doc(gpu=True), jd, art, "n", "k", "/c", "/r", None)
    assert "--gpus" in gpu and gpu[gpu.index("--gpus") + 1] == "all"
    # mutation: `--user None` on Windows
    assert "--user" not in gpu and "None" not in gpu


def test_run_args_keeps_a_path_with_a_space_as_one_argument(tmp_path):
    jd = tmp_path / "my runs" / "j"
    args = run_args(local_doc(), jd, jd / "n" / "artifacts", "n", "k", "/c", "/r", None)
    assert args[args.index("-v") + 1] == f"{jd}:/work:ro"


def test_host_uid_is_uid_gid_on_posix_and_none_on_windows(monkeypatch):
    if os.name == "nt":
        assert host_uid() is None
    else:
        assert host_uid() == f"{os.getuid()}:{os.getgid()}"


def test_deploy_runs_the_container_named_from_the_job_dir_and_returns_the_name(tmp_path):
    fake = FakeDocker(inspect=None, volume_labels={
        (volume_names("knapsack")[1], "talos.cargo_home"): "/usr/local/cargo\n",
        (volume_names("knapsack")[1], "talos.rustup_home"): "/usr/local/rustup\n"})
    jd = job_dir(tmp_path)
    name = transport(fake).deploy(jd)
    run_dir = tmp_path / "runs" / "job1"
    assert name == container_name(run_dir, "3", "ab12" * 4)
    run_cmd = [c for c in fake.calls if c[1] == "run"][0]
    assert run_cmd[run_cmd.index("--name") + 1] == name
    # mutation: the artifacts dir left for Docker to create is root-owned on Linux
    assert (jd / name / "artifacts").is_dir()
    assert f"{jd / name / 'artifacts'}:/artifacts" in run_cmd
    assert "-e" in run_cmd and "CARGO_HOME=/usr/local/cargo" in run_cmd
    # mutation: a stale container with this name makes `docker run` fail with a name clash
    assert ["rm", "-f", name] in [c[1:] for c in fake.calls]


def test_deploy_prunes_exited_containers_of_the_same_run_first(tmp_path):
    fake = FakeDocker(inspect=None, ps_ids="aaa\nbbb\n", volume_labels={
        (volume_names("knapsack")[1], "talos.cargo_home"): "/c",
        (volume_names("knapsack")[1], "talos.rustup_home"): "/r"})
    jd = job_dir(tmp_path)
    transport(fake).deploy(jd)
    ps = [c for c in fake.calls if c[1] == "ps"][0]
    key = run_key(tmp_path / "runs" / "job1")
    # mutation: pruning every exited container on the machine, or none
    assert f"label={'talos.run'}={key}" in ps and "status=exited" in ps
    assert ["rm", "aaa", "bbb"] in [c[1:] for c in fake.calls]


def test_deploy_without_the_cargo_volume_is_a_command_error_not_a_crash(tmp_path):
    fake = FakeDocker(inspect=None)
    with pytest.raises(C3CommandError):
        transport(fake).deploy(job_dir(tmp_path))


@pytest.mark.parametrize("st, expect", [
    (state("created"), "PENDING"),
    (state("running"), "RUNNING"),
    (state("exited", 0, finished=T0 + timedelta(seconds=30)), "SUCCEEDED"),
    (state("exited", 1, finished=T0 + timedelta(seconds=30)), "FAILED"),
    (state("exited", 137, finished=T0 + timedelta(seconds=1800)), "TIMED_OUT"),
    (state("exited", 0, finished=T0 + timedelta(seconds=1801)), "TIMED_OUT"),
])
def test_status_maps_docker_state_onto_the_c3_vocabulary(st, expect):
    fake = FakeDocker(inspect=st)
    assert transport(fake).status("n") == expect
    # mutation: killing a container that has already exited
    assert not any(c[1] == "kill" for c in fake.calls)


def test_status_kills_a_running_container_past_its_limit_and_reports_timed_out():
    fake = FakeDocker(inspect=state("running"))
    t = transport(fake, now=lambda: T0 + timedelta(seconds=1800))
    assert t.status("n") == "TIMED_OUT"
    assert ["kill", "n"] in [c[1:] for c in fake.calls]
    # mutation: a limit read from a hard-coded constant instead of the container's label
    fake2 = FakeDocker(inspect=state("running"))
    assert transport(fake2, now=lambda: T0 + timedelta(seconds=1799)).status("n") == "RUNNING"


def test_status_on_a_missing_container_is_a_command_error():
    with pytest.raises(C3CommandError):
        transport(FakeDocker(inspect=None)).status("n")


def test_cancel_removes_the_container_and_tolerates_a_missing_one():
    fake = FakeDocker(inspect=state("running"))
    transport(fake).cancel("n")
    assert ["rm", "-f", "n"] in [c[1:] for c in fake.calls]
    transport(FakeDocker(inspect=None)).cancel("n")  # no raise


def test_fetch_reports_the_file_on_the_bind_mount_and_runs_nothing(tmp_path):
    fake = FakeDocker()
    dest = tmp_path / "n" / "artifacts" / "results.json"
    assert transport(fake).fetch("n", "results.json", dest) is False
    dest.parent.mkdir(parents=True)
    dest.write_text("{}")
    assert transport(fake).fetch("n", "results.json", dest) is True
    assert fake.calls == []


def test_whoami_and_balance_are_not_part_of_the_local_path():
    t = transport(FakeDocker())
    with pytest.raises(NotImplementedError):
        t.whoami()
    with pytest.raises(NotImplementedError):
        t.balance_gbp()


def test_a_missing_docker_binary_or_a_hang_is_a_command_error():
    def missing(cmd, **kw):
        raise FileNotFoundError("docker")
    with pytest.raises(C3CommandError):
        DockerTransport(run=missing).docker("info")
    import subprocess

    def hung(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, kw.get("timeout"))
    with pytest.raises(C3CommandError):
        DockerTransport(run=hung).docker("info")
```

- [ ] **Step 2: Run to verify they fail**

Run: `python3 -m pytest tests/test_local_transport.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'talos.local_transport'`

- [ ] **Step 3: Implement `talos/local_transport.py`**

```python
"""Docker transport for the local backend: one detached container per evaluate call, driven
through the `docker` CLI. It satisfies talos.c3_transport.C3Transport, so C3Bench runs a local
job exactly as it runs a C3 one. Every subprocess call goes through the injected runner.

The flags in run_args are the sandbox for LLM-authored code on the user's machine. Loosening
them is a human decision (AGENTS.md, "What requires a human")."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from talos.c3_bench import C3CommandError
from talos.challenges import DEV_IMAGE_TAG, MONOREPO_REF
from talos.executables import argv0

APP = "/app"
WORK = "/work"
ARTIFACTS = "/artifacts"
PIDS_LIMIT = 4096
LABEL_LIMIT = "talos.time_limit_s"
LABEL_RUN = "talos.run"
LABEL_CARGO = "talos.cargo_home"
LABEL_RUSTUP = "talos.rustup_home"


def host_uid() -> str | None:
    """`uid:gid` for `docker run --user`, so files on the bind mounts belong to the user. None
    on Windows: there is no getuid, and Docker Desktop maps bind-mount ownership itself."""
    if os.name == "nt":
        return None
    return f"{os.getuid()}:{os.getgid()}"


def volume_key() -> str:
    """Both pins, so a pin bump gets fresh volumes and never reuses a target directory built
    against another monorepo (AGENTS.md invariant 3)."""
    return hashlib.sha256(f"{MONOREPO_REF}\0{DEV_IMAGE_TAG}".encode()).hexdigest()[:12]


def volume_names(challenge: str) -> tuple[str, str]:
    key = volume_key()
    return f"talos-app-{challenge}-{key}", f"talos-cargo-{challenge}-{key}"


def run_key(run_dir: Path) -> str:
    return hashlib.sha256(str(Path(run_dir).resolve()).encode()).hexdigest()[:8]


def container_name(run_dir: Path, purpose: str, request_hash: str) -> str:
    """Also the job id C3Bench records, so it must match c3_bench._JOB_ID_RE. The request hash
    is derived from a hash of the rand hash, never the rand hash itself (c3_jobdir.request_hash),
    so the name is safe to show in `docker ps`."""
    return f"talos-{run_key(run_dir)}-{purpose}-{request_hash}"


def parse_time(s: str) -> datetime:
    """Docker's RFC 3339 with nanoseconds; Python's fromisoformat takes at most microseconds."""
    s = s.rstrip("Z")
    if "." in s:
        head, frac = s.split(".", 1)
        s = f"{head}.{frac[:6]}"
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def run_args(local: dict, job_dir: Path, artifacts: Path, name: str, key: str, cargo_home: str,
             rustup_home: str, uid: str | None) -> list[str]:
    """The job container's argv. Each `-v` value is one element, so a path with a space stays
    whole. The cargo and rustup homes are passed explicitly because HOME is overridden."""
    app_vol, cargo_vol = volume_names(local["challenge"])
    args = ["docker", "run", "-d", "--name", name,  # [0] is dropped by DockerTransport.docker
            "--label", f"{LABEL_LIMIT}={local['time_limit_s']}", "--label", f"{LABEL_RUN}={key}",
            "-v", f"{job_dir}:{WORK}:ro", "-v", f"{artifacts}:{ARTIFACTS}",
            "-v", f"{app_vol}:{APP}", "-v", f"{cargo_vol}:{cargo_home}",
            "-e", f"C3_JOB_WORKDIR={WORK}", "-e", f"C3_ARTIFACTS_DIR={ARTIFACTS}",
            "-e", "HOME=/tmp", "-e", f"CARGO_HOME={cargo_home}", "-e", f"RUSTUP_HOME={rustup_home}",
            "--cpus", str(local["cpus"]), "--memory", f"{local['memory_gib']}g",
            "--network", "none", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--pids-limit", str(PIDS_LIMIT)]
    if uid:
        args += ["--user", uid]
    if local["gpu"]:
        args += ["--gpus", "all"]
    return args + [local["image"], "bash", f"{WORK}/job.sh"]


class DockerTransport:
    name = "docker"
    label = "local Docker"

    def __init__(self, run: Callable = subprocess.run,
                 now: Callable[[], datetime] | None = None, uid: str | None = None):
        self._run = run
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._uid = uid

    def docker(self, *args: str, timeout: int = 600, ok: tuple[int, ...] = (0,)) -> str:
        """One `docker` call. A missing binary, a hang, or an exit code outside `ok` is a
        C3CommandError, which C3Bench treats as a poll failure or a failed deploy, never a
        traceback. Never echoes argv: a bind-mount path is fine, but the habit is what keeps
        the rand hash out of messages elsewhere."""
        try:
            r = self._run([argv0("docker"), *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)
        except OSError as e:
            raise C3CommandError(f"docker {args[0]} could not be run: {str(e)[:200]}") from None
        except subprocess.TimeoutExpired:
            raise C3CommandError(f"docker {args[0]} timed out after {timeout}s") from None
        if r.returncode not in ok:
            raise C3CommandError(f"docker {args[0]} failed ({r.returncode}): "
                                 f"{(r.stderr or r.stdout)[-500:]}")
        return r.stdout

    # ── C3Transport ────────────────────────────────────────────────────
    def whoami(self) -> dict:
        raise NotImplementedError("the local backend has no account")

    def balance_gbp(self) -> float | None:
        raise NotImplementedError("the local backend has no balance")

    def deploy(self, job_dir: Path) -> str:
        job_dir = Path(job_dir).resolve()
        local = json.loads((job_dir / "local.json").read_text(encoding="utf-8"))
        run_dir = job_dir.parent.parent  # runs/<job_id>/local/<purpose>
        key = run_key(run_dir)
        name = container_name(run_dir, job_dir.name, local["request_hash"])
        _, cargo_vol = volume_names(local["challenge"])
        cargo_home = self.volume_label(cargo_vol, LABEL_CARGO)
        rustup_home = self.volume_label(cargo_vol, LABEL_RUSTUP)
        artifacts = job_dir / name / ARTIFACTS.strip("/")
        artifacts.mkdir(parents=True, exist_ok=True)  # else Docker creates it root-owned
        self._prune(key)
        self.docker("rm", "-f", name, ok=(0, 1))  # a leftover with this exact name
        self.docker(*run_args(local, job_dir, artifacts, name, key, cargo_home, rustup_home,
                              self._uid)[1:])
        return name

    def _prune(self, key: str) -> None:
        """Exited containers of earlier iterations of this run. Only exited ones: a resumed
        process may still be about to reattach to the newest, and only this run's."""
        ids = self.docker("ps", "-aq", "--filter", f"label={LABEL_RUN}={key}",
                          "--filter", "status=exited").split()
        if ids:
            self.docker("rm", *ids, ok=(0, 1))

    def _inspect(self, job_id: str) -> dict:
        try:
            return json.loads(self.docker("inspect", job_id))[0]
        except (ValueError, IndexError, TypeError) as e:
            raise C3CommandError(f"docker inspect returned no document: {e}") from None

    def status(self, job_id: str) -> str:
        doc = self._inspect(job_id)
        st = doc["State"]
        limit = int(doc["Config"]["Labels"][LABEL_LIMIT])
        status = st.get("Status")
        if status == "created":
            return "PENDING"
        started = parse_time(st["StartedAt"])
        if status == "running":
            if (self._now() - started).total_seconds() >= limit:
                self.docker("kill", job_id, ok=(0, 1))
                return "TIMED_OUT"
            return "RUNNING"
        if status in ("exited", "dead"):
            if (parse_time(st["FinishedAt"]) - started).total_seconds() >= limit:
                return "TIMED_OUT"  # the one we, or a previous process, killed
            return "SUCCEEDED" if st.get("ExitCode") == 0 else "FAILED"
        raise C3CommandError(f"unexpected container status {status!r}")

    def cancel(self, job_id: str) -> None:
        try:
            self.docker("rm", "-f", job_id)
        except C3CommandError:
            pass  # best effort: it may already be gone

    def fetch(self, job_id: str, name: str, dest: Path) -> bool:
        return Path(dest).exists()  # the artifacts directory is a bind mount

    # ── helpers for prepare ────────────────────────────────────────────
    def image_present(self, image: str) -> bool:
        try:
            self.docker("image", "inspect", image)
            return True
        except C3CommandError:
            return False

    def volume_exists(self, volume: str) -> bool:
        try:
            self.docker("volume", "inspect", volume)
            return True
        except C3CommandError:
            return False

    def volume_label(self, volume: str, key: str) -> str:
        out = self.docker("volume", "inspect", "--format", f'{{{{index .Labels "{key}"}}}}',
                          volume).strip()
        if not out:
            raise C3CommandError(f"volume {volume} has no {key} label; run `talos run` again "
                                 f"so prepare can recreate it")
        return out
```

Note the `[1:]` in `deploy`: `run_args` returns argv with a literal `docker` first so tests can assert on the whole command on every OS, and `docker()` prepends the real binary (`argv0`, a full path on Windows) itself.

- [ ] **Step 4: Run to verify they pass**

Run: `python3 -m pytest tests/test_local_transport.py -v && python3 -m ruff check talos/local_transport.py`
Expected: all PASS, ruff clean

- [ ] **Step 5: Commit**

```bash
git add talos/local_transport.py tests/test_local_transport.py
git commit -m "local_transport: a Docker transport that satisfies the C3 transport protocol"
```

---

### Task 7: The prepare step and the runtime check

**Files:**
- Modify: `talos/local_transport.py` (append)
- Test: `tests/test_local_transport.py` (append)

**Interfaces:**
- Consumes: `DockerTransport` helpers from Task 6; `CHALLENGES`, `dev_image`, `MONOREPO_REF` from challenges.
- Produces: `docker_runtimes(run=subprocess.run) -> list[str]` (raises `C3CommandError`); `has_gpu_runtime(run=subprocess.run) -> bool`; `prepare(challenge: str, run=subprocess.run, uid: str | None = None, log=print) -> str | None` returning the GPU name for a GPU challenge and `None` otherwise; `clone_script(uid) -> str`; `warm_script(challenge, uid) -> str`; constants `READY_MARKER = ".talos-ready"`, `WARM_MARKER = ".talos-warm"`, `MONOREPO_TARBALL`.

If the spike (Task 1) found `CARGO_HOME` or `RUSTUP_HOME` unset in the image, `prepare` records `$HOME/.cargo` and `$HOME/.rustup` as read from the image, which is what the `sh -c` lines below already do.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_local_transport.py` (extend the import with `docker_runtimes, has_gpu_runtime, prepare, READY_MARKER, WARM_MARKER`):

```python
class FakePrepareDocker:
    """`docker` for prepare: which image and volumes exist, which markers the /app volume has."""

    def __init__(self, image=False, volumes=(), markers=(), runtimes=("runc",), gpu_name="L40S"):
        self.image, self.volumes, self.markers = image, set(volumes), set(markers)
        self.runtimes, self.gpu_name = runtimes, gpu_name
        self.calls = []

    def __call__(self, cmd, **kw):
        self.calls.append(cmd)
        rc, out = 0, ""
        if cmd[1:3] == ["image", "inspect"]:
            rc = 0 if self.image else 1
        elif cmd[1] == "pull":
            self.image = True
        elif cmd[1:3] == ["volume", "inspect"]:
            vol = cmd[-1]
            if vol not in self.volumes:
                rc = 1
            elif "--format" in cmd:
                out = "/usr/local/cargo\n" if "cargo_home" in cmd[-2] else "/usr/local/rustup\n"
        elif cmd[1:3] == ["volume", "create"]:
            self.volumes.add(cmd[-1])
        elif cmd[1] == "info":
            out = json.dumps({r: {"path": r} for r in self.runtimes})
        elif cmd[1] == "run":
            script = cmd[-1]
            if cmd[-2] == "-c" and script.startswith("test -e"):
                rc = 0 if script.split("/")[-1] in self.markers else 1
            elif "nvidia-smi" in cmd:
                out = f"{self.gpu_name}\n"
            elif "echo ${CARGO_HOME" in script:
                out = "/usr/local/cargo\n"
            elif "echo ${RUSTUP_HOME" in script:
                out = "/usr/local/rustup\n"
            elif "codeload" in script:
                self.markers.add(READY_MARKER)
            elif "build_algorithm" in script:
                self.markers.add(WARM_MARKER)
        return types.SimpleNamespace(returncode=rc, stdout=out, stderr="" if rc == 0 else out)


def runs(fake):
    return [c for c in fake.calls if c[1] == "run"]


def test_prepare_from_nothing_pulls_creates_volumes_clones_and_warms(tmp_path):
    fake = FakePrepareDocker()
    lines = []
    assert prepare("knapsack", run=fake, uid="1000:1000", log=lines.append) is None
    app, cargo = volume_names("knapsack")
    assert ["pull", dev_image("knapsack")] in [c[1:] for c in fake.calls]
    create = [c for c in fake.calls if c[1:3] == ["volume", "create"]]
    # mutation: the cargo volume created without its labels leaves deploy unable to mount it
    assert any(c[-1] == cargo and "--label" in c
               and any(a.startswith("talos.cargo_home=/usr/local/cargo") for a in c)
               and any(a.startswith("talos.rustup_home=/usr/local/rustup") for a in c)
               for c in create)
    assert any(c[-1] == app for c in create)
    scripts = [c[-1] for c in runs(fake) if c[-2] == "-c"]
    clone = [s for s in scripts if "codeload" in s][0]
    warm = [s for s in scripts if "build_algorithm" in s][0]
    assert MONOREPO_REF in clone and "--strip-components=1" in clone
    # mutation: a warm build that builds nothing leaves the registry cold and the first job
    # fails with --network none
    assert "tig-algorithms/src/knapsack" in warm and "talos_cand" in warm
    assert f"touch /app/{READY_MARKER}" in clone and f"touch /app/{WARM_MARKER}" in warm
    # mutation: no chown, so the job container (host uid) cannot write /app
    assert "chown -R 1000:1000 /app /usr/local/cargo" in clone
    assert "chown -R 1000:1000 /app /usr/local/cargo" in warm
    # mutation: prepare's containers with --network none cannot download anything
    for c in runs(fake):
        assert "--network" not in c
    assert "--gpus" not in " ".join(" ".join(c) for c in fake.calls)
    assert any("pulling" in ln for ln in lines) and any("warm" in ln.lower() for ln in lines)


@pytest.mark.parametrize("have, absent_step", [
    ("image", "pull"), ("volumes", "create"), ("ready", "codeload"), ("warm", "build_algorithm")])
def test_prepare_skips_each_step_whose_marker_is_present(tmp_path, have, absent_step):
    app, cargo = volume_names("knapsack")
    fake = FakePrepareDocker(
        image=have in ("image", "volumes", "ready", "warm"),
        volumes=(app, cargo) if have in ("volumes", "ready", "warm") else (),
        markers={"ready": (READY_MARKER,), "warm": (READY_MARKER, WARM_MARKER)}.get(have, ()))
    prepare("knapsack", run=fake, uid=None, log=lambda *a: None)
    # mutation: a step that runs unconditionally re-clones (or re-pulls 13 GB) on every run
    assert not any(absent_step in " ".join(c) for c in fake.calls), absent_step


def test_prepare_without_a_uid_does_not_chown(tmp_path):
    fake = FakePrepareDocker()
    prepare("knapsack", run=fake, uid=None, log=lambda *a: None)
    assert not any("chown" in c[-1] for c in runs(fake))


def test_prepare_for_a_gpu_challenge_warms_with_the_gpu_and_returns_its_name(tmp_path):
    fake = FakePrepareDocker(runtimes=("runc", "nvidia"), gpu_name="NVIDIA L40S")
    assert prepare("hypergraph", run=fake, uid=None, log=lambda *a: None) == "NVIDIA L40S"
    warm = [c for c in runs(fake) if c[-2] == "-c" and "build_algorithm" in c[-1]][0]
    assert "--gpus" in warm
    smi = [c for c in runs(fake) if "nvidia-smi" in c][0]
    assert "--gpus" in smi and "--query-gpu=name" in smi


def test_docker_runtimes_and_the_gpu_check():
    assert docker_runtimes(FakePrepareDocker(runtimes=("runc", "nvidia"))) == ["nvidia", "runc"]
    assert has_gpu_runtime(FakePrepareDocker(runtimes=("runc", "nvidia"))) is True
    assert has_gpu_runtime(FakePrepareDocker(runtimes=("runc",))) is False

    def down(cmd, **kw):
        return types.SimpleNamespace(returncode=1, stdout="",
                                     stderr="Cannot connect to the Docker daemon")
    # mutation: a daemon that is down reported as "no GPU" instead of "no Docker"
    with pytest.raises(C3CommandError):
        docker_runtimes(down)
```

- [ ] **Step 2: Run to verify they fail**

Run: `python3 -m pytest tests/test_local_transport.py -k "prepare or runtimes" -v`
Expected: FAIL with `ImportError: cannot import name 'prepare'`

- [ ] **Step 3: Implement**

Append to `talos/local_transport.py` (add `from talos.challenges import CHALLENGES, DEV_IMAGE_TAG, MONOREPO_REF, dev_image` to the import):

```python
READY_MARKER = ".talos-ready"
WARM_MARKER = ".talos-warm"
MONOREPO_TARBALL = "https://codeload.github.com/tig-foundation/tig-monorepo/tar.gz/"


def docker_runtimes(run: Callable = subprocess.run) -> list[str]:
    """The runtimes the daemon reports, sorted. A daemon that is down raises C3CommandError,
    which setup turns into "install or start Docker"; it is never "no GPU"."""
    out = DockerTransport(run=run).docker("info", "--format", "{{json .Runtimes}}", timeout=60)
    try:
        return sorted(json.loads(out))
    except (ValueError, TypeError) as e:
        raise C3CommandError(f"docker info returned no runtimes: {e}") from None


def has_gpu_runtime(run: Callable = subprocess.run) -> bool:
    return "nvidia" in docker_runtimes(run)


def _chown(uid: str | None, cargo_home: str) -> str:
    return f"chown -R {uid} {APP} {cargo_home}\n" if uid else ""


def clone_script(uid: str | None, cargo_home: str) -> str:
    return ("set -euo pipefail\n"
            f"curl -fsSL \"{MONOREPO_TARBALL}{MONOREPO_REF}\" | tar xz -C {APP} "
            "--strip-components=1\n"
            f"touch {APP}/{READY_MARKER}\n" + _chown(uid, cargo_home))


def warm_script(challenge: str, uid: str | None, cargo_home: str) -> str:
    """Builds the first algorithm the pinned monorepo ships for the challenge, so the registry
    volume is populated with networking on, once, by code that is not LLM-authored."""
    return ("set -euo pipefail\n"
            f"cd {APP}\n"
            f"name=$(ls -d tig-algorithms/src/{challenge}/*/ | grep -v talos_cand | head -1 "
            "| xargs basename)\n"
            "build_algorithm \"$name\"\n"
            f"touch {APP}/{WARM_MARKER}\n" + _chown(uid, cargo_home))


def prepare(challenge: str, run: Callable = subprocess.run, uid: str | None = None,
            log: Callable[[str], None] = print) -> str | None:
    """Idempotent: pull the image, create the volumes, clone the pin, warm the registry. Each
    step is skipped when its marker is present, and one line is printed per step performed.
    Returns the GPU name for a GPU challenge, None otherwise. Every container here runs with
    networking on and as root; the chown at the end of each script hands the volumes to the
    host user for the job containers."""
    spec = CHALLENGES[challenge]
    image = dev_image(challenge)
    t = DockerTransport(run=run, uid=uid)
    gpu = ["--gpus", "all"] if spec.is_gpu else []
    if not t.image_present(image):
        log(f"pulling {image} (about 13 GB, once per image tag)")
        t.docker("pull", image, timeout=7200)
    app_vol, cargo_vol = volume_names(challenge)
    if not t.volume_exists(cargo_vol):
        cargo_home = t.docker("run", "--rm", image, "sh", "-c",
                              "echo ${CARGO_HOME:-$HOME/.cargo}").strip()
        rustup_home = t.docker("run", "--rm", image, "sh", "-c",
                               "echo ${RUSTUP_HOME:-$HOME/.rustup}").strip()
        t.docker("volume", "create", "--label", f"{LABEL_CARGO}={cargo_home}",
                 "--label", f"{LABEL_RUSTUP}={rustup_home}", cargo_vol)
        log(f"created volume {cargo_vol}")
    cargo_home = t.volume_label(cargo_vol, LABEL_CARGO)
    if not t.volume_exists(app_vol):
        t.docker("volume", "create", app_vol)
        log(f"created volume {app_vol}")
    mounts = ["-v", f"{app_vol}:{APP}", "-v", f"{cargo_vol}:{cargo_home}"]

    def marker(name: str) -> bool:
        try:
            t.docker("run", "--rm", *mounts, image, "sh", "-c", f"test -e {APP}/{name}",
                     timeout=120)
            return True
        except C3CommandError:
            return False

    if not marker(READY_MARKER):
        log(f"cloning tig-monorepo at {MONOREPO_REF[:12]} into {app_vol}")
        t.docker("run", "--rm", *mounts, image, "bash", "-c", clone_script(uid, cargo_home),
                 timeout=1800)
    if not marker(WARM_MARKER):
        log("warm build: one clean build so later builds are incremental and offline")
        t.docker("run", "--rm", *gpu, *mounts, image, "bash", "-c",
                 warm_script(challenge, uid, cargo_home), timeout=7200)
    if spec.is_gpu:
        out = t.docker("run", "--rm", *gpu, image, "nvidia-smi", "--query-gpu=name",
                       "--format=csv,noheader", timeout=120)
        return out.splitlines()[0].strip()
    return None
```

- [ ] **Step 4: Run to verify they pass**

Run: `python3 -m pytest tests/test_local_transport.py -v && python3 -m ruff check talos/local_transport.py`
Expected: all PASS, ruff clean

- [ ] **Step 5: Commit**

```bash
git add talos/local_transport.py tests/test_local_transport.py
git commit -m "local_transport: prepare (pull, volumes, clone, warm build) and the runtime check"
```

---

### Task 8: Config fields

**Files:**
- Modify: `talos/config.py`
- Test: `tests/test_cli.py` (config tests live there; see `test_config_without_backend_loads_as_modal`)

**Interfaces:**
- Produces: `Config.local_cpus: int | None = None`, `Config.local_memory_gib: int | None = None`, written only when set and read with `.get`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_cli.py` after `test_config_without_backend_loads_as_modal`:

```python
def test_config_round_trips_the_local_limits_and_omits_them_when_unset(tmp_path):
    save(tmp_path, Config(provider="anthropic", model="m", mode="single-shot", api_base=None,
                          backend="local", local_cpus=8, local_memory_gib=12), None)
    cfg = load(tmp_path)
    assert cfg.local_cpus == 8 and cfg.local_memory_gib == 12
    save(tmp_path, Config(provider="anthropic", model="m", mode="single-shot", api_base=None), None)
    # mutation: writing null keys makes a Modal config say something about a local container
    assert "local_cpus" not in json.loads((tmp_path / "talos.config.json").read_text())
    assert load(tmp_path).local_cpus is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_cli.py -k round_trips_the_local -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'local_cpus'`

- [ ] **Step 3: Implement**

In `talos/config.py`: add the fields after `backend`:

```python
    local_cpus: int | None = None       # local backend: the job container's --cpus
    local_memory_gib: int | None = None  # local backend: the job container's --memory, GiB
```

In `to_dict`, after the two pops: `return {k: v for k, v in d.items() if not (k.startswith("local_") and v is None)}`.

In `load`, add `local_cpus=d.get("local_cpus"), local_memory_gib=d.get("local_memory_gib"),` to the `Config(...)` call.

- [ ] **Step 4: Run to verify it passes**

Run: `python3 -m pytest tests/test_cli.py -k "config" -q`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add talos/config.py tests/test_cli.py
git commit -m "config: local_cpus and local_memory_gib for the local backend"
```

---

### Task 9: CLI wiring — setup, run, compile, budget, hardware class

**Files:**
- Modify: `talos/cli.py` (`BACKENDS`, imports, `make_bench`, `bench_hardware_class`, `cmd_setup`, `cmd_run` compute-budget block, `execute_job`, `compile_backend`/`cmd_compile`)
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `LocalSettings` (Task 3); `DockerTransport`, `host_uid`, `prepare`, `has_gpu_runtime`, `docker_runtimes` (Tasks 6, 7); `local_hardware_class` (Task 2); `Config.local_*` (Task 8).
- Produces: `BACKENDS = ("modal", "c3", "local")`; `LOCAL_DEFAULT_MEMORY_GIB = 8`; `default_local_memory_gib() -> int`; `check_local(run=None) -> bool` (raises `ConfigError`; returns whether the nvidia runtime is present); `local_settings(cfg: Config | None) -> LocalSettings`; `make_bench(backend, run_dir, pending, c3_api_key=None, local=None)`; `bench_hardware_class(backend, challenge, local=None, gpu_name=None, host=None)`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_cli.py`:

```python
def _local_docker(runtimes=("runc",), fail=False):
    def run(cmd, **kw):
        if fail:
            return types.SimpleNamespace(returncode=1, stdout="",
                                         stderr="Cannot connect to the Docker daemon")
        assert cmd[:2] == ["docker", "info"], cmd
        return types.SimpleNamespace(returncode=0, stdout=json.dumps({r: {} for r in runtimes}),
                                     stderr="")
    return run


def test_setup_local_asks_limits_checks_docker_and_writes_no_secret(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "validate_provider", lambda p: None)
    monkeypatch.setattr(cli, "deploy_bench",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("deployed")))
    monkeypatch.setattr(cli, "check_c3",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("c3")))
    monkeypatch.setattr(cli, "subprocess", types.SimpleNamespace(run=_local_docker(("runc", "nvidia"))))
    # prompts: backend, provider, model, api key, cpus, memory
    rc = cli.main(["setup"], ask=scripted(["local", "claude-cli", "", "single-shot", "6", "10"]))
    assert rc == 0
    cfg = load(tmp_path)
    assert cfg.backend == "local" and cfg.local_cpus == 6 and cfg.local_memory_gib == 10
    assert not (tmp_path / ".talos" / "secrets.json").exists()
    assert "GPU challenges: available" in capsys.readouterr().out


def test_setup_local_without_docker_writes_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "validate_provider", lambda p: None)
    monkeypatch.setattr(cli, "subprocess", types.SimpleNamespace(run=_local_docker(fail=True)))
    rc = cli.main(["setup"], ask=scripted(["local", "claude-cli", "", "single-shot", "", ""]))
    # mutation: ignoring the docker check writes a config whose first run fails at the baseline
    assert rc == 1 and "Docker" in capsys.readouterr().err
    assert not (tmp_path / "talos.config.json").exists()


def test_default_local_memory_floors_and_falls_back(monkeypatch):
    monkeypatch.setattr(cli, "_total_memory_gib", lambda: 6)
    assert cli.default_local_memory_gib() == 4
    monkeypatch.setattr(cli, "_total_memory_gib", lambda: 30)
    assert cli.default_local_memory_gib() == 26
    monkeypatch.setattr(cli, "_total_memory_gib", lambda: None)
    assert cli.default_local_memory_gib() == cli.LOCAL_DEFAULT_MEMORY_GIB


def test_make_bench_local_is_the_c3_bench_over_docker_and_the_local_class(tmp_path):
    from talos.bench import PendingJobStore
    from talos.c3_bench import C3Bench
    from talos.c3_jobdir import LocalSettings
    from talos.local_transport import DockerTransport
    b = cli.make_bench("local", tmp_path, PendingJobStore.memory(), local=LocalSettings(8, 12))
    assert isinstance(b, C3Bench) and isinstance(b._t, DockerTransport)
    # mutation: a local bench billing C3's hourly rate
    assert b.usd_per_hour == 0.0 and b.subdir == "local"
    cls = cli.bench_hardware_class("local", "knapsack", local=LocalSettings(8, 12), host="Box")
    assert cls == "local-box-cpu8-mem12"
    assert cli.bench_hardware_class("local", "hypergraph", local=LocalSettings(8, 12),
                                    gpu_name="NVIDIA L40S", host="box") == "local-box-gpu-nvidia-l40s"


def test_local_settings_come_from_the_config_or_the_machine(monkeypatch):
    monkeypatch.setattr(cli.os, "cpu_count", lambda: 16)
    monkeypatch.setattr(cli, "_total_memory_gib", lambda: 30)
    s = cli.local_settings(Config(provider="x", model="m", mode="single-shot", api_base=None,
                                  backend="local", local_cpus=6, local_memory_gib=10))
    assert (s.cpus, s.memory_gib) == (6, 10)
    # the agentic sandbox's `talos compile` has no config: the machine's defaults
    assert (cli.local_settings(None).cpus, cli.local_settings(None).memory_gib) == (16, 26)


def _local_config(root, cpus=8, memory=12):
    save(root, Config(provider="claude-cli", model="m", mode="single-shot", api_base=None,
                      backend="local", local_cpus=cpus, local_memory_gib=memory), None)


def test_run_local_skips_the_compute_budget_question_and_leaves_the_cap_unset(tmp_path,
                                                                                monkeypatch):
    monkeypatch.chdir(tmp_path)
    _local_config(tmp_path)
    monkeypatch.setattr(cli, "make_provider", lambda *a, **k: object())
    monkeypatch.setattr(cli, "fetch_challenge_info", lambda name: knapsack_info())
    _stub_mainnet(monkeypatch)
    monkeypatch.setattr(cli, "prepare", lambda ch, **k: None)
    monkeypatch.setattr(cli, "has_gpu_runtime", lambda: False)
    seen = {}

    class B(_RefusingBench):
        def evaluate(self, request):
            raise BenchCancelled("stop")

    monkeypatch.setattr(cli, "make_bench", lambda *a, **k: B("unreachable"))
    real = cli.execute_job

    def spy(spec, store, cfg, resume):
        seen["compute"] = spec.budget.compute_usd
        return real(spec, store, cfg, resume)
    monkeypatch.setattr(cli, "execute_job", spy)
    # prompts: direction, iteration budget, hours, [compute: skipped, an extra prompt would
    # raise "unexpected prompt"], mode, track, hyperparameters (blank = the default for each)
    rc = cli.main(["run", "--challenge", "knapsack"],
                  ask=scripted(["go", "1", "1", "", "", ""]))
    assert rc == 1 and seen["compute"] is None
    # --yes must not apply DEFAULT_COMPUTE_USD either
    rc = cli.main(["run", "--challenge", "knapsack", "--direction", "go",
                   "--budget-iterations", "1", "--yes"])
    assert rc == 1 and seen["compute"] is None


def test_run_local_prepares_before_the_baseline_and_exports_the_backend(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TALOS_BACKEND", raising=False)
    _local_config(tmp_path)
    monkeypatch.setattr(cli, "make_provider", lambda *a, **k: object())
    monkeypatch.setattr(cli, "fetch_challenge_info", lambda name: knapsack_info())
    _stub_mainnet(monkeypatch)
    order = []
    monkeypatch.setattr(cli, "prepare", lambda ch, **k: order.append(("prepare", ch)))

    class B(_RefusingBench):
        def evaluate(self, request):
            order.append(("evaluate", os.environ.get("TALOS_BACKEND")))
            raise BenchCancelled("stop")

    monkeypatch.setattr(cli, "make_bench", lambda *a, **k: B("unreachable"))
    rc = cli.main(["run", "--challenge", "knapsack", "--direction", "go",
                   "--budget-iterations", "1", "--yes"])
    # mutation: preparing after the baseline, or not at all
    assert rc == 1 and order == [("prepare", "knapsack"), ("evaluate", "local")]


def test_run_local_reports_a_docker_failure_and_fails_the_job(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    _local_config(tmp_path)
    monkeypatch.setattr(cli, "make_provider", lambda *a, **k: object())
    monkeypatch.setattr(cli, "fetch_challenge_info", lambda name: knapsack_info())
    _stub_mainnet(monkeypatch)
    monkeypatch.setattr(cli, "prepare", lambda ch, **k: (_ for _ in ()).throw(
        C3CommandError("docker info could not be run: [Errno 2] No such file")))
    monkeypatch.setattr(cli, "make_bench", lambda *a, **k: _RefusingBench("no bench"))
    rc = cli.main(["run", "--challenge", "knapsack", "--direction", "go",
                   "--budget-iterations", "1", "--yes"])
    assert rc == 1 and "Docker" in capsys.readouterr().err
    st = json.loads(next((tmp_path / "runs").glob("*/state.json")).read_text())
    # mutation: a run that stops here left at its initial status shows as live for ever
    assert st["status"] == "failed" and "docker" in st["stop_reason"].lower()


def test_run_local_refuses_a_gpu_challenge_without_the_nvidia_runtime(tmp_path, monkeypatch,
                                                                       capsys):
    monkeypatch.chdir(tmp_path)
    _local_config(tmp_path)
    monkeypatch.setattr(cli, "make_provider", lambda *a, **k: object())
    monkeypatch.setattr(cli, "fetch_challenge_info", lambda name: knapsack_info())
    _stub_mainnet(monkeypatch)
    monkeypatch.setattr(cli, "has_gpu_runtime", lambda: False)
    monkeypatch.setattr(cli, "prepare",
                        lambda ch, **k: (_ for _ in ()).throw(AssertionError("prepared")))
    monkeypatch.setattr(cli, "make_bench", lambda *a, **k: _RefusingBench("no bench"))
    rc = cli.main(["run", "--challenge", "hypergraph", "--direction", "go",
                   "--budget-iterations", "1", "--yes"])
    err = capsys.readouterr().err
    assert rc == 1 and "NVIDIA" in err and "modal" in err and "c3" in err
    st = json.loads(next((tmp_path / "runs").glob("*/state.json")).read_text())
    assert st["status"] == "failed"


def test_compile_local_prepares_then_evaluates(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "algorithm").mkdir()
    (tmp_path / "algorithm" / "mod.rs").write_text("fn x(){}")
    order = []
    monkeypatch.setattr(cli, "prepare", lambda ch, **k: order.append("prepare"))

    class B(_RefusingBench):
        def evaluate(self, request):
            order.append("evaluate")
            from talos.bench import EvalResult
            from talos.types import CompileResult
            return EvalResult(CompileResult(ok=True, artifact_id="a", output="ok"), [], None,
                              "not_won")

    monkeypatch.setattr(cli, "make_bench", lambda *a, **k: B("x"))
    rc = cli.main(["compile", "--challenge", "knapsack", "--backend", "local"])
    assert rc == 0 and order == ["prepare", "evaluate"]
```

`test_make_bench_picks_the_backend_and_hardware_class` needs no change: its `ConfigError` case uses `"aws"`, and no test asserts the `--backend` choices list. An explicit `--budget-compute-usd 0` stopping before the baseline is backend-independent and already pinned by `tests/test_loop.py::test_baseline_is_budget_checked_before_the_first_bench_call`; the local backend's zero spend does not change it (`0 >= 0`).

- [ ] **Step 2: Run to verify they fail**

Run: `python3 -m pytest tests/test_cli.py -k "local" -v`
Expected: FAIL (`AttributeError: module 'talos.cli' has no attribute 'prepare'` and the like)

- [ ] **Step 3: Implement**

In `talos/cli.py`:

Imports:

```python
import socket
from talos.c3_jobdir import LocalSettings
from talos.challenges import (CHALLENGES, MONOREPO_REF, c3_hardware_class, c3_image,
                              hardware_class, local_hardware_class)
from talos.local_transport import (DockerTransport, docker_runtimes, has_gpu_runtime, host_uid,
                                   prepare)
```

Constants: `BACKENDS = ("modal", "c3", "local")`, `LOCAL_DEFAULT_MEMORY_GIB = 8`, `LOCAL_MIN_MEMORY_GIB = 4`.

After `check_c3`:

```python
def _total_memory_gib() -> int | None:
    """Physical memory in GiB, or None where os.sysconf cannot say (Windows)."""
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30)
    except (AttributeError, ValueError, OSError):
        return None


def default_local_memory_gib() -> int:
    total = _total_memory_gib()
    if total is None:
        return LOCAL_DEFAULT_MEMORY_GIB
    return max(LOCAL_MIN_MEMORY_GIB, total - 4)


def local_settings(cfg: Config | None) -> LocalSettings:
    """The container limits: from the config, or the machine's own for a `talos compile` in
    the agentic sandbox, which has no config and scores no nonce (so no hardware class)."""
    cpus = (cfg.local_cpus if cfg and cfg.local_cpus else None) or os.cpu_count() or 1
    mem = (cfg.local_memory_gib if cfg and cfg.local_memory_gib else None) \
        or default_local_memory_gib()
    return LocalSettings(cpus=cpus, memory_gib=mem)


def check_local(run=None) -> bool:
    """Docker must answer; returns whether the nvidia runtime is present."""
    try:
        runtimes = docker_runtimes(run or subprocess.run)
    except C3CommandError as e:
        raise ConfigError(f"Docker check failed: {e}; install and start Docker "
                          f"(docker.com), then run `talos setup` again") from None
    return "nvidia" in runtimes
```

`make_bench`:

```python
def make_bench(backend: str, run_dir: Path, pending, c3_api_key: str | None = None,
               local: LocalSettings | None = None):
    if backend == "modal":
        from talos.bench import ModalBench
        return ModalBench()
    if backend == "c3":
        from talos.c3_bench import C3Bench
        return C3Bench(run_dir, pending=pending, api_key=c3_api_key)
    if backend == "local":
        from talos.c3_bench import C3Bench
        return C3Bench(run_dir, pending=pending, transport=DockerTransport(uid=host_uid()),
                       local=local or local_settings(None), usd_per_hour=0.0)
    raise ConfigError(f"unknown backend {backend!r}; run `talos setup`")


def bench_hardware_class(backend: str, challenge: str, local: LocalSettings | None = None,
                         gpu_name: str | None = None, host: str | None = None) -> str:
    spec = CHALLENGES[challenge]
    if backend == "local":
        return local_hardware_class(spec, local.cpus, local.memory_gib, gpu_name,
                                    host or socket.gethostname())
    return c3_hardware_class(spec) if backend == "c3" else hardware_class(spec)
```

`cmd_setup`: after the `if backend == "modal":` token prompts add

```python
    local = None
    if backend == "local":
        try:
            local = LocalSettings(
                cpus=_ask_number(ask, "CPUs for the local container", str(os.cpu_count() or 1),
                                 int),
                memory_gib=_ask_number(ask, "Memory for the local container in GiB",
                                       str(default_local_memory_gib()), int))
        except ConfigError as e:  # three non-numbers, as the run wizard treats it
            print(str(e), file=sys.stderr)
            return 2
``` In the `try:` block add a branch:

```python
        elif backend == "local":
            gpu = check_local()
            print("Local: jobs run in Docker on this machine. "
                  f"GPU challenges: {'available (nvidia runtime found)' if gpu else 'not available (no nvidia runtime)'}.")
```

and change `else: deploy_bench(...)` to `elif backend == "modal":`. In `save(...)` pass `local_cpus=local.cpus if local else None, local_memory_gib=local.memory_gib if local else None` in the `Config(...)`.

`cmd_run` compute-budget block: change the guard to `if budget.compute_usd is None and cfg.backend != "local":` and update the comment: `# spec §5.2: compute spend is always capped, except on the local backend where it is always zero.`

`execute_job`: replace the C3 image check block with

```python
        local = local_settings(cfg) if cfg.backend == "local" else None
        gpu_name = None
        if cfg.backend == "c3" and not image_available(spec.challenge):
            ...  # unchanged
        if cfg.backend == "local":
            if CHALLENGES[spec.challenge].is_gpu and not has_gpu_runtime():
                state.status, state.stop_reason = "failed", "no NVIDIA container runtime"
                store.save(state)
                print(f"{spec.challenge} needs a GPU and Docker reports no NVIDIA runtime; "
                      f"install the NVIDIA container toolkit, or run this challenge on the "
                      f"modal or c3 backend", file=sys.stderr)
                return 1
            try:
                # Before the baseline: the first run per challenge pulls the image and does one
                # clean build, and the user should see that happening rather than a silent wait.
                gpu_name = prepare(spec.challenge, uid=host_uid())
            except C3CommandError as e:
                state.status, state.stop_reason = "failed", f"docker: {str(e)[:120]}"
                store.save(state)
                print(f"Docker failed while preparing {spec.challenge}: {e}", file=sys.stderr)
                return 1
```

and `bench = make_bench(cfg.backend, store.run_dir, pending, c3_api_key=c3_api_key, local=local)`, `hardware = bench_hardware_class(cfg.backend, spec.challenge, local=local, gpu_name=gpu_name)`. Keep `os.environ["TALOS_BACKEND"] = ...` as it is.

`cmd_compile`: after `backend = compile_backend(...)` and `cfg = load(...)`:

```python
    if backend == "local":
        try:
            prepare(args.challenge, uid=host_uid())
        except C3CommandError as e:
            print(f"Docker failed while preparing {args.challenge}: {e}", file=sys.stderr)
            return 1
    bench = make_bench(backend, Path.cwd() / ".talos" / "compile", PendingJobStore.memory(),
                       c3_api_key=resolve_c3_api_key(cfg) if backend == "c3" else None,
                       local=local_settings(cfg) if backend == "local" else None)
```

- [ ] **Step 4: Run to verify they pass**

Run: `python3 -m pytest tests/test_cli.py -q && python3 -m ruff check talos/cli.py`
Expected: all PASS, ruff clean

- [ ] **Step 5: Commit**

```bash
git add talos/cli.py tests/test_cli.py
git commit -m "cli: the local backend in setup, run, compile, the budget and the hardware class"
```

---

### Task 10: Live test, docs, and the measured numbers

**Files:**
- Modify: `tests/test_live.py` (append), `AGENTS.md`, `README.md`, `docs/compute-backends.md`

**Interfaces:**
- Consumes: everything above.
- Produces: `tests/test_live.py::test_local_knapsack_job`; the documentation the spec's §8 lists; MEASURED build times.

- [ ] **Step 1: Add the live test**

Append to `tests/test_live.py`:

```python
def test_local_knapsack_job(tmp_path):
    """Manual: one real local Docker job. Run:
    TALOS_LIVE_BACKEND=local .venv/bin/pytest -m live tests/test_live.py -k local -s
    Needs Docker. Costs time, not money: the first run pulls a 13 GB image and does one clean
    build (about 8 minutes); the job itself is one incremental build plus 4 nonces."""
    if os.environ.get("TALOS_LIVE_BACKEND") != "local":
        pytest.skip("set TALOS_LIVE_BACKEND=local")
    import time
    from talos.bench import EvalRequest
    from talos.c3_bench import C3Bench
    from talos.c3_jobdir import LocalSettings
    from talos.challenges import CHALLENGES
    from talos.local_transport import DockerTransport, host_uid, prepare
    ch = os.environ.get("TALOS_LIVE_CHALLENGE", "knapsack")
    info = mainnet.fetch_challenge_info(ch)
    name, _algorithm_id, _adoption = mainnet.top_algorithm(ch)
    files = mainnet.fetch_algorithm_files(ch, name)
    tr, ho = draw_nonce_sets(info.tracks[:1], new_rand_hash(), training_count=2, holdout_count=2)
    t0 = time.monotonic()
    gpu = prepare(ch, uid=host_uid())
    t_prep = time.monotonic() - t0
    local = LocalSettings(cpus=os.cpu_count() or 1, memory_gib=8)
    b = C3Bench(tmp_path, transport=DockerTransport(uid=host_uid()), local=local,
                usd_per_hour=0.0, poll_s=5.0)
    t1 = time.monotonic()
    r = b.evaluate(EvalRequest(ch, files, tr, ho, info.max_fuel, None, CHALLENGES[ch].beat))
    t_job = time.monotonic() - t1
    assert r.compile.ok, r.compile.output[-3000:]
    assert len(r.training) == 2 and r.holdout is not None and len(r.holdout) == 2
    assert any(x.ok and x.quality > 0 for x in r.training), [x.to_dict() for x in r.training]
    assert r.holdout_reason == "forced" and b.cost_mark() == 0.0
    art = next((tmp_path / "local" / "adhoc").glob("talos-*/artifacts/results.json"))
    if os.name != "nt":
        assert art.stat().st_uid == os.getuid(), "artifacts must belong to the user"
    print({"gpu": gpu, "prepare_s": round(t_prep), "job_s": round(t_job),
           "training": [x.to_dict() for x in r.training]})
```

- [ ] **Step 2: Run it, in the background, with a log**

Run: `TALOS_LIVE_BACKEND=local python3 -m pytest -m live tests/test_live.py -k local -s > <scratchpad>/live-local.log 2>&1` with `run_in_background: true`, `timeout: 3600000`. Poll every 2 minutes. Run it twice: the first run measures prepare from nothing (or from the spike's pulled image), the second measures a warm prepare (all markers present, should take seconds) and an incremental job.

Expected: `1 passed` twice, and the printed dict from each run. Record both `prepare_s` and `job_s` values; they are MEASURED for step 3.

- [ ] **Step 3: Documentation**

`AGENTS.md`:

- "What this project is": replace `and LLM-authored code never executes on the user's machine.` with `and LLM-authored code never executes outside a container. Only the ``local`` backend runs that container on the user's own machine, with networking off, capabilities dropped, and CPU and memory limits (``talos/local_transport.py::run_args``).` and after the C3 sentence add `The local backend (Docker on the user's machine) is the third, chosen the same way and running the same batch job through a Docker transport.`
- Invariant 1: after `talos/baseline.py::cache_key`, add `; the local backend's class is ``talos/challenges.py::local_hardware_class``, which carries the host name and the container's CPU and memory limits, so changing either at ``talos setup`` invalidates every local baseline`.
- "What requires a human": add `- Loosening the local job container's isolation flags (``talos/local_transport.py::run_args``): networking, capabilities, the pids limit, the user it runs as. They are the sandbox for LLM-authored code on the user's machine.`
- "Where to look": add rows `| Local backend: Docker transport, container flags | ``talos/local_transport.py::DockerTransport``, ``talos/local_transport.py::run_args`` |` and `| Local backend: image pull, volumes, clone, warm build | ``talos/local_transport.py::prepare`` |`.
- The "Modal and C3 transports" row: `Modal, C3 and local transports`.

`README.md`:

- Prerequisites: after "nothing Rust- or CUDA-related is needed locally." add ` The ``local`` backend needs Docker (and, for GPU challenges, the NVIDIA container toolkit); the compiler still runs inside the container.`
- Backend table: add `| ``local`` | Docker running on this machine. First run per challenge pulls the 13 GB dev image and does one clean build (MEASURED: the first run's `prepare_s` from step 2, on this machine); later builds are incremental (MEASURED: the second run's `job_s` from step 2). GPU challenges need the NVIDIA container toolkit and are unverified (see docs/compute-backends.md). | Nothing. The compute budget question is skipped. |`
- Setup list: step 1 `modal`, `c3` or `local`; add step 9: `**CPUs and memory for the local container**: ``local`` backend only. Defaults are every core and total memory minus 4 GiB. Both are part of the local baseline cache key: change them and local baselines are re-measured.`
- Checks list: `- on ``local``: that ``docker info`` answers, and says whether the NVIDIA runtime is present.`
- Live smoke test: add the `local` command and the MEASURED numbers, with the note that it costs time, not money.

`docs/compute-backends.md`:

- Transports: add a paragraph: on the local backend Talos runs the same job directory in a Docker container through ``talos/local_transport.py::DockerTransport``; the container name is the job id.
- New section `## Local backend` covering: the two volumes and their key (`talos-app-<challenge>-<key>`, `talos-cargo-<challenge>-<key>`, key from both pins); the prepare steps and their markers; the container flags (copy the §4.4 table from the spec); job directories at `runs/<job_id>/local/<purpose>/` and the artifacts under `<container-name>/artifacts/`; reclaiming space after a pin bump with `docker volume ls --filter name=talos- ` and `docker volume rm`; the MEASURED prepare and job times from step 2; and `### Local GPU support (unverified)` stating that the GPU branch has unit tests only and that `TALOS_LIVE_BACKEND=local TALOS_LIVE_CHALLENGE=vector_search` is the test to run on a GPU host, with a line to record the result.
- Rename `## C3 dev images` content: add a sentence that the local backend pulls the same image with `docker pull`.

- [ ] **Step 4: Run the gate**

Run: `make check`
Expected: ruff clean, pytest all pass (the docs-references test must resolve every new `path::symbol`), agentify contract holds.

- [ ] **Step 5: Commit**

```bash
git add tests/test_live.py AGENTS.md README.md docs/compute-backends.md
git commit -m "docs: the local backend; live test for one local knapsack job"
```

---

### Task 11: Finish the branch

- [ ] **Step 1: Mutation-check the new tests**

For each of these, break the code, run the named test, confirm it fails, restore:

| Break | Test that must fail |
|---|---|
| Delete `"--network", "none",` from `run_args` | `test_run_args_carry_every_hardening_flag_and_the_mounts` |
| `return "FAILED"` for every exited container in `status` | `test_status_maps_docker_state_onto_the_c3_vocabulary[TIMED_OUT cases]` |
| Drop the `unstage_algorithm` line in `c3_job.main` | `test_a_leftover_candidate_from_a_previous_job_is_removed_before_staging` |
| `rate = GBP_PER_HOUR[profile] * USD_PER_GBP if not self.usd_per_hour else ...` | `test_local_settings_select_the_local_subdir_flavour_and_a_zero_rate` |
| Remove `and cfg.backend != "local"` from the compute-budget guard | `test_run_local_skips_the_compute_budget_question_and_leaves_the_cap_unset` |
| Make `prepare` skip the marker checks (always clone) | `test_prepare_skips_each_step_whose_marker_is_present[ready]` |

- [ ] **Step 2: Claim table for the PR body**

Before writing the PR body, list every number in it: `| claim | value | MEASURED / ESTIMATED | command | when |`. The spike timings, the live-test timings and the test count delta are MEASURED in this session; the 13 GB image size is MEASURED from `docker images` output or labelled ESTIMATE.

- [ ] **Step 3: Push and open the PR**

Probe first: `timeout 10 ssh -o BatchMode=yes -o ConnectTimeout=8 -T git@github.com`. The PR targets `ghcr-images` if that branch is still unmerged, else `main`; say which in the body. The body states in words that the GPU branch is unverified (spec §7), and ends with the attribution line from the session reminder. Then follow the `review-pr` skill to green.

## Spike results (Task 1, MEASURED 2026-09-23)

Script: the session scratchpad's `spike.sh` (throwaway; log kept as `spike.log` there). Image
`ghcr.io/tig-foundation/tig-monorepo/knapsack/dev:0.0.7`, 16-core / 30 GB Linux box.

| Question | Result | Evidence |
|---|---|---|
| `CARGO_HOME` in the image | unset; cargo at `/root/.cargo` (`$HOME/.cargo`, HOME=/root) | `spike.log` image env line |
| `RUSTUP_HOME` in the image | unset; `/root/.rustup` | same |
| Offline incremental build as host uid | rc=127 in 0.016 s: `cargo: command not found` | `offline build rc=` line |
| Why | `/root` is mode 700 root; a non-root user cannot traverse it, so neither cargo nor the rustup toolchain is reachable (`docker run --user 1000:1000 … ls /root/.cargo/bin` → Permission denied) | follow-up probe |
| Bind-mount file owned by host uid | yes (`-rw-r--r-- 1 1000 1000 … x`) | `ls -ln` line |
| Clean `build_algorithm fast_and_furious` | 12m0.168s real | `time` under "clean build" |
| Incremental `build_algorithm` | not measured by the spike (the run failed before compiling); measured by the live test instead | — |

Decision (spec §6 fallback 2): the job container runs as root, without `--user` and without
the `HOME=/tmp` override. The `chown` of the volumes in prepare is dropped: root needs none.
Whether an offline build works once the registry is warm was settled by the live test's
second run (it does: `1 passed`, `job_s: 844`).

Revised after the whole-branch review (2026-09-23): the first version bind-mounted a
world-writable host directory at `/artifacts` and chowned it back from a helper container.
The reviewer showed that root in the container could leave a setuid-root file on the host
through that mount, and that the hand-over was skipped on timeout and cancel. Now nothing
writable on the host is mounted: `fetch` copies `results.json` and `build.log` out of the
stopped container with `docker cp`, the deploy prune keeps a stopped container until the next
iteration, and the helper container and the 777 directory are gone. Live run 1 also showed
that capability-less root obeys the mode of a user-owned mount (PermissionError on
`build.log`), which the copy-out design sidesteps.
