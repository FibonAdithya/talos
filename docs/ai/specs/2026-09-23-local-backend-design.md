# Local compute backend — design

Date: 2026-09-23. Branch: `local-backend`, cut from `ghcr-images` (two commits ahead of
`main`, no PR yet). It depends on those commits: the local backend runs the official GHCR
dev image that `talos/challenges.py::dev_image` names, and `main` still points C3 at the
Docker Hub mirror.

This spec adds a third compute backend, `local`, chosen at `talos setup` beside `modal`
and `c3`. It compiles and scores candidates in a Docker container on the user's own
machine, so a user is not forced to hold a Modal or C3 account to run Talos.

## 1. Problem

Both existing backends need a hosted account with credit. A user with a capable machine
and Docker has everything the benchmark needs (the TIG dev image is public on GHCR) and
nowhere to point Talos at it.

Two facts about the existing code make this cheap:

- The C3 backend already runs one whole iteration inside a container from a job directory
  (`talos/c3_jobdir.py::write_job_dir` writes it, `talos/c3_job.py::main` runs it and
  writes `results.json` atomically after every nonce). Nothing in that runner is specific
  to C3's scheduler.
- `talos/c3_bench.py::C3Bench` drives that job through a four-method transport protocol
  (`talos/c3_transport.py::C3Transport`: deploy, status, cancel, fetch). A Docker container
  satisfies the same protocol.

So the local backend is the C3 bench with a Docker transport, and its new code is the
transport, a local flavour of the job directory, a prepare step, and setup wiring.

## 2. Decisions taken during design

Recorded so the reasoning survives.

| Question | Decision | Why |
|---|---|---|
| Isolation | Docker container only; no bare-metal path | Keeps the README promise that nothing Rust-related is installed locally, and keeps LLM-authored code inside a sandbox. Bare metal was rejected, and a Docker-plus-bare-metal fallback rejected as doubling the surface. |
| GPU challenges | Supported when Docker reports the `nvidia` runtime | The user asked for it. The branch cannot be run on the development machine (no GPU); see §7 for how it ships. |
| Transport shape | One detached `docker run` per evaluate, polled like a C3 job | Reuses the runner, its tests, and every C3Bench behaviour (reattach on resume, stop, resubmit-once, timeout fill-in). A per-step `docker exec` design ("local Modal") and a long-lived per-job container were rejected as duplicating client logic for no speed gain. |
| Speed | Persistent named volumes for the monorepo checkout and the cargo registry | A C3 job builds from clean in about 466 s (MEASURED 2026-09-14). A persistent cargo target directory makes every build after the first incremental. |

## 3. The project principle this changes

`AGENTS.md` states that LLM-authored code never executes on the user's machine. With the
local backend it does, inside a container with networking off, capabilities dropped, and
CPU and memory limits. The sentence becomes: LLM-authored code never executes outside a
container, and only the `local` backend runs that container on the user's own hardware.
The container hardening flags (§4.4) join the "What requires a human" list: loosening them
is a human decision.

## 4. Design

### 4.1 Transport: `talos/local_transport.py::DockerTransport`

Implements `C3Transport`. Every subprocess call goes through an injected `run`
(`subprocess.run` by default), the same way `CliTransport` does, so unit tests never touch
Docker. `whoami` and `balance_gbp` are not part of the local path and raise
`NotImplementedError`; `check_local` (§4.6) is what setup calls instead.

| Method | Docker | Notes |
|---|---|---|
| `deploy(job_dir)` | `docker run -d --name <name> <flags> <image> bash /work/job.sh` | Returns the container **name** as the job id, not the hex id: the name is chosen before the run (§4.4), so the artifacts directory `<job_dir>/<name>/artifacts/` can be created and bind-mounted before the container exists, and it matches `[A-Za-z0-9._-]+` so `_safe_job_id` accepts it. Every later `docker` command takes the name. Reads `local.json` from the job dir (§4.2) for image, limits and time limit. A name collision (a container left by a killed process whose pending record did not match) is removed with `docker rm -f` first. The transport creates the artifacts directory itself before the run, so Docker does not create it root-owned. |
| `status(job_id)` | `docker inspect --format '{{json .State}}'` | Mapped as in the table below. |
| `cancel(job_id)` | `docker rm -f <id>` | Idempotent; a missing container is not an error. |
| `fetch(job_id, name, dest)` | none | The artifacts directory is a bind mount, so `dest` already exists or does not. Returns whether it exists. `build.log` and `results.json` are the two names C3Bench asks for. |

Status mapping. Docker has no wall-clock limit of its own, so the transport enforces the
one the job directory writer computed (`talos/c3_jobdir.py::time_limit_s`), which it
stores on the container as a label at deploy and reads back at status so a resumed process
enforces the same limit:

| `docker inspect` state | Reported status |
|---|---|
| `Status: created` | `PENDING` |
| `Status: running`, elapsed since `StartedAt` within the limit | `RUNNING` |
| `Status: running`, elapsed over the limit | `docker kill`, then `TIMED_OUT` |
| `Status: exited`, `ExitCode: 0` | `SUCCEEDED` |
| `Status: exited`, `ExitCode != 0` | `FAILED` |
| `Status: exited`, `FinishedAt - StartedAt` at or over the limit | `TIMED_OUT`, whatever the exit code: this is the container we (or a previous process) killed |
| container not found | `C3CommandError` (a poll failure; C3Bench tolerates up to `poll_failures_max`) |

C3Bench's "no capacity" branch (`pending_timeout_s` while `QUEUED`) never fires locally:
a container is `created` for milliseconds before `running`.

### 4.2 Job directory: a `local` flavour of `write_job_dir`

`talos/c3_jobdir.py::write_job_dir` gains a `flavour` argument, `"c3"` (default) or
`"local"`. Both flavours write the same `payload.json` and copy the same `JOB_MODULES`,
so `request_hash` is unchanged and a request hashes identically on every backend. The
local flavour differs in two files:

- No `.c3`. Instead `local.json`: `{"image", "cpus", "memory_gib", "gpu": bool,
  "workers", "time_limit_s"}`. `DockerTransport.deploy` reads it; nothing else does.
- `job.sh` does not download the monorepo tarball (it is on the `/app` volume, §4.3). It
  is:

      #!/bin/bash
      set -euo pipefail
      cd "$C3_JOB_WORKDIR"
      exec python3 -m talos.c3_job

  The environment variables keep their C3 names (`C3_JOB_WORKDIR`, `C3_ARTIFACTS_DIR`);
  `talos/c3_job.py` reads them and renaming them there would touch the C3 path for no
  gain. The transport sets them to `/work` and `/artifacts`.

`workers` is the CPU count for a CPU challenge and 1 for a GPU one, mirroring
`talos/challenges.py::c3_workers`. The payload's `workers` field is what the runner reads,
so the local flavour passes the local count there.

Job directories land at `runs/<job_id>/local/<purpose>/`, mirroring `runs/<job_id>/c3/`.
The container's artifacts directory is `<job_dir>/<container-name>/artifacts/`, the path
C3Bench already expects for pulled artifacts (`job_dir / job_id / "artifacts"`), so
`_collect` is unchanged.

### 4.3 Persistent volumes and the prepare step

Two named Docker volumes per challenge, both keyed by the pins so a pin bump gets fresh
volumes and can never reuse a target directory built against another monorepo:

| Volume | Mounted at | Holds |
|---|---|---|
| `talos-app-<challenge>-<key>` | `/app` | The monorepo checkout at `MONOREPO_REF`, and its cargo target directory once built. |
| `talos-cargo-<challenge>-<key>` | the image's `CARGO_HOME` | The crate registry, so builds work with networking off. Mounting a named volume over an image directory copies the image's contents into the volume on first use, so whatever registry the image ships is kept. |

`<key>` is the first 12 hex characters of `sha256(MONOREPO_REF + "\0" + DEV_IMAGE_TAG)`.
The image's `CARGO_HOME` path is read once by the prepare step (`docker run --rm <image>
sh -c 'echo ${CARGO_HOME:-$HOME/.cargo}'`) and stored in the volume's label, not guessed.

`talos/local_transport.py::prepare(challenge, run)` runs before the baseline in
`execute_job` (where the C3 path runs `image_available`) and before `talos compile`. It
is idempotent and each step is skipped when its marker is present:

1. `docker image inspect <image>` fails → `docker pull <image>` with its progress
   streamed to the terminal. The images are about 13 GB; this is the one slow first step
   and the user sees it happening.
2. Volume `/app` lacks `/app/.talos-ready` → clone the monorepo at `MONOREPO_REF` into it
   (the same tarball URL `job_sh_text` uses on C3) and write the marker.
3. Volume `/app` lacks `/app/.talos-warm` → warm build: `build_algorithm <name>` for the
   first algorithm directory under `tig-algorithms/src/<challenge>/` that is not
   `talos_cand`, then write the marker. This is the only container that runs with
   networking on, and it never contains LLM-authored code. For a GPU challenge it gets
   `--gpus all` like the job container, since the dev image's build may probe the GPU.
4. On POSIX, `chown -R <uid>:<gid>` both volumes so the job containers can run as the host
   user (§4.4). Steps 2 to 4 are one `docker run --rm` as root.
5. For a GPU challenge, `docker run --rm --gpus all <image> nvidia-smi --query-gpu=name
   --format=csv,noheader` gives the GPU name for the hardware class (§4.5).

`prepare` prints one line per step it performs and nothing for a step it skips.

Reclaiming space after a pin bump is a documented `docker volume rm` of the old key, not a
Talos command.

### 4.4 The job container

`DockerTransport.run_args(local_json, job_dir, artifacts_dir)` builds the argv. Listed in
full because these flags are the sandbox:

| Flag | Value | Why |
|---|---|---|
| `-d --name` | `talos-<run-key>-<purpose>-<request-hash>` | `<run-key>` is 8 hex chars of `sha256(run_dir)`, so two run directories on one machine cannot collide. |
| `--label` | `talos.time_limit_s=<n>` | Read back by `status` (§4.1). |
| `-v` | `<job_dir>:/work:ro` | The payload and the talos modules; the runner only reads them. |
| `-v` | `<artifacts_dir>:/artifacts` | Where `results.json` and `build.log` land. |
| `-v` | `talos-app-…:/app` | §4.3. |
| `-v` | `talos-cargo-…:<CARGO_HOME>` | §4.3; the path is read from the volume's label at each deploy. |
| `-e` | `C3_JOB_WORKDIR=/work`, `C3_ARTIFACTS_DIR=/artifacts`, `HOME=/tmp` | `HOME` because the host uid has no home in the image. |
| `--user` | `<uid>:<gid>` | POSIX only (`os.getuid` does not exist on Windows; Docker Desktop maps bind-mount ownership itself). Without it every file under `runs/` is root-owned. |
| `--cpus`, `--memory` | from `local.json` | The limits are part of the hardware class (§4.5). |
| `--network none` | | The job never needs the network: the checkout and the registry are on the volumes. |
| `--cap-drop ALL --security-opt no-new-privileges` | | Nothing in `build_algorithm`, `tig-runtime` or `tig-verifier` needs a capability. |
| `--pids-limit 4096` | | A fork bomb in a candidate fails the container, not the machine. |
| `--gpus all` | GPU challenges only | Requires the `nvidia` runtime; `check_local` refuses a GPU challenge without it. |

The entrypoint is `bash /work/job.sh`.

One change to the runner: `talos/c3_job.py::main` calls `inside.unstage_algorithm` before
`inside.stage_algorithm`. On C3 the candidate directory never pre-exists, so this is a
no-op there. On a persistent `/app` a previous candidate's extra file would otherwise
survive into the next build.

### 4.5 Hardware class and the baseline cache

`talos/challenges.py::local_hardware_class(spec, cpus, memory_gib, gpu_name, host)`:

- CPU challenge: `local-<host>-cpu<cpus>-mem<memory_gib>`.
- GPU challenge: `local-<host>-gpu-<gpu_name slug>`.

`<host>` is `socket.gethostname()` lowercased with non-alphanumerics collapsed to `-`. The
cache at `~/.talos/baselines` is per user, so it is already per machine; the hostname only
guards a home directory synced between machines, and a changed hostname causes a
re-measure, which is the safe failure. Changing the CPU or memory answer at setup changes
the class and so invalidates every local baseline, as invariant 1 requires. A local class
can never equal a Modal (`cpu4-mem8192`) or C3 (`c3-…`) class.

`talos/cli.py::bench_hardware_class` takes the config as well as the backend, so it can
read the local limits; the GPU name comes from `prepare` and is passed through.

Timing noise from other work on the machine can only slow a candidate past its per-track
cap (`talos/loop.py::Loop._timeouts`), which fails it and never flatters it; invariant 1's
one asymmetry holds unchanged.

### 4.6 Config and setup

`talos/config.py::Config` gains `local_cpus: int | None = None` and
`local_memory_gib: int | None = None`, written to `talos.config.json` only when set and
read back with `.get`, so an existing config file loads unchanged. No new secret.

`talos setup` with backend `local` asks, after the provider questions:

1. `CPUs for the local container` — default `os.cpu_count()`.
2. `Memory for the local container in GiB` — default total memory minus 4, floor 4.
   Total memory comes from `os.sysconf` on POSIX and is unknown on Windows, where the
   default is 8.

Both go through `_ask_number` with `int`. Then `talos/cli.py::check_local(run)` runs
`docker info --format '{{json .Runtimes}}'`; a failure is a `ConfigError` naming Docker
as the missing piece. `check_local` returns whether the `nvidia` runtime is present, and
setup prints one line saying whether GPU challenges will be available. Setup does not
pull any image: it does not know the challenge, and the images are large.

`BACKENDS` becomes `("modal", "c3", "local")`. `make_bench("local", …)` builds
`C3Bench(run_dir, pending=pending, transport=DockerTransport(…), flavour="local",
usd_per_hour=0.0)`; see §4.7 for those two new parameters. `compile_backend` accepts
`local` like the others, and `talos compile` on `local` runs `prepare` first. In the
agentic sandbox `talos compile` has no config to read; it uses `cpus=os.cpu_count()` and
`memory_gib=8`, which cannot matter because `talos compile` scores no nonce.

At `talos run` on `local`, `execute_job` runs `prepare` where the C3 path runs
`image_available`, and refuses a GPU challenge without the `nvidia` runtime with a
message naming Modal and C3, setting the job `failed` with a stop reason the way the C3
missing-image branch does.

### 4.7 C3Bench parameters

`C3Bench.__init__` gains `flavour: str = "c3"` and `usd_per_hour: float | None = None`.
`evaluate` uses `self.run_dir / flavour / purpose` and passes `flavour` to `write_job_dir`;
`_collect` charges
`(t_end - t_run) / 3600 * usd_per_hour` when it is given and the existing
`GBP_PER_HOUR[profile] * USD_PER_GBP` when it is None. Error messages that say "C3" take
the transport's `name` attribute (`"cli"`, `"mcp"`, and now `"docker"`) so a local
failure does not tell the user to check C3. Nothing else in the class changes, and the
C3 tests run unchanged.

### 4.8 Budget

Local compute spend is exactly zero. With `--budget-compute-usd` unset on `local`, the
run wizard skips the compute question and leaves the cap `None`, and the `--yes` path does
not apply `DEFAULT_COMPUTE_USD`. A cap given explicitly is honoured (a positive one is
harmless; zero stops before the baseline, as invariant 5 says it must). The status line
and the final summary still print the compute figure, which reads `$0.00`.

## 5. Tests

One file per new module, with the same injected-runner pattern as `tests/test_c3_bench.py`
and `tests/test_c3_transport.py`. Each test names the mutation it catches.

`tests/test_local_transport.py`:

- `deploy` argv: every flag in §4.4 present, `--gpus` only for a GPU challenge, `--user`
  only on POSIX (parametrised over a fake `os.name`). Catches a dropped hardening flag.
- `status` mapping table, one case per row, including elapsed over the limit → `docker
  kill` is issued and `TIMED_OUT` returned; and a container killed by us → `TIMED_OUT`,
  not `FAILED`. Catches a wrong mapping and a missing kill.
- `status` on a missing container raises `C3CommandError`. Catches a bare exception.
- `cancel` on a missing container does not raise.
- `fetch` returns True only when the file exists and never runs a command.
- `prepare`: each step runs only when its marker is absent (parametrised over the four
  markers); the warm-build container is the only one without `--network none`; the GPU
  name query runs only for a GPU challenge. Catches a step that runs unconditionally and a
  warm build that lost its network.
- `run_args` container name differs for two run directories with the same purpose and
  request hash.

`tests/test_c3_jobdir.py` additions:

- The local flavour writes `local.json` and no `.c3`; `job.sh` has no `curl` line;
  `request_hash` is identical for both flavours. Catches a flavour that changed the hash.

`tests/test_c3_bench.py` additions:

- With `usd_per_hour=0.0` a completed job charges nothing; with `None` it charges the C3
  rate (existing test, kept). Catches a `0.0` treated as "not given".
- With `flavour="local"` the job directory is under `run_dir/local/` and `write_job_dir`
  receives the local flavour.

`tests/test_c3_job.py` addition:

- A leftover `talos_cand/extra.rs` in the monorepo fixture is gone after `main` stages the
  payload's files. Catches the missing unstage.

`tests/test_challenges.py` addition:

- `local_hardware_class` for CPU and GPU specs; two hostnames that differ only in case or
  punctuation give the same class; a Modal and a C3 class never start with `local-`.

`tests/test_config.py` and `tests/test_cli.py` additions:

- Config round-trip with and without the local fields; a pre-existing config without them
  loads.
- `setup` on `local`: asks the two questions, calls `check_local`, writes the config,
  writes no secrets file. `check_local` failure is reported and nothing is written.
- `run` on `local` skips the compute-budget question and leaves the cap `None`; `--yes` does
  not apply the default; an explicit `--budget-compute-usd 0` stops before the baseline.
- `run` on a GPU challenge without the runtime fails the job with a stop reason.

`tests/test_live.py::test_local_knapsack_job`: one real local job on knapsack, gated on
`TALOS_LIVE_BACKEND=local`. It costs time, not money, and the implementation runs it on
the development machine. It records the first-build and the incremental-build durations
(MEASURED) for `docs/compute-backends.md`.

## 6. Spike before implementation

Two assumptions in §4.3 and §4.4 are settled by an experiment on the knapsack image
already present on the development machine, before the plan's other tasks start:

1. A build succeeds with `--network none` once the cargo volume has been warmed by one
   networked build. If not: the job container keeps networking on, and §3's sentence says
   so.
2. A job container runs as the host uid after `prepare` has chowned the volumes, and the
   artifacts under `runs/` are owned by the user. If not: containers run as root and
   `docs/compute-backends.md` documents `sudo rm -rf runs/<id>/local`.

The spike also reads the image's `CARGO_HOME` and measures one clean and one incremental
`build_algorithm`. Its script and numbers go in the plan, labelled MEASURED.

## 7. GPU branch: how it ships

The development machine has no GPU. The GPU branch is the `--gpus all` flag, `workers=1`,
the `nvidia-smi` name query in `prepare`, and the GPU hardware class. All four have unit
tests. The live test has a GPU variant (`TALOS_LIVE_BACKEND=local TALOS_LIVE_CHALLENGE=
vector_search`) that has not been run when this ships. The PR body says so in those words,
and `docs/compute-backends.md` marks local GPU support as unverified until someone runs it
on a GPU host and records the result.

## 8. Documentation changes

- `AGENTS.md`: the "What this project is" paragraph (§3); invariant 1 names
  `local_hardware_class`; the "What requires a human" list gains the container hardening
  flags (`talos/local_transport.py::run_args`); the "Where to look" table gains rows for
  the transport and the prepare step.
- `README.md`: a `local` row in the backend table (needs Docker; GPU challenges need the
  NVIDIA container toolkit; first run per challenge pulls a 13 GB image and does one clean
  build), the two setup questions, and a note that the compute budget question is skipped.
- `docs/compute-backends.md`: a local section covering the volumes and their keys, the
  prepare steps, the container flags, how to reclaim volumes after a pin bump, the
  measured build times from the live test, and the GPU verification status.

## 9. Out of scope

- Podman or any non-Docker runtime. The transport shells out to `docker`; a `podman`
  alias may work and is not tested.
- Sharing local baselines between machines.
- A Talos command to prune volumes or images.
- Running the container on a remote Docker host (`DOCKER_HOST`). It may work unchanged;
  it is not tested and the bind mounts would not.
