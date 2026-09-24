# Compute backends

How Talos uses Modal, C3 and local Docker, what a C3 job and a local job cost in time, and
what maintainers do to keep the C3 and local backends working. For choosing and configuring a backend, see
[README.md](../README.md#3-pick-a-compute-backend).

## Transports

On Modal, `talos setup` deploys the benchmark app to your account, and Talos calls its compile
and score functions.

On the C3 backend Talos uses C3's hosted MCP endpoint (`https://api.cthree.cloud/mcp`) when
a C3 API key is configured, and the `c3` CLI when it is not. The key path needs no C3
install, which is what makes the C3 backend usable on Windows, and it uploads job.sh already
marked executable. Both paths submit the same job directory, written to
`runs/<job_id>/c3/<n>/`.

On the local backend Talos writes the same job directory in a local flavour and runs it in a
Docker container on this machine through `talos/local_transport.py::DockerTransport`, which
satisfies the C3 transport protocol (deploy, status, cancel, fetch). The container's name is
the job id. See [Local backend](#local-backend).

## C3 job directories

On the C3 backend, each iteration gets `runs/<job_id>/c3/<n>/` in the
[run directory](../README.md#where-results-land), and the baseline measurement gets
`runs/<job_id>/c3/baseline/`. Each holds the files submitted to C3 for that job (a .c3
config, a job.sh entrypoint, a payload.json, and copies of the modules the container needs)
and the pulled artifacts, under `<c3-job-id>/artifacts/`. `<c3-job-id>` is C3's own job id,
distinct from the Talos `<job_id>`. The payload carries the job's `rand_hash` and is
uploaded to C3's workspace store as part of the job directory.

## C3 timings

On C3, one iteration is one batch job with about 12 minutes of fixed overhead before any
nonce is scored — MEASURED 2026-09-14 on a spike run (knapsack, 4 vCPU): about 4 minutes
to script start, 466 seconds to build the candidate, then 1 to 2 seconds per nonce at
mainnet fuel. The release smoke test on 2026-09-15 (MEASURED, knapsack, 4 vCPU, two
training and two held-out nonces) took 12 min 20 s from submission to result: about 2 minutes
queued, 470 seconds to build, 1.3 to 1.7 seconds per nonce; the client's cost estimate was
$0.027 and the account balance fell by £0.02. Modal has no equivalent per-job overhead.

## C3 dev images

C3 pulls the official TIG dev image straight from GHCR,
`ghcr.io/tig-foundation/tig-monorepo/<challenge>/dev:<DEV_IMAGE_TAG>`, the same reference
Modal builds from, so a candidate is compiled in one image whichever backend runs it. There
is no mirror to maintain: a `DEV_IMAGE_TAG` bump takes effect on the next job. `talos run` on
C3 checks that the tag exists on GHCR (an anonymous pull-scoped token, then a manifest GET)
before the baseline is measured, and fails with a `DEV_IMAGE_TAG` hint if it is missing. The
local backend pulls the same image with `docker pull` the first time a challenge is run.

## Local backend

The local backend is the C3 bench with a Docker transport. One evaluate call is one detached
`docker run` of the challenge's dev image, polled with `docker inspect` until it exits, with
`results.json` and `build.log` copied out of the stopped container by `docker cp`. Nothing is billed; the compute figure on the status line
reads `$0.00`.

### Volumes and the prepare step

Two named Docker volumes per challenge, keyed by the first 12 hex characters of
`sha256(MONOREPO_REF + "\0" + DEV_IMAGE_TAG)`, so a pin bump gets fresh volumes and never
reuses a target directory built against another monorepo:

| Volume | Mounted at | Holds |
|---|---|---|
| `talos-app-<challenge>-<key>` | `/app` | The monorepo checkout at `MONOREPO_REF` and its cargo target directory. |
| `talos-cargo-<challenge>-<key>` | the image's cargo home (`/root/.cargo` in the 0.0.7 images) | The crate registry, so builds work with networking off. |

What the volumes buy is the clone, the dependency compile and `--network none`; they do not
make a build incremental. The 0.0.7 image's `build_so` runs an LLVM fuel-instrumentation pass
(`opt`, `llc`, `clang`) over every dependency's IR, standard library included, on every
build, and that pass is most of the build time (see [Local timings](#local-timings)).

`talos/local_transport.py::prepare` runs before the baseline on every `talos run` (and before
`talos compile`). Each step is skipped when its marker is present, and one line is printed per
step performed:

1. `docker pull` of the dev image if `docker image inspect` does not find it (about 13 GB).
2. Create the cargo volume, labelled with the image's cargo and rustup home paths.
3. Create the app volume.
4. Clone the monorepo tarball at the pin into `/app` (marker `/app/.talos-ready`).
5. Warm-up build: `build_algorithm` of the first algorithm the pinned monorepo ships for the
   challenge, with networking on, so the registry and the dependency compile are populated by
   code that is not LLM-authored (marker `/app/.talos-warm`).
6. For a GPU challenge, read the GPU name with `nvidia-smi` for the hardware class.

Both volumes are shared by every job of the challenge on this machine, and the runner stages
the candidate into the checkout on `/app` (`talos/c3_job.py::main`), so jobs take turns: the
local job script (`talos/c3_jobdir.py::local_job_sh_text`) runs the runner under
`flock /app/.talos-lock`, and the clone and warm-up scripts hold the same lock. Two runs of one challenge at once, or a `talos compile` beside a
run, therefore serialise on the build instead of compiling each other's files; a waiting job's
time limit still counts from its start.

To reclaim space after a pin bump, list and remove the old volumes by hand:

```bash
docker volume ls --filter name=talos-
docker volume rm <old volume names>
```

### The job container

Every flag is in `talos/local_transport.py::run_args`; loosening them is a human decision
(`AGENTS.md`, "What requires a human").

| Flag | Why |
|---|---|
| `-v <job dir>:/work:ro` | The payload and the talos modules; the runner only reads them. |
| (no mount for `/artifacts`) | `results.json` and `build.log` stay inside the container; Talos copies them out with `docker cp` once the job has stopped. Nothing writable on the host is mounted, so root in the container cannot leave a file (setuid or otherwise) on the host. |
| `-v talos-app-…:/app`, `-v talos-cargo-…:<cargo home>` | The volumes above. |
| `-e CARGO_HOME=… -e RUSTUP_HOME=…` | Cargo and rustup are told where their files are, so the mounted registry is the one used whatever the image's profile does. |
| (no `--user`) | The job runs as root inside the container: the image keeps cargo and rustup under `/root`, mode 700, so a non-root user cannot build. With every capability dropped, no network, and no writable host mount, what root can reach is the container's own filesystem and the two volumes. Files under `runs/` are written by Talos itself, so they belong to the user. |
| `--cpus N --memory Mg` | From `talos setup`, which defaults them to what `docker info` reports and refuses more: Docker rejects a container with more CPUs than the daemon has (MEASURED 2026-09-24, Docker 29.1.3: "range of CPUs is from 0.01 to 16.00"), and accepts one with more memory than the daemon has (MEASURED: `--memory 999g` on a 30 GiB daemon), which would leave the limit meaning nothing. On Docker Desktop both figures are the VM's (Settings > Resources), not the machine's. Part of the hardware class. |
| `--network none` | The checkout and the registry are on the volumes; the job never needs the network. |
| `--cap-drop ALL --security-opt no-new-privileges --pids-limit 4096` | Nothing in the build or the runtime needs a capability; a fork bomb fails the container, not the machine. |
| `--gpus all` | GPU challenges only. |

Job directories land at `runs/<job_id>/local/<purpose>/`, and the container's artifacts under
`<job dir>/<container name>/artifacts/` (copied out of the container). The container name is
`talos-<8 hex of the run dir>-<purpose>-<request hash>`; the request hash is derived from a
hash of the rand hash, never the rand hash itself. Docker has no wall-clock limit of its own,
so the transport stores the job's limit as a container label and kills the container when it
is exceeded, reporting `TIMED_OUT` exactly as C3 would. A stopped container is kept until the
next deploy of the same run, because its artifacts are read from it; exited containers of
earlier iterations are removed then. The local build allowance in the time limit is one hour
(`talos/c3_jobdir.py::LOCAL_BUILD_ALLOWANCE_S`), against 20 minutes on C3, because the
candidate build takes about 15 minutes on 16 cores and longer on fewer.

The hardware class is `local-<host>-cpu<N>-mem<M>` (or `local-<host>-gpu-<name>`), so a local
baseline never matches a Modal or C3 one, and a changed CPU or memory setting is a re-measure.

### Local timings

MEASURED 2026-09-23 on a 16-core, 30 GB machine (`tests/test_live.py::test_local_knapsack_job`,
knapsack, two training and two held-out nonces):

| Step | Time |
|---|---|
| Spike: clean `build_algorithm` of a shipped algorithm, no cache | 12m0s |
| Live run 1: warm-up build in `prepare` (clone done, registry warm) | 12m (14:31 to 14:43 UTC) |
| Live run 1: the job's `build_algorithm` of the candidate, cargo cache warm | 14m44s (container start to exit) |
| Live run 2: `prepare` with every marker present | 1 s (`prepare_s: 1`) |
| Live run 2: one job, build plus 4 nonces (1.3 to 1.7 s per nonce) | 14m4s (`job_s: 844`; `1 passed` in 14m7s) |
| Live run 3, the shipped copy-out transport: prepare with markers present / one job | 1 s / 14m59s (`job_s: 899`; `1 passed` in 15m3s) |

The job's build is no faster than the warm-up build: the instrumentation pass, not the Rust
compile, is the cost. Caching the instrumented objects per IR file would remove it and is
out of scope here.

### Local GPU support (unverified)

The GPU branch is the `--gpus all` flag, one scoring worker, the `nvidia-smi` name query in
`prepare`, and the GPU hardware class. All four have unit tests. The live test has not been run
on a GPU host as of 2026-09-23. To verify it, on a machine with the NVIDIA container toolkit:

```bash
TALOS_LIVE_BACKEND=local TALOS_LIVE_CHALLENGE=vector_search .venv/bin/pytest -m live tests/test_live.py -k local -s
```

Record the printed `gpu`, `prepare_s` and `job_s` here when it passes.
