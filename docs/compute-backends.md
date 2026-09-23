# Compute backends

How Talos uses Modal and C3, what a C3 job costs in time, and what maintainers do to keep the
C3 backend working. For choosing and configuring a backend, see
[README.md](../README.md#3-pick-a-compute-backend).

## Transports

On Modal, `talos setup` deploys the benchmark app to your account, and Talos calls its compile
and score functions.

On the C3 backend Talos uses C3's hosted MCP endpoint (`https://api.cthree.cloud/mcp`) when
a C3 API key is configured, and the `c3` CLI when it is not. The key path needs no C3
install, which is what makes the C3 backend usable on Windows, and it uploads job.sh already
marked executable. Both paths submit the same job directory, written to
`runs/<job_id>/c3/<n>/`.

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
before the baseline is measured, and fails with a `DEV_IMAGE_TAG` hint if it is missing.
