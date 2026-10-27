# Troubleshooting

Run `talos doctor` first. It checks the config, the LLM credential, the backend, mainnet and
the dev images, and says what to fix for each failure (see
[`talos doctor`](usage.md#talos-doctor)). If that does not explain the problem, find the
message below. Text in angle brackets, such as `<job_id>`, is filled in by Talos.

## Setup and credentials

### `modal deploy failed`

**Symptom:** `talos setup` stops with `modal deploy failed: <Modal's output>`.

**Cause:** the `modal deploy` command that setup runs returned non-zero. The text after the
colon is the end of Modal's own error output, and says why.

**Fix:** act on Modal's message, then run `talos setup` again. If the message is about the
token, create a new one at modal.com/settings/tokens and give it to setup.

### The Modal app is missing or older than the client

**Symptom:** a run stops with one of:

- ``Modal function <name> not found in app talos-bench: <error>. Run `talos setup` to deploy.``
- ``the deployed talos-bench app predates this client (<error>). Run `talos setup` to redeploy.``

**Cause:** the app deployed in your Modal account does not match this version of Talos. This
happens after pulling a new version without redeploying, because the score function's
arguments change between versions.

**Fix:** `talos setup`, then `talos run --resume <job_id>` if a job stopped.

### No API key for the provider

**Symptom:** ``no API key for provider <provider>: run `talos setup` or set <ENV_VAR>``, exit
code 2.

**Cause:** neither `.talos/secrets.json` nor the provider's environment variable holds a key.

**Fix:** `talos setup`, or export the variable the message names.

### C3 balance is low

**Symptom:** `talos setup` prints ``warning: C3 credit balance is low (£<balance>); run `c3 topup` ``,
or `talos doctor` prints `backend: warn: £<balance> left; c3 topup`. On a new account, or when
the balance cannot be read, `talos doctor` prints
`backend: warn: could not read the balance, or it is £0.00; c3 topup` instead.

**Cause:** the C3 account has less than £1 of credit, or none.

**Fix:** `c3 topup`.

### Local container limits above Docker's

**Symptom:** `talos setup` on the `local` backend stops with
`Docker has <n> CPUs and <m> GiB; the container cannot have more (Docker Desktop: raise them under Settings > Resources)`,
exit code 2.

**Cause:** the CPUs or memory you gave are above what the Docker daemon has. On Docker Desktop
those are the VM's figures, not the machine's.

**Fix:** run `talos setup` again and accept the defaults it proposes, or raise the limits in
Docker Desktop under Settings > Resources first.

## Starting a job

### `--budget-usd cannot be enforced`

**Symptom:** `talos run` stops before spending anything with one of:

- `model <model> has no price-table entry, so --budget-usd cannot be enforced; add --budget-hours or --budget-iterations`
- `codex-cli does not report what its calls cost, so --budget-usd cannot be enforced; add --budget-hours or --budget-iterations`

**Cause:** Talos cannot measure the LLM spend of this model or provider (it is `unpriced` or
`not tracked`), so a dollar cap would never stop the job. See [Budget](usage.md#budget).

**Fix:** add `--budget-hours` or `--budget-iterations`.

### Agentic codex refused

**Symptom:** `codex agentic mode runs agent-authored commands on this machine and cannot restrict reads; set TALOS_ALLOW_CODEX_AGENTIC=1 to accept that`.

**Cause:** `codex-cli` has no sandbox that restricts what the agent reads or runs, so agentic
codex is opt-in. See [Agentic mode](usage.md#agentic-mode).

**Fix:** `export TALOS_ALLOW_CODEX_AGENTIC=1` if you accept that, or use `--mode single-shot`,
or `claude-cli`.

With `--start template` the message is instead
`codex agentic mode is not available with --start template: …`, and no opt-in applies. Use
`claude-cli` or `--mode single-shot` for a template start.

### Dev image not on GHCR (C3)

**Symptom:** `talos run` on the C3 backend stops with
``image <image> is not on GHCR (or GHCR is unreachable): the C3 backend pulls the official TIG dev image from GHCR; check that `DEV_IMAGE_TAG` in talos/challenges.py names a published tag``.
The job is left `failed` with reason `dev image not on GHCR`.

**Cause:** GHCR could not be reached, or the pinned dev image tag is not published for this
challenge.

**Fix:** run `talos doctor`; its `images` check says which challenges are missing. If GHCR is
reachable and the tag is missing, see
[Upgrading the TIG dev image](maintainers.md#upgrading-the-tig-dev-image).

### Resume refused after a dev image upgrade

**Symptom:** `job <job_id> was created with a different or unrecorded TIG build (monorepo <ref>, dev image <tag>); start a new job with dev image <tag> to remeasure the baseline and candidates together`,
exit code 2.

**Cause:** the job was started under different TIG pins than this version of Talos has. Its
baseline and candidates cannot be compared with scores from the new build.

**Fix:** start a new job with `talos run`. See
[Upgrading the TIG dev image](maintainers.md#upgrading-the-tig-dev-image).

## A job that stopped

### Job `paused`

**Symptom:** the run ends with `paused: <error>` or `Status: paused (bench: <error>)`, and
`talos status` lists the job as `paused`.

**Cause:** a compute call could not complete. The text after `bench:` says which case it is,
and is also in `state.json` (`stop_reason`) under `runs/<job_id>/`. It is one of:

- the backend could not be reached, or had no hardware free, for longer than Talos retries;
- the Modal app is missing or older than the client (see
  [that entry](#the-modal-app-is-missing-or-older-than-the-client));
- a C3 deploy failed, or a C3 job `failed twice`, `timed out before writing results`,
  `succeeded without results.json` or wrote unreadable results.

**Fix:** read the reason first. If the backend was unreachable or out of capacity, run
`talos doctor` to check it, then `talos run --resume <job_id>`. If a C3 job timed out or
failed twice, resuming submits the same work again and is likely to fail and bill again: a
timeout comes from the files being built and scored, not from the infrastructure. Look at
the job's files under `runs/<job_id>/c3/` before resuming, or start a new job.

### Job `failed`

**Symptom:** the run ends with `Status: failed (<reason>)`, and `talos status` lists the job as
`failed`.

**Cause:** the reason is in `state.json` (`stop_reason`) and in the last lines of
`timeline.jsonl`, both under `runs/<job_id>/`. Reasons Talos sets itself include
`provider auth: …` (the LLM rejected the credential), `provider: …` (another LLM error),
`docker: …`, `no NVIDIA container runtime` and `dev image not on GHCR`.

**Fix:** fix what the reason names (for a provider or credential reason, `talos setup`), then
`talos run --resume <job_id>`.

### A sandbox compile on C3 still running after the agent timed out

**Symptom:** an agentic iteration fails with `claude timed out after 1800s` (or `codex …`),
and the C3 account keeps a job running.

**Cause:** each `talos compile` the agent runs from its sandbox is one C3 job. The 30-minute
agent timeout kills the agent but does not cancel that job, which runs on to its own time
limit and bills for it.

**Fix:** list your C3 jobs with `c3 squeue` and cancel the leftover one with
`c3 cancel <c3_job_id>`.

## Windows

See [What differs on Windows](windows.md#what-differs-on-windows).
