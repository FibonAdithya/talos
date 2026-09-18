# C3 over its hosted MCP endpoint, not the `c3` CLI

Status: implemented 2026-09-18, PR #12. The key check has run (§8). Q1–Q7 are answered by
measurement, and the live job in §6 ran over MCP on 2026-09-18 (§8 Q8, Q9). Still unmeasured:
`cancel_job` (the live job succeeded, so nothing cancelled it) and an explicit count of `429`s.

## 1. Problem

The C3 backend runs the `c3` CLI for every operation (`talos/c3_bench.py`,
`talos/cli.py::check_c3`). Supplying a C3 API key replaces `c3 login`, but it does not
replace the CLI. On Windows that causes three problems:

1. **Install.** The CLI's only documented install is `curl -fsSL https://cthree.cloud/install.sh | sh`,
   which PowerShell cannot run. A Windows build exists: `https://cthree.cloud/releases/latest/c3-windows-amd64.exe`
   returned `206` with an `MZ` header and a length of 22,547,968 bytes (MEASURED 2026-09-17).
   `c3-windows-arm64.exe` returned `404` (MEASURED). Getting it onto PATH is manual.
2. **The execute bit.** `c3_jobdir.write_job_dir` runs `chmod +x job.sh`, which does nothing on
   Windows. It is unknown whether C3 then executes `job.sh` directly (which would fail) or through
   `bash`. UNVERIFIED.
3. **Parsing printed output.** Talos parses CLI stdout: a warning line before JSON
   (`parse_json_stdout`), and a regex over `Credit balance: £…` for the balance.

## 2. What C3 offers besides the CLI

- `api.cthree.cloud` is a JSON HTTP API. It is undocumented; `GET /` returns
  `{"error":{"code":"NOT_FOUND",...}}` (MEASURED). Building on it means reverse-engineering the CLI.
- A hosted **MCP server** at `https://api.cthree.cloud/mcp` is documented at
  <https://docs.cthree.cloud/mcp>. DOCUMENTED claims (the measurements below confirm the read-only ones):
  - It "calls the same API the CLI calls with the same credential". Every tool mirrors one `c3`
    command and returns `cli_equivalent`.
  - Auth is `Authorization: Bearer c3_key_...` (or OAuth). A revoked key returns `403`.
  - It can run with nothing installed on the client.
- The docs describe C3's CLI as "the primary interface for ordinary compute".

Measured against the endpoint on 2026-09-17, without a key:

| Request | Result |
|---|---|
| `POST /mcp` `initialize`, Python's default `User-Agent: Python-urllib/3.x` | `403`, Cloudflare `error 1010 browser_signature_banned` ("Do not retry") |
| Same request with `User-Agent: talos/0.1.0` | `401 UNAUTHORIZED`, `WWW-Authenticate: Bearer realm="c3-mcp", resource_metadata=".../.well-known/oauth-protected-resource/mcp"` |

So the client must send its own `User-Agent`, and even `initialize` needs the key.

Measured with an API key on 2026-09-17 (script: `mcp_probe.py`, `mcp_artifacts.py`, standard
library `urllib` with `User-Agent: talos/0.1.0`). All calls were read-only; no job was created.

| Call | Result |
|---|---|
| `initialize` | `200 application/json`, `protocolVersion 2025-06-18`, server `c3 1.0.0`, **no** `Mcp-Session-Id` header |
| `tools/list` | `200`; 17 tools, including every one in §3 |
| `whoami` | `200`; `structuredContent` with `user_id`, `email`, `org`, `perms`, `access`, `admin`, `email_verified` |
| `balance` | `200`; `structuredContent.balance_gbp` is a number (`9.322322`), plus `low_balance`, `tier` |
| `list_jobs` | `200`; `structuredContent.jobs[]` with `job_id`, `status`, `hardware_profile`, `project` |
| `get_job` on a job submitted by `c3 deploy` from Talos | `200`; top-level `status: "SUCCEEDED"`, `hardware_profile: "cpu-d3-4vcpu-16gb"`, `exit_code`, `failure` |
| `list_artifacts` on the same job | `files[]`: `artifacts/build.log` (31,115 bytes), `artifacts/results.json` (19,212 bytes), each with `size_bytes`, `sha256`, `download_url`; `expires_in` |
| `read_artifact` `artifacts/results.json` | `inline: true`, `encoding: utf8`, `content` parses with keys `compile`, `holdout`, `holdout_reason`, `started`, `training` |
| `read_artifact` inline `content` vs its `sha256` | match |
| `download_url` fetched with default and with Talos user agent | both `200`, 31,115 bytes, `sha256` match (the storage host does not block the default agent) |
| `read_artifact` on a path that does not exist | tool result `isError: true`, `NOT_FOUND`, with the available paths listed |

No output schema is published for any tool (`outputSchema` absent), so result field names are
observed, not contracted. The client must tolerate extra fields and fail loudly on missing ones.

## 3. Operation mapping

| Talos today | CLI | MCP tool | Notes |
|---|---|---|---|
| setup: identity | `c3 whoami` | `whoami` | |
| setup: balance | `c3 balance` + regex | `balance` | `balance_gbp` is a number, so the regex goes (MEASURED) |
| submit | `c3 deploy --json` in the job dir | `deploy` with inline `files` | inline limit 20 MiB decoded, 500 files; a Talos job dir is ~10 files, well under 1 MiB ESTIMATE (unverified) |
| poll | `c3 squeue --json`, row filtered by id | `get_job(job_id)` | fetches one job, not the whole queue; top-level `status` (MEASURED) |
| stop | `c3 cancel <id>` | `cancel_job(job_id)` | |
| collect | `c3 pull <id> --json`, then read `results.json` / `build.log` from disk | `read_artifact(job_id, path)` | paths `artifacts/results.json`, `artifacts/build.log`; inline up to 1 MiB, otherwise a short-lived download URL (MEASURED); `build.log` is capped at `BUILD_OUTPUT_CAP` = 1,000,000 characters, which can exceed 1 MiB in bytes |

Every command Talos uses has a tool (MEASURED in `tools/list`). `topup`, `upgrade` and API-key management are
not exposed, and Talos does not call them.

## 4. Design

### 4.1 Transport seam

`C3Bench` keeps all of its policy: request hashing, reattach on resume, the anchored retry
window, pending timeout and cancel, `_JobFailed` resubmission, timeout filling, and cost
estimates. Only the calls to `c3` move behind a transport:

```python
class C3Transport(Protocol):
    def whoami(self) -> dict: ...
    def balance_gbp(self) -> float | None: ...          # None: could not read it
    def deploy(self, job_dir: Path) -> str: ...          # returns the C3 job id
    def status(self, job_id: str) -> str: ...            # upper-cased C3 status
    def cancel(self, job_id: str) -> None: ...           # best effort
    def fetch(self, job_id: str, name: str, dest: Path) -> bool: ...  # False: artifact absent
```

- `CliTransport` is today's code, moved without changing behaviour. Its tests move with it.
- `McpTransport` implements the same six methods over HTTP.
- Both raise `C3CommandError` for every failure that `C3Bench` already treats as a transient CLI
  failure, so `_wait`'s failure counter, `_deploy`'s `BenchUnavailable` mapping and `_collect`'s
  handling stay as they are.

`_pull` becomes `fetch` of `artifacts/results.json` and `artifacts/build.log` (the paths
`list_artifacts` reported, MEASURED) into `job_dir/<job_id>/artifacts/`, the layout `_collect`
already reads. `fetch` uses `read_artifact`:

- `inline: true`: write `content` (decoding base64 when `encoding` says so).
- Otherwise: GET `download_url`.
- Either way, compare the bytes' SHA-256 with the reported `sha256` and raise `C3CommandError` on a
  mismatch, so a truncated transfer never reaches `_result_from`.
- `isError` with `NOT_FOUND`: return `False`, which `_collect` already treats as "no results".

### 4.2 Choosing a transport

- A C3 API key is configured (`resolve_c3_api_key`: `.talos/secrets.json`, then `C3_API_KEY`):
  **MCP**.
- No key: **CLI**, using the `c3 login` session, exactly as today.

There is no separate setting. A user who wants the CLI with a key can leave the key out of Talos and
run `c3 login`. `talos setup` prints which one it will use.

### 4.3 `McpTransport` wire details

- Standard library only (`urllib.request`), matching `talos/providers/openai_compat.py`. The MCP
  Python SDK would add a dependency for about 60 lines of JSON-RPC.
- Every request sends `User-Agent: talos/<version>` (§2: the default agent is blocked),
  `Content-Type: application/json`, `Accept: application/json, text/event-stream`, and
  `Authorization: Bearer <key>`.
- Session: `initialize`, then `notifications/initialized`, once per `McpTransport`. The server
  issued no `Mcp-Session-Id` (MEASURED), so calls are effectively stateless. Still keep and resend
  one if a later server version returns it, and on `404` for a session start a new one once.
- Responses were `application/json` (MEASURED). Also accept `text/event-stream`, which the MCP
  transport allows: use the `data:` line carrying the JSON-RPC response with the request's `id`.
- Read tool results from `result.structuredContent` (present on every call measured). Fall back to
  parsing `result.content[].text` as JSON.
- Errors, all raised as `C3CommandError`:
  - HTTP `401`/`403`: bad or revoked key. Setup reports it as a key problem, not a network one.
  - JSON-RPC `error`, or a tool result with `isError: true`: the tool's text.
  - Timeout, `URLError`, unparseable body.
- The key never appears in argv, logs, exception text or `state.json`. Error text passes through
  the existing `_redact`, extended to strip `c3_key_[A-Za-z0-9]+`.

### 4.4 Deploy payload

`write_job_dir` still writes the job directory, so `runs/<job_id>/c3/<n>/` keeps the same record
of what was sent. `McpTransport.deploy` reads that directory back and sends:

- `files`: every file except `.c3`, as `{path, content}`, with `executable: true` for `job.sh`.
  The schema documents `executable` as "Mark the file executable (mode 0755)" (MEASURED from
  `tools/list`), which removes the Windows execute-bit question (§1.2). Every Talos job file is
  UTF-8 text, so `encoding` stays at its default.
- The `.c3` settings as the tool's own arguments, which the schema defines one for one:
  `project` (required), `job_name`, `script` (required with `files`), `hardware`,
  `walltime_seconds` (integer seconds, "same as `time:` in .c3"), `docker_image`,
  `docker_requires_accelerator` (`cuda` | `none`). The values come from the same inputs as
  `c3_config_text`, through one function both use, so the two cannot drift.
- `.c3` itself is not sent. Whether `deploy` reads an inline `.c3` is undocumented, and the
  arguments make it unnecessary.

**Invariant** (AGENTS.md, invariant 1): baseline and candidate must run on the same hardware
class. `c3_hardware_class` is part of the baseline cache key. `McpTransport` must send exactly the
profile and image the `.c3` names. A test asserts the deploy arguments equal the `.c3` values for
a CPU and a GPU challenge.

`payload.json` contains the job's rand hash. It is uploaded today too, so this is no change in
exposure.

### 4.5 Resume compatibility

A pending record stores `job_id`. `get_job`, `list_artifacts` and `read_artifact` all worked on a
job that Talos had submitted with `c3 deploy` (MEASURED), so job ids are shared. A job submitted
under one transport can be resumed under the other, for example after adding a key. No refusal
or transport field is needed.

### 4.6 Setup and messages

- `check_c3` goes through the chosen transport, so a key-only Windows user needs no CLI.
- The CLI-missing message stops telling key users to run `c3 login`.
- The README's C3 row says: with an API key, nothing to install; without one, the `c3` CLI and
  `c3 login`.

## 5. Tests

Every test names the mutation it catches.

| Test | Mutation it catches |
|---|---|
| transport chosen: key → MCP, no key → CLI | selection inverted or constant |
| every MCP request carries `User-Agent: talos/…` and `Authorization: Bearer` | either header dropped (the first is a Cloudflare 403) |
| key absent from every raised message and from the pending record | key interpolated into error text |
| JSON and SSE responses both parse; SSE picks the matching `id` | SSE unsupported, or the first `data:` line taken |
| `isError: true` and JSON-RPC `error` both raise `C3CommandError` | tool error treated as success |
| `401`/`403` surface as a key problem in setup | auth failure reported as "unreachable" |
| deploy sends `executable: true` for `job.sh` only | flag dropped, or set on every file |
| deploy hardware and image equal the `.c3` values (CPU and GPU challenge) | a hard-coded profile (breaks invariant 1) |
| `read_artifact` inline, URL fallback over 1 MiB, missing artifact → `False` | URL branch missing; absent file raises instead of `False` |
| a `build.log` over 1 MiB is fetched whole | inline-only fetch truncates |
| existing `C3Bench` policy tests run against both transports (parametrised fake) | policy accidentally forked per transport |
| fetch rejects content whose SHA-256 differs from `sha256` | integrity check dropped |
| `NOT_FOUND` from `read_artifact` returns `False`; any other `isError` raises | every error treated as "absent", hiding auth failures |
| `status` reads top-level `status` from `get_job` and upper-cases it | reading `current_activity.event_type` (`JOB_SUCCEEDED`) instead |
| the deploy arguments and `c3_config_text` come from one function | `.c3` and MCP arguments drift apart |

The HTTP layer is tested with an injected `post(url, headers, body) -> (status, headers, body)`
function, plus one test against a `http.server` bound to `127.0.0.1` in a thread for real header
and SSE handling. No test talks to `api.cthree.cloud`; that is the live test's job.

Per the C3 test-safety note, any test that could reach the real `c3` or MCP endpoint must stub
the transport before mutation-checking.

## 6. Live verification

1. **Key check (no cost):** done 2026-09-17, see §2 and §8.
2. **One live job (costs credit, ~12 minutes of `cpu-d3-4vcpu-16gb`, ESTIMATE ~£0.03):**
   `tests/test_live.py -k c3`, with `TALOS_LIVE_BACKEND=c3` and a key, so the run uses MCP.
   Settles §8 Q2 and Q7, and shows `job.sh` running from an inline upload.
3. **Windows:** the CI matrix covers the unit tests. A real Windows run needs the user's machine.

## 7. Out of scope

- Removing `CliTransport`. `c3 login` users keep it.
- OAuth. Talos is a CLI; a browser OAuth flow is more than a key paste.
- `prepare_upload` (presigned PUT uploads). Inline `files` covers Talos's sizes.

## 8. Questions for the key check, and answers

| # | Question | Answer | Evidence |
|---|---|---|---|
| Q1 | Does `initialize` + `tools/list` work from `urllib` with the user agent and key? Is a session id issued? | Yes. No session id. | MEASURED, §2 |
| Q2 | `deploy.files` entry schema | `{path, content, encoding?: utf8\|base64, executable?: bool}`; `path` and `content` required | MEASURED, `tools/list` |
| Q3 | Inline `.c3`, or tool arguments? Hardware argument name; `walltime_seconds` units | Tool arguments: `hardware`, `walltime_seconds` in seconds, `docker_image`, `docker_requires_accelerator`, `script`, `project` (required) | MEASURED, `tools/list`; honoured by a real deploy is UNVERIFIED until §6.2 |
| Q4 | `get_job` status field and values | top-level `status`; `SUCCEEDED` observed, same spelling as `c3 squeue` | MEASURED, one terminal job; other states UNVERIFIED |
| Q5 | `balance` shape | `structuredContent.balance_gbp`, a number | MEASURED |
| Q6 | Can MCP read a CLI-submitted job? | Yes: `get_job`, `list_artifacts`, `read_artifact` | MEASURED |
| Q7 | Artifact paths | `artifacts/results.json`, `artifacts/build.log`; inline when small, `sha256` and `download_url` always given | MEASURED |
| Q8 | Rate limits at a 20 s poll interval | No rate limit stopped a job polled every 20 s for 12 min 16 s (about 36 `get_job` calls, ESTIMATE from the duration). A single `429` would have been absorbed by `_wait`'s failure tolerance without being logged, so "no `429` at all" is UNVERIFIED | MEASURED 2026-09-18, job `job_1789723631604_rlzgbv`: `1 passed in 736.27s` |
| Q9 | Does `deploy` with `files` run `job.sh` marked `executable`, and does `cancel_job` cancel? | `deploy`: yes. An inline upload with `executable: true` on `job.sh` and the settings as tool arguments ran to `SUCCEEDED`, `exit_code 0`, on `cpu-d3-4vcpu-16gb` (the profile `.c3` names); `results.json` (15,339 bytes) and `build.log` (31,115 bytes) were fetched with `read_artifact` and passed the sha256 check; compile ok, 2 training and 2 held-out results. `cancel_job`: UNVERIFIED, the job was never cancelled | MEASURED 2026-09-18, job `job_1789723631604_rlzgbv`, `{'transport': 'mcp', 'keyed': True}`; `get_job`: started 09:28:32.024Z, finished 09:38:59.487Z (627.5 s running). Cost: Talos's own figure is $0.0266 (ESTIMATE, from the £0.11/h list price); the balance read £9.306636 afterwards against £9.322322 on 2026-09-17 (£0.0157 lower, with no record of what else ran between the two readings) |
