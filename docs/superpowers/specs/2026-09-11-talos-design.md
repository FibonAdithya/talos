# Talos: single-user autoresearch for TIG

**Status:** accepted design, 2026-09-11.
**Scope of this spec:** one implementation plan. Auto-submission to TIG, multi-agent
swarms and a hosted service are explicitly out of scope and listed under Non-goals.

## 1. Problem

TIG rewards innovators whose algorithm gets adopted by benchmarkers. Improving on the
current state of the art (SOTA) is an iterative research loop: take the top algorithm,
form a hypothesis, change the code, benchmark it faithfully, keep it if it wins, repeat.

Prometheus Swarm (`tig-foundation/prometheus-swarm`) already runs that loop as a
multi-agent swarm, but a non-technical innovator cannot get it running: a host must
provision a coordination server on Railway, contributors must clone a repo, run a setup
UI, and either install Docker or configure C3, and the benchmark it runs is the swarm's
own wall-clock quality score rather than TIG's fuel-metered harness.

Talos is the minimum product that lets one person run that loop alone, with a faithful
benchmark, until they beat SOTA or run out of budget, and walk away with a package they
can submit themselves.

## 2. Goals

- A user with Python and Git, no Docker and no Rust, can go from `git clone` to a running
  research loop in two wizards: `talos setup` once, `talos run` per job.
- The loop starts from the current top-adopted mainnet algorithm and measures it once, at
  the start of the job, as the bar to beat.
- Scoring is TIG-faithful: upstream `tig-runtime` and `tig-verifier` from the official
  per-challenge dev image, fuel-metered, over a fixed nonce set. Baseline and candidate
  are always scored on identical nonces, fuel and hardware class.
- LLM-authored code never executes on the user's machine.
- The user's "direction" text steers the agent as tacit knowledge from iteration one.
- A win is confirmed on held-out nonces the loop has never seen.
- The loop stops on win, on budget, or on an unrecoverable error, and always hands back the
  best-so-far as a local package.
- Any of: an API key (Anthropic, OpenAI, Google, OpenRouter, OpenAI-compatible endpoint),
  a `claude` CLI login, or a `codex` CLI login can drive the agent.

## 3. Non-goals

- Submitting to TIG on the user's behalf. The package is submit-ready; the user submits.
- Multi-agent swarms, cross-user inspiration, a coordination server or a dashboard.
  Agent count is a later knob; day one is one agent per job.
- A hosted web service or holding any user credential on infrastructure we run.
- Local Docker benchmarking. Compute is Modal only.
- Rewriting Prometheus. Talos lifts specific modules from it as a library (section 12).

## 4. Decisions and rationale

| Decision | Chosen | Why |
|---|---|---|
| Where the loop runs | User's machine, Python CLI | User asked for clone, setup wizard, run wizard. No service to operate. |
| Where code executes | Modal functions in the TIG dev image | Modal builds an image from GHCR or a Dockerfile once and caches it, fans out from Python with `map`, bills per second. C3 cannot pull GHCR images and has no docker-in-docker, so every job there rebuilds the toolchain. |
| Compute credential | User's own Modal token, entered in setup | No auth, metering or abuse handling on our side. User pays their own compute. |
| Benchmark | Upstream `tig-runtime` + `tig-verifier`, fuel-metered | "Beat SOTA" must mean what benchmarkers measure. Prometheus's `benchmark.py` has no fuel metering (verified by grep). |
| Baseline | Top-adoption mainnet algorithm, measured once at job start | User's explicit request. Cached per challenge and monorepo commit. |
| Win condition | Beat baseline on training nonces, then on held-out nonces | Prevents winning by overfitting the nonces the loop has seen. |
| Exit | Stop and hand back; no auto-submit | User's decision; auto-submit deferred. |
| Providers | API keys and headless `claude` / `codex` subscriptions, single-shot and agentic | User's request; Prometheus already has the three CLI providers. |
| Agent count | One | Simplest correct product. Swarm mechanics are a later knob. |

## 5. User surface

### 5.1 `talos setup`

Run once. Interactive, terminal only.

1. Provider kind: `anthropic`, `openai`, `google`, `openrouter`, `custom` (OpenAI-compatible
   base URL + model id + key variable name), `claude-cli`, `codex-cli`.
2. Model id, with a sensible default per provider.
3. For API kinds: the key. For CLI kinds: nothing; Talos checks the binary is on `PATH`
   and runs one trivial headless call to confirm a logged-in session.
4. Modal token id and secret, with a link to where to create them.
5. Validation: one cheap call per credential. A failing credential is reported and the
   wizard re-prompts. Nothing is written until every credential validates.

Writes `talos.config.json` (provider, model, mode, defaults) and `.talos/secrets.json`
(mode 0600, gitignored). CLI providers store no secret at all.

### 5.2 `talos run`

Run per job. Interactive unless every value is given as a flag.

1. Challenge: one of the eight TIG challenges. GPU challenges are labelled with their
   Modal GPU class and approximate cost per benchmark.
2. Direction: free text, multi-line, or a path to a file. This becomes the first entry in
   the job's tacit knowledge.
3. Budget: for API providers, dollars of LLM spend and hours of wall clock. For CLI
   providers, iterations and hours. Modal spend is always shown and capped in dollars.
4. Mode for CLI providers: `single-shot` (default) or `agentic`, with the cost warning that
   agentic uses roughly five to twenty times the tokens.
5. Starts the loop. The terminal shows a status line (job id, iteration, best delta vs
   baseline, spend, time left) and streams one line per event. `Ctrl-C` stops cleanly at
   the next safe point and packages the best-so-far.

Flags: `--challenge`, `--direction`/`--direction-file`, `--budget-usd`, `--budget-hours`,
`--budget-iterations`, `--mode`, `--resume <job_id>`, `--yes`.

### 5.3 `talos compile`

Internal command exposed for agentic mode. Reads the algorithm files from the current
worktree, calls Modal `compile`, prints compiler output, exits non-zero on failure. It is
the only Bash command the agentic sandbox allows.

### 5.4 Run directory

`runs/<job_id>/` holds `job.json` (immutable inputs), `state.json` (mutable progress),
`timeline.jsonl` (one event per line), `tacit.md`, `baseline/`, `best/`, `iterations/<n>/`
and on exit `package/`. Everything a user or a future agent needs is here; nothing lives
only in memory.

## 6. Architecture

```
user's machine                                   Modal (user's account)
--------------------------------                 ---------------------------------
talos setup  -> talos.config.json                talos-bench app
                .talos/secrets.json                image: TIG dev image per challenge
talos run    -> loop process                       compile(challenge, files) -> artifact
                 |- provider client  --> LLM API or local claude/codex CLI
                 |- bench client     --> score(challenge, artifact, nonce_set, fuel)
                 |- baseline resolver --> mainnet API + monorepo raw files (read only)
                 |- runs/<job>/      <-- state, timeline, package
```

Trust boundaries: the loop process holds the LLM key for the life of the run. Modal holds
compiled artifacts and per-nonce outputs in the user's own Modal account. The mainnet API
and GitHub are read-only. No Talos-operated service exists.

## 7. Components

Each is a module with one job, testable alone.

### 7.1 Config and secrets (`talos/config.py`)

Load and validate `talos.config.json` and `.talos/secrets.json`. Resolve the provider's
credential in this order: secrets file, environment variable, CLI login session. Refuse to
start a run with an unset or empty credential rather than falling back silently.

### 7.2 Provider client (`talos/providers/`)

One interface, `complete(system, messages) -> text` for single-shot and
`run_agentic(worktree, prompt, timeout) -> hypothesis` for agentic. Backends: OpenAI-
compatible HTTP (covers OpenAI, OpenRouter, custom, DeepSeek-style endpoints), Anthropic,
Google, `claude` CLI, `codex` CLI. Token usage is returned on every call and priced from a
per-model table shipped with Talos, so budget accounting uses measured usage.

### 7.3 Baseline resolver (`talos/baseline.py`)

Port of Prometheus `server/mainnet_seed.py`. `top_algorithm(challenge)` returns the
highest-adoption compiled algorithm from `https://mainnet-api.tig.foundation`.
`fetch_algorithm_files(challenge, name)` returns the file bundle from the monorepo's raw
files at the branch `<challenge>/<name>`. The job records algorithm name, adoption, and
monorepo commit. Baseline scores are cached under `~/.talos/baselines/<challenge>/<commit>/
<name>/` keyed also by nonce set, fuel and hardware class, so a second job on the same
challenge skips measurement.

### 7.4 Bench client and Modal app (`talos/bench.py`, `modal_app/`)

Modal app `talos-bench`, deployed once per Talos version into the user's account by
`talos setup`. Image: `modal.Image.from_registry` on the official TIG dev image for the
challenge, pinned by digest in a table shipped with Talos, with `tig-runtime` and
`tig-verifier` built for that challenge at image build time so per-call cost is zero.

- `compile(challenge, files: dict[path, str]) -> {ok, artifact_id, stderr}`. Writes the
  files into the monorepo's algorithm tree under a fixed name, runs the monorepo's
  `build_algorithm`, stores the `.so` (and PTX for CUDA challenges) in a Modal Volume
  under a content hash, and returns that hash. Removes the algorithm from the tree
  afterwards so the tracked `mod.rs` is left exactly as found.
- `score_nonce(challenge, artifact_id, seed, nonce, fuel, timeout) -> NonceResult`. Runs
  `tig-runtime` then `tig-verifier` for one nonce. Returns quality, runtime, fuel used,
  and an error class from `{ok, no_solution, invalid, out_of_fuel, panic, timeout}`.
- `score(challenge, artifact_id, nonce_set, fuel)` on the client side maps `score_nonce`
  over the set and returns the ordered list of `NonceResult`.

Hardware is pinned per challenge: a fixed CPU count and memory for the five CPU
challenges, one fixed GPU type for `hypergraph`, `neuralnet_optimizer` and `vector_search`.
The class is recorded on every result so a baseline is never compared against a candidate
run on a different class.

A nonce set is `(seed, first_nonce, count)`. The seed is 32 random bytes drawn once per
job and stored in `job.json`. It is never placed in any prompt, log line or file the
agent can read.

### 7.5 Scoring rule (`talos/scoring.py`)

Lifted from tig-pentesting's `bundle_scoring.py`. Given baseline and candidate
`NonceResult` lists over the same nonces:

- Per nonce, the candidate wins if it produced a valid solution of strictly better
  quality, or equal quality in strictly less fuel; ties are ties; any candidate error is
  a loss.
- The bundle delta is the mean signed per-nonce quality difference normalised by the
  baseline quality, with each candidate error counted as the worst observed loss.
- `beats(baseline, candidate, margin)` is true when the delta exceeds the challenge's
  margin and the candidate's error rate is below the challenge's error ceiling.

Margins and error ceilings live in one table per challenge, versioned with Talos, never in
prompts.

### 7.6 Research loop (`talos/loop.py`)

Lifted from Prometheus `scripts/run_loop.py` single-shot mode with all server I/O removed.
One iteration:

1. Load `state.json`: current best files, best delta, runs since improvement, hypothesis
   log for the current best, tacit knowledge.
2. Build the hypothesis prompt from `prompts.py`: challenge brief, current best code, the
   direction text and tacit knowledge, the failed hypotheses tried against this exact best
   (recalled once stagnation passes a threshold), and the baseline delta target.
3. Hypothesis call returns a one-paragraph idea and a strategy tag.
4. Code call returns search-and-replace edits against the current best (Prometheus
   `search_replace.py`). Apply; refuse an edit that touches anything but the algorithm
   files.
5. Compile on Modal. On failure, feed the compiler output back for up to three fix
   rounds. Then log `failed:compile` and go to step 9.
6. Score on the training nonce set.
7. If `beats(baseline, candidate)`: enter confirmation (7.7). Else if the candidate beats
   the current best: it becomes the best, stagnation resets. Else stagnation increments.
8. Log the hypothesis outcome against the best it was tried on.
9. Stagnation handling, in order as the counter grows: recall failed hypotheses in the
   prompt; distill one tacit-knowledge lesson from the failures and append it to
   `tacit.md`; reset to the best with a forced change of strategy tag. Thresholds are
   config values with Prometheus's defaults.
10. Record spend, check budget, write `state.json`, append events, repeat.

Agentic mode replaces steps 3 to 5 with one headless `claude` or `codex` call inside a
git worktree containing only the algorithm files, `CHALLENGE.md`, `tacit.md` and a
`.talos/` scratch dir. The agent may edit algorithm files, run `talos compile`, and must
write `.talos/hypothesis.json` before stopping. Network is denied; the sandbox settings
are Prometheus's `sandbox-settings.json` with `cargo` replaced by `talos compile`. The
call is wall-clock bounded (default 1800 s).

### 7.7 Stop rule (`talos/stop.py`)

At job start, two disjoint nonce sets are drawn from the job seed: training (default 32
nonces) and held-out (default 32 nonces, a different nonce range). The baseline is scored
on both at start.

A candidate that beats the baseline on training moves the job to `confirming`: it is
scored on held-out. Win requires `beats` on held-out too. Then state is `won`. A miss
marks the candidate `false_positive`, keeps it as the current best for research purposes,
and returns to `researching`. A candidate is confirmed at most once.

Non-terminal states: `queued`, `measuring_baseline`, `researching`, `confirming`, and
`paused` (Modal unreachable past the retry window; resumable). Terminal states: `won`,
`exhausted` (any budget dimension hit), `failed` (unrecoverable error), `cancelled`
(Ctrl-C). Every terminal state builds the package.

### 7.8 Budget (`talos/budget.py`)

Dimensions: LLM dollars (API providers), iterations (CLI providers), wall-clock hours,
Modal dollars. Modal dollars are computed from measured container seconds times the
per-class rate in a table shipped with Talos, and marked "estimated" in the UI. Each
dimension is checked before every LLM call and before every Modal call. A check that
fails ends the run at the next safe point; a benchmark in flight completes and is
recorded. Spend is persisted in `state.json` after every call so a resume continues from
the true total.

### 7.9 Package (`talos/package.py`)

`runs/<job>/package/` contains: the best algorithm files; `diff_vs_baseline.patch`;
`scores.md` with per-nonce tables for baseline and candidate on training and held-out;
`hypotheses.md`, the full log with outcomes; `evidence_draft.md`, the TIG advance
evidence template with the challenge, method description from the winning hypotheses,
and benchmark section pre-filled, other sections left as the template's prompts;
`README.md` explaining how to submit. Also zipped as `package.zip`.

### 7.10 Resume

`talos run --resume <job_id>` reloads `job.json` and `state.json`, verifies the Modal app
and cached artifacts exist, and continues from the last completed step. An iteration
interrupted before its `score` completed is discarded and re-run.

## 8. Data flow for one job

1. `talos run` writes `job.json` and `state.json` (`queued`), draws the job seed and the
   two nonce sets.
2. `measuring_baseline`: resolve the top algorithm, compile, score on training and
   held-out, or load from cache. Record spend.
3. `researching`: iterate per 7.6.
4. `confirming` on a training beat; `won` on a held-out beat, else back to `researching`.
5. Terminal: build the package, print its path and the final delta, and for API
   providers, print the measured LLM and Modal spend.

## 9. Error handling

- **Baseline compile failure** aborts with the raw compiler output. Nothing downstream is
  meaningful. The likely cause is dev-image drift; the message says so and names the
  pinned digest.
- **Candidate compile failure** after three fix rounds is logged and skipped.
- **Per-nonce runtime errors** (`no_solution`, `invalid`, `out_of_fuel`, `panic`,
  `timeout`) score that nonce as a loss. A candidate over the error ceiling is logged as
  `failed:runtime` and skipped.
- **Modal errors** retry with exponential backoff up to a bounded window (default 15
  minutes). Beyond that the run pauses in a resumable state with a clear message; it does
  not spend LLM budget on iterations it cannot score.
- **Provider errors**: rate limits wait and retry; authentication and billing errors stop
  the run at once and name the credential to fix; malformed outputs (no edits, edits that
  do not apply) count as a failed iteration.
- **Edit scope**: any edit outside the algorithm files is rejected and the iteration
  fails, in both modes.
- **Seed hygiene**: the job seed appears in `job.json` only. Prompts, timeline, agent
  worktree and package contain nonce indices but never the seed.
- **Mainnet or GitHub unreachable** at start aborts before any spend, with the URL that
  failed.

## 10. Security and privacy

- Credentials: `.talos/secrets.json` is 0600 and gitignored; CLI providers store none.
  Secrets are never logged, never written to `runs/`, and never included in the package.
- Untrusted code runs only in Modal containers in the user's own account.
- The agentic sandbox has no network and can edit only algorithm files.
- Nothing leaves the user's machine except LLM calls to their provider and bench calls to
  their Modal account. Talos has no telemetry.

## 11. Testing

Every test names the mutation it catches.

- **Scoring**: per-nonce win/tie/loss and bundle delta on hand-built results. Mutation:
  flip the strict inequality on quality to non-strict; test fails on the equal-quality
  case.
- **Stop rule**: a training beat followed by a held-out miss must return to
  `researching` with `false_positive` recorded. Mutation: skip confirmation; test fails.
- **Budget**: spend exactly at the cap stops; one call under does not. Mutation: `>=`
  to `>`; the boundary test fails. Zero budget is a legitimate value and stops before the
  first call.
- **Edit application**: an edit outside the algorithm files is rejected. Mutation: drop
  the path check; test fails.
- **Loop with fakes**: a fake provider with scripted hypotheses and a fake bench with
  scripted per-nonce scores run a whole job in under a second. Covers stagnation
  thresholds, tacit distillation trigger, resume after a kill mid-iteration, and the
  false-positive path.
- **Wizards**: scripted stdin drives `setup` and `run`; asserts the files written, the
  0600 mode, that a rejected credential writes nothing, and that CLI providers write no
  secret.
- **Live smoke**, manual and marked slow: per challenge, compile and score the real
  mainnet top algorithm on Modal, assert result shape and that no nonce reports `panic`.
  Run before every release and whenever the pinned image digest changes.

## 12. Reuse map

| Talos module | Lifted from | Change |
|---|---|---|
| `baseline.py` | prometheus `server/mainnet_seed.py` | drop swarm reshape; keep files as-is |
| `prompts.py` | prometheus `scripts/prompts.py` | remove peer inspiration and server context; add baseline delta target |
| `search_replace.py` | prometheus `scripts/search_replace.py` | none |
| loop skeleton, tacit distillation, stagnation | prometheus `scripts/run_loop.py` | remove server I/O, HPO, cleaner, C3 |
| agentic sandbox | prometheus `.swarm/sandbox-settings.json`, `agentic_backends.py` | `cargo` allowlist replaced by `talos compile` |
| scoring | tig-pentesting `bundle_scoring.py` | none beyond interface |
| seed redaction, anti-hardcoding stance | tig-pentesting `scorer_api.py` | reimplemented in `bench.py` and `stop.py` |
| bench in the real harness | prometheus `c3_tig_bench.py` (JSON `test_algorithm` variant) | runs in the dev image on Modal instead of rebuilding a toolchain |

Prometheus is GPLv3. Talos will carry the same licence.

## 13. Defaults

| Setting | Default |
|---|---|
| Training nonces / held-out nonces | 32 / 32 |
| Fuel | the challenge's mainnet fuel setting at job start |
| Compile fix rounds | 3 |
| Stagnation thresholds: recall / distill / reset | 2 / 3 / 5 |
| Agentic call timeout | 1800 s |
| Modal retry window | 15 min |
| Budget | must be given; no default |

## 14. Open items deferred, not blocking

- Auto-submission with a TIG API key.
- More than one agent per job, and cross-job inspiration.
- A local dashboard page reading `timeline.jsonl`.
