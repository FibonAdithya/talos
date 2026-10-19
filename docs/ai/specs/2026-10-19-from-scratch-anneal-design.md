# From-scratch start and annealing search

Date: 2026-10-19. Status: design, not implemented. Non-authoritative (see `docs/ai/README.md`).

## 1. Why

Today every Talos job starts from the code of the mainnet top-adoption algorithm and climbs from
the current best, one candidate at a time. SkyDiscover reports that starting from an existing
system pushed its agent toward small tweaks, while starting from scratch gave it room to explore
fundamentally different designs. TIG has a local example of the second path: the AIDDA
Prometheus swarm started `neuralnet_optimizer` from the blank template
(`challenge_configs.initial_algorithm_code` in the 2026-08-12 export), and its best result was
submitted as `prometheus_aidda` (`c006_a036`), which mainnet shows as merged in round 125.
Talos's four real knapsack jobs, all starting from the mainnet top, ran 25 iterations and never
beat their baseline; the best was +0.12% on training (read from `runs/*/state.json` on
2026-10-19).

The goal is a Talos run that starts from TIG's empty template, develops its own design, and keeps
going until a candidate beats the current mainnet top on training and held-out nonces, or the
budget runs out.

## 2. What the user decided

- **Start point.** A new option starts the first candidate from the challenge's `template.rs`
  instead of the mainnet top algorithm's code.
- **The LLM sees the score, never the code.** The mainnet top is still measured once as the
  baseline and is still the bar. The LLM is told the baseline's name and its per-track scores,
  and the gap between those and its own candidates. It never sees the baseline's source.
- **No phase switch and no "close enough" threshold.** Exploration and refinement run side by
  side for the whole job. The user rejected a cut-over at some distance from the mainnet score
  as an arbitrary cap.
- **The gap to mainnet is always shown**: for the best candidate so far and for the parent each
  candidate is built from, in the prompt on every iteration, in the live CLI output, and in the
  hand-back package.
- **Lineage tracking.** Every candidate records its parent. The run is a tree, not a chain.
- **Simulated annealing acceptance.** A candidate that improves on its parent is always
  accepted. A worse candidate is accepted with a probability that falls with its quality drop.
  The temperature cools as the budget is spent and reheats after a stretch with no new best.
- **Design family is a label only.** Each candidate carries a short design-family name the LLM
  gives its approach (for example "rolling-horizon LP"). It is shown in the tree. It does not
  gate acceptance.
- **Parallel chains, configurable.** The number of annealing chains run in parallel is a flag.
- **Start point and search strategy are separate options.** Annealing and lineage also work
  when starting from the mainnet algorithm. Starting from the template also works with today's
  greedy search.
- **First target:** `energy_arbitrage` (`c008`, CPU) on C3, with £40 of C3 credit available and
  more addable.

## 3. Goals, non-goals, anti-goals

Goals:

1. `talos run --start template` writes the first candidate from `template.rs`.
2. `talos run --search anneal` runs the annealing search with a lineage tree; `--search greedy`
   is today's behaviour and stays the default.
3. `talos run --chains N` runs N chains concurrently under one job and one budget.
4. Every candidate's gap to the mainnet baseline is visible in prompts, CLI output and package.
5. A win means exactly what it means today (section 6).

Non-goals:

- Crossover or merging between chains or between design families. Later work.
- A hosted service, swarm server or cross-user sharing (still out of scope, per the 2026-09-11
  design).
- Changing any challenge's `BeatRule`, nonce counts, pins or price tables. Those need a human
  (`AGENTS.md`).
- Learned or embedding-based novelty measures.

Anti-goals (failures even if the feature "works"):

- The baseline's source reaching the LLM in a `--start template` job by any path: prompt,
  agentic worktree, transcript, tacit file, or a failed-attempt description that quotes code.
- A win that a candidate earned under a looser rule than today's, for example an accepted-worse
  candidate being confirmed or packaged as a win.
- Any existing invariant in `AGENTS.md` weakened without that being written down.

## 4. Options and defaults

| Flag | Values | Default | Frozen in `job.json` |
|---|---|---|---|
| `--start` | `mainnet`, `template` | `mainnet` | yes |
| `--search` | `greedy`, `anneal` | `greedy` | yes |
| `--chains` | integer >= 1 | 1 | yes |
| annealing settings (section 7) | numbers | ESTIMATES, section 7 | yes |

All four are frozen with the job, and `--resume` refuses a different value, as `--scoring` does
today. `--chains` applies to both search strategies: with `greedy`, N chains are N independent
hill climbs.

## 5. Start from the template

**What the LLM is given instead of the mainnet code.** With `--start mainnet` the LLM learns
the problem mostly by reading the algorithm it is editing. With `--start template` it gets the
challenge definition from the pinned monorepo (`MONOREPO_REF`):
`tig-challenges/src/<challenge>/README.md` and the non-baseline source files of that directory
(for energy arbitrage: `mod.rs`, `battery.rs`, `market.rs`, `network.rs`, `scenarios.rs`,
`constants.rs`, `utils.rs`). *My call:* the `baselines/` subdirectory (`greedy.rs`,
`conservative.rs`, `energy_v1/`) is not shown. It is the challenge's own reference policy, not a
mainnet algorithm, and quality is measured against it, but showing it would hand the LLM a
working design to tweak, which is what this mode exists to avoid. The README already describes
it in prose. In single-shot mode this text goes into the prompt; in agentic mode it is written
into the worktree as `CHALLENGE.md`, beside the template, and the read rules cover it.

**The first candidate.** Chain roots are the template itself. The template does not run
(`policy` returns "Not implemented"), so a root has no score. Until a chain has a scored
candidate, its parent is the template and every scored candidate is accepted as the chain's
first state.

**Whole-file writes.** Today's single-shot edit format forbids whole-file rewrites. A
`--start template` job needs them. *My call:* single-shot mode gains a whole-file response
format, allowed for any candidate whose parent is a chain root and for any candidate whose
design family is new to that chain; all other candidates use search/replace as today. The
scope rule (invariant 4) is unchanged: a file outside the algorithm files fails the iteration.
Agentic mode needs no new format.

**Hyperparameters.** *My call:* in a `--start template` job the baseline still runs with the
mainnet map pinned for it (`JobSpec.hyperparameters`), and candidates run with `None`. The
candidate is a different algorithm; the mainnet map belongs to the baseline's code
(`JobSpec.baseline_algorithm`) and its keys mean nothing to a design written from scratch. This
is a second deliberate exception to invariant 1's "identical hyperparameters" and must be
written into `AGENTS.md` invariant 1 and `docs/architecture.md`. The prompt's
`hyperparameters_block` is omitted for template starts.

**Which challenges** (*my call*). `--start template` is supported for `energy_arbitrage` only
at first. The list of challenge files shown to the LLM is per-challenge data in
`talos/challenges.py`, and `talos run --start template` refuses a challenge without one. Each
challenge added later brings its own file list and its own leak test, because other challenges
may keep a reference solver inline rather than under `baselines/`.

**One file set for the whole job** (*my call*). A candidate stays within the template's file set
(`mod.rs` for energy arbitrage) for the whole job. New file names would mean loosening
`talos/edits.py::_resolve` and `talos/agentic.py::read_back`, which `AGENTS.md` treats as
enforcement points; that is a separate decision for a human.

**No codex** (*my call*). `--start template` refuses `codex-cli` agentic mode, even with
`TALOS_ALLOW_CODEX_AGENTIC=1`. Codex ignores the sandbox settings and can read
`runs/<job>/state.json`, which holds the baseline's full source in `baseline.files`, and no grep
test can detect such a read. `claude-cli` agentic and single-shot are allowed.

**Keeping the baseline code out.** `talos/agentic.py::prepare_worktree` already places the
worktree outside `runs/`, and `runs/<job>/baseline/` holds the baseline source. The builder adds
tests in the style of the existing `rand_hash` greps: with `--start template`, take a
distinctive line from the baseline's files and assert it appears in no prompt, no timeline
entry, no agentic worktree file and no hand-back package file other than the baseline copy that
the package already carries. The package may keep the baseline source, since it is for the
user.

## 6. What a win is (unchanged)

The best candidate ever found, across all chains, is tracked separately from each chain's
current state. Only a candidate that would become the best goes through native-to-metered
validation (`Loop._validate_and_finish`), and only the best goes to held-out confirmation
(`Loop._confirm`) under the challenge's `BeatRule` via `beats` or `beats_focused`. An
accepted-worse candidate is never validated, confirmed or packaged as the result. "Becomes the
best" keeps today's rule: mean delta above the current best's, or beats the baseline.

**The best below mainnet.** The user asked to always see the gap from "the best discovered" to
mainnet, so in a `--start template` job the job-wide best is the highest-scoring candidate so
far, even when its `mean_rel_delta` is negative. Before any candidate has scored, there is no
best; `Loop._best_delta()` and `Loop._current_files()` must not fall back to the baseline's
score or files in a template start (today they return 0.0 and `state.baseline.files`). Nothing
is held-out confirmed until the best beats the baseline on training, as today.

**Validation of every new best** (*my call*). Every new job-best is validated on the metered
runtime, as today, including bests below mainnet. On a from-scratch climb that is frequent, and
each validation is a metered compile and score of the training nonces. On `energy_arbitrage`
(CPU, £0.11/h on C3) the extra cost is accepted in exchange for never building a long lineage on
code that runs out of fuel when metered. The validation count is reported in the package so its
cost can be judged after the first run.

**Native scores drive acceptance** (*my call*). Every tree node stores the score it was produced
with; in `--scoring native` jobs the accept rule compares native scores only, including for a
parent that has since been validated, so one scale is used throughout a chain. A metered score
is stored beside it where one exists. A candidate that fails validation (`fuel_proxy_miss`,
`nondeterministic`, `native_metered_build_mismatch`, `unscoreable`, `error_ceiling`) is recorded
as demoted and is also rejected as its chain's next state, so no chain builds on code that fails
TIG's metered runtime.

## 7. Annealing search

**Quality measure.** A candidate's score is its `mean_rel_delta` against the mainnet baseline
(`talos/scoring.py::bundle_delta`): per-track relative delta to the baseline's mean quality,
averaged over tracks. This is already "a fraction of the mainnet score", so the same
temperature means the same thing on every challenge. A candidate that fails to compile, fails
the dead-code guard, or exceeds the error ceiling has no score and is never accepted.

**Accept rule.** For candidate `c` built from parent `p` on a chain:

- `drop = score(p) - score(c)`.
- If `drop <= 0`, accept.
- Otherwise accept with probability `exp(-drop / T)`.
- An accepted candidate becomes the chain's current state. A rejected one is recorded in the
  tree as a leaf and the chain stays where it was.

**Temperature.**

- `T = T0 * cooling(f)`, where `f` is the fraction of the budget spent, taken as the largest
  spent fraction over the budget dimensions that have a cap (`talos/budget.py`).
- After `stall` consecutive iterations on a chain with no new best for that chain, the chain
  reheats: `T` is raised by a reheat factor (capped at `T0`) and the chain's current state
  jumps back to a strong node in its own subtree. Which node is left to the builder; the
  chain's best is the simple default.
- `cooling`, `T0`, the reheat factor and `stall` are settings frozen in `job.json`. Defaults are
  ESTIMATES with no measurement behind them. Starting values for the builder:
  `T0 = 0.05` (a 5% drop is accepted with probability 1/e at the start), linear cooling to
  `0.1 * T0` at budget exhausted, reheat to `T0`, `stall = 5`. The first energy-arbitrage run
  is where they get measured; its accept and reject log is the data.

**Randomness.** The accept draw uses a generator seeded from the job (derived from the job id
and the chain index, never from `rand_hash`, which must stay out of anything the agent can
read). Each draw and its probability are written to the tree, so a resume that replays the
state repeats the same decisions. Resume continues from the saved generator state.

**Parent choice.** In `anneal`, a chain's next candidate is built from its current state. In
`greedy`, from the chain's best, as today.

**What happens to today's stagnation tools.** *My call:* failed-hypothesis recall
(`prompts.py::recall_block`) is keyed to the parent node rather than to the job-wide best, so
"already tried against this exact code" stays true in a tree. Distillation into `tacit.md` stays
and is shared across chains, since lessons are about the problem. The forced strategy-tag
rotation (`Loop._forced_tag`) is not used with `--search anneal`; reheating replaces it.

## 8. Lineage tree

Each node records: node id, chain index, parent node id, iteration, design-family label (from
the LLM's hypothesis, a new required key next to `strategy_tag`), hypothesis title,
`mean_rel_delta` and per-track deltas (or the failure outcome), accepted or rejected, the accept
probability and draw for worse candidates, the temperature at the time, and whether it was ever
the job's best. File contents for accepted nodes must be recoverable, because a reheat can jump
back to any of them; where they are stored is left to the builder (the iteration directories
already hold them). `state.json` must stay small enough to write atomically every step
(invariant 6).

**Visible output.**

- Prompt, every iteration: the parent's gap to the mainnet baseline per track and mean, the
  best's gap, and the baseline's name. With `--start template` the wording says the baseline's
  code is not available.
- CLI live output: one line per iteration with chain, node, parent, design family, score, gap to
  mainnet, accepted or rejected (with probability when worse), and temperature.
- Package: the tree as JSON plus a readable summary (best path from root to the best node with
  each step's gap to mainnet, and a per-design-family table). The current package contents stay.

## 9. Parallel chains

- One job, one budget, N chains. Chains do not share a current state, nodes or recall; they
  share `tacit.md` and the job-wide best.
- **Budget (invariant 5; the user's decision).** Every compute and LLM call still checks the
  budget first, and a cap of 0 still stops before the first call. Costs are measured only after a
  call (`cost_mark` / `cost_usd_since` in `talos/bench.py` and `talos/c3_bench.py`; tokens after
  the reply), so today a run can end over its cap by at most one call. With N chains, N calls
  can pass the check together, so a run can end over its cap by at most one call per chain. The
  user accepted that bound on 2026-10-19; no pre-call cost estimator or reservation is built.
  `AGENTS.md` invariant 5 and `README.md#budget` must state the bound.
- **Resume.** `JobState.pending_job` is a single dict today. With chains, each chain needs its
  own pending record so a C3 job in flight for any chain can be reattached.
- **Local backend** (*my call*). On the `local` backend, chains run their LLM steps
  concurrently but compute calls one at a time. The baseline's timings, native budgets and
  per-nonce timeouts were measured with one container on the host; N containers sharing its CPUs
  would give each candidate less compute than the baseline had, which breaks invariant 1 in
  effect. On Modal and C3 each compute call gets its own container or machine, so chains run
  compute concurrently there.
- **C3 concurrency.** The Free plan caps concurrent C3 jobs at 3 chips and returns 409
  `CONCURRENCY_LIMIT` past that (measured 2026-09-23). More chains than slots must queue and
  retry rather than fail an iteration.
- **Win.** When the best confirms on held-out data, the job is `won`. Chains still in flight are
  stopped; whether their C3 jobs are cancelled or allowed to finish is left to the builder,
  provided their cost is still charged.

## 10. Constraints

All invariants in `AGENTS.md` hold, with the one written exception in section 5
(hyperparameters for template starts). In particular: baseline and candidates on identical
nonces, fuel and hardware class; `rand_hash` out of agent reach; pins in every cache key; budget
checked before every call; `state.json` atomic; costs shown as estimates.

## 11. Left to the builder

- Data layout of the tree and where node files live.
- The reheat target node and how `cooling` is shaped, within section 7.
- How the CLI line is formatted.
- How chains are run concurrently (threads, async, processes), provided state writes stay atomic
  and serialised.
- Whether chain-level stop on win cancels C3 jobs.

## 12. Measured and estimated numbers in this document

| Claim | Value | MEASURED / ESTIMATED | Source | When |
|---|---|---|---|---|
| `prometheus_aidda` merged | round 125 | MEASURED | `get-algorithms` on mainnet block `c6dd2306...` | 2026-10-19 |
| Swarm seed was the blank template | yes | MEASURED | `raw/swarm.db`, `challenge_configs` | 2026-10-19 |
| Talos knapsack jobs | 25 iterations, 0 won, best +0.12% | MEASURED | `runs/*/state.json` | 2026-10-19 |
| energy_arbitrage mainnet top | `gridflow` `c008_a052`, 74% adoption, merged round 139 | MEASURED | mainnet `get-algorithms` | 2026-10-19 |
| C3 CPU price | £0.11/h | from `talos/c3_bench.py::GBP_PER_HOUR` | code | 2026-10-19 |
| C3 Free-plan concurrency | 3 chips | MEASURED in an earlier session | `c3 whoami` | 2026-09-23 |
| Annealing defaults | section 7 | ESTIMATE (unverified) | none | n/a |
