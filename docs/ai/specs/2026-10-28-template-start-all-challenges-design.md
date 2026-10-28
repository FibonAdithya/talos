# `--start template` for every challenge

Date: 2026-10-28. Status: draft for approval.

## 1. Intent

`talos run --start template` lets the LLM write a solver from TIG's empty template instead of
editing the top mainnet algorithm. Today it is refused for every challenge but
`energy_arbitrage` (`talos/cli.py:1099`, `--start template supports energy_arbitrage only`).
The beta announcement should be able to say it works on any challenge. This change makes the
other seven challenges accept it: knapsack, job_scheduling, satisfiability, vehicle_routing
(CPU) and hypergraph, vector_search, neuralnet_optimizer (GPU).

The 2026-10-19 design (`docs/ai/specs/2026-10-19-from-scratch-anneal-design.md`, "Which
challenges") left this open on purpose: each challenge added later brings its own file list
and its own leak check, because a challenge may keep a reference solver inline instead of under
`baselines/`.

## 2. Goals, non-goals, anti-goals

Goals:

1. `--start template` is accepted for all eight challenges, with `--fake` and on a real backend.
2. A GPU template start compiles: the root holds a `.cu` file as well as `mod.rs`.
3. The LLM sees each challenge's definition files at the pinned monorepo commit, and never a
   file under `baselines/` or the mainnet baseline's code.
4. Resume, the package (`diff_vs_template.patch`, `submission/`) and `report.html` handle a
   root with more than one file.

Non-goals:

- Changing which files a candidate may edit during a job. The file set still never grows
  (`talos/edits.py`, `talos/agentic.py::read_back`); the GPU root simply starts with two files.
- Changing the codex-agentic refusal for template starts.
- Moving any challenge from `untested` to `tested`. That status means a job got past its
  baseline, and this change does not try to.
- Changing nonce counts, fuel, beat rules or pins.

Anti-goals:

- Any path that fetches or shows a `baselines/` file or the baseline algorithm's source.
- Editing the challenge files we show. They are quoted as the pinned source.

## 3. Facts this design rests on

| claim | value | MEASURED / ESTIMATED | command or file | when |
|---|---|---|---|---|
| Template start refused for knapsack | exit 2, `--start template supports energy_arbitrage only, not knapsack` | MEASURED | `talos run --fake --start template --challenge knapsack ...` from an `origin/main` checkout (9fdde7c) | 2026-10-28 |
| energy_arbitrage fake template job runs to a package | yes | MEASURED | same command with `--challenge energy_arbitrage` | 2026-10-28 |
| Challenge files at pin `839c718` | the lists in section 4 | MEASURED | `gh api repos/tig-foundation/tig-monorepo/git/trees/<pin>?recursive=1` | 2026-10-28 |
| GPU challenges ship `template.cu` beside `template.rs` | hypergraph, vector_search, neuralnet_optimizer | MEASURED | same tree listing | 2026-10-28 |
| GPU build fails with no `.cu` in the algorithm dir | `no .cu files in the <name> algorithm` | from code | `talos/inside.py:142-144` | 2026-10-28 |
| knapsack, job_scheduling, vehicle_routing reference solvers live under `baselines/` | `mod.rs` only calls `baselines::...` | MEASURED | grep of the fetched `mod.rs` files | 2026-10-28 |
| hypergraph has an inline greedy partitioner | `greedy_bipartition` kernel in `kernels.cu`, launched from `mod.rs` at instance generation; sets `greedy_baseline_connectivity_metric`, the quality zero point | MEASURED | grep of the fetched files | 2026-10-28 |
| No local GPU on the dev machine | `nvidia-smi: command not found` | MEASURED | `nvidia-smi -L` | 2026-10-28 |

## 4. File lists (decided)

Every non-`baselines/` file under `tig-challenges/src/<name>/` at the pin, by relative path:

| challenge | files |
|---|---|
| energy_arbitrage | unchanged |
| knapsack | `README.md`, `mod.rs` |
| job_scheduling | `README.md`, `mod.rs`, `scenarios.rs` |
| satisfiability | `README.md`, `mod.rs` |
| vehicle_routing | `README.md`, `mod.rs` |
| hypergraph | `README.md`, `mod.rs`, `kernels.cu` |
| vector_search | `README.md`, `mod.rs`, `kernels.cu` |
| neuralnet_optimizer | `README.md`, `mod.rs`, `kernels.cu`, `nn/mod.rs`, `nn/batch_norm.rs`, `nn/linear.rs`, `nn/mlp.rs`, `nn/tensor.rs`, `nn/kernels.cu` |

**Hypergraph shows its greedy partitioner** (the user's decision, 2026-10-28). Its `mod.rs` and
`kernels.cu` define how instances are generated and scored; the greedy is the quality zero
point, not a competitive design, and its kernel is compiled into the same PTX module as the
candidate's, so a candidate can call it whether or not the LLM sees it. The files are shown
unmodified.

## 5. The GPU root (decided)

For a GPU challenge the template root is `{"mod.rs": template.rs, "algorithm.cu": template.cu}`;
for a CPU challenge it stays `{"mod.rs": template.rs}`. `algorithm.cu` is my call: the build
globs `*.cu` in the algorithm directory (`talos/inside.py:142`), so any name compiles, and a
name other than `kernels.cu` keeps it distinct from the challenge's own `kernels.cu` in prompts.
TIG's metered build (`tig-binary/scripts/build_ptx` at the pin, lines 253 and 260) globs
`*.cu` too. Merged mainnet GPU algorithms use free-form names (`freud_opt/freud_opt.cu`,
`hyper_opt_extrem_v11/kernels_t22.cu`, `.../track_t21.cu`, MEASURED from the pinned tree
listing), so TIG does not require one fixed name. Whether the upload page itself checks the
name has not been tried (ESTIMATE (unverified)). No GPU template job exists yet, so a rename
later would break no stored job; after release it would, because `talos/loop.py::_node_files`
reads iteration files by the root's names.

Because the root has both files from iteration 1, whole-file and search/replace edits can
change either, and neither may add a file. That keeps `talos/edits.py` and
`talos/agentic.py::read_back` unchanged.

## 6. Where the code changes

1. `talos/challenges.py::TEMPLATE_START_FILES`: the seven new lists.
2. Template fetch: `talos/mainnet.py` gains a way to fetch `template.cu` for GPU challenges
   (one function returning the root files is fine), and `cli.FAKE_MAINNET` gets the fake
   equivalent so `--fake` GPU template jobs run.
3. Job creation (`talos/cli.py` around 1227-1276): write `challenge/template.cu` for GPU
   challenges, and create parent directories so `nn/linear.rs` etc. can be written (today
   `(docs_dir / name).write_text` raises for a nested name).
4. `talos/loop.py::_root_files`: build the root from the template files (section 5).
5. Resume (`talos/cli.py` 848-853): the `.cu` reaches the root after a resume. Read it from
   `runs/<job>/challenge/template.cu` or re-fetch it, the builder's call; the root must be
   identical either way.
6. `talos/package.py` (around 354) and `talos/report.py::run_sources` (around 121): use the
   same root, so `diff_vs_template.patch` and the report's root node include `algorithm.cu`.
   One helper that turns `runs/<job>/challenge/` into the root files, used by loop, package and
   report, is preferred over three copies.
7. Agentic mode: `talos/agentic.py::prepare_worktree` writes `CHALLENGE.md` quoting only
   `template.rs` (line 142). For a GPU challenge it also quotes `template.cu`. claude-cli
   agentic GPU template starts are in scope (my call); codex stays refused.
8. Docs: `docs/usage.md` (`--start` row and section), `docs/architecture.md:100`, and anything
   else that says "energy_arbitrage only". Mention that every new best of a template job gets a
   metered validation, which costs more on GPU challenges.

## 7. Tests

| test | mutation it catches |
|---|---|
| For every challenge in `TEMPLATE_START_FILES`, `fetch_challenge_docs` fetches no URL containing `/baselines/` | adding a `baselines/` entry to any list, or walking the directory |
| Every challenge in `CHALLENGES` has a `TEMPLATE_START_FILES` entry, and each list contains `README.md` and `mod.rs` | leaving a challenge out; a typo that drops `mod.rs` |
| A `live`-marked test lists the pinned tree and asserts each list equals the non-`baselines/` files of that challenge | a missed file (e.g. `scenarios.rs`), or a list gone stale after a pin bump |
| `--fake` template job for a CPU challenge other than energy_arbitrage (knapsack) runs to a package | the CLI refusal left in place |
| `--fake` template job for a GPU challenge: the first candidate's files include `algorithm.cu`, and `diff_vs_template.patch` names it | `_root_files` returning only `mod.rs`; package/report still using the single-file root |
| A fake neuralnet_optimizer template job writes `challenge/nn/linear.rs` and the prompt quotes it | the missing `mkdir(parents=True)`; `cli.py:838` reading only top-level names |
| Resume of a GPU template job keeps `algorithm.cu` in the root | resume rebuilding the root from `template.rs` only |
| `live`: fetching hypergraph's docs at the pin returns `kernels.cu` containing `greedy_bipartition`, and neuralnet_optimizer's returns `nn/linear.rs` | omitting a `.cu` or `nn/` file from a list; any filtering of fetched text |
| The GPU agentic worktree's `CHALLENGE.md` quotes `template.cu` | `prepare_worktree` still quoting only `template.rs` |

**Fixtures** (my call). Tests that run in `make check` use the fake mainnet, which returns a
short stub per listed file (extend `cli.FAKE_MAINNET.fetch_challenge_docs` so it returns one
entry per name in the challenge's list, not only `README.md`). They test the plumbing: lists,
nested paths, root files, resume, package, report. Checks on real file content run only under
the `live` mark against the pinned monorepo. No copies of TIG files are committed as fixtures.

## 8. Verification before merge (decided)

- `make check` at the branch tip.
- A `--fake --start template` job for each of the eight challenges, one iteration each.
- One real `--start template` job on knapsack with the `local` backend (Docker, free), 1-2
  iterations, to show a template candidate compiles and is scored by tig-runtime. No GPU
  template job runs for real in this change: there is no local GPU, and a paid run was not
  chosen.

## 9. Left to the builder

- The shape of the template-files helper and where it lives.
- Re-fetch vs read-from-disk for `template.cu` on resume (section 6.5).
- Prompt wording for the `.cu` in the template contract line (`talos/prompts.py:207`), as long
  as the LLM is told the `.cu` file holds its kernels.
