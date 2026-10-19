# From-scratch start and annealing search: implementation plan (part 1 of 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `talos run --start template --search anneal --chains N` writes the first candidate from TIG's empty template, explores with simulated annealing over a lineage tree, and still only wins under today's rule.

**Architecture:** A new pure module `talos/lineage.py` holds the tree, chain and annealing maths. `Loop` keeps one tree of nodes in `JobState`; every iteration belongs to one chain (taken in turns), builds on that chain's parent node, and records a node. The win path (validation, held-out confirmation, package) is unchanged and acts only on the job-wide best.

**Tech Stack:** Python 3.10, pytest, ruff (`make check`).

**Spec:** `docs/ai/specs/2026-10-19-from-scratch-anneal-design.md` (commit 51cc66f). Read it before any task.

**Scope of part 1.** Chains run **in turns, not concurrently**: iteration `n` belongs to chain `(n - 1) % chains`. That delivers the separate lineages with no change to budget checking, `pending_job`, C3 concurrency or local-backend scheduling. Spec section 9's concurrency items (budget overshoot of one call per chain, per-chain `pending_job`, C3 409 queueing, local serialisation) are **part 2**, a separate plan. Do not implement them here, and do not edit `AGENTS.md` invariant 5 in this plan.

## Global Constraints

- `make check` (ruff, `pytest -m "not live"`, agentify check) passes after every task. On this machine `python3` has no pytest and the repo `.venv` is 3.10. Wherever a step says `python3 -m pytest`, run it with a 3.11 venv: `uv venv --python 3.11 <dir> && uv pip install --python <dir>/bin/python -r requirements-dev.txt -e .`, then `make check PYTHON=<dir>/bin/python`. Baseline at 51cc66f (MEASURED 2026-10-19): 755 passed, 7 deselected; `tests/test_loop.py` collects 67.
- ruff's pinned rule set does not include E501. Check that no changed line exceeds 100 characters with a short Python `len(line) > 100` scan.
- Never run `tests/test_live.py`, Modal, C3 or a real LLM. Every test uses `FakeBench` / `FakeProvider` / fake mainnet.
- Mutation-check runs isolate bytecode (`PYTHONDONTWRITEBYTECODE=1` or a fresh `PYTHONPYCACHEPREFIX`), or a stale `.pyc` reports a false MISSED.
- Defaults that keep today's behaviour exactly: `start="mainnet"`, `search="greedy"`, `chains=1`. Every existing test must pass unmodified, except the prompt tests Task 5 names.
- Annealing defaults, all ESTIMATES: `t0 = 0.05`, `floor = 0.1` (T at the end of a cooling run as a fraction of `t0`), `stall = 5`.
- The accept draw is seeded from `job_id`, chain index and iteration, never from `rand_hash`.
- With `--start template` no byte of the baseline's source reaches a prompt, the timeline, an iteration directory, or an agentic worktree.
- Candidates in a `--start template` job run with hyperparameters `None`; the baseline keeps `JobSpec.hyperparameters`.
- `--start template` supports `energy_arbitrage` only; the file list lives in `talos/challenges.py::TEMPLATE_START_FILES`.
- `--start template` refuses `codex-cli` agentic, even with `TALOS_ALLOW_CODEX_AGENTIC=1`.
- One file set per job: a candidate has exactly its root's file names (`mod.rs` for a template start).
- Stage explicit paths only; never `git add -A` / `.`.

## Review Focus

1. **Resume in the middle of an annealing job.** A job killed after iteration 3 and resumed must make the same accept decisions and tree as one that never stopped. Test in Task 7.
2. **A template whose candidates never compile.** Chain stays at the root, stall counts up, reheat goes back to the root, nothing crashes, and the package says no candidate. Test in Task 8.
3. **A job created before this feature, resumed after it.** `job.json` and `state.json` without the new keys load as greedy / mainnet / one chain and run exactly as before. Test in Task 2.
4. **A hypothesis with no `design_family`.** It is labelled `unlabelled`; the first such hypothesis on a chain counts as a new family, later ones do not. Test in Task 8.
5. **More chains than iterations.** `--chains 3 --budget-iterations 2` leaves chain 2 untouched (its `current` and `best` stay at the root) and the package still builds. Test in Task 10.

---

### Task 1: `talos/lineage.py`, the pure tree and annealing maths

**Files:**
- Create: `talos/lineage.py`
- Test: `tests/test_lineage.py`

**Interfaces:**
- Consumes: `talos.budget.Budget`, `talos.budget.Spend`.
- Produces:
  - `ROOT: int = 0`, the node id of every chain's root.
  - `@dataclass(frozen=True) AnnealSettings(t0: float = 0.05, floor: float = 0.1, stall: int = 5)` with `to_dict() -> dict` and `from_dict(d: dict) -> AnnealSettings`.
  - `@dataclass Node(id: int, chain: int, parent: int, family: str, title: str, outcome: str, score: float | None = None, tracks: dict[str, float] | None = None, metered_score: float | None = None, accepted: bool = False, probability: float | None = None, draw: float | None = None, temperature: float | None = None, best: bool = False)`, with `to_dict` / `from_dict`. Also `rank_score(self) -> float | None`: `metered_score` if not None, else `score`.
  - `@dataclass Chain(index: int, current: int = ROOT, best: int = ROOT, best_score: float | None = None, stall: int = 0, reheated_at: float = 0.0)`, with `to_dict` / `from_dict`.
  - `chain_for(iteration: int, chains: int) -> int`
  - `spent_fraction(budget: Budget, spend: Spend, now: float) -> float`
  - `temperature(settings: AnnealSettings, spent: float, reheated_at: float) -> float`
  - `accept_probability(drop: float, t: float) -> float`
  - `draw(job_id: str, chain: int, iteration: int) -> float`
  - `decide(parent_score: float | None, score: float, t: float, u: float) -> tuple[bool, float | None]`

- [ ] **Step 1: Write the failing tests**

```python
import math
import pytest
from talos.budget import Budget, Spend
from talos.lineage import (AnnealSettings, Chain, Node, ROOT, accept_probability, chain_for,
                           decide, draw, spent_fraction, temperature)

S = AnnealSettings()

def test_chains_take_turns():
    # mutation: `iteration % chains` puts iteration 1 on chain 1
    assert [chain_for(n, 3) for n in range(1, 8)] == [0, 1, 2, 0, 1, 2, 0]
    assert chain_for(5, 1) == 0

def test_temperature_cools_linearly_to_floor():
    # mutation: dropping the floor term cools to 0, which rejects every worse candidate late on
    assert temperature(S, 0.0, 0.0) == pytest.approx(0.05)
    assert temperature(S, 0.5, 0.0) == pytest.approx(0.0275)
    assert temperature(S, 1.0, 0.0) == pytest.approx(0.005)

def test_reheat_restarts_cooling_over_the_remaining_budget():
    # mutation: ignoring reheated_at leaves T at the cooled value after a reheat
    assert temperature(S, 0.6, 0.6) == pytest.approx(0.05)
    assert temperature(S, 0.8, 0.6) == pytest.approx(0.0275)
    assert temperature(S, 1.0, 1.0) == pytest.approx(0.005)  # no division by zero at the end

def test_accept_probability():
    assert accept_probability(0.05, 0.05) == pytest.approx(math.exp(-1))
    assert accept_probability(0.0, 0.05) == 1.0
    assert accept_probability(-0.1, 0.05) == 1.0
    # mutation: exp(-drop / t) with t == 0 raises ZeroDivisionError
    assert accept_probability(0.01, 0.0) == 0.0

def test_spent_fraction_takes_the_largest_capped_dimension():
    b = Budget(usd=10, hours=None, iterations=20, compute_usd=None)
    assert spent_fraction(b, Spend(started_at=0.0, llm_usd=2.5, iterations=10), 0.0) == 0.5
    h = Budget(usd=None, hours=2, iterations=None, compute_usd=None)
    assert spent_fraction(h, Spend(started_at=0.0), 3600.0) == pytest.approx(0.5)
    # mutation: `if budget.compute_usd:` treats a 0 cap as no cap and returns 0.0
    z = Budget(usd=None, hours=None, iterations=5, compute_usd=0.0)
    assert spent_fraction(z, Spend(started_at=0.0), 0.0) == 1.0
    # clamped: an overshoot never gives a fraction above 1
    assert spent_fraction(b, Spend(started_at=0.0, llm_usd=50.0), 0.0) == 1.0

def test_draw_is_deterministic_and_keyed_by_chain_and_iteration():
    # mutation: random.random() makes a resumed job decide differently
    assert draw("j", 0, 3) == draw("j", 0, 3)
    assert 0.0 <= draw("j", 0, 3) < 1.0
    assert len({draw("j", c, n) for c in range(3) for n in range(1, 30)}) == 87

def test_decide():
    assert decide(None, -0.5, 0.05, 0.99) == (True, None)      # first scored node on a chain
    assert decide(-0.2, -0.1, 0.05, 0.99) == (True, None)      # improves on its parent
    # 0.0 - (-0.05) is exactly 0.05, so decide's p equals this p bit for bit. (-0.1 - -0.15 is
    # 0.04999999999999999, whose p is one ulp above exp(-1): a correct `u < p` would accept.)
    p = accept_probability(0.05, 0.05)
    assert p == math.exp(-1.0)
    assert decide(0.0, -0.05, 0.05, p - 1e-9) == (True, pytest.approx(p))
    # mutation: `u <= p` accepts a draw exactly at p
    assert decide(0.0, -0.05, 0.05, p) == (False, pytest.approx(p))

def test_round_trips():
    n = Node(id=4, chain=1, parent=2, family="lp", title="t", outcome="accepted", score=-0.1,
             tracks={"a": -0.1}, metered_score=-0.09, accepted=True, probability=0.4,
             draw=0.2, temperature=0.03, best=True)
    assert Node.from_dict(n.to_dict()) == n
    assert n.rank_score() == -0.09
    c = Chain(index=2, current=4, best=4, best_score=-0.09, stall=1, reheated_at=0.3)
    assert Chain.from_dict(c.to_dict()) == c
    assert AnnealSettings.from_dict(S.to_dict()) == S
    assert ROOT == 0
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `python3 -m pytest tests/test_lineage.py -q`
Expected: collection error, `ModuleNotFoundError: talos.lineage`.

- [ ] **Step 3: Implement `talos/lineage.py`**

Pure, no I/O, imports only `talos.budget`, `hashlib`, `math` and `dataclasses`. Formulas:
- `temperature`: `progress = 1.0 if reheated_at >= 1 else clamp((spent - reheated_at) / (1 - reheated_at), 0, 1)`; `T = t0 * (1 - (1 - floor) * progress)`.
- `spent_fraction`: the max over the dimensions whose cap `is not None`: `llm_usd / usd`, `(now - started_at) / (hours * 3600)`, `iterations / iterations cap`, `compute_usd / compute_usd cap`. A cap of 0 counts as 1.0. Clamp the result to [0, 1]; return 0.0 when no dimension is capped.
- `draw`: `int.from_bytes(sha256(f"{job_id}:{chain}:{iteration}".encode()).digest()[:8], "big") / 2**64`.
- `decide`: `parent_score is None` or `score >= parent_score` gives `(True, None)`; otherwise `p = accept_probability(parent_score - score, t)` and `(u < p, p)`.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `python3 -m pytest tests/test_lineage.py -q` → all pass.

- [ ] **Step 5: Commit**

```bash
git add talos/lineage.py tests/test_lineage.py
git commit -m "lineage: tree, chain and annealing maths"
```

---

### Task 2: Job fields for start, search, chains and the tree

**Files:**
- Modify: `talos/state.py` (`JobSpec`, `JobState`)
- Test: `tests/test_state.py`

**Interfaces:**
- Consumes: Task 1's `Node`, `Chain`.
- Produces: `JobSpec.start: str = "mainnet"`, `JobSpec.search: str = "greedy"`, `JobSpec.chains: int = 1`, `JobSpec.anneal: dict | None = None` (an `AnnealSettings.to_dict()`); `JobState.nodes: list[Node]` and `JobState.chains: list[Chain]`, both defaulting to `[]`, serialised as lists of dicts under the keys `"nodes"` and `"chains"`.

- [ ] **Step 1: Write the failing tests** in `tests/test_state.py`

`test_new_spec_fields_round_trip` builds a `JobSpec` with `start="template", search="anneal", chains=3, anneal=AnnealSettings().to_dict()`, round-trips it through `to_dict`/`from_dict` and asserts equality. Mutation caught: a field missing from the dataclass (constructor `TypeError`) or `from_dict` dropping one.

`test_tree_round_trips_through_state` puts one `Node` and two `Chain`s on a fresh `JobState`, saves with `JobStore.save`, loads, and asserts `loaded.nodes == state.nodes and loaded.chains == state.chains`. Mutation caught: `to_dict` omitting `"nodes"`.

`test_job_from_before_lineage_loads_with_defaults` (Review Focus 3): take a `JobSpec.to_dict()` and a `JobState.to_dict()`, delete the new keys (`start`, `search`, `chains`, `anneal`; `nodes`, `chains`), load both, and assert `(spec.start, spec.search, spec.chains, spec.anneal) == ("mainnet", "greedy", 1, None)` and `state.nodes == [] and state.chains == []`. Mutation caught: `d["nodes"]` instead of `d.get("nodes", [])`.

- [ ] **Step 2: Run them and confirm they fail** (`python3 -m pytest tests/test_state.py -q`: `TypeError` on the unknown fields).

- [ ] **Step 3: Add the fields.** `JobSpec` keeps `cls(**d)`; the new fields have defaults. `JobState.to_dict` writes `[n.to_dict() for n in self.nodes]` and the same for chains; `from_dict` reads them with `.get(..., [])`.

- [ ] **Step 4: Run `python3 -m pytest tests/test_state.py -q`** → pass. Run the full suite → pass.

- [ ] **Step 5: Commit** `talos/state.py tests/test_state.py`: "state: start, search, chains and the lineage tree".

---

### Task 3: Challenge files for a template start

**Files:**
- Modify: `talos/challenges.py`, `talos/mainnet.py`, `talos/cli.py` (`FAKE_MAINNET` only)
- Test: `tests/test_mainnet.py`

**Interfaces:**
- Produces:
  - `talos/challenges.py::TEMPLATE_START_FILES: dict[str, tuple[str, ...]] = {"energy_arbitrage": ("README.md", "mod.rs", "battery.rs", "market.rs", "network.rs", "scenarios.rs", "constants.rs", "utils.rs")}`
  - `talos/mainnet.py::fetch_challenge_docs(name: str, get_text=_get_text) -> dict[str, str]`, keyed by the names above, fetched from `f"{GH_RAW}/{MONOREPO_REF}/tig-challenges/src/{name}/{file}"`. It raises `MainnetError` for a challenge without an entry.
  - `FAKE_MAINNET.fetch_challenge_docs = lambda ch: {"README.md": "Fake challenge: maximise k.\n"}`

- [ ] **Step 1: Write the failing tests**

```python
def test_challenge_docs_come_from_the_pin_and_skip_the_reference_policy():
    urls = []
    docs = mainnet.fetch_challenge_docs("energy_arbitrage",
                                        get_text=lambda u: urls.append(u) or f"text of {u}")
    assert set(docs) == set(TEMPLATE_START_FILES["energy_arbitrage"])
    prefix = f"{mainnet.GH_RAW}/{MONOREPO_REF}/tig-challenges/src/energy_arbitrage/"
    # mutation: fetching at a branch name rather than the pinned commit
    assert urls and all(u.startswith(prefix) for u in urls)
    # mutation: walking the directory picks up baselines/greedy.rs, a working design to tweak
    assert not any("baselines" in u for u in urls)

def test_challenge_docs_refuse_a_challenge_without_a_file_list():
    # mutation: `TEMPLATE_START_FILES.get(name, ())` returns {} and a knapsack template job
    # would run with no problem definition at all
    with pytest.raises(mainnet.MainnetError):
        mainnet.fetch_challenge_docs("knapsack", get_text=lambda u: "x")
```

The file list was checked against the pin on 2026-10-19 (MEASURED, `gh api repos/tig-foundation/tig-monorepo/contents/tig-challenges/src/energy_arbitrage?ref=839c718…`): `README.md`, `battery.rs`, `constants.rs`, `market.rs`, `mod.rs`, `network.rs`, `scenarios.rs`, `utils.rs`, plus the `baselines/` directory. Together 57,621 bytes. The challenge's `mod.rs` has the same base name as the candidate's algorithm file `mod.rs`, so every place that shows these files labels them by their full monorepo path (Task 5).

- [ ] **Step 2: Run them and confirm they fail** (`AttributeError: fetch_challenge_docs`).
- [ ] **Step 3: Implement** the table, the function and the fake entry.
- [ ] **Step 4: Run `python3 -m pytest tests/test_mainnet.py -q`** → pass.
- [ ] **Step 5: Commit** `talos/challenges.py talos/mainnet.py talos/cli.py tests/test_mainnet.py`: "mainnet: challenge files for a template start".

---

### Task 4: Whole-file edit responses

**Files:**
- Modify: `talos/edits.py`, `talos/prompts.py` (the format constant only)
- Test: `tests/test_edits.py`

**Interfaces:**
- Produces:
  - `talos/prompts.py::WHOLE_FILE_FORMAT: str`, the instruction text. The block is the line `<<<<<<< FILE <name>`, then the full file, then the line `>>>>>>> END FILE`; one block per file; files not given are unchanged; only the algorithm files shown.
  - `talos/edits.py::apply_whole_file_response(files: dict[str, str], response_text: str) -> EditOutcome`: `applied` is the number of files replaced; `rejected` lists names that `_resolve` does not map into `files`; `misses` is always `[]`. It raises `EditError("response contained no FILE blocks")` when no block parses.

- [ ] **Step 1: Write the failing tests**

`test_whole_file_replaces_the_named_file`: `{"mod.rs": "old\n"}` plus one block for `mod.rs` holding `"fn a() {}\n"` gives `files == {"mod.rs": "fn a() {}\n"}` and `applied == 1`.
`test_whole_file_accepts_the_candidate_directory_spelling`: a block named `talos_cand/mod.rs` is applied to `mod.rs`.
`test_whole_file_rejects_a_new_file_name` (spec section 5, one file set): a block for `helpers.rs` gives `rejected == ["helpers.rs"]` and `"helpers.rs" not in outcome.files`. Mutation caught: adding unknown names to `files`.
`test_whole_file_rejects_another_algorithms_directory`: `other_algo/mod.rs` is rejected.
`test_whole_file_without_blocks_raises`: plain prose raises `EditError`. Mutation caught: returning an outcome with `applied == 0` instead of raising.
`test_whole_file_strips_a_code_fence_inside_the_block`: a block whose first content line is `` ```rust `` and last is `` ``` `` gives `files["mod.rs"]` without either fence line. Mutation caught: copying the block body verbatim, which hands the compiler a stray fence.

- [ ] **Step 2: Run them and confirm they fail** (`ImportError`).
- [ ] **Step 3: Implement** with a line scanner like `search_replace.parse_blocks`, tolerant of surrounding prose and code fences. Reuse `_resolve` for the scope decision.
- [ ] **Step 4: Run `python3 -m pytest tests/test_edits.py -q`** → pass.
- [ ] **Step 5: Commit** `talos/edits.py talos/prompts.py tests/test_edits.py`: "edits: whole-file responses within the algorithm file set".

---

### Task 5: Prompts: design family, gap to mainnet, template start

**Files:**
- Modify: `talos/prompts.py`
- Test: `tests/test_prompts.py`

**Interfaces:**
- Consumes: Task 4's `WHOLE_FILE_FORMAT`.
- Produces, on `PromptContext`:
  - `start: str = "mainnet"`
  - `challenge_docs: dict[str, str] = field(default_factory=dict)`
  - `parent_gap: dict | None = None` and `best_gap: dict | None = None`, each `{"node": int, "mean": float, "tracks": dict[str, float]}`. `None` means unscored, for example the template root or no best yet.
  - `whole_file: bool = False`
  - `anchor` keeps its name; it now means the parent node id (`0` is the root).
- New functions:
  - `gap_block(ctx) -> str`
  - `challenge_block(ctx) -> str`, empty unless `ctx.start == "template"`. Otherwise it is the line `The challenge definition (from the pinned TIG monorepo; read it, do not edit it):` followed by one section per doc, `README.md` first and the rest sorted, each headed `--- tig-challenges/src/{ctx.challenge}/{name} ---`. The full path keeps the challenge's own `mod.rs` apart from the algorithm file `mod.rs`. Task 9's `CHALLENGE.md` reuses `challenge_block`.
- `parse_hypothesis` returns a fourth key, `"design_family": str(d.get("design_family") or "unlabelled").strip()[:80] or "unlabelled"`.

Exact text the tests pin:
- `gap_block` with a parent gap: `f"Parent (node #{node}): {mean:+.3%} vs mainnet \"{baseline_name}\""`, followed by `; track {t} {d:+.3%}` per track in sorted order.
- With no parent gap: `"Parent: the empty template (not scored yet)"` for a template start. For a mainnet start, `"Parent: the mainnet baseline itself (+0.000%)"`.
- The same pattern with the prefix `Best so far` for `best_gap`, or `"Best so far: none yet"`.
- When `ctx.start == "template"`, the block ends with the sentence `"The baseline's code is not available to you; only its score is."`

- [ ] **Step 1: Write the failing tests**

`test_hypothesis_asks_for_a_design_family`: the system prompt contains `"design_family"`. Mutation caught: the key is never requested, so every node is `unlabelled`.
`test_parse_hypothesis_design_family`: a JSON reply with `design_family` `"rolling-horizon LP"` round-trips; a reply without it gives `"unlabelled"`.
`test_gap_block_shows_parent_and_best_per_track`: with the file's `ctx()` helper (its `baseline_name` is `"qk_v3"`), `parent_gap={"node": 3, "mean": -0.12, "tracks": {"b": -0.2, "a": -0.04}}` and `best_gap={"node": 5, "mean": -0.05, "tracks": {"a": -0.05, "b": -0.05}}`, assert the exact parent line `'Parent (node #3): -12.000% vs mainnet "qk_v3"; track a -4.000%; track b -20.000%'` is in `gap_block(ctx)`, and so is `"Best so far (node #5): -5.000%"`. Mutation caught: unsorted tracks, or the parent and best gaps swapped. With both gaps `None` and `start="mainnet"`, assert `"Parent: the mainnet baseline itself (+0.000%)"` and `"Best so far: none yet"`.
`test_template_start_shows_challenge_docs_and_hides_baseline_code`: `start="template"`, `challenge_docs={"README.md": "CHALLENGE-DOC"}`. Assert `"CHALLENGE-DOC"` and `"tig-challenges/src/knapsack/README.md"` are in the hypothesis user prompt and in the edit user prompt, and `"The baseline's code is not available to you"` is in the hypothesis prompt. With `start="mainnet"`, assert `challenge_block(ctx) == ""`.
`test_edit_prompt_switches_to_whole_file_format`: `whole_file=True` puts `WHOLE_FILE_FORMAT` in the edit system prompt and not `SEARCH_REPLACE_FORMAT`; `whole_file=False` does the reverse.
`test_hyperparameters_block_absent_for_template_start`: with `hyperparameters=None`, the hypothesis prompt has no `"Hyperparameters:"`.

- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement.**
  - `hypothesis_prompts`: replace only the second sentence of the first part, `"Your current best is {best_delta:+.3%} relative to it on the training nonces."`, with `gap_block(ctx)`. Keep `Baseline to beat: mainnet algorithm "{name}".`: `test_hypothesis_prompt_carries_direction_tacit_and_target` asserts the name is in the user prompt, and `gap_block` without gaps does not name it. Add `"design_family"` (a short name for the overall approach) to the JSON keys asked for. Keep the substring `"strategy_tag" (one of` in the system prompt, because `talos/cli.py::_fake_script` uses it to spot a hypothesis prompt. For a template start the system prompt says the solver is being written from the empty template, and `challenge_block` precedes the source.
  - `edit_prompts` and `compile_fix_prompts` also append `challenge_block(ctx)`.
  - `PromptContext.best_delta` stays, because the agentic `claude_md` reads it until Task 9.
  - The only existing tests allowed to change are the three `parse_hypothesis` equality asserts in `tests/test_prompts.py` (lines 82, 96 and 102 at 51cc66f). They compare whole dicts and now need `"design_family": "unlabelled"`. No existing test pins the old "Your current best is" sentence (checked: `grep -n "current best is" tests/` finds nothing).
- [ ] **Step 4: Run `python3 -m pytest tests/test_prompts.py -q`** → pass.
- [ ] **Step 5: Commit** `talos/prompts.py tests/test_prompts.py`: "prompts: design family, gap to mainnet, template start".

---

### Task 6: Loop records a lineage tree; greedy chains take turns

**Files:**
- Modify: `talos/loop.py`
- Test: `tests/test_loop.py`

**Interfaces:**
- Consumes: Tasks 1, 2 and 5.
- Produces, on `Loop`:
  - `_chain(n: int) -> Chain`, which creates `spec.chains` `Chain`s in `state.chains` on first use. **A job from before this feature** (Review Focus 3) has `state.chains == []` but may have `state.best`. In that case chain 0 is seeded with `current = best = state.best.iteration` and `best_score = _best_delta()`. Without that, a resumed old job's greedy parent would silently fall back to the baseline.
  - `_parent_id(chain: Chain) -> int`: `chain.best` for greedy, `chain.current` for anneal (Task 7).
  - `_root_files() -> dict[str, str]`: `state.baseline.files` for a mainnet start. Task 8 adds the template.
  - `_root_score() -> float | None`: `0.0` for a mainnet start.
  - `_node(node_id: int) -> Node | None`.
  - `_node_files(node_id: int) -> dict[str, str]`: the root files for `ROOT`; `state.best.files` when `state.best` is that node (this covers the seeded legacy best, which has no `Node`); otherwise each root file name read from `store.run_dir / "iterations" / f"{node_id:04d}"`. Do not call `store.iteration_dir`, because it creates the directory.
  - `_current_files()` returns the files of the chain's parent for iteration `self._n` (the iteration in progress; `iterate` and `_resume_pending` set `_n` before anything reads it).
- **A node's id is its iteration number.** Every iteration, failed ones included, appends exactly one `Node`, so `state.nodes[i].id == i + 1`. `ROOT` (0) is never in `state.nodes`.
- Each iteration record gains `chain`, `against` (the parent node id), `design_family`, `accepted`, `probability`, `draw` and `temperature`. That covers the record built in `iterate` and the one built in `_resume_pending`. A record without a parsed hypothesis has `design_family == ""`. `_finish_iteration` appends a `Node` built from the record.
- **Order inside `_finish_iteration`.** The node append and every chain update (current, best, best_score, stall, and Task 7's reheat) happen **before** its `self._save()`, and so before the `iteration_done` event and `_distill`. A stop requested from an `iteration_done` handler, or an `Interrupted` raised inside `_distill`'s LLM call, must not lose a chain update. Task 7's resume test depends on this.
- `_context()` sets `parent_gap` and `best_gap` for **every** start (moved here from Task 8). Task 5 removed the "current best" sentence, and without this step default jobs would lose it from the prompt. `parent_gap` is `{"node": id, "mean": node.rank_score(), "tracks": node.tracks}` for a non-root parent, and `None` for `ROOT`. For a seeded legacy best with no `Node`, use `state.best.delta`. `best_gap` comes from `state.best` (`delta["mean_rel_delta"]`, and per-track `rel_delta` from `delta["tracks"]`), or is `None`.

**Behaviour:**
- Iteration `n` uses chain `chain_for(n, spec.chains)`, and its parent is `_parent_id(chain)`.
- A candidate the job-wide rule calls improved (validated, in native mode) is accepted and becomes `chain.best` with `best_score = rank_score()`.
- A candidate that fails native validation is never accepted and never becomes a chain state, in either search (spec §6).
- In greedy, a candidate that is not a job-wide improvement is accepted only if its score beats `chain.best_score` (`None` counts as minus infinity). It then gets outcome `"accepted"` and becomes `chain.best`.
- With one chain and a mainnet start this never happens. `chain.best_score` equals the job's `_best_delta()`, and any score above it either beats that or beats the baseline (margin 0). A root's `best_score` is `_root_score()`. The path is reachable only with a template start or more than one chain; Task 8's `test_template_greedy_second_chain_keeps_its_own_worse_best` covers it.
- **Chain best, both searches:** whenever an accepted, scored node's `rank_score()` is above `chain.best_score`, it becomes `chain.best`. That includes Task 7's accepted-but-not-job-improved anneal nodes, such as a second chain's first scored node.
- `chain.stall` resets to 0 on a new chain best and otherwise goes up by one, failed iterations included.
- Recall (`_context().failed_hypotheses`) is filtered to the iteration's own chain, using `h.get("chain", 0)`.
- `_resume_pending` takes its parent from the chain's state, not from `_anchor()`. Delete `_anchor`.

- [ ] **Step 1: Write the failing tests**

First extend the helpers in `tests/test_loop.py`: `spec(..., **spec_kw)` and `make(..., spec_kw=None)` pass any extra `JobSpec` field through (`start`, `search`, `chains`, `anneal`, and `hyperparameters` for Task 8); existing calls are unchanged.

`test_default_job_is_unchanged_and_records_a_tree`: run the existing `test_win_requires_training_then_holdout_beat` script. Assert the run is still won at iteration 1, and `st.nodes[0].id == 1`, `st.nodes[0].parent == 0`, `st.nodes[0].chain == 0`, `st.nodes[0].accepted is True`, `st.nodes[0].best is True`, `st.chains[0].best == 1`.
`test_two_greedy_chains_take_turns_and_keep_their_own_parent`: `chains=2`, `iterations=4`, and a script where iteration 1 (chain 0) improves to k=5, iteration 2 (chain 1) edits from `let k = 1;` to k=3, iteration 3 (chain 0) edits from `let k = 5;`, and iteration 4 (chain 1) edits from `let k = 3;`. Assert `[n.chain for n in st.nodes] == [0, 1, 0, 1]` and `[n.parent for n in st.nodes] == [0, 0, 1, 2]`. A held-out score that never improves keeps the job running. Note that k=3 (+2%) also beats the baseline (margin 0), so iteration 2 becomes the job best under today's "mean above best, or beats" rule. That is expected and does not change the parents. Also assert that iteration 3's hypothesis prompt (from `fp.calls`) contains `"Parent (node #1): +4.000%"`. Mutation caught: building chain 1's iteration 2 on the job best (k=5) makes its SEARCH text miss, and the iteration fails as `failed:edit`; leaving `parent_gap` unset drops the parent line.
`test_recall_is_per_chain`: two chains, `Thresholds(recall=2, distill=99)` (no distill call eats a scripted reply). Iteration 1 (chain 0) fails with title `fail-c0`, iteration 2 (chain 1) fails with title `fail-c1`, then iteration 3 (chain 0) and iteration 4 (chain 1). In iteration 4's hypothesis prompt, assert `"fail-c1"` **is** present (positive control: recall is showing) and `"fail-c0"` is absent. Mutation caught: an unfiltered `_failed()`. The positive control catches recall never being shown, which would make the absence check pass vacuously.
`test_job_from_before_lineage_resumes_on_its_best` (Review Focus 3): build a state as a pre-lineage job would have it. Set `iteration = 1`, `best = Candidate(iteration=1, files={"mod.rs": "fn solve() { let k = 5; }\n"}, ...)` with a +4% delta, and `nodes == chains == []`, with no `iterations/0001/` on disk. Run one iteration with `edit(6, frm=5)`. Assert that iteration is not `failed:edit`, `st.nodes[0].parent == 1` and `st.nodes[0].id == 2`. Mutation caught: seeding chains at `ROOT` makes the SEARCH text `let k = 5;` miss against the baseline's `let k = 1;`; reading the best's files only from disk raises `FileNotFoundError`.
`test_parent_files_come_from_disk_after_a_restart`: run 1 iteration with `chains=1`, build a new `Loop` from `store.read_spec()` and `store.load()`, and assert `loop._current_files()` equals the iteration-1 files written under `iterations/0001/`. Mutation caught: keeping parent files only in memory.

- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement.** Keep `state.best`, `_best_delta()`, validation and `_confirm` exactly as they are. The tree is added beside them.
- [ ] **Step 4: Run `python3 -m pytest tests/test_loop.py -q`**, then the full suite → pass. All 67 existing loop tests must pass unchanged.
- [ ] **Step 5: Commit** `talos/loop.py tests/test_loop.py`: "loop: lineage tree and chains taken in turns".

---

### Task 7: Annealing acceptance and reheating

**Files:**
- Modify: `talos/loop.py`
- Test: `tests/test_loop.py`

**Interfaces:**
- Consumes: Task 6's chain plumbing, and Task 1's `temperature`, `decide`, `draw`, `spent_fraction`, `AnnealSettings`.
- Produces: events `accept` (`chain`, `node`, `parent`, `family`, `score`, `accepted`, `probability`, `temperature`) and `reheat` (`chain`, `node`, `temperature`).

**Behaviour with `spec.search == "anneal"`** (settings from `AnnealSettings.from_dict(spec.anneal)`):

1. A job-wide improvement is accepted as in Task 6. It is validated first in native mode, and a validated new best is always accepted.
2. A candidate that fails validation is rejected (`accepted=False`, outcome `failed:validation`).
3. Any other scored candidate (one with a `mean_rel_delta`, under the error ceiling):
   - Let `parent_score` be the parent node's `score`, or `_root_score()` for the root.
   - Compute `t = temperature(settings, spent_fraction(spec.budget, state.spend, clock()), chain.reheated_at)` and `u = draw(spec.job_id, chain.index, n)`.
   - Call `accepted, p = decide(parent_score, score, t, u)`.
   - The outcome is `"accepted"` when accepted, otherwise `"failed:score"`.
   - Record `probability`, `draw` and `temperature` on the node.
4. An accepted node becomes `chain.current`. A rejected one leaves the chain where it was. An accepted node above `chain.best_score` also becomes `chain.best` (Task 6's both-searches rule).
5. Inside `_finish_iteration`, after `spend.iterations` is incremented and `chain.stall` updated and **before its `_save()`** (Task 6's ordering rule): if `chain.stall >= settings.stall`, set `chain.current = chain.best`, set `chain.reheated_at` to the spent fraction at that point, and reset `chain.stall` to 0. Emit `reheat` after the save, beside `iteration_done`.
6. `_forced_tag()` returns `None` for anneal, and no `reset` event is emitted.

- [ ] **Step 1: Write the failing tests** (all `search="anneal"`, `anneal=AnnealSettings().to_dict()`, fake bench quality `100 + k - 1`, baseline quality 100)

**Why the worse candidates sit at or below baseline parity.** The beat rule's margin is 0 (`test_any_improvement_over_the_baseline_wins`), and "improved" is `mean > best or beats(baseline)`. A candidate at +4% after a +5% best therefore still beats the baseline and becomes the job best. It is not an annealing step. Every "worse" candidate below must score ≤ 0.0 against the baseline.

`test_anneal_accepts_a_worse_candidate_when_the_draw_is_under_p`:
- Iteration 1 moves k=1 to k=6 (+5%, a job best, accepted). Iteration 2 moves k=6 back to k=1 (`edit(1, frm=6)`): 0.0 against the baseline, so not a win, and a drop of 0.05 from its parent.
- Monkeypatch `talos.loop.draw` to return 0.0, so any worse candidate is accepted. At iteration 2's decision `spend.iterations == 1`, so T = 0.05 × (1 − 0.9 × 1/20) = 0.04775 and p = exp(−0.05/0.04775) ≈ 0.351 (ESTIMATE, arithmetic; the test reads T back from the node).
- Assert `st.nodes[1].accepted and st.nodes[1].outcome == "accepted"` and `st.chains[0].current == 2`.
- Assert `st.nodes[1].probability == pytest.approx(math.exp(-0.05 / st.nodes[1].temperature))`, `st.nodes[1].temperature == pytest.approx(0.04775)`, and `st.best.iteration == 1`. Mutation caught: making an accepted-worse node the job best; computing T from the post-increment iteration count.

`test_anneal_rejects_when_the_draw_is_over_p`: the same, with `draw` returning 0.999999. Assert `not st.nodes[1].accepted`, `st.nodes[1].outcome == "failed:score"` and `st.chains[0].current == 1`. Mutation caught: accepting every worse candidate.

`test_accepted_worse_candidate_is_never_confirmed`: in the accept case, assert `fb.holdout_runs == 1`. Only iteration 1, which beat the baseline, was confirmed. Mutation caught: confirming any accepted node.

`test_anneal_builds_on_the_current_state_not_the_best`:
- Continue the accept case with iteration 3, `edit(0, frm=1)` (k=0, −1%, not a win).
- Assert `st.nodes[2].parent == 2` and that iteration 3 is not `failed:edit`.
- Mutation caught: parent = `chain.best` (k=6) makes iteration 3's SEARCH text `let k = 1;` miss.

`test_stall_reheats_back_to_the_chain_best`:
- `AnnealSettings(stall=2)`; iteration 1 (k=6) is the job best.
- Iteration 2 (`edit(1, frm=6)`, 0.0) and iteration 3 (`edit(0, frm=1)`, −1%) are accepted-worse (draw 0.0), each lower than the last.
- Assert a `reheat` event follows iteration 3, `st.chains[0].current == 1`, `st.chains[0].stall == 0` and `st.chains[0].reheated_at == pytest.approx(3 / 20)` (iterations budget 20).
- Mutation caught: `>` instead of `>=` against `stall`; reheating without resetting the stall count.

`test_anneal_never_forces_a_strategy_tag`: six failing iterations with `Thresholds(reset=5)` produce no `reset` event and no "MUST be" sentence in any prompt.

`test_native_demoted_candidate_is_not_a_chain_state`:
- `scoring="native"`, iterations budget 1; the metered path fails validation with `fuel_proxy_miss`, using `FakeBench(metered_scores=...)` the way existing validation tests do. Mutation caught: running the anneal accept rule on a demoted candidate (decide(0.0, +x) would accept it).
- Assert the node has `accepted is False`, outcome `failed:validation`, and `st.chains[0].current == 0`.

`test_resumed_anneal_makes_the_same_decisions` (Review Focus 1):
- Both runs use `Thresholds(distill=99)`, so the list-scripted `FakeProvider` sees exactly two calls per iteration and the "advance past three iterations" offset is 6 replies. Choose a script where at least one of the 5 nodes is a worse candidate, so a real draw decides something. Assert that some `n.probability is not None`, otherwise the test compares nothing.
- Run 5 iterations straight through (real `draw`) and record `[(n.accepted, n.draw, n.parent) for n in st.nodes]`, plus `st.chains`.
- In a fresh `tmp_path`, run the same script with an `on_event` that calls `loop.request_stop()` on iteration 3's `iteration_done` event; the run ends `cancelled`.
- Build a new `Loop` from `store.read_spec()` and `store.load()`, with the provider script advanced past the first three iterations' replies. Set `state.status = "researching"`, as `execute_job` does on resume, and run on.
- Assert the five triples and the chains are equal. Mutation caught: an unseeded draw; a draw keyed on something a resume changes (for example `state.iteration` read after the increment). This test does **not** catch a chain update made after `_finish_iteration`'s save, because `run()`'s final `_save()` still persists the in-memory update after a soft stop. The next test covers that.

`test_tree_and_reheat_are_saved_before_iteration_done`: use the stall test's script with an `on_event` that, on each `iteration_done`, records `store.load()`'s `len(nodes)`, `chains[0].current` and `chains[0].stall`. Assert that after iteration 3 they read `(3, 1, 0)`, the reheat already applied. Mutation caught: appending the node or applying the reheat after the save, which a hard kill at that point would lose.

All Task 7 tests use a score function whose held-out nonces never improve (quality 100 when `ns.start >= 1_000_000`, as `test_false_positive_returns_to_research` does), so a training win is a false positive and the job keeps iterating. The stall check runs after `_finish_iteration` has incremented `spend.iterations`; that ordering is what makes `reheated_at == 3/20`.

- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement.** Import `draw` into `talos.loop`'s namespace (`from talos.lineage import draw`) so the tests can monkeypatch `talos.loop.draw`.
- [ ] **Step 4: Run `python3 -m pytest tests/test_loop.py -q`** and the full suite → pass.
- [ ] **Step 5: Commit** `talos/loop.py tests/test_loop.py`: "loop: simulated annealing acceptance and reheating".

---

### Task 8: Loop starts from the template

**Files:**
- Modify: `talos/loop.py`
- Test: `tests/test_loop.py`

**Interfaces:**
- Consumes: Tasks 3, 4, 5, 6 and 7.
- Produces:
  - `Loop.__init__(..., challenge_docs: dict[str, str] | None = None)`.
  - `_root_files()` is `{"mod.rs": self.template_rs}` for a template start.
  - `_root_score()` is `None` for a template start.
  - `_best_delta()` is `-math.inf` for a template start with no best.
  - `_candidate_hyperparameters() -> dict | None` is `None` for a template start, else `spec.hyperparameters`.

**Behaviour with `spec.start == "template"`:**
- `_request` passes `_candidate_hyperparameters()`. The baseline and calibration requests keep `spec.hyperparameters`.
- `_context()` sets `start`, `challenge_docs` and `hyperparameters=None`. The gaps are already set by Task 6.
- `single_shot_propose_and_edit`, **template starts only** (a mainnet start always uses search/replace): after the hypothesis is parsed, set `whole_file` to `ctx.anchor == ROOT or hypothesis["design_family"] not in {families of nodes on this chain}`. Build the edit prompt from `dataclasses.replace(ctx, whole_file=whole_file)`; never mutate the shared `ctx`, which `_score_candidate` reuses for compile-fix prompts. Then call `apply_whole_file_response` when `whole_file` is set, and `apply_edit_response` otherwise. A rejected path fails the iteration as today. Compile-fix rounds keep search/replace.
- `validation_failure` receives `_best_delta()`, so with no best any validated candidate counts as improved.
- `_mark_won`, `_confirm` and `beats` are untouched: a negative best is never confirmed.

- [ ] **Step 1: Write the failing tests**

Use a helper `make_template(tmp_path, script, ...)`. It wraps `make` with `spec(..., start="template", search="anneal", anneal=AnnealSettings().to_dict())` and `template_rs="pub fn solve_challenge() { todo!() }\n"`. It sets the baseline files to `{"mod.rs": "fn solve() { let k = 1; } // BASELINE-ONLY-7f3a\n"}` and passes `challenge_docs={"README.md": "doc"}`. Whole-file replies hold `fn solve() { let k = N; }`.

`test_template_start_first_candidate_is_written_whole_from_the_template`: the iteration-1 edit prompt contains `"pub fn solve_challenge() { todo!() }"` and `WHOLE_FILE_FORMAT`.

`test_template_start_never_shows_baseline_code`:
- Four iterations: one compile failure followed by a fix round (search/replace), one improvement, and two failures, so distillation runs at `Thresholds(distill=2)`.
- Assert `"BASELINE-ONLY-7f3a"` appears in no `fp.calls` entry, not in `timeline.jsonl`, and in no file under `iterations/`.
- Mutation caught: `_root_files()` returning `state.baseline.files`.

`test_template_candidates_run_without_hyperparameters`:
- `spec.hyperparameters={"t": {"x": 1}}`.
- Every candidate `EvalRequest` in `fb.calls` has `hyperparameters is None`.
- Mutation caught: `_request` passing `spec.hyperparameters`.

`test_template_best_below_mainnet_is_kept_and_shown`:
- Baseline quality 200; the first candidate scores k=11, quality 110, −45%.
- Assert `st.best.iteration == 1` and `st.best.delta["mean_rel_delta"] == pytest.approx(-0.45)`. Mutation caught: `_best_delta()` returning 0.0 drops every from-scratch best.
- Assert the next hypothesis prompt contains `"Best so far (node #1): -45.000%"`.
- Assert `fb.holdout_runs == 0`.

`test_template_new_family_gets_whole_file_and_same_family_gets_diffs` (Review Focus 4):
- Iteration 1 has family `"lp"`; iteration 2, family `"lp"`; iteration 3 has no family (`unlabelled`); iteration 4 is also `unlabelled`.
- Assert the edit system prompts use whole-file, search/replace, whole-file, search/replace.

`test_template_greedy_second_chain_keeps_its_own_worse_best` (the greedy chain-only acceptance path from Task 6, reachable only here):
- `start="template"`, `search="greedy"`, `chains=2`, baseline quality 200, iterations 3.
- Iteration 1 (chain 0, root) is a whole-file write with k=21: quality 120, −40%, the job best.
- Iteration 2 (chain 1, root) is a whole-file write with k=11: quality 110, −45%. That is not a job improvement, but it beats chain 1's `None`.
- Iteration 3 (chain 1) uses the same `design_family` as iteration 2 and `edit(12, frm=11)`.
- Assert `st.nodes[1].outcome == "accepted"`, `st.chains[1].best == 2`, `st.nodes[2].parent == 2`, iteration 3 is not `failed:edit`, and `st.best.iteration == 1`.
- Mutation caught: dropping the chain-only acceptance leaves chain 1 at `ROOT`. Iteration 3 is then whole-file, its search/replace reply does not parse as FILE blocks, and it fails `failed:edit` with parent 0.

`test_template_that_never_compiles` (Review Focus 2):
- `fb._compile_ok = lambda files: False`, `AnnealSettings(stall=2)`, 5 iterations.
- Use a callable `FakeProvider` script in the style of `talos/cli.py::_fake_script`: a hypothesis for hypothesis prompts, a whole-file block for `WHOLE_FILE_FORMAT` prompts, and a search/replace block that applies for compile-fix prompts. A list script drifts out of step with the compile-fix rounds and the distill call.
- The run ends `exhausted`, `st.best is None`, every node has `accepted is False`, there is at least one `reheat` event, and `st.chains[0].current == 0`. Also build the package and assert its README says no candidate. Mutation caught: a reheat that reads `chain.best_score` or a `ROOT` node's score crashes on `None`.

- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the tests** → pass; full suite → pass.
- [ ] **Step 5: Commit** `talos/loop.py tests/test_loop.py`: "loop: start from the empty template".

---

### Task 9: Agentic mode for template starts

**Files:**
- Modify: `talos/agentic.py`, `talos/cli.py` (`_codex_refused` only)
- Test: `tests/test_agentic.py`, `tests/test_cli.py`

**Interfaces:**
- Produces: `codex_agentic_refused(provider_kind: str, mode: str, start: str = "mainnet") -> bool`, which is true for codex agentic with `start == "template"` whatever the environment says. Also `TEMPLATE_CODEX_REFUSAL: str`, and `_codex_refused(cfg, start="mainnet")`.

- [ ] **Step 1: Write the failing tests**

`test_template_worktree_carries_challenge_docs`:
- `prepare_worktree(ctx)` with `start="template"`, files = the template, and docs `{"README.md": "DOC-TEXT"}`.
- `CHALLENGE.md` contains `"DOC-TEXT"`, the heading `tig-challenges/src/<challenge>/README.md`, and the template.
- Mutation caught: writing only `template_rs` to `CHALLENGE.md` (today's behaviour) leaves the agent with no problem definition.
- The baseline-source leak for agentic runs is covered by Task 8's `_root_files` test: `ctx.files` is the only route by which source code reaches the worktree.

`test_template_claude_md_shows_the_gap_and_family_key`: `claude_md(ctx)` contains `gap_block(ctx)` and `"design_family"`, and does not contain `"Your current best is"`. `claude_md` formats `ctx.best_delta`, and for a template start with no best that is `-inf` and prints `-inf%`, so the sentence is replaced by `gap_block`, not kept beside it. Mutation caught: appending the gap block and leaving the old sentence.

`CHALLENGE.md` for a template start is the template section as today, followed by `challenge_block(ctx)` (Task 5's headings). For a mainnet start it is unchanged.

`test_read_back_reads_design_family`: a `hypothesis.json` with `design_family` set round-trips; one without it gives `"unlabelled"`.

`test_codex_refused_for_template_even_when_opted_in`: with `monkeypatch.setenv("TALOS_ALLOW_CODEX_AGENTIC", "1")`, `codex_agentic_refused("codex-cli", "agentic", "template") is True` and `codex_agentic_refused("codex-cli", "agentic", "mainnet") is False`. Mutation caught: reusing the env opt-in.

- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement.** `attach_agentic` passes `loop.spec.start` to the refusal check.
- [ ] **Step 4: Run the tests** → pass.
- [ ] **Step 5: Commit** `talos/agentic.py talos/cli.py tests/test_agentic.py tests/test_cli.py`: "agentic: template starts; codex refused".

---

### Task 10: CLI flags, refusals and output lines

**Files:**
- Modify: `talos/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Produces: `talos run --start {mainnet,template} --search {greedy,anneal} --chains N`. Flags only; the wizard asks nothing new. **The argparse default of all three is `None`**, as `--scoring`'s is. A new job resolves `None` to `mainnet` / `greedy` / `1`. A resume refuses only a flag that was given and differs (`args.start is not None and args.start != spec.start`). With a non-`None` default, resuming a template or anneal job without repeating its flags would be refused. `JobSpec.anneal = AnnealSettings().to_dict()` when `--search anneal`, else `None`.
- `_codex_refused(cfg, start)` is called with the resolved start on a new job and with `spec.start` on a resume.
- With `--start template`, the challenge docs and the template (`fetch_challenge_docs`, `fetch_template`; `FAKE_MAINNET` under `--fake`) are fetched before `write_spec`. A fetch failure prints `mainnet unreachable` and returns 1, with no job created. They are written to `runs/<job>/challenge/<file>` and `runs/<job>/challenge/template.rs`. `execute_job` reads back only the `TEMPLATE_START_FILES` names and passes them as `Loop(challenge_docs=...)`.
- `_event_line` handles `accept`: `chain {c} #{node} <- #{parent} [{family}] {score:+.3%} vs mainnet: accepted (p=0.42, T=0.031)`. It shows `accepted (improved)` when `probability` is `None`, and `rejected (p=…, T=…)` when the candidate is rejected. It handles `reheat`: `chain {c} reheats from #{node}, T={t:.3f}`. `OUTCOME_TEXT["accepted"] = "kept as the chain's next step"`.

- [ ] **Step 1: Write the failing tests**

`test_start_template_refuses_an_unsupported_challenge`: `run --fake --yes --start template --challenge knapsack` returns 2, and stderr names `energy_arbitrage`.
`test_chains_must_be_positive`: `--chains 0` returns 2.
`test_resume_refuses_a_different_start_search_or_chains`: three cases, each returning 2. Mutation caught: dropping a frozen-field check.
`test_resume_without_the_flags_keeps_the_jobs_own`: resume a `--start template --search anneal --chains 2` fake job with none of the three flags and assert it is not refused (rc != 2). Mutation caught: an argparse default of `"mainnet"` / `"greedy"` / `1`.
`test_template_start_refuses_codex_agentic`: a config with `codex-cli` and agentic mode, with the opt-in env set, returns 2 before anything is spent.
`test_accept_and_reheat_event_lines`: `_event_line("accept", {...})` equals the exact strings above for the accepted, accepted-improved and rejected cases. `_event_line("reheat", ...)` likewise.

`test_fake_template_anneal_run_end_to_end` (Review Focus 5):
- A fake run on `energy_arbitrage` (`FAKE_MAINNET` scoring) with `--start template --search anneal --chains 3 --budget-iterations 2`.
- Extend `_fake_script` so a prompt with `WHOLE_FILE_FORMAT` in the system prompt gets a whole-file reply `fn solve() { let k = 1; }`, and `_fake_scores` reads k as today. That reply is at baseline parity: delta 0.0, which is a job best because a template start's floor is minus infinity, and never a win because the margin is strict. Iteration 1 runs on chain 0 and iteration 2 on chain 1.
- Assert `rc == 1` and status `exhausted`; `runs/<job>/challenge/README.md` and `runs/<job>/challenge/template.rs` exist; `[n.chain for n in state.nodes] == [0, 1]`; `state.chains[2].current == 0 and state.chains[2].best == 0`; and `package/lineage.json` exists (Task 11).
- Run this test after Task 11, or mark it `xfail` until then.

- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement.** Store `start`, `search`, `chains` and `anneal` in the `JobSpec` built in `cmd_run`. The "Job ..." summary line adds `start {start}, search {search}, {chains} chain(s)`.
- [ ] **Step 4: Run `python3 -m pytest tests/test_cli.py -q`** → pass.
- [ ] **Step 5: Commit** `talos/cli.py tests/test_cli.py`: "cli: --start, --search, --chains".

---

### Task 11: Package: lineage, template wording, validation count

**Files:**
- Modify: `talos/package.py`
- Test: `tests/test_package.py`

**Interfaces:**
- Produces:
  - `lineage_markdown(spec: JobSpec, state: JobState) -> str`
  - `package/lineage.json`: one object, `{"nodes": [n.to_dict() for n in state.nodes], "chains": [c.to_dict() for c in state.chains]}`. It is written whenever `state.nodes` is non-empty, including when there is no best, so it is written before `build_package`'s early returns.
  - For a template start, **no `hyperparameters.json`**. The candidate ran with none, and a file of the mainnet values beside the submission would invite submitting with them. The README section lists the baseline's values instead.
  - `package/lineage.md`.
  - For a template start, `diff_vs_template.patch` instead of `diff_vs_baseline.patch`, diffed against `{"mod.rs": <contents of runs/<job>/challenge/template.rs>}`, which Task 10 writes. `build_package`'s signature stays unchanged.

`lineage.md` holds:
- The path from the root to the best node: one line per node, `#id [family] title: {score:+.3%} vs mainnet`.
- A family table: `| family | nodes | accepted | best score |`.
- The line `Metered validations: {len(validations)} ({demoted} demoted)`.

README for a template start:
- Adds `"Written from TIG's empty template; the baseline's code was never shown to the LLM."`
- Omits the sentence about carrying over the baseline's Unique Algorithm Identifier.
- The hyperparameters section says the baseline ran with the listed values and the candidate ran with none.

- [ ] **Step 1: Write the failing tests**

`test_lineage_files_show_the_best_path_and_families`: a hand-built state with nodes 1 (root child, family lp, accepted), 2 (child of 1, family lp, accepted, best) and 3 (family greedy, rejected). Also assert `json.loads(lineage.json)` has keys `{"nodes", "chains"}` and three nodes. Assert the best-path lines for #1 and #2 in order, and #3 absent from the path. Assert the family table rows `| lp | 2 | 2 |` and `| greedy | 1 | 0 |` (the prefix of each row). Assert the validations line.
`test_template_package_diffs_against_the_template_and_drops_the_uai_line`: the test itself writes `runs/<job>/challenge/template.rs`, as Task 10 does for a real job, and `spec.hyperparameters` is set. `diff_vs_template.patch` exists and `diff_vs_baseline.patch` does not, and `hyperparameters.json` does not exist. The README lacks `"Unique Algorithm Identifier"` and contains `"never shown to the LLM"`. No package file other than `baseline`-named ones contains the baseline marker string. Mutation caught: diffing against the baseline, which hands the user a patch built from mainnet code.
`test_package_without_a_best_still_writes_lineage`: `state.best is None` with nodes present gives `lineage.json`.

- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run `python3 -m pytest tests/test_package.py tests/test_cli.py -q`** → pass, including Task 10's end-to-end test.
- [ ] **Step 5: Commit** `talos/package.py tests/test_package.py`: "package: lineage and template-start wording".

---

### Task 12: Documentation and the gate

**Files:**
- Modify: `AGENTS.md` (invariants 1 and 2 only), `README.md` (a section on `--start`, `--search` and `--chains`), `docs/architecture.md` (a new section, "From-scratch start and annealing search")
- Test: `tests/test_docs_references.py` (existing; it checks that every `file::symbol` reference resolves)

- [ ] **Step 1: Write the text.**
  - **Invariant 1** gains the written exception: with `--start template`, candidates run with hyperparameters `None` while the baseline keeps the mainnet map (`talos/loop.py::Loop._candidate_hyperparameters`).
  - **Invariant 2** gains: with `--start template` the baseline's source never reaches the LLM (`talos/loop.py::Loop._root_files`), tested in `tests/test_loop.py`. Codex is refused for template starts (`talos/agentic.py::codex_agentic_refused`).
  - **README:** the three flags, their defaults, the energy-arbitrage-only limit, and the line that chains take turns (concurrency comes later).
  - **Architecture:** the accept rule, the temperature schedule with the ESTIMATE defaults, reheating, and what the tree records.
- [ ] **Step 2: Run `make check`** → ruff clean, all non-live tests pass, agentify check passes. Paste the pytest summary line into the commit message body.
- [ ] **Step 3: Commit** `AGENTS.md README.md docs/architecture.md`: "docs: from-scratch start and annealing search".

---

## Audit record (2026-10-19)

Audited against the code at 51cc66f before dispatch; the fixes below are already applied in the tasks above.

| Sev | Task | Defect | Fix applied |
|---|---|---|---|
| Critical | 7 | The beat rule's margin is 0, so the "worse" k=5 candidate (+4%) after a +5% best beats the baseline, becomes the job best and is confirmed. `st.best.iteration == 1`, `holdout_runs == 1` and the reheat test fail against a correct implementation. | Worse candidates are k=1 (0.0) and k=0 (−1%), with explicit T and p arithmetic. |
| Critical | 6 | A pre-lineage job resumed with `state.best` set got fresh chains at `ROOT`, so greedy silently built on the baseline. Review Focus 3 tested loading only, never running. | Seed chain 0 from `state.best`; `_node_files` serves `state.best.files`; new test `test_job_from_before_lineage_resumes_on_its_best`. |
| Blocking | 1 | `-0.1 - (-0.15)` is `0.04999999999999999`, so `decide`'s p is one ulp above the test's p, and a correct `u < p` accepts where the test expects a reject. | Use parent 0.0 and score −0.05 (exact), with p from `accept_probability`. |
| Blocking | 5 | Three existing `parse_hypothesis` whole-dict asserts break once `design_family` is added, and the plan allowed no existing test to change except a sentence no test pins. | Allowance names those three asserts. |
| Blocking | 5 | The expected gap line used baseline `"b"`, but `tests/test_prompts.py::ctx` uses `"qk_v3"`. | Expected line uses `qk_v3`. |
| Blocking | 5/6/8 | Gaps were populated only in Task 8 under the template heading. After Task 5 removes the best sentence, default jobs would lose their best delta from the prompt. | Gap population moved to Task 6 for all starts, with an assertion in the two-chain test. |
| Blocking | 5/9 | The challenge-doc heading format was pinned by tests but never specified, and the challenge `mod.rs` collides with the algorithm's `mod.rs`. | Format specified (full monorepo path); `CHALLENGE.md` reuses `challenge_block`. |
| Blocking | 6/7 | Where chain, node and reheat updates sit relative to `_finish_iteration`'s save was unspecified, so a hard kill (or an `Interrupted` raised in `_distill`) could lose a reheat. | Ordering rule plus `test_tree_and_reheat_are_saved_before_iteration_done`. |
| Blocking | 7 | Anneal never made an accepted, non-job-improved node the chain best. A second chain's best stayed at `ROOT`, so it reheated to the template and its stall never reset. | Chain-best rule for both searches. |
| Blocking | 6/8 | The greedy chain-only acceptance path had no test, and it is unreachable with a mainnet start. | `test_template_greedy_second_chain_keeps_its_own_worse_best`. |
| Blocking | 6 | `test_recall_is_per_chain` only asserted an absence, so it passed if recall never showed. | Positive control plus `distill=99`. |
| Blocking | 10 | Unspecified argparse defaults: a `"mainnet"` default makes every resume of a template job be refused. | Defaults are `None`; new resume-without-flags test. |
| Blocking | 11 | The `lineage.json` shape was "a list plus a dict". A template package also kept `hyperparameters.json` of mainnet values the candidate never ran with. | `{"nodes", "chains"}` object; no `hyperparameters.json` for template starts. |
| Nit | 2 | `"rand_hash" not in spec.redacted()` was already true and caught nothing new. | Replaced with a stated mutation. |
| Nit | 4 | No test for a code fence inside a FILE block. | Added. |
| Nit | 7/8 | The resume test's list script drifts with distill; the never-compiles test's list script drifts with compile-fix rounds. | `distill=99`; callable script. |
| Nit | 6 | Node id = iteration was implicit. | Stated. |
| Nit | 8 | Whole-file scope (template only) and mutating the shared `ctx` were unstated. | Stated; use `dataclasses.replace`. |

Flagged and left alone:
- `talos/prompts.py::_parent` and `talos/package.py::_hypothesis_line` label `against` as "the baseline" / "best #N". In a template job node 0 is the template, and in anneal N is a node rather than a best. The wording is wrong but harmless, and changing it would edit tests outside this plan's allowance.
- In a multi-chain native job, greedy `chain.best_score` uses `rank_score()` (metered once validated) while unvalidated nodes carry native scores. Spec §6 asks for native scores throughout. A single chain is unaffected; settle it in part 2.
- Challenge docs add 57,621 bytes (MEASURED file sizes) to each hypothesis, edit and compile-fix prompt in a template job: about 15k tokens each (ESTIMATE, unverified). This is a cost the user should see after the first run, not a defect.
