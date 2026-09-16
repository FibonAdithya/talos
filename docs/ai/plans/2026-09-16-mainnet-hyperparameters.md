# Mainnet hyperparameters Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every nonce Talos runs, baseline and candidate alike, passes the per-track hyperparameters of the baseline algorithm's best-quality mainnet benchmark to `tig-runtime`, frozen into `job.json` at job start.

**Architecture:** `talos/mainnet.py` gains `top_hyperparameters`, which reads every player's recent benchmarks and picks the best one per track for one algorithm at one fuel. `talos run` resolves the top algorithm and its map once, before the job exists, and stores both in `JobSpec`. The map travels in `EvalRequest` to both backends (Modal starmap args, C3 `payload.json`) and ends as `--hyperparameters <json>` on the `tig-runtime` argv in `talos/inside.py::run_nonce`. The baseline cache key includes the map; the prompt and the package show it.

**Tech Stack:** Python 3.10+, pytest, ruff 0.16.7 (`select = ["E4","E7","E9","F"]`, no E501), agentify for the docs-reference test.

**Spec:** `docs/ai/specs/2026-09-16-mainnet-hyperparameters-design.md`

## Global Constraints

- Branch `mainnet-hyperparameters` in the worktree `/home/fibonadithya/TIG/Talos/.claude/worktrees/mainnet-hyperparameters`, based on PR #5's head `7bf0972`. Run every command from the worktree root. Never `cd` into the main checkout.
- Python: `PY=/home/fibonadithya/TIG/Talos/.venv/bin/python`. It imports `talos` from the worktree when run from the worktree root (checked 2026-09-16). Never `pip`; the venv is managed with `uv`.
- **Guarded test command.** Every pytest run, including every mutation check, goes through `.superpowers/sdd/t.sh`. The real `c3` CLI and Modal credentials are live on this machine, and a mutated cli test once submitted a billed C3 job. The script refuses subprocess calls through a pytest plugin, puts a shim first on PATH where `c3`, `modal`, `claude` and `codex` exit 97, and points Modal at bogus credentials. It exists, gitignored, with its plugin and shims under `.superpowers/sdd/`. If anything is missing, recreate it exactly as in Task 0. Wherever this plan writes `T <args>`, run `.superpowers/sdd/t.sh <args>`. It is a script, not a shell function, so it works across separate Bash calls.
  Full suite: `.superpowers/sdd/t.sh -m "not live" --ignore tests/test_contract.py --ignore tests/test_docs_references.py > /tmp/talos-hp-suite.log 2>&1; tail -3 /tmp/talos-hp-suite.log`.
  MEASURED at `267f6fc` (2026-09-16): `298 passed, 2 deselected`.
- Never run `talos run`, `talos compile` or `talos setup` outside pytest. Never run the `live` tests.
- No line over 100 characters. ruff does not check it. After each task run:
  `"$PY" -c "import sys,pathlib; [print(p,i+1,len(l)) for p in sys.argv[1:] for i,l in enumerate(pathlib.Path(p).read_text().splitlines()) if len(l)>100]" <changed .py files>` and expect no output.
- `"$PY" -m ruff check .` passes after every task.
- Every new test has a `# mutation:` comment naming the code change it catches (repo convention). After a task's tests pass, apply that mutation, confirm the named test fails under `.superpowers/sdd/t.sh`, then restore with `git checkout -- <file>` (the file must have no other uncommitted change at that point) and re-run to green.
- `rand_hash` never enters a prompt, event, package file or exception message. The mainnet benchmark's own `rand_hash` is never read into Talos state.
- `None` and `{}` are different hyperparameters: `{}` is passed to `tig-runtime` as `{}`. Never write `if hp:` for a single track's map; use `is None`.
- Commit per task with explicit paths (never `git add -A`, `git add .`, `git commit -a`). Run `git status --short` first. Commit messages follow the repo style `area: what changed` and end with
  `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`.
- Do not push. Do not open a PR. The orchestrator does both after Task 12.

---

### Task 0: Verify the guards (no commit)

**Files:** none tracked (`.superpowers/` is gitignored).

- [ ] **Step 1: Confirm the guard files exist, or recreate them**

```bash
ls .superpowers/sdd/t.sh .superpowers/sdd/no_subprocess.py .superpowers/sdd/shim/c3 \
   .superpowers/sdd/shim/modal .superpowers/sdd/shim/claude .superpowers/sdd/shim/codex
```

If any is missing:

```bash
mkdir -p .superpowers/sdd/shim
cat > .superpowers/sdd/no_subprocess.py <<'EOF'
"""pytest plugin: any subprocess call from a test fails it. The real `c3` CLI is logged in on
this machine; a mutated cli test once submitted a billed C3 job."""
import subprocess

import pytest


def _refuse(*args, **kwargs):
    raise AssertionError(f"subprocess call from a test: {args[:1]!r}")


@pytest.fixture(autouse=True)
def _no_subprocess(monkeypatch):
    for name in ("run", "Popen", "check_output", "check_call", "call"):
        monkeypatch.setattr(subprocess, name, _refuse)
EOF
for b in c3 modal claude codex; do
  printf '#!/bin/sh\necho "shim: real %s refused in tests" >&2\nexit 97\n' "$b" > .superpowers/sdd/shim/$b
  chmod +x .superpowers/sdd/shim/$b
done
```

and create `.superpowers/sdd/t.sh` with exactly this content, then `chmod +x .superpowers/sdd/t.sh`:

```sh
#!/bin/sh
# Guarded pytest for Talos: refuses subprocess calls and shims c3/modal/claude/codex (exit 97).
# Run from the worktree root. Arguments go to pytest.
root=$(pwd)
exec env PATH="$root/.superpowers/sdd/shim:$PATH" MODAL_CONFIG_PATH=/nonexistent \
  MODAL_TOKEN_ID=ak-invalid MODAL_TOKEN_SECRET=as-invalid PYTHONPATH="$root/.superpowers/sdd" \
  /home/fibonadithya/TIG/Talos/.venv/bin/python -m pytest -p no_subprocess -p no:cacheprovider -q "$@"
```

The plugin alone is not enough. `C3Bench`, `deploy_bench` and the CLI providers bind `subprocess.run` as a default argument at import time. The PATH shim catches those.

- [ ] **Step 2: Run the guarded suite and record the count**

Run: the full-suite command from Global Constraints.
Expected: `298 passed, 2 deselected`. If the count differs, stop and report it. Do not start Task 1.

---

### Task 1: `top_algorithm` returns the algorithm id

**Files:**
- Modify: `talos/mainnet.py` (`top_algorithm`)
- Modify: `talos/baseline.py` (`resolve_baseline`, the unpack of `top`)
- Modify: `talos/cli.py` (`FAKE_MAINNET`)
- Test: `tests/test_mainnet.py`, `tests/test_baseline.py`, `tests/test_cli.py`, `tests/test_loop.py`, `tests/test_live.py`

**Interfaces:**
- Produces: `mainnet.top_algorithm(name: str, get_json=...) -> tuple[str, str, int] | None`, returning `(algorithm_name, algorithm_id, adoption)`. The id is the `codes[].id` field, e.g. `"c003_a144"`.

- [ ] **Step 1: Change the expected value in the existing test**

In `tests/test_mainnet.py::test_top_algorithm_skips_uncompiled_and_other_challenges`, replace the assert and add a mutation line:

```python
    # mutation: dropping the challenge filter picks "other" (adoption 99)
    # mutation: returning the name where the id belongs gives precommit matching nothing to match
    assert mainnet.top_algorithm("vehicle_routing", get_json=fake_get_json) == (
        "fast_lane_v6", "a2", 50)
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `T tests/test_mainnet.py::test_top_algorithm_skips_uncompiled_and_other_challenges`
Expected: FAIL, `('fast_lane_v6', 50) != ('fast_lane_v6', 'a2', 50)`.

- [ ] **Step 3: Implement**

In `talos/mainnet.py`, replace `top_algorithm` with:

```python
def top_algorithm(name: str, get_json=_get_json) -> tuple[str, str, int] | None:
    """(algorithm_name, algorithm_id, adoption) of the highest-adoption compiled algorithm, or
    None. The id is what a benchmark's precommit names its algorithm by."""
    block_id = _block_id(get_json)
    challenges = get_json(f"{MAINNET_API}/get-challenges?block_id={block_id}")
    algos = get_json(f"{MAINNET_API}/get-algorithms?block_id={block_id}")
    cid = next((c["id"] for c in challenges["challenges"]
                if (c.get("config") or {}).get("name") == name), None)
    if cid is None:
        raise MainnetError(f"challenge {name!r} not found on mainnet")
    compiled = {b["algorithm_id"]: bool((b.get("details") or {}).get("compile_success"))
                for b in algos.get("binarys", [])}
    best: tuple[str, str, int] | None = None
    for algo in algos.get("codes", []):
        details = algo.get("details") or {}
        if details.get("challenge_id") != cid or not compiled.get(algo["id"]):
            continue
        try:
            adoption = int((algo.get("block_data") or {}).get("adoption") or 0)
        except (TypeError, ValueError):
            adoption = 0
        algo_name = details.get("name")
        if adoption > 0 and algo_name and (best is None or adoption > best[2]):
            best = (algo_name, algo["id"], adoption)
    return best
```

In `talos/baseline.py::resolve_baseline`, replace `name, adoption = top` with:

```python
    name, _algorithm_id, adoption = top
```

(Task 7 rewrites this block again. This step only keeps the suite green.)

In `talos/cli.py`, change the `FAKE_MAINNET` entry to:

```python
    top_algorithm=lambda ch: ("fake_base", "c003_a000", 1),
```

- [ ] **Step 4: Update every test double that returns the old 2-tuple**

- `tests/test_baseline.py`: `def fake_mainnet(top=("algo_x", "algo_x_id", 55)):`
- `tests/test_cli.py::_stub_mainnet`: `lambda ch: ("fake_base", "c003_a000", 1)`
- `tests/test_loop.py`: all four `top_algorithm=lambda ch: ("base", 1),` become `top_algorithm=lambda ch: ("base", "base_id", 1),` (currently near lines 336, 356, 374 and 591; find them with `grep -n 'top_algorithm=' tests/test_loop.py`).
- `tests/test_live.py`: `name, adoption = top` becomes `name, _algorithm_id, adoption = top`, and `name, _ = mainnet.top_algorithm(ch)` becomes `name, _algorithm_id, _adoption = mainnet.top_algorithm(ch)`. These are live tests; do not run them.

- [ ] **Step 5: Run the guarded full suite**

Expected: `298 passed, 2 deselected`. The count is unchanged because this task changes an assertion but adds no test.

- [ ] **Step 6: Mutation check**

Mutate `best = (algo_name, algo["id"], adoption)` to `best = (algo_name, algo_name, adoption)`. Expect `test_top_algorithm_skips_uncompiled_and_other_challenges` to FAIL. Restore it and re-run to green.

- [ ] **Step 7: Commit**

```bash
git status --short
git add talos/mainnet.py talos/baseline.py talos/cli.py tests/test_mainnet.py tests/test_baseline.py \
        tests/test_cli.py tests/test_loop.py tests/test_live.py
git commit -m "mainnet: top_algorithm also returns the algorithm id

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `top_hyperparameters`

**Files:**
- Modify: `talos/mainnet.py` (new dataclass and function; add `from statistics import mean`)
- Test: `tests/test_mainnet.py`

**Interfaces:**
- Consumes: `_block_id`, `_get_json`, `MAINNET_API` (existing).
- Produces:
  ```python
  @dataclass(frozen=True)
  class TrackHyperparameters:
      hyperparameters: dict | None
      benchmark_id: str | None
      player_id: str | None
      mean_quality: float | None
      def source(self) -> dict  # {"benchmark_id", "player_id", "mean_quality"}

  def top_hyperparameters(algorithm_id: str, tracks: list[str], fuel: int,
                          get_json=_get_json) -> dict[str, TrackHyperparameters]
  ```
  The result has exactly one key per entry of `tracks`. A track with no eligible benchmark maps to `TrackHyperparameters(None, None, None, None)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_mainnet.py`:

```python
FUEL = 5000000000000


def row(bid, player, algo, track, quality, hp, fuel=FUEL):
    """One benchmark as /get-benchmarks returns it: a precommit and, once submitted, a benchmark.
    `quality=None` models a precommit whose benchmark has not been submitted yet."""
    pre = {"benchmark_id": bid,
           "details": {"fuel_budget": fuel, "hyperparameters": hp, "rand_hash": "00" * 16},
           "settings": {"player_id": player, "algorithm_id": algo, "track_id": track}}
    bench = None if quality is None else {"id": bid,
                                          "details": {"average_quality_by_bundle": quality}}
    return pre, bench


def benchmarks_get_json(rows_by_player, frauds=(), failing=()):
    def gj(url):
        if url.endswith("/get-block"):
            return BLOCK
        if "/get-opow?block_id=b1" in url:
            return {"opow": [{"player_id": p} for p in rows_by_player]}
        if "/get-benchmarks?block_id=b1&player_id=" in url:
            player = url.split("player_id=", 1)[1]
            if player in failing:
                raise mainnet.MainnetError(f"HTTP 500 fetching {url}", status=500)
            rows = rows_by_player[player]
            ids = {pre["benchmark_id"] for pre, _ in rows}
            return {"precommits": [pre for pre, _ in rows],
                    "benchmarks": [b for _, b in rows if b is not None],
                    "proofs": [],
                    "frauds": [{"benchmark_id": f} for f in frauds if f in ids]}
        raise AssertionError(url)
    return gj


def test_top_hyperparameters_takes_the_best_benchmark_of_this_algorithm_across_players():
    gj = benchmarks_get_json({
        "0xp1": [row("b1", "0xp1", "a2", "T1", [100, 100], {"x": 1}),
                 row("b2", "0xp1", "a9", "T1", [900, 900], {"x": 9})],
        "0xp2": [row("b3", "0xp2", "a2", "T1", [150, 150], {"x": 2})],
    })
    got = mainnet.top_hyperparameters("a2", ["T1"], FUEL, get_json=gj)
    # mutation: dropping the algorithm_id filter picks b2 ({"x": 9}), another algorithm's keys
    # mutation: reading only the first player's benchmarks picks b1 ({"x": 1})
    assert got == {"T1": mainnet.TrackHyperparameters({"x": 2}, "b3", "0xp2", 150.0)}
    assert got["T1"].source() == {"benchmark_id": "b3", "player_id": "0xp2", "mean_quality": 150.0}


def test_top_hyperparameters_ignores_other_fuel_frauds_and_unsubmitted_benchmarks():
    gj = benchmarks_get_json({"0xp1": [
        row("ok", "0xp1", "a2", "T1", [100, 100], {"x": 1}),
        row("fuel", "0xp1", "a2", "T1", [999, 999], {"x": 2}, fuel=20000000000),
        row("fraud", "0xp1", "a2", "T1", [999, 999], {"x": 3}),
        row("pending", "0xp1", "a2", "T1", None, {"x": 4}),
        row("empty", "0xp1", "a2", "T1", [], {"x": 5}),
    ]}, frauds=("fraud",))
    got = mainnet.top_hyperparameters("a2", ["T1"], FUEL, get_json=gj)
    # mutation: dropping the fuel filter picks "fuel", values tuned for a different budget
    # mutation: dropping the fraud filter picks "fraud"
    # mutation: treating a missing or empty quality list as eligible raises or picks it
    assert got["T1"].benchmark_id == "ok" and got["T1"].hyperparameters == {"x": 1}


def test_top_hyperparameters_ranks_by_mean_over_bundles():
    gj = benchmarks_get_json({"0xp1": [
        row("first_high", "0xp1", "a2", "T1", [100, 10], {"x": 1}),
        row("mean_high", "0xp1", "a2", "T1", [60, 60], {"x": 2}),
    ]})
    got = mainnet.top_hyperparameters("a2", ["T1"], FUEL, get_json=gj)
    # mutation: ranking by the first bundle or by max() picks first_high (mean 55 < 60)
    assert got["T1"].benchmark_id == "mean_high" and got["T1"].mean_quality == 60.0


@pytest.mark.parametrize("order", [("bb", "ba"), ("ba", "bb")])
def test_top_hyperparameters_breaks_a_tie_on_the_lower_benchmark_id(order):
    gj = benchmarks_get_json({"0xp1": [row(b, "0xp1", "a2", "T1", [70, 70], {"id": b})
                                       for b in order]})
    # mutation: keeping the first (or last) seen makes the choice depend on API order
    assert mainnet.top_hyperparameters("a2", ["T1"], FUEL, get_json=gj)["T1"].benchmark_id == "ba"


def test_top_hyperparameters_covers_every_track_and_keeps_empty_apart_from_null():
    gj = benchmarks_get_json({"0xp1": [
        row("e", "0xp1", "a2", "T_empty", [10, 10], {}),
        row("n", "0xp1", "a2", "T_null", [10, 10], None),
        row("other", "0xp1", "a2", "T_not_asked", [10, 10], {"x": 1}),
    ]})
    got = mainnet.top_hyperparameters("a2", ["T_empty", "T_null", "T_missing"], FUEL, get_json=gj)
    # mutation: returning only tracks that had a benchmark raises KeyError downstream
    assert set(got) == {"T_empty", "T_null", "T_missing"}
    assert got["T_missing"] == mainnet.TrackHyperparameters(None, None, None, None)
    # mutation: `hp or None` collapses {} (run with an empty map) into None (run without the flag)
    assert got["T_empty"].hyperparameters == {} and got["T_empty"].hyperparameters is not None
    assert got["T_null"].hyperparameters is None and got["T_null"].benchmark_id == "n"


def test_top_hyperparameters_raises_when_one_player_cannot_be_read():
    gj = benchmarks_get_json({"0xp1": [row("b1", "0xp1", "a2", "T1", [1, 1], {"x": 1})],
                              "0xp2": []}, failing=("0xp2",))
    # mutation: skipping a failed player chooses from a partial view without saying so
    with pytest.raises(mainnet.MainnetError):
        mainnet.top_hyperparameters("a2", ["T1"], FUEL, get_json=gj)
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `T tests/test_mainnet.py -k top_hyperparameters`
Expected: FAIL (collection error or `AttributeError: module 'talos.mainnet' has no attribute 'TrackHyperparameters'`).

- [ ] **Step 3: Implement**

In `talos/mainnet.py`, add `from statistics import mean` after `from dataclasses import dataclass`. Add after `top_algorithm`:

```python
@dataclass(frozen=True)
class TrackHyperparameters:
    """One track's best mainnet benchmark for an algorithm. Every field is None when no benchmark
    qualified; `hyperparameters` alone is None when that benchmark ran without any."""
    hyperparameters: dict | None
    benchmark_id: str | None
    player_id: str | None
    mean_quality: float | None

    def source(self) -> dict:
        return {"benchmark_id": self.benchmark_id, "player_id": self.player_id,
                "mean_quality": self.mean_quality}


def top_hyperparameters(algorithm_id: str, tracks: list[str], fuel: int,
                        get_json=_get_json) -> dict[str, TrackHyperparameters]:
    """Per track, the hyperparameters of the highest mean-quality benchmark that ran
    `algorithm_id` at exactly `fuel`, among every player's benchmarks in mainnet's recent window.
    Frauds and benchmarks without a quality are skipped; ties go to the lower benchmark id. One
    player that cannot be read raises: the choice is never made from a partial view."""
    block_id = _block_id(get_json)
    opow = get_json(f"{MAINNET_API}/get-opow?block_id={block_id}")
    best: dict[str, tuple[float, str, dict]] = {}
    for player in [row["player_id"] for row in opow.get("opow", [])]:
        data = get_json(f"{MAINNET_API}/get-benchmarks?block_id={block_id}&player_id={player}")
        frauds = {f["benchmark_id"] for f in data.get("frauds") or []}
        quality = {b["id"]: (b.get("details") or {}).get("average_quality_by_bundle")
                   for b in data.get("benchmarks") or []}
        for pre in data.get("precommits") or []:
            bid = pre["benchmark_id"]
            settings, details = pre.get("settings") or {}, pre.get("details") or {}
            bundles = quality.get(bid)
            if (settings.get("algorithm_id") != algorithm_id
                    or settings.get("track_id") not in tracks
                    or details.get("fuel_budget") != fuel
                    or bid in frauds or not bundles):  # None or [] (a list): no quality to rank
                continue
            score = mean(bundles)
            held = best.get(settings["track_id"])
            if held is None or score > held[0] or (score == held[0] and bid < held[1]):
                best[settings["track_id"]] = (score, bid, pre)
    out: dict[str, TrackHyperparameters] = {}
    for track in tracks:
        if track not in best:
            out[track] = TrackHyperparameters(None, None, None, None)
            continue
        score, bid, pre = best[track]
        out[track] = TrackHyperparameters(pre["details"].get("hyperparameters"), bid,
                                          pre["settings"].get("player_id"), float(score))
    return out
```

- [ ] **Step 4: Run the new tests**

Run: `T tests/test_mainnet.py`
Expected: PASS. Six new test functions (seven cases, since the tie test runs twice) plus the existing tests.

- [ ] **Step 5: Mutation checks** (apply one at a time, confirm the named test fails, restore)

| mutation | failing test |
|---|---|
| delete `settings.get("algorithm_id") != algorithm_id or` | `..._best_benchmark_of_this_algorithm_across_players` |
| delete `or details.get("fuel_budget") != fuel` | `..._ignores_other_fuel_frauds_and_unsubmitted_benchmarks` |
| delete `or bid in frauds` | same |
| `score = mean(bundles)` → `score = bundles[0]` | `..._ranks_by_mean_over_bundles` |
| delete `or (score == held[0] and bid < held[1])` | `..._breaks_a_tie_on_the_lower_benchmark_id[order0]` |
| `pre["details"].get("hyperparameters")` → `pre["details"].get("hyperparameters") or None` | `..._keeps_empty_apart_from_null` |
| delete the `if track not in best:` branch's `out[track] = ...` line and `continue` | `..._keeps_empty_apart_from_null` |

- [ ] **Step 6: Line-length check, ruff, guarded full suite**

Expected: `305 passed, 2 deselected` (298 + 7 test cases).

- [ ] **Step 7: Commit**

```bash
git status --short
git add talos/mainnet.py tests/test_mainnet.py
git commit -m "mainnet: top_hyperparameters picks each track's best benchmark of one algorithm

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `run_nonce` passes `--hyperparameters` to `tig-runtime`

**Files:**
- Modify: `talos/inside.py` (`run_nonce`)
- Test: `tests/test_inside.py`

**Interfaces:**
- Produces: `inside.run_nonce(challenge_id, track, rand_hash, nonce, so, fuel, timeout_s, ptx, run=subprocess.run, workdir=None, clock=time.monotonic, hyperparameters: dict | None = None) -> dict`. Keyword-only in practice: every caller passes `hyperparameters=`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_inside.py`:

```python
def _capture_run(seen):
    def run(cmd, **kw):
        seen.append(cmd)
        if cmd[0] == "tig-runtime":
            (Path(cmd[cmd.index("--output") + 1]) / f"{cmd[3]}.json").write_text("{}")
        return Result(0, "quality: 1\n", "")
    return run


@pytest.mark.parametrize("hp, flag", [({"b": [2, 3], "a": 1}, '{"b":[2,3],"a":1}'), ({}, "{}")])
def test_run_nonce_passes_hyperparameters_to_the_runtime_only(tmp_path, hp, flag):
    seen = []
    inside.run_nonce("c003", "n=1", "ab" * 32, 7, Path("/lib/x.so"), 10, 600, None,
                     _capture_run(seen), tmp_path, hyperparameters=hp)
    rt, ver = seen
    # mutation: `if hyperparameters:` skips the flag for {}, which the benchmark ran with
    # mutation: json.dumps without compact separators still parses, but pins a different argv
    assert rt[rt.index("--hyperparameters") + 1] == flag
    # mutation: appending the flag to the verifier call makes clap reject every verification
    assert "--hyperparameters" not in ver


def test_run_nonce_without_hyperparameters_passes_no_flag(tmp_path):
    seen = []
    inside.run_nonce("c003", "n=1", "ab" * 32, 7, Path("/lib/x.so"), 10, 600, None,
                     _capture_run(seen), tmp_path)
    # mutation: always passing the flag sends "null", which tig-runtime rejects as not an object
    assert all("--hyperparameters" not in cmd for cmd in seen)
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `T tests/test_inside.py -k hyperparameters`
Expected: FAIL, `TypeError: run_nonce() got an unexpected keyword argument 'hyperparameters'`. The no-flag test passes already; that is expected.

- [ ] **Step 3: Implement**

In `talos/inside.py::run_nonce`, change the signature and docstring and build the flag:

```python
def run_nonce(challenge_id: str, track: str, rand_hash: str, nonce: int, so: Path, fuel: int,
              timeout_s: int, ptx: Path | None, run=subprocess.run,
              workdir: Path | None = None, clock=time.monotonic,
              hyperparameters: dict | None = None) -> dict:
    """Mirrors scripts/test_algorithm in the monorepo:
    `tig-runtime SETTINGS RAND_HASH NONCE SO --fuel F --output DIR [--hyperparameters JSON]
    [--ptx P --gpu 0]` writes DIR/<nonce>.json, then
    `tig-verifier SETTINGS RAND_HASH NONCE DIR/<nonce>.json [--ptx P --gpu 0]`
    prints `quality: N` and exits 0 on a valid solution. `{}` is passed as `{}`: the algorithm
    receives Some(empty map), not None."""
```

and replace the `cmd = [...]` assignment with:

```python
        hp_args = ([] if hyperparameters is None
                   else ["--hyperparameters", json.dumps(hyperparameters, separators=(",", ":"))])
        cmd = ["tig-runtime", settings, rand_hash, str(nonce), str(so),
               "--fuel", str(fuel), "--output", td] + hp_args + gpu_args
```

The verifier command is unchanged.

- [ ] **Step 4: Run tests**

Run: `T tests/test_inside.py`
Expected: PASS.

- [ ] **Step 5: Mutation checks**

| mutation | failing test |
|---|---|
| `if hyperparameters is None` → build `hp_args` with `if not hyperparameters` | `..._to_the_runtime_only[hp1]` |
| drop `separators=(",", ":")` | `..._to_the_runtime_only[hp0]` |
| `hp_args = ["--hyperparameters", json.dumps(hyperparameters, separators=(",", ":"))]` unconditionally | `test_run_nonce_without_hyperparameters_passes_no_flag` |

- [ ] **Step 6: Line-length check, ruff, guarded full suite**

Expected: `308 passed, 2 deselected`.

- [ ] **Step 7: Commit**

```bash
git status --short
git add talos/inside.py tests/test_inside.py
git commit -m "inside: run_nonce passes --hyperparameters to tig-runtime

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: `EvalRequest.hyperparameters` and the Modal path

**Files:**
- Modify: `talos/bench.py` (`EvalRequest`, new `hyperparameters_for`, `ModalBench._score`, `ModalBench.evaluate`)
- Modify: `modal_app/talos_bench.py` (`_score_impl`, `score_nonce`)
- Test: `tests/test_bench.py`, `tests/test_talos_bench.py`

**Interfaces:**
- Consumes: `inside.run_nonce(..., hyperparameters=...)` (Task 3).
- Produces:
  - `EvalRequest.hyperparameters: dict[str, dict | None] | None = None` (last field)
  - `bench.hyperparameters_for(hyperparameters: dict[str, dict | None] | None, track: str) -> dict | None`
  - Modal starmap argument tuples become `(artifact_id, track, rand_hash, nonce, fuel, timeout_s, hyperparameters)`.

- [ ] **Step 1: Update the three existing starmap doubles**

In `tests/test_bench.py`, all three `for (_a, t, _h, n, _f, _to) in args]` become `for (_a, t, _h, n, _f, _to, _hp) in args]`. Find them with `grep -n "_to) in args" tests/test_bench.py`.

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_bench.py`:

```python
def test_hyperparameters_for_a_track():
    from talos.bench import hyperparameters_for
    hp = {"t": {"x": 1}, "u": None, "e": {}}
    assert hyperparameters_for(hp, "t") == {"x": 1}
    # mutation: `or None` would turn a track's {} into None
    assert hyperparameters_for(hp, "e") == {}
    assert hyperparameters_for(hp, "u") is None and hyperparameters_for(hp, "missing") is None
    assert hyperparameters_for(None, "t") is None


def test_modal_starmap_carries_each_tracks_hyperparameters(monkeypatch):
    seen = []

    class Fn:
        def hydrate(self):
            pass

        def remote(self, files):
            return {"ok": True, "artifact_id": "art", "output": "ok"}

        def starmap(self, args):
            seen.extend((a[1], a[6]) for a in args)
            return [{"track": t, "nonce": n, "ok": True, "quality": 100, "runtime_ms": 1,
                     "error": None} for (_a, t, _h, n, _f, _to, _hp) in args]

    mod = types.ModuleType("modal")
    mod.exception = types.SimpleNamespace(NotFoundError=type("NotFoundError", (Exception,), {}))
    mod.Function = types.SimpleNamespace(from_name=lambda app, name: Fn())
    monkeypatch.setitem(sys.modules, "modal", mod)
    two = [NonceSet("t", "ab" * 32, 0, 1), NonceSet("u", "ab" * 32, 0, 1)]
    r = req(training=two, holdout=[])
    r.hyperparameters = {"t": {"x": 1}, "u": None}
    ModalBench().evaluate(r)
    # mutation: dropping the element from the args runs every nonce without hyperparameters
    # mutation: passing the whole map instead of the track's entry hands tig-runtime {"t": ...}
    assert seen == [("t", {"x": 1}), ("u", None)]
    seen.clear()
    ModalBench().evaluate(req(training=two, holdout=[]))
    assert seen == [("t", None), ("u", None)]
```

Append to `tests/test_talos_bench.py`:

```python
def test_score_impl_hands_the_hyperparameters_to_run_nonce(monkeypatch, tmp_path):
    _sandbox(monkeypatch, tmp_path)
    art = tmp_path / "artifacts" / "knapsack" / "art1"
    art.mkdir(parents=True)
    (art / "algo.so").write_bytes(b"\x7fELF")
    seen = {}

    def fake_run_nonce(*args, **kwargs):
        seen.update(kwargs)
        return {"track": args[1], "nonce": args[3], "ok": True, "quality": 1, "runtime_ms": 1,
                "error": None}
    monkeypatch.setattr(talos_bench.inside, "run_nonce", fake_run_nonce)
    talos_bench._score_impl("knapsack", "c003", "art1", "t", "ab" * 32, 0, 5, 60, {"x": 1})
    # mutation: accepting the argument but not forwarding it runs Modal nonces without the map
    assert seen["hyperparameters"] == {"x": 1}
```

- [ ] **Step 3: Run them and confirm they fail**

Run: `T tests/test_bench.py tests/test_talos_bench.py`
Expected: FAIL. `hyperparameters_for` cannot be imported, the starmap doubles fail to unpack 6-tuples into 7 names, and `_score_impl` takes too many positional arguments.

- [ ] **Step 4: Implement**

In `talos/bench.py`, add as the last field of `EvalRequest`:

```python
    # Per-track mainnet hyperparameters, passed to tig-runtime on every nonce of that track. A
    # track mapped to None, or absent, runs without the flag. None = no track gets any. The loop
    # and the baseline both take it from JobSpec.hyperparameters, never per request.
    hyperparameters: dict[str, dict | None] | None = None
```

Add after `timeout_for`:

```python
def hyperparameters_for(hyperparameters: dict[str, dict | None] | None,
                        track: str) -> dict | None:
    return None if hyperparameters is None else hyperparameters.get(track)
```

Replace `ModalBench._score` and the two `_score` calls in `ModalBench.evaluate`:

```python
    def _score(self, challenge: str, artifact_id: str, nonce_sets: list[NonceSet],
              fuel: int, timeouts: dict[str, int] | None,
              hyperparameters: dict[str, dict | None] | None) -> list[NonceResult]:
        args = [(artifact_id, ns.track, ns.rand_hash, n, fuel, timeout_for(timeouts, ns.track),
                 hyperparameters_for(hyperparameters, ns.track))
                for ns in nonce_sets for n in ns.nonces()]
```

(the rest of the method body is unchanged), and in `evaluate`:

```python
        tr = (self._score(request.challenge, c.artifact_id, request.training, request.fuel,
                          request.timeouts, request.hyperparameters)
              if request.training else [])
```

```python
            ho = (self._score(request.challenge, c.artifact_id, request.holdout, request.fuel,
                              request.timeouts, request.hyperparameters)
                  if request.holdout else [])
```

In `modal_app/talos_bench.py`:

```python
def _score_impl(name: str, challenge_id: str, artifact_id: str, track: str, rand_hash: str,
                nonce: int, fuel: int, timeout_s: int = NONCE_TIMEOUT_S,
                hyperparameters: dict | None = None) -> dict:
```

with its `return` becoming:

```python
    return inside.run_nonce(challenge_id, track, rand_hash, nonce, so, fuel,
                            min(timeout_s, NONCE_TIMEOUT_S), ptx if ptx.exists() else None,
                            workdir=MONOREPO, hyperparameters=hyperparameters)
```

and `_mk_score`:

```python
    def _mk_score(n=_name, cid=_spec.id):
        def score_nonce(artifact_id: str, track: str, rand_hash: str, nonce: int, fuel: int,
                        timeout_s: int = NONCE_TIMEOUT_S,
                        hyperparameters: dict | None = None) -> dict:
            return _score_impl(n, cid, artifact_id, track, rand_hash, nonce, fuel, timeout_s,
                               hyperparameters)
        return score_nonce
```

- [ ] **Step 5: Run tests**

Run: `T tests/test_bench.py tests/test_talos_bench.py`
Expected: PASS.

- [ ] **Step 6: Mutation checks**

| mutation | failing test |
|---|---|
| in `_score` args, replace `hyperparameters_for(hyperparameters, ns.track)` with `hyperparameters` | `test_modal_starmap_carries_each_tracks_hyperparameters` |
| in `evaluate`'s training call, pass `None` instead of `request.hyperparameters` | same |
| in `_score_impl`, drop `hyperparameters=hyperparameters` from the `run_nonce` call | `test_score_impl_hands_the_hyperparameters_to_run_nonce` |
| `hyperparameters_for` returns `(hyperparameters or {}).get(track) or None` | `test_hyperparameters_for_a_track` |

- [ ] **Step 7: Line-length check, ruff, guarded full suite**

Expected: `311 passed, 2 deselected`.

- [ ] **Step 8: Commit**

```bash
git status --short
git add talos/bench.py modal_app/talos_bench.py tests/test_bench.py tests/test_talos_bench.py
git commit -m "bench: EvalRequest carries per-track hyperparameters through the Modal starmap

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: The C3 path

**Files:**
- Modify: `talos/c3_jobdir.py` (`payload`)
- Modify: `talos/c3_job.py` (`_run_one`, `_score`)
- Test: `tests/test_c3_jobdir.py`, `tests/test_c3_job.py`

**Interfaces:**
- Consumes: `EvalRequest.hyperparameters` (Task 4), `inside.run_nonce(..., hyperparameters=...)` (Task 3).
- Produces: `payload.json` key `"hyperparameters"` (the request's map, or `null`). C3 task tuples become `(challenge_id, track, rand_hash, nonce, so, fuel, timeout_s, ptx, workdir, hyperparameters)`.
- Constraint: `talos/c3_job.py` runs inside the container with only `JOB_MODULES` shipped (`__init__, inside, scoring, types, challenges, diagnostics, c3_job`). It must NOT import `talos.bench`. Inline the per-track lookup.

- [ ] **Step 1: Update the existing pool-tuple test**

In `tests/test_c3_job.py::test_the_pool_path_passes_run_one_positional_tuples_and_sorts_the_rows`, the expected tuple gains a trailing `None`:

```python
    assert {t[3]: t for t in seen}[0] == ("c003", "t", HASH, 0, str(so), 7,
                                          inside.NONCE_TIMEOUT_S, None, str(mono), None)
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_c3_jobdir.py`:

```python
def test_payload_carries_the_hyperparameters(tmp_path):
    # mutation: dropping the map from the payload runs every C3 nonce without hyperparameters
    r = req()
    r.hyperparameters = {"t": {"x": 1}}
    assert c3_jobdir.payload(r)["hyperparameters"] == {"t": {"x": 1}}
    assert c3_jobdir.payload(req())["hyperparameters"] is None


def test_request_hash_changes_with_the_hyperparameters():
    # mutation: a hash that ignores the map lets a resume reattach to a job run without it
    with_hp = req()
    with_hp.hyperparameters = {"t": {"x": 1}}
    other_hp = req()
    other_hp.hyperparameters = {"t": {"x": 2}}
    hashes = {c3_jobdir.request_hash(r) for r in (req(), with_hp, other_hp)}
    assert len(hashes) == 3
```

Append to `tests/test_c3_job.py`:

```python
def test_each_track_gets_its_own_hyperparameters_in_the_pool_tasks(tmp_path, monkeypatch):
    mono, work, art = setup(tmp_path, n=1, baseline_q=200)
    payload = json.loads((work / "payload.json").read_text())
    payload["training"].append({"track": "u", "rand_hash": HASH, "start": 0, "count": 1})
    payload["hyperparameters"] = {"t": {"x": 1}, "u": None}
    (work / "payload.json").write_text(json.dumps(payload))
    seen = []

    def stub(task):
        seen.append(task)
        return {"track": task[1], "nonce": task[3], "ok": True, "quality": 120,
                "runtime_ms": 5, "error": None}
    monkeypatch.setattr(c3_job, "_run_one", stub)
    c3_job.main(workdir=work, artifacts_dir=art, run=fake_run(), monorepo=mono,
                log=lambda *a: None, pool_factory=FakePool)
    # mutation: putting the whole map in the task hands tig-runtime {"t": ..., "u": ...}
    assert sorted((t[1], json.dumps(t[9])) for t in seen) == [("t", '{"x": 1}'), ("u", "null")]


def test_the_serial_path_passes_the_hyperparameters_to_tig_runtime(tmp_path):
    mono, work, art = setup(tmp_path, n=1, baseline_q=200)
    payload = json.loads((work / "payload.json").read_text())
    payload["hyperparameters"] = {"t": {"x": 1}}
    (work / "payload.json").write_text(json.dumps(payload))
    run = fake_run()
    c3_job.main(workdir=work, artifacts_dir=art, run=run, monorepo=mono, log=lambda *a: None)
    runtime = [c for c in run.calls if c[0] == "tig-runtime"]
    # mutation: the serial branch not passing t[9] runs local and CPU jobs without the map
    assert runtime and all(c[c.index("--hyperparameters") + 1] == '{"x":1}' for c in runtime)
```

(`setup`, `fake_run`, `FakePool`, `HASH` and `json` already exist in `tests/test_c3_job.py`. `main` without `pool_factory` and with a non-`subprocess.run` runner takes the serial branch.)

- [ ] **Step 3: Run them and confirm they fail**

Run: `T tests/test_c3_jobdir.py tests/test_c3_job.py`
Expected: FAIL. `KeyError: 'hyperparameters'` for the payload; `tuple index out of range` for `t[9]`; `ValueError: 'hyperparameters' is not in list` in the serial test; the updated pool-tuple assert fails.

- [ ] **Step 4: Implement**

In `talos/c3_jobdir.py::payload`, add after `"timeouts": request.timeouts,`:

```python
            "hyperparameters": request.hyperparameters,
```

In `talos/c3_job.py`:

```python
def _run_one(task: tuple) -> dict:
    challenge_id, track, rand_hash, nonce, so, fuel, timeout_s, ptx, workdir, hp = task
    return inside.run_nonce(challenge_id, track, rand_hash, nonce, Path(so), fuel, timeout_s,
                            Path(ptx) if ptx else None, workdir=Path(workdir),
                            hyperparameters=hp)
```

In `_score`, replace the `tasks = [...]` assignment with:

```python
    timeouts = payload.get("timeouts") or {}
    # talos.bench is not shipped to the container, so the per-track lookup is inlined here: a
    # track mapped to None, or absent, runs without the flag; {} is passed as {}.
    hyperparameters = payload.get("hyperparameters") or {}
    tasks = [(payload["challenge_id"], ns["track"], ns["rand_hash"], n, str(so), payload["fuel"],
              timeouts.get(ns["track"], payload["nonce_timeout_s"]),
              str(ptx) if ptx else None, str(monorepo), hyperparameters.get(ns["track"]))
             for ns in sets for n in range(ns["start"], ns["start"] + ns["count"])]
```

(`payload.get("hyperparameters") or {}` is safe: it treats a whole-map `None` as empty, and per-track values are read with `.get` and never tested for truthiness.) In the serial branch:

```python
            r = inside.run_nonce(t[0], t[1], t[2], t[3], so, t[5], t[6], ptx, run=run,
                                 workdir=monorepo, hyperparameters=t[9])
```

- [ ] **Step 5: Run tests**

Run: `T tests/test_c3_jobdir.py tests/test_c3_job.py`
Expected: PASS.

- [ ] **Step 6: Mutation checks**

| mutation | failing test |
|---|---|
| remove `"hyperparameters": request.hyperparameters,` from `payload` | `test_payload_carries_the_hyperparameters`, `test_request_hash_changes_with_the_hyperparameters` |
| task element `hyperparameters.get(ns["track"])` → `payload.get("hyperparameters")` | `test_each_track_gets_its_own_hyperparameters_in_the_pool_tasks` |
| serial branch: drop `hyperparameters=t[9]` | `test_the_serial_path_passes_the_hyperparameters_to_tig_runtime` |

- [ ] **Step 7: Line-length check, ruff, guarded full suite**

Expected: `315 passed, 2 deselected`.

- [ ] **Step 8: Commit**

```bash
git status --short
git add talos/c3_jobdir.py talos/c3_job.py tests/test_c3_jobdir.py tests/test_c3_job.py
git commit -m "c3: the job payload and every nonce task carry the track's hyperparameters

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: `JobSpec` fields

**Files:**
- Modify: `talos/state.py` (`JobSpec`)
- Test: `tests/test_state.py`

**Interfaces:**
- Produces, after `track` on `JobSpec`:
  ```python
  baseline_algorithm: dict | None = None      # {"name": str, "id": str, "adoption": int}
  hyperparameters: dict[str, dict | None] | None = None
  hyperparameters_source: dict[str, dict] | None = None  # track -> TrackHyperparameters.source()
  ```

- [ ] **Step 1: Write the failing test**

Append to `tests/test_state.py`:

```python
def test_spec_hyperparameter_fields_round_trip_default_to_none_and_survive_redaction(tmp_path):
    pinned = replace(spec(), baseline_algorithm={"name": "a", "id": "c003_a1", "adoption": 9},
                     hyperparameters={"n=1": {"x": 1}},
                     hyperparameters_source={"n=1": {"benchmark_id": "b", "player_id": "0xp",
                                                     "mean_quality": 1.0}})
    store = JobStore(tmp_path)
    store.write_spec(pinned)
    # mutation: dropping a field loses the pinned algorithm or map on resume
    assert store.read_spec() == pinned
    old = spec().to_dict()
    for key in ("baseline_algorithm", "hyperparameters", "hyperparameters_source"):
        del old[key]
    # mutation: a field without a default makes every job.json written before it unresumable
    loaded = JobSpec.from_dict(old)
    assert (loaded.baseline_algorithm, loaded.hyperparameters,
            loaded.hyperparameters_source) == (None, None, None)
    red = pinned.redacted()
    # mutation: stripping them from the redacted spec hides from the agent what every run passes
    assert red["hyperparameters"] == {"n=1": {"x": 1}}
    assert red["baseline_algorithm"]["id"] == "c003_a1"
    assert "ab" * 32 not in json.dumps(red)
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `T tests/test_state.py`
Expected: FAIL, `TypeError: ... unexpected keyword argument 'baseline_algorithm'`.

- [ ] **Step 3: Implement**

In `talos/state.py::JobSpec`, after `track: str | None = None  # ...`:

```python
    # Pinned at job start by `talos run --hyperparameters mainnet` (the default). The map belongs
    # to this algorithm's code: resolve_baseline measures this algorithm rather than whatever
    # tops mainnet adoption by then. None on all three = no hyperparameters, as before.
    baseline_algorithm: dict | None = None  # {"name", "id", "adoption"}
    hyperparameters: dict[str, dict | None] | None = None  # track -> map passed to tig-runtime
    hyperparameters_source: dict[str, dict] | None = None  # track -> benchmark it came from
```

`from_dict` and `redacted` need no change: dataclass defaults cover old files, and `redacted` only removes the rand hash.

- [ ] **Step 4: Run tests**

Run: `T tests/test_state.py`
Expected: PASS.

- [ ] **Step 5: Mutation check**

| mutation | failing test |
|---|---|
| delete the `hyperparameters_source` line from `JobSpec` | `test_spec_hyperparameter_fields_round_trip_default_to_none_and_survive_redaction` (`TypeError` in `replace`) |
| in `JobSpec.redacted`, add `d.pop("hyperparameters")` before `return d` | same (`KeyError`) |

The default-to-None assert has no clean single-line mutation: removing one default makes every later field a `TypeError` at class creation. The round trip through a dict with the keys deleted is what guards old `job.json` files.

- [ ] **Step 6: Line-length check, ruff, guarded full suite**

Expected: `316 passed, 2 deselected`.

- [ ] **Step 7: Commit**

```bash
git status --short
git add talos/state.py tests/test_state.py
git commit -m "state: the job spec pins the baseline algorithm and its hyperparameters

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Baseline cache key, pinned algorithm, and the failure hint

**Files:**
- Modify: `talos/baseline.py` (`cache_key`, new `effective_hyperparameters`, `resolve_baseline`)
- Test: `tests/test_baseline.py`

**Interfaces:**
- Consumes: `EvalRequest.hyperparameters` (Task 4); `top_algorithm` 3-tuple (Task 1).
- Produces:
  ```python
  def effective_hyperparameters(hp: dict[str, dict | None] | None) -> dict[str, dict] | None
  def cache_key(challenge, monorepo_ref, name, training, holdout, fuel, hardware_class,
                hyperparameters=None) -> str
  def resolve_baseline(challenge, training, holdout, fuel, bench, cache_dir, hardware_class, rule,
                       mainnet=_mainnet, log=lambda msg: None, algorithm: dict | None = None,
                       hyperparameters: dict[str, dict | None] | None = None)
                       -> tuple[BaselineRecord, str]
  ```

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_baseline.py`:

```python
def test_cache_key_without_hyperparameters_is_unchanged_from_before_they_existed():
    # Value computed at 267f6fc with the pre-change cache_key. Existing cached baselines stay hits.
    # mutation: always putting "hyperparameters": None in the hashed payload changes this key
    key = cache_key("knapsack", "ref", "algo", TR, HO, 5, "cpu4-mem8192")
    assert key == "776fde051d7068800fc50361"


def test_cache_key_follows_what_the_hyperparameters_change_at_runtime():
    k = lambda hp: cache_key("knapsack", "ref", "algo", TR, HO, 5, "cpu4-mem8192", hp)  # noqa: E731
    plain = k(None)
    # tracks mapped to None run exactly as tracks without an entry, and as no map at all
    assert plain == k({}) == k({"t": None})
    # mutation: ignoring the map serves a baseline measured without it to a job that uses it
    assert k({"t": {"x": 1}}) not in (plain, k({"t": {"x": 2}}))
    # mutation: `if v` instead of `is not None` drops {} and collides it with running flagless
    assert k({"t": {}}) != plain


def test_a_pinned_algorithm_is_measured_instead_of_asking_mainnet(tmp_path):
    def no_top(ch, **kw):
        raise AssertionError("top_algorithm must not be called for a pinned algorithm")
    mn = fake_mainnet()
    mn.top_algorithm = no_top
    fb = FakeBench(lambda ch, files, ns: [10 for _ in ns.nonces()])
    hp = {"t": {"x": 1}}
    rec, _ = resolve_baseline("knapsack", TR, HO, 5, fb, tmp_path, "cpu4-mem8192", rule=BeatRule(),
                              mainnet=mn, algorithm={"name": "pinned", "id": "p1", "adoption": 7},
                              hyperparameters=hp)
    # mutation: calling top_algorithm again measures whatever tops mainnet now, not the algorithm
    # the frozen map belongs to
    assert (rec.name, rec.adoption, rec.files) == ("pinned", 7, {"mod.rs": "// pinned"})
    # mutation: not passing the map measures the baseline without it while candidates use it
    assert fb.calls[0].hyperparameters == hp


def test_an_unscoreable_baseline_with_hyperparameters_suggests_running_without(tmp_path):
    fb = FakeBench(lambda ch, files, ns: [None for _ in ns.nonces()])
    with pytest.raises(BaselineError) as ei:
        resolve_baseline("knapsack", TR, HO, 5, fb, tmp_path, "cpu4-mem8192", rule=BeatRule(),
                         mainnet=fake_mainnet(), hyperparameters={"t": {"x": 1}})
    # mutation: dropping the hint leaves the user guessing whether the map broke the algorithm
    assert "--hyperparameters none" in str(ei.value)
    with pytest.raises(BaselineError) as plain:
        resolve_baseline("knapsack", TR, HO, 5, fb, tmp_path, "cpu4-mem8192", rule=BeatRule(),
                         mainnet=fake_mainnet(), hyperparameters={"t": None})
    # mutation: hinting whenever the argument is not None blames a map that passed nothing
    assert "--hyperparameters none" not in str(plain.value)
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `T tests/test_baseline.py`
Expected: the pinned-key test PASSES now (it records the current value). The other three FAIL (`TypeError` on the extra arguments).

- [ ] **Step 3: Implement**

In `talos/baseline.py`, add before `cache_key`:

```python
def effective_hyperparameters(hp: dict[str, dict | None] | None) -> dict[str, dict] | None:
    """The per-track map as it changes a run. A track mapped to None runs exactly as a track with
    no entry, so both normalise away; a map with nothing left is None. A track's {} is kept: it is
    passed to tig-runtime and is not the same input as no flag."""
    if hp is None:
        return None
    kept = {track: v for track, v in hp.items() if v is not None}
    return kept or None  # the whole map, not a track's value: empty means "no flag anywhere"
```

Replace `cache_key`:

```python
def cache_key(challenge: str, monorepo_ref: str, name: str, training: list[NonceSet],
              holdout: list[NonceSet], fuel: int, hardware_class: str,
              hyperparameters: dict[str, dict | None] | None = None) -> str:
    h = hashlib.sha256()
    payload = {"challenge": challenge, "ref": monorepo_ref, "name": name, "fuel": fuel,
               "hw": hardware_class,
               "training": [(n.track, n.rand_hash, n.start, n.count) for n in training],
               "holdout": [(n.track, n.rand_hash, n.start, n.count) for n in holdout]}
    effective = effective_hyperparameters(hyperparameters)
    if effective is not None:
        # Only when present, so a key computed before hyperparameters existed is unchanged.
        payload["hyperparameters"] = effective
    h.update(json.dumps(payload, sort_keys=True).encode())
    return h.hexdigest()[:24]
```

In `resolve_baseline`, change the signature and the start of the body:

```python
def resolve_baseline(challenge: str, training: list[NonceSet], holdout: list[NonceSet],
                     fuel: int, bench, cache_dir: Path, hardware_class: str, rule,
                     mainnet=_mainnet, log=lambda msg: None, algorithm: dict | None = None,
                     hyperparameters: dict[str, dict | None] | None = None,
                     ) -> tuple[BaselineRecord, str]:
    if algorithm is not None:
        # Pinned at job start with the hyperparameters, which belong to this algorithm's code.
        name, adoption = algorithm["name"], algorithm["adoption"]
    else:
        top = mainnet.top_algorithm(challenge)
        if top is None:
            raise BaselineError(f"no adopted, compiled algorithm found on mainnet for {challenge}")
        name, _algorithm_id, adoption = top
    template = mainnet.fetch_template(challenge)
    key = cache_key(challenge, MONOREPO_REF, name, training, holdout, fuel, hardware_class,
                    hyperparameters)
```

Change the `bench.evaluate` call to pass the map:

```python
    r = bench.evaluate(EvalRequest(challenge=challenge, files=files, training=training,
                                   holdout=holdout, fuel=fuel, baseline_training=None, rule=rule,
                                   hyperparameters=hyperparameters))
```

Replace the two `_require_scoreable` lines with:

```python
    try:
        _require_scoreable(challenge, fuel, "training", training, tr)
        _require_scoreable(challenge, fuel, "held-out", holdout, ho)
    except BaselineError as e:
        if effective_hyperparameters(hyperparameters) is None:
            raise
        raise BaselineError(f"{e}. The baseline ran with mainnet hyperparameters; start a new job "
                            f"with --hyperparameters none to rule them out") from None
```

- [ ] **Step 4: Run tests**

Run: `T tests/test_baseline.py`
Expected: PASS.

- [ ] **Step 5: Mutation checks**

| mutation | failing test |
|---|---|
| always set `payload["hyperparameters"] = effective` (even when None) | `test_cache_key_without_hyperparameters_is_unchanged_from_before_they_existed` |
| delete the `if effective is not None:` block | `test_cache_key_follows_what_the_hyperparameters_change_at_runtime` |
| `if v is not None` → `if v` | same |
| ignore `algorithm` (always call `top_algorithm`) | `test_a_pinned_algorithm_is_measured_instead_of_asking_mainnet` |
| drop `hyperparameters=hyperparameters` from the `EvalRequest` | same |
| `if effective_hyperparameters(hyperparameters) is None:` → `if hyperparameters is None:` | `test_an_unscoreable_baseline_with_hyperparameters_suggests_running_without` |

- [ ] **Step 6: Line-length check, ruff, guarded full suite**

Expected: `320 passed, 2 deselected`.

- [ ] **Step 7: Commit**

```bash
git status --short
git add talos/baseline.py tests/test_baseline.py
git commit -m "baseline: key on the hyperparameters, measure the pinned algorithm with them

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: The prompts show the hyperparameters

**Files:**
- Modify: `talos/prompts.py` (`PromptContext`, new `hyperparameters_block`, `hypothesis_prompts`, `edit_prompts`)
- Modify: `talos/agentic.py` (`claude_md`, and its `from talos.prompts import` line)
- Test: `tests/test_prompts.py`, `tests/test_agentic.py`

**Interfaces:**
- Produces: `PromptContext.hyperparameters: dict[str, dict | None] | None = None` (last field); `prompts.hyperparameters_block(ctx: PromptContext) -> str`, which returns `""` when `ctx.hyperparameters is None`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_prompts.py`:

```python
HP = {"t": {"b": 2, "a": 1}, "u": None}


def test_hypothesis_and_edit_prompts_show_every_tracks_hyperparameters():
    c = ctx(hyperparameters=HP)
    _, hyp_user = hypothesis_prompts(c)
    _, edit_user = edit_prompts(c, {"title": "t", "description": "d"})
    for user in (hyp_user, edit_user):
        # mutation: leaving the block out of either prompt lets the model rename a key unawares
        assert 'track t: {"a":1,"b":2}' in user
        assert "track u: none (solve_challenge receives None)" in user
        assert "do not rename or remove" in user
    import re
    assert not re.search(r"\b[0-9a-f]{64}\b", hyp_user + edit_user)


def test_no_hyperparameters_block_without_a_map():
    _, user = hypothesis_prompts(ctx())
    # mutation: printing the block for None tells the model values are passed when none are
    assert "Hyperparameters:" not in user


def test_a_focused_prompt_shows_only_the_focus_tracks_hyperparameters():
    c = ctx(hyperparameters={"t": {"x": 1}, "u": {"x": 2}}, track="t", guard_tracks=["u"])
    _, user = hypothesis_prompts(c)
    # mutation: listing every track spends the focused prompt on tracks it must not tune for
    assert 'track t: {"x":1}' in user and '{"x":2}' not in user
    assert "guard tracks run with their own values" in user
```

Append to `tests/test_agentic.py`, which already has a `ctx(**kw)` helper that forwards keyword arguments to `PromptContext`:

```python
def test_claude_md_shows_the_hyperparameters():
    from talos.agentic import claude_md
    text = claude_md(ctx(hyperparameters={"t": {"x": 1}}))
    # mutation: the agentic brief omitting the block while single-shot prompts carry it
    assert 'track t: {"x":1}' in text and "do not rename or remove" in text
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `T tests/test_prompts.py tests/test_agentic.py`
Expected: FAIL, `TypeError: ... unexpected keyword argument 'hyperparameters'`.

- [ ] **Step 3: Implement**

In `talos/prompts.py::PromptContext`, add the last field:

```python
    hyperparameters: dict[str, dict | None] | None = None
```

Add after `focus_sentence`:

```python
def hyperparameters_block(ctx: PromptContext) -> str:
    """What every run passes to solve_challenge, so edits keep those keys readable. Empty when the
    job runs without hyperparameters. A focused job shows only its own track."""
    if ctx.hyperparameters is None:
        return ""
    shown = [ctx.track] if ctx.track else sorted(ctx.hyperparameters)
    lines = []
    for track in shown:
        hp = ctx.hyperparameters.get(track)
        value = ("none (solve_challenge receives None)" if hp is None
                 else json.dumps(hp, sort_keys=True, separators=(",", ":")))
        lines.append(f"track {track}: {value}")
    text = ("Hyperparameters: every run of this code, the baseline and your candidate alike, "
            "passes these values to solve_challenge as `hyperparameters`, per track. They came "
            "from the best mainnet benchmark of this algorithm. Keep every key readable: you may "
            "add keys, but do not rename or remove existing ones.\n" + "\n".join(lines))
    if ctx.track and len(ctx.hyperparameters) > 1:
        text += "\nThe guard tracks run with their own values too."
    return text
```

In `hypothesis_prompts`, insert before `parts.append("Current algorithm source:\n" + _files_block(ctx.files))`:

```python
    hp = hyperparameters_block(ctx)
    if hp:
        parts.append(hp)
```

In `edit_prompts`, replace the `user = (...)` assignment with:

```python
    focus = focus_sentence(ctx)
    hp = hyperparameters_block(ctx)
    user = (f"Implement this hypothesis:\nTitle: {hypothesis['title']}\n"
            f"Description: {hypothesis['description']}\n\n"
            + (focus + "\n\n" if focus else "")
            + (hp + "\n\n" if hp else "")
            + f"Current algorithm source files:\n{_files_block(ctx.files)}")
```

(`focus = focus_sentence(ctx)` already exists just above; do not duplicate it.)

In `talos/agentic.py`, add `hyperparameters_block` to the existing `from talos.prompts import (...)` list. In `claude_md`, after `focus = ...`, add:

```python
    hp = hyperparameters_block(ctx)
    hp = (hp + "\n\n") if hp else ""
```

and change `{focus}Rules:` in the template to `{focus}{hp}Rules:`.

- [ ] **Step 4: Run tests**

Run: `T tests/test_prompts.py tests/test_agentic.py`
Expected: PASS.

- [ ] **Step 5: Mutation checks**

| mutation | failing test |
|---|---|
| remove the `parts.append(hp)` insertion | `test_hypothesis_and_edit_prompts_show_every_tracks_hyperparameters` |
| remove `+ (hp + "\n\n" if hp else "")` from `edit_prompts` | same |
| delete the `if ctx.hyperparameters is None: return ""` guard, and write `(ctx.hyperparameters or {})` in the two places the function reads the map | `test_no_hyperparameters_block_without_a_map` (the header is printed with no lines) |
| `shown = sorted(ctx.hyperparameters)` unconditionally | `test_a_focused_prompt_shows_only_the_focus_tracks_hyperparameters` |
| `{focus}{hp}Rules:` → `{focus}Rules:` | `test_claude_md_shows_the_hyperparameters` |

- [ ] **Step 6: Line-length check, ruff, guarded full suite**

Expected: `324 passed, 2 deselected`.

- [ ] **Step 7: Commit**

```bash
git status --short
git add talos/prompts.py talos/agentic.py tests/test_prompts.py tests/test_agentic.py
git commit -m "prompts: show the hyperparameters every run passes, per track

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: The loop passes the spec's map everywhere

**Files:**
- Modify: `talos/loop.py` (`measure_baseline`, `_request`, `_context`)
- Test: `tests/test_loop.py`

**Interfaces:**
- Consumes: `JobSpec.baseline_algorithm`, `JobSpec.hyperparameters` (Task 6); `resolve_baseline(..., algorithm=, hyperparameters=)` (Task 7); `EvalRequest.hyperparameters` (Task 4); `PromptContext.hyperparameters` (Task 8).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_loop.py` (add `from dataclasses import replace` to the imports if absent):

```python
def test_the_spec_hyperparameters_reach_baseline_candidates_and_prompts(tmp_path):
    hp = {"t": {"x": 1}}
    b = Budget(usd=None, hours=None, iterations=1, compute_usd=None)
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(5)], budget=b)
    loop.spec = replace(loop.spec, hyperparameters=hp,
                        baseline_algorithm={"name": "pinned", "id": "p1", "adoption": 3})
    loop.state.baseline = None

    def no_top(ch):
        raise AssertionError("the pinned algorithm must be used")
    mainnet = types.SimpleNamespace(
        top_algorithm=no_top,
        fetch_algorithm_files=lambda ch, name: BASE_FILES,
        fetch_template=lambda ch: "pub fn solve_challenge(")
    loop.measure_baseline(tmp_path / "cache", "cpu4-mem8192", mainnet=mainnet)
    # mutation: measure_baseline not passing algorithm= asks mainnet again
    assert loop.state.baseline.name == "pinned"
    # mutation: not passing hyperparameters= measures the baseline without the map
    assert fb.calls[0].hyperparameters == hp
    loop.run()
    # mutation: _request without hyperparameters= scores candidates without the map
    assert len(fb.calls) >= 2 and all(c.hyperparameters == hp for c in fb.calls)
    # mutation: _context without hyperparameters= leaves the model unaware of the keys
    assert any('track t: {"x":1}' in user for _system, user in fp.calls)
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `T tests/test_loop.py::test_the_spec_hyperparameters_reach_baseline_candidates_and_prompts`
Expected: FAIL, `AssertionError: the pinned algorithm must be used`.

- [ ] **Step 3: Implement**

In `talos/loop.py::measure_baseline`, change the `resolve_baseline` call to:

```python
        rec, template = resolve_baseline(self.spec.challenge, self.spec.training,
                                         self.spec.holdout, self.spec.fuel, _BudgetedBench(self),
                                         cache_dir, hardware_class, rule=self.rule,
                                         log=lambda m: self._event("baseline", message=m),
                                         algorithm=self.spec.baseline_algorithm,
                                         hyperparameters=self.spec.hyperparameters, **kw)
```

In `_request`, add `hyperparameters=self.spec.hyperparameters` to the `EvalRequest(...)`:

```python
        return EvalRequest(challenge=self.spec.challenge, files=files, training=training,
                           holdout=holdout, fuel=self.spec.fuel, baseline_training=base,
                           rule=self.rule, prior_functions=prior,
                           timeouts=self._timeouts(baseline_training),
                           hyperparameters=self.spec.hyperparameters)
```

In `_context`, add as the last `PromptContext` argument:

```python
                             hyperparameters=self.spec.hyperparameters)
```

(replacing the closing `)` of the `guard_tracks=...` argument so the call stays syntactically complete).

- [ ] **Step 4: Run tests**

Run: `T tests/test_loop.py`
Expected: PASS.

- [ ] **Step 5: Mutation checks**

Remove each of the four added keyword arguments in turn (`algorithm=`, `hyperparameters=` in `measure_baseline`, `hyperparameters=` in `_request`, `hyperparameters=` in `_context`). Each makes the new test fail at the assert its mutation comment names. Restore after each.

- [ ] **Step 6: Line-length check, ruff, guarded full suite**

Expected: `325 passed, 2 deselected`.

- [ ] **Step 7: Commit**

```bash
git status --short
git add talos/loop.py tests/test_loop.py
git commit -m "loop: baseline, candidates and prompts all take the spec's hyperparameters

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: The package records the hyperparameters

**Files:**
- Modify: `talos/package.py` (`import json`, new `_hyperparameters_section`, `_readme`, `build_package`)
- Test: `tests/test_package.py`

**Interfaces:**
- Consumes: `JobSpec.hyperparameters`, `JobSpec.hyperparameters_source` (Task 6).
- Produces: `README.md` gains a `## Hyperparameters` section, and `hyperparameters.json` is written when `spec.hyperparameters is not None`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_package.py`:

```python
def test_package_records_the_hyperparameters_and_their_source(tmp_path):
    spec, st, store = make(tmp_path)
    spec = replace(spec, hyperparameters={"t": {"x": 1}},
                   hyperparameters_source={"t": {"benchmark_id": "bm1", "player_id": "0xp",
                                                 "mean_quality": 150.0}})
    pkg = build_package(spec, st, store)
    readme = (pkg / "README.md").read_text()
    # mutation: omitting the section leaves the user submitting without the values they beat with
    assert "## Hyperparameters" in readme
    assert '- t: `{"x": 1}` (mainnet benchmark bm1 by 0xp, mean quality 150)' in readme
    # mutation: not writing the file leaves nothing to paste into a benchmarker config
    import json
    assert json.loads((pkg / "hyperparameters.json").read_text()) == {"t": {"x": 1}}


def test_package_without_hyperparameters_has_no_section_or_file(tmp_path):
    spec, st, store = make(tmp_path)
    pkg = build_package(spec, st, store)
    # mutation: writing the section for None claims values were used when none were
    assert "## Hyperparameters" not in (pkg / "README.md").read_text()
    assert not (pkg / "hyperparameters.json").exists()
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `T tests/test_package.py`
Expected: the first FAILS (no section). The second PASSES already.

- [ ] **Step 3: Implement**

In `talos/package.py`, add `import json` after `import difflib`. Add before `_readme`:

```python
def _hyperparameters_section(spec: JobSpec) -> str:
    if spec.hyperparameters is None:
        return ""
    lines = []
    for track in spec.tracks:
        hp = spec.hyperparameters.get(track)
        src = (spec.hyperparameters_source or {}).get(track)
        value = "none" if hp is None else f"`{json.dumps(hp, sort_keys=True)}`"
        origin = (f" (mainnet benchmark {src['benchmark_id']} by {src['player_id']}, "
                  f"mean quality {src['mean_quality']:g})" if src else "")
        lines.append(f"- {track}: {value}{origin}")
    return ("\n## Hyperparameters\n\nThe baseline and every candidate ran with these per-track "
            "hyperparameters, from the best mainnet benchmark of the baseline algorithm. The "
            "measured improvement holds only with them: benchmark with the same values (they are "
            "also in hyperparameters.json).\n\n" + "\n".join(lines) + "\n")
```

At the end of `_readme`, replace `return head` with:

```python
    return head + _hyperparameters_section(spec)
```

In `build_package`, after `(pkg / "README.md").write_text(_readme(spec, state))`:

```python
    if spec.hyperparameters is not None:
        (pkg / "hyperparameters.json").write_text(json.dumps(spec.hyperparameters, indent=1))
```

- [ ] **Step 4: Run tests**

Run: `T tests/test_package.py`
Expected: PASS.

- [ ] **Step 5: Mutation checks**

| mutation | failing test |
|---|---|
| `return head` (section dropped) | `test_package_records_the_hyperparameters_and_their_source` |
| delete the `hyperparameters.json` write | same |
| drop the `if spec.hyperparameters is not None:` guard around the `hyperparameters.json` write (writes `null`) | `test_package_without_hyperparameters_has_no_section_or_file` |

- [ ] **Step 6: Line-length check, ruff, guarded full suite**

Expected: `327 passed, 2 deselected`.

- [ ] **Step 7: Commit**

```bash
git status --short
git add talos/package.py tests/test_package.py
git commit -m "package: record the hyperparameters and where they came from

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 11: `talos run --hyperparameters`

**Files:**
- Modify: `talos/cli.py` (imports, `FAKE_MAINNET`, new `_mainnet_hyperparameters`, `cmd_run`, `main` parser)
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `mainnet.top_algorithm` 3-tuple (Task 1), `mainnet.top_hyperparameters`, `TrackHyperparameters` (Task 2), `JobSpec` fields (Task 6).
- Produces: `cli._mainnet_hyperparameters(api, challenge: str, info) -> tuple[dict | None, dict | None, dict | None]`.

**Safety:** every run of `tests/test_cli.py`, including mutation checks, goes through `.superpowers/sdd/t.sh`. A cli test that reaches a real bench is the failure the guards exist for.

- [ ] **Step 1: Add the autouse network guard and update the wizard test**

At the top of `tests/test_cli.py`, add `from talos.mainnet import MainnetError, TrackHyperparameters` (replacing the existing `from talos.mainnet import MainnetError`), and after the imports:

```python
@pytest.fixture(autouse=True)
def _no_mainnet_hyperparameters(monkeypatch):
    """`talos run` reads the top algorithm and its hyperparameters before the job exists. No test
    here may reach mainnet for them; a test that cares overrides these."""
    monkeypatch.setattr("talos.mainnet.top_algorithm", lambda ch: ("fake_base", "c003_a000", 1))
    monkeypatch.setattr("talos.mainnet.top_hyperparameters",
                        lambda algorithm_id, tracks, fuel:
                        {t: TrackHyperparameters(None, None, None, None) for t in tracks})
```

In `test_wizard_labels_gpu_challenges_asks_mode_and_survives_a_typo`, append one answer and move the prompt asserts:

```python
    answers = iter(["knapsack", "go", "abc", "3", "4", "5", "agentic", "", ""])
```

```python
    assert prompts[-3] == "Mode (single-shot or agentic)"
    assert prompts[-2] == "Track to optimise (all, or one of: n=1)"
    assert prompts[-1] == "Hyperparameters (mainnet or none)"
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_cli.py`:

```python
def _cli_config(tmp_path):
    save(tmp_path, Config(provider="claude-cli", model="claude-opus-5", mode="single-shot",
                          api_base=None), None)


def test_run_pins_the_top_algorithm_and_its_hyperparameters(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    _cli_config(tmp_path)
    monkeypatch.setattr(cli, "fetch_challenge_info", lambda name: knapsack_info())
    monkeypatch.setattr("talos.mainnet.top_algorithm", lambda ch: ("algo", "c003_a7", 9))
    asked = []

    def top_hp(algorithm_id, tracks, fuel):
        asked.append((algorithm_id, tracks, fuel))
        return {"n=1": TrackHyperparameters({"x": 1}, "bm1", "0xp", 150.0)}
    monkeypatch.setattr("talos.mainnet.top_hyperparameters", top_hp)
    seen = {}
    monkeypatch.setattr(cli, "execute_job",
                        lambda spec, store, cfg, resume: seen.update(spec=spec) or 0)
    rc = cli.main(["run", "--challenge", "knapsack", "--direction", "go",
                   "--budget-iterations", "3", "--yes"])
    assert rc == 0
    # mutation: passing the algorithm name, or another fuel, matches no precommit on mainnet
    assert asked == [("c003_a7", ["n=1"], 7)]
    spec = seen["spec"]
    # mutation: not pinning the algorithm lets the baseline measure a different one later
    assert spec.baseline_algorithm == {"name": "algo", "id": "c003_a7", "adoption": 9}
    assert spec.hyperparameters == {"n=1": {"x": 1}}
    assert spec.hyperparameters_source == {"n=1": {"benchmark_id": "bm1", "player_id": "0xp",
                                                   "mean_quality": 150.0}}
    # mutation: a silent default hides from the user that their job runs with mainnet values
    assert "hyperparameters: 1/1 tracks from mainnet" in capsys.readouterr().out


def test_run_hyperparameters_none_reads_nothing_from_mainnet(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    _cli_config(tmp_path)
    monkeypatch.setattr(cli, "fetch_challenge_info", lambda name: knapsack_info())
    def boom(*a, **k):
        pytest.fail("mainnet must not be read for --hyperparameters none")
    monkeypatch.setattr("talos.mainnet.top_algorithm", boom)
    monkeypatch.setattr("talos.mainnet.top_hyperparameters", boom)
    seen = {}
    monkeypatch.setattr(cli, "execute_job",
                        lambda spec, store, cfg, resume: seen.update(spec=spec) or 0)
    rc = cli.main(["run", "--challenge", "knapsack", "--direction", "go",
                   "--budget-iterations", "3", "--yes", "--hyperparameters", "none"])
    # mutation: ignoring the flag fetches and applies the map anyway
    assert rc == 0
    assert (seen["spec"].baseline_algorithm, seen["spec"].hyperparameters) == (None, None)
    assert "hyperparameters: none" in capsys.readouterr().out


def test_run_with_no_matching_benchmark_pins_the_algorithm_but_no_map(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _cli_config(tmp_path)
    monkeypatch.setattr(cli, "fetch_challenge_info", lambda name: knapsack_info())
    seen = {}
    monkeypatch.setattr(cli, "execute_job",
                        lambda spec, store, cfg, resume: seen.update(spec=spec) or 0)
    assert cli.main(["run", "--challenge", "knapsack", "--direction", "go",
                     "--budget-iterations", "3", "--yes"]) == 0
    # mutation: storing {"n=1": None} makes the package claim hyperparameters were used
    assert seen["spec"].hyperparameters is None and seen["spec"].hyperparameters_source is None
    assert seen["spec"].baseline_algorithm == {"name": "fake_base", "id": "c003_a000",
                                               "adoption": 1}


def test_run_reports_a_hyperparameters_fetch_failure_and_creates_nothing(tmp_path, monkeypatch,
                                                                         capsys):
    monkeypatch.chdir(tmp_path)
    _cli_config(tmp_path)
    monkeypatch.setattr(cli, "fetch_challenge_info", lambda name: knapsack_info())

    def down(*a, **k):
        raise MainnetError("HTTP 503 fetching get-benchmarks")
    monkeypatch.setattr("talos.mainnet.top_hyperparameters", down)
    monkeypatch.setattr(cli, "execute_job",
                        lambda spec, store, cfg, resume: pytest.fail("must not start a job"))
    rc = cli.main(["run", "--challenge", "knapsack", "--direction", "go",
                   "--budget-iterations", "3", "--yes"])
    # mutation: letting MainnetError escape tracebacks; resolving after write_spec leaves a run dir
    assert rc == 1 and "mainnet unreachable" in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()


def test_wizard_hyperparameters_answer_none_and_a_bad_answer(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    _cli_config(tmp_path)
    monkeypatch.setattr(cli, "fetch_challenge_info", lambda name: knapsack_info())
    seen = {}
    monkeypatch.setattr(cli, "execute_job",
                        lambda spec, store, cfg, resume: seen.update(spec=spec) or 0)
    wizard = ["knapsack", "go", "3", "4", "5", "single-shot", ""]
    assert cli.main(["run"], ask=scripted(wizard + ["none"])) == 0
    # mutation: ignoring the wizard answer applies mainnet values the user declined
    assert seen["spec"].hyperparameters is None and seen["spec"].baseline_algorithm is None
    # mutation: accepting any answer starts a job whose choice nobody made
    assert cli.main(["run"], ask=scripted(wizard + ["maybe"])) == 2
    assert "unknown hyperparameters choice 'maybe'" in capsys.readouterr().err


def test_resume_refuses_a_hyperparameters_flag(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    fake_config(tmp_path)
    assert fake_run(monkeypatch) == 0
    run_dir = next((tmp_path / "runs").glob("*/job.json")).parent
    # mutation: letting it through on resume would score a job against a baseline measured
    # with different hyperparameters
    assert cli.main(["run", "--resume", run_dir.name, "--hyperparameters", "none"]) == 2
    assert "fixed its hyperparameters" in capsys.readouterr().err


def test_fake_run_carries_hyperparameters_into_the_package(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fake_config(tmp_path)
    assert fake_run(monkeypatch) == 0
    run_dir = next((tmp_path / "runs").glob("*/job.json")).parent
    job = json.loads((run_dir / "job.json").read_text())
    # mutation: the fake provider path skipping FAKE_MAINNET's map leaves the end-to-end run
    # exercising none of this feature
    assert job["hyperparameters"] == {"n=1": {"fake_boost": 1}}
    assert "## Hyperparameters" in (run_dir / "package" / "README.md").read_text()
```

Two notes:
- The wizard sequence for a `claude-cli` config with a single track is: challenge, direction, `Iteration budget`, `Wall-clock budget in hours`, `Compute budget in USD`, `Mode (single-shot or agentic)`, `Track to optimise ...`, then `Hyperparameters (mainnet or none)`. Confirm it against the existing wizard test before relying on it.
- The fake run's `README.md` gets the section only if the run produced a best candidate. `test_fake_run_with_a_track_wins_and_packages_per_track` shows the fake run wins, so `README.md` goes through `_readme`.

- [ ] **Step 3: Run them and confirm they fail**

Run: `T tests/test_cli.py`
Expected: the new tests FAIL (`unrecognized arguments: --hyperparameters`, missing spec fields). The updated wizard test FAILS on `StopIteration`/prompt order. The other existing tests still pass.

- [ ] **Step 4: Implement**

In `talos/cli.py`:

Imports: add `from talos import mainnet as mainnet_api` and change the mainnet import to `from talos.mainnet import ChallengeInfo, MainnetError, TrackHyperparameters, fetch_challenge_info`.

`FAKE_MAINNET` becomes:

```python
FAKE_MAINNET = types.SimpleNamespace(
    top_algorithm=lambda ch: ("fake_base", "c003_a000", 1),
    top_hyperparameters=lambda algorithm_id, tracks, fuel: {
        t: TrackHyperparameters({"fake_boost": 1}, "fake-benchmark", "0xfake", 100.0)
        for t in tracks},
    fetch_algorithm_files=lambda ch, name: {"mod.rs": "fn solve() { let k = 1; }\n"},
    fetch_template=lambda ch: "pub fn solve_challenge(")
```

Add after `_track_arg`:

```python
def _mainnet_hyperparameters(api, challenge: str, info) -> tuple[dict | None, dict | None,
                                                                 dict | None]:
    """(baseline_algorithm, hyperparameters, hyperparameters_source) for a new job. The algorithm
    is pinned whenever mainnet has one, so the baseline measured later is the code the map
    belongs to. The map is None when no track has a benchmark of it at this fuel."""
    top = api.top_algorithm(challenge)
    if top is None:
        return None, None, None  # resolve_baseline reports the missing algorithm
    name, algorithm_id, adoption = top
    algorithm = {"name": name, "id": algorithm_id, "adoption": adoption}
    per_track = api.top_hyperparameters(algorithm_id, info.tracks, info.max_fuel)
    found = {t: th for t, th in per_track.items() if th.benchmark_id is not None}
    if not found:
        return algorithm, None, None
    return (algorithm, {t: per_track[t].hyperparameters for t in info.tracks},
            {t: th.source() for t, th in found.items()})
```

In `cmd_run`'s resume branch, after the `--track` refusal:

```python
        if args.hyperparameters is not None:
            print(f"job {spec.job_id} fixed its hyperparameters at start; start a new job to "
                  f"change them", file=sys.stderr)
            return 2
```

In `cmd_run`, after the `unknown track` check and before `rand_hash = new_rand_hash()`:

```python
    choice = args.hyperparameters
    if choice is None:
        choice = "mainnet" if args.yes else ask("Hyperparameters (mainnet or none)", "mainnet")
    if choice not in ("mainnet", "none"):
        print(f"unknown hyperparameters choice {choice!r}; use mainnet or none", file=sys.stderr)
        return 2
    algorithm = hyperparameters = hp_source = None
    if choice == "mainnet":
        api = FAKE_MAINNET if cfg.provider == "fake" else mainnet_api
        try:
            algorithm, hyperparameters, hp_source = _mainnet_hyperparameters(api, challenge, info)
        except MainnetError as e:
            print(f"mainnet unreachable: {e}", file=sys.stderr)
            return 1
```

Add to the `JobSpec(...)` call, after `track=track`:

```python
                   track=track, baseline_algorithm=algorithm, hyperparameters=hyperparameters,
                   hyperparameters_source=hp_source)
```

Replace the final `print(f"Job {job_id}: ...")` with:

```python
    if hyperparameters is None:
        hp_line = "hyperparameters: none"
    else:
        used = sum(1 for v in hyperparameters.values() if v is not None)
        hp_line = f"hyperparameters: {used}/{len(info.tracks)} tracks from mainnet"
    print(f"Job {job_id}: {scope}, fuel {info.max_fuel}, budget {budget.to_dict()}; {hp_line}")
```

In `main`, after the `--track` argument:

```python
    r.add_argument("--hyperparameters", choices=["mainnet", "none"],
                   help="per-track hyperparameters from the baseline algorithm's best mainnet "
                        "benchmark (mainnet, the default) or none")
```

- [ ] **Step 5: Run tests**

Run: `T tests/test_cli.py`
Expected: PASS, all of them.

- [ ] **Step 6: Mutation checks** (guarded command only)

| mutation | failing test |
|---|---|
| in `_mainnet_hyperparameters`, pass `name` instead of `algorithm_id` to `top_hyperparameters` | `test_run_pins_the_top_algorithm_and_its_hyperparameters` |
| drop `baseline_algorithm=algorithm` from `JobSpec(...)` | same, and `..._no_matching_benchmark_pins_the_algorithm_but_no_map` |
| `if not found:` block removed | `test_run_with_no_matching_benchmark_pins_the_algorithm_but_no_map` |
| `except MainnetError` removed | `test_run_reports_a_hyperparameters_fetch_failure_and_creates_nothing` |
| `choice = "mainnet"` unconditionally | `test_run_hyperparameters_none_reads_nothing_from_mainnet`, `test_wizard_hyperparameters_answer_none_and_a_bad_answer` |
| remove the resume refusal | `test_resume_refuses_a_hyperparameters_flag` |
| `api = mainnet_api` unconditionally | `test_fake_run_carries_hyperparameters_into_the_package` (the autouse stub returns no benchmark) |

- [ ] **Step 7: Line-length check, ruff, guarded full suite**

Expected: `334 passed, 2 deselected`.

- [ ] **Step 8: Commit**

```bash
git status --short
git add talos/cli.py tests/test_cli.py
git commit -m "cli: talos run --hyperparameters pins mainnet's per-track values at job start

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 12: Docs, and the full gate

**Files:**
- Modify: `AGENTS.md` (invariants 1 and 3)
- Modify: `README.md` (`talos run` flags, the re-run-setup note)

- [ ] **Step 1: AGENTS.md invariant 1**

In invariant 1, after the sentence ending `a cached baseline measured under different hardware or fuel is a different key, never a hit.`, add:

```markdown
   Both sides also run with identical hyperparameters: the per-track map is
   frozen into `job.json` (`talos/state.py::JobSpec`) together with the
   algorithm it belongs to, every request takes it from there
   (`talos/loop.py::Loop._request`, `talos/baseline.py::resolve_baseline`), and
   `talos/inside.py::run_nonce` passes it to `tig-runtime`.
```

- [ ] **Step 2: AGENTS.md invariant 3**

At the end of invariant 3, add:

```markdown
   The baseline cache key also includes the hyperparameter map
   (`talos/baseline.py::effective_hyperparameters`); a key computed without one
   is unchanged from before the map existed.
```

- [ ] **Step 3: README.md**

After the `--track <name>` bullet in the `talos run` flags list, add:

```markdown
- `--hyperparameters mainnet|none` — `mainnet` (the default; the interactive prompt asks)
  runs the baseline and every candidate with the per-track hyperparameters of the baseline
  algorithm's best-quality benchmark on mainnet at the job's fuel, frozen at job start. A
  track with no such benchmark runs without any. `none` runs every nonce without
  hyperparameters, as Talos did before. It cannot be changed on `--resume`. The package
  lists the values and their source benchmark in `README.md` and `hyperparameters.json`.
```

At the end of the paragraph that ends `the C3 job ships its own code and needs nothing.`, add:

```markdown
The Modal score function also takes the track's hyperparameters as an argument, so the same
applies after upgrading past that change: run `talos setup` again on Modal first.
```

- [ ] **Step 4: Docs-reference and contract tests (need agentify on Python 3.11+)**

Run (needs network for the git dependency):

```bash
uv run --python 3.11 --isolated \
  --with "agentify @ git+https://github.com/FibonAdithya/agentic-coding-scaffold@v0.1.2" \
  --with pytest -- python -m pytest tests/test_contract.py tests/test_docs_references.py -q
```

Expected: PASS. If a reference does not resolve, fix the citation (symbol names must exist exactly as cited). If `uv` cannot fetch the dependency, report that as blocked with the error text. Do not skip the step silently.

- [ ] **Step 5: Full gate**

```bash
"$PY" -m ruff check .
.superpowers/sdd/t.sh -m "not live" --ignore tests/test_contract.py \
  --ignore tests/test_docs_references.py > /tmp/talos-hp-suite.log 2>&1; tail -3 /tmp/talos-hp-suite.log
```

Expected: ruff clean; `334 passed, 2 deselected`. Paste both outputs into the task report.

- [ ] **Step 6: rand_hash leak check across new outputs**

```bash
T tests/test_package.py tests/test_prompts.py tests/test_loop.py -k "hash or leak or hyperparameters"
```

Expected: PASS. (The existing hash-grep tests cover the package, prompts and timeline. The new prompt test also asserts no 64-hex string.)

- [ ] **Step 7: Commit**

```bash
git status --short
git add AGENTS.md README.md
git commit -m "docs: --hyperparameters, and the map in the scoring and cache-key invariants

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 8: Report**

Report: the final `git log --oneline 7bf0972..HEAD`, the suite line (MEASURED, with the command), the docs-test result, and every mutation check that did not fail as the table said it would.

---

## Self-review

**Spec coverage**

| spec section | task |
|---|---|
| §3 selection, eligibility, ties, all tracks, `{}` vs null, failure raises | 2 |
| §3 `top_algorithm` returns the id | 1 |
| §4 `JobSpec` fields, old files, redaction | 6 |
| §4 pinned algorithm used by `resolve_baseline` | 7, 9 |
| §5 `run_nonce` flag, runtime only, `{}` passed | 3 |
| §5 `EvalRequest`, Modal signature | 4 |
| §5 C3 payload, tasks, request hash | 5 |
| §5 every request from the spec | 7, 9 |
| §6 cache key normalisation, unchanged old keys, failure hint | 7 |
| §7 CLI flag, wizard, resume refusal, fetch failure, summary line | 11 |
| §8 prompt block, focused view, agentic brief | 8, 9 |
| §9 package section and `hyperparameters.json` | 10 |
| §10 AGENTS.md invariants | 12 |
| README (`talos setup` again, flag) | 12 |

**Test counts:** 298 → +0 (T1) → 305 (T2, +7 cases) → 308 (T3, +3) → 311 (T4, +3) → 315 (T5, +4) → 316 (T6) → 320 (T7, +4) → 324 (T8, +4) → 325 (T9) → 327 (T10, +2) → 334 (T11, +7). ESTIMATE (unverified): each "Expected" count assumes the named tests are the only additions; the implementer reports the measured line at each task.

**Type consistency checked:** `top_algorithm` 3-tuple (T1) in T7 and T11; `TrackHyperparameters(hyperparameters, benchmark_id, player_id, mean_quality)` and `.source()` (T2) in T11; `hyperparameters=` keyword on `run_nonce` (T3) in T4 and T5; `EvalRequest.hyperparameters` (T4) in T5, T7 and T9; `JobSpec.baseline_algorithm` as `{"name","id","adoption"}` (T6) in T7, T9 and T11; `effective_hyperparameters` (T7) cited in T12; `PromptContext.hyperparameters` (T8) in T9.
