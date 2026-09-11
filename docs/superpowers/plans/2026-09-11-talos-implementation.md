# Talos Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `talos` CLI that, after one setup wizard, runs an LLM research loop against the top mainnet TIG algorithm on Modal's faithful benchmark until it beats it on training and held-out nonces or exhausts a budget, and hands back a submit-ready package.

**Architecture:** Pure-Python package `talos/` with one module per responsibility (mainnet, nonces, scoring, edits, budget, state, bench client, providers, prompts, loop, package, cli, agentic). A separate `modal_app/` holds the Modal app whose container-side logic lives in a pure module (`modal_app/inside.py`) tested with a fake subprocess runner. The loop is driven end to end in tests by `FakeProvider` and `FakeBench`; nothing in the test suite touches the network, Modal, or a real CLI.

**Tech Stack:** Python 3.10+, `modal` 1.5.x (pinned `>=1.5,<2`), `rich` for terminal status, stdlib `urllib` for HTTP, `pytest` + `ruff` for checks. GPLv3.

**Spec:** `docs/superpowers/specs/2026-09-11-talos-design.md`

## Global Constraints

- Python `>=3.10`. No dependency beyond `modal` and `rich` at runtime; `pytest` and `ruff` for dev.
- LLM-authored code never executes on the user's machine. Only `modal_app/` runs it, on Modal.
- The job `rand_hash` never appears in a prompt, a timeline event, the agent worktree, or the package. Only `job.json` holds it.
- Dev image: `ghcr.io/tig-foundation/tig-monorepo/<challenge>/dev:0.0.7`. Monorepo pinned at commit `84a5787f5b14a630bdf40f52bccf37887d3d8464`.
- Defaults (spec §13): training 32 nonces per track, held-out 32 per track starting at nonce 1,000,000; compile fix rounds 3; stagnation recall/distill/reset = 2/3/5; agentic timeout 1800 s; Modal retry window 15 min; beat rule margin 0.005, track tolerance 0.0, error ceiling 0.05; CPU 4 vCPU / 8 GiB, GPU one L40S; per-nonce timeout 600 s; budget has no default.
- Secrets: `.talos/secrets.json` mode 0600, holds only `api_key`. Modal credentials go through `modal token set`.
- `make check` = `ruff check . && pytest -q -m "not live"`. Every task ends with it green. The lint set is pinned in `pyproject.toml` (`[tool.ruff.lint] select = ["E4", "E7", "E9", "F"]`) because ruff 0.16 defaults to a much wider set (isort, BLE, B008) that this code does not target.
- Environment: `python3 -m venv` is broken on the dev machine (no `ensurepip`, no system `pip`); `uv` is on PATH. Create the venv with `uv venv --python 3.10 .venv` and install with `uv pip install --python .venv/bin/python -e '.[dev]'`. Never use `.venv/bin/pip`.
- CLI subprocesses (`claude`, `codex`, `tig-runtime`, `tig-verifier`, `build_algorithm`) are invoked by bare name; `subprocess.run` resolves PATH and raises `FileNotFoundError` when absent. Never `shutil.which()` them into argv — tests assert on argv[0].
- Commit after every task with a message in the form `<area>: <what>`; stage explicit paths only.
- Tests state the mutation they catch in a comment on the test.

---

## File structure

```
pyproject.toml            package metadata, console script `talos`, pytest markers
Makefile                  `make check`
talos/__init__.py
talos/types.py            NonceSet, NonceResult, Usage, Completion, CompileResult
talos/challenges.py       static per-challenge table (id, gpu, beat rule, hardware)
talos/mainnet.py          get-block, get-challenges, top algorithm, fetch files, template
talos/nonces.py           draw disjoint training / held-out NonceSets from a job hash
talos/scoring.py          track and bundle deltas, beats()
talos/search_replace.py   SEARCH/REPLACE parsing and application (lifted from Prometheus)
talos/edits.py            apply an LLM edit response to files with path scoping
talos/budget.py           Budget, Spend, exhausted()
talos/state.py            JobSpec, JobState, atomic save/load, timeline events
talos/bench.py            Bench protocol, ModalBench, FakeBench
talos/providers/__init__.py  Provider protocol, factory
talos/providers/pricing.py   per-model $/Mtok table, estimate_cost
talos/providers/openai_compat.py, anthropic.py, google.py, claude_cli.py, codex_cli.py, fake.py
talos/prompts.py          hypothesis / edit / fix / distill prompts and parsers
talos/baseline.py         resolve + compile + score the mainnet top algorithm, cache
talos/loop.py             the research loop, stagnation, confirmation, stop
talos/package.py          hand-back package
talos/agentic.py          worktree, sandbox settings, claude/codex iterate
talos/cli.py              `talos setup | run | compile`
talos/data/evidence_template.md   copied from the monorepo
talos/data/rust_rules.md          Rust rules block (lifted from Prometheus prompts.py)
modal_app/talos_bench.py  Modal app: per-challenge compile / score_nonce functions
modal_app/inside.py       container-side pure logic (stage, build, run nonce)
tests/...                 one test file per module, plus tests/test_live.py (marker live)
```

---

### Task 1: Project scaffold and core types

**Files:**
- Create: `pyproject.toml`, `Makefile`, `talos/__init__.py`, `talos/types.py`, `tests/__init__.py`, `tests/test_types.py`

**Interfaces:**
- Produces: `talos.types.NonceSet(track: str, rand_hash: str, start: int, count: int)` frozen dataclass with `nonces() -> range`; `NonceResult(track, nonce, ok, quality, runtime_ms, error)`; `Usage(input_tokens, output_tokens, cost_usd)`; `Completion(text, usage)`; `CompileResult(ok, artifact_id, output)`. Error strings are exactly one of `None, "no_solution", "invalid", "out_of_fuel", "panic", "timeout", "compile"`.

- [ ] **Step 1: Write pyproject, Makefile, package init**

`pyproject.toml`:
```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "talos-tig"
version = "0.1.0"
description = "Single-user autoresearch loop for TIG challenges"
requires-python = ">=3.10"
license = {text = "GPL-3.0-or-later"}
dependencies = ["modal>=1.5,<2", "rich>=13"]

[project.optional-dependencies]
dev = ["pytest>=8", "ruff>=0.5"]

[project.scripts]
talos = "talos.cli:main"

[tool.setuptools]
packages = ["talos", "talos.providers", "modal_app"]

[tool.setuptools.package-data]
talos = ["data/*.md"]

[tool.pytest.ini_options]
markers = ["live: hits Modal or the network; run manually"]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E4", "E7", "E9", "F"]
```

`Makefile`:
```make
PYTHON ?= python3
.PHONY: check
check:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m pytest -q -m "not live"
```

`talos/__init__.py`:
```python
"""Talos: single-user autoresearch for TIG."""
__version__ = "0.1.0"
```

- [ ] **Step 2: Write the failing test**

`tests/test_types.py`:
```python
from talos.types import NonceSet, NonceResult, ERROR_KINDS


def test_nonce_set_range():
    # mutation: off-by-one in nonces() (count-1 or start+1) fails this
    ns = NonceSet(track="n_nodes=600", rand_hash="ab", start=5, count=3)
    assert list(ns.nonces()) == [5, 6, 7]


def test_nonce_result_rejects_unknown_error():
    # mutation: dropping the validation lets a typo like "panik" through
    import pytest
    with pytest.raises(ValueError):
        NonceResult(track="t", nonce=0, ok=False, quality=None, runtime_ms=0, error="panik")
    assert "out_of_fuel" in ERROR_KINDS
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv venv --python 3.10 .venv && uv pip install --python .venv/bin/python -e '.[dev]' && .venv/bin/pytest tests/test_types.py -v`
Expected: FAIL with `ModuleNotFoundError: talos.types`

- [ ] **Step 4: Write types.py**

```python
"""Core value types shared by every Talos module. No I/O here."""
from __future__ import annotations

from dataclasses import dataclass, asdict

ERROR_KINDS = (None, "no_solution", "invalid", "out_of_fuel", "panic", "timeout", "compile")


@dataclass(frozen=True)
class NonceSet:
    track: str
    rand_hash: str
    start: int
    count: int

    def nonces(self) -> range:
        return range(self.start, self.start + self.count)


@dataclass
class NonceResult:
    track: str
    nonce: int
    ok: bool
    quality: int | None
    runtime_ms: int
    error: str | None = None

    def __post_init__(self) -> None:
        if self.error not in ERROR_KINDS:
            raise ValueError(f"unknown error kind {self.error!r}")
        if self.ok and self.quality is None:
            raise ValueError("ok result must carry a quality")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "NonceResult":
        return cls(**d)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None


@dataclass
class Completion:
    text: str
    usage: Usage


@dataclass
class CompileResult:
    ok: bool
    artifact_id: str | None
    output: str
```

- [ ] **Step 5: Run tests and ruff**

Run: `make check PYTHON=.venv/bin/python`
Expected: 2 passed, ruff clean.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml Makefile talos/__init__.py talos/types.py tests/__init__.py tests/test_types.py
git commit -m "scaffold: package, make check, core types"
```

---

### Task 2: Challenge table and mainnet client

**Files:**
- Create: `talos/challenges.py`, `talos/mainnet.py`, `tests/test_mainnet.py`

**Interfaces:**
- Produces: `talos.challenges.CHALLENGES: dict[str, ChallengeSpec]` keyed by name, `ChallengeSpec(name, id, is_gpu, beat: BeatRule, cpu, memory_mib, gpu)`; `BeatRule(margin, track_tolerance, error_ceiling)`; `MONOREPO_REF`, `DEV_IMAGE_TAG`, `dev_image(name) -> str`.
- Produces: `talos.mainnet.ChallengeInfo(id, name, is_gpu, tracks: list[str], max_fuel: int)`; `fetch_challenge_info(name, get_json=_get_json) -> ChallengeInfo`; `top_algorithm(name, get_json=_get_json) -> tuple[str, int] | None`; `fetch_algorithm_files(name, algorithm, get_text=_get_text, get_json=_get_json) -> dict[str, str]`; `fetch_template(name, get_text=_get_text) -> str`; `MainnetError`.

- [ ] **Step 1: Write challenges.py**

```python
"""Static per-challenge facts. Everything a job needs that is not read from mainnet."""
from __future__ import annotations

from dataclasses import dataclass

MONOREPO_REF = "84a5787f5b14a630bdf40f52bccf37887d3d8464"
DEV_IMAGE_TAG = "0.0.7"


def dev_image(name: str) -> str:
    return f"ghcr.io/tig-foundation/tig-monorepo/{name}/dev:{DEV_IMAGE_TAG}"


@dataclass(frozen=True)
class BeatRule:
    margin: float = 0.005
    track_tolerance: float = 0.0
    error_ceiling: float = 0.05


@dataclass(frozen=True)
class ChallengeSpec:
    name: str
    id: str
    is_gpu: bool
    beat: BeatRule = BeatRule()
    cpu: int = 4
    memory_mib: int = 8192
    gpu: str | None = None


def _cpu(name: str, cid: str) -> ChallengeSpec:
    return ChallengeSpec(name=name, id=cid, is_gpu=False)


def _gpu(name: str, cid: str) -> ChallengeSpec:
    return ChallengeSpec(name=name, id=cid, is_gpu=True, gpu="L40S")


CHALLENGES: dict[str, ChallengeSpec] = {
    s.name: s
    for s in (
        _cpu("satisfiability", "c001"),
        _cpu("vehicle_routing", "c002"),
        _cpu("knapsack", "c003"),
        _gpu("vector_search", "c004"),
        _gpu("hypergraph", "c005"),
        _gpu("neuralnet_optimizer", "c006"),
        _cpu("job_scheduling", "c007"),
        _cpu("energy_arbitrage", "c008"),
    )
}
```

- [ ] **Step 2: Write the failing tests**

`tests/test_mainnet.py`:
```python
import pytest

from talos import mainnet

BLOCK = {"block": {"id": "b1"}}
CHALLENGES = {"challenges": [
    {"id": "c002", "config": {"name": "vehicle_routing", "type": "cpu",
                              "active_tracks": {"n_nodes=600": {}, "n_nodes=700": {}},
                              "max_fuel_budget": 5000000000000}},
    {"id": "c005", "config": {"name": "hypergraph", "type": "gpu",
                              "active_tracks": {"k=8": {}}, "max_fuel_budget": 7}},
]}
ALGOS = {"codes": [
    {"id": "a1", "details": {"challenge_id": "c002", "name": "hgs_v1"}, "block_data": {"adoption": "30"}},
    {"id": "a2", "details": {"challenge_id": "c002", "name": "fast_lane_v6"}, "block_data": {"adoption": "50"}},
    {"id": "a3", "details": {"challenge_id": "c002", "name": "broken"}, "block_data": {"adoption": "90"}},
    {"id": "a4", "details": {"challenge_id": "c005", "name": "other"}, "block_data": {"adoption": "99"}},
], "binarys": [
    {"algorithm_id": "a1", "details": {"compile_success": True}},
    {"algorithm_id": "a2", "details": {"compile_success": True}},
    {"algorithm_id": "a3", "details": {"compile_success": False}},
    {"algorithm_id": "a4", "details": {"compile_success": True}},
]}


def fake_get_json(url: str):
    if url.endswith("/get-block"):
        return BLOCK
    if "/get-challenges?" in url:
        return CHALLENGES
    if "/get-algorithms?" in url:
        return ALGOS
    if "api.github.com" in url and "/contents/" in url:
        return [{"type": "file", "path": "tig-algorithms/src/vehicle_routing/fast_lane_v6/mod.rs",
                 "name": "mod.rs"},
                {"type": "file", "path": "tig-algorithms/src/vehicle_routing/fast_lane_v6/ls.rs",
                 "name": "ls.rs"}]
    raise AssertionError(url)


def fake_get_text(url: str):
    return f"// contents of {url.rsplit('/', 1)[-1]}"


def test_challenge_info_reads_tracks_and_fuel_sorted():
    # mutation: forgetting sorted() makes track order depend on dict order
    info = mainnet.fetch_challenge_info("vehicle_routing", get_json=fake_get_json)
    assert info.id == "c002" and not info.is_gpu
    assert info.tracks == ["n_nodes=600", "n_nodes=700"]
    assert info.max_fuel == 5000000000000


def test_challenge_info_unknown_name_raises():
    with pytest.raises(mainnet.MainnetError):
        mainnet.fetch_challenge_info("nope", get_json=fake_get_json)


def test_top_algorithm_skips_uncompiled_and_other_challenges():
    # mutation: dropping the compile_success filter picks "broken" (adoption 90)
    # mutation: dropping the challenge filter picks "other" (adoption 99)
    assert mainnet.top_algorithm("vehicle_routing", get_json=fake_get_json) == ("fast_lane_v6", 50)


def test_top_algorithm_none_when_no_adoption():
    def gj(url):
        if "/get-algorithms?" in url:
            return {"codes": [], "binarys": []}
        return fake_get_json(url)
    assert mainnet.top_algorithm("vehicle_routing", get_json=gj) is None


def test_fetch_algorithm_files_relative_paths():
    # mutation: keeping the full repo path as the key breaks staging inside Modal
    files = mainnet.fetch_algorithm_files("vehicle_routing", "fast_lane_v6",
                                          get_text=fake_get_text, get_json=fake_get_json)
    assert set(files) == {"mod.rs", "ls.rs"}
    assert files["mod.rs"].startswith("// contents of mod.rs")


def test_fetch_template_url():
    seen = []
    def gt(url):
        seen.append(url)
        return "pub fn solve_challenge"
    assert "solve_challenge" in mainnet.fetch_template("knapsack", get_text=gt)
    assert seen == ["https://raw.githubusercontent.com/tig-foundation/tig-monorepo/"
                    "84a5787f5b14a630bdf40f52bccf37887d3d8464/tig-algorithms/src/knapsack/template.rs"]
```

- [ ] **Step 3: Run to verify failure**

Run: `.venv/bin/pytest tests/test_mainnet.py -v`
Expected: FAIL, `ModuleNotFoundError: talos.mainnet`

- [ ] **Step 4: Write mainnet.py**

Port of Prometheus `server/mainnet_seed.py` (GPLv3), reduced to what Talos needs. Single-file algorithms live at `tig-algorithms/src/<challenge>/<name>.rs`; multi-file ones at `tig-algorithms/src/<challenge>/<name>/…`. Both are on the branch `<challenge>/<name>`.

```python
"""Read-only mainnet and monorepo access. Every function takes its HTTP getter as a
parameter so tests never touch the network."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

from talos.challenges import MONOREPO_REF

MAINNET_API = "https://mainnet-api.tig.foundation"
GH_REPO = "tig-foundation/tig-monorepo"
GH_API = f"https://api.github.com/repos/{GH_REPO}"
GH_RAW = f"https://raw.githubusercontent.com/{GH_REPO}"
HTTP_TIMEOUT = 15
UA = "talos-tig"


class MainnetError(RuntimeError):
    pass


def _get(url: str, accept: str) -> bytes:
    req = urllib.request.Request(url, headers={"Accept": accept, "User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        raise MainnetError(f"HTTP {e.code} fetching {url}") from None
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise MainnetError(f"network error fetching {url}: {e}") from None


def _get_json(url: str):
    return json.loads(_get(url, "application/json"))


def _get_text(url: str) -> str:
    return _get(url, "text/plain").decode("utf-8")


@dataclass(frozen=True)
class ChallengeInfo:
    id: str
    name: str
    is_gpu: bool
    tracks: list[str]
    max_fuel: int


def _block_id(get_json) -> str:
    return get_json(f"{MAINNET_API}/get-block")["block"]["id"]


def fetch_challenge_info(name: str, get_json=_get_json) -> ChallengeInfo:
    block_id = _block_id(get_json)
    resp = get_json(f"{MAINNET_API}/get-challenges?block_id={block_id}")
    for c in resp["challenges"]:
        cfg = c.get("config") or {}
        if cfg.get("name") != name:
            continue
        tracks = sorted((cfg.get("active_tracks") or {}).keys())
        if not tracks:
            raise MainnetError(f"challenge {name} has no active tracks on mainnet")
        return ChallengeInfo(id=c["id"], name=name, is_gpu=cfg.get("type") == "gpu",
                             tracks=tracks, max_fuel=int(cfg["max_fuel_budget"]))
    raise MainnetError(f"challenge {name!r} not found on mainnet")


def top_algorithm(name: str, get_json=_get_json) -> tuple[str, int] | None:
    """(algorithm_name, adoption) of the highest-adoption compiled algorithm, or None."""
    block_id = _block_id(get_json)
    challenges = get_json(f"{MAINNET_API}/get-challenges?block_id={block_id}")
    algos = get_json(f"{MAINNET_API}/get-algorithms?block_id={block_id}")
    cid = next((c["id"] for c in challenges["challenges"]
                if (c.get("config") or {}).get("name") == name), None)
    if cid is None:
        raise MainnetError(f"challenge {name!r} not found on mainnet")
    compiled = {b["algorithm_id"]: bool((b.get("details") or {}).get("compile_success"))
                for b in algos.get("binarys", [])}
    best: tuple[str, int] | None = None
    for algo in algos.get("codes", []):
        details = algo.get("details") or {}
        if details.get("challenge_id") != cid or not compiled.get(algo["id"]):
            continue
        try:
            adoption = int((algo.get("block_data") or {}).get("adoption") or 0)
        except (TypeError, ValueError):
            adoption = 0
        algo_name = details.get("name")
        if adoption > 0 and algo_name and (best is None or adoption > best[1]):
            best = (algo_name, adoption)
    return best


def _walk(path: str, ref: str, get_json) -> list[str]:
    entries = get_json(f"{GH_API}/contents/{path}?ref={ref}")
    if isinstance(entries, dict):  # a single file, not a directory
        return [entries["path"]]
    out: list[str] = []
    for e in entries:
        if e["type"] == "dir":
            out.extend(_walk(e["path"], ref, get_json))
        elif e["type"] == "file":
            out.append(e["path"])
    return out


def fetch_algorithm_files(name: str, algorithm: str, get_text=_get_text,
                          get_json=_get_json) -> dict[str, str]:
    """{relative_path: contents}. A single-file algorithm comes back as {"mod.rs": ...}."""
    ref = f"{name}/{algorithm}"
    base_dir = f"tig-algorithms/src/{name}/{algorithm}"
    try:
        paths = _walk(base_dir, ref, get_json)
    except MainnetError:
        single = f"tig-algorithms/src/{name}/{algorithm}.rs"
        return {"mod.rs": get_text(f"{GH_RAW}/{ref}/{single}")}
    files: dict[str, str] = {}
    for p in paths:
        rel = p[len(base_dir) + 1:] if p.startswith(base_dir + "/") else p.rsplit("/", 1)[-1]
        files[rel] = get_text(f"{GH_RAW}/{ref}/{p}")
    if not files:
        raise MainnetError(f"no files found for {name}/{algorithm}")
    return files


def fetch_template(name: str, get_text=_get_text) -> str:
    return get_text(f"{GH_RAW}/{MONOREPO_REF}/tig-algorithms/src/{name}/template.rs")
```

- [ ] **Step 5: Run tests, ruff**

Run: `make check PYTHON=.venv/bin/python`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add talos/challenges.py talos/mainnet.py tests/test_mainnet.py
git commit -m "mainnet: challenge table, tracks and fuel, top algorithm, file fetch"
```

---

### Task 3: Nonce sets and scoring

**Files:**
- Create: `talos/nonces.py`, `talos/scoring.py`, `tests/test_nonces.py`, `tests/test_scoring.py`

**Interfaces:**
- Produces: `talos.nonces.new_rand_hash() -> str` (64 hex chars), `draw_nonce_sets(tracks, rand_hash, training_count=32, holdout_count=32) -> tuple[list[NonceSet], list[NonceSet]]` with `HOLDOUT_START = 1_000_000`.
- Produces: `talos.scoring.TrackDelta(track, base_mean, cand_mean, rel_delta, cand_errors, n)`, `BundleDelta(tracks: list[TrackDelta], mean_rel_delta, worst_rel_delta, error_rate)`, `bundle_delta(baseline, candidate) -> BundleDelta`, `beats(baseline, candidate, rule) -> bool`, `ScoringError`.

- [ ] **Step 1: Failing tests for nonces**

`tests/test_nonces.py`:
```python
from talos.nonces import draw_nonce_sets, new_rand_hash, HOLDOUT_START


def test_rand_hash_is_64_hex():
    h = new_rand_hash()
    assert len(h) == 64 and int(h, 16) >= 0
    assert new_rand_hash() != h


def test_sets_are_per_track_and_disjoint():
    # mutation: holdout start == training start makes the sets overlap
    tr, ho = draw_nonce_sets(["a=1", "b=2"], "ff" * 32, training_count=4, holdout_count=3)
    assert [s.track for s in tr] == ["a=1", "b=2"] and [s.track for s in ho] == ["a=1", "b=2"]
    assert list(tr[0].nonces()) == [0, 1, 2, 3]
    assert list(ho[0].nonces()) == [HOLDOUT_START, HOLDOUT_START + 1, HOLDOUT_START + 2]
    assert set(tr[0].nonces()).isdisjoint(ho[0].nonces())
    assert all(s.rand_hash == "ff" * 32 for s in tr + ho)
```

- [ ] **Step 2: Failing tests for scoring**

`tests/test_scoring.py`:
```python
import pytest

from talos.challenges import BeatRule
from talos.scoring import ScoringError, beats, bundle_delta
from talos.types import NonceResult


def R(track, nonce, q, err=None):
    return NonceResult(track=track, nonce=nonce, ok=err is None, quality=q,
                       runtime_ms=1, error=err)


BASE = [R("t1", 0, 100), R("t1", 1, 100), R("t2", 0, 200), R("t2", 1, 200)]


def test_track_and_bundle_delta():
    cand = [R("t1", 0, 110), R("t1", 1, 110), R("t2", 0, 200), R("t2", 1, 200)]
    d = bundle_delta(BASE, cand)
    by = {t.track: t for t in d.tracks}
    assert by["t1"].rel_delta == pytest.approx(0.10)
    assert by["t2"].rel_delta == pytest.approx(0.0)
    assert d.mean_rel_delta == pytest.approx(0.05)
    assert d.worst_rel_delta == pytest.approx(0.0)
    assert d.error_rate == 0.0


def test_error_scored_as_worst_observed_floored_at_zero():
    # mutation: scoring an error as None/skip inflates the candidate mean
    cand = [R("t1", 0, 110), R("t1", 1, None, "panic"), R("t2", 0, 200), R("t2", 1, 200)]
    d = bundle_delta(BASE, cand)
    by = {t.track: t for t in d.tracks}
    # worst observed on t1 across both = 100 -> error counts as 100
    assert by["t1"].cand_mean == pytest.approx(105)
    assert by["t1"].cand_errors == 1
    assert d.error_rate == pytest.approx(0.25)


def test_mismatched_nonces_raise():
    # mutation: silently zipping unequal lists compares different instances
    with pytest.raises(ScoringError):
        bundle_delta(BASE, BASE[:3])
    with pytest.raises(ScoringError):
        bundle_delta(BASE, [R("t1", 0, 1), R("t1", 5, 1), R("t2", 0, 1), R("t2", 1, 1)])


def test_beats_margin_boundary():
    # mutation: `>` instead of `>=` on the margin fails the equal case
    rule = BeatRule(margin=0.05, track_tolerance=0.0, error_ceiling=0.05)
    cand = [R("t1", 0, 110), R("t1", 1, 110), R("t2", 0, 200), R("t2", 1, 200)]
    assert beats(BASE, cand, rule)  # mean delta exactly 0.05
    rule2 = BeatRule(margin=0.0501, track_tolerance=0.0, error_ceiling=0.05)
    assert not beats(BASE, cand, rule2)


def test_beats_rejects_track_regression_and_errors():
    # mutation: dropping the worst-track check accepts a regression on t2
    rule = BeatRule(margin=0.01, track_tolerance=0.0, error_ceiling=0.05)
    regress = [R("t1", 0, 150), R("t1", 1, 150), R("t2", 0, 199), R("t2", 1, 199)]
    assert not beats(BASE, regress, rule)
    # mutation: dropping the error ceiling accepts a 25% error rate
    errs = [R("t1", 0, 150), R("t1", 1, 150), R("t2", 0, 250), R("t2", 1, None, "timeout")]
    assert not beats(BASE, errs, rule)


def test_zero_baseline_mean_raises():
    zero = [R("t1", 0, 0), R("t1", 1, 0)]
    with pytest.raises(ScoringError):
        bundle_delta(zero, [R("t1", 0, 5), R("t1", 1, 5)])
```

- [ ] **Step 3: Run to verify failures**

Run: `.venv/bin/pytest tests/test_nonces.py tests/test_scoring.py -v`
Expected: FAIL with module not found for both.

- [ ] **Step 4: Write nonces.py**

```python
"""Draw the per-track training and held-out nonce sets for one job."""
from __future__ import annotations

import secrets

from talos.types import NonceSet

HOLDOUT_START = 1_000_000


def new_rand_hash() -> str:
    return secrets.token_hex(32)


def draw_nonce_sets(tracks: list[str], rand_hash: str, training_count: int = 32,
                    holdout_count: int = 32) -> tuple[list[NonceSet], list[NonceSet]]:
    training = [NonceSet(track=t, rand_hash=rand_hash, start=0, count=training_count)
                for t in tracks]
    holdout = [NonceSet(track=t, rand_hash=rand_hash, start=HOLDOUT_START, count=holdout_count)
               for t in tracks]
    return training, holdout
```

- [ ] **Step 5: Write scoring.py**

```python
"""Compare a candidate's per-nonce results against the baseline's on identical nonces."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from statistics import mean

from talos.challenges import BeatRule
from talos.types import NonceResult


class ScoringError(ValueError):
    pass


@dataclass
class TrackDelta:
    track: str
    base_mean: float
    cand_mean: float
    rel_delta: float
    cand_errors: int
    n: int


@dataclass
class BundleDelta:
    tracks: list[TrackDelta]
    mean_rel_delta: float
    worst_rel_delta: float
    error_rate: float

    def to_dict(self) -> dict:
        return asdict(self)


def _by_track(results: list[NonceResult]) -> dict[str, dict[int, NonceResult]]:
    out: dict[str, dict[int, NonceResult]] = {}
    for r in results:
        out.setdefault(r.track, {})[r.nonce] = r
    return out


def _quality_or_worst(r: NonceResult, worst: int) -> int:
    return r.quality if r.ok and r.quality is not None else worst


def bundle_delta(baseline: list[NonceResult], candidate: list[NonceResult]) -> BundleDelta:
    b, c = _by_track(baseline), _by_track(candidate)
    if b.keys() != c.keys():
        raise ScoringError(f"track mismatch: {sorted(b)} vs {sorted(c)}")
    tracks: list[TrackDelta] = []
    total = errors = 0
    for track in sorted(b):
        if b[track].keys() != c[track].keys():
            raise ScoringError(f"nonce mismatch on {track}")
        observed = [r.quality for r in list(b[track].values()) + list(c[track].values())
                    if r.ok and r.quality is not None]
        worst = max(0, min(observed)) if observed else 0
        bq = [_quality_or_worst(r, worst) for r in b[track].values()]
        cq = [_quality_or_worst(r, worst) for r in c[track].values()]
        bm, cm = mean(bq), mean(cq)
        if bm <= 0:
            raise ScoringError(f"baseline mean quality on {track} is {bm}; cannot normalise")
        n_err = sum(1 for r in c[track].values() if not r.ok)
        tracks.append(TrackDelta(track=track, base_mean=bm, cand_mean=cm,
                                 rel_delta=(cm - bm) / bm, cand_errors=n_err, n=len(cq)))
        total += len(cq)
        errors += n_err
    return BundleDelta(tracks=tracks,
                       mean_rel_delta=mean(t.rel_delta for t in tracks),
                       worst_rel_delta=min(t.rel_delta for t in tracks),
                       error_rate=errors / total)


def beats(baseline: list[NonceResult], candidate: list[NonceResult], rule: BeatRule) -> bool:
    d = bundle_delta(baseline, candidate)
    return (d.mean_rel_delta >= rule.margin
            and d.worst_rel_delta >= -rule.track_tolerance
            and d.error_rate <= rule.error_ceiling)
```

- [ ] **Step 6: Run tests, ruff**

Run: `make check PYTHON=.venv/bin/python`
Expected: pass. If `test_beats_margin_boundary` is flaky on float equality, compare with `>= rule.margin - 1e-12` and note it in the code.

- [ ] **Step 7: Commit**

```bash
git add talos/nonces.py talos/scoring.py tests/test_nonces.py tests/test_scoring.py
git commit -m "scoring: nonce sets, track and bundle deltas, beat rule"
```

---

### Task 4: Search/replace edits with path scoping

**Files:**
- Create: `talos/search_replace.py` (copied from Prometheus), `talos/edits.py`, `tests/test_search_replace.py`, `tests/test_edits.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `talos.search_replace.parse_blocks(text) -> list[Block]`, `apply_blocks(files, blocks) -> (new_files, misses)`, `format_misses(misses) -> str`, `Block(file, search, replace)`, `Miss(block, reason)` (reason in `not_found | ambiguous | no_file | empty_search`).
- Produces: `talos.edits.EditOutcome(files: dict[str, str], applied: int, misses: list[Miss], rejected: list[str])`, `apply_edit_response(files, response_text) -> EditOutcome`, `EditError`.

- [ ] **Step 1: Copy the engine and its tests from Prometheus**

The engine is GPLv3 and pure. Clone and copy:

```bash
gh repo clone tig-foundation/prometheus-swarm /tmp/prometheus-swarm -- --depth 1
cp /tmp/prometheus-swarm/scripts/search_replace.py talos/search_replace.py
cp /tmp/prometheus-swarm/scripts/test_search_replace.py tests/test_search_replace.py
```

Then edit `tests/test_search_replace.py`: change `from search_replace import ...` to `from talos.search_replace import ...`, delete the `print("PASS ...")` lines and the `if __name__ == "__main__":` runner at the bottom so pytest collects it. Also delete the `if __name__ == "__main__":` self-test block at the bottom of `talos/search_replace.py`. Add this header line under the module docstring of `talos/search_replace.py`:

```python
# Lifted from tig-foundation/prometheus-swarm scripts/search_replace.py (GPLv3).
```

- [ ] **Step 2: Run the copied tests**

Run: `.venv/bin/pytest tests/test_search_replace.py -v`
Expected: all pass. If any fail, the import rewrite is wrong; fix the import, do not touch the engine.

- [ ] **Step 3: Failing tests for edits.py**

`tests/test_edits.py`:
```python
import pytest

from talos.edits import EditError, apply_edit_response

FILES = {"mod.rs": "fn a() {}\nfn b() {}\n", "ls.rs": "fn c() {}\n"}


def blk(path, s, r):
    return f"<<<<<<< SEARCH {path}\n{s}\n=======\n{r}\n>>>>>>> REPLACE\n"


def test_applies_blocks_to_known_files():
    out = apply_edit_response(FILES, blk("mod.rs", "fn a() {}", "fn a() { 1 }"))
    assert out.applied == 1 and not out.misses and not out.rejected
    assert out.files["mod.rs"].startswith("fn a() { 1 }")
    assert out.files["ls.rs"] == FILES["ls.rs"]


def test_rejects_edit_outside_algorithm_files():
    # mutation: dropping the path check lets an edit to Cargo.toml or ../x through
    out = apply_edit_response(FILES, blk("Cargo.toml", "x", "y") + blk("../mod.rs", "x", "y"))
    assert out.applied == 0
    assert sorted(out.rejected) == ["../mod.rs", "Cargo.toml"]


def test_no_blocks_is_an_error():
    # mutation: returning an empty outcome silently wastes an iteration
    with pytest.raises(EditError):
        apply_edit_response(FILES, "I would change the loop bound.")


def test_miss_reported_not_guessed():
    out = apply_edit_response(FILES, blk("mod.rs", "fn zzz() {}", "fn zzz() { 1 }"))
    assert out.applied == 0 and len(out.misses) == 1 and out.misses[0].reason == "not_found"
```

- [ ] **Step 4: Run to verify failure**

Run: `.venv/bin/pytest tests/test_edits.py -v`
Expected: FAIL, module not found.

- [ ] **Step 5: Write edits.py**

```python
"""Apply an LLM edit response to the algorithm files. Any block naming a path that is not
one of the algorithm's own files is rejected outright, never resolved by basename."""
from __future__ import annotations

from dataclasses import dataclass, field

from talos.search_replace import Block, Miss, apply_blocks, parse_blocks


class EditError(ValueError):
    pass


@dataclass
class EditOutcome:
    files: dict[str, str]
    applied: int
    misses: list[Miss] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)


def _in_scope(block: Block, files: dict[str, str]) -> bool:
    if block.file is None:
        return len(files) == 1
    return block.file in files


def apply_edit_response(files: dict[str, str], response_text: str) -> EditOutcome:
    blocks = parse_blocks(response_text)
    if not blocks:
        raise EditError("response contained no SEARCH/REPLACE blocks")
    rejected = [b.file for b in blocks if not _in_scope(b, files)]
    kept = [b for b in blocks if _in_scope(b, files)]
    new_files, misses = apply_blocks(files, kept)
    applied = len(kept) - len(misses)
    return EditOutcome(files=new_files, applied=applied, misses=misses,
                       rejected=[r for r in rejected if r is not None])
```

Note: a block with `file=None` on a multi-file algorithm is out of scope and lands in `rejected` as skipped, not as a path; `rejected` lists only named paths.

- [ ] **Step 6: Run tests, ruff, commit**

Run: `make check PYTHON=.venv/bin/python`
Expected: pass.

```bash
git add talos/search_replace.py talos/edits.py tests/test_search_replace.py tests/test_edits.py
git commit -m "edits: search/replace engine from prometheus, path-scoped application"
```

---

### Task 5: Budget

**Files:**
- Create: `talos/budget.py`, `tests/test_budget.py`

**Interfaces:**
- Produces: `Budget(usd: float | None, hours: float | None, iterations: int | None, modal_usd: float | None)`, `Spend(llm_usd=0.0, modal_usd=0.0, iterations=0, started_at: float)`, `exhausted(budget, spend, now) -> str | None` returning the dimension name (`"usd" | "hours" | "iterations" | "modal_usd"`), `BudgetExhausted(Exception)` with `.dimension`, `Budget.validate()`.

- [ ] **Step 1: Failing tests**

`tests/test_budget.py`:
```python
import pytest

from talos.budget import Budget, Spend, exhausted


def test_zero_budget_is_exhausted_before_first_call():
    # mutation: `if budget.usd:` treats 0 as unset and never stops
    b = Budget(usd=0.0, hours=None, iterations=None, modal_usd=None)
    assert exhausted(b, Spend(started_at=0.0), now=0.0) == "usd"


def test_boundary_at_cap_is_exhausted():
    # mutation: `>` instead of `>=` lets spend exactly at the cap continue
    b = Budget(usd=1.0, hours=None, iterations=3, modal_usd=None)
    assert exhausted(b, Spend(llm_usd=1.0, started_at=0.0), now=0.0) == "usd"
    assert exhausted(b, Spend(llm_usd=0.99, iterations=3, started_at=0.0), now=0.0) == "iterations"
    assert exhausted(b, Spend(llm_usd=0.99, iterations=2, started_at=0.0), now=0.0) is None


def test_hours_uses_clock_not_wall():
    # mutation: reading time.time() inside makes this untestable and flaky
    b = Budget(usd=None, hours=1.0, iterations=None, modal_usd=None)
    assert exhausted(b, Spend(started_at=100.0), now=100.0 + 3599) is None
    assert exhausted(b, Spend(started_at=100.0), now=100.0 + 3600) == "hours"


def test_all_none_is_invalid():
    # a job with no cap at all must be refused at wizard time
    with pytest.raises(ValueError):
        Budget(usd=None, hours=None, iterations=None, modal_usd=None).validate()
    Budget(usd=None, hours=2.0, iterations=None, modal_usd=None).validate()
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_budget.py -v` → module not found.

- [ ] **Step 3: Write budget.py**

```python
"""Budget caps and measured spend. Pure; the caller passes the clock."""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class Budget:
    usd: float | None
    hours: float | None
    iterations: int | None
    modal_usd: float | None

    def validate(self) -> None:
        if all(v is None for v in (self.usd, self.hours, self.iterations, self.modal_usd)):
            raise ValueError("at least one budget dimension must be set")
        for name in ("usd", "hours", "iterations", "modal_usd"):
            v = getattr(self, name)
            if v is not None and v < 0:
                raise ValueError(f"budget {name} must be >= 0")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Spend:
    started_at: float
    llm_usd: float = 0.0
    modal_usd: float = 0.0
    iterations: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class BudgetExhausted(Exception):
    def __init__(self, dimension: str):
        super().__init__(f"budget exhausted: {dimension}")
        self.dimension = dimension


def exhausted(budget: Budget, spend: Spend, now: float) -> str | None:
    if budget.usd is not None and spend.llm_usd >= budget.usd:
        return "usd"
    if budget.hours is not None and (now - spend.started_at) >= budget.hours * 3600:
        return "hours"
    if budget.iterations is not None and spend.iterations >= budget.iterations:
        return "iterations"
    if budget.modal_usd is not None and spend.modal_usd >= budget.modal_usd:
        return "modal_usd"
    return None
```

- [ ] **Step 4: Run, commit**

Run: `make check PYTHON=.venv/bin/python` → pass.

```bash
git add talos/budget.py tests/test_budget.py
git commit -m "budget: caps, measured spend, boundary-inclusive exhaustion"
```

---

### Task 6: Job spec, state, and timeline

**Files:**
- Create: `talos/state.py`, `tests/test_state.py`

**Interfaces:**
- Produces:
  - `JobSpec(job_id, challenge, direction, provider, model, mode, budget: Budget, rand_hash, tracks: list[str], training: list[NonceSet], holdout: list[NonceSet], fuel: int, created_at: float, monorepo_ref: str, challenge_id: str)` with `to_dict()/from_dict()`, `redacted() -> dict` (no `rand_hash`, no nonce-set hashes).
  - `BaselineRecord(name, adoption, artifact_id, files, training: list[NonceResult], holdout: list[NonceResult])`.
  - `Candidate(iteration, files, artifact_id, training: list[NonceResult], delta: dict, hypothesis: dict)`.
  - `JobState(status, iteration, best: Candidate | None, baseline: BaselineRecord | None, runs_since_improvement, hypotheses: list[dict], spend: Spend, confirmed: list[int], false_positives: list[int], stop_reason: str | None, tacit: str, strategy_counts: dict[str, int])`.
  - `STATUSES` tuple; `TERMINAL = {"won", "exhausted", "failed", "cancelled"}`.
  - `JobStore(run_dir: Path)` with `write_spec(spec)`, `read_spec()`, `save(state)` (atomic), `load()`, `event(kind, **data)` appending to `timeline.jsonl`, `iteration_dir(n)`.

- [ ] **Step 1: Failing tests**

`tests/test_state.py`:
```python
import json

from talos.budget import Budget, Spend
from talos.state import JobSpec, JobState, JobStore, TERMINAL
from talos.types import NonceSet


def spec():
    return JobSpec(job_id="j1", challenge="knapsack", direction="try tabu", provider="fake",
                   model="m", mode="single-shot",
                   budget=Budget(usd=1.0, hours=None, iterations=None, modal_usd=None),
                   rand_hash="ab" * 32, tracks=["n=1"],
                   training=[NonceSet("n=1", "ab" * 32, 0, 2)],
                   holdout=[NonceSet("n=1", "ab" * 32, 1_000_000, 2)],
                   fuel=10, created_at=1.0, monorepo_ref="deadbeef", challenge_id="c003")


def test_spec_roundtrip_and_redaction(tmp_path):
    store = JobStore(tmp_path)
    store.write_spec(spec())
    back = store.read_spec()
    assert back == spec()
    red = spec().redacted()
    # mutation: forgetting to strip nonce-set hashes leaks the seed via `training`
    assert "ab" * 32 not in json.dumps(red)


def test_state_roundtrip_atomic(tmp_path):
    store = JobStore(tmp_path)
    st = JobState.fresh(Spend(started_at=1.0))
    st.status = "researching"
    st.iteration = 3
    store.save(st)
    assert not list(tmp_path.glob("*.tmp"))  # mutation: non-atomic write leaves temp
    assert store.load().iteration == 3 and store.load().status == "researching"


def test_timeline_appends_json_lines(tmp_path):
    store = JobStore(tmp_path)
    store.event("hypothesis", iteration=1, title="x")
    store.event("score", iteration=1, delta=0.1)
    lines = (tmp_path / "timeline.jsonl").read_text().splitlines()
    assert [json.loads(line)["kind"] for line in lines] == ["hypothesis", "score"]
    assert "ts" in json.loads(lines[0])


def test_terminal_set():
    assert TERMINAL == {"won", "exhausted", "failed", "cancelled"}
```

- [ ] **Step 2: Run to verify failure** → module not found.

- [ ] **Step 3: Write state.py**

```python
"""Everything a job persists. JSON on disk under runs/<job_id>/; writes are atomic."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from talos.budget import Budget, Spend
from talos.types import NonceResult, NonceSet

STATUSES = ("queued", "measuring_baseline", "researching", "confirming", "paused",
            "won", "exhausted", "failed", "cancelled")
TERMINAL = {"won", "exhausted", "failed", "cancelled"}


@dataclass
class JobSpec:
    job_id: str
    challenge: str
    direction: str
    provider: str
    model: str
    mode: str
    budget: Budget
    rand_hash: str
    tracks: list[str]
    training: list[NonceSet]
    holdout: list[NonceSet]
    fuel: int
    created_at: float
    monorepo_ref: str
    challenge_id: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["budget"] = self.budget.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "JobSpec":
        d = dict(d)
        d["budget"] = Budget(**d["budget"])
        d["training"] = [NonceSet(**n) for n in d["training"]]
        d["holdout"] = [NonceSet(**n) for n in d["holdout"]]
        return cls(**d)

    def redacted(self) -> dict:
        d = self.to_dict()
        d.pop("rand_hash")
        for key in ("training", "holdout"):
            d[key] = [{"track": n["track"], "start": n["start"], "count": n["count"]}
                      for n in d[key]]
        return d


@dataclass
class BaselineRecord:
    name: str
    adoption: int
    artifact_id: str
    files: dict[str, str]
    training: list[NonceResult]
    holdout: list[NonceResult]

    def to_dict(self) -> dict:
        return {"name": self.name, "adoption": self.adoption, "artifact_id": self.artifact_id,
                "files": self.files, "training": [r.to_dict() for r in self.training],
                "holdout": [r.to_dict() for r in self.holdout]}

    @classmethod
    def from_dict(cls, d: dict) -> "BaselineRecord":
        return cls(name=d["name"], adoption=d["adoption"], artifact_id=d["artifact_id"],
                   files=d["files"], training=[NonceResult.from_dict(r) for r in d["training"]],
                   holdout=[NonceResult.from_dict(r) for r in d["holdout"]])


@dataclass
class Candidate:
    iteration: int
    files: dict[str, str]
    artifact_id: str
    training: list[NonceResult]
    delta: dict
    hypothesis: dict
    holdout: list[NonceResult] | None = None

    def to_dict(self) -> dict:
        return {"iteration": self.iteration, "files": self.files, "artifact_id": self.artifact_id,
                "training": [r.to_dict() for r in self.training], "delta": self.delta,
                "hypothesis": self.hypothesis,
                "holdout": [r.to_dict() for r in self.holdout] if self.holdout else None}

    @classmethod
    def from_dict(cls, d: dict) -> "Candidate":
        return cls(iteration=d["iteration"], files=d["files"], artifact_id=d["artifact_id"],
                   training=[NonceResult.from_dict(r) for r in d["training"]], delta=d["delta"],
                   hypothesis=d["hypothesis"],
                   holdout=[NonceResult.from_dict(r) for r in d["holdout"]] if d.get("holdout") else None)


@dataclass
class JobState:
    status: str
    iteration: int
    best: Candidate | None
    baseline: BaselineRecord | None
    runs_since_improvement: int
    hypotheses: list[dict]
    spend: Spend
    confirmed: list[int] = field(default_factory=list)
    false_positives: list[int] = field(default_factory=list)
    stop_reason: str | None = None
    tacit: str = ""
    strategy_counts: dict[str, int] = field(default_factory=dict)

    @classmethod
    def fresh(cls, spend: Spend) -> "JobState":
        return cls(status="queued", iteration=0, best=None, baseline=None,
                   runs_since_improvement=0, hypotheses=[], spend=spend)

    def to_dict(self) -> dict:
        return {"status": self.status, "iteration": self.iteration,
                "best": self.best.to_dict() if self.best else None,
                "baseline": self.baseline.to_dict() if self.baseline else None,
                "runs_since_improvement": self.runs_since_improvement,
                "hypotheses": self.hypotheses, "spend": self.spend.to_dict(),
                "confirmed": self.confirmed, "false_positives": self.false_positives,
                "stop_reason": self.stop_reason, "tacit": self.tacit,
                "strategy_counts": self.strategy_counts}

    @classmethod
    def from_dict(cls, d: dict) -> "JobState":
        return cls(status=d["status"], iteration=d["iteration"],
                   best=Candidate.from_dict(d["best"]) if d.get("best") else None,
                   baseline=BaselineRecord.from_dict(d["baseline"]) if d.get("baseline") else None,
                   runs_since_improvement=d["runs_since_improvement"],
                   hypotheses=d["hypotheses"], spend=Spend(**d["spend"]),
                   confirmed=d.get("confirmed", []), false_positives=d.get("false_positives", []),
                   stop_reason=d.get("stop_reason"), tacit=d.get("tacit", ""),
                   strategy_counts=d.get("strategy_counts", {}))


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


class JobStore:
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def write_spec(self, spec: JobSpec) -> None:
        _atomic_write(self.run_dir / "job.json", json.dumps(spec.to_dict(), indent=1))

    def read_spec(self) -> JobSpec:
        return JobSpec.from_dict(json.loads((self.run_dir / "job.json").read_text()))

    def save(self, state: JobState) -> None:
        _atomic_write(self.run_dir / "state.json", json.dumps(state.to_dict(), indent=1))

    def load(self) -> JobState:
        return JobState.from_dict(json.loads((self.run_dir / "state.json").read_text()))

    def event(self, kind: str, **data) -> None:
        row = {"ts": time.time(), "kind": kind, **data}
        with (self.run_dir / "timeline.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")

    def iteration_dir(self, n: int) -> Path:
        d = self.run_dir / "iterations" / f"{n:04d}"
        d.mkdir(parents=True, exist_ok=True)
        return d
```

- [ ] **Step 4: Run, commit**

Run: `make check PYTHON=.venv/bin/python` → pass.

```bash
git add talos/state.py tests/test_state.py
git commit -m "state: job spec with redaction, atomic state, timeline"
```

---

### Task 7: Modal bench app and client

**Files:**
- Create: `modal_app/__init__.py`, `modal_app/inside.py`, `modal_app/talos_bench.py`, `talos/bench.py`, `tests/test_inside.py`, `tests/test_bench.py`

**Interfaces:**
- Consumes: `CHALLENGES`, `MONOREPO_REF`, `dev_image` (Task 2); `NonceSet`, `NonceResult`, `CompileResult` (Task 1).
- Produces (container side, pure): `modal_app.inside.stage_algorithm(monorepo: Path, challenge, files, name) -> None`, `unstage_algorithm(monorepo, challenge, name) -> None`, `build(monorepo, challenge, name, run) -> tuple[bool, str]`, `artifact_paths(monorepo, challenge, name) -> tuple[Path, Path | None]`, `run_nonce(challenge_id, track, rand_hash, nonce, so, fuel, timeout_s, ptx, run, workdir) -> dict` (a `NonceResult.to_dict()` shape), `classify(runtime_rc, verifier_rc, quality, timed_out) -> tuple[bool, str | None]`.
- Produces (client side): `talos.bench.Bench` protocol with `compile(challenge, files) -> CompileResult` and `score(challenge, artifact_id, nonce_sets, fuel) -> list[NonceResult]` and `cost_usd_since(mark) -> float`; `ModalBench(app_name="talos-bench")`; `FakeBench(scores: Callable[[str, dict[str,str], NonceSet], list[int | None]], compile_ok=lambda files: True)`; `BenchUnavailable(Exception)`.
- Produces: `modal_app/talos_bench.py` deployable with `modal deploy modal_app/talos_bench.py`, app name `talos-bench`, functions `compile_<challenge>` and `score_nonce_<challenge>` for all eight challenges, Volume `talos-artifacts` at `/artifacts`.

- [ ] **Step 1: Failing tests for inside.py**

`tests/test_inside.py`:
```python
import json
from pathlib import Path

import pytest

from modal_app import inside


def make_monorepo(tmp_path: Path) -> Path:
    mono = tmp_path / "mono"
    (mono / "tig-algorithms" / "src" / "knapsack").mkdir(parents=True)
    (mono / "tig-algorithms" / "src" / "knapsack" / "mod.rs").write_text("// c003_a001\n")
    return mono


class Result:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


def test_stage_writes_files_and_registers_module_once(tmp_path):
    mono = make_monorepo(tmp_path)
    inside.stage_algorithm(mono, "knapsack", {"mod.rs": "fn x(){}", "ls.rs": "fn y(){}"}, "talos_cand")
    d = mono / "tig-algorithms" / "src" / "knapsack" / "talos_cand"
    assert (d / "mod.rs").read_text() == "fn x(){}" and (d / "ls.rs").exists()
    modrs = (mono / "tig-algorithms" / "src" / "knapsack" / "mod.rs").read_text()
    assert modrs.count("pub mod talos_cand;") == 1
    # mutation: unstage that leaves the line behind breaks the next compile
    inside.unstage_algorithm(mono, "knapsack", "talos_cand")
    assert not d.exists()
    assert (mono / "tig-algorithms" / "src" / "knapsack" / "mod.rs").read_text() == "// c003_a001\n"


def test_stage_rejects_path_escape(tmp_path):
    mono = make_monorepo(tmp_path)
    with pytest.raises(ValueError):
        inside.stage_algorithm(mono, "knapsack", {"../evil.rs": "x"}, "talos_cand")


def test_build_invokes_build_algorithm_and_reports(tmp_path):
    calls = []

    def run(cmd, **kw):
        calls.append((cmd, kw.get("cwd")))
        return Result(0, "ok", "")

    ok, out = inside.build(tmp_path, "knapsack", "talos_cand", run)
    assert ok and calls[0][0] == ["build_algorithm", "talos_cand"] and calls[0][1] == tmp_path


def test_classify():
    # exit codes from tig-runtime/src/main.rs and tig-verifier/src/main.rs at MONOREPO_REF
    # mutation: keying ok off runtime rc instead of verifier rc + quality
    assert inside.classify(87, 0, 500, False) == (True, None)        # out of fuel but solved
    assert inside.classify(87, 1, None, False) == (False, "out_of_fuel")
    assert inside.classify(84, 1, None, False) == (False, "no_solution")  # "Runtime Error" exit
    assert inside.classify(0, 1, None, False) == (False, "invalid")       # verifier rejected it
    assert inside.classify(-11, 1, None, False) == (False, "panic")       # killed by a signal
    assert inside.classify(101, 1, None, False) == (False, "panic")       # rust panic exit code
    assert inside.classify(0, 0, None, True) == (False, "timeout")
    assert inside.classify(0, 0, 7, False) == (True, None)


def test_run_nonce_builds_commands_and_parses_quality(tmp_path):
    seen = []

    def run(cmd, **kw):
        seen.append(cmd)
        if cmd[0] == "tig-runtime":
            # tig-runtime's --output names a FOLDER and it writes <nonce>.json inside it.
            # mutation: passing a file path there makes tig-runtime mkdir a folder of that name
            out_dir = Path(cmd[cmd.index("--output") + 1])
            assert out_dir.is_dir(), "--output must name an existing folder"
            (out_dir / f"{cmd[3]}.json").write_text('{"solution": "e30="}')
            return Result(0, "", "")
        return Result(0, "quality: 4242\n", "")

    row = inside.run_nonce("c003", "n=1", "ab" * 32, 7, Path("/lib/x.so"), 10, 600, None, run, tmp_path)
    assert row["ok"] and row["quality"] == 4242 and row["nonce"] == 7 and row["track"] == "n=1"
    rt, ver = seen
    settings = json.loads(rt[1])
    assert settings == {"algorithm_id": "", "challenge_id": "c003", "track_id": "n=1",
                        "block_id": "", "player_id": ""}
    assert rt[2] == "ab" * 32 and rt[3] == "7" and rt[4] == "/lib/x.so"
    assert "--fuel" in rt and rt[rt.index("--fuel") + 1] == "10"
    # tig-verifier takes positional SETTINGS RAND_HASH NONCE SOLUTION_FILE and has no subcommand
    # mutation: inserting a "verify_solution" argv[1] makes clap reject every call
    assert ver[:4] == ["tig-verifier", rt[1], "ab" * 32, "7"] and ver[4].endswith("/7.json")
    assert len(ver) == 5
    assert "--ptx" not in rt  # CPU challenge


def test_run_nonce_gpu_passes_ptx_and_gpu_to_both_binaries(tmp_path):
    seen = []

    def run(cmd, **kw):
        seen.append(cmd)
        if cmd[0] == "tig-runtime":
            (Path(cmd[cmd.index("--output") + 1]) / f"{cmd[3]}.json").write_text("{}")
        return Result(0, "quality: 1\n", "")

    inside.run_nonce("c005", "k=1", "ab" * 32, 3, Path("/a.so"), 10, 600, Path("/a.ptx"), run, tmp_path)
    rt, ver = seen
    for cmd in (rt, ver):  # mutation: dropping --gpu from the verifier call breaks GPU challenges
        assert cmd[cmd.index("--ptx") + 1] == "/a.ptx" and cmd[cmd.index("--gpu") + 1] == "0"


def test_run_nonce_classifies_no_solution(tmp_path):
    def run(cmd, **kw):
        if cmd[0] == "tig-runtime":
            # tig-runtime writes an empty solution and exits 84 when the algorithm returns Err
            (Path(cmd[cmd.index("--output") + 1]) / f"{cmd[3]}.json").write_text('{"solution": ""}')
            return Result(84, "", "Runtime Error: no solution")
        return Result(1, "", "Verification error: Invalid solution")

    row = inside.run_nonce("c003", "n=1", "ab" * 32, 1, Path("/x.so"), 10, 600, None, run, tmp_path)
    assert not row["ok"] and row["error"] == "no_solution" and row["quality"] is None
```

- [ ] **Step 2: Run to verify failure** → `ModuleNotFoundError: modal_app`.

- [ ] **Step 3: Write modal_app/inside.py**

```python
"""Container-side logic for the Modal bench. Pure functions that take the subprocess runner
as a parameter, so they are unit-tested on any machine. Runs inside the TIG dev image,
where `build_algorithm`, `tig-runtime` and `tig-verifier` are on PATH and the monorepo
checkout is the working directory."""
from __future__ import annotations

import json
import re
import subprocess
import tempfile
import time
from pathlib import Path

ALGO_NAME = "talos_cand"
_QUALITY_RE = re.compile(r"quality:\s*(-?\d+)")
# Exit codes, from tig-runtime/src/main.rs at MONOREPO_REF.
RUNTIME_ERROR_RC = 84  # compute_solution returned Err: the algorithm gave up / no solution
OUT_OF_FUEL_RC = 87    # the algorithm library exits 87 when fuel runs out
RUST_PANIC_RC = 101    # a Rust panic that unwinds to main


def _algo_root(monorepo: Path, challenge: str) -> Path:
    return monorepo / "tig-algorithms" / "src" / challenge


def stage_algorithm(monorepo: Path, challenge: str, files: dict[str, str], name: str) -> None:
    root = _algo_root(monorepo, challenge)
    target = root / name
    for rel in files:
        p = (target / rel).resolve()
        if not str(p).startswith(str(target.resolve()) + "/") and p != target.resolve():
            raise ValueError(f"file path escapes algorithm dir: {rel}")
    target.mkdir(parents=True, exist_ok=True)
    for rel, text in files.items():
        p = target / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    mod_rs = root / "mod.rs"
    line = f"pub mod {name};"
    existing = mod_rs.read_text()
    if line not in existing.splitlines():
        if not existing.endswith("\n"):
            existing += "\n"
        mod_rs.write_text(existing + line + "\n")


def unstage_algorithm(monorepo: Path, challenge: str, name: str) -> None:
    import shutil
    root = _algo_root(monorepo, challenge)
    shutil.rmtree(root / name, ignore_errors=True)
    mod_rs = root / "mod.rs"
    line = f"pub mod {name};"
    kept = [ln for ln in mod_rs.read_text().splitlines() if ln != line]
    mod_rs.write_text("\n".join(kept) + "\n")


def build(monorepo: Path, challenge: str, name: str, run=subprocess.run) -> tuple[bool, str]:
    r = run(["build_algorithm", name], cwd=monorepo, capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    return r.returncode == 0, out[-20000:]


def artifact_paths(monorepo: Path, challenge: str, name: str) -> tuple[Path, Path | None]:
    lib = monorepo / "tig-algorithms" / "lib" / challenge
    so = lib / "amd64" / f"{name}.so"
    ptx = lib / "ptx" / f"{name}.ptx"
    return so, (ptx if ptx.exists() else None)


def classify(runtime_rc: int, verifier_rc: int, quality: int | None,
             timed_out: bool) -> tuple[bool, str | None]:
    """A run that saved a solution before running out of fuel still counts if it verifies.
    tig-verifier exits 1 for every rejection, so the runtime exit code carries the reason."""
    if timed_out:
        return False, "timeout"
    if verifier_rc == 0 and quality is not None:
        return True, None
    if runtime_rc == OUT_OF_FUEL_RC:
        return False, "out_of_fuel"
    if runtime_rc < 0 or runtime_rc == RUST_PANIC_RC:
        return False, "panic"
    if runtime_rc == RUNTIME_ERROR_RC:
        return False, "no_solution"
    if runtime_rc != 0:
        return False, "panic"
    return False, "invalid"


def run_nonce(challenge_id: str, track: str, rand_hash: str, nonce: int, so: Path, fuel: int,
              timeout_s: int, ptx: Path | None, run=subprocess.run,
              workdir: Path | None = None) -> dict:
    """Mirrors scripts/test_algorithm in the monorepo:
    `tig-runtime SETTINGS RAND_HASH NONCE SO --fuel F --output DIR [--ptx P --gpu 0]` writes
    DIR/<nonce>.json, then `tig-verifier SETTINGS RAND_HASH NONCE DIR/<nonce>.json [--ptx P --gpu 0]`
    prints `quality: N` and exits 0 on a valid solution."""
    settings = json.dumps({"algorithm_id": "", "challenge_id": challenge_id, "track_id": track,
                           "block_id": "", "player_id": ""}, separators=(",", ":"))
    gpu_args = ["--ptx", str(ptx), "--gpu", "0"] if ptx else []
    with tempfile.TemporaryDirectory(dir=workdir) as td:
        out_file = Path(td) / f"{nonce}.json"
        cmd = ["tig-runtime", settings, rand_hash, str(nonce), str(so),
               "--fuel", str(fuel), "--output", td] + gpu_args
        t0 = time.time()
        timed_out = False
        try:
            r1 = run(cmd, capture_output=True, text=True, timeout=timeout_s)
            rt_rc = r1.returncode
        except subprocess.TimeoutExpired:
            timed_out, rt_rc = True, -1
        runtime_ms = int((time.time() - t0) * 1000)
        quality = None
        ver_rc = 1
        if not timed_out and out_file.exists():
            r2 = run(["tig-verifier", settings, rand_hash, str(nonce), str(out_file)] + gpu_args,
                     capture_output=True, text=True, timeout=timeout_s)
            ver_rc = r2.returncode
            m = _QUALITY_RE.search(r2.stdout or "")
            quality = int(m.group(1)) if m else None
    ok, err = classify(rt_rc, ver_rc, quality, timed_out)
    return {"track": track, "nonce": nonce, "ok": ok, "quality": quality if ok else None,
            "runtime_ms": runtime_ms, "error": err}
```

- [ ] **Step 4: Run inside tests** → pass.

- [ ] **Step 5: Write modal_app/talos_bench.py**

One app, eight image variants, sixteen functions registered in a loop with `serialized=True` so Modal does not need to import them by attribute name.

```python
"""Modal app `talos-bench`. Deployed once into the user's Modal account by `talos setup`.
Every function runs inside the official TIG dev image for its challenge plus a pinned
monorepo checkout at /app. Artifacts live on the `talos-artifacts` Volume keyed by the
content hash of the submitted files."""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import modal

from modal_app import inside
from talos.challenges import CHALLENGES, MONOREPO_REF, dev_image

APP_NAME = "talos-bench"
ARTIFACTS = "/artifacts"
MONOREPO = Path("/app")
NONCE_TIMEOUT_S = 600

app = modal.App(APP_NAME)
volume = modal.Volume.from_name("talos-artifacts", create_if_missing=True)


def _image(name: str) -> modal.Image:
    return (
        modal.Image.from_registry(dev_image(name), add_python="3.11")
        .apt_install("git")
        .run_commands(
            "git clone https://github.com/tig-foundation/tig-monorepo.git /app",
            f"cd /app && git checkout {MONOREPO_REF}",
        )
        .env({"CHALLENGE": name})
        .add_local_python_source("modal_app", "talos")
    )


def content_hash(files: dict[str, str]) -> str:
    h = hashlib.sha256()
    for k in sorted(files):
        h.update(k.encode())
        h.update(b"\0")
        h.update(files[k].encode())
        h.update(b"\0")
    return h.hexdigest()[:32]


def _compile_impl(name: str, files: dict[str, str]) -> dict:
    art_id = content_hash(files)
    dest = Path(ARTIFACTS) / name / art_id
    if (dest / "algo.so").exists():
        return {"ok": True, "artifact_id": art_id, "output": "cached"}
    inside.stage_algorithm(MONOREPO, name, files, inside.ALGO_NAME)
    try:
        ok, out = inside.build(MONOREPO, name, inside.ALGO_NAME)
        if not ok:
            return {"ok": False, "artifact_id": None, "output": out}
        so, ptx = inside.artifact_paths(MONOREPO, name, inside.ALGO_NAME)
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(so, dest / "algo.so")
        if ptx:
            shutil.copy2(ptx, dest / "algo.ptx")
        volume.commit()
        return {"ok": True, "artifact_id": art_id, "output": out}
    finally:
        inside.unstage_algorithm(MONOREPO, name, inside.ALGO_NAME)


def _score_impl(name: str, challenge_id: str, artifact_id: str, track: str, rand_hash: str,
                nonce: int, fuel: int) -> dict:
    volume.reload()
    d = Path(ARTIFACTS) / name / artifact_id
    so, ptx = d / "algo.so", d / "algo.ptx"
    return inside.run_nonce(challenge_id, track, rand_hash, nonce, so, fuel, NONCE_TIMEOUT_S,
                            ptx if ptx.exists() else None, workdir=MONOREPO)


for _name, _spec in CHALLENGES.items():
    _kw = dict(image=_image(_name), volumes={ARTIFACTS: volume}, serialized=True)
    if _spec.is_gpu:
        _kw["gpu"] = _spec.gpu
    else:
        _kw["cpu"] = _spec.cpu
        _kw["memory"] = _spec.memory_mib

    def _mk_compile(n=_name):
        def compile_fn(files: dict) -> dict:
            return _compile_impl(n, files)
        return compile_fn

    def _mk_score(n=_name, cid=_spec.id):
        def score_nonce(artifact_id: str, track: str, rand_hash: str, nonce: int, fuel: int) -> dict:
            return _score_impl(n, cid, artifact_id, track, rand_hash, nonce, fuel)
        return score_nonce

    app.function(name=f"compile_{_name}", timeout=3600, **_kw)(_mk_compile())
    app.function(name=f"score_nonce_{_name}", timeout=NONCE_TIMEOUT_S + 120, **_kw)(_mk_score())
```

If `modal deploy` rejects `serialized=True` combined with `name=` in the installed Modal version, define the sixteen functions explicitly with `@app.function(...)` decorators instead; the bodies stay one-liners calling `_compile_impl` / `_score_impl`.

- [ ] **Step 6: Failing tests for the client**

`tests/test_bench.py`:
```python
from talos.bench import FakeBench
from talos.types import NonceSet


def test_fake_bench_scores_per_nonce_and_tracks_cost():
    def scores(challenge, files, ns):
        return [100 + n for n in ns.nonces()]
    fb = FakeBench(scores)
    c = fb.compile("knapsack", {"mod.rs": "x"})
    assert c.ok and c.artifact_id
    mark = fb.cost_mark()
    res = fb.score("knapsack", c.artifact_id, [NonceSet("t", "ab", 0, 3)], fuel=1)
    assert [r.quality for r in res] == [100, 101, 102]
    assert all(r.track == "t" for r in res)
    assert fb.cost_usd_since(mark) > 0  # mutation: not charging makes budget tests vacuous


def test_fake_bench_none_means_error():
    fb = FakeBench(lambda ch, files, ns: [None, 5])
    c = fb.compile("knapsack", {"mod.rs": "x"})
    res = fb.score("knapsack", c.artifact_id, [NonceSet("t", "ab", 0, 2)], fuel=1)
    assert not res[0].ok and res[0].error == "no_solution" and res[1].quality == 5


def test_fake_bench_compile_failure():
    fb = FakeBench(lambda ch, files, ns: [], compile_ok=lambda files: "BUG" not in files["mod.rs"])
    assert not fb.compile("knapsack", {"mod.rs": "BUG"}).ok
```

- [ ] **Step 7: Write talos/bench.py**

```python
"""Bench client. ModalBench talks to the deployed `talos-bench` app; FakeBench drives the
loop in tests. Both return the same shapes."""
from __future__ import annotations

import hashlib
import time
from typing import Callable, Protocol

from talos.challenges import CHALLENGES
from talos.types import CompileResult, NonceResult, NonceSet

# Rough Modal list prices, $/second, used only for budget accounting (marked estimated).
CPU_USD_PER_CORE_SECOND = 0.0000131
MEM_USD_PER_GIB_SECOND = 0.00000222
GPU_USD_PER_SECOND = {"L40S": 0.000542}


class BenchUnavailable(Exception):
    pass


class Bench(Protocol):
    def compile(self, challenge: str, files: dict[str, str]) -> CompileResult: ...
    def score(self, challenge: str, artifact_id: str, nonce_sets: list[NonceSet],
              fuel: int) -> list[NonceResult]: ...
    def cost_mark(self) -> float: ...
    def cost_usd_since(self, mark: float) -> float: ...


def _seconds_cost(challenge: str, seconds: float) -> float:
    spec = CHALLENGES[challenge]
    if spec.is_gpu:
        return seconds * GPU_USD_PER_SECOND[spec.gpu]
    return seconds * (spec.cpu * CPU_USD_PER_CORE_SECOND
                      + spec.memory_mib / 1024 * MEM_USD_PER_GIB_SECOND)


class ModalBench:
    def __init__(self, app_name: str = "talos-bench", retry_window_s: int = 900):
        self.app_name = app_name
        self.retry_window_s = retry_window_s
        self._cost = 0.0

    def _fn(self, name: str):
        import modal
        try:
            fn = modal.Function.from_name(self.app_name, name)
            fn.hydrate()  # from_name is lazy; hydrate forces the lookup so "not deployed" fails here
            return fn
        except Exception as e:  # noqa: BLE001 - any lookup failure means not deployed
            raise BenchUnavailable(f"Modal function {name} not found in app {self.app_name}: "
                                   f"{e}. Run `talos setup` to deploy.") from None

    def _with_retry(self, call: Callable[[], object]):
        deadline = time.time() + self.retry_window_s
        delay = 5.0
        while True:
            try:
                return call()
            except BenchUnavailable:
                raise
            except Exception as e:  # noqa: BLE001 - Modal raises many transport types
                if time.time() + delay > deadline:
                    raise BenchUnavailable(f"Modal unreachable for {self.retry_window_s}s: {e}")
                time.sleep(delay)
                delay = min(delay * 2, 60)

    def compile(self, challenge: str, files: dict[str, str]) -> CompileResult:
        fn = self._fn(f"compile_{challenge}")
        t0 = time.time()
        out = self._with_retry(lambda: fn.remote(files))
        self._cost += _seconds_cost(challenge, time.time() - t0)
        return CompileResult(ok=out["ok"], artifact_id=out.get("artifact_id"), output=out["output"])

    def score(self, challenge: str, artifact_id: str, nonce_sets: list[NonceSet],
              fuel: int) -> list[NonceResult]:
        fn = self._fn(f"score_nonce_{challenge}")
        args = [(artifact_id, ns.track, ns.rand_hash, n, fuel) for ns in nonce_sets
                for n in ns.nonces()]
        rows = self._with_retry(lambda: list(fn.starmap(args)))
        results = [NonceResult.from_dict(r) for r in rows]
        self._cost += sum(_seconds_cost(challenge, r.runtime_ms / 1000) for r in results)
        return results

    def cost_mark(self) -> float:
        return self._cost

    def cost_usd_since(self, mark: float) -> float:
        return self._cost - mark


class FakeBench:
    """scores(challenge, files, nonce_set) -> list of quality per nonce, None = error."""

    def __init__(self, scores: Callable[[str, dict[str, str], NonceSet], list[int | None]],
                 compile_ok: Callable[[dict[str, str]], bool] = lambda files: True,
                 usd_per_nonce: float = 0.01):
        self._scores = scores
        self._compile_ok = compile_ok
        self._usd_per_nonce = usd_per_nonce
        self._files: dict[str, dict[str, str]] = {}
        self._cost = 0.0
        self.compile_calls = 0
        self.score_calls = 0

    def compile(self, challenge: str, files: dict[str, str]) -> CompileResult:
        self.compile_calls += 1
        if not self._compile_ok(files):
            return CompileResult(ok=False, artifact_id=None, output="error[E0308]: mismatched types")
        art = hashlib.sha256(repr(sorted(files.items())).encode()).hexdigest()[:16]
        self._files[art] = dict(files)
        return CompileResult(ok=True, artifact_id=art, output="ok")

    def score(self, challenge: str, artifact_id: str, nonce_sets: list[NonceSet],
              fuel: int) -> list[NonceResult]:
        self.score_calls += 1
        out: list[NonceResult] = []
        for ns in nonce_sets:
            qs = self._scores(challenge, self._files[artifact_id], ns)
            for n, q in zip(ns.nonces(), qs):
                self._cost += self._usd_per_nonce
                if q is None:
                    out.append(NonceResult(ns.track, n, False, None, 1, "no_solution"))
                else:
                    out.append(NonceResult(ns.track, n, True, q, 1, None))
        return out

    def cost_mark(self) -> float:
        return self._cost

    def cost_usd_since(self, mark: float) -> float:
        return self._cost - mark
```

- [ ] **Step 8: Run, ruff, commit**

Run: `make check PYTHON=.venv/bin/python` → pass. `modal_app/talos_bench.py` imports `modal`, which is installed; it is not executed by tests.

```bash
git add modal_app/__init__.py modal_app/inside.py modal_app/talos_bench.py talos/bench.py tests/test_inside.py tests/test_bench.py
git commit -m "bench: modal app in the TIG dev image, pure container logic, client and fake"
```

---

### Task 8: Providers and pricing

**Files:**
- Create: `talos/providers/__init__.py`, `talos/providers/pricing.py`, `talos/providers/openai_compat.py`, `talos/providers/anthropic_provider.py`, `talos/providers/google.py`, `talos/providers/claude_cli.py`, `talos/providers/codex_cli.py`, `talos/providers/fake.py`, `tests/test_providers.py`
- Modify: `pyproject.toml` (add `anthropic>=1,<2` to dependencies)

**Interfaces:**
- Consumes: `Usage`, `Completion` (Task 1).
- Produces: `talos.providers.Provider` protocol: `complete(system: str, user: str) -> Completion`, attribute `name: str`, attribute `metered: bool` (False for CLI providers); `ProviderAuthError`, `ProviderRateLimited`, `ProviderError`; `make_provider(kind, model, api_key=None, api_base=None) -> Provider` for kinds `anthropic | openai | google | openrouter | custom | claude-cli | codex-cli | fake`; `DEFAULT_MODELS: dict[str, str]`; `validate_provider(p) -> str | None` (None on success, message on failure; makes one tiny call).
- Produces: `talos.providers.pricing.estimate_cost(model, usage) -> float | None` and `PRICES: dict[str, tuple[float, float]]` in $ per million input/output tokens.
- Produces: `talos.providers.fake.FakeProvider(script: list[str] | Callable[[str, str], str], usd_per_call=0.01)` recording `calls: list[tuple[str, str]]`.

- [ ] **Step 1: Failing tests**

`tests/test_providers.py`:
```python
import json
import types

import httpx
import pytest

from talos.providers import (ProviderAuthError, ProviderRateLimited, make_provider,
                             validate_provider)
from talos.providers.fake import FakeProvider
from talos.providers.openai_compat import OpenAICompat
from talos.providers.pricing import estimate_cost
from talos.providers.claude_cli import ClaudeCli
from talos.providers.codex_cli import CodexCli
from talos.types import Usage


def test_pricing_known_and_unknown():
    # mutation: swapping input/output rates changes 1M-in vs 1M-out asymmetry
    assert estimate_cost("claude-opus-5", Usage(1_000_000, 0)) == pytest.approx(5.0)
    assert estimate_cost("claude-opus-5", Usage(0, 1_000_000)) == pytest.approx(25.0)
    assert estimate_cost("some/unknown-model", Usage(10, 10)) is None


def test_fake_provider_scripted_and_metered():
    p = FakeProvider(["first", "second"])
    a = p.complete("sys", "u1")
    b = p.complete("sys", "u2")
    assert (a.text, b.text) == ("first", "second") and p.calls == [("sys", "u1"), ("sys", "u2")]
    assert a.usage.cost_usd == pytest.approx(0.01) and p.metered


def test_openai_compat_request_shape_and_usage():
    seen = {}
    def post(url, body, headers):
        seen.update(url=url, body=body, headers=headers)
        return {"choices": [{"message": {"content": "hello"}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 3}}
    p = OpenAICompat(api_base="https://api.openai.com/v1", api_key="k", model="gpt-5", post=post)
    c = p.complete("SYS", "USER")
    assert c.text == "hello" and c.usage.input_tokens == 12 and c.usage.output_tokens == 3
    assert seen["url"] == "https://api.openai.com/v1/chat/completions"
    assert seen["headers"]["Authorization"] == "Bearer k"
    assert seen["body"]["messages"][0] == {"role": "system", "content": "SYS"}
    assert seen["body"]["messages"][1] == {"role": "user", "content": "USER"}


def test_openai_compat_maps_http_errors():
    # mutation: treating 401 as retryable would loop forever on a bad key
    from talos.providers.openai_compat import HTTPError
    def post401(url, body, headers):
        raise HTTPError(401, "bad key")
    def post429(url, body, headers):
        raise HTTPError(429, "slow down")
    with pytest.raises(ProviderAuthError):
        OpenAICompat("https://x/v1", "k", "m", post=post401).complete("s", "u")
    with pytest.raises(ProviderRateLimited):
        OpenAICompat("https://x/v1", "k", "m", post=post429).complete("s", "u")


def test_claude_cli_parses_json_result_and_cost():
    def run(cmd, input=None, **kw):
        # mutation: shutil.which() in argv[0] makes this machine-dependent
        assert cmd[:2] == ["claude", "-p"] and "--output-format" in cmd and "json" in cmd
        assert "--system-prompt" in cmd and input == "USER"
        class R:
            returncode = 0
            stdout = json.dumps({"result": "the code", "total_cost_usd": 0.42,
                                 "usage": {"input_tokens": 5, "output_tokens": 7}})
            stderr = ""
        return R()
    p = ClaudeCli(model="claude-opus-5", run=run)
    c = p.complete("SYS", "USER")
    assert c.text == "the code" and c.usage.cost_usd == 0.42 and not p.metered


def test_codex_cli_reads_last_message_file(tmp_path):
    def run(cmd, **kw):
        # mutation: shutil.which() in argv[0] makes this machine-dependent
        assert cmd[:2] == ["codex", "exec"] and "-m" in cmd
        out = cmd[cmd.index("-o") + 1]
        open(out, "w").write("edited code")

        class R:
            returncode = 0
            stdout = ""
            stderr = ""
        return R()
    p = CodexCli(model="gpt-5-codex", run=run)
    c = p.complete("SYS", "USER")
    assert c.text == "edited code" and not p.metered


def test_make_provider_kinds_and_validate():
    fp = make_provider("fake", "m")
    assert isinstance(fp, FakeProvider) and validate_provider(fp) is None
    with pytest.raises(ValueError):
        make_provider("nope", "m")


def test_anthropic_provider_maps_errors_and_prices_usage():
    import anthropic

    from talos.providers.anthropic_provider import AnthropicProvider

    class Block:
        type = "text"
        text = "done"

    class Msg:
        stop_reason = "end_turn"
        content = [Block()]
        usage = types.SimpleNamespace(input_tokens=1_000_000, output_tokens=0)

    class Stream:
        def __init__(self, exc):
            self.exc = exc

        def __enter__(self):
            if self.exc:
                raise self.exc
            return self

        def __exit__(self, *a):
            return False

        def get_final_message(self):
            return Msg()

    def client(exc=None):
        return types.SimpleNamespace(messages=types.SimpleNamespace(stream=lambda **kw: Stream(exc)))

    p = AnthropicProvider(model="claude-opus-5", api_key="k", client=client())
    c = p.complete("s", "u")
    assert c.text == "done" and c.usage.cost_usd == pytest.approx(5.0) and p.metered
    # mutation: mapping RateLimitError to ProviderError makes the loop fail instead of wait
    resp = httpx.Response(429, request=httpx.Request("POST", "https://x"))
    err = anthropic.RateLimitError("slow", response=resp, body=None)
    with pytest.raises(ProviderRateLimited):
        AnthropicProvider("claude-opus-5", "k", client=client(err)).complete("s", "u")
    resp = httpx.Response(401, request=httpx.Request("POST", "https://x"))
    err = anthropic.AuthenticationError("bad", response=resp, body=None)
    with pytest.raises(ProviderAuthError):
        AnthropicProvider("claude-opus-5", "k", client=client(err)).complete("s", "u")
```

- [ ] **Step 2: Run to verify failure** → module not found.

- [ ] **Step 3: Write pricing.py**

Prices in $ per million tokens (input, output). Anthropic first-party rates as of 2026-06; OpenAI and Google entries are placeholders the executor must fill from each vendor's current price page before shipping, or leave absent so `estimate_cost` returns `None` and the CLI shows "unpriced".

```python
"""Per-model prices in USD per million tokens: (input, output). Unknown models cost None,
which the CLI reports as "unpriced" rather than zero."""
from __future__ import annotations

from talos.types import Usage

PRICES: dict[str, tuple[float, float]] = {
    "claude-fable-5-1": (10.0, 50.0),
    "claude-fable-5": (10.0, 50.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


def estimate_cost(model: str, usage: Usage) -> float | None:
    key = model.split("/", 1)[-1] if "/" in model else model  # openrouter "vendor/model"
    p = PRICES.get(key)
    if p is None:
        return None
    return usage.input_tokens / 1e6 * p[0] + usage.output_tokens / 1e6 * p[1]
```

- [ ] **Step 4: Write providers/__init__.py**

```python
"""Provider protocol and factory. Every backend returns a Completion with measured usage."""
from __future__ import annotations

from typing import Protocol

from talos.types import Completion


class ProviderError(RuntimeError):
    pass


class ProviderAuthError(ProviderError):
    """Bad or missing credential, billing failure. Never retried."""


class ProviderRateLimited(ProviderError):
    """429 or equivalent. The loop waits and retries."""


class Provider(Protocol):
    name: str
    metered: bool

    def complete(self, system: str, user: str) -> Completion: ...


DEFAULT_MODELS = {
    "anthropic": "claude-opus-5",
    "openai": "gpt-5",
    "google": "gemini-2.5-pro",
    "openrouter": "anthropic/claude-opus-5",
    "custom": "",
    "claude-cli": "claude-opus-5",
    "codex-cli": "gpt-5-codex",
    "fake": "fake",
}

KINDS = tuple(DEFAULT_MODELS)


def make_provider(kind: str, model: str, api_key: str | None = None,
                  api_base: str | None = None) -> Provider:
    if kind == "anthropic":
        from talos.providers.anthropic_provider import AnthropicProvider
        return AnthropicProvider(model=model, api_key=api_key)
    if kind in ("openai", "openrouter", "custom"):
        from talos.providers.openai_compat import OpenAICompat
        base = api_base or {"openai": "https://api.openai.com/v1",
                            "openrouter": "https://openrouter.ai/api/v1"}.get(kind)
        if not base:
            raise ValueError("custom provider needs api_base")
        return OpenAICompat(api_base=base, api_key=api_key or "", model=model)
    if kind == "google":
        from talos.providers.google import GoogleProvider
        return GoogleProvider(model=model, api_key=api_key or "")
    if kind == "claude-cli":
        from talos.providers.claude_cli import ClaudeCli
        return ClaudeCli(model=model)
    if kind == "codex-cli":
        from talos.providers.codex_cli import CodexCli
        return CodexCli(model=model)
    if kind == "fake":
        from talos.providers.fake import FakeProvider
        return FakeProvider(lambda s, u: "<<<<<<< SEARCH\nx\n=======\ny\n>>>>>>> REPLACE")
    raise ValueError(f"unknown provider kind {kind!r}; choose one of {KINDS}")


def validate_provider(p: Provider) -> str | None:
    """One tiny call. None on success, else a message naming what to fix."""
    try:
        c = p.complete("Reply with the single word OK.", "Say OK.")
    except ProviderAuthError as e:
        return f"credential rejected: {e}"
    except ProviderError as e:
        return f"provider error: {e}"
    return None if c.text.strip() else "provider returned an empty reply"
```

- [ ] **Step 5: Write openai_compat.py**

```python
"""OpenAI-compatible chat completions over urllib. Covers OpenAI, OpenRouter, DeepSeek-style
endpoints and any local server. `post` is injectable for tests."""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from talos.providers import ProviderAuthError, ProviderError, ProviderRateLimited
from talos.providers.pricing import estimate_cost
from talos.types import Completion, Usage


class HTTPError(Exception):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:300]}")
        self.status = status
        self.body = body


def _post_json(url: str, body: dict, headers: dict) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise HTTPError(e.code, e.read().decode("utf-8", "replace")) from None
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise ProviderError(f"network error: {e}") from None


def _map_http(e: HTTPError) -> ProviderError:
    if e.status in (401, 403, 402):
        return ProviderAuthError(str(e))
    if e.status == 429:
        return ProviderRateLimited(str(e))
    return ProviderError(str(e))


class OpenAICompat:
    metered = True

    def __init__(self, api_base: str, api_key: str, model: str, post=_post_json,
                 max_tokens: int = 16000):
        self.name = f"openai-compat:{api_base}"
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self._post = post
        self.max_tokens = max_tokens

    def complete(self, system: str, user: str) -> Completion:
        body = {"model": self.model,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}],
                "max_completion_tokens": self.max_tokens}
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            resp = self._post(f"{self.api_base}/chat/completions", body, headers)
        except HTTPError as e:
            if e.status == 400 and "max_completion_tokens" in e.body:
                body["max_tokens"] = body.pop("max_completion_tokens")
                try:
                    resp = self._post(f"{self.api_base}/chat/completions", body, headers)
                except HTTPError as e2:
                    raise _map_http(e2) from None
            else:
                raise _map_http(e) from None
        try:
            text = resp["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            raise ProviderError(f"unexpected response shape: {str(resp)[:300]}") from None
        u = resp.get("usage") or {}
        usage = Usage(int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0)))
        usage.cost_usd = estimate_cost(self.model, usage)
        return Completion(text=text, usage=usage)
```

- [ ] **Step 6: Write anthropic_provider.py**

Uses the official SDK. Adaptive thinking and effort are passed for every model; the setup wizard's validation call surfaces a model that rejects them.

```python
"""Anthropic backend via the official SDK."""
from __future__ import annotations

from talos.providers import ProviderAuthError, ProviderError, ProviderRateLimited
from talos.providers.pricing import estimate_cost
from talos.types import Completion, Usage


class AnthropicProvider:
    metered = True

    def __init__(self, model: str, api_key: str | None, effort: str = "high",
                 max_tokens: int = 32000, client=None):
        import anthropic
        self.name = "anthropic"
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self._client = client or anthropic.Anthropic(api_key=api_key)

    def complete(self, system: str, user: str) -> Completion:
        import anthropic
        try:
            with self._client.messages.stream(
                model=self.model, max_tokens=self.max_tokens, system=system,
                thinking={"type": "adaptive"}, output_config={"effort": self.effort},
                messages=[{"role": "user", "content": user}],
            ) as stream:
                msg = stream.get_final_message()
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as e:
            raise ProviderAuthError(str(e)) from None
        except anthropic.RateLimitError as e:
            raise ProviderRateLimited(str(e)) from None
        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                raise ProviderRateLimited(f"server error {e.status_code}") from None
            raise ProviderError(str(e)) from None
        except anthropic.APIConnectionError as e:
            raise ProviderRateLimited(f"connection error: {e}") from None
        if msg.stop_reason == "refusal":
            raise ProviderError("model refused the request")
        text = "".join(b.text for b in msg.content if b.type == "text")
        usage = Usage(msg.usage.input_tokens, msg.usage.output_tokens)
        usage.cost_usd = estimate_cost(self.model, usage)
        return Completion(text=text, usage=usage)
```

- [ ] **Step 7: Write google.py**

```python
"""Gemini generateContent over urllib."""
from __future__ import annotations

from talos.providers import ProviderError
from talos.providers.openai_compat import HTTPError, _map_http, _post_json
from talos.providers.pricing import estimate_cost
from talos.types import Completion, Usage

BASE = "https://generativelanguage.googleapis.com/v1beta/models"


class GoogleProvider:
    metered = True

    def __init__(self, model: str, api_key: str, post=_post_json):
        self.name = "google"
        self.model = model
        self.api_key = api_key
        self._post = post

    def complete(self, system: str, user: str) -> Completion:
        body = {"system_instruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}]}
        url = f"{BASE}/{self.model}:generateContent"
        try:
            resp = self._post(url, body, {"x-goog-api-key": self.api_key})  # never in the URL
        except HTTPError as e:
            raise _map_http(e) from None
        try:
            text = "".join(p.get("text", "") for p in resp["candidates"][0]["content"]["parts"])
        except (KeyError, IndexError, TypeError):
            raise ProviderError(f"unexpected response shape: {str(resp)[:300]}") from None
        u = resp.get("usageMetadata") or {}
        usage = Usage(int(u.get("promptTokenCount", 0)), int(u.get("candidatesTokenCount", 0)))
        usage.cost_usd = estimate_cost(self.model, usage)
        return Completion(text=text, usage=usage)
```

- [ ] **Step 8: Write claude_cli.py and codex_cli.py**

```python
"""Headless `claude -p` using the user's Claude subscription login. No key is stored."""
from __future__ import annotations

import json
import subprocess

from talos.providers import ProviderAuthError, ProviderError
from talos.types import Completion, Usage


class ClaudeCli:
    metered = False

    def __init__(self, model: str, run=subprocess.run, timeout_s: int = 1800):
        self.name = "claude-cli"
        self.model = model
        self._run = run
        self.timeout_s = timeout_s

    def complete(self, system: str, user: str) -> Completion:
        cmd = ["claude", "-p", "--output-format", "json", "--model", self.model,
               "--system-prompt", system]
        try:
            r = self._run(cmd, input=user, capture_output=True, text=True, timeout=self.timeout_s)
        except FileNotFoundError:
            raise ProviderAuthError("claude CLI not found on PATH; install Claude Code") from None
        except subprocess.TimeoutExpired:
            raise ProviderError("claude CLI timed out") from None
        if r.returncode != 0:
            err = (r.stderr or r.stdout or "")[-500:]
            if "login" in err.lower() or "auth" in err.lower():
                raise ProviderAuthError(f"claude CLI not logged in: {err}")
            raise ProviderError(f"claude CLI failed: {err}")
        try:
            data = json.loads(r.stdout)
        except json.JSONDecodeError:
            return Completion(text=r.stdout, usage=Usage())
        u = data.get("usage") or {}
        usage = Usage(int(u.get("input_tokens", 0)), int(u.get("output_tokens", 0)),
                      cost_usd=data.get("total_cost_usd"))
        return Completion(text=data.get("result", ""), usage=usage)
```

```python
"""Headless `codex exec` using the user's ChatGPT/Codex login. The system prompt is folded
into the prompt because codex exec has no system flag."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from talos.providers import ProviderAuthError, ProviderError
from talos.types import Completion, Usage


class CodexCli:
    metered = False

    def __init__(self, model: str, run=subprocess.run, timeout_s: int = 1800):
        self.name = "codex-cli"
        self.model = model
        self._run = run
        self.timeout_s = timeout_s

    def complete(self, system: str, user: str) -> Completion:
        prompt = f"{system}\n\n---\n\n{user}"
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "last.md"
            cmd = ["codex", "exec", "-m", self.model, "--skip-git-repo-check", "-o", str(out), prompt]
            try:
                r = self._run(cmd, capture_output=True, text=True, timeout=self.timeout_s, cwd=td)
            except FileNotFoundError:
                raise ProviderAuthError("codex CLI not found on PATH") from None
            except subprocess.TimeoutExpired:
                raise ProviderError("codex CLI timed out") from None
            if r.returncode != 0:
                err = (r.stderr or r.stdout or "")[-500:]
                if "login" in err.lower() or "auth" in err.lower():
                    raise ProviderAuthError(f"codex CLI not logged in: {err}")
                raise ProviderError(f"codex CLI failed: {err}")
            text = out.read_text() if out.exists() else r.stdout
        return Completion(text=text, usage=Usage())
```

- [ ] **Step 9: Write fake.py**

```python
"""Scripted provider for tests and the fake end-to-end run."""
from __future__ import annotations

from talos.types import Completion, Usage


class FakeProvider:
    metered = True

    def __init__(self, script, usd_per_call: float = 0.01):
        self.name = "fake"
        self._script = script
        self._i = 0
        self.usd_per_call = usd_per_call
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> Completion:
        self.calls.append((system, user))
        if callable(self._script):
            text = self._script(system, user)
        else:
            text = self._script[min(self._i, len(self._script) - 1)]
            self._i += 1
        return Completion(text=text, usage=Usage(100, 50, cost_usd=self.usd_per_call))
```

- [ ] **Step 10: Add the dependency, run, commit**

In `pyproject.toml` change `dependencies = ["modal>=1.5,<2", "rich>=13"]` to `dependencies = ["modal>=1.5,<2", "rich>=13", "anthropic>=1,<2"]`, then `uv pip install --python .venv/bin/python -e '.[dev]'`.

Run: `make check PYTHON=.venv/bin/python` → pass.

```bash
git add pyproject.toml talos/providers tests/test_providers.py
git commit -m "providers: anthropic sdk, openai-compatible, google, claude and codex cli, fake, pricing"
```

---

### Task 9: Prompts and parsers

**Files:**
- Create: `talos/prompts.py`, `talos/data/rust_rules.md`, `tests/test_prompts.py`

**Interfaces:**
- Produces: `STRATEGY_TAGS: list[str]`; `hypothesis_prompts(ctx: PromptContext) -> tuple[str, str]` (system, user); `edit_prompts(ctx, hypothesis: dict) -> tuple[str, str]`; `compile_fix_prompts(ctx, files, compiler_output) -> tuple[str, str]`; `edit_repair_prompts(ctx, files, misses_text) -> tuple[str, str]`; `distill_prompts(ctx, failed: list[dict]) -> tuple[str, str]`; `parse_hypothesis(text) -> dict` with keys `title, description, strategy_tag`; `parse_distillation(text) -> str | None`; `PromptContext(challenge, template_rs, direction, tacit, files, baseline_name, best_delta, failed_hypotheses: list[dict], forced_tag: str | None, is_gpu)`.

- [ ] **Step 1: Copy the Rust rules block**

From the Prometheus clone made in Task 4, copy the text of the `RUST_RULES_DETAILED` string in `scripts/prompts.py` (the multi-line string literal, without the Python quotes) into `talos/data/rust_rules.md`. Prepend the line `<!-- Lifted from tig-foundation/prometheus-swarm scripts/prompts.py (GPLv3) -->`. If the clone is gone, re-run the `gh repo clone` line from Task 4.

- [ ] **Step 2: Failing tests**

`tests/test_prompts.py`:
```python
import pytest

from talos.prompts import (PromptContext, STRATEGY_TAGS, distill_prompts, edit_prompts,
                           hypothesis_prompts, parse_distillation, parse_hypothesis)


def ctx(**kw):
    base = dict(challenge="knapsack", template_rs="pub fn solve_challenge(", direction="Try tabu",
                tacit="- lesson one", files={"mod.rs": "fn x(){}"}, baseline_name="qk_v3",
                best_delta=-0.01, failed_hypotheses=[], forced_tag=None, is_gpu=False)
    base.update(kw)
    return PromptContext(**base)


def test_hypothesis_prompt_carries_direction_tacit_and_target():
    system, user = hypothesis_prompts(ctx())
    assert "Try tabu" in user and "lesson one" in user and "qk_v3" in user
    assert "solve_challenge" in system
    assert all(t in system for t in STRATEGY_TAGS)


def test_hypothesis_prompt_never_contains_hash():
    # a rand_hash is 64 hex chars; the context has no field for it, so make sure
    # nothing that looks like one is interpolated from files or tacit
    system, user = hypothesis_prompts(ctx())
    import re
    assert not re.search(r"\b[0-9a-f]{64}\b", system + user)


def test_failed_hypotheses_and_forced_tag_appear_when_given():
    # mutation: dropping the recall block loses the "do not repeat" signal
    c = ctx(failed_hypotheses=[{"title": "Bigger tabu tenure", "outcome": "failed:score"}],
            forced_tag="decomposition")
    _, user = hypothesis_prompts(c)
    assert "Bigger tabu tenure" in user and "decomposition" in user


def test_edit_prompt_shows_files_and_format():
    hyp = {"title": "Bitset tabu", "description": "Use a bitset for the tabu list",
           "strategy_tag": "local_search"}
    system, user = edit_prompts(ctx(), hyp)
    # mutation: dropping the description from the user prompt leaves the coder without the idea
    assert "<<<<<<< SEARCH" in system and "mod.rs" in user and "fn x(){}" in user
    assert "Use a bitset for the tabu list" in user


def test_parse_hypothesis_tolerates_prose_and_validates_tag():
    text = 'Sure.\n{"title": "A", "description": "B", "strategy_tag": "local_search"}\nThanks'
    h = parse_hypothesis(text)
    assert h == {"title": "A", "description": "B", "strategy_tag": "local_search"}
    # mutation: accepting an unknown tag breaks strategy_counts bookkeeping
    bad = '{"title": "A", "description": "B", "strategy_tag": "magic"}'
    assert parse_hypothesis(bad)["strategy_tag"] == "hybrid"
    with pytest.raises(ValueError):
        parse_hypothesis("no json here")


def test_distill_roundtrip():
    system, user = distill_prompts(ctx(), [{"title": "Bigger tabu tenure", "outcome": "failed:score"}])
    assert "Bigger tabu tenure" in user  # mutation: dropping the failure list from the prompt
    assert parse_distillation("LESSON: Prefer cheap moves early.") == "Prefer cheap moves early."
    assert parse_distillation("nothing useful") is None
```

- [ ] **Step 3: Run to verify failure** → module not found.

- [ ] **Step 4: Write prompts.py**

```python
"""Prompt builders and parsers. Pure functions over PromptContext; no I/O beyond reading the
packaged Rust rules once."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from importlib import resources

STRATEGY_TAGS = ["construction", "local_search", "metaheuristic", "constraint_relaxation",
                 "decomposition", "hybrid", "data_structure", "parameter_tuning"]

SEARCH_REPLACE_FORMAT = """\
Respond ONLY with one or more edit blocks in exactly this format:

<<<<<<< SEARCH path/to/file.rs
<lines to find, copied exactly>
=======
<replacement lines>
>>>>>>> REPLACE

Rules: the SEARCH text must match the current file exactly once; keep each block small;
emit several blocks for several places; never emit whole-file rewrites; never touch files
other than the algorithm files shown."""


def _rust_rules() -> str:
    return resources.files("talos.data").joinpath("rust_rules.md").read_text()


@dataclass
class PromptContext:
    challenge: str
    template_rs: str
    direction: str
    tacit: str
    files: dict[str, str]
    baseline_name: str
    best_delta: float
    failed_hypotheses: list[dict] = field(default_factory=list)
    forced_tag: str | None = None
    is_gpu: bool = False


def _files_block(files: dict[str, str]) -> str:
    return "\n\n".join(f"--- {name} ---\n{text}" for name, text in sorted(files.items()))


def hypothesis_prompts(ctx: PromptContext) -> tuple[str, str]:
    system = (
        f"You are a research engineer improving a Rust solver for the TIG challenge "
        f"\"{ctx.challenge}\". The goal is to beat the current mainnet state of the art on "
        f"TIG's own benchmark: higher verifier quality per nonce under a fixed fuel budget, "
        f"across every active track.\n\n"
        f"The solver must keep this contract (template.rs):\n```rust\n{ctx.template_rs}\n```\n\n"
        f"Propose ONE specific change. Reply with a JSON object with keys \"title\" "
        f"(short), \"description\" (what to change and why it should raise quality), and "
        f"\"strategy_tag\" (one of: {', '.join(STRATEGY_TAGS)}). No other text."
    )
    parts = [f"Baseline to beat: mainnet algorithm \"{ctx.baseline_name}\". "
             f"Your current best is {ctx.best_delta:+.3%} relative to it on the training nonces.",
             f"Direction from the user:\n{ctx.direction}"]
    if ctx.tacit.strip():
        parts.append(f"Tacit knowledge (lessons so far):\n{ctx.tacit}")
    if ctx.failed_hypotheses:
        lines = "\n".join(f"- {h.get('title', '')} [{h.get('outcome', '')}]"
                          for h in ctx.failed_hypotheses)
        parts.append("Already tried against this exact code and did NOT help; do not repeat:\n"
                     + lines)
    if ctx.forced_tag:
        parts.append(f"You have stagnated. Your strategy_tag MUST be \"{ctx.forced_tag}\" "
                     f"this time; change the approach, not the constants.")
    parts.append("Current algorithm source:\n" + _files_block(ctx.files))
    return system, "\n\n".join(parts)


def edit_prompts(ctx: PromptContext, hypothesis: dict) -> tuple[str, str]:
    system = (
        f"You are editing a Rust solver for the TIG challenge \"{ctx.challenge}\".\n\n"
        f"{SEARCH_REPLACE_FORMAT}\n\n{_rust_rules()}"
    )
    user = (f"Implement this hypothesis:\nTitle: {hypothesis['title']}\n"
            f"Description: {hypothesis['description']}\n\n"
            f"Current algorithm source files:\n{_files_block(ctx.files)}")
    return system, user


def compile_fix_prompts(ctx: PromptContext, files: dict[str, str],
                        compiler_output: str) -> tuple[str, str]:
    system = (f"You are fixing a Rust compile error in a TIG \"{ctx.challenge}\" solver.\n\n"
              f"{SEARCH_REPLACE_FORMAT}\n\n{_rust_rules()}")
    user = (f"The build failed with:\n```\n{compiler_output[-6000:]}\n```\n\n"
            f"Current files:\n{_files_block(files)}\n\nEmit edit blocks that fix the build "
            f"without abandoning the intended change.")
    return system, user


def edit_repair_prompts(ctx: PromptContext, files: dict[str, str],
                        misses_text: str) -> tuple[str, str]:
    system = (f"You are repairing edit blocks for a TIG \"{ctx.challenge}\" solver.\n\n"
              f"{SEARCH_REPLACE_FORMAT}")
    user = (f"These blocks did not match the files exactly once:\n{misses_text}\n\n"
            f"Current files:\n{_files_block(files)}\n\nRe-emit only the failed blocks with "
            f"SEARCH text copied exactly from the files above.")
    return system, user


def distill_prompts(ctx: PromptContext, failed: list[dict]) -> tuple[str, str]:
    system = ("You distill one reusable lesson from failed optimisation attempts. Reply with "
              "exactly one line starting with \"LESSON: \" or the single word NONE.")
    lines = "\n".join(f"- {h.get('title', '')}: {h.get('description', '')} "
                      f"[{h.get('outcome', '')}]" for h in failed)
    user = (f"Challenge: {ctx.challenge}. Direction: {ctx.direction}\n\nFailed attempts:\n"
            f"{lines}\n\nWhat general lesson, independent of these exact constants, should "
            f"guide the next attempts?")
    return system, user


_JSON_RE = re.compile(r"\{.*?\}", re.S)


def parse_hypothesis(text: str) -> dict:
    for m in _JSON_RE.finditer(text):
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if "title" in d and "description" in d:
            tag = d.get("strategy_tag")
            return {"title": str(d["title"]), "description": str(d["description"]),
                    "strategy_tag": tag if tag in STRATEGY_TAGS else "hybrid"}
    raise ValueError("no hypothesis JSON object found in response")


def parse_distillation(text: str) -> str | None:
    for line in text.splitlines():
        if line.strip().startswith("LESSON:"):
            lesson = line.split("LESSON:", 1)[1].strip()
            return lesson or None
    return None
```

- [ ] **Step 5: Run, commit**

Run: `make check PYTHON=.venv/bin/python` → pass.

```bash
git add talos/prompts.py talos/data/rust_rules.md tests/test_prompts.py
git commit -m "prompts: hypothesis, edit, fix, repair, distill builders and parsers"
```

---

### Task 10: Baseline resolver and cache

**Files:**
- Create: `talos/baseline.py`, `tests/test_baseline.py`

**Interfaces:**
- Consumes: `mainnet.top_algorithm`, `mainnet.fetch_algorithm_files`, `mainnet.fetch_template` (Task 2); `Bench` (Task 7); `BaselineRecord` (Task 6); `NonceSet` (Task 1).
- Produces: `resolve_baseline(challenge, training, holdout, fuel, bench, cache_dir: Path, hardware_class: str, mainnet=talos.mainnet) -> tuple[BaselineRecord, str]` returning the record and the template source; `BaselineError`; `cache_key(challenge, monorepo_ref, name, training, holdout, fuel, hardware_class) -> str`.

- [ ] **Step 1: Failing tests**

`tests/test_baseline.py`:
```python
import types

import pytest

from talos.baseline import BaselineError, cache_key, resolve_baseline
from talos.bench import FakeBench
from talos.types import NonceSet

TR = [NonceSet("t", "ab" * 32, 0, 2)]
HO = [NonceSet("t", "ab" * 32, 1_000_000, 2)]


def fake_mainnet(top=("algo_x", 55)):
    return types.SimpleNamespace(
        top_algorithm=lambda ch, **kw: top,
        fetch_algorithm_files=lambda ch, name, **kw: {"mod.rs": f"// {name}"},
        fetch_template=lambda ch, **kw: "pub fn solve_challenge(",
    )


def test_resolve_compiles_and_scores_both_sets(tmp_path):
    fb = FakeBench(lambda ch, files, ns: [10 + n for n in ns.nonces()])
    rec, template = resolve_baseline("knapsack", TR, HO, 5, fb, tmp_path, "cpu4",
                                     mainnet=fake_mainnet())
    assert rec.name == "algo_x" and rec.adoption == 55 and rec.artifact_id
    assert [r.quality for r in rec.training] == [10, 11]
    assert [r.nonce for r in rec.holdout] == [1_000_000, 1_000_001]
    assert "solve_challenge" in template
    assert fb.compile_calls == 1 and fb.score_calls == 2


def test_cache_hit_skips_bench(tmp_path):
    # mutation: ignoring the cache re-measures and charges the user twice
    fb = FakeBench(lambda ch, files, ns: [1 for _ in ns.nonces()])
    resolve_baseline("knapsack", TR, HO, 5, fb, tmp_path, "cpu4", mainnet=fake_mainnet())
    resolve_baseline("knapsack", TR, HO, 5, fb, tmp_path, "cpu4", mainnet=fake_mainnet())
    assert fb.compile_calls == 1 and fb.score_calls == 2


def test_cache_key_changes_with_fuel_and_hardware():
    a = cache_key("knapsack", "ref", "algo", TR, HO, 5, "cpu4")
    assert a != cache_key("knapsack", "ref", "algo", TR, HO, 6, "cpu4")
    assert a != cache_key("knapsack", "ref", "algo", TR, HO, 5, "l40s")


def test_no_top_algorithm_is_an_error(tmp_path):
    fb = FakeBench(lambda ch, files, ns: [1])
    with pytest.raises(BaselineError):
        resolve_baseline("knapsack", TR, HO, 5, fb, tmp_path, "cpu4", mainnet=fake_mainnet(top=None))


def test_baseline_compile_failure_is_an_error_with_output(tmp_path):
    fb = FakeBench(lambda ch, files, ns: [1], compile_ok=lambda f: False)
    with pytest.raises(BaselineError) as ei:
        resolve_baseline("knapsack", TR, HO, 5, fb, tmp_path, "cpu4", mainnet=fake_mainnet())
    assert "E0308" in str(ei.value)
```

- [ ] **Step 2: Run to verify failure** → module not found.

- [ ] **Step 3: Write baseline.py**

```python
"""Resolve the mainnet top algorithm, compile and score it once, cache the result."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from talos import mainnet as _mainnet
from talos.challenges import MONOREPO_REF
from talos.state import BaselineRecord
from talos.types import NonceResult, NonceSet


class BaselineError(RuntimeError):
    pass


def cache_key(challenge: str, monorepo_ref: str, name: str, training: list[NonceSet],
              holdout: list[NonceSet], fuel: int, hardware_class: str) -> str:
    h = hashlib.sha256()
    payload = {"challenge": challenge, "ref": monorepo_ref, "name": name, "fuel": fuel,
               "hw": hardware_class,
               "training": [(n.track, n.rand_hash, n.start, n.count) for n in training],
               "holdout": [(n.track, n.rand_hash, n.start, n.count) for n in holdout]}
    h.update(json.dumps(payload, sort_keys=True).encode())
    return h.hexdigest()[:24]


def resolve_baseline(challenge: str, training: list[NonceSet], holdout: list[NonceSet],
                     fuel: int, bench, cache_dir: Path, hardware_class: str,
                     mainnet=_mainnet, log=lambda msg: None) -> tuple[BaselineRecord, str]:
    top = mainnet.top_algorithm(challenge)
    if top is None:
        raise BaselineError(f"no adopted, compiled algorithm found on mainnet for {challenge}")
    name, adoption = top
    template = mainnet.fetch_template(challenge)
    key = cache_key(challenge, MONOREPO_REF, name, training, holdout, fuel, hardware_class)
    cache_file = Path(cache_dir) / challenge / f"{key}.json"
    if cache_file.exists():
        log(f"baseline {name}: cache hit {key}")
        return BaselineRecord.from_dict(json.loads(cache_file.read_text())), template
    files = mainnet.fetch_algorithm_files(challenge, name)
    log(f"baseline {name} (adoption {adoption}): compiling {len(files)} file(s)")
    c = bench.compile(challenge, files)
    if not c.ok:
        raise BaselineError(f"baseline {name} failed to compile; likely dev-image drift at "
                            f"{MONOREPO_REF}.\n{c.output[-4000:]}")
    log("baseline: scoring training nonces")
    tr: list[NonceResult] = bench.score(challenge, c.artifact_id, training, fuel)
    log("baseline: scoring held-out nonces")
    ho: list[NonceResult] = bench.score(challenge, c.artifact_id, holdout, fuel)
    rec = BaselineRecord(name=name, adoption=adoption, artifact_id=c.artifact_id, files=files,
                         training=tr, holdout=ho)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(rec.to_dict(), indent=1))
    return rec, template
```

Note: the cache stores `training` and `holdout` results keyed by a hash that includes each set's `rand_hash`, so the cache file itself contains no secret beyond what `job.json` already holds for that job. The cache dir defaults to `~/.talos/baselines/` in the CLI (Task 13).

- [ ] **Step 4: Run, commit**

Run: `make check PYTHON=.venv/bin/python` → pass.

```bash
git add talos/baseline.py tests/test_baseline.py
git commit -m "baseline: resolve mainnet top algorithm, measure once, cache"
```

---

### Task 11: The research loop

**Files:**
- Create: `talos/loop.py`, `tests/test_loop.py`

**Interfaces:**
- Consumes: `JobSpec`, `JobState`, `JobStore`, `Candidate`, `BaselineRecord`, `TERMINAL` (Task 6); `Bench` (Task 7); `Provider`, `ProviderAuthError`, `ProviderRateLimited`, `ProviderError` (Task 8); prompts and parsers (Task 9); `apply_edit_response`, `EditError` (Task 4); `bundle_delta`, `beats` (Task 3); `Budget`, `Spend`, `exhausted`, `BudgetExhausted` (Task 5); `CHALLENGES` (Task 2); `resolve_baseline`, `BaselineError` (Task 10); `BenchUnavailable` (Task 7).
- Produces: `Loop(spec, state, store, provider, bench, template_rs, clock=time.time, sleep=time.sleep, on_event=None, thresholds=Thresholds())` with `run() -> JobState`, `iterate() -> None`, `measure_baseline(cache_dir, hardware_class, mainnet=None) -> None`; `Thresholds(recall=2, distill=3, reset=5, compile_fix_rounds=3, edit_repair_rounds=1)`; `Interrupted(Exception)`; `Loop.request_stop()`.
- Produces: an `EditStrategy` hook so Task 14 can plug agentic mode in: `Loop.propose_and_edit: Callable[[PromptContext], tuple[dict, dict[str, str]]]` returning `(hypothesis, new_files)`; default is `Loop.single_shot_propose_and_edit`.

- [ ] **Step 1: Failing tests**

`tests/test_loop.py`:
```python
from talos.budget import Budget, Spend
from talos.bench import FakeBench
from talos.loop import Loop, Thresholds
from talos.providers.fake import FakeProvider
from talos.state import BaselineRecord, JobSpec, JobState, JobStore
from talos.types import NonceResult, NonceSet

HASH = "ab" * 32
TR = [NonceSet("t", HASH, 0, 4)]
HO = [NonceSet("t", HASH, 1_000_000, 4)]
BASE_FILES = {"mod.rs": "fn solve() { let k = 1; }\n"}


def spec(budget=None):
    return JobSpec(job_id="j", challenge="knapsack", direction="go", provider="fake", model="m",
                   mode="single-shot",
                   budget=budget or Budget(usd=None, hours=None, iterations=20, modal_usd=None),
                   rand_hash=HASH, tracks=["t"], training=TR, holdout=HO, fuel=1, created_at=0.0,
                   monorepo_ref="r", challenge_id="c003")


def baseline(q=100):
    tr = [NonceResult("t", n, True, q, 1) for n in range(4)]
    ho = [NonceResult("t", 1_000_000 + n, True, q, 1) for n in range(4)]
    return BaselineRecord(name="base", adoption=1, artifact_id="base-art", files=BASE_FILES,
                          training=tr, holdout=ho)


def hyp(title, tag="local_search"):
    return f'{{"title": "{title}", "description": "d", "strategy_tag": "{tag}"}}'


def edit(k, frm=1):
    """An edit block turning `let k = {frm};` into `let k = {k};`. The SEARCH text must match the
    files the loop is currently editing (the best so far), not always the baseline."""
    return f"<<<<<<< SEARCH mod.rs\nlet k = {frm};\n=======\nlet k = {k};\n>>>>>>> REPLACE\n"


def quality_from_files(challenge, files, ns):
    # quality = 100 + k on every nonce; k parsed from the code. "let k = 7" -> 107
    import re
    k = int(re.search(r"let k = (\d+);", files["mod.rs"]).group(1))
    return [100 + k - 1 for _ in ns.nonces()]  # k=1 -> 100 (baseline parity)


def make(tmp_path, script, scores=quality_from_files, budget=None, thresholds=None):
    store = JobStore(tmp_path)
    sp = spec(budget)
    store.write_spec(sp)
    st = JobState.fresh(Spend(started_at=0.0))
    st.baseline = baseline()
    st.status = "researching"
    st.best = None
    fb = FakeBench(scores)
    fb.compile("knapsack", BASE_FILES)  # register the baseline files under an artifact id
    fp = FakeProvider(script)
    loop = Loop(sp, st, store, fp, fb, template_rs="pub fn solve_challenge(",
                clock=lambda: 0.0, sleep=lambda s: None, thresholds=thresholds or Thresholds())
    return loop, fp, fb, store


def test_win_requires_training_then_holdout_beat(tmp_path):
    # mutation: skipping the held-out confirmation declares a win after training alone
    loop, fp, fb, store = make(tmp_path, [hyp("bump k"), edit(5)])
    st = loop.run()
    assert st.status == "won" and st.best.iteration == 1
    assert st.best.holdout is not None and st.confirmed == [1]
    assert fb.score_calls == 1 + 1  # training, then held-out


def test_false_positive_returns_to_research(tmp_path):
    # training says win, held-out says no -> keep researching, record false positive
    def scores(challenge, files, ns):
        import re
        k = int(re.search(r"let k = (\d+);", files["mod.rs"]).group(1))
        if ns.start >= 1_000_000:
            return [100 for _ in ns.nonces()]  # held-out never improves
        return [100 + k - 1 for _ in ns.nonces()]
    b = Budget(usd=None, hours=None, iterations=2, modal_usd=None)
    # iteration 2 edits the iteration-1 best (k=5), so its SEARCH text is `let k = 5;`
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(5), hyp("b"), edit(6, frm=5)], scores, b)
    st = loop.run()
    assert st.status == "exhausted" and st.stop_reason == "iterations"
    assert st.false_positives == [1, 2] and st.best.iteration == 2


def test_compile_fix_rounds_then_skip(tmp_path):
    # mutation: unlimited fix rounds never terminates on a stubborn error
    bad = "<<<<<<< SEARCH mod.rs\nlet k = 1;\n=======\nlet k = BUG;\n>>>>>>> REPLACE\n"
    script = [hyp("a"), bad, bad, bad, bad, hyp("b"), edit(5)]

    def compile_ok(files):
        return "BUG" not in files["mod.rs"]

    loop, fp, fb, store = make(tmp_path, script)
    fb._compile_ok = compile_ok
    st = loop.run()
    assert st.status == "won"
    assert st.hypotheses[0]["outcome"] == "failed:compile"
    assert fb.compile_calls == 1 + 4 + 1  # baseline reg + 1 edit + 3 fixes + winning edit


def test_budget_stops_before_llm_call(tmp_path):
    b = Budget(usd=0.015, hours=None, iterations=None, modal_usd=None)  # one fake call = 0.01
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(2), hyp("b"), edit(3)], budget=b)
    st = loop.run()
    assert st.status == "exhausted" and st.stop_reason == "usd"
    assert len(fp.calls) == 2  # hypothesis + edit, then the compile's budget check refuses
    assert fb.compile_calls == 1  # only the baseline registration in make(); no candidate compile


def test_over_error_ceiling_is_failed_runtime(tmp_path):
    # spec §9: a candidate over the error ceiling is logged failed:runtime and never becomes best
    # mutation: dropping the ceiling check lets a half-crashing candidate become the best
    def scores(challenge, files, ns):
        import re
        k = int(re.search(r"let k = (\d+);", files["mod.rs"]).group(1))
        if k == 1:
            return [100 for _ in ns.nonces()]
        return [None if n % 2 else 100 + k for n in ns.nonces()]  # half the nonces error out

    b = Budget(usd=None, hours=None, iterations=1, modal_usd=None)
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(9)], scores, b)
    st = loop.run()
    assert st.hypotheses[0]["outcome"] == "failed:runtime" and st.best is None
    assert st.status == "exhausted"


def test_stagnation_recall_distill_reset(tmp_path):
    # non-improving edits (k stays 1) drive stagnation through recall -> distill -> reset
    script = []
    for i in range(6):
        script += [hyp(f"h{i}"), edit(1)]
    script.insert(6, "LESSON: constants are not the answer.")  # distill call after 3rd failure
    b = Budget(usd=None, hours=None, iterations=6, modal_usd=None)
    loop, fp, fb, store = make(tmp_path, script, budget=b,
                               thresholds=Thresholds(recall=2, distill=3, reset=5))
    st = loop.run()
    assert "constants are not the answer" in st.tacit
    users = [u for _, u in fp.calls]
    assert any("do not repeat" in u for u in users)          # recall block appeared
    assert any("strategy_tag MUST be" in u for u in users)   # reset forced a tag


def test_resume_discards_incomplete_iteration(tmp_path):
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(5)])
    loop.state.iteration = 3
    store.iteration_dir(4).joinpath("hypothesis.json").write_text("{}")  # started, never finished
    store.save(loop.state)
    loop2 = Loop(store.read_spec(), store.load(), store, fp, fb, template_rs="x",
                 clock=lambda: 0.0, sleep=lambda s: None)
    st = loop2.run()
    assert st.status == "won" and st.best.iteration == 4
    assert not store.iteration_dir(4).joinpath("hypothesis.json").read_text() == "{}"


def test_request_stop_cancels_at_safe_point(tmp_path):
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(1), hyp("b"), edit(5)])
    loop.request_stop()
    st = loop.run()
    assert st.status == "cancelled" and len(fp.calls) == 0
```

- [ ] **Step 2: Run to verify failure** → module not found.

- [ ] **Step 3: Write loop.py**

```python
"""The research loop. One Loop per job; all state lives in JobState and is saved after every
step so the run can be resumed. No network code here beyond calling the provider and bench."""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from typing import Callable

from talos.baseline import resolve_baseline
from talos.bench import BenchUnavailable
from talos.budget import BudgetExhausted, exhausted
from talos.challenges import CHALLENGES
from talos.edits import EditError, apply_edit_response
from talos.prompts import (PromptContext, STRATEGY_TAGS, compile_fix_prompts, distill_prompts,
                           edit_prompts, edit_repair_prompts, hypothesis_prompts,
                           parse_distillation, parse_hypothesis)
from talos.providers import ProviderAuthError, ProviderError, ProviderRateLimited
from talos.scoring import ScoringError, beats, bundle_delta
from talos.search_replace import format_misses
from talos.state import Candidate, JobSpec, JobState, JobStore, TERMINAL
from talos.types import Completion


@dataclass
class Thresholds:
    recall: int = 2
    distill: int = 3
    reset: int = 5
    compile_fix_rounds: int = 3
    edit_repair_rounds: int = 1
    rate_limit_wait_s: int = 30
    rate_limit_max_waits: int = 20


class Interrupted(Exception):
    pass


class Loop:
    def __init__(self, spec: JobSpec, state: JobState, store: JobStore, provider, bench,
                 template_rs: str, clock=time.time, sleep=time.sleep,
                 on_event: Callable[[str, dict], None] | None = None,
                 thresholds: Thresholds = Thresholds()):
        self.spec, self.state, self.store = spec, state, store
        self.provider, self.bench = provider, bench
        self.template_rs = template_rs
        self.clock, self.sleep = clock, sleep
        self.on_event = on_event or (lambda kind, data: None)
        self.t = thresholds
        self.rule = CHALLENGES[spec.challenge].beat
        self._stop = False
        self.propose_and_edit = self.single_shot_propose_and_edit

    # ── plumbing ──────────────────────────────────────────────────────

    def request_stop(self) -> None:
        self._stop = True

    def _event(self, kind: str, **data) -> None:
        self.store.event(kind, iteration=self.state.iteration, **data)
        self.on_event(kind, data)

    def _save(self) -> None:
        self.store.save(self.state)

    def _check_budget(self) -> None:
        dim = exhausted(self.spec.budget, self.state.spend, self.clock())
        if dim:
            raise BudgetExhausted(dim)
        if self._stop:
            raise Interrupted()

    def _llm(self, system: str, user: str) -> Completion:
        self._check_budget()
        waits = 0
        while True:
            try:
                c = self.provider.complete(system, user)
                break
            except ProviderRateLimited as e:
                waits += 1
                if waits > self.t.rate_limit_max_waits:
                    raise ProviderError(f"gave up after {waits} rate-limit waits: {e}")
                self._event("rate_limited", wait_s=self.t.rate_limit_wait_s)
                self.sleep(self.t.rate_limit_wait_s)
        if c.usage.cost_usd:
            self.state.spend.llm_usd += c.usage.cost_usd
        self._save()
        return c

    def _bench_compile(self, files):
        self._check_budget()
        mark = self.bench.cost_mark()
        r = self.bench.compile(self.spec.challenge, files)
        self.state.spend.modal_usd += self.bench.cost_usd_since(mark)
        self._save()
        return r

    def _bench_score(self, artifact_id, nonce_sets):
        self._check_budget()
        mark = self.bench.cost_mark()
        r = self.bench.score(self.spec.challenge, artifact_id, nonce_sets, self.spec.fuel)
        self.state.spend.modal_usd += self.bench.cost_usd_since(mark)
        self._save()
        return r

    # ── baseline ──────────────────────────────────────────────────────

    def measure_baseline(self, cache_dir, hardware_class: str, mainnet=None) -> None:
        self.state.status = "measuring_baseline"
        self._save()
        kw = {"mainnet": mainnet} if mainnet else {}
        mark = self.bench.cost_mark()
        rec, template = resolve_baseline(self.spec.challenge, self.spec.training,
                                         self.spec.holdout, self.spec.fuel, self.bench,
                                         cache_dir, hardware_class,
                                         log=lambda m: self._event("baseline", message=m), **kw)
        self.state.spend.modal_usd += self.bench.cost_usd_since(mark)
        self.state.baseline = rec
        self.template_rs = template
        self.state.status = "researching"
        self._save()
        self._event("baseline_ready", name=rec.name, adoption=rec.adoption)

    # ── context ───────────────────────────────────────────────────────

    def _current_files(self) -> dict[str, str]:
        return dict(self.state.best.files if self.state.best else self.state.baseline.files)

    def _current_training(self):
        return self.state.best.training if self.state.best else self.state.baseline.training

    def _best_delta(self) -> float:
        return self.state.best.delta["mean_rel_delta"] if self.state.best else 0.0

    def _failed_for_current(self) -> list[dict]:
        anchor = self.state.best.iteration if self.state.best else 0
        return [h for h in self.state.hypotheses
                if h.get("against") == anchor and h.get("outcome", "").startswith("failed")]

    def _forced_tag(self) -> str | None:
        if self.state.runs_since_improvement < self.t.reset:
            return None
        counts = self.state.strategy_counts
        return min(STRATEGY_TAGS, key=lambda tag: counts.get(tag, 0))

    def _context(self) -> PromptContext:
        failed = self._failed_for_current() if self.state.runs_since_improvement >= self.t.recall else []
        return PromptContext(challenge=self.spec.challenge, template_rs=self.template_rs,
                             direction=self.spec.direction, tacit=self.state.tacit,
                             files=self._current_files(),
                             baseline_name=self.state.baseline.name,
                             best_delta=self._best_delta(), failed_hypotheses=failed,
                             forced_tag=self._forced_tag(),
                             is_gpu=CHALLENGES[self.spec.challenge].is_gpu)

    # ── single-shot propose + edit ────────────────────────────────────

    def single_shot_propose_and_edit(self, ctx: PromptContext) -> tuple[dict, dict[str, str]]:
        system, user = hypothesis_prompts(ctx)
        hypothesis = parse_hypothesis(self._llm(system, user).text)
        self._event("hypothesis", **hypothesis)
        system, user = edit_prompts(ctx, hypothesis)
        text = self._llm(system, user).text
        outcome = apply_edit_response(ctx.files, text)
        for _ in range(self.t.edit_repair_rounds):
            if not outcome.misses:
                break
            system, user = edit_repair_prompts(ctx, outcome.files, format_misses(outcome.misses))
            repaired = apply_edit_response(outcome.files, self._llm(system, user).text)
            outcome = repaired
        if outcome.applied == 0:
            raise EditError("no edit block applied")
        return hypothesis, outcome.files

    # ── one iteration ─────────────────────────────────────────────────

    def iterate(self) -> None:
        n = self.state.iteration + 1
        it_dir = self.store.iteration_dir(n)
        ctx = self._context()
        anchor = self.state.best.iteration if self.state.best else 0
        record = {"iteration": n, "against": anchor, "title": "", "description": "",
                  "strategy_tag": "", "outcome": "started"}
        try:
            hypothesis, files = self.propose_and_edit(ctx)
        except (EditError, ValueError) as e:
            record.update(outcome="failed:edit", error=str(e))
            self._finish_iteration(n, record, improved=False)
            return
        record.update(hypothesis)
        (it_dir / "hypothesis.json").write_text(json.dumps(hypothesis))
        self.state.strategy_counts[hypothesis["strategy_tag"]] = (
            self.state.strategy_counts.get(hypothesis["strategy_tag"], 0) + 1)

        # compile with bounded fix rounds
        comp = self._bench_compile(files)
        for _ in range(self.t.compile_fix_rounds):
            if comp.ok:
                break
            self._event("compile_failed", output=comp.output[-2000:])
            system, user = compile_fix_prompts(ctx, files, comp.output)
            try:
                files = apply_edit_response(files, self._llm(system, user).text).files
            except EditError:
                break
            comp = self._bench_compile(files)
        if not comp.ok:
            record.update(outcome="failed:compile")
            self._finish_iteration(n, record, improved=False)
            return
        for name, text in files.items():
            (it_dir / name).parent.mkdir(parents=True, exist_ok=True)
            (it_dir / name).write_text(text)

        # score on training
        results = self._bench_score(comp.artifact_id, self.spec.training)
        try:
            delta = bundle_delta(self.state.baseline.training, results)
        except ScoringError as e:
            record.update(outcome="failed:score", error=str(e))
            self._finish_iteration(n, record, improved=False)
            return
        cand = Candidate(iteration=n, files=files, artifact_id=comp.artifact_id,
                         training=results, delta=delta.to_dict(), hypothesis=hypothesis)
        self._event("scored", mean_rel_delta=delta.mean_rel_delta,
                    worst_rel_delta=delta.worst_rel_delta, error_rate=delta.error_rate)
        if delta.error_rate > self.rule.error_ceiling:  # spec §9: over the ceiling is a runtime failure
            record.update(outcome="failed:runtime", error_rate=delta.error_rate)
            self._finish_iteration(n, record, improved=False)
            return
        improved = delta.mean_rel_delta > self._best_delta() or self.state.best is None and delta.mean_rel_delta > 0
        if improved:
            self.state.best = cand
            record["outcome"] = "improved"
        else:
            record["outcome"] = "failed:score"

        if beats(self.state.baseline.training, results, self.rule):
            self._confirm(cand)
        self._finish_iteration(n, record, improved=improved)

    def _confirm(self, cand: Candidate) -> None:
        self.state.status = "confirming"
        self._save()
        self._event("confirming")
        ho = self._bench_score(cand.artifact_id, self.spec.holdout)
        cand.holdout = ho
        if beats(self.state.baseline.holdout, ho, self.rule):
            self.state.confirmed.append(cand.iteration)
            self.state.status = "won"
            self.state.stop_reason = "beat baseline on training and held-out nonces"
            self._event("won", holdout=bundle_delta(self.state.baseline.holdout, ho).to_dict())
        else:
            self.state.false_positives.append(cand.iteration)
            self.state.status = "researching"
            self._event("false_positive", holdout=bundle_delta(self.state.baseline.holdout, ho).to_dict())
        self._save()

    def _finish_iteration(self, n: int, record: dict, improved: bool) -> None:
        self.state.iteration = n
        self.state.spend.iterations += 1
        self.state.hypotheses.append(record)
        if improved:
            self.state.runs_since_improvement = 0
        else:
            self.state.runs_since_improvement += 1
        self._save()
        self._event("iteration_done", outcome=record["outcome"],
                    runs_since_improvement=self.state.runs_since_improvement)
        if not improved and self.state.runs_since_improvement == self.t.distill:
            self._distill()
        if not improved and self.state.runs_since_improvement >= self.t.reset:
            self._event("reset", forced_tag=self._forced_tag())

    def _distill(self) -> None:
        failed = self._failed_for_current()
        if not failed:
            return
        system, user = distill_prompts(self._context(), failed)
        try:
            lesson = parse_distillation(self._llm(system, user).text)
        except ProviderError:
            return
        if lesson:
            self.state.tacit = (self.state.tacit.rstrip() + f"\n- LLM: {lesson}\n").lstrip()
            (self.store.run_dir / "tacit.md").write_text(self.state.tacit)
            self._save()
            self._event("distilled", lesson=lesson)

    # ── driver ────────────────────────────────────────────────────────

    def _discard_incomplete_iteration(self) -> None:
        n = self.state.iteration + 1
        d = self.store.run_dir / "iterations" / f"{n:04d}"
        if d.exists():
            shutil.rmtree(d)
            self._event("discarded_incomplete", discarded=n)

    def run(self) -> JobState:
        self._discard_incomplete_iteration()
        if self.state.status in TERMINAL:
            return self.state
        self.state.status = "researching"
        self._save()
        try:
            while self.state.status not in TERMINAL:
                self._check_budget()
                self.iterate()
        except BudgetExhausted as e:
            self.state.status, self.state.stop_reason = "exhausted", e.dimension
        except Interrupted:
            self.state.status, self.state.stop_reason = "cancelled", "user requested stop"
        except ProviderAuthError as e:
            self.state.status, self.state.stop_reason = "failed", f"provider auth: {e}"
        except ProviderError as e:
            self.state.status, self.state.stop_reason = "failed", f"provider: {e}"
        except BenchUnavailable as e:
            self.state.status, self.state.stop_reason = "paused", f"bench: {e}"
        self._save()
        self._event("stopped", status=self.state.status, reason=self.state.stop_reason)
        return self.state
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_loop.py -v`
Expected: all pass. Two things to watch: `test_compile_fix_rounds_then_skip` counts compile calls exactly, so the loop must not compile a fourth time after the third fix; `test_stagnation_recall_distill_reset` expects the distill LLM call to consume the script entry at index 6, which is why it is inserted there (after hypothesis+edit pairs for iterations 1 to 3). If the distill fires one iteration earlier or later, the failure is in `_finish_iteration`'s equality check, not the test.

- [ ] **Step 5: ruff, commit**

Run: `make check PYTHON=.venv/bin/python` → pass.

```bash
git add talos/loop.py tests/test_loop.py
git commit -m "loop: iterate, compile-fix, confirm on held-out, stagnation, budget, resume"
```

---

### Task 12: Hand-back package

**Files:**
- Create: `talos/package.py`, `talos/data/evidence_template.md`, `tests/test_package.py`

**Interfaces:**
- Consumes: `JobSpec`, `JobState`, `JobStore` (Task 6); `bundle_delta` (Task 3).
- Produces: `build_package(spec, state, store) -> Path` returning `runs/<job>/package/`, also writing `package.zip` beside it; `scores_markdown(baseline_results, candidate_results) -> str`; `evidence_draft(spec, state) -> str`.

- [ ] **Step 1: Fetch the evidence template**

```bash
curl -fsSL https://raw.githubusercontent.com/tig-foundation/tig-monorepo/main/tig-algorithms/advances/evidence_template.md -o talos/data/evidence_template.md
head -3 talos/data/evidence_template.md
```
Expected: first lines read `## UNIQUE ALGORITHM IDENTIFIER (UAI)`.

- [ ] **Step 2: Failing tests**

`tests/test_package.py`:
```python
import zipfile

from talos.budget import Budget, Spend
from talos.package import build_package, evidence_draft, scores_markdown
from talos.state import BaselineRecord, Candidate, JobSpec, JobState, JobStore
from talos.types import NonceResult, NonceSet

HASH = "cd" * 32


def make(tmp_path, status="won"):
    spec = JobSpec(job_id="j", challenge="knapsack", direction="go", provider="fake", model="m",
                   mode="single-shot", budget=Budget(usd=1.0, hours=None, iterations=None, modal_usd=None),
                   rand_hash=HASH, tracks=["t"], training=[NonceSet("t", HASH, 0, 2)],
                   holdout=[NonceSet("t", HASH, 1_000_000, 2)], fuel=1, created_at=0.0,
                   monorepo_ref="r", challenge_id="c003")
    base = BaselineRecord("base", 1, "a", {"mod.rs": "fn a() {}\nlet k = 1;\n"},
                          [NonceResult("t", 0, True, 100, 1), NonceResult("t", 1, True, 100, 1)],
                          [NonceResult("t", 1_000_000, True, 100, 1), NonceResult("t", 1_000_001, True, 100, 1)])
    cand = Candidate(3, {"mod.rs": "fn a() {}\nlet k = 9;\n"}, "b",
                     [NonceResult("t", 0, True, 110, 1), NonceResult("t", 1, True, 111, 1)],
                     {"mean_rel_delta": 0.105}, {"title": "Bump k", "description": "d", "strategy_tag": "hybrid"},
                     holdout=[NonceResult("t", 1_000_000, True, 108, 1), NonceResult("t", 1_000_001, True, 109, 1)])
    st = JobState.fresh(Spend(started_at=0.0))
    st.status, st.best, st.baseline = status, cand, base
    st.hypotheses = [{"iteration": 3, "against": 0, "title": "Bump k", "description": "d",
                      "strategy_tag": "hybrid", "outcome": "improved"}]
    store = JobStore(tmp_path)
    store.write_spec(spec)
    store.save(st)
    return spec, st, store


def test_package_contents_and_no_hash(tmp_path):
    spec, st, store = make(tmp_path)
    pkg = build_package(spec, st, store)
    names = {p.name for p in pkg.iterdir()}
    assert {"mod.rs", "diff_vs_baseline.patch", "scores.md", "hypotheses.md",
            "evidence_draft.md", "README.md"} <= names
    assert "let k = 9" in (pkg / "diff_vs_baseline.patch").read_text()
    # mutation: dumping job.json into the package leaks rand_hash
    for p in pkg.rglob("*"):
        if p.is_file():
            assert HASH not in p.read_text(errors="ignore")
    z = zipfile.ZipFile(tmp_path / "package.zip")
    assert "mod.rs" in {n.rsplit("/", 1)[-1] for n in z.namelist()}


def test_scores_markdown_has_per_nonce_rows_and_delta():
    base = [NonceResult("t", 0, True, 100, 1), NonceResult("t", 1, False, None, 1, "panic")]
    cand = [NonceResult("t", 0, True, 120, 1), NonceResult("t", 1, True, 130, 1)]
    md = scores_markdown(base, cand)
    assert "| t | 0 | 100 | 120 |" in md and "panic" in md and "mean_rel_delta" in md


def test_evidence_draft_prefills_challenge_and_method(tmp_path):
    spec, st, store = make(tmp_path)
    text = evidence_draft(spec, st)
    assert "knapsack" in text and "Bump k" in text and "UNIQUE ALGORITHM IDENTIFIER" in text


def test_package_for_exhausted_job_still_builds(tmp_path):
    spec, st, store = make(tmp_path, status="exhausted")
    pkg = build_package(spec, st, store)
    assert "not confirmed" in (pkg / "README.md").read_text()
```

- [ ] **Step 3: Run to verify failure** → module not found.

- [ ] **Step 4: Write package.py**

```python
"""Build the hand-back package: files, diff, per-nonce tables, hypothesis log, evidence draft."""
from __future__ import annotations

import difflib
import shutil
from importlib import resources
from pathlib import Path

from talos.scoring import ScoringError, bundle_delta
from talos.state import JobSpec, JobState, JobStore
from talos.types import NonceResult


def scores_markdown(baseline: list[NonceResult], candidate: list[NonceResult]) -> str:
    rows = ["| track | nonce | baseline | candidate | note |", "|---|---|---|---|---|"]
    by_b = {(r.track, r.nonce): r for r in baseline}
    for c in sorted(candidate, key=lambda r: (r.track, r.nonce)):
        b = by_b.get((c.track, c.nonce))
        bq = b.quality if b and b.ok else (b.error if b else "missing")
        cq = c.quality if c.ok else c.error
        note = ""
        if b and b.ok and c.ok:
            note = "+" if c.quality > b.quality else ("=" if c.quality == b.quality else "-")
        rows.append(f"| {c.track} | {c.nonce} | {bq} | {cq} | {note} |")
    try:
        d = bundle_delta(baseline, candidate)
        summary = (f"\nmean_rel_delta: {d.mean_rel_delta:+.4%}  worst_rel_delta: "
                   f"{d.worst_rel_delta:+.4%}  error_rate: {d.error_rate:.2%}\n")
    except ScoringError as e:
        summary = f"\n(no bundle delta: {e})\n"
    return "\n".join(rows) + "\n" + summary


def _diff(base: dict[str, str], cand: dict[str, str]) -> str:
    out = []
    for name in sorted(set(base) | set(cand)):
        a = base.get(name, "").splitlines(keepends=True)
        b = cand.get(name, "").splitlines(keepends=True)
        out.extend(difflib.unified_diff(a, b, fromfile=f"baseline/{name}", tofile=f"candidate/{name}"))
    return "".join(out)


def evidence_draft(spec: JobSpec, state: JobState) -> str:
    template = resources.files("talos.data").joinpath("evidence_template.md").read_text()
    winners = [h for h in state.hypotheses if h.get("outcome") == "improved"]
    method = "\n".join(f"- {h['title']}: {h['description']}" for h in winners) or "(none recorded)"
    filled = template.replace(
        "PLEASE IDENTIFY WHICH TIG CHALLENGE THE METHOD ADDRESSES.\n\n> YOUR RESPONSE HERE",
        f"PLEASE IDENTIFY WHICH TIG CHALLENGE THE METHOD ADDRESSES.\n\n> {spec.challenge} "
        f"({spec.challenge_id})", 1)
    filled = filled.replace(
        "PLEASE DESCRIBE THE METHOD THAT YOU HAVE SELECTED FOR ASSESSMENT.\n\n> YOUR RESPONSE HERE",
        f"PLEASE DESCRIBE THE METHOD THAT YOU HAVE SELECTED FOR ASSESSMENT.\n\n> Draft from the "
        f"Talos hypothesis log; rewrite as a discrete method:\n{method}", 1)
    bench = ""
    if state.best and state.baseline:
        bench = ("\n\n## TALOS BENCHMARK APPENDIX (auto-generated)\n\n"
                 f"Baseline: mainnet `{state.baseline.name}` at monorepo `{spec.monorepo_ref}`, "
                 f"fuel {spec.fuel}, tracks {', '.join(spec.tracks)}.\n\n### Training nonces\n\n"
                 + scores_markdown(state.baseline.training, state.best.training))
        if state.best.holdout:
            bench += "\n### Held-out nonces\n\n" + scores_markdown(state.baseline.holdout, state.best.holdout)
    return filled + bench


def _readme(spec: JobSpec, state: JobState) -> str:
    confirmed = state.best is not None and state.best.iteration in state.confirmed
    head = ("# Talos hand-back\n\n"
            f"Challenge: {spec.challenge}. Baseline: mainnet `{state.baseline.name}`. "
            f"Status: {state.status}. Reason: {state.stop_reason}\n\n")
    if confirmed:
        head += ("This candidate beat the baseline on both the training and the held-out nonce "
                 "sets. See scores.md.\n\n")
    else:
        head += ("This is the best candidate found, but it was **not confirmed** against the "
                 "held-out nonces. Treat scores.md as indicative only.\n\n")
    head += ("## Submitting\n\n1. Copy the algorithm files into "
             f"`tig-algorithms/src/{spec.challenge}/<your_name>/` in a monorepo checkout.\n"
             "2. Add the copyright header the TIG Inbound Game License requires.\n"
             "3. Follow docs/guides in the monorepo to submit. For Advance Rewards, complete "
             "evidence_draft.md.\n")
    return head


def build_package(spec: JobSpec, state: JobState, store: JobStore) -> Path:
    pkg = store.run_dir / "package"
    if pkg.exists():
        shutil.rmtree(pkg)
    pkg.mkdir()
    if state.best is None or state.baseline is None:
        (pkg / "README.md").write_text(_readme(spec, state) if state.baseline else
                                        "# Talos hand-back\n\nNo candidate was produced.\n")
        shutil.make_archive(str(store.run_dir / "package"), "zip", pkg)
        return pkg
    for name, text in state.best.files.items():
        (pkg / name).parent.mkdir(parents=True, exist_ok=True)
        (pkg / name).write_text(text)
    (pkg / "diff_vs_baseline.patch").write_text(_diff(state.baseline.files, state.best.files))
    scores = "# Training nonces\n\n" + scores_markdown(state.baseline.training, state.best.training)
    if state.best.holdout:
        scores += "\n# Held-out nonces\n\n" + scores_markdown(state.baseline.holdout, state.best.holdout)
    (pkg / "scores.md").write_text(scores)
    hyps = "\n".join(f"- #{h['iteration']} [{h.get('strategy_tag','')}] {h.get('title','')}: "
                     f"{h.get('description','')} -> {h.get('outcome','')}" for h in state.hypotheses)
    (pkg / "hypotheses.md").write_text("# Hypotheses\n\n" + hyps + "\n")
    (pkg / "evidence_draft.md").write_text(evidence_draft(spec, state))
    (pkg / "README.md").write_text(_readme(spec, state))
    shutil.make_archive(str(store.run_dir / "package"), "zip", pkg)
    return pkg
```

- [ ] **Step 5: Run, commit**

Run: `make check PYTHON=.venv/bin/python` → pass.

```bash
git add talos/package.py talos/data/evidence_template.md tests/test_package.py
git commit -m "package: files, diff, per-nonce tables, hypothesis log, evidence draft, zip"
```

---

### Task 13: CLI: setup, run, compile

**Files:**
- Create: `talos/config.py`, `talos/cli.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `talos.config.Config(provider, model, mode, api_base, secrets_path, config_path)` with `load(root) -> Config`, `save(root, config, api_key: str | None)`, `resolve_api_key(config) -> str | None` (secrets file, then env var named by provider), `ConfigError`.
- Produces: `talos.cli.main(argv=None) -> int` with subcommands `setup`, `run`, `compile`, `status`; wizard prompts via an injectable `ask(prompt, default=None, secret=False) -> str`.
- Produces: `talos.cli.deploy_bench(run=subprocess.run) -> None` which runs `modal token set` when tokens are given and `modal deploy modal_app/talos_bench.py`.

- [ ] **Step 1: Failing tests**

`tests/test_cli.py`:
```python
import json
import stat

from talos import cli
from talos.config import Config, load, resolve_api_key, save


def scripted(answers):
    it = iter(answers)
    def ask(prompt, default=None, secret=False):
        try:
            v = next(it)
        except StopIteration:
            raise AssertionError(f"unexpected prompt: {prompt}")
        return v if v != "" else (default or "")
    return ask


def test_setup_writes_config_and_0600_secret(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = []
    monkeypatch.setattr(cli, "validate_provider", lambda p: None)
    monkeypatch.setattr(cli, "deploy_bench", lambda token_id, token_secret, run=None: calls.append("deploy"))
    rc = cli.main(["setup"], ask=scripted(["anthropic", "", "sk-test", "ak-1", "as-1"]))
    assert rc == 0
    cfg = json.loads((tmp_path / "talos.config.json").read_text())
    assert cfg["provider"] == "anthropic" and cfg["model"] == "claude-opus-5"
    sec = tmp_path / ".talos" / "secrets.json"
    assert json.loads(sec.read_text()) == {"api_key": "sk-test"}
    assert stat.S_IMODE(sec.stat().st_mode) == 0o600  # mutation: dropping chmod fails this
    assert calls == ["deploy"]


def test_setup_rejected_key_writes_nothing(tmp_path, monkeypatch):
    # mutation: writing before validation leaves a bad key on disk
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "validate_provider", lambda p: "credential rejected")
    monkeypatch.setattr(cli, "deploy_bench", lambda *a, **k: None)
    rc = cli.main(["setup"], ask=scripted(["anthropic", "", "bad", "ak", "as"]))
    assert rc != 0
    assert not (tmp_path / "talos.config.json").exists()
    assert not (tmp_path / ".talos").exists()


def test_setup_cli_provider_stores_no_secret(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "validate_provider", lambda p: None)
    monkeypatch.setattr(cli, "deploy_bench", lambda *a, **k: None)
    rc = cli.main(["setup"], ask=scripted(["claude-cli", "", "", "ak", "as"]))  # mode defaults
    assert rc == 0
    assert not (tmp_path / ".talos" / "secrets.json").exists()
    assert json.loads((tmp_path / "talos.config.json").read_text())["provider"] == "claude-cli"


def test_run_requires_budget_and_creates_job(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    save(tmp_path, Config(provider="fake", model="fake", mode="single-shot", api_base=None), None)
    fake_info = type("I", (), {"id": "c003", "name": "knapsack", "is_gpu": False,
                               "tracks": ["n=1"], "max_fuel": 7})()
    monkeypatch.setattr(cli, "fetch_challenge_info", lambda name: fake_info)
    started = {}
    def fake_execute(spec, store, cfg, resume):
        started.update(spec=spec)
        return 0
    monkeypatch.setattr(cli, "execute_job", fake_execute)
    rc = cli.main(["run", "--challenge", "knapsack", "--direction", "go", "--budget-iterations", "3",
                   "--yes"])
    assert rc == 0
    spec = started["spec"]
    assert spec.challenge == "knapsack" and spec.fuel == 7 and spec.tracks == ["n=1"]
    assert spec.budget.iterations == 3
    assert len(spec.rand_hash) == 64
    job_files = list((tmp_path / "runs").glob("*/job.json"))
    assert len(job_files) == 1
    # mutation: zero-budget guard using truthiness would accept a job with no cap
    rc2 = cli.main(["run", "--challenge", "knapsack", "--direction", "go", "--yes"])
    assert rc2 != 0


def test_resolve_api_key_prefers_file_then_env(tmp_path, monkeypatch):
    save(tmp_path, Config(provider="openai", model="gpt-5", mode="single-shot", api_base=None), "from-file")
    cfg = load(tmp_path)
    assert resolve_api_key(cfg) == "from-file"
    (tmp_path / ".talos" / "secrets.json").unlink()
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    assert resolve_api_key(load(tmp_path)) == "from-env"
    monkeypatch.setenv("OPENAI_API_KEY", "")
    assert resolve_api_key(load(tmp_path)) is None  # empty env is unset, not a key
```

- [ ] **Step 2: Run to verify failure** → module not found.

- [ ] **Step 3: Write config.py**

```python
"""talos.config.json and .talos/secrets.json."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path

ENV_KEYS = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY",
            "google": "GEMINI_API_KEY", "openrouter": "OPENROUTER_API_KEY",
            "custom": "TALOS_CUSTOM_API_KEY"}


class ConfigError(RuntimeError):
    pass


@dataclass
class Config:
    provider: str
    model: str
    mode: str
    api_base: str | None
    config_path: Path | None = None
    secrets_path: Path | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("config_path")
        d.pop("secrets_path")
        return d


def load(root: Path) -> Config:
    root = Path(root)
    cp = root / "talos.config.json"
    if not cp.exists():
        raise ConfigError("talos.config.json not found; run `talos setup` first")
    d = json.loads(cp.read_text())
    return Config(provider=d["provider"], model=d["model"], mode=d.get("mode", "single-shot"),
                  api_base=d.get("api_base"), config_path=cp, secrets_path=root / ".talos" / "secrets.json")


def save(root: Path, config: Config, api_key: str | None) -> None:
    root = Path(root)
    (root / "talos.config.json").write_text(json.dumps(config.to_dict(), indent=1) + "\n")
    if api_key:
        sdir = root / ".talos"
        sdir.mkdir(exist_ok=True)
        sp = sdir / "secrets.json"
        sp.write_text(json.dumps({"api_key": api_key}) + "\n")
        os.chmod(sp, 0o600)


def resolve_api_key(config: Config) -> str | None:
    if config.secrets_path and config.secrets_path.exists():
        key = json.loads(config.secrets_path.read_text()).get("api_key")
        if key:
            return key
    env = ENV_KEYS.get(config.provider)
    val = os.environ.get(env, "") if env else ""
    return val or None
```

- [ ] **Step 4: Write cli.py**

```python
"""`talos setup | run | compile | status`. Wizard prompts go through `ask` so tests can script
them; `rich` renders the live status line."""
from __future__ import annotations

import argparse
import getpass
import json
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from talos.budget import Budget, Spend
from talos.challenges import CHALLENGES, MONOREPO_REF
from talos.config import Config, ConfigError, load, resolve_api_key, save
from talos.mainnet import MainnetError, fetch_challenge_info
from talos.nonces import draw_nonce_sets, new_rand_hash
from talos.providers import DEFAULT_MODELS, KINDS, make_provider, validate_provider
from talos.state import JobSpec, JobState, JobStore

BASELINE_CACHE = Path.home() / ".talos" / "baselines"
MODAL_APP_FILE = Path(__file__).resolve().parent.parent / "modal_app" / "talos_bench.py"


def default_ask(prompt: str, default: str | None = None, secret: bool = False) -> str:
    suffix = f" [{default}]" if default else ""
    v = getpass.getpass(f"{prompt}{suffix}: ") if secret else input(f"{prompt}{suffix}: ")
    return v.strip() or (default or "")


def deploy_bench(token_id: str | None, token_secret: str | None, run=subprocess.run) -> None:
    modal = shutil.which("modal") or [sys.executable, "-m", "modal"]
    base = modal if isinstance(modal, list) else [modal]
    if token_id and token_secret:
        r = run(base + ["token", "set", "--token-id", token_id, "--token-secret", token_secret],
                capture_output=True, text=True)
        if r.returncode != 0:
            raise ConfigError(f"modal token set failed: {r.stderr[-500:]}")
    r = run(base + ["deploy", str(MODAL_APP_FILE)], capture_output=True, text=True)
    if r.returncode != 0:
        raise ConfigError(f"modal deploy failed: {(r.stderr or r.stdout)[-2000:]}")


def cmd_setup(args, ask) -> int:
    root = Path.cwd()
    kinds = ", ".join(k for k in KINDS if k != "fake")
    kind = ask(f"Provider ({kinds})", "anthropic")
    if kind not in KINDS:
        print(f"unknown provider {kind!r}", file=sys.stderr)
        return 2
    model = ask("Model", DEFAULT_MODELS.get(kind) or None)
    api_base = ask("API base URL") if kind == "custom" else None
    api_key = None
    if kind in ("anthropic", "openai", "google", "openrouter", "custom"):
        api_key = ask("API key", secret=True)
    mode = "single-shot"
    if kind in ("claude-cli", "codex-cli"):
        mode = ask("Mode (single-shot or agentic)", "single-shot")
    token_id = ask("Modal token id (create at modal.com/settings/tokens)")
    token_secret = ask("Modal token secret", secret=True)
    provider = make_provider(kind, model, api_key=api_key, api_base=api_base)
    err = validate_provider(provider)
    if err:
        print(f"Provider check failed: {err}", file=sys.stderr)
        return 1
    try:
        deploy_bench(token_id or None, token_secret or None)
    except ConfigError as e:
        print(str(e), file=sys.stderr)
        return 1
    save(root, Config(provider=kind, model=model, mode=mode, api_base=api_base), api_key)
    print("Setup complete. Run `talos run` to start a job.")
    return 0


def _budget_from_args(args) -> Budget:
    b = Budget(usd=args.budget_usd, hours=args.budget_hours, iterations=args.budget_iterations,
               modal_usd=args.budget_modal_usd)
    b.validate()
    return b


def execute_job(spec: JobSpec, store: JobStore, cfg: Config, resume: bool) -> int:
    from talos.bench import ModalBench
    from talos.loop import Loop
    from talos.package import build_package
    from talos.agentic import attach_agentic
    provider = make_provider(cfg.provider, cfg.model, api_key=resolve_api_key(cfg), api_base=cfg.api_base)
    bench = ModalBench()
    state = store.load() if resume else JobState.fresh(Spend(started_at=time.time()))
    if not resume:
        store.save(state)
    spec_cls = CHALLENGES[spec.challenge]
    hardware = spec_cls.gpu or f"cpu{spec_cls.cpu}"

    def on_event(kind, data):
        line = f"[{time.strftime('%H:%M:%S')}] it={state.iteration} {kind} " + \
               " ".join(f"{k}={str(v)[:60]}" for k, v in data.items())
        print(line, flush=True)

    loop = Loop(spec, state, store, provider, bench, template_rs="", on_event=on_event)
    if cfg.mode == "agentic":
        attach_agentic(loop, cfg.provider, cfg.model)
    signal.signal(signal.SIGINT, lambda *_: loop.request_stop())
    try:
        if state.baseline is None:
            loop.measure_baseline(BASELINE_CACHE, hardware)
        else:
            from talos.mainnet import fetch_template
            loop.template_rs = fetch_template(spec.challenge)
        final = loop.run()
    except Exception as e:  # noqa: BLE001 - report, package what we have, exit non-zero
        state.status, state.stop_reason = "failed", str(e)
        store.save(state)
        final = state
    pkg = build_package(spec, final, store)
    print(f"\nStatus: {final.status} ({final.stop_reason})")
    print(f"LLM spend: ${final.spend.llm_usd:.2f}   Modal spend (estimated): ${final.spend.modal_usd:.2f}")
    print(f"Package: {pkg}")
    return 0 if final.status == "won" else 1


def cmd_run(args, ask) -> int:
    root = Path.cwd()
    try:
        cfg = load(root)
    except ConfigError as e:
        print(str(e), file=sys.stderr)
        return 2
    if args.resume:
        store = JobStore(root / "runs" / args.resume)
        return execute_job(store.read_spec(), store, cfg, resume=True)
    challenge = args.challenge or ask(f"Challenge ({', '.join(CHALLENGES)})", "vehicle_routing")
    if challenge not in CHALLENGES:
        print(f"unknown challenge {challenge!r}", file=sys.stderr)
        return 2
    direction = args.direction
    if args.direction_file:
        direction = Path(args.direction_file).read_text()
    if not direction:
        direction = ask("Direction for the agent (what to explore)")
    metered = cfg.provider not in ("claude-cli", "codex-cli")
    if not args.yes and args.budget_usd is None and args.budget_hours is None \
            and args.budget_iterations is None:
        if metered:
            args.budget_usd = float(ask("LLM budget in USD", "20"))
        else:
            args.budget_iterations = int(ask("Iteration budget", "50"))
        args.budget_hours = float(ask("Wall-clock budget in hours", "4"))
    try:
        budget = _budget_from_args(args)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2
    try:
        info = fetch_challenge_info(challenge)
    except MainnetError as e:
        print(f"mainnet unreachable: {e}", file=sys.stderr)
        return 1
    rand_hash = new_rand_hash()
    training, holdout = draw_nonce_sets(info.tracks, rand_hash)
    job_id = time.strftime("%Y%m%d-%H%M%S") + f"-{challenge}"
    spec = JobSpec(job_id=job_id, challenge=challenge, direction=direction, provider=cfg.provider,
                   model=cfg.model, mode=cfg.mode, budget=budget, rand_hash=rand_hash,
                   tracks=info.tracks, training=training, holdout=holdout, fuel=info.max_fuel,
                   created_at=time.time(), monorepo_ref=MONOREPO_REF, challenge_id=info.id)
    store = JobStore(root / "runs" / job_id)
    store.write_spec(spec)
    (store.run_dir / "tacit.md").write_text(f"- USER: {direction.strip()}\n")
    print(f"Job {job_id}: {len(info.tracks)} tracks, fuel {info.max_fuel}, budget {budget.to_dict()}")
    return execute_job(spec, store, cfg, resume=False)


def cmd_compile(args, ask) -> int:
    from talos.bench import ModalBench
    d = Path(args.dir)
    files = {str(p.relative_to(d)): p.read_text() for p in d.rglob("*")
             if p.is_file() and p.suffix in (".rs", ".cu")}
    r = ModalBench().compile(args.challenge, files)
    print(r.output[-4000:])
    return 0 if r.ok else 1


def cmd_status(args, ask) -> int:
    root = Path.cwd() / "runs"
    for job in sorted(root.glob("*/state.json")):
        st = json.loads(job.read_text())
        print(f"{job.parent.name}: {st['status']} it={st['iteration']} "
              f"llm=${st['spend']['llm_usd']:.2f} modal=${st['spend']['modal_usd']:.2f}")
    return 0


def main(argv=None, ask=default_ask) -> int:
    p = argparse.ArgumentParser(prog="talos")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup")
    r = sub.add_parser("run")
    r.add_argument("--challenge")
    r.add_argument("--direction")
    r.add_argument("--direction-file")
    r.add_argument("--budget-usd", type=float)
    r.add_argument("--budget-hours", type=float)
    r.add_argument("--budget-iterations", type=int)
    r.add_argument("--budget-modal-usd", type=float)
    r.add_argument("--resume")
    r.add_argument("--yes", action="store_true")
    c = sub.add_parser("compile")
    c.add_argument("--challenge", required=True)
    c.add_argument("--dir", default="algorithm")
    sub.add_parser("status")
    args = p.parse_args(argv)
    return {"setup": cmd_setup, "run": cmd_run, "compile": cmd_compile, "status": cmd_status}[args.cmd](args, ask)


if __name__ == "__main__":
    sys.exit(main())
```

`talos/agentic.py` does not exist yet; `execute_job` imports it lazily and Task 14 creates it. For this task's tests `execute_job` is monkeypatched, so the import never runs. To keep `make check` green before Task 14, create a stub `talos/agentic.py` containing only:

```python
"""Agentic mode. Filled in by the agentic task."""
def attach_agentic(loop, provider_kind: str, model: str) -> None:
    raise NotImplementedError("agentic mode not implemented yet")
```

- [ ] **Step 5: Run, commit**

Run: `make check PYTHON=.venv/bin/python` → pass.

```bash
git add talos/config.py talos/cli.py talos/agentic.py tests/test_cli.py
git commit -m "cli: setup and run wizards, compile, status, modal deploy"
```

---

### Task 14: Agentic mode

**Files:**
- Modify: `talos/agentic.py` (replace the stub)
- Create: `tests/test_agentic.py`

**Interfaces:**
- Consumes: `Loop`, `PromptContext` (Tasks 9, 11); `apply_edit_response` is not used here; files are read back from the worktree.
- Produces: `attach_agentic(loop, provider_kind, model) -> None` which sets `loop.propose_and_edit`; `prepare_worktree(run_dir, ctx) -> Path`; `sandbox_settings(worktree) -> dict`; `claude_md(ctx) -> str`; `read_back(worktree, ctx) -> tuple[dict, dict[str, str]]`; `AgenticError`.

- [ ] **Step 1: Failing tests**

`tests/test_agentic.py`:
```python
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from talos.agentic import (AgenticError, attach_agentic, claude_md, prepare_worktree, read_back,
                           sandbox_settings)
from talos.prompts import PromptContext
from talos.state import JobStore


def ctx():
    return PromptContext(challenge="knapsack", template_rs="pub fn solve_challenge(", direction="go",
                         tacit="- USER: go", files={"mod.rs": "fn a(){}", "ls.rs": "fn b(){}"},
                         baseline_name="b", best_delta=0.0)


def test_prepare_worktree_layout(tmp_path):
    wt = prepare_worktree(tmp_path, ctx())
    assert (wt / "algorithm" / "mod.rs").read_text() == "fn a(){}"
    assert (wt / "CHALLENGE.md").exists() and (wt / "tacit.md").read_text() == "- USER: go"
    assert (wt / ".claude" / "settings.json").exists()
    assert (wt / "AGENTS.md").exists()


def test_sandbox_denies_network_and_scopes_edits(tmp_path):
    # Claude Code applies deny before allow, so a broad deny like Edit(**) or Bash(*) would
    # silently block the very edits and compile command we allow.
    # mutation: adding "Edit(**)" or "Bash(*)" to deny breaks agentic mode entirely
    s = sandbox_settings(tmp_path)
    allow, deny = s["permissions"]["allow"], s["permissions"]["deny"]
    # Claude Code's documented prefix form is `Bash(cmd:*)`; a bare `*` is not a prefix match
    assert "Bash(talos compile:*)" in allow
    assert "Edit(algorithm/**)" in allow and "Edit(.talos/hypothesis.json)" in allow
    assert "WebFetch" in deny and "WebSearch" in deny and "Write(**)" in deny
    assert {"Bash(curl:*)", "Bash(wget:*)", "Bash(git:*)", "Bash(ssh:*)", "Bash(python:*)"} <= set(deny)
    assert "Edit(**)" not in deny and "Bash(*)" not in deny and "Bash(*:*)" not in deny
    assert s["permissions"]["defaultMode"] == "dontAsk"  # unlisted tools are refused, not prompted


def test_read_back_requires_hypothesis_and_returns_files(tmp_path):
    wt = prepare_worktree(tmp_path, ctx())
    with pytest.raises(AgenticError):
        read_back(wt, ctx())
    (wt / ".talos" / "hypothesis.json").write_text(json.dumps(
        {"title": "T", "description": "D", "strategy_tag": "local_search"}))
    (wt / "algorithm" / "mod.rs").write_text("fn a(){ 1 }")
    (wt / "algorithm" / "evil.rs").write_text("x")  # new file: not one of the algorithm's files
    hyp, files = read_back(wt, ctx())
    assert hyp["title"] == "T" and files["mod.rs"] == "fn a(){ 1 }"
    assert "evil.rs" not in files  # mutation: globbing the dir would pick it up


def test_claude_md_mentions_compile_and_hypothesis():
    text = claude_md(ctx())
    assert "talos compile" in text and ".talos/hypothesis.json" in text and "knapsack" in text


def test_agentic_error_is_a_failed_iteration_not_a_crash():
    # mutation: a RuntimeError base escapes Loop.iterate's except clause and kills the whole run
    assert issubclass(AgenticError, ValueError)


def test_attach_agentic_timeout_is_agentic_error_and_talos_is_on_path(tmp_path):
    seen = {}

    def run(cmd, **kw):
        seen.update(cmd=cmd, env=kw.get("env"), cwd=kw.get("cwd"))
        raise subprocess.TimeoutExpired(cmd, 1)

    loop = types.SimpleNamespace(store=JobStore(tmp_path), propose_and_edit=None,
                                 _check_budget=lambda: None, _event=lambda *a, **k: None)
    attach_agentic(loop, "claude-cli", "m", timeout_s=1, run=run)
    with pytest.raises(AgenticError):  # mutation: letting TimeoutExpired escape crashes the loop
        loop.propose_and_edit(ctx())
    assert seen["cmd"][0] == "claude"  # mutation: shutil.which() makes argv[0] machine-dependent
    assert "--permission-mode" in seen["cmd"] and seen["cwd"] == tmp_path / "agentic"
    # mutation: dropping the PATH prepend means `talos compile` is not found inside the sandbox
    assert seen["env"]["PATH"].split(os.pathsep)[0] == str(Path(sys.executable).parent)
```

- [ ] **Step 2: Run to verify failure** → `ImportError` on the stub.

- [ ] **Step 3: Write agentic.py**

```python
"""Agentic mode: one headless claude or codex call per iteration inside a sandboxed worktree.
The agent edits algorithm files, may run `talos compile`, and must write .talos/hypothesis.json.
The loop reads the files back and owns compile, score and publish as in single-shot mode."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from talos.prompts import PromptContext, STRATEGY_TAGS, _rust_rules


class AgenticError(ValueError):
    """A ValueError so Loop.iterate records it as a failed iteration instead of crashing the run."""


def sandbox_settings(worktree: Path) -> dict:
    """Deny is evaluated before allow in Claude Code, so never deny a glob that covers an
    allowed path. Unlisted tools are refused by `dontAsk` mode rather than prompted for."""
    return {"permissions": {
        "allow": ["Read(algorithm/**)", "Read(CHALLENGE.md)", "Read(tacit.md)", "Read(AGENTS.md)",
                  "Read(.talos/hypothesis.json)", "Edit(algorithm/**)",
                  "Edit(.talos/hypothesis.json)", "Bash(talos compile:*)"],
        "deny": ["WebFetch", "WebSearch", "Write(**)", "Bash(curl:*)", "Bash(wget:*)", "Bash(git:*)",
                 "Bash(ssh:*)", "Bash(python:*)", "Bash(pip:*)", "Bash(nc:*)", "Bash(rm:*)"],
        "defaultMode": "dontAsk"}}


def claude_md(ctx: PromptContext) -> str:
    return f"""# Talos agentic iteration: {ctx.challenge}

You are improving a Rust solver for the TIG challenge "{ctx.challenge}". Beat the mainnet
baseline "{ctx.baseline_name}" on TIG's benchmark (higher verifier quality per nonce under a
fixed fuel budget on every active track). Your current best is {ctx.best_delta:+.3%} vs baseline.

Rules:
- Edit ONLY files under `algorithm/`. Do not create new files. Do not touch anything else.
- You may run `talos compile --challenge {ctx.challenge} --dir algorithm` to check the build.
  Nothing else may be executed. There is no network.
- Make ONE focused change per iteration that implements a single hypothesis.
- Before you stop, EDIT the existing file `.talos/hypothesis.json` (it starts as `{{}}`) so it
  holds keys "title", "description", "strategy_tag" (one of: {", ".join(STRATEGY_TAGS)}).
  Use the Edit tool; creating new files is not permitted.

Direction from the user:
{ctx.direction}

Tacit knowledge so far is in `tacit.md`. The solver contract is in `CHALLENGE.md`.

{_rust_rules()}
"""


def prepare_worktree(run_dir: Path, ctx: PromptContext) -> Path:
    wt = Path(run_dir) / "agentic"
    if wt.exists():
        shutil.rmtree(wt)
    (wt / "algorithm").mkdir(parents=True)
    (wt / ".talos").mkdir()
    (wt / ".talos" / "hypothesis.json").write_text("{}\n")  # agent must Edit, Write is denied
    (wt / ".claude").mkdir()
    for name, text in ctx.files.items():
        p = wt / "algorithm" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    (wt / "CHALLENGE.md").write_text(f"# {ctx.challenge} solver contract (template.rs)\n\n"
                                     f"```rust\n{ctx.template_rs}\n```\n")
    (wt / "tacit.md").write_text(ctx.tacit)
    md = claude_md(ctx)
    (wt / "CLAUDE.md").write_text(md)
    (wt / "AGENTS.md").write_text(md)
    (wt / ".claude" / "settings.json").write_text(json.dumps(sandbox_settings(wt), indent=1))
    return wt


def read_back(worktree: Path, ctx: PromptContext) -> tuple[dict, dict[str, str]]:
    hp = worktree / ".talos" / "hypothesis.json"
    if not hp.exists() or hp.read_text().strip() in ("", "{}"):
        raise AgenticError("agent did not fill in .talos/hypothesis.json")
    try:
        h = json.loads(hp.read_text())
        hypothesis = {"title": str(h["title"]), "description": str(h["description"]),
                      "strategy_tag": h.get("strategy_tag") if h.get("strategy_tag") in STRATEGY_TAGS else "hybrid"}
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise AgenticError(f"bad hypothesis.json: {e}") from None
    files = {name: (worktree / "algorithm" / name).read_text() for name in ctx.files
             if (worktree / "algorithm" / name).exists()}
    if files == ctx.files:
        raise AgenticError("agent changed no algorithm file")
    return hypothesis, files


def _agent_env() -> dict[str, str]:
    """The agent runs `talos compile`; make sure the interpreter that runs Talos is first on PATH so
    the console script resolves even when the venv is not activated in the agent's shell."""
    env = dict(os.environ)
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    return env


def _run_agent(cmd: list[str], wt: Path, timeout_s: int, run) -> None:
    try:
        r = run(cmd, cwd=wt, capture_output=True, text=True, timeout=timeout_s, env=_agent_env())
    except FileNotFoundError:
        raise AgenticError(f"{cmd[0]} CLI not found on PATH") from None
    except subprocess.TimeoutExpired:
        raise AgenticError(f"{cmd[0]} timed out after {timeout_s}s") from None
    (wt / ".talos" / "agent_stdout.txt").write_text(r.stdout or "")
    (wt / ".talos" / "agent_stderr.txt").write_text(r.stderr or "")
    if r.returncode != 0:
        raise AgenticError(f"{cmd[0]} exited {r.returncode}: {(r.stderr or '')[-500:]}")


def _run_claude(wt: Path, model: str, prompt: str, timeout_s: int, run) -> None:
    _run_agent(["claude", "-p", "--model", model, "--settings", str(wt / ".claude" / "settings.json"),
                "--permission-mode", "dontAsk", prompt], wt, timeout_s, run)


def _run_codex(wt: Path, model: str, prompt: str, timeout_s: int, run) -> None:
    _run_agent(["codex", "exec", "-m", model, "--sandbox", "workspace-write",
                "--skip-git-repo-check", prompt], wt, timeout_s, run)


def attach_agentic(loop, provider_kind: str, model: str, timeout_s: int = 1800,
                   run=subprocess.run) -> None:
    runner = {"claude-cli": _run_claude, "codex-cli": _run_codex}.get(provider_kind)
    if runner is None:
        raise AgenticError(f"agentic mode needs claude-cli or codex-cli, not {provider_kind}")

    def propose_and_edit(ctx: PromptContext) -> tuple[dict, dict[str, str]]:
        wt = prepare_worktree(loop.store.run_dir, ctx)
        prompt = ("Read CLAUDE.md (or AGENTS.md), then make one improvement to the solver under "
                  "algorithm/, check it with `talos compile`, and write .talos/hypothesis.json.")
        if ctx.failed_hypotheses:
            prompt += "\nAlready tried and failed against this code: " + "; ".join(
                h.get("title", "") for h in ctx.failed_hypotheses)
        if ctx.forced_tag:
            prompt += f"\nYou have stagnated: use strategy_tag \"{ctx.forced_tag}\"."
        loop._check_budget()
        runner(wt, model, prompt, timeout_s, run)
        hypothesis, files = read_back(wt, ctx)
        loop._event("hypothesis", **hypothesis)
        return hypothesis, files

    loop.propose_and_edit = propose_and_edit
```

Verified on the dev machine during the plan audit (2026-09-11): `claude --help` lists `--settings <file-or-json>`, `--permission-mode` with choices `acceptEdits, auto, bypassPermissions, manual, dontAsk, plan`, `--system-prompt`, `--output-format`; `codex exec --help` lists `-m/--model`, `-s/--sandbox`, `--skip-git-repo-check`, `-o/--output-last-message`. Permission rules use Claude Code's documented prefix form `Bash(cmd:*)`. Codex ignores `.claude/settings.json`; its `workspace-write` sandbox denies network but allows other commands — the read-back scope check is what enforces "algorithm files only" there.

- [ ] **Step 4: Run, commit**

Run: `make check PYTHON=.venv/bin/python` → pass.

```bash
git add talos/agentic.py tests/test_agentic.py
git commit -m "agentic: sandboxed worktree, claude/codex iterate, read-back with scope check"
```

---

### Task 15: Live smoke test, README, release check

**Files:**
- Create: `tests/test_live.py`
- Modify: `README.md`

- [ ] **Step 1: Write the live smoke test (marker `live`, never in `make check`)**

`tests/test_live.py`:
```python
"""Manual: compile and score the real mainnet top algorithm on Modal for one challenge.
Run: TALOS_LIVE_CHALLENGE=knapsack .venv/bin/pytest -m live tests/test_live.py -s
Needs `talos setup` done (Modal deployed) and network."""
import os

import pytest

from talos import mainnet
from talos.bench import ModalBench
from talos.nonces import draw_nonce_sets, new_rand_hash

pytestmark = pytest.mark.live


def test_baseline_compiles_and_scores():
    ch = os.environ.get("TALOS_LIVE_CHALLENGE", "knapsack")
    info = mainnet.fetch_challenge_info(ch)
    name, adoption = mainnet.top_algorithm(ch)
    files = mainnet.fetch_algorithm_files(ch, name)
    bench = ModalBench()
    c = bench.compile(ch, files)
    assert c.ok, c.output[-3000:]
    tr, _ = draw_nonce_sets(info.tracks[:1], new_rand_hash(), training_count=2, holdout_count=0)
    res = bench.score(ch, c.artifact_id, tr, info.max_fuel)
    assert len(res) == 2
    assert all(r.error != "panic" for r in res), [r.to_dict() for r in res]
    assert any(r.ok for r in res), [r.to_dict() for r in res]
    print({"algorithm": name, "adoption": adoption, "results": [r.to_dict() for r in res]})
```

- [ ] **Step 2: Run it once for real on a CPU challenge**

Run: `.venv/bin/talos setup` (real credentials), then
`TALOS_LIVE_CHALLENGE=knapsack .venv/bin/pytest -m live tests/test_live.py -s`
Expected: PASS and a printed dict with two `ok: True` results. Record the printed output in the commit message body as MEASURED. If `modal deploy` or the image build fails, fix `modal_app/talos_bench.py` first; the plan's `from_registry(..., add_python="3.11")` and the `git clone` step are the two likely failure points.

- [ ] **Step 3: Fake end-to-end run**

Add to `talos/cli.py` a hidden flag `--fake` on `run` (`r.add_argument("--fake", action="store_true", help=argparse.SUPPRESS)`). Keep the `execute_job(spec, store, cfg, resume)` signature unchanged (Task 13's test monkeypatches it with exactly those four parameters); route the fakes through `cfg.provider == "fake"` instead:

- In `cmd_run`, when `args.fake`: set `cfg = Config(provider="fake", model="fake", mode="single-shot", api_base=None)` and use a canned `ChallengeInfo(id="c003", name=challenge, is_gpu=False, tracks=["n=1"], max_fuel=1)` instead of calling `fetch_challenge_info`.
- In `execute_job`, when `cfg.provider == "fake"`: `bench = FakeBench(scores)` where `scores` parses `let k = (\d+);` from `mod.rs` and returns `100 + k - 1` per nonce (like `tests/test_loop.py`); `provider = FakeProvider(script)` where `script(system, user)` returns a hypothesis JSON when `"strategy_tag" (one of` is in `system`, else an edit block bumping `k` by one (parse the current `k` from `user`); and `loop.measure_baseline(BASELINE_CACHE / "fake", hardware, mainnet=types.SimpleNamespace(top_algorithm=lambda ch: ("fake_base", 1), fetch_algorithm_files=lambda ch, name: {"mod.rs": "fn solve() { let k = 1; }\n"}, fetch_template=lambda ch: "pub fn solve_challenge("))`. With margin 0.005, `k=2` gives +1% and wins on training and held-out at iteration 1.

Then run:

```bash
.venv/bin/talos run --challenge knapsack --direction "test" --budget-iterations 5 --yes --fake
```
Expected: prints iteration events, ends with `Status: won` or `exhausted`, and `runs/<job>/package/` exists with `scores.md`.

- [ ] **Step 4: README**

Replace `README.md` with usage: requirements (Python 3.10+, Git, a Modal account, one LLM credential), the three commands, what each wizard asks, where results land, the budget semantics, that Modal spend is an estimate, and a "How it works" paragraph pointing at the spec.

- [ ] **Step 5: Final check and commit**

Run: `make check PYTHON=.venv/bin/python` → pass. Report the test count from the pytest summary line as MEASURED.

```bash
git add tests/test_live.py talos/cli.py README.md
git commit -m "release: live smoke test, fake end-to-end run, README"
```

---

## Self-review

**Spec coverage.** §5.1 setup wizard: Task 13. §5.2 run wizard, flags, resume, Ctrl-C: Tasks 11 and 13. §5.3 `talos compile`: Task 13. §5.4 run directory: Task 6. §7.1 config: Task 13. §7.2 providers: Task 8. §7.3 baseline and cache: Task 10. §7.4 bench, image, hardware, tracks, fuel: Tasks 2 and 7. §7.5 scoring: Task 3. §7.6 loop, stagnation, agentic swap: Tasks 11 and 14. §7.7 stop rule and states: Tasks 6 and 11. §7.8 budget: Tasks 5 and 11. §7.9 package: Task 12. §7.10 resume: Task 11. §9 error handling: compile fix rounds and edit scope in Tasks 4 and 11, Modal retry and `paused` in Tasks 7 and 11, provider error classes in Task 8, seed hygiene in Tasks 6, 9 and 12. §10 security: 0600 secrets in Task 13, sandbox in Task 14. §11 testing: fakes in Tasks 7 and 8, live smoke in Task 15, wizard tests in Task 13.

**Gaps acknowledged.** The spec's "reject edits outside algorithm files in agentic mode" is enforced by reading back only the known file names (Task 14) rather than by the CLI sandbox alone. Modal per-second prices in `talos/bench.py` are ESTIMATE (unverified) and must be labelled as such in the CLI output, which Task 13 does with "(estimated)". OpenAI and Google prices are absent from the pricing table on purpose; those models report "unpriced" until the executor adds current rates.

**Audit notes (2026-09-11), flagged and deliberately left.** (1) `cache_key` in Task 10 includes each nonce set's `rand_hash`, which is drawn fresh per job, so the spec §7.3 claim that "a second job on the same challenge skips measurement" does not hold; the cache only helps a resumed job, and a resumed job already carries its baseline in `state.json`. Making the cache cross-job would need a per-challenge deterministic hash, which weakens the seed-hygiene story; left for the user to decide. (2) Task 13's setup wizard returns non-zero on a rejected credential instead of re-prompting (spec §5.1); the wizard tests pin this. (3) A kill during held-out confirmation leaves `state.best` pointing at an iteration whose directory is then discarded and re-run under the same number on resume; harmless for scoring, untidy in `hypotheses.md`. (4) `fetch_algorithm_files` returns every file in the algorithm's directory (README.md included); they are staged and editable, and `talos compile` only uploads `.rs`/`.cu`.

**Type consistency.** `Bench.score` returns `list[NonceResult]` everywhere; `FakeBench.scores` callback signature `(challenge, files, nonce_set)` is the same in Tasks 7, 10 and 11. `Candidate.delta` is a dict from `BundleDelta.to_dict()`, read as `delta["mean_rel_delta"]` in Task 11. `Loop.propose_and_edit` returns `(hypothesis_dict, files_dict)` in both Task 11 and Task 14. `JobStore.iteration_dir(n)` is used in Tasks 6, 11. `exhausted()` returns the dimension string used as `stop_reason` in Task 11 and asserted in tests.
