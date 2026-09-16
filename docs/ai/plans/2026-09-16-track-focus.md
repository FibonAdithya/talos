# Track focus (`--track`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `talos run --track <name>` optimises one mainnet track: training scores that track only, confirmation adds every other track's training nonces as a regression guard, prompts name the target, and the package reports per section.

**Architecture:** The job spec records the track; nonce sets and the baseline stay as they are (all tracks, same cache key). Two pure helpers in `talos/scoring.py` slice results and build the per-iteration nonce sets; the loop, the package and the prompts read them. A new focused beat rule runs at confirmation. No bench or backend changes.

**Tech Stack:** Python 3.10+, pytest, ruff. Gate is `make check` run from a Python 3.11+ venv (the agentify checker needs 3.11; the repo `.venv` is 3.10, so use `uv venv --python 3.11 <scratch>/venv311 && uv pip install --python <scratch>/venv311/bin/python -r requirements-dev.txt -e .`). Unit tests alone run fine on `.venv/bin/pytest`.

**Spec:** `docs/ai/specs/2026-09-16-track-focus-design.md`

## Global Constraints

- Without `--track`, every observable behaviour is byte-identical to today; every existing test must keep passing unmodified except the one wizard test named in Task 5.
- `rand_hash` never enters a prompt, event, package, or exception message. `select` keys on `(track, nonce)` only.
- Baseline and candidate are compared only on identical `NonceSet` lists (AGENTS.md invariant 1).
- No line over 100 characters (ruff does not check E501; count with Python, not awk).
- Every new test carries a `# mutation:` comment naming the code change it catches, matching the repo's convention.
- Branch: `codex-model-list`. Commit per task with explicit paths; never `git add -A`.

---

### Task 1: `JobSpec.track`

**Files:**
- Modify: `talos/state.py:19-56`
- Test: `tests/test_state.py`

**Interfaces:**
- Produces: `JobSpec.track: str | None` (default `None`, last field). `JobSpec.from_dict` tolerates a dict without the key.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_state.py`:

```python
def test_spec_track_round_trips_and_defaults_to_none(tmp_path):
    # mutation: dropping the `.get` default in from_dict makes every job.json written before
    # the field existed unresumable; dropping the field loses the focus on resume
    focused = replace(spec(), track="n=1")
    store = JobStore(tmp_path)
    store.write_spec(focused)
    assert store.read_spec().track == "n=1"
    old = spec().to_dict()
    del old["track"]
    assert JobSpec.from_dict(old).track is None
    assert spec().track is None
```

Add `from dataclasses import replace` to the imports at the top of `tests/test_state.py`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_state.py -k track -q`
Expected: FAIL with `TypeError: replace() got an unexpected keyword argument 'track'`.

- [ ] **Step 3: Add the field**

In `talos/state.py`, class `JobSpec`, add after `challenge_id: str`:

```python
    track: str | None = None  # one active track to optimise; None = all tracks
```

In `JobSpec.from_dict`, before `return cls(**d)`:

```python
        d["track"] = d.get("track")
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/pytest tests/test_state.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add talos/state.py tests/test_state.py
git commit -m "state: optional track on the job spec"
```

---

### Task 2: Slicing helpers and the focused beat rule

**Files:**
- Modify: `talos/scoring.py`
- Test: `tests/test_scoring.py`

**Interfaces:**
- Produces:
  - `select(results: list[NonceResult], sets: list[NonceSet]) -> list[NonceResult]`
  - `focus_sets(track: str | None, training: list[NonceSet], holdout: list[NonceSet]) -> tuple[list[NonceSet], list[NonceSet]]`
  - `beats_focused(baseline: list[NonceResult], candidate: list[NonceResult], rule: BeatRule, track: str) -> bool`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_scoring.py` (add `from talos.scoring import beats_focused, focus_sets, select` and `from talos.types import NonceSet` to the imports):

```python
H = "ab" * 32
TR2 = [NonceSet("t1", H, 0, 2), NonceSet("t2", H, 0, 2)]
HO2 = [NonceSet("t1", H, 1_000_000, 2), NonceSet("t2", H, 1_000_000, 2)]


def test_select_keeps_rows_inside_the_sets_in_input_order():
    # mutation: keying on track alone keeps held-out rows in a training slice; keying on nonce
    # alone keeps t2's rows in a t1 slice
    rows = [R("t2", 0, 1), R("t1", 1_000_000, 2), R("t1", 1, 3), R("t1", 0, 4), R("t1", 2, 5)]
    out = select(rows, [NonceSet("t1", H, 0, 2)])
    assert [(r.track, r.nonce) for r in out] == [("t1", 1), ("t1", 0)]


def test_focus_sets_narrows_training_and_appends_guards_to_holdout():
    # mutation: forgetting the guard leaves shared-code regressions on t2 unseen; using t2's
    # held-out set as the guard would compare against baseline rows the guard never measured
    tr, ho = focus_sets("t1", TR2, HO2)
    assert tr == [TR2[0]]
    assert ho == [HO2[0], TR2[1]]


def test_focus_sets_without_a_track_is_the_identity():
    # mutation: the focus path leaking into unfocused jobs changes every existing request
    assert focus_sets(None, TR2, HO2) == (TR2, HO2)


RULE = BeatRule(margin=0.005, track_tolerance=0.0, error_ceiling=0.05)


def test_beats_focused_fails_on_a_guard_regression():
    # mutation: dropping the guard check confirms a candidate that broke t2
    cand = [R("t1", 0, 120), R("t1", 1, 120), R("t2", 0, 199), R("t2", 1, 200)]
    assert not beats_focused(BASE, cand, RULE, "t1")
    # (the unfocused rule rejects this too, via worst_rel_delta; the point is that the guard
    # check, not the margin, is what fails here: the focus track is +20%)
    tol = BeatRule(margin=0.005, track_tolerance=0.01, error_ceiling=0.05)
    assert beats_focused(BASE, cand, tol, "t1")  # a 0.25% drop is inside a 1% tolerance


def test_beats_focused_applies_the_margin_to_the_focus_track_only():
    # mutation: applying the margin to the mean confirms a candidate whose focus track is flat
    # but whose guard track happened to rise
    cand = [R("t1", 0, 100), R("t1", 1, 100), R("t2", 0, 220), R("t2", 1, 220)]
    assert not beats_focused(BASE, cand, RULE, "t1")
    assert beats_focused(BASE, cand, RULE, "t2")


def test_beats_focused_counts_errors_on_the_focus_track_only():
    # mutation: using the bundle error rate lets a focus-track error rate of 50% pass because
    # 40 clean guard rows dilute it to 1/42 = 2.4%, under the 5% ceiling; ignoring errors
    # entirely passes it too (the focus track is +50% on quality)
    base = [R("t1", 0, 100), R("t1", 1, 100)] + [R("t2", n, 200) for n in range(40)]
    cand = ([R("t1", 0, 200), R("t1", 1, None, "panic")]
            + [R("t2", n, 200) for n in range(40)])
    assert bundle_delta(base, cand).error_rate < RULE.error_ceiling  # the diluted rate
    assert not beats_focused(base, cand, RULE, "t1")


def test_beats_focused_on_a_single_track_equals_beats():
    # mutation: the two rules diverging when there is nothing to guard
    # 200 -> 201 is rel_delta 0.005 exactly, the margin: a `>` in one rule and `>=` in the other
    # diverges here and nowhere else in this list
    base = [R("t1", 0, 200), R("t1", 1, 200)]
    for q in (200, 201, 202, 220):
        cand = [R("t1", 0, q), R("t1", 1, q)]
        assert beats_focused(base, cand, RULE, "t1") == beats(base, cand, RULE)
    assert beats_focused(base, [R("t1", 0, 201), R("t1", 1, 201)], RULE, "t1")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_scoring.py -q`
Expected: collection error `ImportError: cannot import name 'beats_focused'`.

- [ ] **Step 3: Implement**

In `talos/scoring.py`, add `from talos.types import NonceResult, NonceSet` (replacing the existing `NonceResult` import) and append:

```python
def select(results: list[NonceResult], sets: list[NonceSet]) -> list[NonceResult]:
    """The rows whose (track, nonce) fall inside one of `sets`, in input order. Strays are
    dropped, not raised on; bundle_delta still raises on any mismatch that survives."""
    wanted = {(s.track, n) for s in sets for n in s.nonces()}
    return [r for r in results if (r.track, r.nonce) in wanted]


def focus_sets(track: str | None, training: list[NonceSet],
               holdout: list[NonceSet]) -> tuple[list[NonceSet], list[NonceSet]]:
    """The nonce sets one iteration scores. Focused: training is the track's own training set;
    held-out is the track's held-out set followed by every other track's TRAINING set, the
    regression guard (already measured for the baseline). Unfocused: unchanged."""
    if track is None:
        return training, holdout
    focus_tr = [s for s in training if s.track == track]
    focus_ho = [s for s in holdout if s.track == track]
    guards = [s for s in training if s.track != track]
    return focus_tr, focus_ho + guards


def beats_focused(baseline: list[NonceResult], candidate: list[NonceResult], rule: BeatRule,
                  track: str) -> bool:
    """Confirmation rule for a focused job: the focus track clears the margin with its own error
    rate under the ceiling, and no guard track drops below -track_tolerance."""
    d = bundle_delta(baseline, candidate)
    by = {t.track: t for t in d.tracks}
    if track not in by:
        raise ScoringError(f"focus track {track!r} has no results")
    focus = by[track]
    guards_ok = all(t.rel_delta >= -rule.track_tolerance for t in d.tracks if t.track != track)
    return (focus.rel_delta >= rule.margin
            and focus.cand_errors / focus.n <= rule.error_ceiling
            and guards_ok)
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/pytest tests/test_scoring.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add talos/scoring.py tests/test_scoring.py
git commit -m "scoring: select, focus_sets and the focused beat rule"
```

---

### Task 3: The loop builds focused requests and confirms with the guard

**Files:**
- Modify: `talos/loop.py:20` (imports), `:135-139` (`_request`), `:214-225` (`_context`), `:302-345` (`_score_candidate`), `:346-383` (`_confirm`)
- Test: `tests/test_loop.py`

**Interfaces:**
- Consumes: `select`, `focus_sets`, `beats_focused` from Task 2; `JobSpec.track` from Task 1.
- Produces: `PromptContext` is built with `track=` and `guard_tracks=` keyword arguments (Task 4 adds those fields; until then the context test in this task passes `track` through only after Task 4 lands, so **do Task 4 before Step 3c below** or add the fields as part of this task's Step 3c — see the note there).

- [ ] **Step 1: Extend the test fixtures**

In `tests/test_loop.py`, replace the `TR`, `HO`, `spec` and `baseline` helpers with:

```python
TR = [NonceSet("t", HASH, 0, 4)]
HO = [NonceSet("t", HASH, 1_000_000, 4)]
TR2 = [NonceSet("t", HASH, 0, 4), NonceSet("u", HASH, 0, 4)]
HO2 = [NonceSet("t", HASH, 1_000_000, 4), NonceSet("u", HASH, 1_000_000, 4)]
BASE_FILES = {"mod.rs": "fn solve() { let k = 1; }\n"}


def spec(budget=None, track=None, two_tracks=False):
    tracks = ["t", "u"] if two_tracks else ["t"]
    return JobSpec(job_id="j", challenge="knapsack", direction="go", provider="fake", model="m",
                   mode="single-shot",
                   budget=budget or Budget(usd=None, hours=None, iterations=20, compute_usd=None),
                   rand_hash=HASH, tracks=tracks, training=TR2 if two_tracks else TR,
                   holdout=HO2 if two_tracks else HO, fuel=1, created_at=0.0,
                   monorepo_ref="r", challenge_id="c003", track=track)


def baseline(q=100, holdout_n=4, tracks=("t",)):
    tr = [NonceResult(t, n, True, q, 1) for t in tracks for n in range(4)]
    ho = [NonceResult(t, 1_000_000 + n, True, q, 1) for t in tracks for n in range(holdout_n)]
    return BaselineRecord(name="base", adoption=1, artifact_id="base-art", files=BASE_FILES,
                          training=tr, holdout=ho)
```

and give `make` two new keyword parameters, threading them through:

```python
def make(tmp_path, script, scores=quality_from_files, budget=None, thresholds=None, holdout_n=4,
         track=None, two_tracks=False):
    store = JobStore(tmp_path)
    sp = spec(budget, track=track, two_tracks=two_tracks)
    store.write_spec(sp)
    st = JobState.fresh(Spend(started_at=0.0))
    st.baseline = baseline(holdout_n=holdout_n, tracks=("t", "u") if two_tracks else ("t",))
    ...  # rest unchanged
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_loop.py`:

```python
def test_focused_request_narrows_training_and_guards_holdout(tmp_path):
    # mutation: forgetting the guard sends only t's held-out set; sending the whole baseline
    # makes the in-job holdout decision raise a nonce mismatch on every iteration
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(5)], track="t", two_tracks=True)
    loop.run()
    req = fb.calls[0]
    assert req.training == [TR2[0]]
    assert req.holdout == [HO2[0], TR2[1]]
    assert {(r.track, r.nonce) for r in req.baseline_training} == {("t", n) for n in range(4)}


def test_focused_win_is_confirmed_on_the_track_and_the_guard(tmp_path):
    # mutation: comparing the guard rows against the baseline's held-out rows (wrong nonces)
    # raises a ScoringError and records a false positive instead of the win
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(5)], track="t", two_tracks=True)
    st = loop.run()
    assert st.status == "won" and st.confirmed == [1]
    assert {r.track for r in st.best.holdout} == {"t", "u"}


def test_focused_guard_regression_is_a_false_positive(tmp_path):
    # mutation: confirming on the focus track alone declares a win that broke track u
    def scores(challenge, files, ns):
        import re
        k = int(re.search(r"let k = (\d+);", files["mod.rs"]).group(1))
        if ns.track == "u" and k != 1:
            return [90 for _ in ns.nonces()]  # the edit regresses the other track
        return [100 + k - 1 for _ in ns.nonces()]
    b = Budget(usd=None, hours=None, iterations=1, compute_usd=None)
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(5)], scores, b, track="t",
                               two_tracks=True)
    st = loop.run()
    assert st.status == "exhausted" and st.false_positives == [1] and st.confirmed == []
    events = [json.loads(ln) for ln in (tmp_path / "timeline.jsonl").read_text().splitlines()]
    fp_event = next(e for e in events if e["kind"] == "false_positive")
    assert fp_event["holdout"]["worst_rel_delta"] < 0


def test_unfocused_request_is_unchanged(tmp_path):
    # mutation: the focus path leaking into unfocused jobs changes every existing request
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(5)], two_tracks=True)
    loop.run()
    req = fb.calls[0]
    assert req.training == TR2 and req.holdout == HO2
    assert len(req.baseline_training) == 8


def test_focused_context_names_track_and_guards(tmp_path):
    # mutation: dropping either field leaves the model aiming at every track
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(5)], track="t", two_tracks=True)
    ctx = loop._context()
    assert ctx.track == "t" and ctx.guard_tracks == ["u"]
    loop2, *_ = make(tmp_path / "b", [hyp("a"), edit(5)], two_tracks=True)
    ctx2 = loop2._context()
    assert ctx2.track is None and ctx2.guard_tracks == []
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_loop.py -k focused -q`
Expected: the request test fails on `req.training == [TR2[0]]` (today it sends both sets); the context test fails with `TypeError: PromptContext.__init__() got an unexpected keyword argument 'track'` once Step 3c is in, or on `ctx.track` before it.

- [ ] **Step 3a: Imports**

In `talos/loop.py` change the scoring import to:

```python
from talos.scoring import ScoringError, beats, beats_focused, bundle_delta, focus_sets, select
```

- [ ] **Step 3b: `_request` and a `_focus` helper**

Replace `_request`:

```python
    def _focus(self) -> tuple[list, list]:
        return focus_sets(self.spec.track, self.spec.training, self.spec.holdout)

    def _request(self, files: dict[str, str], baseline_training) -> EvalRequest:
        training, holdout = self._focus()
        base = select(baseline_training, training) if baseline_training is not None else None
        return EvalRequest(challenge=self.spec.challenge, files=files, training=training,
                           holdout=holdout, fuel=self.spec.fuel, baseline_training=base,
                           rule=self.rule)
```

- [ ] **Step 3c: `_context`**

Add to the `PromptContext(...)` call in `_context`:

```python
                             track=self.spec.track,
                             guard_tracks=([t for t in self.spec.tracks if t != self.spec.track]
                                           if self.spec.track else []),
```

Note: `PromptContext` gains these two fields in Task 4. If executing this task first, add to `talos/prompts.py::PromptContext` now, after `is_gpu: bool = False`:

```python
    track: str | None = None
    guard_tracks: list[str] = field(default_factory=list)
```

- [ ] **Step 3d: `_score_candidate` slices the baseline**

In `_score_candidate`, after `results = res.training`, replace the two uses of `self.state.baseline.training`:

```python
        results = res.training
        base_tr = select(self.state.baseline.training, self._focus()[0])
        try:
            delta = bundle_delta(base_tr, results)
        ...
        wins = beats(base_tr, results, self.rule)
```

- [ ] **Step 3e: `_confirm` compares on the request's held-out sets**

Replace the `try:` block that computes `won` and `holdout_delta`:

```python
        _, holdout_sets = self._focus()
        # The guard sets are TRAINING sets of the other tracks, so their baseline rows live in
        # baseline.training; select() picks the right rows from both lists by (track, nonce).
        base_ho = select(self.state.baseline.holdout + self.state.baseline.training, holdout_sets)
        try:
            if self.spec.track is None:
                won = beats(base_ho, ho, self.rule)
            else:
                won = beats_focused(base_ho, ho, self.rule, self.spec.track)
            holdout_delta = bundle_delta(base_ho, ho).to_dict()
```

- [ ] **Step 4: Run the whole loop suite**

Run: `.venv/bin/pytest tests/test_loop.py -q`
Expected: all pass, including every pre-existing test.

- [ ] **Step 5: Commit**

```bash
git add talos/loop.py tests/test_loop.py talos/prompts.py
git commit -m "loop: focused requests, guard confirmation, track in the prompt context"
```

(Include `talos/prompts.py` only if the note in Step 3c applied.)

---

### Task 4: Prompts and the agentic instructions name the track

**Files:**
- Modify: `talos/prompts.py:33-77` (`PromptContext`, `hypothesis_prompts`), `:78-87` (`edit_prompts`), `talos/agentic.py:55-81` (`claude_md`)
- Test: `tests/test_prompts.py`, `tests/test_agentic.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `PromptContext.track: str | None = None`, `PromptContext.guard_tracks: list[str] = []`; a module-level `focus_sentence(ctx) -> str` in `talos/prompts.py` (empty string when unfocused) that `agentic.claude_md` reuses.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_prompts.py`:

```python
def test_focused_prompts_name_the_track_and_the_guards():
    # mutation: dropping the track from the system prompt leaves the model optimising every
    # track; dropping the guard list hides that the other tracks are re-scored
    c = ctx(track="n_items=5000,budget=25", guard_tracks=["n_items=1000,budget=5"])
    system, user = hypothesis_prompts(c)
    assert "n_items=5000,budget=25" in system and "n_items=1000,budget=5" in system
    assert "regression guard" in system
    system2, user2 = edit_prompts(c, {"title": "t", "description": "d"})
    assert "n_items=5000,budget=25" in (system2 + user2)


def test_unfocused_prompts_do_not_mention_a_guard():
    # mutation: emitting the focus sentence with an empty track changes every existing prompt
    system, user = hypothesis_prompts(ctx())
    assert "regression guard" not in system and "across every active track" in system
```

Append to `tests/test_agentic.py`:

```python
def test_claude_md_names_the_focus_track():
    # mutation: the agentic goal line still says "every active track" for a focused job
    c = ctx()
    c.track, c.guard_tracks = "n=1", ["n=2"]
    text = claude_md(c)
    assert "n=1" in text and "n=2" in text and "regression guard" in text
    assert "regression guard" not in claude_md(ctx())
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_prompts.py tests/test_agentic.py -k 'focus or guard' -q`
Expected: FAIL. If Task 3 already added the fields, the failures are on the `in system` asserts; otherwise `TypeError` on the unknown keyword.

- [ ] **Step 3: Implement**

In `talos/prompts.py`, add the fields to `PromptContext` after `is_gpu: bool = False` (skip if Task 3 did it):

```python
    track: str | None = None
    guard_tracks: list[str] = field(default_factory=list)
```

Add after `_files_block`:

```python
def focus_sentence(ctx: PromptContext) -> str:
    """The target sentence for a focused job; empty when the job optimises every track."""
    if ctx.track is None:
        return ""
    guards = ", ".join(ctx.guard_tracks) or "none"
    return (f"Optimise for track \"{ctx.track}\" only. The other active tracks ({guards}) are "
            f"re-scored as a regression guard when a candidate wins, and none of them may get "
            f"worse: confine changes to the code path that serves \"{ctx.track}\".")
```

In `hypothesis_prompts`, compute two strings first and interpolate them; do not mix implicit
string concatenation with `+` inside the existing parenthesised literal:

```python
    scope = f"on track \"{ctx.track}\"" if ctx.track else "across every active track"
    focus = (focus_sentence(ctx) + "\n\n") if ctx.track else ""
    system = (
        f"You are a research engineer improving a Rust solver for the TIG challenge "
        f"\"{ctx.challenge}\". The goal is to beat the current mainnet state of the art on "
        f"TIG's own benchmark: higher verifier quality per nonce under a fixed fuel budget, "
        f"{scope}.\n\n{focus}"
        f"The solver must keep this contract (template.rs):\n```rust\n{ctx.template_rs}\n```\n\n"
        f"Propose ONE specific change. Reply with a JSON object with keys \"title\" "
        f"(short), \"description\" (what to change and why it should raise quality), and "
        f"\"strategy_tag\" (one of: {', '.join(STRATEGY_TAGS)}). No other text."
    )
```

In `edit_prompts`, after the hypothesis description line in `user`:

```python
    focus = focus_sentence(ctx)
    user = (f"Implement this hypothesis:\nTitle: {hypothesis['title']}\n"
            f"Description: {hypothesis['description']}\n\n"
            + (focus + "\n\n" if focus else "")
            + f"Current algorithm source files:\n{_files_block(ctx.files)}")
```

In `talos/agentic.py::claude_md`, import `focus_sentence` from `talos.prompts` and change the goal sentence:

```python
    scope = f'on track "{ctx.track}"' if ctx.track else "on every active track"
    focus = (focus_sentence(ctx) + "\n\n") if ctx.track else ""
    return f"""# Talos agentic iteration: {ctx.challenge}

You are improving a Rust solver for the TIG challenge "{ctx.challenge}". Beat the mainnet
baseline "{ctx.baseline_name}" on TIG's benchmark (higher verifier quality per nonce under a
fixed fuel budget {scope}). Your current best is {ctx.best_delta:+.3%} vs baseline.

{focus}Rules:
```

(keep the rest of the template as it is).

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/pytest tests/test_prompts.py tests/test_agentic.py tests/test_loop.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add talos/prompts.py talos/agentic.py tests/test_prompts.py tests/test_agentic.py
git commit -m "prompts: name the focus track and the regression guard"
```

---

### Task 5: `--track` on the CLI

**Files:**
- Modify: `talos/cli.py:362-520` (`cmd_run`), `:557-600` (`main` argparse)
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `JobSpec.track` (Task 1).
- Produces: `talos run --track <name>`; wizard prompt `Track to optimise (all, or one of: ...)`.

- [ ] **Step 1: Update the one wizard test that pins the last prompt**

In `tests/test_cli.py::test_wizard_labels_gpu_challenges_asks_mode_and_survives_a_typo`, the answers list gains one trailing answer and the last-prompt assertion moves:

```python
    answers = iter(["knapsack", "go", "abc", "3", "4", "5", "agentic", ""])
    ...
    assert prompts[-2] == "Mode (single-shot or agentic)"
    assert prompts[-1] == "Track to optimise (all, or one of: n=1)"
    assert seen["spec"].track is None
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_cli.py`:

```python
def test_run_track_flag_lands_in_the_spec(tmp_path, monkeypatch):
    # mutation: parsing the flag but not storing it makes every focused job an all-tracks job
    monkeypatch.chdir(tmp_path)
    fake_config(tmp_path)
    monkeypatch.setattr(cli, "fetch_challenge_info", lambda name: knapsack_info())
    seen = {}
    monkeypatch.setattr(cli, "execute_job",
                        lambda spec, store, cfg, resume: seen.update(spec=spec) or 0)
    rc = cli.main(["run", "--challenge", "knapsack", "--direction", "go",
                   "--budget-iterations", "3", "--yes", "--track", "n=1"])
    assert rc == 0 and seen["spec"].track == "n=1"
    # the nonce draw is unchanged: every track is still in the spec, so the baseline cache key is
    assert [s.track for s in seen["spec"].training] == ["n=1"]


def test_run_rejects_a_track_mainnet_does_not_have(tmp_path, monkeypatch, capsys):
    # mutation: skipping validation starts a job whose focus set is empty
    monkeypatch.chdir(tmp_path)
    fake_config(tmp_path)
    monkeypatch.setattr(cli, "fetch_challenge_info", lambda name: knapsack_info())
    monkeypatch.setattr(cli, "execute_job",
                        lambda spec, store, cfg, resume: pytest.fail("must not start a job"))
    rc = cli.main(["run", "--challenge", "knapsack", "--direction", "go",
                   "--budget-iterations", "3", "--yes", "--track", "n=9"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "n=9" in err and "n=1" in err
    assert not (tmp_path / "runs").exists()  # validation runs before the run directory exists


def test_resume_with_a_different_track_is_refused(tmp_path, monkeypatch, capsys):
    # mutation: letting --track through on resume would score a job on sets its baseline
    # comparison was never built for
    monkeypatch.chdir(tmp_path)
    fake_config(tmp_path)
    assert fake_run(monkeypatch, ["--track", "n=1"]) == 0
    run_dir = next((tmp_path / "runs").glob("*/job.json")).parent
    assert cli.main(["run", "--resume", run_dir.name, "--track", "n=2"]) == 2
    assert "started with track" in capsys.readouterr().err


def test_fake_run_with_a_track_wins_and_packages_per_track(tmp_path, monkeypatch, capsys):
    # mutation: the whole focused path; the fake challenge has one track, so there is no guard
    monkeypatch.chdir(tmp_path)
    fake_config(tmp_path)
    rc = fake_run(monkeypatch, ["--track", "n=1"])
    out = capsys.readouterr().out
    assert rc == 0 and "Status: won" in out
    assert "track n=1 of 1 tracks" in out
    run_dir = next((tmp_path / "runs").glob("*/job.json")).parent
    scores = (run_dir / "package" / "scores.md").read_text()
    assert "# Training nonces (track n=1)" in scores
    assert "# Held-out nonces (track n=1)" in scores
    assert "Regression guard" not in scores
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_cli.py -k 'track or wizard_labels' -q`
Expected: FAIL with `error: unrecognized arguments: --track n=1` (argparse exits 2, but the spec assertion and the wizard prompt assertion fail).

- [ ] **Step 4: Implement**

In `main`, after `r.add_argument("--direction-file")`:

```python
    r.add_argument("--track", help="one active track to optimise; default all tracks")
```

In `cmd_run`, resume branch, after the `--mode` check:

```python
        if args.track and args.track != spec.track:
            print(f"job {spec.job_id} was started with track {spec.track or 'all'}; start a new "
                  f"job to change track", file=sys.stderr)
            return 2
```

In `cmd_run`, after `info` is known (after the `challenge table drift` check, before `rand_hash = new_rand_hash()`):

```python
    track = args.track
    if track is None and not args.yes:
        answer = ask(f"Track to optimise (all, or one of: {', '.join(info.tracks)})", "all")
        track = None if answer.strip() in ("", "all") else answer.strip()
    if track is not None and track not in info.tracks:
        print(f"unknown track {track!r} for {challenge}; active tracks: "
              f"{', '.join(info.tracks)}", file=sys.stderr)
        return 2
```

Pass it to the spec: `..., challenge_id=info.id, track=track)`.

Replace the job start line:

```python
    scope = f"track {track} of {len(info.tracks)} tracks" if track else f"{len(info.tracks)} tracks"
    print(f"Job {job_id}: {scope}, fuel {info.max_fuel}, budget {budget.to_dict()}")
```

- [ ] **Step 5: Run the CLI suite**

Run: `.venv/bin/pytest tests/test_cli.py -q`
Expected: all pass. The per-track package headings in the last test need Task 6; if Task 6 is not done yet, that single test fails on the headings and passes after Task 6.

- [ ] **Step 6: Commit**

```bash
git add talos/cli.py tests/test_cli.py
git commit -m "cli: --track picks one active track to optimise"
```

---

### Task 6: The package reports per track

**Files:**
- Modify: `talos/package.py:14-33` (`scores_markdown` callers), `:47-71` (`evidence_draft`), `:83-114` (`_won_training`, `_readme`), `:131-149` (`build_package`)
- Test: `tests/test_package.py`

**Interfaces:**
- Consumes: `select`, `focus_sets` (Task 2); `JobSpec.track` (Task 1).
- Produces: `scores_sections(spec, state) -> list[tuple[str, str]]` (heading, table) used by both `scores.md` and the evidence appendix.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_package.py` (add `from dataclasses import replace` to the imports):

```python
def make_focused(tmp_path):
    """Two tracks t and u; the job focuses t; the candidate's held-out carries t's held-out rows
    and u's training rows (the guard), exactly as the loop stores them."""
    spec, st, store = make(tmp_path)
    spec = replace(spec, tracks=["t", "u"], track="t",
                   training=[NonceSet("t", HASH, 0, 2), NonceSet("u", HASH, 0, 2)],
                   holdout=[NonceSet("t", HASH, 1_000_000, 2), NonceSet("u", HASH, 1_000_000, 2)])
    st.baseline.training += [NonceResult("u", 0, True, 50, 1), NonceResult("u", 1, True, 50, 1)]
    st.baseline.holdout += [NonceResult("u", 1_000_000, True, 50, 1),
                            NonceResult("u", 1_000_001, True, 50, 1)]
    st.best.holdout += [NonceResult("u", 0, True, 50, 1), NonceResult("u", 1, True, 49, 1)]
    (tmp_path / "job.json").unlink()
    store.write_spec(spec)
    return spec, st, store


def test_focused_package_splits_scores_into_track_and_guard_sections(tmp_path):
    # mutation: comparing the whole baseline against the single-track candidate raises a
    # ScoringError and prints "(no bundle delta" instead of the delta; dropping the guard
    # section hides the u regression the confirmation saw
    spec, st, store = make_focused(tmp_path)
    pkg = build_package(spec, st, store)
    scores = (pkg / "scores.md").read_text()
    assert "# Training nonces (track t)" in scores
    assert "# Held-out nonces (track t)" in scores
    assert "# Regression guard (other tracks, training nonces)" in scores
    assert "(no bundle delta" not in scores
    guard = scores.split("# Regression guard")[1]
    assert "| u | 1 | 50 | 49 | - |" in guard
    readme = (pkg / "README.md").read_text()
    assert "Optimised for track t" in readme and "regression guard" in readme
    evidence = (pkg / "evidence_draft.md").read_text()
    assert "Optimised for track t" in evidence and "Regression guard" in evidence


def test_unfocused_package_keeps_todays_headings(tmp_path):
    # mutation: the focus headings leaking into unfocused packages
    spec, st, store = make(tmp_path)
    scores = (build_package(spec, st, store) / "scores.md").read_text()
    assert "# Training nonces\n" in scores and "(track" not in scores
    assert "Regression guard" not in scores
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_package.py -k 'focused or todays' -q`
Expected: the focused test fails on the `(track t)` heading; the unfocused test passes already (it guards the refactor).

- [ ] **Step 3: Implement**

In `talos/package.py`, change the scoring import to
`from talos.scoring import ScoringError, beats, bundle_delta, focus_sets, select` and add after `_diff`:

```python
def scores_sections(spec: JobSpec, state: JobState) -> list[tuple[str, str]]:
    """(heading, table) pairs for scores.md and the evidence appendix. Focused jobs get the
    track's training and held-out tables plus a guard table for the other tracks; unfocused
    jobs keep the two plain headings."""
    base, best = state.baseline, state.best
    training_sets, holdout_sets = focus_sets(spec.track, spec.training, spec.holdout)
    if spec.track is None:
        out = [("Training nonces", scores_markdown(base.training, best.training))]
        if best.holdout:
            out.append(("Held-out nonces", scores_markdown(base.holdout, best.holdout)))
        return out
    t = spec.track
    out = [(f"Training nonces (track {t})",
            scores_markdown(select(base.training, training_sets), best.training))]
    if best.holdout:
        focus_ho = [s for s in holdout_sets if s.track == t]
        guards = [s for s in holdout_sets if s.track != t]
        out.append((f"Held-out nonces (track {t})",
                    scores_markdown(select(base.holdout, focus_ho), select(best.holdout, focus_ho))))
        if guards:
            out.append(("Regression guard (other tracks, training nonces)",
                        scores_markdown(select(base.training, guards),
                                        select(best.holdout, guards))))
    return out
```

In `evidence_draft`, replace the `bench` block:

```python
    bench = ""
    if state.best and state.baseline:
        focus = f" Optimised for track {spec.track}." if spec.track else ""
        bench = ("\n\n## TALOS BENCHMARK APPENDIX (auto-generated)\n\n"
                 f"Baseline: mainnet `{state.baseline.name}` at monorepo `{spec.monorepo_ref}`, "
                 f"fuel {spec.fuel}, tracks {', '.join(spec.tracks)}.{focus}\n")
        for heading, table in scores_sections(spec, state):
            bench += f"\n### {heading}\n\n{table}"
    return filled + bench
```

In `_won_training`, slice the baseline:

```python
def _won_training(spec: JobSpec, state: JobState) -> bool:
    try:
        training_sets, _ = focus_sets(spec.track, spec.training, spec.holdout)
        return beats(select(state.baseline.training, training_sets), state.best.training,
                     CHALLENGES[spec.challenge].beat)
    except (ScoringError, KeyError):
        return False
```

In `_readme`, after the `head = (...)` challenge line and before the `if confirmed:` branch:

```python
    if spec.track:
        head += (f"Optimised for track {spec.track}; the other tracks were re-scored on "
                 "confirmation as a regression guard (see the last section of scores.md).\n\n")
```

In `build_package`, replace the `scores = ...` lines:

```python
    scores = "\n".join(f"# {heading}\n\n{table}" for heading, table in scores_sections(spec, state))
    (pkg / "scores.md").write_text(scores)
```

Check the unfocused output is byte-identical to before: the old code joined `"# Training nonces\n\n" + table` and, with held-out, `"\n# Held-out nonces\n\n" + table`; `scores_markdown` ends with `\n`, so `"\n".join` reproduces the same text. The unfocused test in Step 1 pins the headings; `test_package_contents_and_no_hash` pins the rest.

- [ ] **Step 4: Run the package and CLI suites**

Run: `.venv/bin/pytest tests/test_package.py tests/test_cli.py -q`
Expected: all pass, including Task 5's fake focused run.

- [ ] **Step 5: Commit**

```bash
git add talos/package.py tests/test_package.py
git commit -m "package: per-track score sections and the regression guard table"
```

---

### Task 7: Documentation and the gate

**Files:**
- Modify: `README.md:77-88` (run flags), `:140-160` (where results land), `:165-176` (budget); `AGENTS.md:44-52` (invariant 1)
- Test: `tests/test_docs_references.py` (existing; checks that code paths named in the authoritative docs exist)

- [ ] **Step 1: README run flags**

After the `--mode` bullet add:

```markdown
- `--track <name>` — one active track of the challenge to optimise (the interactive prompt
  lists them; default all). Training scores that track only. When a candidate wins on
  training, the confirmation job scores the track's held-out nonces plus every other track's
  training nonces as a regression guard: no other track may get worse. The model still sees
  and may edit every file; the flag narrows what is scored and what it is told to target.
```

- [ ] **Step 2: README results and budget**

In "Where results land", extend the `scores.md` parenthetical: `scores.md` (per-nonce tables for baseline and candidate on training and held-out nonces; a focused job adds a regression-guard table for the other tracks).

In "Budget", after the sentence about C3 per-job overhead, add: `A focused job's confirmation scores the other tracks' training nonces too, about a minute more per winning iteration on C3.`

- [ ] **Step 3: AGENTS.md invariant 1**

Append to invariant 1, after the cache-key sentence:

```markdown
   With `--track`, the loop slices both sides with the same `NonceSet` lists
   (`talos/scoring.py::select`, `talos/scoring.py::focus_sets`); the guard compares
   the other tracks' training nonces against the cached baseline training results
   for those same nonces, and the confirmation rule is
   `talos/scoring.py::beats_focused`.
```

- [ ] **Step 4: Run the gate**

From a 3.11+ venv (see Tech Stack):

```bash
make check PYTHON=<scratch>/venv311/bin/python
```

Expected: ruff `All checks passed!`, pytest all passed with 2 deselected (the live tests), agentify check exit 0. Paste the three result lines into the task report.

- [ ] **Step 5: Line-length check on every touched file**

```bash
.venv/bin/python -c "
import subprocess
files = subprocess.run(['git','diff','--name-only','main'], capture_output=True, text=True).stdout.split()
for f in files:
    if f.endswith('.py'):
        for i, l in enumerate(open(f), 1):
            if len(l.rstrip('\n')) > 100: print(f, i, len(l))
print('done')"
```

Expected: only `done` plus the pre-existing long lines listed in the codex-model-list branch (none of them in code this plan adds).

- [ ] **Step 6: Commit**

```bash
git add README.md AGENTS.md
git commit -m "docs: --track, the regression guard, and the focused scores sections"
```

---

## Self-review

**Spec coverage.** §3 → Task 1. §4, §6 rule → Task 2. §5, §6 confirmation → Task 3. §7 → Task 4. §8 → Task 5. §9 → Task 6. §10, §12 → Task 7. §11 test table: every row maps to a test above (state round trip T1; select, focused rule, identity T2; request shape, guard false positive, unfocused request T3; validation, resume refusal, fake end-to-end T5; prompts T4; package T6).

**Placeholder scan.** No TBD/TODO. Every code step shows the code. "rest unchanged" appears only where the surrounding lines are quoted verbatim from the current file.

**Type consistency.** `focus_sets(track, training, holdout)` is called with that argument order in Tasks 3 and 6. `select(results, sets)` everywhere. `beats_focused(baseline, candidate, rule, track)` in Tasks 2 and 3. `PromptContext.track` / `.guard_tracks` in Tasks 3 and 4. `scores_sections(spec, state)` only inside Task 6.

**Ordering.** Tasks 1 and 2 first (independent). Task 3 depends on 1 and 2 and on the two `PromptContext` fields; the note in Task 3 Step 3c makes it self-sufficient if Task 4 has not run. Task 5's last test needs Task 6. Task 7 last.
