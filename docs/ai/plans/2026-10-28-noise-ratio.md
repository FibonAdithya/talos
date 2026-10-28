# Noise Ratio Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show the paired standard error and a noise ratio next to every scored delta (issue
#69). This is display only.

**Architecture:** `bundle_delta` computes a per-track SE and the SE of the mean from the same
substituted qualities it already uses. A single `noise()` function turns a `BundleDelta`
plus the beat rule into `(ratio, band, state)`. The loop records those values in the
`scored` event and the hypothesis record. The status line, `scores.md` and `report.html`
only format what was recorded (or, for tables built from rows, call `noise()`).

**Tech Stack:** Python 3.11 standard library (`math`, `statistics`), pytest. No new
dependencies.

**Spec:** `docs/ai/specs/2026-10-28-noise-ratio-design.md`. Read it first; this plan does
not repeat its reasoning.

**Worktree:** `/home/fibonadithya/TIG/Talos-noise-ratio`, branch `noise-ratio-69` from
`origin/main` at 3357e3e. Run every command from the worktree root.

## Global Constraints

- No change to any decision: `beats`, `improves`, `clears`, `beats_focused`,
  `holdout_decision`, `validation_failure` and `BeatRule` behave exactly as now.
- `talos/scoring.py` may import only from the standard library. It runs inside the C3 and
  local job containers.
- `talos/prompts.py` is not touched. `talos status --json` and `talos show --json` output
  is unchanged.
- `evidence_draft` output is byte-identical to today's.
- The bands are `clear` (ratio ≥ 2.0), `marginal` (≥ 1.0 and < 2.0) and `noise` (< 1.0). The
  states are `ok`, `n1`, `identical`, `no_spread` and `over_ceiling`. These strings are
  exact.
- Every reader of the new keys uses `.get`, because state.json and timeline.jsonl files
  written before this change lack them.
- Tests run with `.venv/bin/python -m pytest -m "not live"`. Never run bare `pytest`: an
  unmarked live test bills Modal.
- Done means `make check PYTHON=.venv/bin/python` passes.

## Review Focus

1. **A demoted native candidate.** Its record keeps the native noise fields, and its node
   score is native, so the two agree. Pinned in Task 2
   (`test_a_demoted_record_keeps_the_native_noise`).
2. **A narrow terminal.** `_one_line` truncates the line, and the noise text must survive
   because it comes first. Pinned in Task 3
   (`test_event_line_scored_noise_survives_truncation`).
3. **A focused job (`--track t`).** The training delta covers only the focus track, so
   `track_rel_se` names only that track. Pinned in Task 2
   (`test_focused_scored_event_has_the_focus_track_se_only`).
4. **A whisker wider than the axis.** It is clipped to the plot area and never drawn outside
   it. Pinned in Task 5 (`test_whisker_is_clipped_to_the_plot`).
5. **An old run.** It has hypothesis records and a `best.delta` without the new keys, and the
   report and package must still build. Pinned in Task 4
   (`test_package_builds_from_a_state_without_noise_keys`) and Task 5
   (`test_old_records_draw_no_whisker`).

---

### Task 1: SE and `noise()` in scoring

**Files:**
- Modify: `talos/scoring.py` (`TrackDelta`, `BundleDelta`, `bundle_delta`; add `noise`)
- Test: `tests/test_scoring.py`

**Interfaces:**
- Produces:
  - `TrackDelta.rel_se: float | None = None`: the last field, with a default.
  - `BundleDelta.mean_rel_se: float | None = None`: the last field, with a default.
  - `NOISE_CLEAR = 2.0`, `NOISE_MARGINAL = 1.0`.
  - `noise(d: BundleDelta, rule: BeatRule) -> tuple[float | None, str | None, str]`
    returns `(ratio, band, state)`.

- [ ] **Step 1: Write the failing tests** in `tests/test_scoring.py`. Add `noise` and
  `TrackDelta`/`BundleDelta` to the import. Use the existing `R` helper.

```python
def test_paired_se_matches_hand_computation_on_two_tracks():
    # spec Tests 1. mutations: population stdev; dividing by the candidate mean; summing
    # track SEs instead of root-sum-of-squares; dividing the combined SE by sqrt(T)
    base = [R("A", i, 100) for i in range(4)] + [R("B", 0, 200), R("B", 1, 200)]
    cand = [R("A", 0, 101), R("A", 1, 103), R("A", 2, 99), R("A", 3, 105),
            R("B", 0, 200), R("B", 1, 204)]
    d = bundle_delta(base, cand)
    by = {t.track: t for t in d.tracks}
    assert by["A"].rel_se == pytest.approx(math.sqrt(5 / 3) / 100, rel=1e-9)
    assert by["B"].rel_se == pytest.approx(0.01, rel=1e-9)
    assert d.mean_rel_se == pytest.approx(math.sqrt(2 / 3) / 100, rel=1e-9)
    ratio, band, state = noise(d, BeatRule())
    assert ratio == pytest.approx(1.5 / math.sqrt(2 / 3), rel=1e-9)
    assert (band, state) == ("marginal", "ok")
    assert d.to_dict()["mean_rel_se"] == d.mean_rel_se
    assert d.to_dict()["tracks"][0]["rel_se"] == by["A"].rel_se


def test_se_is_none_when_any_track_has_one_nonce():
    # spec Tests 2. mutation: averaging over the tracks that have an SE hides the n=1 track
    d = bundle_delta([R("a", 0, 100)], [R("a", 0, 102)])
    assert d.tracks[0].rel_se is None and d.mean_rel_se is None
    assert noise(d, BeatRule()) == (None, None, "n1")
    base = [R("a", 0, 100)] + [R("b", i, 100) for i in range(4)]
    cand = [R("a", 0, 102)] + [R("b", 0, 100), R("b", 1, 102), R("b", 2, 100), R("b", 3, 102)]
    d = bundle_delta(base, cand)
    by = {t.track: t for t in d.tracks}
    assert by["a"].rel_se is None and by["b"].rel_se is not None
    assert d.mean_rel_se is None and noise(d, BeatRule())[2] == "n1"


def test_se_scores_a_failed_nonce_at_the_worst_quality():
    # spec Tests 3. mutation: leaving the failed nonce out of the differences gives [2, 0]
    base = [R("t", 0, 10), R("t", 1, 20), R("t", 2, 30)]
    cand = [R("t", 0, 12), R("t", 1, None, "panic"), R("t", 2, 30)]
    d = bundle_delta(base, cand)
    assert d.tracks[0].rel_delta == pytest.approx(-2 / 15, rel=1e-9)
    assert d.tracks[0].rel_se == pytest.approx(0.1855921454276674, rel=1e-9)
    assert noise(d, BeatRule(error_ceiling=0.34))[2] == "ok"


def test_noise_is_withheld_over_the_error_ceiling():
    # spec Tests 4 and 10. mutation: checking n=1 before the ceiling reports "n1" for a
    # broken one-nonce candidate
    base = [R("t", 0, 10), R("t", 1, 20), R("t", 2, 30)]
    cand = [R("t", 0, 12), R("t", 1, None, "panic"), R("t", 2, 30)]
    d = bundle_delta(base, cand)
    assert noise(d, BeatRule(error_ceiling=0.05)) == (None, None, "over_ceiling")
    assert d.mean_rel_se is not None
    one = bundle_delta([R("t", 0, 100)], [R("t", 0, None, "panic")])
    assert noise(one, BeatRule())[2] == "over_ceiling"


def test_zero_spread_states():
    # spec Tests 5. mutation: dividing by a zero SE raises ZeroDivisionError or returns inf
    same = bundle_delta(BASE, list(BASE))
    assert same.mean_rel_se == 0.0
    assert noise(same, BeatRule()) == (None, None, "identical")
    plus = [R(r.track, r.nonce, r.quality + 1) for r in BASE]
    assert noise(bundle_delta(BASE, plus), BeatRule()) == (None, None, "no_spread")


def test_se_pairs_rows_by_nonce_not_by_input_order():
    # The bench may return rows in another order than the baseline's. mutation: pairing
    # b[track].values() with c[track].values() in dict order. Reversed, that pairs
    # [100, 110, 120, 130] with [135, 119, 113, 101]: differences [35, 9, -7, -29], a far
    # larger stdev than the true [1, 3, -1, 5]
    base = [R("A", 0, 100), R("A", 1, 110), R("A", 2, 120), R("A", 3, 130)]
    cand = [R("A", 0, 101), R("A", 1, 113), R("A", 2, 119), R("A", 3, 135)]
    fwd = bundle_delta(base, cand)
    rev = bundle_delta(base, list(reversed(cand)))
    # differences [1, 3, -1, 5] in either order: stdev sqrt(20/3), mean base 115
    expected = math.sqrt(20 / 3) / 2 / 115
    assert fwd.tracks[0].rel_se == pytest.approx(expected, rel=1e-9)
    assert rev.tracks[0].rel_se == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("delta,band", [(2.0, "clear"), (1.999, "marginal"),
                                        (1.0, "marginal"), (0.999, "noise"),
                                        (-2.0, "clear")])
def test_noise_band_boundaries(delta, band):
    # spec Tests 6. mutations: > for >= at either boundary; the band using the signed delta
    d = BundleDelta(tracks=[TrackDelta("t", 1.0, 1.0, delta, 0, 4, rel_se=1.0)],
                    mean_rel_delta=delta, worst_rel_delta=delta, error_rate=0.0,
                    mean_rel_se=1.0)
    ratio, got, state = noise(d, BeatRule())
    assert (ratio, got, state) == (abs(delta), band, "ok")
```

- [ ] **Step 2: Run them and confirm they fail.**
  Run `.venv/bin/python -m pytest tests/test_scoring.py -q -k "se_ or noise or zero_spread"`.
  Expected: an ImportError on `noise`.

- [ ] **Step 3: Implement.** In `bundle_delta`, per track, take the paired differences
  `cq[i] − bq[i]` in nonce order. If `n ≥ 2`, `rel_se = statistics.stdev(diffs) /
  math.sqrt(n) / bm`; otherwise `None`. `mean_rel_se` is `None` if any track's `rel_se` is
  `None`; otherwise it is `math.sqrt(sum(se²)) / T`.
  - `bq` and `cq` are built from `.values()` of two dicts. Pair them by nonce key, not by
    dict order: build both lists from `sorted(b[track])`.
  - `noise()` checks, in order:
    1. `d.error_rate > rule.error_ceiling` → `over_ceiling`
    2. `mean_rel_se is None` → `n1`
    3. `mean_rel_se == 0` → `identical` if every `t.rel_delta == 0`, else `no_spread`
    4. otherwise `ok`, with `ratio = abs(mean_rel_delta) / mean_rel_se` and the band from
       the two constants.

- [ ] **Step 4: Run the whole scoring file and confirm it passes.**
  Run `.venv/bin/python -m pytest tests/test_scoring.py -q`. Expected: all pass, the existing
  tests included (decisions unchanged, spec Tests 8).

- [ ] **Step 5: Commit**

```bash
git add talos/scoring.py tests/test_scoring.py
git commit -m "Paired SE and noise ratio for every bundle delta (#69)"
```

---

### Task 2: Record the noise figures in the loop

**Files:**
- Modify: `talos/loop.py`: the `scored` event block (around line 700–717) and
  `Loop._validate_and_finish` (the `record.update(outcome="improved", ...)` line, around 793)
- Test: `tests/test_loop.py`

**Interfaces:**
- Consumes: `noise`, `BundleDelta.mean_rel_se`, `TrackDelta.rel_se` (Task 1).
- Produces:
  - `Loop._noise_fields(self, d: BundleDelta) -> dict`, with keys `mean_rel_se`,
    `noise_ratio`, `noise_band` and `noise_state`.
  - The `scored` event additionally carries those four keys plus
    `track_rel_se: dict[str, float | None]`.
  - The hypothesis record carries the four keys. After a passed validation they hold the
    metered values.

- [ ] **Step 1: Write the failing tests** in `tests/test_loop.py`. Use the existing `make`,
  `native`, `hyp`, `edit`, `quality_from_files` and `FakeBench`. Read the scored events from
  `timeline.jsonl` the way `test_a_native_score_carries_no_runtime_ratio_against_the_metered_baseline`
  does.

```python
ONE = Budget(usd=None, hours=None, iterations=1, compute_usd=None)

def _scored_events(store):
    rows = [json.loads(x) for x in (store.run_dir / "timeline.jsonl").read_text().splitlines()]
    return [r for r in rows if r["kind"] == "scored"]


def test_scored_event_and_record_carry_the_noise_figures(tmp_path):
    # baseline 100 on nonces 0..3; candidate [100, 102, 100, 102]: differences [0, 2, 0, 2],
    # delta 1%, SE sqrt(1/3)%, ratio sqrt(3). mutation: not passing the fields to _event
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(2)], budget=ONE,
                               scores=lambda ch, files, ns: [100 + 2 * (n % 2)
                                                             for n in ns.nonces()])
    st = loop.run()
    ev = _scored_events(store)[0]
    se = math.sqrt(1 / 3) / 100
    assert ev["mean_rel_se"] == pytest.approx(se, rel=1e-9)
    assert ev["noise_ratio"] == pytest.approx(math.sqrt(3), rel=1e-9)
    assert (ev["noise_band"], ev["noise_state"]) == ("marginal", "ok")
    assert ev["track_rel_se"] == {"t": pytest.approx(se, rel=1e-9)}
    rec = st.hypotheses[0]
    assert rec["mean_rel_se"] == pytest.approx(se, rel=1e-9)
    assert (rec["noise_band"], rec["noise_state"]) == ("marginal", "ok")


def test_over_ceiling_scored_event_withholds_the_ratio(tmp_path):
    # one of four nonces fails: 25% > the 5% ceiling. mutation: calling noise() without the
    # rule's ceiling, or computing the band in the loop
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(2)], budget=ONE,
                               scores=lambda ch, files, ns: [None] + [101] * (ns.count - 1))
    loop.run()
    ev = _scored_events(store)[0]
    assert ev["noise_state"] == "over_ceiling"
    assert ev["noise_ratio"] is None and ev["noise_band"] is None
    assert ev["mean_rel_se"] is not None


def test_a_validated_record_carries_the_metered_noise(tmp_path):
    # spec Tests 11. native: 104 on every nonce, so zero spread. metered: [error, 104, 104, 104];
    # the error scores at the worst observed (100), so the differences are [0, 4, 4, 4]: delta 3%,
    # SE 1%, ratio 3, error rate 0.25, which is at the ceiling and not over it.
    # mutation: no overwrite in _validate_and_finish leaves 0.0 / no_spread in the record
    from talos.challenges import CHALLENGES

    def metered(ch, files, ns):
        q = quality_from_files(ch, files, ns)
        return [None] + q[1:] if "let k = 5;" in files["mod.rs"] else q
    fb = FakeBench(quality_from_files, metered_scores=metered)
    loop, fp, _, store = native(tmp_path, [hyp("a"), edit(5)], bench=fb, budget=ONE)
    loop.rule = replace(CHALLENGES["knapsack"].beat, error_ceiling=0.25)
    st = loop.run()
    assert st.validations[0]["outcome"] == "validated"
    rec = st.hypotheses[0]
    assert rec["mean_rel_se"] == pytest.approx(0.01, rel=1e-9)
    assert rec["noise_ratio"] == pytest.approx(3.0, rel=1e-9)
    assert (rec["noise_band"], rec["noise_state"]) == ("clear", "ok")
    ev = _scored_events(store)[0]
    assert ev["mean_rel_se"] == 0.0 and ev["noise_state"] == "no_spread"


def test_a_demoted_record_keeps_the_native_noise(tmp_path):
    # Review Focus 1. Metered fails one nonce of four: 25% is over the default 5% ceiling, so
    # validation demotes it as "error_ceiling" and, unlike "nondeterministic", returns a
    # metered delta. mutation: overwriting the noise fields from that delta before the
    # demotion return records "over_ceiling" instead of the native "no_spread"
    def metered(ch, files, ns):
        q = quality_from_files(ch, files, ns)
        return [None] + q[1:] if "let k = 5;" in files["mod.rs"] else q
    fb = FakeBench(quality_from_files, metered_scores=metered)
    loop, fp, _, store = native(tmp_path, [hyp("a"), edit(5)], bench=fb, budget=ONE)
    st = loop.run()
    assert st.hypotheses[0]["outcome"] == "failed:validation"
    assert st.validations[0]["reason"] == "error_ceiling"
    ev = _scored_events(store)[0]
    assert ev["noise_state"] == "no_spread"
    for k in ("mean_rel_se", "noise_ratio", "noise_band", "noise_state"):
        assert st.hypotheses[0][k] == ev[k]


def test_focused_scored_event_has_the_focus_track_se_only(tmp_path):
    # Review Focus 3. mutation: computing the SE against the unselected baseline raises a
    # track mismatch, or lists track u
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(2)], budget=ONE, two_tracks=True,
                               track="t")
    loop.run()
    assert set(_scored_events(store)[0]["track_rel_se"]) == {"t"}
```

  Before relying on them, check three fixture facts in `tests/test_loop.py` and
  `talos/types.py`:
  - `NonceSet.nonces()` yields 0..3 for `TR`.
  - `FakeBench` turns a `None` score into a failed row.
  - The second test's `edit(2)` reaches scoring. Its compile is not rejected as dead code.

  Adjust the scores lambda if any of these does not hold, and say so in the report.

- [ ] **Step 2: Run them and confirm they fail.**
  Run `.venv/bin/python -m pytest tests/test_loop.py -q -k "noise or focus_track_se"`.
  Expected: KeyError on `mean_rel_se`.

- [ ] **Step 3: Implement.**
  - Add `_noise_fields(d)`, which calls `noise(d, self.rule)`.
  - At the scored step, `record.update(**fields)`. Pass `**fields` and
    `track_rel_se={t.track: t.rel_se for t in delta.tracks}` to `self._event("scored", ...)`.
  - In `_validate_and_finish`, add `**self._noise_fields(d)` to the
    `record.update(outcome="improved", ...)` call only. Do not add it to the demoted branch.

- [ ] **Step 4: Run the loop tests and confirm they pass.**
  Run `.venv/bin/python -m pytest tests/test_loop.py -q`. Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add talos/loop.py tests/test_loop.py
git commit -m "Record the noise figures with each scored candidate (#69)"
```

---

### Task 3: The status line

**Files:**
- Modify: `talos/cli.py::_event_line`, the `scored` branch (around line 580)
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: the `scored` event keys from Task 2.
- Produces: `_noise_text(data: dict) -> str | None` in `talos/cli.py`. It returns `None` for
  `over_ceiling` and for an event without `noise_state`.

- [ ] **Step 1: Write the failing tests** in `tests/test_cli.py`, next to
  `test_event_line_scored_prints_percentages`. Leave that test and
  `test_event_line_scored_says_how_many_nonces_were_not_run` unchanged. They are the
  old-event case (spec Tests 7).

```python
SCORED = {"mean_rel_delta": 0.004, "worst_rel_delta": -0.0012, "error_rate": 0.0,
          "runtime_ratio": 1.02}
TAIL = "worst track -0.120%, errors 0.0%, runtime x1.02)"


def test_event_line_scored_shows_the_noise_figure():
    # mutation: the ratio formatted with :.2f, or the band left out
    data = {**SCORED, "mean_rel_se": 0.0031, "noise_ratio": 1.29, "noise_band": "marginal",
            "noise_state": "ok"}
    assert cli._event_line("scored", data) == \
        "scored +0.400% vs baseline (±0.310%, 1.3x noise, marginal; " + TAIL


@pytest.mark.parametrize("state,se,text", [
    ("n1", None, "n=1"), ("identical", 0.0, "identical on every nonce"),
    ("no_spread", 0.0, "±0, no spread")])
def test_event_line_scored_names_the_undefined_cases(state, se, text):
    data = {**SCORED, "mean_rel_se": se, "noise_ratio": None, "noise_band": None,
            "noise_state": state}
    assert cli._event_line("scored", data) == \
        f"scored +0.400% vs baseline ({text}; " + TAIL


@pytest.mark.parametrize("se", [None, 0.0, 0.009])
def test_event_line_scored_over_the_ceiling_has_no_noise(se):
    # spec Tests 10. mutation: switching on the SE value instead of noise_state prints "n=1"
    # or "no spread" for a broken candidate
    data = {**SCORED, "mean_rel_se": se, "noise_ratio": None, "noise_band": None,
            "noise_state": "over_ceiling"}
    assert cli._event_line("scored", data) == "scored +0.400% vs baseline (" + TAIL


def test_event_line_scored_noise_survives_truncation():
    # Review Focus 2. mutation: appending the noise after the runtime puts it past the cut
    data = {**SCORED, "mean_rel_se": 0.0031, "noise_ratio": 1.29, "noise_band": "marginal",
            "noise_state": "ok"}
    # the noise text with its ";" is 58 characters; width 70 keeps 67 of them, then "..."
    line = cli._event_line("scored", data, width=70)
    assert line.startswith("scored +0.400% vs baseline (±0.310%, 1.3x noise, marginal;")
    assert len(line) == 70 and line.endswith("...")
```

- [ ] **Step 2: Run them and confirm they fail.**
  Run `.venv/bin/python -m pytest tests/test_cli.py -q -k event_line_scored`. Expected: the
  four new tests fail and the two old ones pass.

- [ ] **Step 3: Implement `_noise_text(data)`.**
  - `ok` gives `f"±{se:.3%}, {ratio:.1f}x noise, {band}"`.
  - The other states give the strings in the second test.
  - In the `scored` branch, prefix `f"{noise}; "` inside the parenthesis when `_noise_text`
    returns a string.

- [ ] **Step 4: Run the CLI tests and confirm they pass.**
  Run `.venv/bin/python -m pytest tests/test_cli.py -q`. Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add talos/cli.py tests/test_cli.py
git commit -m "Show the noise figure on the scored status line (#69)"
```

---

### Task 4: `scores.md`

**Files:**
- Modify: `talos/package.py`: `scores_markdown`, `scores_sections`, and the `build_package`
  call at line 340
- Test: `tests/test_package.py`

**Interfaces:**
- Consumes: `noise`, `bundle_delta` (Task 1).
- Produces:
  - `scores_markdown(baseline, candidate, rule: BeatRule | None = None) -> str`
  - `scores_sections(spec, state, *, with_noise: bool = False)`

- [ ] **Step 1: Write the failing tests** in `tests/test_package.py`. Add
  `from talos.challenges import BeatRule` and `from talos.types import NonceResult` if they
  are not already imported. Use the existing `make` fixture. Its best scores `[110, 111]` against `[100, 100]`, so the differences are
  `[10, 11]`, the delta is 10.5%, the SE is 0.5% and the ratio is 21.0.

```python
T4 = [NonceResult("t", n, True, 100, 1) for n in range(4)]
ALT = [NonceResult("t", n, True, 100 + 2 * (n % 2), 1) for n in range(4)]


def test_scores_markdown_with_a_rule_adds_the_noise_lines():
    # differences [0, 2, 0, 2]: SE sqrt(1/3)% = 0.5774%, ratio 1.7
    md = scores_markdown(T4, ALT, BeatRule())
    assert "mean_rel_se: ±0.5774%  noise: 1.7x (marginal)" in md
    assert "per track: t: ±0.58% (n=4)" in md


def test_scores_markdown_without_a_rule_has_no_noise_lines():
    # mutation: defaulting rule to BeatRule() changes evidence_draft and this table
    assert "noise:" not in scores_markdown(T4, ALT)
    assert "per track:" not in scores_markdown(T4, ALT)


def test_scores_markdown_over_the_ceiling_has_no_noise_lines():
    bad = [NonceResult("t", 0, False, None, 1, "panic")] + ALT[1:]
    md = scores_markdown(T4, bad, BeatRule())
    assert "noise:" not in md and "per track:" not in md and "mean_rel_delta" in md


def test_scores_markdown_names_n1():
    md = scores_markdown(T4[:1], [NonceResult("t", 0, True, 101, 1)], BeatRule())
    assert "noise: n=1" in md and "per track: t: n=1" in md


def test_scores_markdown_zero_se_is_not_n1():
    # mutation: `if t.rel_se:` treats a 0.0 SE as missing and prints "n=1"
    plus = [NonceResult("t", n, True, 101, 1) for n in range(4)]
    md = scores_markdown(T4, plus, BeatRule())
    assert "noise: ±0, no spread" in md and "per track: t: ±0.00% (n=4)" in md
    assert "n=1" not in md


def test_evidence_has_no_noise_but_scores_md_does(tmp_path):
    # spec Tests 12. mutation: build_package not passing with_noise=True, or evidence_draft
    # passing it
    spec, st, store = make(tmp_path)
    pkg = build_package(spec, st, store)
    assert "noise: 21.0x (clear)" in (pkg / "scores.md").read_text(encoding="utf-8")
    text = evidence_draft(spec, st)
    assert "noise:" not in text and "mean_rel_se" not in text


def test_package_builds_from_a_state_without_noise_keys(tmp_path):
    # Review Focus 5: the fixture's best.delta is {"mean_rel_delta": 0.105} only, as in a
    # state.json written before this change
    spec, st, store = make(tmp_path)
    assert "mean_rel_se" not in st.best.delta
    build_package(spec, st, store)
```

- [ ] **Step 2: Run them and confirm they fail.**
  Run `.venv/bin/python -m pytest tests/test_package.py -q -k "noise or n1"`. Expected: a
  TypeError on the third argument.

- [ ] **Step 3: Implement.**
  - With a rule, add one line after the existing summary line:
    `mean_rel_se: ±{se:.4%}  noise: {ratio:.1f}x ({band})` for `ok`;
    `noise: n=1`, `noise: identical on every nonce` or `noise: ±0, no spread` for the other
    states; nothing for `over_ceiling`.
  - Then add `per track: ` followed by `{track}: ±{rel_se:.2%} (n={n})`, or `{track}: n=1`,
    joined by `"; "`. Leave it out for `over_ceiling`.
  - `scores_sections` passes `CHALLENGES[spec.challenge].beat` when `with_noise` is true.
    `build_package` passes `with_noise=True`. `evidence_draft` keeps calling it as now.
    (`talos/report.py::_track_tables` calls `score_pairs`, not `scores_sections`, and is
    Task 5's.)

- [ ] **Step 4: Run the package tests and confirm they pass.**
  Run `.venv/bin/python -m pytest tests/test_package.py -q`. Expected: all pass, including
  the unchanged `test_scores_markdown_has_per_nonce_rows_and_delta`.

- [ ] **Step 5: Commit**

```bash
git add talos/package.py tests/test_package.py
git commit -m "Add the SE and noise ratio to scores.md, not to the evidence draft (#69)"
```

---

### Task 5: `report.html`

**Files:**
- Modify: `talos/report.py`: `ReportNode`, `report_nodes`, `_figure`, `_track_tables`, the
  CSS block and the panel/tooltip script
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: the hypothesis-record keys from Task 2, and `noise` and `bundle_delta` from
  Task 1.
- Produces:
  - `ReportNode` gains four fields, last, defaulting to `None`: `se`, `noise_ratio`,
    `noise_band`, `noise_state`.
  - A whisker element `<line class="whisker" data-whisker="{id}" .../>`.
  - Mean-row classes `noise-clear`, `noise-marginal` and `noise-noise` for a positive delta,
    and `noise-muted` for a delta ≤ 0. They go on the existing `<tr class='total'>`, so it
    reads `<tr class='total noise-clear'>`. The tests match that element.

- [ ] **Step 1: Write the failing tests** in `tests/test_report.py`. Use `rendered`,
  `_scored` and `make_run`.

```python
OK = {"mean_rel_se": 0.002, "noise_ratio": 2.5, "noise_band": "clear", "noise_state": "ok"}


def test_whisker_for_ok_nodes_only(tmp_path):
    # spec Tests 13. mutation: drawing on "SE is not null" draws one on the broken node 2
    hyps = [{**_scored(1)[0], **OK},
            {"iteration": 2, "against": 0, "mean_rel_delta": -0.3, "mean_rel_se": 0.01,
             "noise_ratio": None, "noise_band": None, "noise_state": "over_ceiling"}]
    html = rendered(tmp_path, hyps)
    assert 'data-whisker="1"' in html and 'data-whisker="2"' not in html
    assert '"noise_band": "clear"' in html  # the panel's node data


def test_old_records_draw_no_whisker(tmp_path):
    # Review Focus 5 and spec Tests 7
    html = rendered(tmp_path, _scored(3))
    assert "data-whisker" not in html


def test_whisker_is_clipped_to_the_plot(tmp_path):
    # Review Focus 4. An SE of 50 percentage points runs far past the axis.
    # mutation: no clipping puts a y coordinate outside the plot area
    from talos.report import _PLOT_BOTTOM, _TOP
    hyps = [{**h, **OK, "mean_rel_se": 0.5} for h in _scored(3)]
    html = rendered(tmp_path, hyps)
    pairs = re.findall(r'<line class="whisker"[^>]* y1="([\d.]+)"[^>]* y2="([\d.]+)"', html)
    ys = [float(v) for pair in pairs for v in pair]
    assert ys and all(_TOP <= y <= _PLOT_BOTTOM for y in ys)


def test_track_table_mean_row_colour_follows_the_sign(tmp_path):
    # spec Tests 9. The fixture's best scores [110, 111]. Against a baseline of [200, 202] the
    # delta is about -45% at a clear ratio. mutation: colouring by band alone makes it green
    from talos.state import BaselineRecord
    from talos.types import NonceResult
    from tests.run_fixture import BASELINE_FILES
    low = BaselineRecord("base", 1, "a", dict(BASELINE_FILES),
                         [NonceResult("t", 0, True, 200, 1), NonceResult("t", 1, True, 202, 1)],
                         [NonceResult("t", 1_000_000, True, 100, 1),
                          NonceResult("t", 1_000_001, True, 100, 1)])
    html = rendered(tmp_path, _scored(1), best=1, baseline=low)
    row = re.search(r"<tr class='total[^']*'>", html).group(0)
    assert "noise-muted" in row and "noise-clear" not in row
    up = rendered(tmp_path / "up", _scored(1), best=1)  # the default baseline: +10.5%, clear
    assert "noise-clear" in re.search(r"<tr class='total[^']*'>", up).group(0)
```

```python
def test_track_table_zero_se_is_not_n1(tmp_path):
    # mutation: `if t.rel_se:` prints "n=1" for a zero-spread track. A best of [110, 110]
    # against [100, 100] has differences [10, 10]: SE 0.0
    from talos.state import Candidate
    from talos.types import NonceResult
    spec, st, store = make_run(tmp_path, _scored(1), best=1)
    st.best = Candidate(1, st.best.files, "b",
                        [NonceResult("t", 0, True, 110, 1), NonceResult("t", 1, True, 110, 1)],
                        {"mean_rel_delta": 0.1}, st.best.hypothesis)
    html = _render(spec, st, store)
    assert "±0.000%" in html and "<td>n=1</td>" not in html
```

  This test checks the first `total` row only. `score_pairs` puts the training table first.
  The clipping test's regex is a sketch. The builder may read the coordinates any way that
  checks both ends of every whisker. Keep the assertion: every end lies within
  `[_TOP, _PLOT_BOTTOM]`.

- [ ] **Step 2: Run them and confirm they fail.**
  Run `.venv/bin/python -m pytest tests/test_report.py -q -k "whisker or colour_follows"`.
  Expected: they fail on the missing whisker and class.

- [ ] **Step 3: Implement.**
  - `report_nodes` reads the four keys with `h.get` (`se` comes from `mean_rel_se`).
  - `_figure` draws, before the nodes, a whisker for each node with `noise_state == "ok"`
    that is not cut. Pass both ends through the existing clamp in `y()`.
  - `_track_tables` adds an `SE` column (`±{rel_se:.3%}`, or `n=1`). The mean row shows
    `±{mean_rel_se:.3%}, {ratio:.1f}x {band}` with the sign-aware class, using
    `noise(d, CHALLENGES[spec.challenge].beat)`. For `n1`, `identical` and `no_spread` it
    shows the status-line words and gets no `noise-*` class. Over the ceiling it shows no
    ratio and gets no `noise-*` class.
  - Define CSS tokens for the three colours in both the light and the dark `:root` blocks.
    `noise-muted` uses `--muted`.
  - The panel shows the SE, ratio and band for `ok`, the status-line words for `n1`,
    `identical` and `no_spread`, and nothing for `over_ceiling`. Append the ratio to the
    tooltip for `ok`.

- [ ] **Step 4: Run the report tests and confirm they pass.**
  Run `.venv/bin/python -m pytest tests/test_report.py -q`. Expected: all pass. Then open
  one generated report in a browser, or render it to a PNG, and check that the whiskers sit
  behind their nodes in both themes.

- [ ] **Step 5: Commit**

```bash
git add talos/report.py tests/test_report.py
git commit -m "Draw the SE whisker and sign-aware noise band in report.html (#69)"
```

---

### Task 6: Docs and full check

**Files:**
- Modify: `docs/usage.md`: the Watch section example (around line 86), the text under it,
  and the `timeline.jsonl` row (around line 585)
- Modify: any other doc that `grep -rn "scored +\|scores.md" docs/ README.md` shows
  describing the scored line or `scores.md`. Leave `docs/ai/` alone.

- [ ] **Step 1: Update the Watch example line** to the new format, for example
  `#1 scored +1.000% vs baseline (±0.250%, 4.0x noise, clear; worst track +1.000%, errors 0.0%, runtime x1.00)`.
  Below it, explain:
  - `±` is the paired standard error of the mean delta over the nonces.
  - `Nx noise` is the delta over that error. The bands are `clear` ≥ 2, `marginal` 1–2 and
    `noise` < 1.
  - `n=1`, `identical on every nonce` and `±0, no spread` are the cases with no ratio.
  - Over the error ceiling, nothing is shown.
  - Add the known-limitation sentence, taken from the spec's "Known limitation" section.
- [ ] **Step 2: Add the new `scored` fields to the `timeline.jsonl` row:** `mean_rel_se`,
  `noise_ratio`, `noise_band`, `noise_state` and `track_rel_se`.
- [ ] **Step 3: Run the full check.**
  Run `make check PYTHON=.venv/bin/python > /tmp/claude-1000/-home-fibonadithya-TIG-Talos/bd54de45-c1fd-4049-b1d7-af5871ba21b5/scratchpad/check.log 2>&1; echo exit=$?`,
  then read the log's tail. Expected: `exit=0`. `tests/test_docs_references.py` checks the
  doc references.
- [ ] **Step 4: Commit**

```bash
git add docs/usage.md   # plus any other doc changed in this task
git commit -m "docs: the noise figure on the scored line and in the timeline (#69)"
```

---

## Mutation checks (after all tasks)

Isolate the bytecode cache for each run (`PYTHONPYCACHEPREFIX=$(mktemp -d)`), because a
stale `.pyc` gives a false MISSED. Each mutation must turn at least one named test red:

| mutation | test that must fail |
|---|---|
| `statistics.pstdev` in place of `stdev` | `test_paired_se_matches_hand_computation_on_two_tracks` |
| divide the track SE by `cm` instead of `bm` | same |
| `sum(se)` in place of `sqrt(sum(se²))` | same |
| check `n1` before `over_ceiling` in `noise()` | `test_noise_is_withheld_over_the_error_ceiling` |
| pair `.values()` in dict order instead of by sorted nonce | `test_se_pairs_rows_by_nonce_not_by_input_order` |
| overwrite the noise fields in the demoted branch too | `test_a_demoted_record_keeps_the_native_noise` |
| `if t.rel_se:` in the scores.md per-track line | `test_scores_markdown_zero_se_is_not_n1` |
| `if t.rel_se:` in the report SE column | `test_track_table_zero_se_is_not_n1` |
| `>` for `>=` at `NOISE_CLEAR` | `test_noise_band_boundaries[2.0-clear]` |
| leave out `**self._noise_fields(d)` in `_validate_and_finish` | `test_a_validated_record_carries_the_metered_noise` |
| `_noise_text` switches on `mean_rel_se is None` | `test_event_line_scored_over_the_ceiling_has_no_noise[None]` |
| default `rule=BeatRule()` in `scores_markdown` | `test_scores_markdown_without_a_rule_has_no_noise_lines` |
| whisker drawn when `se is not None` | `test_whisker_for_ok_nodes_only` |
| mean-row class from the band alone | `test_track_table_mean_row_colour_follows_the_sign` |
