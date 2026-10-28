# Noise ratio next to each scored delta (issue #69)

Status: implemented on branch noise-ratio-69, 2026-10-28.

## Why

A status line says `#4 scored +0.4% vs baseline`. Nothing tells the user whether +0.4% is a
gain or nonce-to-nonce variation. This matters most for "improved but not won": a candidate
beats the baseline on training and then fails on held-out.

Baseline and candidate are scored on the same nonces (AGENTS.md invariant 1), and every
per-nonce result is kept. A paired standard error is therefore computable for every scored
candidate with no extra compute.

## Goals

- For each scored candidate, compute the standard error (SE) of its delta per track and for
  the mean over tracks, and the noise ratio `|mean delta| / SE of the mean`.
- Show the SE and ratio on the `scored` status line, record them in `timeline.jsonl` and the
  hypothesis record, add them to `scores.md`, and draw them in `report.html`.

## Non-goals

- No change to any decision: `beats`, `improves`, `clears`, `beats_focused`,
  `holdout_decision`, `BeatRule` and `track_tolerance` behave exactly as now. The noise
  figure is display only.
- No change to the LLM prompts. `talos/prompts.py` reads named fields from the hypothesis
  record, so new keys there do not reach a prompt; keep it that way.
- No SE on the `won`, `false_positive` (held-out) or `validated` (metered) status lines. The
  `scores.md` held-out tables do get it, because they use the same table code (see below).
- The TIG submission evidence draft (`talos/package.py::evidence_draft`) is unchanged. Its
  appendix uses the same `scores_sections` as `scores.md`, but it is text the user submits
  to TIG, so the noise lines stay out of it (my call, 2026-10-28; see `scores.md` below).
- `talos status --json` and `talos show --json` are unchanged. Adding keys there would be a
  compatible change (docs/json-output.md); it is a follow-up, not this issue.
- No correction for choosing the best of many candidates. See "Known limitation".

## Anti-goals

- An SE whose centre is not the delta printed beside it.
- A display that makes a regression look good, for example a green band on a −12% delta.
- Any extra scoring, nonces or compute.

## Decisions (agreed with the repo owner, 2026-10-28)

### 1. The SE matches the delta shown

Today `rel_delta` per track is `(mean(c) − mean(b)) / mean(b)`. Because the nonces are
paired, this equals `mean(c_i − b_i) / mean(b)`. The SE is the SE of that same quantity,
treating `mean(b)` as fixed:

    se_track = stdev(c_i − b_i) / sqrt(n) / mean(b)        (stdev with n − 1)
    se_mean  = sqrt(sum over tracks of se_track²) / T        (T = number of tracks)
    ratio    = |mean_rel_delta| / se_mean

`c_i` and `b_i` are the same per-nonce qualities `bundle_delta` already uses, after its
worst-quality substitution for failed nonces. The SE therefore always describes the number
on the line. The tracks are combined as independent because their nonces are disjoint.

The issue's wording ("SE of the paired per-nonce relative delta") would average per-nonce
ratios `(c_i − b_i) / b_i`. That is rejected: its centre differs from the printed delta, and
it divides by zero when a baseline nonce scores 0.

MEASURED 2026-10-28: this formula reproduces `bundle_delta`'s `rel_delta` exactly on all 27
stored candidates in `runs/` (four runs). The script asserts it (session scratch, not
committed):
`/tmp/claude-1000/-home-fibonadithya-TIG-Talos/bd54de45-c1fd-4049-b1d7-af5871ba21b5/scratchpad/measure_se.py`.
Example results: knapsack 20260923 #5, +0.124% ± 0.035%, 3.5×; job_scheduling 20261026 #6,
+0.797% ± 0.278%, 2.9×.

### 2. Rules when the SE is undefined or zero

Qualities are integers, so `c_i − b_i` is exact and "SE is 0" needs no tolerance.

| case | se_mean | ratio | `noise_state` | shown |
|---|---|---|---|---|
| error rate over the ceiling (decision 3; checked first) | as computed | null | `over_ceiling` | nothing |
| any track has n = 1 | null | null | `n1` | `n=1` |
| every track has stdev 0 and every difference is 0 | 0 | null | `identical` | `identical on every nonce` |
| every track has stdev 0, some difference is not 0 | 0 | null | `no_spread` | `±0, no spread` |
| otherwise | value | value | `ok` | `±SE, R× noise, band` |

`noise_state` is recorded with the other fields. Without it, the display code could not tell
the cases apart: over the ceiling, a track with n = 1 gives the same null SE and ratio as a
plain n = 1, and a zero spread gives the same as `no_spread` (found by the builder check;
recording the state is my call, 2026-10-28). Display code switches on `noise_state` alone.

The SE of the mean needs every track's SE, so a single track with n = 1 makes the mean's SE
undefined. A per-track SE is still recorded where that track has n ≥ 2.

### 3. No ratio over the error ceiling

Failed nonces are scored at the worst observed quality. This turns a broken candidate into a
confident-looking ratio (MEASURED 2026-10-28: knapsack 20260923 #6, −12.08% with 40% errors,
13.4×; job_scheduling 20261026 #1, −61.7% with 100% errors, 7.0×). When
`error_rate > rule.error_ceiling`:

- The status line shows no SE and no ratio, and `report.html` draws no whisker and shows no
  ratio for the node.
- `noise_ratio` and `noise_band` are recorded as null, and `noise_state` as
  `over_ceiling`. `mean_rel_se` and the per-track SEs are still recorded: they are facts
  about the numbers, and only their display is misleading.

At or under the ceiling, errored nonces stay in the SE with the same substitution as the
delta. That keeps decision 1 true.

### 4. Bands depend on the ratio; colour also depends on the sign

Bands, from the issue (pi-autoresearch's thresholds):

| ratio | band |
|---|---|
| ≥ 2.0 | `clear` |
| ≥ 1.0 and < 2.0 | `marginal` |
| < 1.0 | `noise` |

2.0 is `clear`; 1.0 is `marginal`.

The band word does not say whether the change is good or bad, so it appears for both signs.
Colour does carry that meaning, so it is applied only when `mean_rel_delta > 0`: green for
`clear`, amber for `marginal`, red for `noise`. A delta ≤ 0 is drawn in the muted text
colour, whatever its band. The terminal has no colour (talos/cli.py prints none), so there
only the word appears.

## Where each piece goes

### Computation: `talos/scoring.py`

Compute the SE inside `bundle_delta`, which already holds the substituted `bq` and `cq` lists.
A second function would have to repeat the substitution and could drift from it.

- `TrackDelta` gains `rel_se: float | None`.
- `BundleDelta` gains `mean_rel_se: float | None`.
- Both take defaults (`None`), so any code that builds them by keyword still works.
- A function `noise(d: BundleDelta, rule: BeatRule) -> tuple[float | None, str | None, str]`
  returns `(ratio, band, noise_state)` and applies decisions 2 and 3, in the order of the
  table in decision 2. It is the only place those rules live. Display code reads the
  recorded values and never re-derives them.

`BundleDelta.to_dict()` now carries the new keys into `Candidate.delta`, and so into
`state.json`'s `best.delta`. Old `state.json` files lack them. Every reader must use `.get`,
and `Bar.of` already reads only the two keys it needs.

`talos/scoring.py` is copied into the C3 and local job containers (it is in `c3/<n>/talos/`).
The change may use only the standard library (`math`, `statistics`), as the file does now.

### Loop: `talos/loop.py` (the `scored` event near line 714)

- Add `mean_rel_se`, `noise_ratio`, `noise_band` and `noise_state` to the `scored` event.
  Add the per-track SEs as `track_rel_se: {track: se or null}`.
- Add `mean_rel_se`, `noise_ratio`, `noise_band` and `noise_state` to the hypothesis
  `record`, next to `mean_rel_delta`. `report.html` reads them from there.
- **Metered validation** (`Loop._validate_and_finish`, near line 793). A native candidate
  that passes validation has its record's `mean_rel_delta` replaced by the metered delta
  `d`, and `report.html` plots the metered score (`talos/lineage.py::Node.rank_score`).
  At that point, also overwrite the four noise fields in the record with values computed
  from `d` (via `noise(d, self.rule)`). Otherwise a metered score would be drawn with a
  native SE, against the first anti-goal (found by the builder check; my call,
  2026-10-28). The `scored` event keeps the native values; it describes the native
  scoring. The `validated` status line still shows no noise (non-goal).

### Status line: `talos/cli.py::_event_line`, kind `scored`

Put the noise text right after the delta, so a narrow terminal truncates the tail and not the
noise text:

    scored +0.400% vs baseline (±0.310%, 1.3x noise, marginal; worst track -0.120%, errors 0.0%, runtime x1.02)
    scored +0.000% vs baseline (identical on every nonce; worst track ...)
    scored +0.020% vs baseline (n=1; worst track ...)
    scored -12.084% vs baseline (worst track -34.810%, errors 40.0%)          <- over the ceiling

The SE is shown with the same 3 decimal places as the delta. The ratio uses one decimal and
an `x`, matching `runtime x1.02` on the same line.

(Retracted 2026-10-28: an earlier draft said `×` was avoided because it is non-ASCII on
Windows. That was wrong: `talos/cli.py::_utf8_stdout` reconfigures stdout to UTF-8, and the status
line already prints `∞` and `≈`. `x` is kept only to match the `runtime` figure. `±` is
fine for the same reason.)

An event written before this change has no `noise_state` key and must render as it does
today.

`_event_line` does not have the rule. It switches on the recorded `noise_state` and formats
the recorded `mean_rel_se`, `noise_ratio` and `noise_band`.

### `scores.md`: `talos/package.py::scores_markdown`

After the existing summary line, add:

    mean_rel_se: ±0.0310%  noise: 3.2x (clear)
    per track: n_items=5000,b=...: ±0.14% (n=32); ...

The same rules apply (`n=1`, `identical on every nonce`, `±0, no spread`, nothing over the
ceiling). `scores_markdown` computes them by calling `noise()`, because these tables are not
hypothesis records.

`scores_markdown` and `scores_sections` have two callers: `build_package` writes `scores.md`
(`talos/package.py:340`), and `evidence_draft` builds the TIG submission appendix
(`talos/package.py:116`). The noise lines go only into `scores.md` (see Non-goals). So:

- `scores_markdown(baseline, candidate, rule: BeatRule | None = None)`: with `rule=None` the
  output is byte-identical to today's. `tests/test_package.py:74` calls it with two
  arguments and keeps passing unchanged.
- `scores_sections(spec, state, *, noise: bool = False)`: passes the challenge's beat rule
  when `noise` is true. `build_package` passes `noise=True`, and `evidence_draft` keeps the
  default.

In `scores.md`, every table gets the lines, held-out and regression-guard tables included.

### `report.html`: `talos/report.py`

- `ReportNode` gains `se`, `noise_ratio`, `noise_band` and `noise_state`, read from the
  hypothesis record with `.get`. For a validated node these are the metered values (see
  Loop), so they match the metered score the figure plots.
- `_figure`: draw a vertical whisker from `score − se` to `score + se` behind each scored node
  whose `noise_state` is `ok`. There is no whisker for `over_ceiling` (decision 3), `n1`,
  `identical` or `no_spread`, nor for a record without the key. The whisker is clipped to
  the axis and left out for cut (off-axis) nodes. It is ±1 SE, so it matches the `±` printed
  on the status line.
- The node panel and tooltip show the SE, ratio and band for `ok`. For the other states
  they show the same words as the status line, and nothing for `over_ceiling`.
- `_track_tables`: add an `SE` column per track. In the `mean` row, show the mean's SE and
  the ratio with its band, coloured by decision 4.
- Colours are CSS tokens defined for both the light and the dark theme, like the existing
  `--add` and `--del`.
- `axis_for` keeps scaling on scores only. Whiskers are clipped, not used to widen the axis.

### Docs

- `docs/usage.md`, where the status line is described: the meaning of `±SE`, `Nx noise` and
  the band, and one sentence on the known limitation below.
- `docs/usage.md`, the `timeline.jsonl` row: the new `scored` fields.
- Grep `docs/` for other descriptions of the scored line or `scores.md` and update them.

## Known limitation (goes into docs/usage.md)

The SE measures nonce-to-nonce variation for one candidate. It does not account for the best
being chosen out of many candidates, which is how a candidate improves on training and then
fails held-out. A ratio of 3× on the best of 7 tries is weaker evidence than 3× on a single
try.

It is not yet known whether the ratio predicts held-out results. MEASURED 2026-10-28: all 8
local runs have empty `confirmed` and `false_positives` lists, so none has a held-out score
to check against.

## Tests

The expected values below are worked out by hand from the formula, not taken from the code
under test.

1. **Two tracks, paired.** Track A: baseline `[100, 100, 100, 100]`, candidate
   `[101, 103, 99, 105]`. Differences are `[1, 3, −1, 5]`, mean 2, so `rel_delta` = 2%. The
   stdev is `sqrt(20/3)`, so `rel_se` = `sqrt(20/3) / 2 / 100` = `sqrt(5/3)`% ≈ 1.29099%.
   Track B: baseline `[200, 200]`, candidate `[200, 204]`. Differences `[0, 4]`, so
   `rel_delta` = 1%. The stdev is `sqrt(8)`, so `rel_se` = `sqrt(8)/sqrt(2)/200` = 1%.
   Mean delta = 1.5%. `mean_rel_se` = `sqrt(5/3 + 1) / 2` = `sqrt(2/3)`% ≈ 0.816497%.
   Ratio = 1.5 / sqrt(2/3) ≈ 1.83712, band `marginal`. Assert each to 1e-9 relative.
2. **Single nonce.** One track with n = 1: `rel_se`, `mean_rel_se` and the ratio are None, and
   the status line contains `n=1`. A second case: one track with n = 1 next to one with
   n = 4. The n = 4 track has a `rel_se`, the mean's SE is None, and the line shows `n=1`.
3. **Failed nonce under the ceiling.** Baseline `[10, 20, 30]`, candidate `[12, error, 30]`.
   The worst observed ok quality is 10, so the candidate is scored `[12, 10, 30]` and the
   differences are `[2, −10, 0]`. `rel_delta` = −13.333…%, and `rel_se` = stdev / sqrt(3) /
   20 ≈ 18.5592%. Use a rule whose `error_ceiling` admits 1 error in 3.
4. **Over the ceiling.** The same data under a rule with `error_ceiling` 0.05: `noise()`
   returns `(None, None, "over_ceiling")`, the status line has no `±`, and `mean_rel_se` is still a number.
5. **Zero spread.** Candidate equal to the baseline on every nonce: the line shows
   `identical on every nonce`. A candidate that is +1 on every nonce of every track: the line
   shows `±0, no spread`. In both cases the ratio is None.
6. **Band boundaries.** Ratios 2.0 → `clear`, 1.999 → `marginal`, 1.0 → `marginal`, 0.999 →
   `noise`. Test with constructed `BundleDelta` values so the ratio is exact.
7. **Old records.** A `scored` event and a hypothesis record without the new keys render as
   they do today (status line, `scores.md`, `report.html`).
8. **Decisions unchanged.** For the fixtures above, `beats`, `improves` and
   `holdout_decision` return the same results as before. Existing tests cover this; run the
   whole suite.
9. **Report colour by sign.** In `report.html`, a negative `clear` mean row gets the muted
   class and not the green one.
10. **Over the ceiling with n = 1.** A one-nonce track whose candidate failed, under
    `error_ceiling` 0.05: `noise_state` is `over_ceiling`, and the status line contains
    neither `n=1` nor `±`. The same check for zero spread over the ceiling: no
    `no spread` text.
11. **Validated node.** A native job whose candidate validates, with metered rows chosen so
    the metered SE differs from the native one by hand computation: after
    `_validate_and_finish`, the record's `mean_rel_se` equals the metered value, and the
    `scored` event in `timeline.jsonl` still holds the native value.
12. **Evidence unchanged.** `evidence_draft` output contains no `noise:` or `mean_rel_se`
    text, while `scores.md` from the same state does.
13. **No whisker over the ceiling.** In `report.html`, a node with `noise_state`
    `over_ceiling` and a non-null `mean_rel_se` has no whisker element.

## Left to the builder

- Names of CSS classes and tokens, and the exact whisker styling.
- Whether the tooltip shows per-track SEs or only the mean's.
- Layout of the per-track SE line in `scores.md` when there are many tracks.
