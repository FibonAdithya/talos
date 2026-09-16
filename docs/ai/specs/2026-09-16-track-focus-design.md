# Track focus (`--track`) — design

Date: 2026-09-16. Branch: `codex-model-list` (cut from `main` at `d5c1942`,
carrying the codex catalog and stdin fixes; this feature lands on top of
them).

This spec adds an optional `--track <name>` to `talos run`. A job started
with it optimises one mainnet track of the challenge instead of all of them.
Without the flag every behaviour is byte-identical to today.

## 1. Why

Mainnet awards adoption per track: `get-algorithms` reports
`num_qualifiers_by_track_by_player` keyed by track name, so an algorithm that
wins one track earns on that track without winning the others (checked
2026-09-16 on the knapsack top algorithm `knap_lean`). Talos today scores
every active track and its beat rule averages across them, which is a
stricter bar than mainnet applies and spreads the model's attention across
the whole algorithm. On knapsack the top algorithm is five independent
solvers of about 100 KB each; iteration 1 of job `20260916-095103-knapsack`
failed after the model edited two tracks at once.

What this feature does not do: restrict which files the model may edit. At
the pinned monorepo commit the algorithm layouts are one `mod.rs`, a split
by concern (`construct.rs`, `ils.rs`, ...), or per-track files with
algorithm-specific names (`track1.rs` ... `track5.rs`, `track_t1.rs` ...,
routed by thresholds the algorithm chooses). There is no rule that maps a
mainnet track name to a file, so the file set stays as it is and the prompt
token count per call does not drop. A file restriction is a possible
follow-up, not part of this spec.

## 2. Decision: guard the other tracks at confirmation

Three options were considered for the tracks that are not the focus.

| option | training | confirmation | rejected because |
|---|---|---|---|
| no guard | focus track only | focus track held-out only | an edit to shared code can regress the other tracks unseen, and the package carries that to mainnet |
| **guard at confirmation** (chosen) | focus track only | focus track held-out, plus every other track's training nonces; no other track may drop | — |
| narrow the baseline | focus track only, baseline re-measured on it | focus track only | a new baseline cache key means a new 12-minute C3 job, and the guard needs the other tracks' baseline anyway |

The guard costs the other tracks' training nonce count (128 on knapsack)
only on iterations that win on training: about a minute of C3 time per
confirmation, none on the others.

## 3. Job spec

`talos/state.py::JobSpec` gains `track: str | None = None`, placed last so
positional construction is unaffected. `from_dict` reads it with `.get`, so a
`job.json` written before this change resumes as an all-tracks job.
`redacted()` keeps it: a track name is not secret. `tracks`, `training` and
`holdout` are unchanged and still cover every active track, for two reasons:

- The baseline cache key (`talos/baseline.py::cache_key`) is built from the
  nonce sets, so an unchanged draw keeps every existing cached baseline a hit.
- The guard needs the other tracks' baseline results.

The baseline is therefore measured, cached, and stored in `state.json`
exactly as today. All slicing happens in the loop.

## 4. Slicing helpers

Both are pure functions in `talos/scoring.py`, no I/O, so the loop and the
package share one slicing and the C3 job could reuse it if it ever needs to.

```
def select(results: list[NonceResult], sets: list[NonceSet]) -> list[NonceResult]
```

Returns the results whose `(track, nonce)` fall inside one of `sets`, in
input order. A result outside every set is dropped, never raised on; the
existing `bundle_delta` raises `ScoringError` on any mismatch that survives.

```
def focus_sets(track: str | None, training: list[NonceSet],
               holdout: list[NonceSet]) -> tuple[list[NonceSet], list[NonceSet]]
```

With `track` set it returns

- training: `[s for s in training if s.track == track]` (one set)
- holdout: the focus track's held-out set, followed by every other track's
  **training** set, in `training` order

and with `track` None it returns `(training, holdout)` unchanged. Callers
pass the spec's `track`, `training` and `holdout`. The guard reuses training nonces because they are already
measured for the baseline and the model never sees nonce identities in
either set (`rand_hash` is redacted from everything the model reads).

## 5. The request per iteration

`Loop._request` becomes:

```
training, holdout = focus_sets(self.spec.track, self.spec.training, self.spec.holdout)
EvalRequest(challenge, files, training=training, holdout=holdout, fuel,
            baseline_training=select(self.state.baseline.training, training),
            rule=self.rule)
```

The bench protocol, `ModalBench`, `C3Bench`, `FakeBench` and the
in-container `holdout_decision` are untouched: each already scores whatever
sets it is given and decides held-out from `beats(baseline_training,
training, rule)`, which with one track is that track's margin. The C3
reattach path rebuilds the request from the spec on resume and hashes it
(`talos/c3_jobdir.py::request_hash`); `focus_sets` is deterministic in the
spec, so the rebuilt request hashes the same.

`Loop._current_training` and `_best_delta` keep working: a candidate's
`training` holds only the focus track's results, and its `delta` is the
single-track bundle delta.

## 6. Scoring

Training: `bundle_delta(select(baseline.training, training_sets),
res.training)` and `beats(...)` on the same inputs. With one track, mean and
worst deltas coincide and the margin applies to that track.

Confirmation: a new rule.

```
def beats_focused(baseline: list[NonceResult], candidate: list[NonceResult],
                  rule: BeatRule, track: str) -> bool
```

`bundle_delta` on the full lists, then: the focus track's `rel_delta >=
rule.margin`; every other track's `rel_delta >= -rule.track_tolerance`; and
the focus track's own error rate (`cand_errors / n` on that track) `<=
rule.error_ceiling`. The error ceiling is applied to the focus track alone
because an unchanged guard track cannot gain errors and a changed one shows
up as a quality drop (an errored nonce scores as the worst observed quality).
`ScoringError` propagates as `beats` does.

`Loop._confirm` uses, when `spec.track` is set,

- candidate: `res.holdout` as returned (focus held-out then guard rows)
- baseline: `select(baseline.holdout + baseline.training, holdout_sets)`,
  which yields the focus track's held-out rows and the guard tracks' training
  rows because the two kinds of set never share a `(track, nonce)`

and `beats_focused` with the spec's track. The `holdout_delta` recorded in
the timeline and the `false_positive` event use `bundle_delta` on the same
pair, so a guard regression is visible in the event as a negative
`worst_rel_delta`. Without a track, `_confirm` runs today's code path.

## 7. Prompts

`talos/prompts.py::PromptContext` gains `track: str | None = None` and
`guard_tracks: list[str]` (default empty). The loop fills both from the spec.

- `hypothesis_prompts`: "across every active track" becomes, when a track is
  set, "on track `<track>`. The other active tracks (`<guards>`) are re-scored
  as a regression guard when a candidate wins, and none of them may get
  worse: confine changes to the code path that serves `<track>`."
- `edit_prompts`: the same sentence after the hypothesis.
- `talos/agentic.py::claude_md`: the same sentence replaces "on every active
  track" in the goal line.

Nothing else in the prompts changes. `compile_fix_prompts`,
`edit_repair_prompts` and `distill_prompts` do not mention tracks today and
do not need to.

## 8. CLI

`talos run --track <name>`:

- Validated after `fetch_challenge_info`: the name must be in `info.tracks`,
  else exit 2 with a message listing the active tracks. In `--fake` mode the
  only track is `n=1`.
- Interactive: without `--track`, without `--yes`, and without `--resume`,
  once the active tracks are known (after the mainnet lookup, so after the
  mode prompt): `Track to optimise (all, or one of: ...)`, default `all`.
  `all` means None. An answer that is neither `all` nor an active track exits
  2 like the flag.
- `--resume` ignores `--track` the way it ignores `--challenge`: the job's
  `job.json` decides. Passing a different `--track` with `--resume` is
  refused with exit 2, mirroring the `--mode` check.
- The job start line becomes `Job <id>: track <name> of <n> tracks, ...` or
  today's text when unfocused.

`talos status` and the per-iteration status line are unchanged.

## 9. Package

`talos/package.py`:

- `scores.md`: `# Training nonces (track <t>)`, `# Held-out nonces (track
  <t>)`, and, when the job has guard tracks, `# Regression guard (other
  tracks, training nonces)` with the candidate's guard rows against the
  baseline's training rows for those tracks. Unfocused jobs keep today's two
  headings. The split uses
  `select` with the focus sets, so it is the same slicing the loop applied.
- `README.md` head: after the challenge line, `Optimised for track <t>; the
  other tracks were re-scored on confirmation as a regression guard.` The
  confirmed / false-positive sentences are unchanged; a guard failure is a
  false positive and `scores.md` shows which track dropped.
- `evidence_draft.md` appendix: `Baseline ... tracks <all>. Optimised for
  track <t>.` and the same three tables.

## 10. Invariants (AGENTS.md)

- Invariant 1 (identical nonces) holds: every comparison slices baseline and
  candidate with the same `NonceSet` list.
- Invariant 2 (`rand_hash`) is untouched; `select` keys on `(track, nonce)`
  and never reads the hash.
- Invariant 3 (pins in cache keys) holds because the nonce draw is unchanged.
- `AGENTS.md` gains, under invariant 1, one sentence: with `--track`, the
  guard compares the other tracks' training nonces against the cached
  baseline training results for those same nonces.

## 11. Tests and the mutation each catches

| test | mutation caught |
|---|---|
| `test_state`: spec round-trips with `track`; a dict without the key loads with `track=None` | dropping the `.get` default breaks resume of old jobs |
| `test_scoring`: `select` keeps only in-set rows, preserves order, drops strays | keying on track only would keep held-out rows in a training slice |
| `test_scoring`: `beats_focused` fails when a guard track drops by more than the tolerance, even with the focus track far over margin | dropping the guard check |
| `test_scoring`: `beats_focused` fails when the focus track is under margin though the mean across tracks clears it | applying the margin to the mean instead of the focus track |
| `test_scoring`: `beats_focused` on a job with the focus track only equals `beats` | the two rules diverging when there is no guard |
| `test_loop`: with a track, the request's training is one set, holdout is focus held-out then guard training sets, and `baseline_training` holds only that track's rows | forgetting the guard, or sending the whole baseline |
| `test_loop`: confirmation with a guard regression records a false positive and stays `researching` | confirming on the focus track alone |
| `test_loop`: without a track, the request equals today's | the focus path leaking into unfocused jobs |
| `test_cli`: unknown `--track` exits 2 and lists the active tracks; `--resume` with a different track exits 2 | skipping validation |
| `test_cli`: fake end-to-end `--track n=1 --yes --fake` wins and the package has the three headings | the whole path |
| `test_prompts`: a focused context names the track and the guards; an unfocused one does not | prompt text regression |
| `test_package`: scores headings and the README sentence for a focused job | package text regression |

`make check` stays the gate; no live test is added. The C3 job runner is not
modified, so its tests are unchanged.

## 12. User documentation

`README.md`: `--track` joins the `talos run` flag list ("one active track of
the challenge to optimise; the others are re-scored as a regression guard
when a candidate wins"), the Budget section notes the guard's extra nonces on
winning iterations, and "Where results land" names the third `scores.md`
section. `AGENTS.md` changes only as §10 says.
