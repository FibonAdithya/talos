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
        if not c.ok or (b is not None and not b.ok):
            note = "err"
        elif b and b.ok and c.ok:
            note = "+" if c.quality > b.quality else ("=" if c.quality == b.quality else "-")
        else:
            note = ""
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
    # deviation from brief: also include outcome "won" (a winning iteration's hypothesis
    # record carries outcome "won", not "improved" — see JobState.confirmed, Task 11)
    winners = [h for h in state.hypotheses if h.get("outcome") in ("improved", "won")]
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


def _readme_no_candidate(spec: JobSpec, state: JobState) -> str:
    baseline = f" Baseline: mainnet `{state.baseline.name}`." if state.baseline else ""
    return ("# Talos hand-back\n\n"
            f"No candidate beat the baseline.{baseline} Status: {state.status}. "
            f"Reason: {state.stop_reason}\n\n"
            "There is nothing to submit from this job.\n")


def _readme(spec: JobSpec, state: JobState) -> str:
    confirmed = state.best.iteration in state.confirmed
    false_positive = state.best.iteration in state.false_positives
    head = ("# Talos hand-back\n\n"
            f"Challenge: {spec.challenge}. Baseline: mainnet `{state.baseline.name}`. "
            f"Status: {state.status}. Reason: {state.stop_reason}\n\n")
    if confirmed:
        head += ("This candidate beat the baseline on both the training and the held-out nonce "
                 "sets. See scores.md.\n\n")
    elif false_positive:
        head += ("This candidate beat the baseline on the training nonces but NOT on the "
                 "held-out nonces (a false positive); see scores.md.\n\n")
    else:
        head += ("This candidate was never scored on the held-out nonces; it did not beat the "
                 "baseline on training.\n\n")
    head += ("## Submitting\n\n1. Copy the algorithm files into "
             f"`tig-algorithms/src/{spec.challenge}/<your_name>/` in a monorepo checkout.\n"
             "2. Add the copyright header the TIG Inbound Game License requires.\n"
             "3. Follow docs/guides in the monorepo to submit. For Advance Rewards, complete "
             "evidence_draft.md.\n")
    return head


def _hypothesis_line(h: dict) -> str:
    line = (f"- #{h['iteration']} (vs best #{h.get('against', '?')}) [{h.get('strategy_tag', '')}] "
            f"{h.get('title', '')}: {h.get('description', '')} -> {h.get('outcome', '')}")
    error = h.get("error")
    if error:
        line += f" — {error[:200]}"
    return line


def build_package(spec: JobSpec, state: JobState, store: JobStore) -> Path:
    pkg = store.run_dir / "package"
    if pkg.exists():
        shutil.rmtree(pkg)
    pkg.mkdir()
    if state.best is None:
        (pkg / "README.md").write_text(_readme_no_candidate(spec, state))
        shutil.make_archive(str(store.run_dir / "package"), "zip", pkg)
        return pkg
    if state.baseline is None:
        (pkg / "README.md").write_text("# Talos hand-back\n\nNo candidate was produced.\n")
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
    hyps = "\n".join(_hypothesis_line(h) for h in state.hypotheses)
    (pkg / "hypotheses.md").write_text("# Hypotheses\n\n" + hyps + "\n")
    (pkg / "evidence_draft.md").write_text(evidence_draft(spec, state))
    (pkg / "README.md").write_text(_readme(spec, state))
    shutil.make_archive(str(store.run_dir / "package"), "zip", pkg)
    return pkg
