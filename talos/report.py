"""The per-run report.html: one self-contained page (inline CSS, SVG and JS) showing a job's
search, built only from the files already in its run directory, so it works on old runs and on
runs still in progress.

The data functions (report_nodes, node_diffs, axis_for, holdout_points, truncate) are pure.
run_sources and read_timeline read the run directory and nothing else.
"""
from __future__ import annotations

import json
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from talos.package import _diff
from talos.state import JobSpec, JobState

KINDS = ("best", "false_positive", "demoted", "improved", "accepted", "rejected", "failed")

Sources = Callable[[int], "dict[str, str] | None"]


@dataclass(frozen=True)
class ReportNode:
    id: int
    parent: int
    chain: int
    family: str
    title: str
    description: str
    strategy_tag: str
    outcome: str
    error: str | None
    score: float | None
    tracks: dict[str, float] | None
    metered_score: float | None
    accepted: bool
    probability: float | None
    draw: float | None
    temperature: float | None
    kind: str
    former_best: bool
    confirmed: bool
    false_positive: bool


def _kind(n: int, score: float | None, outcome: str, accepted: bool, state: JobState,
          demoted: set[int]) -> str:
    if state.best is not None and state.best.iteration == n:
        return "best"
    if n in state.false_positives:
        return "false_positive"
    if n in demoted:
        return "demoted"
    if score is None:
        return "failed"
    if outcome in ("improved", "won"):
        return "improved"
    return "accepted" if accepted else "rejected"


def report_nodes(spec: JobSpec, state: JobState) -> list[ReportNode]:
    """One node per hypothesis record, sorted by id; the root is not included. A stored lineage
    node with the same id supplies the tree fields; a record from before the lineage tree (or
    from before a resume onto it) supplies them itself."""
    stored = {n.id: n for n in state.nodes}
    demoted = {v.get("iteration") for v in state.validations if v.get("outcome") == "demoted"}
    out = []
    for h in state.hypotheses:
        i = h.get("iteration")
        if not isinstance(i, int):
            continue
        outcome = h.get("outcome", "")
        s = stored.get(i)
        if s is not None:
            parent, chain, family = s.parent, s.chain, s.family
            score, tracks, metered = s.rank_score(), s.tracks, s.metered_score
            accepted, probability, draw, temperature = (s.accepted, s.probability, s.draw,
                                                        s.temperature)
        else:
            parent, chain = h.get("against", 0), h.get("chain", 0)
            family = h.get("design_family") or h.get("strategy_tag", "")
            score, tracks, metered = h.get("mean_rel_delta"), None, None
            accepted = h.get("accepted", outcome in ("improved", "won", "accepted"))
            probability, draw, temperature = (h.get("probability"), h.get("draw"),
                                              h.get("temperature"))
        kind = _kind(i, score, outcome, accepted, state, demoted)
        out.append(ReportNode(
            id=i, parent=parent if isinstance(parent, int) else 0, chain=chain or 0,
            family=family or "", title=h.get("title", ""), description=h.get("description", ""),
            strategy_tag=h.get("strategy_tag", ""), outcome=outcome, error=h.get("error"),
            score=score, tracks=tracks, metered_score=metered, accepted=bool(accepted),
            probability=probability, draw=draw, temperature=temperature, kind=kind,
            former_best=bool(s is not None and s.best and kind != "best"),
            confirmed=i in state.confirmed, false_positive=i in state.false_positives))
    return sorted(out, key=lambda n: n.id)


def root_location(spec: JobSpec) -> str:
    """Where the root's files are in the run directory, for the truncation notes."""
    return "challenge/template.rs" if spec.start == "template" else "baseline"


def run_sources(run_dir: Path, spec: JobSpec, state: JobState) -> Sources:
    """node id -> that node's files, or None when they were not saved. The root is the
    baseline's files, or the template as mod.rs in a template job; the current best is
    state.best's files; any other node is what its iteration directory holds of the root's file
    names (edit and compile failures save none)."""
    run_dir = Path(run_dir)
    if spec.start == "template":
        path = run_dir / "challenge" / "template.rs"
        root = ({"mod.rs": path.read_bytes().decode("utf-8", errors="replace")}
                if path.is_file() else None)
        names: list[str] = ["mod.rs"]
    else:
        root = dict(state.baseline.files) if state.baseline is not None else None
        names = sorted(root) if root else []
    cache: dict[int, dict[str, str] | None] = {}

    def lookup(node_id: int) -> dict[str, str] | None:
        if node_id == 0:
            return root
        if state.best is not None and state.best.iteration == node_id:
            return dict(state.best.files)
        if node_id not in cache:
            # Not JobStore.iteration_dir: it creates the directory.
            it_dir = run_dir / "iterations" / f"{node_id:04d}"
            files = {name: (it_dir / name).read_bytes().decode("utf-8", errors="replace")
                     for name in names if (it_dir / name).is_file()}
            cache[node_id] = files or None
        return cache[node_id]

    return lookup


def truncate(text: str, cap: int, note: str) -> str:
    """text cut to `cap` characters, inside a line if need be, then the note on its own line
    with `{n}` replaced by the number of lines (or rest of a line) left out."""
    if len(text) <= cap:
        return text
    kept = text[:cap]
    n = len(text[cap:].splitlines())
    return kept + ("" if kept.endswith("\n") else "\n") + note.replace("{n}", str(n))


def _location(node_id: int, spec: JobSpec) -> str:
    return root_location(spec) if node_id == 0 else f"iterations/{node_id:04d}"


def node_diffs(nodes: list[ReportNode], sources: Sources, spec: JobSpec,
               total: int = 1_000_000, per_node: int = 64_000) -> dict[int, str]:
    """Each node's diff against its parent, or a message saying why there is none. The diffs
    share `total` characters evenly, each capped at `per_node`."""
    root_label = "template" if spec.start == "template" else "baseline"
    raw: dict[int, tuple[str, str | None]] = {}
    for n in nodes:
        mine = sources(n.id)
        if mine is None:
            raw[n.id] = ("no files saved", None)
            continue
        theirs = sources(n.parent)
        if theirs is None:
            raw[n.id] = ("parent's files not saved", None)
            continue
        text = _diff(theirs, mine, root_label if n.parent == 0 else f"#{n.parent}")
        if not text:
            raw[n.id] = ("no change", None)
            continue
        here = _location(n.id, spec)
        if n.parent == 0 and spec.start == "template":
            here += "/mod.rs"  # `diff -u file dir` looks for dir/<file's basename>
        raw[n.id] = (text, f"… truncated ({{n}} more lines); full diff: diff -u "
                           f"{_location(n.parent, spec)} {here}")
    count = sum(1 for _, note in raw.values() if note is not None)
    cap = min(per_node, total // max(1, count))
    return {i: text if note is None else truncate(text, cap, note)
            for i, (text, note) in raw.items()}


@dataclass(frozen=True)
class Axis:
    lo: float
    hi: float
    cut_lo: float | None
    cut_hi: float | None


def axis_for(scores: list[float]) -> Axis:
    """The y range of the search figure. It always includes 0. With at least 8 scores, a score
    more than 3 interquartile ranges (at least 0.01) beyond a quartile is cut: it is drawn on
    the edge and left out of the range, so one outlier cannot squash the other points."""
    cut_lo = cut_hi = None
    if len(scores) >= 8:
        q1, _, q3 = statistics.quantiles(scores, n=4, method="inclusive")
        w = max(q3 - q1, 0.01)
        lo_edge, hi_edge = q1 - 3 * w, q3 + 3 * w
        cut_lo = lo_edge if any(s < lo_edge for s in scores) else None
        cut_hi = hi_edge if any(s > hi_edge for s in scores) else None
    inside = [s for s in scores if (cut_lo is None or s >= cut_lo)
              and (cut_hi is None or s <= cut_hi)]
    lo = min([0.0, *inside])
    hi = max([0.0, *inside])
    if lo == hi:
        hi = lo + 0.01
    return Axis(lo, hi, cut_lo, cut_hi)


def read_timeline(path: Path) -> list[dict]:
    """timeline.jsonl's rows; unreadable lines are skipped, a missing file is empty."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return []
    rows = []
    for line in text.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def holdout_points(timeline: list[dict]) -> dict[int, float]:
    """iteration -> held-out mean delta, from `won` and `false_positive` events that carry one."""
    out = {}
    for row in timeline:
        ho = row.get("holdout")
        if (row.get("kind") in ("won", "false_positive") and isinstance(ho, dict)
                and isinstance(ho.get("mean_rel_delta"), (int, float))
                and isinstance(row.get("iteration"), int)):
            out[row["iteration"]] = float(ho["mean_rel_delta"])
    return out
