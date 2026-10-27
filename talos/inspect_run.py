"""Read-only views of a run directory for `talos status`, `talos show` and `talos diff`.

Everything here reads runs/<job_id>/ with plain file reads and writes nothing: JobStore is not
used, because its constructor creates the directory. The data functions return plain dicts,
which are the `--json` output (docs/json-output.md); the format_* functions render those same
dicts as text, so the two forms cannot disagree.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from talos.budget import effective_budget
from talos.report import holdout_points, read_timeline
from talos.state import TERMINAL, JobSpec, JobState

SCHEMA_VERSION = 1

# Statuses whose clock has stopped: hours spent is measured up to the last timeline event.
_STOPPED = TERMINAL | {"paused"}


class InspectError(Exception):
    """A user-facing error; str() is the message printed on stderr."""


@dataclass(frozen=True)
class Run:
    run_dir: Path
    spec: JobSpec
    state: JobState
    timeline: list[dict]

    @property
    def job_id(self) -> str:
        # The directory name, which is what the user typed; job.json's job_id matches it in
        # every run Talos writes.
        return self.run_dir.name


def _read(run_dir: Path, job_id: str, name: str, parse):
    path = run_dir / name
    if not path.is_file():
        raise InspectError(f"job {job_id} has no {name}")
    try:
        return parse(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise InspectError(f"job {job_id}: cannot read {name}: {e}") from e


def load_run(runs_root: Path, job_id: str) -> Run:
    """The run under runs_root/<job_id>/. Raises InspectError with the message to print."""
    if job_id in ("", ".", "..") or "/" in job_id or "\\" in job_id:
        raise InspectError(f"no job {job_id} under runs/")
    run_dir = Path(runs_root) / job_id
    if not run_dir.is_dir():
        raise InspectError(f"no job {job_id} under runs/")
    state = _read(run_dir, job_id, "state.json", JobState.from_dict)
    spec = _read(run_dir, job_id, "job.json", JobSpec.from_dict)
    return Run(run_dir, spec, state, read_timeline(run_dir / "timeline.jsonl"))


def holdout_status(state: JobState, iteration: int) -> str:
    if iteration in state.confirmed:
        return "confirmed"
    if iteration in state.false_positives:
        return "false_positive"
    return "not_scored"


def _stamps(timeline: list[dict]) -> list[float]:
    return [row["ts"] for row in timeline
            if isinstance(row.get("ts"), (int, float)) and not isinstance(row["ts"], bool)]


def _last_ts(timeline: list[dict]) -> float | None:
    stamps = _stamps(timeline)
    return float(stamps[-1]) if stamps else None


def hours_spent(run: Run, now: float) -> float | None:
    """Working hours: up to the last event for a job that has stopped, up to `now` otherwise."""
    started = run.state.spend.started_at
    if run.state.status in _STOPPED:
        last = _last_ts(run.timeline)
        return None if last is None else (last - started) / 3600
    return (now - started) / 3600


def last_event(timeline: list[dict]) -> dict | None:
    """The last readable event as {kind, ts, iteration, text}, or None."""
    from talos.cli import _event_line

    for row in reversed(timeline):
        ts, kind = row.get("ts"), row.get("kind")
        if not isinstance(ts, (int, float)) or isinstance(ts, bool) or not isinstance(kind, str):
            continue
        data = {k: v for k, v in row.items() if k not in ("ts", "kind")}
        try:
            text = _event_line(kind, data)
        except (KeyError, TypeError, ValueError):
            text = None
        iteration = row.get("iteration")
        return {"kind": kind, "ts": float(ts),
                "iteration": iteration if isinstance(iteration, int) else None,
                "text": text or kind}
    return None


def _llm(spec: JobSpec, llm_usd: float) -> tuple[float | None, bool, str | None]:
    """(llm_usd or None when unpriced, priced, why it is unpriced)."""
    from talos.cli import unpriced_reason

    reason = unpriced_reason(spec.provider, spec.model)
    return (llm_usd if reason is None else None), reason is None, reason


_ROW_KEYS = ("job_id", "challenge", "provider", "model", "status", "iteration",
             "best_mean_rel_delta", "llm_usd", "llm_priced", "llm_unpriced_reason",
             "compute_usd", "last_event_ts", "error")


def _best_delta(state: JobState) -> float | None:
    return state.best.delta.get("mean_rel_delta") if state.best is not None else None


def status_rows(runs_root: Path) -> list[dict]:
    """One row per directory under runs_root that holds job.json or state.json, by name. A
    directory that cannot be read is an `unreadable` row, so it never hides the others."""
    runs_root = Path(runs_root)
    if not runs_root.is_dir():
        return []
    rows = []
    for d in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        if not ((d / "job.json").is_file() or (d / "state.json").is_file()):
            continue
        row = dict.fromkeys(_ROW_KEYS)
        row["job_id"] = d.name
        try:
            run = load_run(runs_root, d.name)
        except InspectError as e:
            row["status"], row["error"] = "unreadable", str(e)
            rows.append(row)
            continue
        spec, state = run.spec, run.state
        llm_usd, priced, reason = _llm(spec, state.spend.llm_usd)
        row.update(challenge=spec.challenge, provider=spec.provider, model=spec.model,
                   status=state.status, iteration=state.iteration,
                   best_mean_rel_delta=_best_delta(state), llm_usd=llm_usd, llm_priced=priced,
                   llm_unpriced_reason=reason, compute_usd=state.spend.compute_usd,
                   last_event_ts=_last_ts(run.timeline))
        rows.append(row)
    return rows


def _cap(cap, spent) -> dict:
    left = None if cap is None or spent is None else max(0, cap - spent)
    return {"cap": cap, "spent": spent, "left": left}


def job_detail(run: Run, now: float) -> dict:
    spec, state = run.spec, run.state
    eff = effective_budget(spec.budget, state.budget_extensions)
    llm_usd, priced, reason = _llm(spec, state.spend.llm_usd)
    best = None
    if state.best is not None:
        it = state.best.iteration
        best = {"iteration": it, "title": state.best.hypothesis.get("title", ""),
                "mean_rel_delta": _best_delta(state),
                "holdout_status": holdout_status(state, it),
                "holdout_mean_rel_delta": holdout_points(run.timeline).get(it)}
    pending = None
    if state.pending_job is not None:
        hyp = state.pending_job.get("hypothesis")
        pending = {"purpose": state.pending_job.get("purpose"),
                   "title": hyp.get("title") if isinstance(hyp, dict) else None}
    return {
        "schema_version": SCHEMA_VERSION,
        "job_id": run.job_id,
        "job": spec.redacted(),
        "status": state.status,
        "stop_reason": state.stop_reason,
        "iteration": state.iteration,
        "runs_since_improvement": state.runs_since_improvement,
        "hardware": state.hardware,
        "scoring": {"requested": spec.scoring, "actual": state.scoring},
        "best": best,
        "budget": {"iterations": _cap(eff.iterations, state.spend.iterations),
                   "hours": _cap(eff.hours, hours_spent(run, now)),
                   "usd": _cap(eff.usd, llm_usd),
                   "compute_usd": _cap(eff.compute_usd, state.spend.compute_usd)},
        "llm_priced": priced,
        "llm_unpriced_reason": reason,
        "last_event_ts": _last_ts(run.timeline),
        "activity": {"pending": pending, "last_event": last_event(run.timeline)},
    }


# ── human output ─────────────────────────────────────────────────────

def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:+.3%}"


def _utc(ts: float | None) -> str:
    if ts is None:
        return "-"
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _table(header: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(r[i]) for r in [header, *rows]) for i in range(len(header))]
    return "\n".join("  ".join(cell.ljust(w) for cell, w in zip(r, widths)).rstrip()
                     for r in [header, *rows])


def format_status_table(rows: list[dict]) -> str:
    from talos.cli import _llm_spend_text

    header = ["JOB", "CHALLENGE", "STATUS", "IT", "BEST", "LLM", "COMPUTE", "LAST EVENT"]
    body = []
    for r in rows:
        if r["error"] is not None:
            body.append([r["job_id"], "-", r["status"], "-", "-", "-", "-", "-"])
            continue
        body.append([r["job_id"], r["challenge"], r["status"], str(r["iteration"]),
                     _pct(r["best_mean_rel_delta"]),
                     _llm_spend_text(r["provider"], r["model"], r["llm_usd"] or 0.0),
                     f"${r['compute_usd']:.2f}", _utc(r["last_event_ts"])])
    return _table(header, body)


def _amount(dim: str, x) -> str:
    if x is None:
        return "-"
    if dim == "iterations":
        return str(x)
    if dim == "hours":
        return f"{x:.2f}h"
    return f"${x:.2f}"


def format_job_detail(d: dict) -> str:
    job = d["job"]
    lines = [f"job: {d['job_id']}",
             f"challenge: {job['challenge']}",
             f"track focus: {job.get('track') or 'all tracks'}",
             f"provider: {job['provider']} / {job['model']}",
             f"mode: {job['mode']}  search: {job.get('search')}  start: {job.get('start')}",
             f"scoring: {d['scoring']['requested']}"
             + (f" (running {d['scoring']['actual']})"
                if d["scoring"]["actual"] not in (None, d["scoring"]["requested"]) else ""),
             f"created: {_utc(job.get('created_at'))}",
             f"status: {d['status']}",
             f"stop reason: {d['stop_reason'] or '-'}",
             f"iteration: {d['iteration']} ({d['runs_since_improvement']} since improvement)"]
    b = d["best"]
    if b is None:
        lines.append("best: none")
    else:
        held = b["holdout_status"].replace("_", " ")
        if b["holdout_mean_rel_delta"] is not None:
            held += f" ({_pct(b['holdout_mean_rel_delta'])})"
        lines.append(f"best: #{b['iteration']} {_pct(b['mean_rel_delta'])} training, "
                     f"held-out {held}: {b['title']}")
    from talos.cli import _llm_spend_text

    lines.append("budget:")
    unpriced = (None if d["llm_priced"]
                else _llm_spend_text(job["provider"], job["model"], 0.0, why=True))
    for dim, label in (("iterations", "iterations"), ("hours", "hours"), ("usd", "llm"),
                       ("compute_usd", "compute")):
        c = d["budget"][dim]
        cap = "∞" if c["cap"] is None else _amount(dim, c["cap"])
        if dim == "usd" and unpriced:
            lines.append(f"  {label}: cap {cap}, spent {unpriced}")
            continue
        lines.append(f"  {label}: cap {cap}, spent {_amount(dim, c['spent'])}, "
                     f"left {'∞' if c['cap'] is None else _amount(dim, c['left'])}")
    act = d["activity"]
    if act["pending"] is not None:
        p = act["pending"]
        what = f"iteration {p['purpose']}" if isinstance(p["purpose"], int) else str(p["purpose"])
        lines.append(f"in flight: {what}" + (f" ({p['title']})" if p["title"] else ""))
    ev = act["last_event"]
    lines.append("last event: -" if ev is None
                 else f"last event: {_utc(ev['ts'])} {ev['text']}")
    lines.append(f"hardware: {d['hardware'] or 'default'}")
    return "\n".join(lines)
