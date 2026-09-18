"""How Talos reaches C3: the `c3` CLI, or the hosted MCP endpoint when a key is configured."""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Protocol

from talos.bench import _redact
from talos.c3_bench import C3CommandError, c3_env, parse_json_stdout
from talos.executables import argv0


class C3Transport(Protocol):
    def whoami(self) -> dict: ...
    def balance_gbp(self) -> float | None: ...
    def deploy(self, job_dir: Path) -> str: ...
    def status(self, job_id: str) -> str: ...
    def cancel(self, job_id: str) -> None: ...
    def fetch(self, job_id: str, name: str, dest: Path) -> bool: ...


class CliTransport:
    """Subprocess `c3`. Uses the `c3 login` session, or C3_API_KEY when a key is given."""

    name = "cli"

    def __init__(self, run=subprocess.run, api_key: str | None = None):
        self._run = run
        self._env = c3_env(api_key)
        self._pulled: dict[tuple[str, str], Path] = {}

    def _c3(self, *args: str, cwd: Path | None = None, timeout: int = 600) -> str:
        try:
            r = self._run([argv0("c3"), *args], capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout,
                          cwd=str(cwd) if cwd else None, env=self._env)
        except OSError as e:
            # FileNotFoundError included: a `c3` that is not on PATH must pause the run like
            # any other CLI failure, not traceback out of evaluate into "job failed".
            raise C3CommandError(f"c3 {args[0]} could not be run: "
                                 f"{_redact(str(e))[:200]}") from None
        except subprocess.TimeoutExpired:
            # Not an OSError. A CLI call that hangs past its timeout is a CLI failure too.
            raise C3CommandError(f"c3 {args[0]} timed out after {timeout}s") from None
        if r.returncode != 0:
            raise C3CommandError(f"c3 {args[0]} failed ({r.returncode}): "
                                 f"{_redact((r.stderr or r.stdout)[-500:])}")
        return r.stdout

    def whoami(self) -> dict:
        return {"text": self._c3("whoami", timeout=60)}

    def balance_gbp(self) -> float | None:
        m = re.search(r"Credit balance:\s*£([0-9.]+)", self._c3("balance", timeout=60))
        return float(m.group(1)) if m else None

    def deploy(self, job_dir: Path) -> str:
        return parse_json_stdout(self._c3("deploy", "--json", cwd=job_dir))["id"]

    def status(self, job_id: str) -> str:
        for row in parse_json_stdout(self._c3("squeue", "--json", timeout=60)):
            if row.get("job_id") == job_id:
                return str(row.get("status", "UNKNOWN")).upper()
        raise C3CommandError(f"job {job_id} not listed by squeue")

    def cancel(self, job_id: str) -> None:
        try:
            self._c3("cancel", job_id, timeout=60)
        except C3CommandError:
            pass  # best effort: the job may already be terminal

    def _pull(self, job_id: str, root: Path) -> Path:
        root.mkdir(parents=True, exist_ok=True)  # a reattach never wrote the job dir
        pulled = root / job_id
        d = pulled
        for attempt in range(2):
            try:
                doc = parse_json_stdout(self._c3("pull", job_id, "--json", cwd=root))
            except (C3CommandError, ValueError):
                if attempt == 1:  # a transient pull failure gets one retry, same job id
                    raise
                continue
            jobs = doc.get("jobs") or []
            d = Path(jobs[0]["directory"]) if jobs and jobs[0].get("directory") else pulled
            if not d.is_absolute():
                d = root / d
            for cand in (d / "artifacts", d):
                if (cand / "results.json").exists() or (cand / "build.log").exists():
                    return cand
            shutil.rmtree(pulled, ignore_errors=True)  # a "skipped" pull with nothing on disk
        return d

    def fetch(self, job_id: str, name: str, dest: Path) -> bool:
        dest = Path(dest)
        # `c3 pull` run in X writes X/<job_id>/artifacts/<name>. When dest is that path, pull in
        # X so the file lands on dest: the layout C3Bench has always left on disk.
        in_place = dest.parent.name == "artifacts" and dest.parent.parent.name == job_id
        root = dest.parents[2] if in_place else dest.parent
        key = (job_id, str(root))
        if key not in self._pulled:  # one pull per job, however many files are fetched
            self._pulled[key] = self._pull(job_id, root)
        src = self._pulled[key] / name
        if not src.exists():
            return False
        if src.resolve() != dest.resolve():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
        return True


def make_transport(api_key: str | None, run=subprocess.run) -> C3Transport:
    if api_key:
        from talos.c3_mcp import McpTransport
        return McpTransport(api_key)
    return CliTransport(run=run)
