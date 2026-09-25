"""Read rustc's build output for what concerns the candidate. The monorepo build compiles every
module the challenge crate lists; before staging pruned the crate to the candidate the output
was dominated by warnings from algorithms Talos does not touch, and a fix prompt fed that spam
has edited those algorithms' paths. The filter stays: tig-challenges and the standard library
are still in the build. Pure functions, no I/O; copied into the C3 job directory alongside
inside.py."""
from __future__ import annotations

import re

from talos.inside import ALGO_NAME

_DIAG_START = re.compile(r"^(warning|error)(\[E\d+\])?:")
_LOCATION = re.compile(r"^\s*-->\s*(\S+?):\d+", re.M)
# rustc groups the unused methods of one impl: "methods `a`, `b`, and `c` are never used".
_NEVER_USED = re.compile(
    r"^warning: (?:functions?|methods?|associated functions?) "
    r"((?:`\w+`(?:, and |, | and )?)+) (?:is|are) never used\n"
    r"\s*-->\s*(\S+?):\d+", re.M)


def _blocks(output: str) -> list[str]:
    """A block is one diagnostic (from its `warning:`/`error:` line to the next blank line) or
    the run of non-diagnostic lines between two of them. Blocks keep their newlines."""
    out, cur = [], []
    for line in output.splitlines(keepends=True):
        if _DIAG_START.match(line) and cur:
            out.append("".join(cur))
            cur = []
        cur.append(line)
        if line.strip() == "":
            out.append("".join(cur))
            cur = []
    if cur:
        out.append("".join(cur))
    return out


def _own(path: str) -> bool:
    return f"/{ALGO_NAME}/" in path


def relevant(output: str) -> str:
    """The build output minus warning blocks located in files that are not the candidate's.
    Errors, warnings without a location (the summary), and non-diagnostic lines all stay."""
    kept = []
    for block in _blocks(output):
        if block.startswith("warning"):
            loc = _LOCATION.search(block)
            if loc and not _own(loc.group(1)):
                continue
        kept.append(block)
    return "".join(kept)


_FN_DEF = re.compile(r"\bfn\s+(\w+)")


def defined_functions(text: str) -> list[str]:
    """Names of every `fn` the file defines, sorted. Cheap to ship in a job payload, unlike
    the file itself."""
    return sorted(set(_FN_DEF.findall(text)))


def dead_new_functions(output: str, prior_functions: dict[str, list[str]]) -> list[str]:
    """Functions the candidate defines, that the code it was edited from did not, that nothing
    calls: the candidate compiled, but the change is not on the solve path and scoring it
    repeats the prior result exactly. Each entry is `<file>: <name>`."""
    found = []
    for names, path in _NEVER_USED.findall(output):
        if not _own(path):
            continue
        rel = path.split(f"/{ALGO_NAME}/", 1)[1]
        for name in re.findall(r"`(\w+)`", names):
            if name not in prior_functions.get(rel, ()):
                found.append(f"{rel}: {name}")
    return found


def first_error(output: str) -> str:
    """The line that says what went wrong: rustc's first `error` line, which comes before
    cargo's closing summaries; with no error line at all, the last non-empty line."""
    lines = [ln.strip() for ln in output.splitlines() if ln.strip()]
    errors = [ln for ln in lines if ln.startswith("error")]
    return (errors or lines[-1:] or [""])[0]
