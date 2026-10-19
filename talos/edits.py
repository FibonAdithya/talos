"""Apply an LLM edit response to the algorithm files. Any block naming a path that is not
one of the algorithm's own files is rejected outright, never resolved by basename. The one
spelling accepted besides the bare name is the candidate's own directory, with or without
the directories above it (`.../talos_cand/<name>`, `talos_cand/<name>`), because that is how
the compiler prints it and a fix response copies it; another algorithm's directory with the
same basename is still rejected."""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

from talos.inside import ALGO_NAME
from talos.search_replace import Block, Miss, apply_blocks, parse_blocks


class EditError(ValueError):
    pass


@dataclass
class EditOutcome:
    files: dict[str, str]
    applied: int
    misses: list[Miss] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)


def _resolve(path: str, files: dict[str, str]) -> str | None:
    """The algorithm-file name `path` denotes, or None when it is out of scope."""
    if path in files:
        return path
    head, sep, name = path.rpartition(f"{ALGO_NAME}/")
    if sep and (head == "" or head.endswith("/")) and name in files:
        return name
    return None


def _in_scope(block: Block, files: dict[str, str]) -> bool:
    if block.file is None:
        return len(files) == 1
    return _resolve(block.file, files) is not None


def apply_edit_response(files: dict[str, str], response_text: str) -> EditOutcome:
    blocks = parse_blocks(response_text)
    if not blocks:
        raise EditError("response contained no SEARCH/REPLACE blocks")
    rejected = [b.file for b in blocks if not _in_scope(b, files)]
    kept = [b if b.file is None else replace(b, file=_resolve(b.file, files))
            for b in blocks if _in_scope(b, files)]
    new_files, misses = apply_blocks(files, kept)
    applied = len(kept) - len(misses)
    return EditOutcome(files=new_files, applied=applied, misses=misses,
                       rejected=[r for r in rejected if r is not None])


_FILE_RE = re.compile(r"^[ \t]*<{5,9}[ \t]*FILE[ \t]+(?P<file>.*?)[ \t]*$")
_END_FILE_RE = re.compile(r"^[ \t]*>{5,9}[ \t]*END FILE[ \t]*$")
_FENCE_RE = re.compile(r"^[ \t]*```[\w+-]*[ \t]*$")


def _parse_file_blocks(text: str) -> list[tuple[str, str]]:
    """(path, body) for each `<<<<<<< FILE <path>` ... `>>>>>>> END FILE` block, in order.
    Prose between blocks is ignored. One code-fence line opening the body and one closing it
    are dropped. A block with no end marker ends the scan."""
    blocks: list[tuple[str, str]] = []
    lines = text.splitlines()
    i, n = 0, len(lines)
    while i < n:
        m = _FILE_RE.match(lines[i])
        i += 1
        if not m or not m.group("file"):
            continue
        body: list[str] = []
        while i < n and not _END_FILE_RE.match(lines[i]):
            body.append(lines[i])
            i += 1
        if i >= n:
            break
        i += 1
        if body and _FENCE_RE.match(body[0]):
            body = body[1:]
        if body and _FENCE_RE.match(body[-1]):
            body = body[:-1]
        blocks.append((m.group("file"), "".join(line + "\n" for line in body)))
    return blocks


def apply_whole_file_response(files: dict[str, str], response_text: str) -> EditOutcome:
    """Replace whole files from FILE blocks. A block naming a path outside `files` is listed in
    `rejected` and not applied, so the file set never grows. For a file named twice the last
    block wins."""
    blocks = _parse_file_blocks(response_text)
    if not blocks:
        raise EditError("response contained no FILE blocks")
    new_files = dict(files)
    replaced: set[str] = set()
    rejected: list[str] = []
    for path, body in blocks:
        name = _resolve(path, files)
        if name is None:
            rejected.append(path)
            continue
        new_files[name] = body
        replaced.add(name)
    return EditOutcome(files=new_files, applied=len(replaced), misses=[], rejected=rejected)
