"""Apply an LLM edit response to the algorithm files. Any block naming a path that is not
one of the algorithm's own files is rejected outright, never resolved by basename. The one
spelling accepted besides the bare name is the candidate's own directory, with or without
the directories above it (`.../talos_cand/<name>`, `talos_cand/<name>`), because that is how
the compiler prints it and a fix response copies it; another algorithm's directory with the
same basename is still rejected."""
from __future__ import annotations

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
