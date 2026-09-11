"""Apply an LLM edit response to the algorithm files. Any block naming a path that is not
one of the algorithm's own files is rejected outright, never resolved by basename."""
from __future__ import annotations

from dataclasses import dataclass, field

from talos.search_replace import Block, Miss, apply_blocks, parse_blocks


class EditError(ValueError):
    pass


@dataclass
class EditOutcome:
    files: dict[str, str]
    applied: int
    misses: list[Miss] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)


def _in_scope(block: Block, files: dict[str, str]) -> bool:
    if block.file is None:
        return len(files) == 1
    return block.file in files


def apply_edit_response(files: dict[str, str], response_text: str) -> EditOutcome:
    blocks = parse_blocks(response_text)
    if not blocks:
        raise EditError("response contained no SEARCH/REPLACE blocks")
    rejected = [b.file for b in blocks if not _in_scope(b, files)]
    kept = [b for b in blocks if _in_scope(b, files)]
    new_files, misses = apply_blocks(files, kept)
    applied = len(kept) - len(misses)
    return EditOutcome(files=new_files, applied=applied, misses=misses,
                       rejected=[r for r in rejected if r is not None])
