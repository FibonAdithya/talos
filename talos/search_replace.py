"""Soft search/replace edits for API (non-agentic) algorithm mutation.

Instead of returning a whole file every iteration, the LLM emits targeted
SEARCH/REPLACE blocks (Aider-style). This is far cheaper on output tokens for
multi-file algorithms, and is the only edit mode exploiters use.

Block format (the prompt instructs exactly this):

    <<<<<<< SEARCH path/to/file.rs
    <original lines to find>
    =======
    <replacement lines>
    >>>>>>> REPLACE

The path after SEARCH is optional for a single-file algorithm (it defaults to
the sole file). Matching is deliberately *soft*: exact first, then
whitespace/indentation-insensitive. A block that can't be located to exactly
ONE place is reported as a miss (never applied to a guessed spot). Callers run a
bounded LLM repair pass on misses and then skip whatever still doesn't match —
the downstream compile-fix loop catches anything that breaks the build.

Pure functions, no I/O and no LLM — unit-testable in isolation.
"""
# Lifted from tig-foundation/prometheus-swarm scripts/search_replace.py (GPLv3).

from __future__ import annotations

import re
from dataclasses import dataclass

_SEARCH_RE = re.compile(r"^[ \t]*<{5,9}[ \t]*SEARCH[ \t]*(?P<file>.*?)[ \t]*$")
_DIVIDER_RE = re.compile(r"^[ \t]*={5,9}[ \t]*$")
_REPLACE_RE = re.compile(r"^[ \t]*>{5,9}[ \t]*REPLACE[ \t]*$")


@dataclass
class Block:
    """One SEARCH/REPLACE edit. `file` is the target relpath, or None when the
    block omitted it (the caller resolves it to the single-file entry)."""
    file: str | None
    search: str
    replace: str


@dataclass
class Miss:
    """A block that could not be applied, with a human/LLM-readable reason."""
    block: Block
    reason: str  # "not_found" | "ambiguous" | "no_file" | "empty_search"


def parse_blocks(text: str) -> list[Block]:
    """Extract all SEARCH/REPLACE blocks from an LLM response.

    Tolerant of surrounding prose and ```fences. Lines between the SEARCH
    marker and the `=======` divider are the search body; lines between the
    divider and the `>>>>>>> REPLACE` marker are the replacement."""
    blocks: list[Block] = []
    lines = text.splitlines()
    i, n = 0, len(lines)
    while i < n:
        m = _SEARCH_RE.match(lines[i])
        if not m:
            i += 1
            continue
        file = (m.group("file") or "").strip() or None
        # File path sometimes lands on the next line instead of the marker.
        i += 1
        search_lines: list[str] = []
        while i < n and not _DIVIDER_RE.match(lines[i]):
            if _SEARCH_RE.match(lines[i]):  # malformed: new block before divider
                break
            search_lines.append(lines[i])
            i += 1
        if i >= n or not _DIVIDER_RE.match(lines[i]):
            break  # unterminated block — give up on the rest
        i += 1  # skip divider
        replace_lines: list[str] = []
        while i < n and not _REPLACE_RE.match(lines[i]):
            replace_lines.append(lines[i])
            i += 1
        if i >= n or not _REPLACE_RE.match(lines[i]):
            break  # unterminated block
        i += 1  # skip REPLACE marker
        blocks.append(Block(
            file=file,
            search="\n".join(search_lines),
            replace="\n".join(replace_lines),
        ))
    return blocks


def _norm_lines(s: str) -> list[str]:
    """Whitespace-insensitive line form for fuzzy matching: strip each line and
    drop blank lines so indentation drift / trailing spaces don't block a match."""
    return [ln.strip() for ln in s.splitlines() if ln.strip()]


def _exact_spans(haystack: str, needle: str) -> list[int]:
    """Start offsets of every exact occurrence of needle in haystack."""
    out, start = [], 0
    while True:
        idx = haystack.find(needle, start)
        if idx == -1:
            return out
        out.append(idx)
        start = idx + 1


def _fuzzy_line_spans(hay_lines: list[str], needle: str) -> list[tuple[int, int]]:
    """(start, end) line-index ranges where the normalized needle matches a
    contiguous run of normalized haystack lines. Blank lines in the haystack
    are skipped when aligning, mirroring `_norm_lines`."""
    target = _norm_lines(needle)
    if not target:
        return []
    # Map each non-blank haystack line to its index.
    idx_norm = [(i, ln.strip()) for i, ln in enumerate(hay_lines) if ln.strip()]
    spans: list[tuple[int, int]] = []
    for k in range(len(idx_norm) - len(target) + 1):
        window = [norm for _, norm in idx_norm[k:k + len(target)]]
        if window == target:
            start_line = idx_norm[k][0]
            end_line = idx_norm[k + len(target) - 1][0]
            spans.append((start_line, end_line))
    return spans


def _resolve_file(block: Block, files: dict[str, str]) -> str | None:
    if block.file and block.file in files:
        return block.file
    if block.file:
        # Allow a basename match (LLM may give "mod.rs" for "sub/mod.rs" — only
        # safe when unambiguous).
        matches = [k for k in files if k.split("/")[-1] == block.file.split("/")[-1]]
        if len(matches) == 1:
            return matches[0]
        return None
    # No path given: only valid for a single-file algorithm.
    return next(iter(files)) if len(files) == 1 else None


def apply_blocks(
    files: dict[str, str], blocks: list[Block]
) -> tuple[dict[str, str], list[Miss]]:
    """Apply SEARCH/REPLACE blocks to a files-map.

    Returns (new_files, misses). Each block must resolve to exactly one location
    (exact match preferred, else a unique whitespace-insensitive match); a block
    that matches zero or multiple places is left unapplied and reported as a
    Miss. Blocks are applied in order against the evolving file content.
    """
    out = dict(files)
    misses: list[Miss] = []
    for block in blocks:
        if not block.search.strip():
            misses.append(Miss(block, "empty_search"))
            continue
        fname = _resolve_file(block, out)
        if fname is None:
            misses.append(Miss(block, "no_file"))
            continue
        content = out[fname]

        # 1) Exact match — must be unique.
        exact = _exact_spans(content, block.search)
        if len(exact) == 1:
            i = exact[0]
            out[fname] = content[:i] + block.replace + content[i + len(block.search):]
            continue
        if len(exact) > 1:
            misses.append(Miss(block, "ambiguous"))
            continue

        # 2) Fuzzy (whitespace-insensitive) line match — must be unique.
        hay_lines = content.splitlines()
        spans = _fuzzy_line_spans(hay_lines, block.search)
        if len(spans) == 1:
            start, end = spans[0]
            new_lines = hay_lines[:start] + block.replace.splitlines() + hay_lines[end + 1:]
            # Preserve a trailing newline if the original had one.
            trailing = "\n" if content.endswith("\n") else ""
            out[fname] = "\n".join(new_lines) + trailing
            continue
        if len(spans) > 1:
            misses.append(Miss(block, "ambiguous"))
            continue

        misses.append(Miss(block, "not_found"))
    return out, misses


def format_misses(misses: list[Miss]) -> str:
    """Compact, LLM-friendly description of unmatched blocks for a repair pass."""
    parts = []
    for m in misses:
        loc = m.block.file or "(unspecified file)"
        parts.append(
            f"- [{m.reason}] in {loc}:\n"
            f"  SEARCH was:\n{_indent(m.block.search)}"
        )
    return "\n".join(parts)


def _indent(s: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + ln for ln in s.splitlines())
