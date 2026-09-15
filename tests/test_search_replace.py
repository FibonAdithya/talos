"""Self-running tests for the search/replace edit engine.

Run directly: `python scripts/test_search_replace.py` (no pytest in this repo).
"""
# Copied verbatim from tig-foundation/prometheus-swarm (scripts/test_search_replace.py), the
# upstream of talos/search_replace.py. It is exempt from this repo's "every test states the
# mutation it catches" rule: editing it would break the correspondence with upstream, which is
# what makes it useful as a conformance check when the upstream file is re-synced.

from talos.search_replace import parse_blocks, apply_blocks


def _blk(file, search, replace):
    body = f"<<<<<<< SEARCH {file}\n{search}\n=======\n{replace}\n>>>>>>> REPLACE"
    return parse_blocks(body)


def test_exact_unique():
    files = {"mod.rs": "use super::*;\nlet x = 1;\n"}
    out, misses = apply_blocks(files, _blk("mod.rs", "let x = 1;", "let x = 42;"))
    assert not misses
    assert "let x = 42;" in out["mod.rs"]


def test_whitespace_insensitive():
    files = {"mod.rs": "fn f() {\n          let x = 1;\n}\n"}
    out, misses = apply_blocks(files, _blk("mod.rs", "let x = 1;", "let x = 2;"))
    assert not misses, misses
    assert "let x = 2;" in out["mod.rs"]


def test_ambiguous_not_applied():
    files = {"mod.rs": "let a = 1;\nlet a = 1;\n"}
    out, misses = apply_blocks(files, _blk("mod.rs", "let a = 1;", "let a = 9;"))
    assert len(misses) == 1 and misses[0].reason == "ambiguous"
    assert out == files


def test_not_found_is_miss():
    files = {"mod.rs": "let a = 1;\n"}
    out, misses = apply_blocks(files, _blk("mod.rs", "let z = 0;", "let z = 1;"))
    assert len(misses) == 1 and misses[0].reason == "not_found"


def test_multifile_targeting():
    files = {"mod.rs": "mod helpers;\n", "helpers.rs": "pub fn h() -> i32 { 1 }\n"}
    out, misses = apply_blocks(
        files, _blk("helpers.rs", "pub fn h() -> i32 { 1 }", "pub fn h() -> i32 { 2 }")
    )
    assert not misses
    assert "{ 2 }" in out["helpers.rs"]
    assert out["mod.rs"] == files["mod.rs"]


def test_pathless_single_file():
    files = {"mod.rs": "let q = 0;\n"}
    out, misses = apply_blocks(
        files, parse_blocks("<<<<<<< SEARCH\nlet q = 0;\n=======\nlet q = 5;\n>>>>>>> REPLACE")
    )
    assert not misses and "let q = 5;" in out["mod.rs"]


def test_pathless_multifile_is_miss():
    # Without a path on a multi-file algorithm the target is ambiguous.
    files = {"mod.rs": "a\n", "helpers.rs": "a\n"}
    out, misses = apply_blocks(
        files, parse_blocks("<<<<<<< SEARCH\na\n=======\nb\n>>>>>>> REPLACE")
    )
    assert len(misses) == 1 and misses[0].reason == "no_file"


def test_multiple_ordered_blocks():
    files = {"mod.rs": "a\nb\nc\n"}
    body = (
        "<<<<<<< SEARCH mod.rs\na\n=======\nA\n>>>>>>> REPLACE\n"
        "<<<<<<< SEARCH mod.rs\nc\n=======\nC\n>>>>>>> REPLACE"
    )
    out, misses = apply_blocks(files, parse_blocks(body))
    assert not misses and out["mod.rs"] == "A\nb\nC\n"


def test_partial_apply_skips_only_misses():
    files = {"mod.rs": "keep1\ntarget\nkeep2\n"}
    body = (
        "<<<<<<< SEARCH mod.rs\ntarget\n=======\nCHANGED\n>>>>>>> REPLACE\n"
        "<<<<<<< SEARCH mod.rs\nNOPE\n=======\nX\n>>>>>>> REPLACE"
    )
    out, misses = apply_blocks(files, parse_blocks(body))
    assert "CHANGED" in out["mod.rs"]      # good block applied
    assert len(misses) == 1                 # bad block reported
