import pytest

from talos.edits import EditError, apply_edit_response

FILES = {"mod.rs": "fn a() {}\nfn b() {}\n", "ls.rs": "fn c() {}\n"}


def blk(path, s, r):
    return f"<<<<<<< SEARCH {path}\n{s}\n=======\n{r}\n>>>>>>> REPLACE\n"


def test_applies_blocks_to_known_files():
    # mutation: `_in_scope` returning `block.file not in files` would reject a
    # block naming a real algorithm file, leaving it unapplied
    out = apply_edit_response(FILES, blk("mod.rs", "fn a() {}", "fn a() { 1 }"))
    assert out.applied == 1 and not out.misses and not out.rejected
    assert out.files["mod.rs"].startswith("fn a() { 1 }")
    assert out.files["ls.rs"] == FILES["ls.rs"]


def test_rejects_edit_outside_algorithm_files():
    # mutation: dropping the path check lets an edit to Cargo.toml or ../x through
    out = apply_edit_response(FILES, blk("Cargo.toml", "x", "y") + blk("../mod.rs", "x", "y"))
    assert out.applied == 0
    assert sorted(out.rejected) == ["../mod.rs", "Cargo.toml"]


def test_no_blocks_is_an_error():
    # mutation: returning an empty outcome silently wastes an iteration
    with pytest.raises(EditError):
        apply_edit_response(FILES, "I would change the loop bound.")


def test_miss_reported_not_guessed():
    # mutation: swallowing misses (reporting applied=len(kept)) would hide that
    # the block never touched the file
    out = apply_edit_response(FILES, blk("mod.rs", "fn zzz() {}", "fn zzz() { 1 }"))
    assert out.applied == 0 and len(out.misses) == 1 and out.misses[0].reason == "not_found"
