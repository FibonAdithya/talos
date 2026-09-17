"""The Modal app module constructs offline: importing it registers functions but contacts
nothing. These tests exercise the pure helpers around the Modal objects."""
import types

from modal_app import talos_bench


def test_content_hash_changes_with_the_monorepo_pin_and_image_tag(monkeypatch):
    files = {"mod.rs": "fn x(){}"}
    base = talos_bench.content_hash(files)
    monkeypatch.setattr(talos_bench, "MONOREPO_REF", "other")
    # mutation: leaving the ref out of the hash serves stale .so files off the persistent
    # Volume after a pin bump, so a rebuilt monorepo is never actually benchmarked
    bumped_ref = talos_bench.content_hash(files)
    assert bumped_ref != base
    monkeypatch.setattr(talos_bench, "DEV_IMAGE_TAG", "9.9.9")
    assert talos_bench.content_hash(files) not in (base, bumped_ref)


def test_content_hash_still_depends_on_the_files():
    # mutation: hashing only the pins would collapse every candidate to one cache entry
    assert talos_bench.content_hash({"mod.rs": "a"}) != talos_bench.content_hash({"mod.rs": "b"})


def _sandbox(monkeypatch, tmp_path):
    (tmp_path / "mono" / "tig-algorithms" / "src" / "knapsack").mkdir(parents=True)
    (tmp_path / "mono" / "tig-algorithms" / "src" / "knapsack" / "mod.rs").write_text("// c003\n")
    monkeypatch.setattr(talos_bench, "MONOREPO", tmp_path / "mono")
    monkeypatch.setattr(talos_bench, "ARTIFACTS", str(tmp_path / "artifacts"))
    monkeypatch.setattr(talos_bench, "volume",
                        types.SimpleNamespace(reload=lambda: None, commit=lambda: None))


def test_compile_reports_a_bad_file_map_instead_of_raising(monkeypatch, tmp_path):
    _sandbox(monkeypatch, tmp_path)
    out = talos_bench._compile_impl("knapsack", {"../evil.rs": "x"})
    # mutation: letting the ValueError escape burns the client's 15-minute retry window on a
    # deterministic error and pauses the run instead of reporting a failed compile
    assert out["ok"] is False and out["artifact_id"] is None
    assert "bench error: ValueError" in out["output"]


def test_compile_reports_a_build_that_produced_no_so(monkeypatch, tmp_path):
    _sandbox(monkeypatch, tmp_path)
    monkeypatch.setattr(talos_bench.inside, "build", lambda *a, **k: (True, "warning: unused"))
    out = talos_bench._compile_impl("knapsack", {"mod.rs": "fn x(){}"})
    # mutation: copying a .so that is not there raises FileNotFoundError out of the container
    assert out["ok"] is False and out["artifact_id"] is None
    assert "build produced no .so at" in out["output"] and "warning: unused" in out["output"]


def test_score_impl_hands_the_hyperparameters_to_run_nonce(monkeypatch, tmp_path):
    _sandbox(monkeypatch, tmp_path)
    art = tmp_path / "artifacts" / "knapsack" / "art1"
    art.mkdir(parents=True)
    (art / "algo.so").write_bytes(b"\x7fELF")
    seen = {}

    def fake_run_nonce(*args, **kwargs):
        seen.update(kwargs)
        return {"track": args[1], "nonce": args[3], "ok": True, "quality": 1, "runtime_ms": 1,
                "error": None}
    monkeypatch.setattr(talos_bench.inside, "run_nonce", fake_run_nonce)
    talos_bench._score_impl("knapsack", "c003", "art1", "t", "ab" * 32, 0, 5, 60, {"x": 1})
    # mutation: accepting the argument but not forwarding it runs Modal nonces without the map
    assert seen["hyperparameters"] == {"x": 1}
