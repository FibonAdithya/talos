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


class FakePool:
    """Stands in for `multiprocessing.Pool`: records the worker count and yields the rows back
    to front, so the batch has to sort them itself."""
    sizes: list[int] = []

    def __init__(self, workers):
        FakePool.sizes.append(workers)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def imap_unordered(self, fn, tasks):
        return [fn(t) for t in reversed(list(tasks))]


def _artifact(tmp_path):
    art = tmp_path / "artifacts" / "knapsack" / "art1"
    art.mkdir(parents=True)
    (art / "algo.so").write_bytes(b"\x7fELF")
    return art


def _batch_task(nonce, timeout_s=60, hp=None):
    return {"track": "t", "rand_hash": "ab" * 32, "nonce": nonce, "fuel": 5,
            "timeout_s": timeout_s, "hyperparameters": hp}


def test_score_batch_runs_the_tasks_through_a_pool_and_returns_them_in_nonce_order(
        monkeypatch, tmp_path):
    _sandbox(monkeypatch, tmp_path)
    _artifact(tmp_path)
    seen = []

    def stub(task):
        seen.append(task)
        return {"track": task[1], "nonce": task[3], "ok": True, "quality": 1, "runtime_ms": 1,
                "error": None}
    monkeypatch.setattr(talos_bench.inside, "run_task", stub)
    FakePool.sizes = []
    ticks = iter([100.0, 107.5])
    out = talos_bench._score_batch_impl("knapsack", "c003", "art1",
                                        [_batch_task(n, hp={"x": n}) for n in range(4)],
                                        workers=4, pool_factory=FakePool,
                                        clock=lambda: next(ticks))
    # mutation: a pool of 1, or no pool, scores the four nonces one after another on a
    # container billed for four cores
    assert FakePool.sizes == [4]
    # mutation: returning the rows as the pool finished them hands the client a reversed batch
    assert [r["nonce"] for r in out["rows"]] == [0, 1, 2, 3]
    # the container's own wall time is what Modal bills for; the client charges from it
    assert out["seconds"] == 7.5
    so = tmp_path / "artifacts" / "knapsack" / "art1" / "algo.so"
    # mutation: dropping a tuple field runs the nonce with the wrong fuel, timeout or map
    assert {t[3]: t for t in seen}[2] == ("c003", "t", "ab" * 32, 2, str(so), 5, 60, None,
                                          str(tmp_path / "mono"), {"x": 2})


def test_score_batch_caps_each_timeout_at_the_flat_nonce_timeout(monkeypatch, tmp_path):
    from talos.inside import NONCE_TIMEOUT_S
    _sandbox(monkeypatch, tmp_path)
    _artifact(tmp_path)
    seen = []

    def stub(task):
        seen.append(task[6])
        return {"track": task[1], "nonce": task[3], "ok": True, "quality": 1, "runtime_ms": 1,
                "error": None}
    monkeypatch.setattr(talos_bench.inside, "run_task", stub)
    talos_bench._score_batch_impl("knapsack", "c003", "art1",
                                  [_batch_task(0, timeout_s=NONCE_TIMEOUT_S * 3),
                                   _batch_task(1, timeout_s=7)],
                                  workers=4, pool_factory=FakePool)
    # mutation: passing the client's timeout through unclamped lets a nonce outlive the Modal
    # function's own timeout, which kills the container and loses the whole batch
    assert sorted(seen) == [7, NONCE_TIMEOUT_S]


def test_score_batch_reports_a_missing_artifact_as_infrastructure(monkeypatch, tmp_path):
    import pytest
    _sandbox(monkeypatch, tmp_path)
    with pytest.raises(FileNotFoundError, match="missing on volume"):
        talos_bench._score_batch_impl("knapsack", "c003", "nope", [_batch_task(0)], workers=4)


def test_registered_batch_functions_carry_the_challenge_kinds_worker_count(monkeypatch):
    from talos.challenges import CHALLENGES, gpu_slug
    calls = []
    monkeypatch.setattr(talos_bench, "_score_batch_impl",
                        lambda name, cid, art, tasks, workers, **kw: calls.append(
                            (name, cid, art, tasks, workers)) or {"rows": [], "seconds": 0.0})
    app = _FakeApp()
    talos_bench.register(app, image=lambda name: f"img-{name}", probe_image=lambda: "slim")
    _kw, cpu_fn = app.registered["score_batch_knapsack"]
    _kw, gpu_fn = app.registered[f"score_batch_hypergraph_{gpu_slug('L40S')}"]
    cpu_fn("art", [_batch_task(0)])
    gpu_fn("art", [_batch_task(1)])
    # mutation: a flat worker count of 4 serialises four nonces on one GPU; a flat 1 idles the
    # CPU container's other three cores
    assert calls == [("knapsack", "c003", "art", [_batch_task(0)], CHALLENGES["knapsack"].cpu),
                     ("hypergraph", CHALLENGES["hypergraph"].id, "art", [_batch_task(1)], 1)]


def test_image_python_follows_the_deploying_interpreter(monkeypatch):
    """Functions are serialized=True, and Modal refuses to deploy a serialized function whose
    image Python differs in minor version from the interpreter that defined it."""
    seen = {}

    class _Chain:
        def __getattr__(self, _name):
            return lambda *a, **k: self

    def fake_from_registry(tag, add_python=None, **_kw):
        seen["add_python"] = add_python
        return _Chain()
    monkeypatch.setattr(talos_bench.modal.Image, "from_registry", fake_from_registry)
    monkeypatch.setattr(talos_bench, "sys", types.SimpleNamespace(version_info=(3, 13, 2)),
                        raising=False)
    talos_bench._image("knapsack")
    # mutation: a pinned "3.11" fails `modal deploy` from every interpreter that is not 3.11
    assert seen["add_python"] == "3.13"


class _FakeApp:
    def __init__(self):
        self.registered = {}

    def function(self, name, **kw):
        def deco(fn):
            self.registered[name] = (kw, fn)
            return fn
        return deco


def test_register_deploys_one_function_set_per_gpu_and_a_probe_per_gpu():
    from talos.challenges import CHALLENGES, MODAL_GPUS, gpu_slug
    app = _FakeApp()
    talos_bench.register(app, image=lambda name: f"img-{name}", probe_image=lambda: "slim")
    names = set(app.registered)
    gpu_names = [n for n, s in CHALLENGES.items() if s.is_gpu]
    cpu_names = [n for n, s in CHALLENGES.items() if not s.is_gpu]
    # CPU functions keep their names: the CPU path is untouched by the fallback
    for n in cpu_names:
        kw, _fn = app.registered[f"compile_{n}"]
        assert "gpu" not in kw and kw["cpu"] == CHALLENGES[n].cpu
        assert f"score_batch_{n}" in names
    for n in gpu_names:
        assert f"compile_{n}" not in names  # the un-suffixed GPU name would be a silent L40S
        for gpu in MODAL_GPUS:
            kw, _fn = app.registered[f"compile_{n}_{gpu_slug(gpu)}"]
            # mutation: passing the whole list as `gpu=` lets Modal pick a different GPU per
            # container, which is exactly the per-call fallback invariant 1 forbids
            assert kw["gpu"] == gpu and isinstance(kw["gpu"], str)
            assert kw["image"] == f"img-{n}"
            skw, _fn = app.registered[f"score_batch_{n}_{gpu_slug(gpu)}"]
            assert skw["gpu"] == gpu
    for gpu in MODAL_GPUS:
        kw, fn = app.registered[f"probe_{gpu_slug(gpu)}"]
        # the probe must not pull a 13 GB dev image: its cold start is the whole measurement
        assert kw["image"] == "slim" and kw["gpu"] == gpu and kw["timeout"] <= 120
        assert fn() == gpu
    expected = (2 * len(cpu_names) + 2 * len(gpu_names) * len(MODAL_GPUS) + len(MODAL_GPUS))
    assert len(names) == expected
