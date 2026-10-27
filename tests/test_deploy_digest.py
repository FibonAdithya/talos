"""The marker function that tells a client whether the deployed Modal app was built from the
Talos source it is running."""
import ast
import re
from pathlib import Path

from modal_app import talos_bench
from talos.deploy_digest import DEPLOYED_FILES, digest, marker_name

ROOT = Path(__file__).resolve().parent.parent


def _module_files(sub: str) -> list[str]:
    """The file a dotted module path names: its .py, or its package __init__.py."""
    out = []
    if (ROOT / f"{sub}.py").is_file():
        out.append(f"{sub}.py")
    if (ROOT / sub / "__init__.py").is_file():
        out.append(f"{sub}/__init__.py")
    return out


def _imports(path: Path) -> set[str]:
    """Repo-relative files that `path` imports from `talos` or `modal_app`, function-level
    imports included. Relative imports are resolved against the importing file's package."""
    found = set()
    pkg = path.relative_to(ROOT).parent.parts
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            mods = [(a.name, []) for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = list(pkg[:len(pkg) - (node.level - 1)])
                mods = [(".".join(base + ([node.module] if node.module else [])),
                         [a.name for a in node.names])]
            else:
                mods = [(node.module, [a.name for a in node.names])]
        else:
            continue
        for mod, names in mods:
            parts = mod.split(".")
            if parts[0] not in ("talos", "modal_app"):
                continue
            found.add(f"{parts[0]}/__init__.py")
            found.update(_module_files("/".join(parts)))
            for n in names:  # `from talos import x`
                found.update(_module_files("/".join(parts + [n])))
    return found


def test_deployed_files_are_the_import_closure_of_the_modal_app():
    closure = {"modal_app/__init__.py", "modal_app/talos_bench.py"}
    todo = list(closure)
    while todo:
        for dep in _imports(ROOT / todo.pop()):
            if dep not in closure:
                closure.add(dep)
                todo.append(dep)
    # mutation: dropping a file from DEPLOYED_FILES, or a function-level `from talos import x`
    # added to a deployed file, leaves the digest blind to code the containers run
    assert closure == set(DEPLOYED_FILES)


def _copy(dst: Path, crlf: bool = False) -> Path:
    for rel in DEPLOYED_FILES:
        data = (ROOT / rel).read_bytes().replace(b"\r\n", b"\n")
        if crlf:
            data = data.replace(b"\n", b"\r\n")
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        (dst / rel).write_bytes(data)
    return dst


def test_digest_ignores_crlf(tmp_path):
    a, b = _copy(tmp_path / "a"), _copy(tmp_path / "b", crlf=True)
    assert (b / DEPLOYED_FILES[0]).read_bytes().count(b"\r\n") > 0 or \
        (b / DEPLOYED_FILES[1]).read_bytes().count(b"\r\n") > 0
    # mutation: hashing raw bytes makes a Windows checkout never match a Linux deploy
    assert digest(a) == digest(b)


def test_digest_changes_with_any_deployed_file(tmp_path):
    base = digest(_copy(tmp_path / "base"))
    for i, rel in enumerate(DEPLOYED_FILES):
        root = _copy(tmp_path / f"c{i}")
        with open(root / rel, "ab") as f:
            f.write(b"#")
        # mutation: a file left out of the hash is code that can change without a redeploy
        assert digest(root) != base, rel


def test_digest_is_12_hex():
    assert re.fullmatch(r"[0-9a-f]{12}", digest())
    assert marker_name() == f"deployed_{digest()}"


def test_digest_depends_on_the_path_not_only_the_bytes(tmp_path):
    a = _copy(tmp_path / "a")
    b = _copy(tmp_path / "b")
    (b / "talos/inside.py").write_bytes((a / "talos/native_runner.py").read_bytes())
    (b / "talos/native_runner.py").write_bytes((a / "talos/inside.py").read_bytes())
    # mutation: hashing bytes without the path lets a swap of two files go unnoticed
    assert digest(a) != digest(b)


class _App:
    def __init__(self):
        self.registered = {}

    def function(self, name, **kw):
        def deco(fn):
            self.registered[name] = (kw, fn)
            return fn
        return deco


def test_register_adds_the_marker():
    app = _App()
    talos_bench.register(app, image=lambda n: "img", probe_image=lambda: "slim")
    kw, fn = app.registered[marker_name()]
    # mutation: a marker that is not registered makes every client report a stale deploy
    assert kw["image"] == "slim" and kw["serialized"] is True and kw["timeout"] == 10
    assert fn() is None


def test_the_marker_is_a_closure_not_a_module_level_function():
    app = _App()
    talos_bench.register(app, image=lambda n: "img", probe_image=lambda: "slim")
    _kw, fn = app.registered[marker_name()]
    # mutation: a module-level function is pickled by reference, and the probe image has no
    # talos source to import it from
    assert "<locals>" in fn.__qualname__
    assert not hasattr(talos_bench, fn.__name__)


def test_deployed_paths_are_under_the_two_packages_the_image_carries():
    # the compile/score image adds only these two packages (add_local_python_source)
    assert all(p.split("/")[0] in ("modal_app", "talos") for p in DEPLOYED_FILES)


def test_the_walker_resolves_packages_and_relative_imports(tmp_path, monkeypatch):
    import tests.test_deploy_digest as me
    (tmp_path / "talos" / "providers").mkdir(parents=True)
    for f in ("talos/__init__.py", "talos/providers/__init__.py", "talos/providers/x.py",
              "talos/y.py"):
        (tmp_path / f).write_text("")
    mod = tmp_path / "talos" / "a.py"
    mod.write_text("from talos import providers\nimport talos.providers.x\nfrom . import y\n"
                   "def f():\n    from .providers import x\n")
    monkeypatch.setattr(me, "ROOT", tmp_path)
    # mutation: skipping packages or relative imports lets a deployed file import code the
    # digest does not cover
    assert me._imports(mod) == {"talos/__init__.py", "talos/providers/__init__.py",
                                "talos/providers/x.py", "talos/y.py"}
