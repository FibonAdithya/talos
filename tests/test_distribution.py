"""The built distribution: which packages pyproject.toml ships, and what scripts/check_wheel.py
accepts. A wheel installs its top-level packages straight into site-packages, so anything other
than `talos` there can collide with another distribution's package of the same name."""
import importlib.util
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _check_wheel():
    spec = importlib.util.spec_from_file_location("check_wheel", ROOT / "scripts" / "check_wheel.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_pyproject_lists_every_package_under_talos_and_nothing_else():
    tomllib = pytest.importorskip("tomllib")
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    listed = set(cfg["tool"]["setuptools"]["packages"])
    on_disk = {".".join(p.parent.relative_to(ROOT).parts)
               for p in (ROOT / "talos").rglob("__init__.py")}
    # mutation: a package missing from the list is missing from the wheel (talos.modal_app gone
    # means `talos deploy` has nothing to deploy); a top-level package such as `modal_app`
    # lands in site-packages under its own name
    assert listed == on_disk
    assert "talos.modal_app" in listed


def _wheel(tmp_path, names):
    path = tmp_path / "talos_tig-0-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as z:
        for n in names:
            z.writestr(n, "")
    return path


def test_check_wheel_accepts_only_talos_and_its_dist_info(tmp_path, capsys):
    cw = _check_wheel()
    good = [*cw.required(), "talos_tig-0.dist-info/METADATA"]
    # mutation: a check that rejects the dist-info directory fails every real wheel
    assert cw.main([str(_wheel(tmp_path, good))]) == 0
    capsys.readouterr()
    bad = _wheel(tmp_path, [*good, "modal_app/__init__.py"])
    # mutation: checking only for missing files lets a stray top-level package ship
    assert cw.main([str(bad)]) == 1
    assert capsys.readouterr().err == f"outside talos/ in {bad}: modal_app/__init__.py\n"


def test_check_wheel_requires_the_modal_app_under_talos():
    # mutation: requiring the old top-level path passes a wheel that has no deployable app
    assert "talos/modal_app/talos_bench.py" in _check_wheel().required()
