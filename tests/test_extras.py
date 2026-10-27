import importlib.util
import json

import pytest

from talos import cli, doctor, extras
from talos.config import Config, save

_real_find_spec = importlib.util.find_spec


def hide(monkeypatch, *names):
    """Make find_spec report the named modules as not installed."""
    def find_spec(name, *a, **k):
        return None if name in names else _real_find_spec(name, *a, **k)
    monkeypatch.setattr("talos.extras.importlib.util.find_spec", find_spec)


def reached(*a, **k):
    raise AssertionError("reached")


def stub_run_paths(monkeypatch):
    """A run that gets past the extras check must fail loudly, not build a provider or bench."""
    monkeypatch.setattr(cli, "execute_job", reached)
    monkeypatch.setattr(cli, "make_bench", reached)
    monkeypatch.setattr(cli, "make_provider", reached)


def scripted(answers, prompts):
    it = iter(answers)

    def ask(prompt, default=None, secret=False):
        prompts.append(prompt)
        v = next(it)
        return v if v != "" else (default or "")
    return ask


RUN = ["run", "--challenge", "knapsack", "--direction", "x", "--budget-iterations", "1", "--yes"]


def test_require_names_the_extra_and_both_install_forms(monkeypatch):
    hide(monkeypatch, "modal")
    with pytest.raises(cli.ConfigError) as e:
        extras.require("modal", "the modal backend")
    msg = str(e.value)
    assert msg.startswith("the modal backend needs the modal extra")
    assert "talos-tig[modal]" in msg and "pip install -e '.[modal]'" in msg


def test_require_passes_when_installed(monkeypatch):
    hide(monkeypatch, "anthropic")
    extras.require("modal", "the modal backend")  # not hidden: the suite env has it


def test_setup_modal_without_extra_exits_2_before_asking_for_a_token(
        tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    hide(monkeypatch, "modal")
    prompts = []
    rc = cli.main(["setup"], ask=scripted(["modal", "anthropic", "", "sk", "SHOULD-NOT-BE-ASKED"],
                                          prompts))
    assert rc == 2
    assert not (tmp_path / "talos.config.json").exists()
    assert len(prompts) == 1, prompts
    assert "talos-tig[modal]" in capsys.readouterr().err


def test_setup_anthropic_without_extra_exits_2(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    hide(monkeypatch, "anthropic")
    monkeypatch.setattr(cli, "validate_provider", reached)
    rc = cli.main(["setup"], ask=scripted(["local", "anthropic", "", "sk"], []))
    assert rc == 2
    assert not (tmp_path / "talos.config.json").exists()
    assert "talos-tig[anthropic]" in capsys.readouterr().err


def test_run_without_modal_extra_creates_no_job(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    save(tmp_path, Config(provider="claude-cli", model="m", mode="single-shot", api_base=None,
                          backend="modal"), None)
    hide(monkeypatch, "modal")
    stub_run_paths(monkeypatch)
    assert cli.main(RUN) == 2
    assert not (tmp_path / "runs").exists()
    assert "talos-tig[modal]" in capsys.readouterr().err


def test_run_without_anthropic_extra_creates_no_job(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    save(tmp_path, Config(provider="anthropic", model="m", mode="single-shot", api_base=None,
                          backend="local"), "sk")
    hide(monkeypatch, "anthropic")
    stub_run_paths(monkeypatch)
    assert cli.main(RUN) == 2
    assert not (tmp_path / "runs").exists()
    assert "talos-tig[anthropic]" in capsys.readouterr().err


def test_resume_checks_the_jobs_provider_not_the_configs(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    save(tmp_path, Config(provider="fake", model="fake", mode="single-shot", api_base=None), None)
    monkeypatch.setattr(cli, "fetch_challenge_info", lambda name: type(
        "I", (), {"id": "c003", "name": "knapsack", "is_gpu": False, "tracks": ["n=1"],
                  "max_fuel": 7})())
    assert cli.main(RUN + ["--fake"]) == 0
    run_dir = next((tmp_path / "runs").glob("*/job.json")).parent
    spec = json.loads((run_dir / "job.json").read_text())
    spec["provider"] = "anthropic"
    (run_dir / "job.json").write_text(json.dumps(spec))
    save(tmp_path, Config(provider="claude-cli", model="m", mode="single-shot", api_base=None,
                          backend="local"), None)
    hide(monkeypatch, "anthropic")
    stub_run_paths(monkeypatch)
    capsys.readouterr()
    assert cli.main(["run", "--resume", run_dir.name]) == 2
    assert "talos-tig[anthropic]" in capsys.readouterr().err


def test_fake_run_and_demo_need_no_extras(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    hide(monkeypatch, "modal", "anthropic")
    monkeypatch.setattr(cli, "fetch_challenge_info", lambda name: type(
        "I", (), {"id": "c003", "name": "knapsack", "is_gpu": False, "tracks": ["n=1"],
                  "max_fuel": 7})())
    assert cli.main(["demo", "--delay", "0"]) == 0
    assert cli.main(RUN + ["--fake"]) == 0


def test_compile_with_modal_backend_needs_the_extra(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.rs").write_text("fn main() {}")
    hide(monkeypatch, "modal")
    monkeypatch.setattr(cli, "make_bench", reached)
    rc = cli.main(["compile", "--challenge", "knapsack", "--dir", "src", "--backend", "modal"])
    assert rc == 2
    assert "talos-tig[modal]" in capsys.readouterr().err


def test_doctor_reports_a_missing_extra(monkeypatch):
    hide(monkeypatch, "modal", "anthropic")
    monkeypatch.setattr(doctor, "modal_lookup", reached)
    monkeypatch.setattr(cli, "make_provider", reached)
    b = doctor.check_backend(Config(provider="claude-cli", model="m", mode="single-shot",
                                    api_base=None, backend="modal"))
    p = doctor.check_provider(Config(provider="anthropic", model="m", mode="single-shot",
                                     api_base=None, backend="local"))
    assert b.name == "backend" and b.status == doctor.FAIL and "talos-tig[modal]" in b.detail
    assert p.name == "provider" and p.status == doctor.FAIL and "talos-tig[anthropic]" in p.detail
