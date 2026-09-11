import json
import stat

from talos import cli
from talos.config import Config, load, resolve_api_key, save


def scripted(answers):
    it = iter(answers)
    def ask(prompt, default=None, secret=False):
        try:
            v = next(it)
        except StopIteration:
            raise AssertionError(f"unexpected prompt: {prompt}")
        return v if v != "" else (default or "")
    return ask


def test_setup_writes_config_and_0600_secret(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = []
    monkeypatch.setattr(cli, "validate_provider", lambda p: None)
    monkeypatch.setattr(cli, "deploy_bench", lambda token_id, token_secret, run=None: calls.append("deploy"))
    rc = cli.main(["setup"], ask=scripted(["anthropic", "", "sk-test", "ak-1", "as-1"]))
    assert rc == 0
    cfg = json.loads((tmp_path / "talos.config.json").read_text())
    assert cfg["provider"] == "anthropic" and cfg["model"] == "claude-opus-5"
    sec = tmp_path / ".talos" / "secrets.json"
    assert json.loads(sec.read_text()) == {"api_key": "sk-test"}
    assert stat.S_IMODE(sec.stat().st_mode) == 0o600  # mutation: dropping chmod fails this
    assert calls == ["deploy"]


def test_setup_rejected_key_writes_nothing(tmp_path, monkeypatch):
    # mutation: writing before validation leaves a bad key on disk
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "validate_provider", lambda p: "credential rejected")
    monkeypatch.setattr(cli, "deploy_bench", lambda *a, **k: None)
    rc = cli.main(["setup"], ask=scripted(["anthropic", "", "bad", "ak", "as"]))
    assert rc != 0
    assert not (tmp_path / "talos.config.json").exists()
    assert not (tmp_path / ".talos").exists()


def test_setup_cli_provider_stores_no_secret(tmp_path, monkeypatch):
    # mutation: prompting for and saving an api_key for a CLI provider would write secrets.json
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "validate_provider", lambda p: None)
    monkeypatch.setattr(cli, "deploy_bench", lambda *a, **k: None)
    rc = cli.main(["setup"], ask=scripted(["claude-cli", "", "", "ak", "as"]))  # mode defaults
    assert rc == 0
    assert not (tmp_path / ".talos" / "secrets.json").exists()
    assert json.loads((tmp_path / "talos.config.json").read_text())["provider"] == "claude-cli"


def test_run_requires_budget_and_creates_job(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    save(tmp_path, Config(provider="fake", model="fake", mode="single-shot", api_base=None), None)
    fake_info = type("I", (), {"id": "c003", "name": "knapsack", "is_gpu": False,
                               "tracks": ["n=1"], "max_fuel": 7})()
    monkeypatch.setattr(cli, "fetch_challenge_info", lambda name: fake_info)
    started = {}
    def fake_execute(spec, store, cfg, resume):
        started.update(spec=spec)
        return 0
    monkeypatch.setattr(cli, "execute_job", fake_execute)
    rc = cli.main(["run", "--challenge", "knapsack", "--direction", "go", "--budget-iterations", "3",
                   "--yes"])
    assert rc == 0
    spec = started["spec"]
    assert spec.challenge == "knapsack" and spec.fuel == 7 and spec.tracks == ["n=1"]
    assert spec.budget.iterations == 3
    assert len(spec.rand_hash) == 64
    job_files = list((tmp_path / "runs").glob("*/job.json"))
    assert len(job_files) == 1
    # mutation: zero-budget guard using truthiness would accept a job with no cap
    rc2 = cli.main(["run", "--challenge", "knapsack", "--direction", "go", "--yes"])
    assert rc2 != 0


def test_resolve_api_key_prefers_file_then_env(tmp_path, monkeypatch):
    # mutation: reading the env var first, or treating "" as a key, fails these three asserts
    save(tmp_path, Config(provider="openai", model="gpt-5", mode="single-shot", api_base=None), "from-file")
    cfg = load(tmp_path)
    assert resolve_api_key(cfg) == "from-file"
    (tmp_path / ".talos" / "secrets.json").unlink()
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    assert resolve_api_key(load(tmp_path)) == "from-env"
    monkeypatch.setenv("OPENAI_API_KEY", "")
    assert resolve_api_key(load(tmp_path)) is None  # empty env is unset, not a key
