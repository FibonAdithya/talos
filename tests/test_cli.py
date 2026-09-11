import json
import stat
import types

import pytest

from talos import cli
from talos.config import Config, ConfigError, load, resolve_api_key, save
from talos.mainnet import MainnetError
from talos.types import CompileResult


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
    # mutation: dropping the default leaves Modal spend unbounded
    assert spec.budget.modal_usd == 20.0
    assert len(spec.rand_hash) == 64
    job_files = list((tmp_path / "runs").glob("*/job.json"))
    assert len(job_files) == 1
    # mutation: zero-budget guard using truthiness would accept a job with no cap; applying the
    # Modal default before validation would let it stand in for the missing LLM/time cap
    rc2 = cli.main(["run", "--challenge", "knapsack", "--direction", "go", "--yes"])
    assert rc2 == 2


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


def fake_config(root):
    """A config whose provider is `fake`: execute_job then drives the whole loop with
    FakeProvider/FakeBench, no network and no Modal."""
    save(root, Config(provider="fake", model="fake", mode="single-shot", api_base=None), None)


def knapsack_info():
    return type("I", (), {"id": "c003", "name": "knapsack", "is_gpu": False,
                          "tracks": ["n=1"], "max_fuel": 7})()


def fake_run(monkeypatch, extra=()):
    monkeypatch.setattr(cli, "fetch_challenge_info", lambda name: knapsack_info())
    return cli.main(["run", "--challenge", "knapsack", "--direction", "go",
                     "--budget-iterations", "5", "--yes", *extra])


def test_run_mode_flag_overrides_config_and_gates_agentic(tmp_path, monkeypatch):
    # mutation: dropping --mode, or letting agentic through for an API provider, fails these
    monkeypatch.chdir(tmp_path)
    save(tmp_path, Config(provider="claude-cli", model="claude-opus-5", mode="single-shot",
                          api_base=None), None)
    monkeypatch.setattr(cli, "fetch_challenge_info", lambda name: knapsack_info())
    seen = {}
    monkeypatch.setattr(cli, "execute_job",
                        lambda spec, store, cfg, resume: seen.update(cfg=cfg, spec=spec) or 0)
    rc = cli.main(["run", "--challenge", "knapsack", "--direction", "go", "--mode", "agentic",
                   "--budget-iterations", "1", "--yes"])
    assert rc == 0
    assert seen["cfg"].mode == "agentic" and seen["spec"].mode == "agentic"
    # the override is per-run: talos.config.json is not rewritten
    assert json.loads((tmp_path / "talos.config.json").read_text())["mode"] == "single-shot"
    save(tmp_path, Config(provider="openai", model="gpt-5", mode="single-shot", api_base=None), None)
    rc = cli.main(["run", "--challenge", "knapsack", "--direction", "go", "--mode", "agentic",
                   "--budget-iterations", "1", "--yes"])
    assert rc == 2


def test_execute_job_refuses_without_credential(tmp_path, monkeypatch, capsys):
    # mutation: without the check the loop starts and spends Modal time before the first 401
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    save(tmp_path, Config(provider="openai", model="gpt-5", mode="single-shot", api_base=None), None)
    monkeypatch.setattr(cli, "fetch_challenge_info", lambda name: knapsack_info())
    rc = cli.main(["run", "--challenge", "knapsack", "--direction", "go",
                   "--budget-iterations", "1", "--yes"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "no API key for provider openai" in err and "OPENAI_API_KEY" in err
    assert list((tmp_path / "runs").glob("*/job.json"))  # the job dir was written
    assert not list((tmp_path / "runs").glob("*/state.json"))  # but nothing ran


def test_fake_run_end_to_end_wins_and_packages(tmp_path, monkeypatch, capsys):
    # mutation: dropping the delta line, the status line or build_package fails these asserts
    monkeypatch.chdir(tmp_path)
    fake_config(tmp_path)
    rc = fake_run(monkeypatch)
    out = capsys.readouterr().out
    assert rc == 0
    assert "Status: won" in out
    assert "Best delta vs baseline: +1.000%" in out
    assert "[status] job=" in out and "best=+1.000%" in out and "left=∞" in out
    run_dir = next((tmp_path / "runs").glob("*/job.json")).parent
    assert f"Package: {run_dir / 'package'}" in out
    assert (run_dir / "package" / "scores.md").exists()
    best = json.loads((run_dir / "state.json").read_text())["best"]
    assert "let k = 2;" in best["files"]["mod.rs"]


def test_resume_restarts_a_cancelled_job(tmp_path, monkeypatch, capsys):
    # mutation: resuming a cancelled job with a terminal status makes run() return immediately
    monkeypatch.chdir(tmp_path)
    fake_config(tmp_path)
    assert fake_run(monkeypatch) == 0
    run_dir = next((tmp_path / "runs").glob("*/job.json")).parent
    st = json.loads((run_dir / "state.json").read_text())
    st.update(status="cancelled", stop_reason="user requested stop", best=None, iteration=0,
              confirmed=[], hypotheses=[])
    st["spend"]["iterations"] = 0
    (run_dir / "state.json").write_text(json.dumps(st))
    capsys.readouterr()
    rc = cli.main(["run", "--resume", run_dir.name])
    out = capsys.readouterr().out
    assert rc == 0 and "Status: won" in out
    events = [json.loads(ln)["kind"]
              for ln in (run_dir / "timeline.jsonl").read_text().splitlines()]
    assert "resumed" in events


def test_resume_of_finished_job_and_of_missing_job(tmp_path, monkeypatch, capsys):
    # mutation: without the terminal-status check a won job silently re-enters the loop and
    # without the job.json check --resume mkdirs a stray empty run directory
    monkeypatch.chdir(tmp_path)
    fake_config(tmp_path)
    assert fake_run(monkeypatch) == 0
    run_dir = next((tmp_path / "runs").glob("*/job.json")).parent
    capsys.readouterr()
    assert cli.main(["run", "--resume", run_dir.name]) == 1
    assert "already won; nothing to resume" in capsys.readouterr().out
    assert cli.main(["run", "--resume", "20990101-000000-knapsack"]) == 2
    assert not (tmp_path / "runs" / "20990101-000000-knapsack").exists()


def test_run_reports_mainnet_failure_and_creates_nothing(tmp_path, monkeypatch, capsys):
    # mutation: creating the job dir before fetch_challenge_info leaves an empty run behind
    monkeypatch.chdir(tmp_path)
    fake_config(tmp_path)
    url = "https://mainnet-api.tig.foundation/get-block"
    def boom(name):
        raise MainnetError(f"network error fetching {url}: x")
    monkeypatch.setattr(cli, "fetch_challenge_info", boom)
    rc = cli.main(["run", "--challenge", "knapsack", "--direction", "go",
                   "--budget-iterations", "1", "--yes"])
    assert rc == 1
    assert url in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()


def test_direction_file_is_read_and_conflicts_are_refused(tmp_path, monkeypatch, capsys):
    # mutation: ignoring --direction-file starts the job with an empty direction and an empty
    # first tacit entry
    monkeypatch.chdir(tmp_path)
    fake_config(tmp_path)
    monkeypatch.setattr(cli, "fetch_challenge_info", lambda name: knapsack_info())
    seen = {}
    monkeypatch.setattr(cli, "execute_job",
                        lambda spec, store, cfg, resume: seen.update(spec=spec) or 0)
    (tmp_path / "dir.md").write_text("explore beam search\n")
    assert cli.main(["run", "--challenge", "knapsack", "--direction-file", "dir.md",
                     "--budget-iterations", "1", "--yes"]) == 0
    assert seen["spec"].direction.strip() == "explore beam search"
    run_dir = next((tmp_path / "runs").glob("*/job.json")).parent
    assert (run_dir / "tacit.md").read_text() == "- USER: explore beam search\n"
    assert cli.main(["run", "--challenge", "knapsack", "--direction", "go", "--direction-file",
                     "dir.md", "--budget-iterations", "1", "--yes"]) == 2
    assert cli.main(["run", "--challenge", "knapsack", "--direction-file", "missing.md",
                     "--budget-iterations", "1", "--yes"]) == 2
    assert "direction file not found" in capsys.readouterr().err


def test_compile_ships_sources_and_returns_compiler_status(tmp_path, monkeypatch, capsys):
    # mutation: returning 0 on a failed compile, or shipping non-source files, fails these
    monkeypatch.chdir(tmp_path)
    (tmp_path / "algorithm").mkdir()
    (tmp_path / "algorithm" / "mod.rs").write_text("fn solve() {}\n")
    (tmp_path / "algorithm" / "notes.txt").write_text("ignore me\n")
    seen = {}
    def stub(ok):
        class B:
            def compile(self, challenge, files):
                seen.update(challenge=challenge, files=files)
                return CompileResult(ok=ok, artifact_id="a1" if ok else None,
                                     output="compiler says")
        return lambda: B()
    monkeypatch.setattr("talos.bench.ModalBench", stub(False))
    assert cli.main(["compile", "--challenge", "knapsack"]) == 1
    assert seen["challenge"] == "knapsack" and list(seen["files"]) == ["mod.rs"]
    assert "compiler says" in capsys.readouterr().out
    monkeypatch.setattr("talos.bench.ModalBench", stub(True))
    assert cli.main(["compile", "--challenge", "knapsack"]) == 0


def test_status_lists_every_run(tmp_path, monkeypatch, capsys):
    # mutation: printing only the newest run, or crashing on a run with no state.json, fails this
    monkeypatch.chdir(tmp_path)
    for name, status in (("20260101-000000-knapsack", "won"), ("20260102-000000-knapsack", "failed")):
        d = tmp_path / "runs" / name
        d.mkdir(parents=True)
        (d / "state.json").write_text(json.dumps(
            {"status": status, "iteration": 2, "spend": {"llm_usd": 1.5, "modal_usd": 0.25}}))
    (tmp_path / "runs" / "20260103-000000-knapsack").mkdir()  # started, nothing saved yet
    assert cli.main(["status"]) == 0
    out = capsys.readouterr().out
    assert "20260101-000000-knapsack: won it=2 llm=$1.50 modal=$0.25" in out
    assert "20260102-000000-knapsack: failed it=2" in out


def test_deploy_bench_failure_never_quotes_the_token_secret():
    # mutation: echoing argv into the error leaks the Modal secret
    def run(cmd, **kw):
        if "deploy" in cmd:
            return types.SimpleNamespace(returncode=1, stdout="", stderr="Error: app deploy failed")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    with pytest.raises(ConfigError) as excinfo:
        cli.deploy_bench("ak-1", "as-supersecret", run=run)
    assert "as-supersecret" not in str(excinfo.value)
    assert "app deploy failed" in str(excinfo.value)
