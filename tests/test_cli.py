import json
import os
import stat
import time
import types

import pytest

from talos import cli
from talos.bench import BenchCancelled, EvalResult
from talos.challenges import DEV_IMAGE_TAG
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
    rc = cli.main(["setup"], ask=scripted(["", "anthropic", "", "sk-test", "ak-1", "as-1"]))
    assert rc == 0
    cfg = json.loads((tmp_path / "talos.config.json").read_text())
    assert cfg["provider"] == "anthropic" and cfg["model"] == "claude-opus-5"
    sec = tmp_path / ".talos" / "secrets.json"
    assert json.loads(sec.read_text()) == {"api_key": "sk-test"}
    # mutation: os.open creates a NEW file 0600 and chmod repairs a pre-existing one, so only
    # widening the open mode AND dropping the chmod fails this (either alone is covered by the other)
    assert stat.S_IMODE(sec.stat().st_mode) == 0o600
    assert calls == ["deploy"]


def test_setup_rejected_key_writes_nothing(tmp_path, monkeypatch):
    # mutation: writing before validation leaves a bad key on disk
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "validate_provider", lambda p: "credential rejected")
    monkeypatch.setattr(cli, "deploy_bench", lambda *a, **k: None)
    rc = cli.main(["setup"], ask=scripted(["", "anthropic", "", "bad", "ak", "as"]))
    assert rc != 0
    assert not (tmp_path / "talos.config.json").exists()
    assert not (tmp_path / ".talos").exists()


def test_setup_cli_provider_stores_no_secret(tmp_path, monkeypatch):
    # mutation: prompting for and saving an api_key for a CLI provider would write secrets.json
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "validate_provider", lambda p: None)
    monkeypatch.setattr(cli, "deploy_bench", lambda *a, **k: None)
    rc = cli.main(["setup"], ask=scripted(["", "claude-cli", "", "", "ak", "as"]))  # mode defaults
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
    assert spec.budget.compute_usd == 20.0
    assert len(spec.rand_hash) == 64
    job_files = list((tmp_path / "runs").glob("*/job.json"))
    assert len(job_files) == 1
    # mutation: zero-budget guard using truthiness would accept a job with no cap; applying the
    # Modal default before validation would let it stand in for the missing LLM/time cap
    rc2 = cli.main(["run", "--challenge", "knapsack", "--direction", "go", "--yes"])
    assert rc2 == 2


def test_setup_rejects_the_fake_provider(tmp_path, monkeypatch, capsys):
    # mutation: accepting "fake" writes a config whose runs never call an LLM at all — the
    # in-process test double is not a provider a user can set up
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "validate_provider", lambda p: None)
    monkeypatch.setattr(cli, "deploy_bench", lambda *a, **k: None)
    assert cli.main(["setup"], ask=scripted(["", "fake"])) == 2
    assert "unknown provider 'fake'" in capsys.readouterr().err
    assert not (tmp_path / "talos.config.json").exists()


def test_wizard_labels_gpu_challenges_asks_mode_and_survives_a_typo(tmp_path, monkeypatch,
                                                                   capsys):
    # spec §5.2: GPU challenges carry their GPU class and approximate cost; CLI providers are
    # asked for a mode, after the 5-20x token warning.
    # mutation: an unlabelled challenge list hides that a GPU challenge costs ~$2/h while a CPU
    # one costs cents; dropping the mode prompt means agentic can only be reached by flag
    # mutation: float(ask(...)) on a non-numeric answer tracebacks out of the wizard
    monkeypatch.chdir(tmp_path)
    save(tmp_path, Config(provider="claude-cli", model="claude-opus-5", mode="single-shot",
                          api_base=None), None)
    monkeypatch.setattr(cli, "fetch_challenge_info", lambda name: knapsack_info())
    seen = {}
    monkeypatch.setattr(cli, "execute_job",
                        lambda spec, store, cfg, resume: seen.update(spec=spec, cfg=cfg) or 0)
    prompts = []
    answers = iter(["knapsack", "go", "abc", "3", "4", "5", "agentic"])

    def ask(prompt, default=None, secret=False):
        prompts.append(prompt)
        return next(answers)

    assert cli.main(["run"], ask=ask) == 0
    captured = capsys.readouterr()
    assert "not a number: 'abc'" in captured.err
    assert "hypergraph (GPU: L40S, ≈$1.95/h estimated)" in prompts[0]
    assert "knapsack" in prompts[0] and "knapsack (GPU" not in prompts[0]
    assert prompts.count("Iteration budget") == 2  # the typo was re-asked, not fatal
    assert prompts[-1] == "Mode (single-shot or agentic)"
    assert "agentic mode uses roughly 5-20x the tokens of single-shot" in captured.out
    assert seen["spec"].budget.iterations == 3 and seen["spec"].budget.hours == 4.0
    assert seen["spec"].budget.compute_usd == 5.0
    assert seen["cfg"].mode == "agentic" and seen["spec"].mode == "agentic"


def test_wizard_gives_up_after_three_non_numbers(tmp_path, monkeypatch, capsys):
    # mutation: an unbounded re-ask loop never terminates on a piped/empty stdin
    monkeypatch.chdir(tmp_path)
    save(tmp_path, Config(provider="claude-cli", model="claude-opus-5", mode="single-shot",
                          api_base=None), None)
    monkeypatch.setattr(cli, "execute_job",
                        lambda spec, store, cfg, resume: pytest.fail("must not start a job"))
    assert cli.main(["run"], ask=scripted(["knapsack", "go", "abc", "def", "ghi"])) == 2
    assert "no number given after 3 tries" in capsys.readouterr().err


def test_challenge_table_drift_stops_before_the_job_exists(tmp_path, monkeypatch, capsys):
    # mutation: not cross-checking mainnet against talos/challenges.py scores every nonce under
    # the wrong challenge id (or on the wrong hardware) and reports the result as if it counted
    monkeypatch.chdir(tmp_path)
    fake_config(tmp_path)

    def drifted(**over):
        fields = {"id": "c003", "name": "knapsack", "is_gpu": False, "tracks": ["n=1"],
                  "max_fuel": 7}
        fields.update(over)
        return type("I", (), fields)()

    argv = ["run", "--challenge", "knapsack", "--direction", "go", "--budget-iterations", "1",
            "--yes"]
    monkeypatch.setattr(cli, "fetch_challenge_info", lambda name: drifted(id="c999"))
    assert cli.main(argv) == 1
    err = capsys.readouterr().err
    assert "challenge table drift" in err and "c999" in err and "talos/challenges.py" in err
    monkeypatch.setattr(cli, "fetch_challenge_info", lambda name: drifted(is_gpu=True))
    assert cli.main(argv) == 1
    assert "challenge table drift" in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()  # nothing was created either time


def test_compile_refuses_an_empty_or_missing_dir(tmp_path, monkeypatch, capsys):
    # mutation: shipping an empty file map pays Modal for a container that can only fail, and
    # reports the failure as a compiler error
    monkeypatch.chdir(tmp_path)

    def boom(*a, **k):
        raise AssertionError("the bench must not be constructed for an empty file map")

    monkeypatch.setattr(cli, "make_bench", boom)
    assert cli.main(["compile", "--challenge", "knapsack"]) == 2  # no algorithm/ at all
    assert "no .rs/.cu files under algorithm" in capsys.readouterr().err
    (tmp_path / "algorithm").mkdir()
    (tmp_path / "algorithm" / "notes.txt").write_text("not a source file\n")
    assert cli.main(["compile", "--challenge", "knapsack", "--dir", "algorithm"]) == 2


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
    # mutation: stamping the printed prefix with state.iteration labels iteration 1's events it=0
    assert "it=1 hypothesis" in out and "it=0 hypothesis" not in out
    run_dir = next((tmp_path / "runs").glob("*/job.json")).parent
    assert f"Package: {run_dir / 'package'}" in out
    assert (run_dir / "package" / "scores.md").exists()
    best = json.loads((run_dir / "state.json").read_text())["best"]
    assert "let k = 2;" in best["files"]["mod.rs"]


def test_fake_flag_needs_no_config_and_never_touches_mainnet(tmp_path, monkeypatch, capsys):
    # mutation: reading the config or calling mainnet under --fake makes the flag useless on a
    # fresh clone; a real clone has neither talos.config.json nor Modal/LLM credentials
    monkeypatch.chdir(tmp_path)
    assert not (tmp_path / "talos.config.json").exists()

    def boom(name):
        raise AssertionError("fetch_challenge_info must not be called under --fake")
    monkeypatch.setattr(cli, "fetch_challenge_info", boom)
    rc = cli.main(["run", "--challenge", "knapsack", "--direction", "test",
                   "--budget-iterations", "5", "--yes", "--fake"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Status: won" in out
    assert not (tmp_path / "talos.config.json").exists()


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


def test_resume_after_the_hours_window_shifts_the_clock(tmp_path, monkeypatch, capsys):
    # mutation: keeping the original started_at makes every late resume exhausted — a job with
    # --budget-hours 4 that is resumed five hours later exits "exhausted (hours)" at once and can
    # never be resumed again
    monkeypatch.chdir(tmp_path)
    fake_config(tmp_path)
    monkeypatch.setattr(cli, "fetch_challenge_info", lambda name: knapsack_info())
    assert cli.main(["run", "--challenge", "knapsack", "--direction", "go",
                     "--budget-iterations", "5", "--budget-hours", "4", "--yes"]) == 0
    run_dir = next((tmp_path / "runs").glob("*/job.json")).parent
    st = json.loads((run_dir / "state.json").read_text())
    st.update(status="cancelled", stop_reason="user requested stop", best=None, iteration=0,
              confirmed=[], hypotheses=[])
    st["spend"]["iterations"] = 0
    cancelled_at = time.time() - 5 * 3600  # cancelled five hours ago, an hour past the window
    st["spend"]["started_at"] = cancelled_at - 60
    (run_dir / "state.json").write_text(json.dumps(st))
    os.utime(run_dir / "state.json", (cancelled_at, cancelled_at))
    capsys.readouterr()
    rc = cli.main(["run", "--resume", run_dir.name])
    out = capsys.readouterr().out
    assert rc == 0 and "Status: won" in out
    resumed = [json.loads(ln) for ln in (run_dir / "timeline.jsonl").read_text().splitlines()
               if json.loads(ln)["kind"] == "resumed"]
    # the pause is approximated as the time since the last save, and recorded, not silent
    assert resumed and resumed[-1]["paused_s"] >= 4 * 3600


def test_resume_uses_the_spec_not_the_current_config(tmp_path, monkeypatch, capsys):
    # mutation: resuming with today's talos.config.json switches provider/model/mode mid-job, so
    # the iterations before and after the resume were produced by different agents
    monkeypatch.chdir(tmp_path)
    fake_config(tmp_path)
    monkeypatch.setattr(cli, "fetch_challenge_info", lambda name: knapsack_info())
    assert fake_run(monkeypatch) == 0
    run_dir = next((tmp_path / "runs").glob("*/job.json")).parent
    spec = json.loads((run_dir / "job.json").read_text())
    spec.update(provider="anthropic", model="spec-model", mode="single-shot")
    (run_dir / "job.json").write_text(json.dumps(spec))
    st = json.loads((run_dir / "state.json").read_text())
    st.update(status="cancelled", best=None, iteration=0, confirmed=[], hypotheses=[])
    st["spend"]["iterations"] = 0
    (run_dir / "state.json").write_text(json.dumps(st))
    # the config now says a different model
    save(tmp_path, Config(provider="anthropic", model="config-model", mode="single-shot",
                          api_base=None), None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-x")
    seen = {}

    def capture(kind, model, api_key=None, api_base=None):
        seen.update(kind=kind, model=model)
        raise RuntimeError("stop before the loop starts")

    monkeypatch.setattr(cli, "make_provider", capture)
    with pytest.raises(RuntimeError):
        cli.main(["run", "--resume", run_dir.name])
    assert seen == {"kind": "anthropic", "model": "spec-model"}
    # mutation: allowing --mode on a resume silently changes how the rest of the job is produced
    assert cli.main(["run", "--resume", run_dir.name, "--mode", "agentic"]) == 2
    assert "start a new job to change mode" in capsys.readouterr().err


def test_resume_without_state_json_is_refused(tmp_path, monkeypatch, capsys):
    # mutation: guarding on job.json alone tracebacks on a run whose credential was refused
    # before the first save (job.json written, state.json never)
    monkeypatch.chdir(tmp_path)
    fake_config(tmp_path)
    monkeypatch.setattr(cli, "fetch_challenge_info", lambda name: knapsack_info())
    monkeypatch.setattr(cli, "execute_job", lambda spec, store, cfg, resume: 2)
    assert cli.main(["run", "--challenge", "knapsack", "--direction", "go",
                     "--budget-iterations", "1", "--yes"]) == 2
    run_dir = next((tmp_path / "runs").glob("*/job.json")).parent
    assert not (run_dir / "state.json").exists()
    assert cli.main(["run", "--resume", run_dir.name]) == 2
    assert "missing state.json" in capsys.readouterr().err


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
            def evaluate(self, request):
                seen.update(challenge=request.challenge, files=request.files,
                            training=request.training, holdout=request.holdout)
                return EvalResult(CompileResult(ok=ok, artifact_id="a1" if ok else None,
                                                output="compiler says"), [], None,
                                  "not_compiled" if not ok else "forced")

        def make(backend, run_dir, pending):
            seen["backend"] = backend
            return B()
        return make
    monkeypatch.setattr(cli, "make_bench", stub(False))
    monkeypatch.delenv("TALOS_BACKEND", raising=False)
    assert cli.main(["compile", "--challenge", "knapsack"]) == 1  # no config here: modal
    assert seen["backend"] == "modal"
    assert seen["challenge"] == "knapsack" and list(seen["files"]) == ["mod.rs"]
    # mutation: a compile that ships nonce sets pays for scoring the agent did not ask for
    assert seen["training"] == [] and seen["holdout"] == []
    assert "compiler says" in capsys.readouterr().out
    monkeypatch.setenv("TALOS_BACKEND", "c3")
    monkeypatch.setattr(cli, "make_bench", stub(True))
    assert cli.main(["compile", "--challenge", "knapsack"]) == 0
    assert seen["backend"] == "c3"


def test_status_lists_every_run(tmp_path, monkeypatch, capsys):
    # mutation: printing only the newest run, or crashing on a run with no state.json, fails this
    monkeypatch.chdir(tmp_path)
    for name, status in (("20260101-000000-knapsack", "won"), ("20260102-000000-knapsack", "failed")):
        d = tmp_path / "runs" / name
        d.mkdir(parents=True)
        (d / "state.json").write_text(json.dumps(
            {"status": status, "iteration": 2, "spend": {"llm_usd": 1.5, "compute_usd": 0.25}}))
    (tmp_path / "runs" / "20260103-000000-knapsack").mkdir()  # started, nothing saved yet
    assert cli.main(["status"]) == 0
    out = capsys.readouterr().out
    assert "20260101-000000-knapsack: won it=2 llm=$1.50 compute=$0.25" in out
    assert "20260102-000000-knapsack: failed it=2" in out


def test_unpriced_model_refuses_a_dollar_only_budget(tmp_path, monkeypatch, capsys):
    # mutation: a None cost silently disables the dollar cap. estimate_cost has no entry for
    # gpt-5, so every Completion carries cost_usd=None, spend.llm_usd stays 0.00 and a
    # --budget-usd-only job runs forever.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    save(tmp_path, Config(provider="openai", model="gpt-5", mode="single-shot", api_base=None),
         None)
    monkeypatch.setattr(cli, "fetch_challenge_info", lambda name: knapsack_info())
    seen = []
    monkeypatch.setattr(cli, "execute_job",
                        lambda spec, store, cfg, resume: seen.append(spec) or 0)
    base = ["run", "--challenge", "knapsack", "--direction", "go", "--yes"]
    assert cli.main(base + ["--budget-usd", "5"]) == 2
    err = capsys.readouterr().err
    assert "gpt-5 has no price-table entry" in err
    assert "--budget-hours or --budget-iterations" in err
    assert not seen and not (tmp_path / "runs").exists()  # refused before anything was created
    # another real cap present: the job runs, with the warning
    assert cli.main(base + ["--budget-usd", "5", "--budget-iterations", "3"]) == 0
    assert seen and "no price-table entry" in capsys.readouterr().err
    # mutation: a priced model must NOT be warned about or refused
    save(tmp_path, Config(provider="anthropic", model="claude-opus-5", mode="single-shot",
                          api_base=None), None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-a")
    assert cli.main(base + ["--budget-usd", "5"]) == 0
    assert "price-table" not in capsys.readouterr().err


def test_status_line_says_unpriced_instead_of_zero_dollars(tmp_path):
    # mutation: printing $0.00 for an unpriced model reports "we measured nothing spent" when
    # the truth is "we cannot measure what is being spent"
    from talos.budget import Budget, Spend
    from talos.state import JobState

    def line(provider, model):
        spec = cli.JobSpec(job_id="j", challenge="knapsack", direction="d", provider=provider,
                           model=model, mode="single-shot",
                           budget=Budget(usd=5.0, hours=None, iterations=None, compute_usd=1.0),
                           rand_hash="ab" * 32, tracks=["t"], training=[], holdout=[], fuel=1,
                           created_at=0.0, monorepo_ref="r", challenge_id="c003")
        return cli._status_line(spec, JobState.fresh(Spend(started_at=0.0)), 0.0)

    assert "llm=unpriced" in line("openai", "gpt-5")
    assert "llm=$0.00" in line("anthropic", "claude-opus-5")
    assert "llm=$0.00" in line("claude-cli", "claude-opus-5")  # a CLI provider bills no tokens


def test_run_refuses_agentic_codex_without_the_opt_in(tmp_path, monkeypatch, capsys):
    # mutation: leaving the refusal to attach_agentic tells the user only after the baseline has
    # been measured, i.e. after Modal has already been paid for
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TALOS_ALLOW_CODEX_AGENTIC", raising=False)
    save(tmp_path, Config(provider="codex-cli", model="gpt-5-codex", mode="agentic",
                          api_base=None), None)
    monkeypatch.setattr(cli, "fetch_challenge_info", lambda name: knapsack_info())
    monkeypatch.setattr(cli, "execute_job", lambda spec, store, cfg, resume: 0)
    argv = ["run", "--challenge", "knapsack", "--direction", "go", "--budget-iterations", "1",
            "--yes"]
    assert cli.main(argv) == 2
    assert "TALOS_ALLOW_CODEX_AGENTIC=1" in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()  # refused before the job dir was created
    monkeypatch.setenv("TALOS_ALLOW_CODEX_AGENTIC", "1")
    assert cli.main(argv) == 0  # the opt-in lets it through


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


def _c3_runner(whoami_rc=0, balance="  Credit balance: £9.89\n", balance_rc=0):
    def run(cmd, **kw):
        if cmd[:2] == ["c3", "whoami"]:
            return types.SimpleNamespace(
                returncode=whoami_rc, stdout="Access approved" if whoami_rc == 0 else "",
                stderr="" if whoami_rc == 0 else "Your C3 session has expired")
        if cmd[:2] == ["c3", "balance"]:
            return types.SimpleNamespace(returncode=balance_rc,
                                         stdout="" if balance_rc else balance,
                                         stderr="c3: request failed" if balance_rc else "")
        raise AssertionError(cmd)
    return run


def test_setup_c3_skips_modal_and_writes_backend(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "validate_provider", lambda p: None)
    monkeypatch.setattr(cli, "deploy_bench",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("deployed")))
    monkeypatch.setattr(cli, "check_c3", lambda run=None: 9.89)
    # prompts: backend, provider, model, api key — no Modal token prompts
    rc = cli.main(["setup"], ask=scripted(["c3", "anthropic", "", "sk-test"]))
    assert rc == 0
    cfg = json.loads((tmp_path / "talos.config.json").read_text())
    # mutation: not persisting the backend makes every run go to Modal after a C3 setup
    assert cfg["backend"] == "c3"
    assert load(tmp_path).backend == "c3"


def test_setup_modal_is_the_default_and_unchanged(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = []
    monkeypatch.setattr(cli, "validate_provider", lambda p: None)
    monkeypatch.setattr(cli, "deploy_bench",
                        lambda token_id, token_secret, run=None: calls.append("deploy"))
    rc = cli.main(["setup"], ask=scripted(["", "anthropic", "", "sk-test", "ak-1", "as-1"]))
    assert rc == 0 and calls == ["deploy"]
    assert json.loads((tmp_path / "talos.config.json").read_text())["backend"] == "modal"


def test_setup_c3_fails_when_the_session_has_expired(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "validate_provider", lambda p: None)
    monkeypatch.setattr(cli, "subprocess", types.SimpleNamespace(run=_c3_runner(whoami_rc=1)))
    rc = cli.main(["setup"], ask=scripted(["c3", "anthropic", "", "sk-test"]))
    # mutation: ignoring whoami's exit code writes a config whose first run fails 20 min later
    assert rc == 1 and "c3 login" in capsys.readouterr().err
    assert not (tmp_path / "talos.config.json").exists()


def test_check_c3_parses_the_balance_and_warns_when_low(capsys):
    assert cli.check_c3(run=_c3_runner()) == 9.89
    assert cli.check_c3(run=_c3_runner(balance="Credit balance: £0.40\n")) == 0.40
    # mutation: dropping the warning lets a run start on 40p and pause at the first job
    assert "low" in capsys.readouterr().err.lower()
    assert cli.check_c3(run=_c3_runner(balance_rc=1)) == 0.0
    err = capsys.readouterr().err
    # mutation: reporting an unreadable balance as "£0.00 is low" invents a number `c3` never gave
    assert "could not read" in err and "low" not in err.lower()
    with pytest.raises(ConfigError):
        cli.check_c3(run=_c3_runner(whoami_rc=1))


def test_config_without_backend_loads_as_modal(tmp_path):
    (tmp_path / "talos.config.json").write_text(json.dumps(
        {"provider": "anthropic", "model": "m", "mode": "single-shot", "api_base": None}))
    # mutation: a KeyError here breaks every config written before this change
    assert load(tmp_path).backend == "modal"


def test_image_available_hits_the_hub_tag_endpoint(monkeypatch):
    monkeypatch.delenv("TALOS_IMAGE_NAMESPACE", raising=False)
    seen = {}

    def fetch(url):
        seen["url"] = url
        # the tag-list endpoint (no tag) answers 200 for any repo that exists at all
        return 200 if url.endswith(f"tig-knapsack-dev/tags/{DEV_IMAGE_TAG}") else 404
    assert cli.image_available("knapsack", fetch=fetch) is True
    assert "hub.docker.com/v2/repositories/fibonadithya/tig-knapsack-dev/tags/" in seen["url"]
    # mutation: dropping the tag from the URL degrades the check to "the repo exists", so a
    # DEV_IMAGE_TAG bump with no re-mirror pays for a job that dies at the pull
    assert seen["url"].endswith(f"/tags/{DEV_IMAGE_TAG}")
    assert cli.image_available("hypergraph", fetch=fetch) is False


def test_run_c3_refuses_a_missing_image_before_the_baseline(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    save(tmp_path, Config(provider="claude-cli", model="m", mode="single-shot", api_base=None,
                          backend="c3"), None)
    monkeypatch.setattr(cli, "image_available", lambda ch, fetch=None: False)
    # the bench is stubbed as well as the check: the check is what is under test, so the test
    # must not depend on it to stay off a real, billable C3 job
    monkeypatch.setattr(cli, "make_bench", lambda *a, **k: _refuse_bench("baseline reached"))
    fake_info = type("I", (), {"id": "c003", "name": "knapsack", "is_gpu": False,
                               "tracks": ["n=1"], "max_fuel": 7})()
    monkeypatch.setattr(cli, "fetch_challenge_info", lambda name: fake_info)
    rc = cli.main(["run", "--challenge", "knapsack", "--direction", "go", "--budget-iterations",
                   "1", "--yes"])
    err = capsys.readouterr().err
    # mutation: checking the image after the baseline pays for a job that fails at the pull
    assert rc == 1 and "mirror_images" in err and "tig-knapsack-dev" in err
    st = json.loads(next((tmp_path / "runs").glob("*/state.json")).read_text())
    # mutation: returning without recording the outcome leaves `talos status` listing the run
    # as live for ever, with no reason
    assert st["status"] == "failed" and st["stop_reason"] == "dev image not mirrored"


def test_compile_backend_resolution_order(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = types.SimpleNamespace(backend=None)
    # no flag, no env, no config: modal (the agentic worktree has no config file)
    monkeypatch.delenv("TALOS_BACKEND", raising=False)
    assert cli.compile_backend(args, tmp_path) == "modal"
    save(tmp_path, Config(provider="fake", model="fake", mode="single-shot", api_base=None,
                          backend="c3"), None)
    assert cli.compile_backend(args, tmp_path) == "c3"
    # mutation: reading the config before the env makes an agentic C3 run compile on Modal
    monkeypatch.setenv("TALOS_BACKEND", "modal")
    assert cli.compile_backend(args, tmp_path) == "modal"
    assert cli.compile_backend(types.SimpleNamespace(backend="c3"), tmp_path) == "c3"


def test_execute_job_exports_the_backend_for_the_agentic_child(tmp_path, monkeypatch):
    # mutation: not exporting TALOS_BACKEND makes every `talos compile` from the agentic
    # sandbox default to Modal on a C3 run
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TALOS_BACKEND", raising=False)
    save(tmp_path, Config(provider="fake", model="fake", mode="single-shot", api_base=None,
                          backend="c3"), None)
    rc = cli.main(["run", "--challenge", "knapsack", "--direction", "go", "--budget-iterations",
                   "1", "--yes", "--fake"])
    assert rc == 0 and os.environ.get("TALOS_BACKEND") == "modal"  # --fake forces modal


def test_make_bench_picks_the_backend_and_hardware_class(tmp_path):
    from talos.bench import ModalBench, PendingJobStore
    from talos.c3_bench import C3Bench
    assert isinstance(cli.make_bench("modal", tmp_path, PendingJobStore.memory()), ModalBench)
    assert isinstance(cli.make_bench("c3", tmp_path, PendingJobStore.memory()), C3Bench)
    with pytest.raises(ConfigError):
        cli.make_bench("aws", tmp_path, PendingJobStore.memory())
    # mutation: using the Modal hardware class for C3 lets a Modal baseline serve a C3 run
    assert cli.bench_hardware_class("c3", "knapsack") == "c3-cpu-d3-4vcpu-16gb"
    assert cli.bench_hardware_class("modal", "knapsack") == "cpu4-mem8192"


def test_sigint_stops_the_bench_too(tmp_path, monkeypatch):
    # mutation: only calling loop.request_stop leaves a 20-minute C3 poll running (and billing)
    # until it returns on its own
    import signal
    monkeypatch.chdir(tmp_path)
    save(tmp_path, Config(provider="fake", model="fake", mode="single-shot", api_base=None), None)
    stops = []
    from talos.bench import FakeBench
    real_init = FakeBench.__init__

    def init(self, *a, **k):
        real_init(self, *a, **k)
        self.request_stop = lambda: stops.append("bench")
    monkeypatch.setattr(FakeBench, "__init__", init)
    handlers = {}
    # setdefault: execute_job's `finally` re-registers the previous handler, which must not
    # overwrite the one under test
    monkeypatch.setattr(cli.signal, "signal", lambda sig, h: handlers.setdefault(sig, h))
    assert cli.main(["run", "--challenge", "knapsack", "--direction", "go",
                     "--budget-iterations", "1", "--yes", "--fake"]) == 0
    handlers[signal.SIGINT](None, None)
    assert stops == ["bench"]


def test_a_cancelled_baseline_is_not_a_failed_run(tmp_path, monkeypatch, capsys):
    # mutation: letting BenchCancelled out of measure_baseline fall into execute_job's blanket
    # `except Exception` records a Ctrl-C during the C3 baseline job as a failed run
    monkeypatch.chdir(tmp_path)
    from talos.bench import BenchCancelled, FakeBench

    def cancelled(self, request):
        raise BenchCancelled("job_x")
    monkeypatch.setattr(FakeBench, "evaluate", cancelled)
    rc = cli.main(["run", "--challenge", "knapsack", "--direction", "go", "--budget-iterations",
                   "1", "--yes", "--fake"])
    assert rc == 1
    run_dir = next((tmp_path / "runs").glob("*/state.json")).parent
    st = json.loads((run_dir / "state.json").read_text())
    assert st["status"] == "cancelled" and "job_x" in st["stop_reason"]
    assert "Status: cancelled" in capsys.readouterr().out


class _RefusingBench:
    """A bench that fails the test instead of reaching a backend: no test may submit a job."""

    def __init__(self, why):
        self._why = why

    def evaluate(self, request):
        raise AssertionError(self._why)

    def request_stop(self):
        pass

    def cost_mark(self):
        return 0.0

    def cost_usd_since(self, mark):
        return 0.0


def _refuse_bench(why):
    return _RefusingBench(why)


def _stub_mainnet(monkeypatch):
    """Everything resolve_baseline reads from mainnet, so a test never hits the network."""
    monkeypatch.setattr("talos.mainnet.top_algorithm", lambda ch: ("fake_base", 1))
    monkeypatch.setattr("talos.mainnet.fetch_template", lambda ch: "pub fn solve_challenge(")
    monkeypatch.setattr("talos.mainnet.fetch_algorithm_files",
                        lambda ch, name: {"mod.rs": "fn solve() {}\n"})


def test_execute_job_exports_the_c3_backend_it_actually_runs_on(tmp_path, monkeypatch):
    # mutation: hardcoding "modal" in the export (or reading anything but cfg.backend) sends
    # every `talos compile` from the agentic sandbox to Modal on a C3 run
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TALOS_BACKEND", raising=False)
    save(tmp_path, Config(provider="claude-cli", model="m", mode="single-shot", api_base=None,
                          backend="c3"), None)
    monkeypatch.setattr(cli, "image_available", lambda ch, fetch=None: True)
    monkeypatch.setattr(cli, "make_provider", lambda *a, **k: object())
    monkeypatch.setattr(cli, "fetch_challenge_info", lambda name: knapsack_info())
    _stub_mainnet(monkeypatch)
    seen = {}

    class B(_RefusingBench):
        def evaluate(self, request):
            seen["backend"] = os.environ.get("TALOS_BACKEND")
            raise BenchCancelled("stop")  # nothing runs past the baseline

    monkeypatch.setattr(cli, "make_bench", lambda *a, **k: B("unreachable"))
    rc = cli.main(["run", "--challenge", "knapsack", "--direction", "go",
                   "--budget-iterations", "1", "--yes"])
    assert rc == 1 and seen["backend"] == "c3"


def test_setup_rejects_an_unknown_backend(tmp_path, monkeypatch, capsys):
    # mutation: accepting any answer writes a config whose first run cannot pick a bench at all
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "deploy_bench", lambda *a, **k: None)
    assert cli.main(["setup"], ask=scripted(["aws"])) == 2
    assert "unknown backend 'aws'" in capsys.readouterr().err
    assert not (tmp_path / "talos.config.json").exists()


def test_run_refuses_a_config_naming_an_unknown_backend(tmp_path, monkeypatch, capsys):
    # mutation: leaving a hand-edited backend to make_bench raises ConfigError from inside
    # execute_job, past every early return, so `talos run` tracebacks instead of reporting
    monkeypatch.chdir(tmp_path)
    save(tmp_path, Config(provider="fake", model="fake", mode="single-shot", api_base=None,
                          backend="aws"), None)
    monkeypatch.setattr(cli, "execute_job", lambda *a, **k: _refuse_bench("started").evaluate(None))
    rc = cli.main(["run", "--challenge", "knapsack", "--direction", "go",
                   "--budget-iterations", "1", "--yes"])
    assert rc == 2 and "unknown backend 'aws'" in capsys.readouterr().err


def test_compile_rejects_an_unknown_challenge(tmp_path, monkeypatch, capsys):
    # mutation: indexing CHALLENGES with an unvalidated name tracebacks in the agent's sandbox
    # instead of printing a message and exiting 2
    monkeypatch.chdir(tmp_path)
    (tmp_path / "algorithm").mkdir()
    (tmp_path / "algorithm" / "mod.rs").write_text("fn solve() {}\n")
    monkeypatch.setattr(cli, "make_bench",
                        lambda *a, **k: _refuse_bench("bench built for an unknown challenge"))
    assert cli.main(["compile", "--challenge", "knapsak"]) == 2
    assert "unknown challenge 'knapsak'" in capsys.readouterr().err


def test_compile_refuses_an_unknown_backend(tmp_path, monkeypatch, capsys):
    # mutation: a stale TALOS_BACKEND in the sandbox env raises ConfigError out of cmd_compile
    # and the agent sees a traceback instead of exit code 2
    monkeypatch.chdir(tmp_path)
    (tmp_path / "algorithm").mkdir()
    (tmp_path / "algorithm" / "mod.rs").write_text("fn solve() {}\n")
    monkeypatch.setenv("TALOS_BACKEND", "aws")
    monkeypatch.setattr(cli, "make_bench",
                        lambda *a, **k: _refuse_bench("bench built for an unknown backend"))
    assert cli.main(["compile", "--challenge", "knapsack"]) == 2
    assert "unknown backend 'aws'" in capsys.readouterr().err
