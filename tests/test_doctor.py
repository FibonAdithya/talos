"""`talos doctor`. Nothing here may reach a real process, Modal, C3, Docker, mainnet or GHCR:
the autouse fixture makes every such path raise, and each test restubs only what it needs."""
import json
import subprocess
import urllib.request

import pytest

from talos import cli, doctor, local_transport, mainnet
from talos.bench import EvalResult, ModalBench
from talos.challenges import CHALLENGES
from talos.config import Config, ConfigError, save
from talos.local_transport import DockerInfo
from talos.mainnet import ChallengeInfo, MainnetError
from talos.types import CompileResult, NonceResult


def _boom(*a, **k):
    raise AssertionError("no real subprocess")


@pytest.fixture(autouse=True)
def _no_real_paths(monkeypatch):
    # c3_transport binds run=subprocess.run as a default at import, so patching subprocess.run
    # alone would not stop a real `c3`: the transports are blocked too.
    monkeypatch.setattr(subprocess, "run", _boom)
    monkeypatch.setattr(cli, "make_transport", _boom)
    monkeypatch.setattr(cli, "CliTransport", _boom)
    monkeypatch.setattr(doctor, "modal_lookup", _boom)
    monkeypatch.setattr(cli, "make_bench", _boom)
    monkeypatch.setattr(cli, "deploy_bench", _boom)
    monkeypatch.setattr(local_transport, "prepare", _boom)
    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    monkeypatch.setattr(mainnet, "_get", _boom)
    for name in ("ANTHROPIC_API_KEY", "TALOS_CUSTOM_API_KEY", "C3_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def mainnet_infos():
    return [ChallengeInfo(id=s.id, name=n, is_gpu=s.is_gpu, tracks=["t=1", "t=2"], max_fuel=100)
            for n, s in CHALLENGES.items()]


def all_ok(monkeypatch, tmp_path, backend="modal", provider="anthropic", api_key="sk-test"):
    """A configured directory where every dependency succeeds."""
    monkeypatch.chdir(tmp_path)
    save(tmp_path, Config(provider=provider, model="m", mode="single-shot", api_base=None,
                          backend=backend), api_key)
    monkeypatch.setattr(cli, "make_provider", lambda *a, **k: object())
    monkeypatch.setattr(cli, "validate_provider", lambda p: None)
    monkeypatch.setattr(doctor, "modal_lookup", lambda app, name: True)
    monkeypatch.setattr(cli, "check_c3", lambda api_key=None, **k: 12.5)
    monkeypatch.setattr(cli, "check_local", lambda run=None: DockerInfo(["nvidia", "runc"], 16, 30))
    monkeypatch.setattr(mainnet, "fetch_challenges", lambda get_json=None: mainnet_infos())
    monkeypatch.setattr(cli, "image_available", lambda challenge, fetch=None: True)


def scripted(answers):
    it = iter(answers)

    def ask(prompt, default=None, secret=False):
        v = next(it)
        if v is EOFError:
            raise EOFError
        return v if v != "" else (default or "")
    return ask


def no_ask(prompt, default=None, secret=False):
    raise AssertionError(f"unexpected prompt: {prompt}")


def line_for(out: str, name: str) -> str:
    found = [ln for ln in out.splitlines() if ln.startswith(f"{name}: ")]
    assert len(found) == 1, out
    return found[0]


def test_modal_app_name_matches_bench():
    # mutation: a typo in MODAL_APP_NAME makes every lookup miss a deployed app
    assert doctor.MODAL_APP_NAME == ModalBench().app_name


def test_all_checks_pass_exit_0(tmp_path, monkeypatch, capsys):
    # mutation: a check dropped from free_checks
    all_ok(monkeypatch, tmp_path)
    rc = cli.main(["doctor"], ask=no_ask)
    out = capsys.readouterr().out
    assert rc == 0, out
    names = ["config", "credential", "provider", "backend", "mainnet", "images"]
    starts = [next(i for i, ln in enumerate(out.splitlines()) if ln.startswith(f"{n}: ok"))
              for n in names]
    assert starts == sorted(starts), out


def _remove_config(mp, tmp_path):
    (tmp_path / "talos.config.json").unlink()


def _no_key(mp, tmp_path):
    (tmp_path / ".talos" / "secrets.json").unlink(missing_ok=True)


def _no_cli(mp, tmp_path):
    mp.setattr(doctor.shutil, "which", lambda name: None)


def _provider_rejects(mp, tmp_path):
    mp.setattr(cli, "validate_provider", lambda p: "credential rejected: zz-bad-key")


def _modal_all_missing(mp, tmp_path):
    mp.setattr(doctor, "modal_lookup", lambda app, name: False)


def _modal_import_error(mp, tmp_path):
    def lookup(app, name):
        raise ImportError("No module named 'modal'")
    mp.setattr(doctor, "modal_lookup", lookup)


def _c3_fails(mp, tmp_path):
    def check(api_key=None, **k):
        raise ConfigError("C3 login check failed: zz-c3; run `c3 login` and retry")
    mp.setattr(cli, "check_c3", check)


def _local_fails(mp, tmp_path):
    def check(run=None):
        raise ConfigError("Docker check failed: zz-docker; install and start Docker")
    mp.setattr(cli, "check_local", check)


def _mainnet_down(mp, tmp_path):
    def fetch(get_json=None):
        raise MainnetError("network error fetching x")
    mp.setattr(mainnet, "fetch_challenges", fetch)


def _mainnet_drift(mp, tmp_path):
    infos = [i if i.name != "vehicle_routing"
             else ChallengeInfo("c999", i.name, i.is_gpu, i.tracks, i.max_fuel)
             for i in mainnet_infos()]
    mp.setattr(mainnet, "fetch_challenges", lambda get_json=None: infos)


def _image_missing(mp, tmp_path):
    mp.setattr(cli, "image_available", lambda challenge, fetch=None: challenge != "hypergraph")


# Worked by hand from talos/challenges.py: 5 CPU challenges x 2 functions, 3 GPU challenges x 3
# GPUs x 2 functions, and one probe per GPU: 10 + 18 + 3.
N_MODAL = 31


@pytest.mark.parametrize("backend,provider,breaks,name,needles", [
    ("modal", "anthropic", _remove_config, "config", ["talos setup"]),
    ("modal", "anthropic", _no_key, "credential", ["ANTHROPIC_API_KEY", "talos setup"]),
    ("modal", "custom", _no_key, "credential", ["TALOS_CUSTOM_API_KEY", "talos setup"]),
    ("modal", "claude-cli", _no_cli, "credential", ["install", "claude"]),
    ("modal", "anthropic", _provider_rejects, "provider",
     ["credential rejected: zz-bad-key", "talos setup"]),
    ("modal", "anthropic", _modal_all_missing, "backend",
     [str(N_MODAL), "compile_satisfiability", "talos setup"]),
    ("modal", "anthropic", _modal_import_error, "backend", ["No module named", "talos setup"]),
    ("c3", "anthropic", _c3_fails, "backend", ["zz-c3", "c3 login"]),
    ("local", "anthropic", _local_fails, "backend", ["zz-docker", "install and start Docker"]),
    ("modal", "anthropic", _mainnet_down, "mainnet", ["unreachable", "check your network"]),
    ("modal", "anthropic", _mainnet_drift, "mainnet",
     ["vehicle_routing", "update talos/challenges.py"]),
    ("modal", "anthropic", _image_missing, "images", ["hypergraph", "talos check-updates"]),
])
def test_each_failure_names_its_fix(tmp_path, monkeypatch, capsys, backend, provider, breaks,
                                    name, needles):
    # mutation: a fix hint dropped from any FAIL detail
    all_ok(monkeypatch, tmp_path, backend=backend, provider=provider,
           api_key=None if provider == "claude-cli" else "sk-test")
    breaks(monkeypatch, tmp_path)
    rc = cli.main(["doctor"], ask=no_ask)
    out = capsys.readouterr().out
    assert rc == 1, out
    ln = line_for(out, name)
    assert ln.startswith(f"{name}: FAIL: "), ln
    for needle in needles:
        assert needle in ln, (needle, ln)


def test_image_missing_names_only_the_missing_challenge(tmp_path, monkeypatch, capsys):
    # mutation: listing every challenge in the images FAIL instead of the missing ones
    all_ok(monkeypatch, tmp_path)
    _image_missing(monkeypatch, tmp_path)
    cli.main(["doctor"], ask=no_ask)
    assert "knapsack" not in line_for(capsys.readouterr().out, "images")


def test_no_config_skips_dependent_checks(tmp_path, monkeypatch, capsys):
    # mutation: an early return after the config failure (mainnet and images never run)
    monkeypatch.chdir(tmp_path)
    ran = []
    monkeypatch.setattr(mainnet, "fetch_challenges",
                        lambda get_json=None: ran.append("mainnet") or mainnet_infos())
    monkeypatch.setattr(cli, "image_available",
                        lambda challenge, fetch=None: ran.append("images") or True)
    rc = cli.main(["doctor"], ask=no_ask)
    out = capsys.readouterr().out
    assert rc == 1
    for name in ("credential", "provider", "backend"):
        assert line_for(out, name) == f"{name}: skip: no config"
    assert line_for(out, "mainnet").startswith("mainnet: ok")
    assert line_for(out, "images").startswith("images: ok")
    assert "mainnet" in ran and "images" in ran


def test_modal_function_names_match_register():
    # mutation: the probe names or a GPU suffix dropped from modal_function_names
    pytest.importorskip("modal")
    from modal_app import talos_bench

    seen = []

    class FakeApp:
        def function(self, name, **kw):
            seen.append(name)
            return lambda f: f

    talos_bench.register(FakeApp(), image=lambda n: None, probe_image=lambda: None)
    assert len(seen) == len(set(seen))
    assert set(seen) == set(doctor.modal_function_names())


def test_modal_missing_function_fails(tmp_path, monkeypatch, capsys):
    # mutation: ignoring the lookup result (or saying "up to date") passes a stale deploy
    all_ok(monkeypatch, tmp_path)
    gone = "score_batch_hypergraph_h100"
    monkeypatch.setattr(doctor, "modal_lookup", lambda app, name: name != gone)
    rc = cli.main(["doctor"], ask=no_ask)
    ln = line_for(capsys.readouterr().out, "backend")
    assert rc == 1
    assert ln.startswith("backend: FAIL: ") and gone in ln
    assert "up to date" not in ln


def test_modal_ok_detail_counts_functions(tmp_path, monkeypatch, capsys):
    # mutation: the OK detail claiming "up to date", or the count dropped
    all_ok(monkeypatch, tmp_path)
    cli.main(["doctor"], ask=no_ask)
    ln = line_for(capsys.readouterr().out, "backend")
    assert ln == f"backend: ok (deployed ({N_MODAL} functions))"


@pytest.mark.parametrize("balance,needle", [
    (0.5, "c3 topup"),
    (0.0, "could not read the balance, or it is £0.00; c3 topup"),
])
def test_c3_balance_warns_and_still_exits_0(tmp_path, monkeypatch, capsys, balance, needle):
    # mutation: a warn counted as FAIL
    all_ok(monkeypatch, tmp_path, backend="c3")
    monkeypatch.setattr(cli, "check_c3", lambda api_key=None, **k: balance)
    rc = cli.main(["doctor"], ask=no_ask)
    ln = line_for(capsys.readouterr().out, "backend")
    assert ln.startswith("backend: warn: ") and needle in ln
    assert rc == 0


def test_local_ok_reports_daemon(tmp_path, monkeypatch, capsys):
    # mutation: the NVIDIA runtime read from the wrong field always says no
    all_ok(monkeypatch, tmp_path, backend="local")
    assert cli.main(["doctor"], ask=no_ask) == 0
    assert line_for(capsys.readouterr().out, "backend") == \
        "backend: ok (16 CPUs, 30 GiB, NVIDIA runtime: yes)"


def recording_bench(calls):
    class StubBench:
        def cost_mark(self):
            return 0.0

        def cost_usd_since(self, mark):
            return 0.01

        def select_hardware(self, challenge, chosen=None):
            calls.append(("select_hardware", challenge))
            return None

        def evaluate(self, req):
            calls.append(("evaluate", req))
            rows = [NonceResult(ns.track, n, True, 5, 1) for ns in req.training
                    for n in ns.nonces()]
            return EvalResult(CompileResult(True, "a1", "built"), rows, None, "forced")

    def make(backend, run_dir, pending, c3_api_key=None, local=None):
        calls.append(("make_bench", backend))
        return StubBench()
    return make


def stub_full_mainnet(monkeypatch):
    monkeypatch.setattr(mainnet, "fetch_challenge_info", lambda name, get_json=None: ChallengeInfo(
        CHALLENGES[name].id, name, CHALLENGES[name].is_gpu, ["t=1", "t=2"], 100))
    monkeypatch.setattr(mainnet, "top_algorithm", lambda name, get_json=None: ("algo", "a1", 5))
    monkeypatch.setattr(mainnet, "fetch_algorithm_files",
                        lambda name, algorithm, **k: {"mod.rs": "// x"})


@pytest.mark.parametrize("answer", ["", "n", EOFError])
def test_full_declined_never_builds_a_bench(tmp_path, monkeypatch, capsys, answer):
    # mutation: the confirm default flipped to yes, or EOF treated as the default
    all_ok(monkeypatch, tmp_path)
    stub_full_mainnet(monkeypatch)
    calls = []
    monkeypatch.setattr(cli, "make_bench", recording_bench(calls))
    rc = cli.main(["doctor", "--full"], ask=scripted([answer]))
    out = capsys.readouterr().out
    assert rc == 1
    assert calls == []
    assert "full: skip: declined" in out


def test_full_with_yes_scores_two_training_nonces_no_holdout(tmp_path, monkeypatch, capsys):
    # mutation: holdout counts of 2, all tracks instead of the first, or select_hardware skipped
    all_ok(monkeypatch, tmp_path)
    stub_full_mainnet(monkeypatch)
    calls = []
    monkeypatch.setattr(cli, "make_bench", recording_bench(calls))
    rc = cli.main(["doctor", "--full", "--yes"], ask=no_ask)
    out = capsys.readouterr().out
    assert rc == 0, out
    kinds = [c[0] for c in calls]
    assert kinds == ["make_bench", "select_hardware", "evaluate"]
    req = calls[2][1]
    assert sum(ns.count for ns in req.training) == 2
    assert all(ns.count == 0 for ns in req.holdout)
    ln = line_for(out, "full")
    assert ln.startswith("full: ok") and "2/2 nonces ok" in ln
    assert "compute spend (estimated) $0.01" in ln


def test_full_skipped_after_a_free_check_fails(tmp_path, monkeypatch, capsys):
    # mutation: --full run even though a free check failed
    all_ok(monkeypatch, tmp_path)
    _image_missing(monkeypatch, tmp_path)
    stub_full_mainnet(monkeypatch)
    calls = []
    monkeypatch.setattr(cli, "make_bench", recording_bench(calls))
    rc = cli.main(["doctor", "--full", "--yes"], ask=no_ask)
    out = capsys.readouterr().out
    assert rc == 1
    assert calls == []
    assert "full: skip: a free check failed" in out


def test_json_full_without_yes_is_skip_and_exit_1(tmp_path, monkeypatch, capsys):
    # mutation: JSON mode asking the question (no_ask raises) or running --full unconfirmed
    all_ok(monkeypatch, tmp_path)
    stub_full_mainnet(monkeypatch)
    calls = []
    monkeypatch.setattr(cli, "make_bench", recording_bench(calls))
    rc = cli.main(["doctor", "--json", "--full"], ask=no_ask)
    doc = json.loads(capsys.readouterr().out)
    assert rc == 1 and doc["ok"] is False
    full = [c for c in doc["checks"] if c["name"] == "full"]
    assert full == [{"name": "full", "status": "skip", "detail": "needs --yes"}]
    assert calls == []


def test_json_has_every_check(tmp_path, monkeypatch, capsys):
    # mutation: a check left out of the JSON, or text lines mixed into stdout
    all_ok(monkeypatch, tmp_path)
    rc = cli.main(["doctor", "--json"], ask=no_ask)
    doc = json.loads(capsys.readouterr().out)
    assert [c["name"] for c in doc["checks"]] == [
        "config", "credential", "provider", "backend", "mainnet", "images"]
    assert rc == 0 and doc["ok"] is True


def test_full_cost_text():
    # mutation: the knapsack figure reused for vehicle_routing, or the GPU range from one GPU
    t = doctor.full_cost_text("c3", "knapsack")
    assert "MEASURED 2026-09-15" in t
    t = doctor.full_cost_text("c3", "vehicle_routing")
    assert "not measured for vehicle_routing" in t and "ESTIMATE (unverified)" in t
    assert "MEASURED" not in t
    # 600 s x (4 x 0.0000131 + 8 x 0.00000222) = 0.0421, worked by hand
    assert "$0.04" in doctor.full_cost_text("modal", "knapsack")
    # 1800 x 0.000542 = 0.9756 and 1800 x 0.001097 = 1.9746
    t = doctor.full_cost_text("modal", "hypergraph")
    assert "$0.98" in t and "$1.97" in t
    # 1.49 x 2400 / 3600 = 0.993 and 3.67 x 2400 / 3600 = 2.447
    t = doctor.full_cost_text("c3", "hypergraph")
    assert "£0.99" in t and "£2.45" in t
    assert "no compute charge" in doctor.full_cost_text("local", "hypergraph")


def test_local_json_full_yes_keeps_stdout_pure_json(tmp_path, monkeypatch, capsys):
    # mutation: prepare called with its default log=print, or the cost line printed to stdout,
    # or remove_compile_containers moved out of the finally / dropped
    all_ok(monkeypatch, tmp_path, backend="local")
    stub_full_mainnet(monkeypatch)
    calls = []

    def prepare(challenge, run=None, log=print):
        calls.append(("prepare", challenge))
        log("pulling zz-image (about 13 GB, once)")
        return None

    monkeypatch.setattr(local_transport, "prepare", prepare)
    monkeypatch.setattr(cli, "local_settings", lambda cfg, info=None: object())
    monkeypatch.setattr(cli, "make_bench", recording_bench(calls))
    monkeypatch.setattr(cli, "remove_compile_containers",
                        lambda run_dir: calls.append(("remove", run_dir)))
    rc = cli.main(["doctor", "--json", "--full", "--yes"], ask=no_ask)
    cap = capsys.readouterr()
    doc = json.loads(cap.out)  # one JSON document and nothing else
    assert rc == 0 and doc["ok"] is True
    assert [c["name"] for c in doc["checks"]][-1] == "full"
    assert "pulling zz-image" in cap.err
    assert "no compute charge" in cap.err
    assert [c[0] for c in calls] == ["prepare", "make_bench", "select_hardware", "evaluate",
                                     "remove"]


@pytest.mark.parametrize("exc", [
    json.JSONDecodeError("Expecting value", "<html>captive portal</html>", 0),
    OSError("connection reset by peer"),
])
def test_mainnet_garbage_or_dropped_connection_is_unreachable(tmp_path, monkeypatch, capsys,
                                                              exc):
    # mutation: the except in check_mainnet narrowed back to MainnetError (a traceback)
    all_ok(monkeypatch, tmp_path)

    def fetch(get_json=None):
        raise exc
    monkeypatch.setattr(mainnet, "fetch_challenges", fetch)
    rc = cli.main(["doctor"], ask=no_ask)
    ln = line_for(capsys.readouterr().out, "mainnet")
    assert rc == 1
    assert ln.startswith("mainnet: FAIL: unreachable") and "check your network" in ln
