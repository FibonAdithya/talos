"""`talos setup | run | compile | status`. Wizard prompts go through `ask` so tests can script
them; each loop event is printed to stdout as one line, and every finished iteration ends with a
summary line: outcome, best delta, LLM and compute spend and the wall-clock time left."""
from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import types
import urllib.error
import urllib.request
from dataclasses import replace
from pathlib import Path

from talos import mainnet as mainnet_api
from talos.budget import Budget, Spend
from talos.c3_bench import C3CommandError
from talos.c3_jobdir import LocalSettings
from talos.c3_transport import CliTransport, make_transport
from talos.challenges import (CHALLENGES, MONOREPO_REF, c3_hardware_class, c3_image,
                              hardware_class, local_hardware_class)
from talos.config import (Config, ConfigError, ENV_KEYS, load, resolve_api_key,
                          resolve_c3_api_key, save)
from talos.diagnostics import first_error
from talos.local_transport import DockerTransport, docker_runtimes, has_gpu_runtime, host_uid, prepare
from talos.mainnet import ChallengeInfo, MainnetError, TrackHyperparameters, fetch_challenge_info
from talos.masked_input import ask_secret
from talos.nonces import draw_nonce_sets, new_rand_hash
from talos.providers import DEFAULT_MODELS, KINDS, make_provider, validate_provider
from talos.providers.codex_cli import list_codex_models
from talos.providers.pricing import estimate_cost
from talos.state import JobSpec, JobState, JobStore
from talos.types import Usage

BASELINE_CACHE = Path.home() / ".talos" / "baselines"
MODAL_APP_FILE = Path(__file__).resolve().parent.parent / "modal_app" / "talos_bench.py"
CLI_PROVIDERS = ("claude-cli", "codex-cli")
UNMETERED = CLI_PROVIDERS + ("fake",)
DEFAULT_COMPUTE_USD = 20.0
BACKENDS = ("modal", "c3", "local")
LOCAL_DEFAULT_MEMORY_GIB = 8
LOCAL_MIN_MEMORY_GIB = 4
IMAGE_HINT = ("the C3 backend pulls the official TIG dev image from GHCR; check that "
              "`DEV_IMAGE_TAG` in talos/challenges.py names a published tag")


def default_ask(prompt: str, default: str | None = None, secret: bool = False) -> str:
    suffix = f" [{default}]" if default else ""
    v = ask_secret(f"{prompt}{suffix}: ") if secret else input(f"{prompt}{suffix}: ")
    return v.strip() or (default or "")


def deploy_bench(token_id: str | None, token_secret: str | None, run=subprocess.run) -> None:
    modal = shutil.which("modal") or [sys.executable, "-m", "modal"]
    base = modal if isinstance(modal, list) else [modal]
    if token_id and token_secret:
        r = run(base + ["token", "set", "--token-id", token_id, "--token-secret", token_secret],
                capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            raise ConfigError(f"modal token set failed: {r.stderr[-500:]}")
    r = run(base + ["deploy", str(MODAL_APP_FILE)], capture_output=True, text=True,
            encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise ConfigError(f"modal deploy failed: {(r.stderr or r.stdout)[-2000:]}")


def check_c3(run=None, api_key: str | None = None, transport=None) -> float:
    """Confirms C3 is reachable and authenticated — over MCP when a key is configured, over the
    `c3` CLI otherwise — and returns the credit balance in GBP. 0.0 means "could not read it"."""
    from talos.c3_mcp import McpAuthError

    def key_rejected(e: Exception) -> str:
        return (f"C3 rejected the API key: {e}; check `c3 apikey list` and "
                f"run `talos setup` again")

    if transport is not None:
        t = transport
    elif run is not None:
        t = CliTransport(run=run, api_key=api_key)  # an injected runner means the CLI, keyed or not
    else:
        t = make_transport(api_key, run=subprocess.run)  # resolved now: tests patch cli.subprocess
    try:
        t.whoami()
    except McpAuthError as e:
        raise ConfigError(key_rejected(e)) from None
    except C3CommandError as e:
        keyed = api_key or os.environ.get("C3_API_KEY")
        fix = ("check the C3 API key (`c3 apikey list`)" if keyed
               else "install the `c3` CLI and run `c3 login`, or give setup a C3 API key "
                    "(`c3 apikey create`),")
        raise ConfigError(f"C3 login check failed: {e}; {fix} and retry") from None
    try:
        balance = t.balance_gbp()
    except McpAuthError as e:
        # McpAuthError is a C3CommandError; without this clause it falls into the branch
        # below and only warns, letting a revoked key through as if it had credit
        raise ConfigError(key_rejected(e)) from None
    except C3CommandError as e:
        print(f"warning: could not read the C3 balance: {e}", file=sys.stderr)
        return 0.0
    if balance is None:
        print("warning: could not read the C3 balance", file=sys.stderr)
        return 0.0
    if balance < 1.0:
        print(f"warning: C3 credit balance is low (£{balance:.2f}); run `c3 topup`",
              file=sys.stderr)
    return balance


def _total_memory_gib() -> int | None:
    """Physical memory in GiB, or None where os.sysconf cannot say (Windows)."""
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30)
    except (AttributeError, ValueError, OSError):
        return None


def default_local_memory_gib() -> int:
    total = _total_memory_gib()
    if total is None:
        return LOCAL_DEFAULT_MEMORY_GIB
    return max(LOCAL_MIN_MEMORY_GIB, total - 4)


def local_settings(cfg: Config | None) -> LocalSettings:
    """The container limits: from the config, or the machine's own for a `talos compile` in
    the agentic sandbox, which has no config and scores no nonce (so no hardware class)."""
    cpus = (cfg.local_cpus if cfg and cfg.local_cpus else None) or os.cpu_count() or 1
    mem = ((cfg.local_memory_gib if cfg and cfg.local_memory_gib else None)
           or default_local_memory_gib())
    return LocalSettings(cpus=cpus, memory_gib=mem)


def check_local(run=None) -> bool:
    """Docker must answer; returns whether the nvidia runtime is present."""
    try:
        runtimes = docker_runtimes(run or subprocess.run)
    except C3CommandError as e:
        raise ConfigError(f"Docker check failed: {e}; install and start Docker "
                          f"(docker.com), then run `talos setup` again") from None
    return "nvidia" in runtimes


def image_available(challenge: str, fetch=None) -> bool:
    """True when the manifest for the challenge's dev image tag is on GHCR, which is what C3
    pulls. Anonymous access to GHCR needs a pull-scoped token first, so this is two requests.
    `fetch(url, headers)` returns `(status, body)`; status 0 means unreachable."""
    name, tag = c3_image(challenge).removeprefix("ghcr.io/").rsplit(":", 1)
    if fetch is None:
        def fetch(u, headers):
            try:
                req = urllib.request.Request(u, headers=headers)
                with urllib.request.urlopen(req, timeout=20) as resp:
                    return resp.status, resp.read().decode("utf-8", "replace")
            except urllib.error.HTTPError as e:
                return e.code, ""
            except (OSError, http.client.HTTPException):
                # URLError (an OSError) wraps connect failures only; a body read timeout, a
                # reset connection or a truncated response arrive bare. All of them are
                # unreachable: reported as "not available", the message says so
                return 0, ""
    status, body = fetch(f"https://ghcr.io/token?scope=repository:{name}:pull", {})
    if status != 200:
        return False
    try:
        token = json.loads(body)["token"]
    except (ValueError, KeyError, TypeError):
        return False
    accept = ", ".join(("application/vnd.oci.image.index.v1+json",
                        "application/vnd.oci.image.manifest.v1+json",
                        "application/vnd.docker.distribution.manifest.list.v2+json",
                        "application/vnd.docker.distribution.manifest.v2+json"))
    status, _ = fetch(f"https://ghcr.io/v2/{name}/manifests/{tag}",
                      {"Authorization": f"Bearer {token}", "Accept": accept})
    return status == 200


def make_bench(backend: str, run_dir: Path, pending, c3_api_key: str | None = None,
               local: LocalSettings | None = None):
    if backend == "modal":
        from talos.bench import ModalBench
        return ModalBench()
    if backend == "c3":
        from talos.c3_bench import C3Bench
        return C3Bench(run_dir, pending=pending, api_key=c3_api_key)
    if backend == "local":
        from talos.c3_bench import C3Bench
        return C3Bench(run_dir, pending=pending, transport=DockerTransport(uid=host_uid()),
                       local=local or local_settings(None), usd_per_hour=0.0)
    raise ConfigError(f"unknown backend {backend!r}; run `talos setup`")


def bench_hardware_class(backend: str, challenge: str, local: LocalSettings | None = None,
                         gpu_name: str | None = None, host: str | None = None) -> str:
    spec = CHALLENGES[challenge]
    if backend == "local":
        return local_hardware_class(spec, local.cpus, local.memory_gib, gpu_name,
                                    host or socket.gethostname())
    return c3_hardware_class(spec) if backend == "c3" else hardware_class(spec)


def _model_prompt(kind: str) -> tuple[str, str | None]:
    """Prompt text and default for the model question. CLI providers know their own models:
    codex publishes a catalog, claude accepts short aliases. The static default stands in when
    the catalog cannot be read."""
    default = DEFAULT_MODELS.get(kind) or None
    if kind == "codex-cli":
        models = list_codex_models()
        if models:
            print("Models your codex CLI accepts: " + ", ".join(models))
            return "Model", models[0]
    if kind == "claude-cli":
        return "Model (an alias such as fable, opus or sonnet, or a full model id)", default
    return "Model", default


def _track_arg(answer: str | None) -> str | None:
    """The focus track a flag or wizard answer names; None (every track) for no answer or
    "all", which is how the wizard prompt and the README spell the default."""
    if answer is None or answer.strip() in ("", "all"):
        return None
    return answer.strip()


def _mainnet_hyperparameters(api, challenge: str, info) -> tuple[dict | None, dict | None,
                                                                 dict | None]:
    """(baseline_algorithm, hyperparameters, hyperparameters_source) for a new job. The algorithm
    is pinned whenever mainnet has one, so the baseline measured later is the code the map
    belongs to. The map is None when no track has a benchmark of it at this fuel."""
    top = api.top_algorithm(challenge)
    if top is None:
        return None, None, None  # resolve_baseline reports the missing algorithm
    name, algorithm_id, adoption = top
    algorithm = {"name": name, "id": algorithm_id, "adoption": adoption}
    per_track = api.top_hyperparameters(algorithm_id, info.tracks, info.max_fuel)
    found = {t: th for t, th in per_track.items() if th.benchmark_id is not None}
    if not found:
        return algorithm, None, None
    return (algorithm, {t: per_track[t].hyperparameters for t in info.tracks},
            {t: th.source() for t, th in found.items()})


def cmd_setup(args, ask) -> int:
    root = Path.cwd()
    backend = ask(f"Compute backend ({' or '.join(BACKENDS)})", "modal")
    if backend not in BACKENDS:
        print(f"unknown backend {backend!r}", file=sys.stderr)
        return 2
    kinds = ", ".join(k for k in KINDS if k != "fake")
    kind = ask(f"Provider ({kinds})", "anthropic")
    # "fake" is the in-process test double: it is in KINDS, but setting it up would write a
    # config whose runs never touch an LLM at all.
    if kind not in KINDS or kind == "fake":
        print(f"unknown provider {kind!r}", file=sys.stderr)
        return 2
    model = ask(*_model_prompt(kind))
    api_base = ask("API base URL") if kind == "custom" else None
    api_key = None
    if kind in ("anthropic", "openai", "google", "openrouter", "custom"):
        api_key = ask("API key", secret=True)
    mode = "single-shot"
    if kind in CLI_PROVIDERS:
        mode = ask("Mode (single-shot or agentic)", "single-shot")
    token_id = token_secret = c3_api_key = None
    if backend == "c3":
        c3_api_key = ask("C3 API key (blank to use your `c3 login` session)", secret=True) or None
    if backend == "modal":
        token_id = ask("Modal token id (create at modal.com/settings/tokens)")
        token_secret = ask("Modal token secret", secret=True)
    local = None
    if backend == "local":
        try:
            local = LocalSettings(
                cpus=_ask_number(ask, "CPUs for the local container", str(os.cpu_count() or 1),
                                 int),
                memory_gib=_ask_number(ask, "Memory for the local container in GiB",
                                       str(default_local_memory_gib()), int))
        except ConfigError as e:  # three non-numbers, as the run wizard treats it
            print(str(e), file=sys.stderr)
            return 2
    provider = make_provider(kind, model, api_key=api_key, api_base=api_base)
    err = validate_provider(provider)
    if err:
        print(f"Provider check failed: {err}", file=sys.stderr)
        return 1
    try:
        if backend == "c3":
            env_key = os.environ.get("C3_API_KEY") or None
            check_key = c3_api_key or env_key
            if c3_api_key:
                print("C3: using the hosted MCP endpoint with your API key; no c3 CLI needed.")
            elif env_key:
                print("C3: using the hosted MCP endpoint with the C3_API_KEY in your "
                      "environment; a shell without it falls back to the c3 CLI.")
            else:
                print("C3: using the c3 CLI and its `c3 login` session.")
            check_c3(api_key=check_key)
        elif backend == "local":
            gpu = check_local()
            print("Local: jobs run in Docker on this machine. GPU challenges: "
                  + ("available (nvidia runtime found)." if gpu
                     else "not available (no nvidia runtime)."))
        else:
            deploy_bench(token_id or None, token_secret or None)
    except ConfigError as e:
        print(str(e), file=sys.stderr)
        return 1
    save(root, Config(provider=kind, model=model, mode=mode, api_base=api_base,
                      backend=backend, local_cpus=local.cpus if local else None,
                      local_memory_gib=local.memory_gib if local else None),
         api_key, c3_api_key=c3_api_key)
    print("Setup complete. Run `talos run` to start a job.")
    return 0


def _ask_number(ask, prompt: str, default: str, cast=float, tries: int = 3):
    """Wizard answers are typed by a human: "abc" or "20 usd" must re-prompt rather than
    traceback out of `float(...)` and throw away everything already answered."""
    for _ in range(tries):
        raw = ask(prompt, default)
        try:
            return cast(raw)
        except (TypeError, ValueError):
            print(f"not a number: {raw!r}", file=sys.stderr)
    raise ConfigError(f"{prompt}: no number given after {tries} tries")


def _challenge_prompt() -> str:
    """spec §5.2: GPU challenges are marked, so the expensive choices are visible before one
    is picked."""
    names = [f"{name} (GPU)" if cs.is_gpu else name for name, cs in CHALLENGES.items()]
    return f"Challenge ({', '.join(names)})"


def _codex_refused(cfg: Config) -> bool:
    """spec §10: say so before a single dollar is spent, not at the first iteration."""
    from talos.agentic import CODEX_AGENTIC_REFUSAL, codex_agentic_refused
    if codex_agentic_refused(cfg.provider, cfg.mode):
        print(CODEX_AGENTIC_REFUSAL, file=sys.stderr)
        return True
    return False


def _budget_from_args(args) -> Budget:
    b = Budget(usd=args.budget_usd, hours=args.budget_hours, iterations=args.budget_iterations,
               compute_usd=args.budget_compute_usd)
    b.validate()
    return b


# ── the fake end-to-end run (provider "fake"): no Modal, no network ───────────

def _fake_scores(challenge: str, files: dict[str, str], ns):
    k = int(re.search(r"let k = (\d+);", files["mod.rs"]).group(1))
    return [100 + k - 1 for _ in ns.nonces()]  # k=1 is baseline parity


def _fake_script(system: str, user: str) -> str:
    if '"strategy_tag" (one of' in system:
        return ('{"title": "bump k", "description": "raise k by one", '
                '"strategy_tag": "local_search"}')
    k = int(re.search(r"let k = (\d+);", user).group(1))
    return f"<<<<<<< SEARCH mod.rs\nlet k = {k};\n=======\nlet k = {k + 1};\n>>>>>>> REPLACE\n"


FAKE_MAINNET = types.SimpleNamespace(
    top_algorithm=lambda ch: ("fake_base", "c003_a000", 1),
    top_hyperparameters=lambda algorithm_id, tracks, fuel: {
        t: TrackHyperparameters({"fake_boost": 1}, "fake-benchmark", "0xfake", 100.0)
        for t in tracks},
    fetch_algorithm_files=lambda ch, name: {"mod.rs": "fn solve() { let k = 1; }\n"},
    fetch_template=lambda ch: "pub fn solve_challenge(")


def unpriced(provider: str, model: str) -> bool:
    """True when the provider bills for tokens but Talos has no price for this model. Every
    Completion then carries cost_usd=None, so llm_usd stays 0.00: it must be reported as
    "unpriced", never as a measured $0.00, and --budget-usd cannot be enforced against it."""
    return provider not in UNMETERED and estimate_cost(model, Usage(1, 1)) is None


def _status_line(spec: JobSpec, state: JobState, now: float) -> str:
    best = f"{state.best.delta['mean_rel_delta']:+.3%}" if state.best else "n/a"
    if spec.budget.hours is None:
        left = "∞"
    else:
        left = f"{max(0.0, spec.budget.hours - (now - state.spend.started_at) / 3600):.1f}h"
    llm = "unpriced" if unpriced(spec.provider, spec.model) else f"${state.spend.llm_usd:.2f}"
    return f"best {best} | llm {llm} | compute ≈${state.spend.compute_usd:.2f} | {left} left"


EVENT_LINE_WIDTH = 100
OUTCOME_TEXT = {
    "failed:score": "no improvement",
    "failed:edit": "no candidate: edit failed",
    "failed:compile": "no candidate: did not compile",
    "failed:dead_code": "no candidate: new code never called",
    "failed:runtime": "rejected: error rate over the ceiling",
}


def _one_line(text: str, width: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[:width - 3] + "..."


def _holdout_text(data: dict) -> str:
    if "holdout" not in data:
        return str(data.get("error", ""))
    h = data["holdout"]
    return (f"held-out {h['mean_rel_delta']:+.3%} vs baseline "
            f"(worst track {h['worst_rel_delta']:+.3%})")


def _event_line(kind: str, data: dict, width: int = EVENT_LINE_WIDTH) -> str | None:
    """One loop event as one terminal line, or None for an event the terminal does not show.
    timeline.jsonl keeps every event with every field; this is the summary of it."""
    if kind == "stopped":
        return None  # the summary block printed after the loop reports it
    if kind == "baseline":
        text = data["message"]
    elif kind == "baseline_ready":
        text = "baseline ready"
    elif kind == "hypothesis":
        tag = f" ({data['strategy_tag']})" if data.get("strategy_tag") else ""
        text = f"trying: {data.get('title', '')}{tag}"
    elif kind == "scored":
        runtime = data.get("runtime_ratio")
        text = (f"scored {data['mean_rel_delta']:+.3%} vs baseline "
                f"(worst track {data['worst_rel_delta']:+.3%}, errors {data['error_rate']:.1%}"
                + (f", runtime x{runtime:.2f})" if runtime is not None else ")"))
    elif kind == "compile_failed":
        # a timeline written before the loop recorded `error` has only the output's tail
        error = data.get("error") or first_error(data.get("output") or "")
        text = f"compile failed: {error}" if error else "compile failed"
    elif kind == "edits_rejected":
        text = f"edit rejected: {len(data['paths'])} path(s) outside the algorithm files"
    elif kind == "dead_code":
        text = "new code never called: " + ", ".join(data["names"])
    elif kind == "iteration_done":
        text = OUTCOME_TEXT.get(data["outcome"], data["outcome"])
        if data["outcome"] == "improved":
            text = "new best"
        else:
            text += f" ({data['runs_since_improvement']} in a row)"
    elif kind == "reset":
        text = f"switching strategy to: {data['forced_tag']}"
    elif kind == "distilled":
        text = f"lesson: {data['lesson']}"
    elif kind == "confirming":
        text = "confirming on held-out nonces"
    elif kind == "won":
        text = f"won: {_holdout_text(data)}"
    elif kind == "false_positive":
        text = f"not confirmed: {_holdout_text(data)}"
    elif kind == "rate_limited":
        text = f"rate limited, waiting {data['wait_s']}s"
    elif kind == "resumed_pending":
        text = f"reattached to bench job {data['job_id']}"
    elif kind == "discarded_incomplete":
        text = f"discarded {data['discarded']} unfinished iteration(s) from the previous run"
    else:
        # A kind added to the loop later still reaches the terminal.
        text = f"{kind} " + " ".join(f"{k}={v}" for k, v in data.items())
    return _one_line(text, width)


def execute_job(spec: JobSpec, store: JobStore, cfg: Config, resume: bool) -> int:
    # spec §7.1: refuse before anything is spent when the credential is missing.
    if cfg.provider not in UNMETERED and resolve_api_key(cfg) is None:
        env = ENV_KEYS.get(cfg.provider, "TALOS_CUSTOM_API_KEY")
        print(f"no API key for provider {cfg.provider}: run `talos setup` or set {env}",
              file=sys.stderr)
        return 2
    from talos.package import build_package
    state = store.load() if resume else JobState.fresh(Spend(started_at=time.time()))
    if resume:
        if state.status in ("won", "exhausted"):
            print(f"job {spec.job_id} already {state.status}; nothing to resume")
            print(f"Package: {build_package(spec, state, store)}")
            return 1
        if state.status in ("cancelled", "failed", "paused"):
            previous = state.status
            # The wall-clock budget measures time the job was WORKING. started_at is shifted
            # forward by the pause (approximated as the time since the last save), or a job
            # resumed after its window has passed exits "exhausted (hours)" at once, for ever.
            paused_s = max(0.0, time.time() - (store.run_dir / "state.json").stat().st_mtime)
            state.spend.started_at += paused_s
            state.status, state.stop_reason = "researching", None
            store.save(state)
            store.event("resumed", previous_status=previous, paused_s=round(paused_s, 1))
    else:
        store.save(state)

    from talos.agentic import attach_agentic
    from talos.bench import BenchCancelled, PendingJobStore
    fake = cfg.provider == "fake"

    def _set_pending(d):
        state.pending_job = d
        store.save(state)

    # The backend records the job it has in flight in the run's own state, so a resumed run
    # reattaches to it instead of abandoning one C3 is still billing for.
    pending = PendingJobStore(get=lambda: state.pending_job, set=_set_pending)
    local = gpu_name = None
    if fake:
        from talos.bench import FakeBench
        from talos.providers.fake import FakeProvider
        provider, bench = FakeProvider(_fake_script), FakeBench(_fake_scores)
        cache_dir, mainnet = store.run_dir / "baseline_cache", FAKE_MAINNET
    else:
        provider = make_provider(cfg.provider, cfg.model, api_key=resolve_api_key(cfg),
                                 api_base=cfg.api_base)
        c3_api_key = resolve_c3_api_key(cfg) if cfg.backend == "c3" else None
        local = local_settings(cfg) if cfg.backend == "local" else None
        bench = make_bench(cfg.backend, store.run_dir, pending, c3_api_key=c3_api_key,
                           local=local)
        if c3_api_key:
            # `talos compile` in the agentic sandbox has no secrets.json to read the key from.
            os.environ["C3_API_KEY"] = c3_api_key
        cache_dir, mainnet = BASELINE_CACHE, None
        # Before the baseline, not after: C3 pulls the image at job start, so a missing tag
        # costs a whole job (and its queue wait) to report a pull failure.
        if cfg.backend == "c3" and not image_available(spec.challenge):
            # A run that stops here is over: left at its initial status `talos status` would
            # list it as live for ever, with no reason recorded.
            state.status, state.stop_reason = "failed", "dev image not on GHCR"
            store.save(state)
            print(f"image {c3_image(spec.challenge)} is not on GHCR (or GHCR is "
                  f"unreachable): {IMAGE_HINT}", file=sys.stderr)
            return 1
        if cfg.backend == "local":
            if CHALLENGES[spec.challenge].is_gpu and not has_gpu_runtime():
                state.status, state.stop_reason = "failed", "no NVIDIA container runtime"
                store.save(state)
                print(f"{spec.challenge} needs a GPU and Docker reports no NVIDIA runtime; "
                      f"install the NVIDIA container toolkit, or run this challenge on the "
                      f"modal or c3 backend", file=sys.stderr)
                return 1
            try:
                # Before the baseline: the first run per challenge pulls the image and does one
                # clean build, and the user should see that happening rather than a silent wait.
                gpu_name = prepare(spec.challenge, uid=host_uid())
            except C3CommandError as e:
                state.status, state.stop_reason = "failed", f"docker: {str(e)[:120]}"
                store.save(state)
                print(f"Docker failed while preparing {spec.challenge}: {e}", file=sys.stderr)
                return 1
    from talos.loop import Loop
    hardware = bench_hardware_class(cfg.backend, spec.challenge, local=local, gpu_name=gpu_name)
    # `talos compile` inside the agentic sandbox runs in a worktree with no talos.config.json.
    os.environ["TALOS_BACKEND"] = "modal" if fake else cfg.backend

    def on_event(kind, data):
        # loop._n is the iteration the event belongs to; state.iteration only catches up when an
        # iteration finishes, so it labels iteration 1's events "#0".
        n = getattr(loop, "_n", state.iteration)
        prefix = f"[{time.strftime('%H:%M:%S')}] " + (f"#{n} " if n else "")
        width = shutil.get_terminal_size((EVENT_LINE_WIDTH + len(prefix), 24)).columns
        text = _event_line(kind, data, max(40, width - len(prefix)))
        if text is None:
            return
        if kind == "iteration_done":
            text = f"{text} | {_status_line(spec, state, time.time())}"
        if kind == "hypothesis":
            print(flush=True)  # a blank line between iterations
        print(prefix + text, flush=True)

    loop = Loop(spec, state, store, provider, bench, template_rs="", on_event=on_event)
    if cfg.mode == "agentic":
        attach_agentic(loop, cfg.provider, cfg.model)
    previous_sigint = signal.signal(signal.SIGINT,
                                   lambda *_: (loop.request_stop(), bench.request_stop()))
    try:
        if state.baseline is None:
            loop.measure_baseline(cache_dir, hardware, mainnet=mainnet)
        elif mainnet is not None:
            loop.template_rs = mainnet.fetch_template(spec.challenge)
        else:
            from talos.mainnet import fetch_template
            loop.template_rs = fetch_template(spec.challenge)
        final = loop.run()
    except BenchCancelled as e:
        # Only measure_baseline can raise it here: run() records its own cancelled outcome.
        # Without this the blanket handler below would call a stopped run a failed one.
        state.status, state.stop_reason = "cancelled", f"stopped; bench job cancelled: {e}"
        store.save(state)
        final = state
    except Exception as e:  # noqa: BLE001 - report, package what we have, exit non-zero
        state.status, state.stop_reason = "failed", str(e)
        store.save(state)
        final = state
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
    pkg = build_package(spec, final, store)
    print(f"\nStatus: {final.status} ({final.stop_reason})")
    if final.best is not None:
        print(f"Best delta vs baseline: {final.best.delta['mean_rel_delta']:+.3%}")
    else:
        print("Best delta vs baseline: n/a (no candidate)")
    llm = (f"unpriced (no price-table entry for {spec.model})"
           if unpriced(spec.provider, spec.model) else f"${final.spend.llm_usd:.2f}")
    print(f"LLM spend: {llm}   Compute spend (estimated): ${final.spend.compute_usd:.2f}")
    print(f"Package: {pkg}")
    return 0 if final.status == "won" else 1


def cmd_run(args, ask) -> int:
    root = Path.cwd()
    # spec §11: --fake needs neither a config file nor network, so it must be resolved before
    # `load(root)` is even attempted; a fresh clone with no talos.config.json still runs it.
    if args.fake:
        cfg = Config(provider="fake", model="fake", mode="single-shot", api_base=None,
                     backend="modal")  # the fake bench is in-process: no backend to pick
    else:
        try:
            cfg = load(root)
        except ConfigError as e:
            print(str(e), file=sys.stderr)
            return 2
    # A backend that reached the config by hand would otherwise raise out of make_bench, which
    # sits past every early return in execute_job: a traceback rather than a message.
    if cfg.backend not in BACKENDS:
        print(f"unknown backend {cfg.backend!r} in talos.config.json; run `talos setup`",
              file=sys.stderr)
        return 2
    if args.resume:
        run_dir = root / "runs" / args.resume
        # A run that was refused before its first save (a bad credential, say) has job.json but
        # no state.json; loading it would traceback instead of reporting a resumable job.
        missing = [f for f in ("job.json", "state.json") if not (run_dir / f).exists()]
        if missing:
            print(f"no job to resume at {run_dir}: missing {', '.join(missing)}", file=sys.stderr)
            return 2
        store = JobStore(run_dir)
        spec = store.read_spec()
        # job.json records how the job was started, and every iteration so far was produced that
        # way. A resume runs the same provider, model and mode whatever the config says today.
        if args.mode and args.mode != spec.mode:
            print(f"job {spec.job_id} was started in {spec.mode} mode; start a new job to "
                  f"change mode", file=sys.stderr)
            return 2
        if args.track is not None and _track_arg(args.track) != spec.track:
            print(f"job {spec.job_id} was started with track {spec.track or 'all'}; start a new "
                  f"job to change track", file=sys.stderr)
            return 2
        if args.hyperparameters is not None:
            print(f"job {spec.job_id} fixed its hyperparameters at start; start a new job to "
                  f"change them", file=sys.stderr)
            return 2
        cfg = replace(cfg, provider=spec.provider, model=spec.model, mode=spec.mode)
        if _codex_refused(cfg):
            return 2
        return execute_job(spec, store, cfg, resume=True)
    if args.mode:
        if args.mode == "agentic" and cfg.provider not in CLI_PROVIDERS:
            print(f"--mode agentic needs a CLI provider ({' or '.join(CLI_PROVIDERS)}), "
                  f"not {cfg.provider!r}", file=sys.stderr)
            return 2
        cfg = replace(cfg, mode=args.mode)
    if _codex_refused(cfg):
        return 2
    challenge = args.challenge or ask(_challenge_prompt(), "vehicle_routing")
    if challenge not in CHALLENGES:
        print(f"unknown challenge {challenge!r}", file=sys.stderr)
        return 2
    if args.direction and args.direction_file:
        print("pass either --direction or --direction-file, not both", file=sys.stderr)
        return 2
    direction = args.direction
    if args.direction_file:
        df = Path(args.direction_file)
        if not df.exists():
            print(f"direction file not found: {df}", file=sys.stderr)
            return 2
        direction = df.read_text(encoding="utf-8")
    if not direction:
        direction = ask("Direction for the agent (what to explore)")
    metered = cfg.provider not in CLI_PROVIDERS
    try:
        if not args.yes and args.budget_usd is None and args.budget_hours is None \
                and args.budget_iterations is None:
            if metered:
                args.budget_usd = _ask_number(ask, "LLM budget in USD", "20")
            else:
                args.budget_iterations = _ask_number(ask, "Iteration budget", "50", int)
            args.budget_hours = _ask_number(ask, "Wall-clock budget in hours", "4")
        try:
            budget = _budget_from_args(args)
        except ValueError as e:
            print(str(e), file=sys.stderr)
            return 2
        # spec §5.2: compute spend is always capped, except on the local backend where it is
        # always zero. The default is applied only after the budget has been validated, so it
        # can never stand in for the LLM/time/iteration cap the run needs.
        if budget.compute_usd is None and cfg.backend != "local":
            if args.yes:
                budget = replace(budget, compute_usd=DEFAULT_COMPUTE_USD)
                print(f"No --budget-compute-usd given; capping compute spend at "
                      f"${DEFAULT_COMPUTE_USD:.2f}.")
            else:
                budget = replace(budget, compute_usd=_ask_number(
                    ask, "Compute budget in USD", str(DEFAULT_COMPUTE_USD)))
    except ConfigError as e:
        print(str(e), file=sys.stderr)
        return 2
    # spec §5.2 step 4: mode for CLI providers, with the cost warning before the prompt.
    if not args.yes and args.mode is None and cfg.provider in CLI_PROVIDERS:
        print("agentic mode uses roughly 5-20x the tokens of single-shot")
        mode = ask("Mode (single-shot or agentic)", cfg.mode)
        if mode not in ("single-shot", "agentic"):
            print(f"unknown mode {mode!r}", file=sys.stderr)
            return 2
        cfg = replace(cfg, mode=mode)
        if _codex_refused(cfg):
            return 2
    # A dollar cap on an unpriced model is a cap that can never fire: the provider reports no
    # cost, so llm_usd never rises. Refuse rather than run a job with no effective cap at all.
    if unpriced(cfg.provider, cfg.model):
        if budget.hours is None and budget.iterations is None:
            print(f"model {cfg.model} has no price-table entry, so --budget-usd cannot be "
                  f"enforced; add --budget-hours or --budget-iterations", file=sys.stderr)
            return 2
        print(f"warning: model {cfg.model} has no price-table entry; LLM spend is not measured "
              f"and --budget-usd is not enforced for this run", file=sys.stderr)
    cs = CHALLENGES[challenge]
    if args.fake:
        info = ChallengeInfo(id=cs.id, name=challenge, is_gpu=cs.is_gpu, tracks=["n=1"], max_fuel=1)
    else:
        try:
            info = fetch_challenge_info(challenge)
        except MainnetError as e:
            print(f"mainnet unreachable: {e}", file=sys.stderr)
            return 1
        # The Modal side scores against the static table: its challenge id goes into the runtime
        # settings and its is_gpu picks the CPU or GPU function. If mainnet has moved, every
        # score would be measured under the wrong challenge, so stop before the job exists.
        if info.id != cs.id or info.is_gpu != cs.is_gpu:
            print(f"challenge table drift: mainnet says {info.id}/{info.is_gpu}, Talos has "
                  f"{cs.id}/{cs.is_gpu}; update talos/challenges.py", file=sys.stderr)
            return 1
    track = _track_arg(args.track)
    if args.track is None and not args.yes:
        track = _track_arg(ask(f"Track to optimise (all, or one of: {', '.join(info.tracks)})",
                               "all"))
    if track is not None and track not in info.tracks:
        print(f"unknown track {track!r} for {challenge}; active tracks: "
              f"{', '.join(info.tracks)}", file=sys.stderr)
        return 2
    choice = args.hyperparameters
    if choice is None:
        # A blank answer is the default, as `_track_arg` treats a blank track answer: the wizard
        # test's raw `ask` returns "" rather than the default.
        choice = ("mainnet" if args.yes
                  else ask("Hyperparameters (mainnet or none)", "mainnet").strip() or "mainnet")
    if choice not in ("mainnet", "none"):
        print(f"unknown hyperparameters choice {choice!r}; use mainnet or none", file=sys.stderr)
        return 2
    algorithm = hyperparameters = hp_source = None
    if choice == "mainnet":
        api = FAKE_MAINNET if cfg.provider == "fake" else mainnet_api
        try:
            algorithm, hyperparameters, hp_source = _mainnet_hyperparameters(api, challenge, info)
        except MainnetError as e:
            print(f"mainnet unreachable: {e}", file=sys.stderr)
            return 1
    rand_hash = new_rand_hash()
    training, holdout = draw_nonce_sets(info.tracks, rand_hash)
    stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{challenge}"
    job_id, n = stamp, 1
    while (root / "runs" / job_id / "job.json").exists():  # two runs in the same second
        n += 1
        job_id = f"{stamp}-{n}"
    spec = JobSpec(job_id=job_id, challenge=challenge, direction=direction, provider=cfg.provider,
                   model=cfg.model, mode=cfg.mode, budget=budget, rand_hash=rand_hash,
                   tracks=info.tracks, training=training, holdout=holdout, fuel=info.max_fuel,
                   created_at=time.time(), monorepo_ref=MONOREPO_REF, challenge_id=info.id,
                   track=track, baseline_algorithm=algorithm, hyperparameters=hyperparameters,
                   hyperparameters_source=hp_source)
    store = JobStore(root / "runs" / job_id)
    store.write_spec(spec)
    (store.run_dir / "tacit.md").write_text(f"- USER: {direction.strip()}\n",
                                            encoding="utf-8", newline="\n")
    scope = f"track {track} of {len(info.tracks)} tracks" if track else f"{len(info.tracks)} tracks"
    if hyperparameters is None and algorithm is not None:
        hp_line = (f"hyperparameters: none (no mainnet benchmark of {algorithm['name']} "
                   f"at fuel {info.max_fuel})")
    elif hyperparameters is None:
        hp_line = "hyperparameters: none"
    else:
        used = sum(1 for v in hyperparameters.values() if v is not None)
        hp_line = f"hyperparameters: {used}/{len(info.tracks)} tracks from mainnet"
    print(f"Job {job_id}: {scope}, fuel {info.max_fuel}, budget {budget.to_dict()}; {hp_line}")
    return execute_job(spec, store, cfg, resume=False)


def compile_backend(args, root: Path) -> str:
    """--backend, then TALOS_BACKEND (exported by execute_job for the agentic sandbox, whose
    worktree has no config file), then the config in `root`, then modal."""
    backend = getattr(args, "backend", None) or os.environ.get("TALOS_BACKEND")
    if not backend:
        try:
            backend = load(root).backend
        except ConfigError:
            return "modal"
    if backend not in BACKENDS:
        raise ConfigError(f"unknown backend {backend!r}; choose one of {', '.join(BACKENDS)}")
    return backend


def cmd_compile(args, ask) -> int:
    """Compile only: no nonce is scored. On C3 the job dir is the fixed `.talos/compile/c3/adhoc`,
    which every run rewrites, so two `talos compile` runs at once in one directory are
    unsupported."""
    if args.challenge not in CHALLENGES:
        print(f"unknown challenge {args.challenge!r}", file=sys.stderr)
        return 2
    d = Path(args.dir)
    files = ({p.relative_to(d).as_posix(): p.read_text(encoding="utf-8") for p in d.rglob("*")
              if p.is_file() and p.suffix in (".rs", ".cu")} if d.is_dir() else {})
    # An empty file map builds nothing on the far side: refuse here rather than pay for a
    # container that can only report a mystery failure.
    if not files:
        print(f"no .rs/.cu files under {d}", file=sys.stderr)
        return 2
    from talos.bench import EvalRequest, PendingJobStore
    try:
        backend = compile_backend(args, Path.cwd())
    except ConfigError as e:
        print(str(e), file=sys.stderr)
        return 2
    try:
        cfg = load(Path.cwd())
    except ConfigError:
        cfg = None  # the agentic sandbox: the key comes from C3_API_KEY
    if backend == "local":
        try:
            prepare(args.challenge, uid=host_uid())
        except C3CommandError as e:
            print(f"Docker failed while preparing {args.challenge}: {e}", file=sys.stderr)
            return 1
    bench = make_bench(backend, Path.cwd() / ".talos" / "compile", PendingJobStore.memory(),
                       c3_api_key=resolve_c3_api_key(cfg) if backend == "c3" else None,
                       local=local_settings(cfg) if backend == "local" else None)
    r = bench.evaluate(EvalRequest(args.challenge, files, [], [], 0, None,
                                   CHALLENGES[args.challenge].beat)).compile
    print(r.output[-4000:])
    return 0 if r.ok else 1


def cmd_status(args, ask) -> int:
    root = Path.cwd() / "runs"
    for job in sorted(root.glob("*/state.json")):
        st = json.loads(job.read_text(encoding="utf-8"))
        print(f"{job.parent.name}: {st['status']} it={st['iteration']} "
              f"llm=${st['spend']['llm_usd']:.2f} compute=${st['spend']['compute_usd']:.2f}")
    return 0


def main(argv=None, ask=default_ask) -> int:
    p = argparse.ArgumentParser(prog="talos")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup")
    r = sub.add_parser("run")
    r.add_argument("--challenge")
    r.add_argument("--direction")
    r.add_argument("--direction-file")
    r.add_argument("--track", help="one active track to optimise, or all (the default)")
    r.add_argument("--hyperparameters", choices=["mainnet", "none"],
                   help="per-track hyperparameters from the baseline algorithm's best mainnet "
                        "benchmark (mainnet, the default) or none")
    r.add_argument("--mode", choices=["single-shot", "agentic"])
    r.add_argument("--budget-usd", type=float)
    r.add_argument("--budget-hours", type=float)
    r.add_argument("--budget-iterations", type=int)
    r.add_argument("--budget-compute-usd", type=float)
    r.add_argument("--resume")
    r.add_argument("--yes", action="store_true")
    r.add_argument("--fake", action="store_true", help=argparse.SUPPRESS)
    c = sub.add_parser("compile")
    c.add_argument("--challenge", required=True)
    c.add_argument("--dir", default="algorithm")
    c.add_argument("--backend", choices=list(BACKENDS))
    sub.add_parser("status")
    args = p.parse_args(argv)
    return {"setup": cmd_setup, "run": cmd_run, "compile": cmd_compile, "status": cmd_status}[args.cmd](args, ask)


if __name__ == "__main__":
    sys.exit(main())
