"""`talos setup | run | compile | status`. Wizard prompts go through `ask` so tests can script
them; after every finished iteration a plain one-line status is printed to stdout with the job id,
iteration, best delta, LLM and Modal spend and the wall-clock time left."""
from __future__ import annotations

import argparse
import getpass
import json
import re
import shutil
import signal
import subprocess
import sys
import time
import types
from dataclasses import replace
from pathlib import Path

from talos.budget import Budget, Spend
from talos.challenges import CHALLENGES, MONOREPO_REF, hardware_class
from talos.config import Config, ConfigError, ENV_KEYS, load, resolve_api_key, save
from talos.mainnet import ChallengeInfo, MainnetError, fetch_challenge_info
from talos.nonces import draw_nonce_sets, new_rand_hash
from talos.providers import DEFAULT_MODELS, KINDS, make_provider, validate_provider
from talos.providers.pricing import estimate_cost
from talos.state import JobSpec, JobState, JobStore
from talos.types import Usage

BASELINE_CACHE = Path.home() / ".talos" / "baselines"
MODAL_APP_FILE = Path(__file__).resolve().parent.parent / "modal_app" / "talos_bench.py"
CLI_PROVIDERS = ("claude-cli", "codex-cli")
UNMETERED = CLI_PROVIDERS + ("fake",)
DEFAULT_COMPUTE_USD = 20.0


def default_ask(prompt: str, default: str | None = None, secret: bool = False) -> str:
    suffix = f" [{default}]" if default else ""
    v = getpass.getpass(f"{prompt}{suffix}: ") if secret else input(f"{prompt}{suffix}: ")
    return v.strip() or (default or "")


def deploy_bench(token_id: str | None, token_secret: str | None, run=subprocess.run) -> None:
    modal = shutil.which("modal") or [sys.executable, "-m", "modal"]
    base = modal if isinstance(modal, list) else [modal]
    if token_id and token_secret:
        r = run(base + ["token", "set", "--token-id", token_id, "--token-secret", token_secret],
                capture_output=True, text=True)
        if r.returncode != 0:
            raise ConfigError(f"modal token set failed: {r.stderr[-500:]}")
    r = run(base + ["deploy", str(MODAL_APP_FILE)], capture_output=True, text=True)
    if r.returncode != 0:
        raise ConfigError(f"modal deploy failed: {(r.stderr or r.stdout)[-2000:]}")


def cmd_setup(args, ask) -> int:
    root = Path.cwd()
    kinds = ", ".join(k for k in KINDS if k != "fake")
    kind = ask(f"Provider ({kinds})", "anthropic")
    # "fake" is the in-process test double: it is in KINDS, but setting it up would write a
    # config whose runs never touch an LLM at all.
    if kind not in KINDS or kind == "fake":
        print(f"unknown provider {kind!r}", file=sys.stderr)
        return 2
    model = ask("Model", DEFAULT_MODELS.get(kind) or None)
    api_base = ask("API base URL") if kind == "custom" else None
    api_key = None
    if kind in ("anthropic", "openai", "google", "openrouter", "custom"):
        api_key = ask("API key", secret=True)
    mode = "single-shot"
    if kind in CLI_PROVIDERS:
        mode = ask("Mode (single-shot or agentic)", "single-shot")
    token_id = ask("Modal token id (create at modal.com/settings/tokens)")
    token_secret = ask("Modal token secret", secret=True)
    provider = make_provider(kind, model, api_key=api_key, api_base=api_base)
    err = validate_provider(provider)
    if err:
        print(f"Provider check failed: {err}", file=sys.stderr)
        return 1
    try:
        deploy_bench(token_id or None, token_secret or None)
    except ConfigError as e:
        print(str(e), file=sys.stderr)
        return 1
    save(root, Config(provider=kind, model=model, mode=mode, api_base=api_base), api_key)
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
    """spec §5.2: GPU challenges are labelled with their Modal GPU class and approximate cost,
    so the expensive choices are visible before one is picked."""
    from talos.bench import GPU_USD_PER_SECOND
    names = []
    for name, cs in CHALLENGES.items():
        if cs.is_gpu:
            names.append(f"{name} (GPU: {cs.gpu}, "
                         f"≈${GPU_USD_PER_SECOND[cs.gpu] * 3600:.2f}/h estimated)")
        else:
            names.append(name)
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
    top_algorithm=lambda ch: ("fake_base", 1),
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
    return (f"[status] job={spec.job_id} it={state.iteration} best={best} "
            f"llm={llm} compute≈${state.spend.compute_usd:.2f} left={left}")


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
    fake = cfg.provider == "fake"
    if fake:
        from talos.bench import FakeBench
        from talos.providers.fake import FakeProvider
        provider, bench = FakeProvider(_fake_script), FakeBench(_fake_scores)
        cache_dir, mainnet = store.run_dir / "baseline_cache", FAKE_MAINNET
    else:
        from talos.bench import ModalBench
        provider = make_provider(cfg.provider, cfg.model, api_key=resolve_api_key(cfg),
                                 api_base=cfg.api_base)
        bench = ModalBench()
        cache_dir, mainnet = BASELINE_CACHE, None
    from talos.loop import Loop
    hardware = hardware_class(CHALLENGES[spec.challenge])

    def on_event(kind, data):
        # loop._n is the iteration the event belongs to; state.iteration only catches up when an
        # iteration finishes, so it labels iteration 1's events "it=0".
        n = getattr(loop, "_n", state.iteration)
        line = f"[{time.strftime('%H:%M:%S')}] it={n} {kind} " + \
               " ".join(f"{k}={str(v)[:60]}" for k, v in data.items())
        print(line, flush=True)
        if kind == "iteration_done":
            print(_status_line(spec, state, time.time()), flush=True)

    loop = Loop(spec, state, store, provider, bench, template_rs="", on_event=on_event)
    if cfg.mode == "agentic":
        attach_agentic(loop, cfg.provider, cfg.model)
    previous_sigint = signal.signal(signal.SIGINT, lambda *_: loop.request_stop())
    try:
        if state.baseline is None:
            loop.measure_baseline(cache_dir, hardware, mainnet=mainnet)
        elif mainnet is not None:
            loop.template_rs = mainnet.fetch_template(spec.challenge)
        else:
            from talos.mainnet import fetch_template
            loop.template_rs = fetch_template(spec.challenge)
        final = loop.run()
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
        cfg = Config(provider="fake", model="fake", mode="single-shot", api_base=None)
    else:
        try:
            cfg = load(root)
        except ConfigError as e:
            print(str(e), file=sys.stderr)
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
        direction = df.read_text()
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
        # spec §5.2: compute spend is always capped. The default is applied only after the budget
        # has been validated, so it can never stand in for the LLM/time/iteration cap the run
        # needs.
        if budget.compute_usd is None:
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
                   created_at=time.time(), monorepo_ref=MONOREPO_REF, challenge_id=info.id)
    store = JobStore(root / "runs" / job_id)
    store.write_spec(spec)
    (store.run_dir / "tacit.md").write_text(f"- USER: {direction.strip()}\n")
    print(f"Job {job_id}: {len(info.tracks)} tracks, fuel {info.max_fuel}, budget {budget.to_dict()}")
    return execute_job(spec, store, cfg, resume=False)


def cmd_compile(args, ask) -> int:
    d = Path(args.dir)
    files = ({str(p.relative_to(d)): p.read_text() for p in d.rglob("*")
              if p.is_file() and p.suffix in (".rs", ".cu")} if d.is_dir() else {})
    # An empty file map builds nothing on the far side: refuse here rather than pay for a
    # container that can only report a mystery failure.
    if not files:
        print(f"no .rs/.cu files under {d}", file=sys.stderr)
        return 2
    from talos.bench import ModalBench
    r = ModalBench().compile(args.challenge, files)
    print(r.output[-4000:])
    return 0 if r.ok else 1


def cmd_status(args, ask) -> int:
    root = Path.cwd() / "runs"
    for job in sorted(root.glob("*/state.json")):
        st = json.loads(job.read_text())
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
    sub.add_parser("status")
    args = p.parse_args(argv)
    return {"setup": cmd_setup, "run": cmd_run, "compile": cmd_compile, "status": cmd_status}[args.cmd](args, ask)


if __name__ == "__main__":
    sys.exit(main())
