"""`talos setup | run | compile | status`. Wizard prompts go through `ask` so tests can script
them; `rich` renders the live status line."""
from __future__ import annotations

import argparse
import getpass
import json
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from talos.budget import Budget, Spend
from talos.challenges import CHALLENGES, MONOREPO_REF
from talos.config import Config, ConfigError, load, resolve_api_key, save
from talos.mainnet import MainnetError, fetch_challenge_info
from talos.nonces import draw_nonce_sets, new_rand_hash
from talos.providers import DEFAULT_MODELS, KINDS, make_provider, validate_provider
from talos.state import JobSpec, JobState, JobStore

BASELINE_CACHE = Path.home() / ".talos" / "baselines"
MODAL_APP_FILE = Path(__file__).resolve().parent.parent / "modal_app" / "talos_bench.py"


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
    if kind not in KINDS:
        print(f"unknown provider {kind!r}", file=sys.stderr)
        return 2
    model = ask("Model", DEFAULT_MODELS.get(kind) or None)
    api_base = ask("API base URL") if kind == "custom" else None
    api_key = None
    if kind in ("anthropic", "openai", "google", "openrouter", "custom"):
        api_key = ask("API key", secret=True)
    mode = "single-shot"
    if kind in ("claude-cli", "codex-cli"):
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


def _budget_from_args(args) -> Budget:
    b = Budget(usd=args.budget_usd, hours=args.budget_hours, iterations=args.budget_iterations,
               modal_usd=args.budget_modal_usd)
    b.validate()
    return b


def execute_job(spec: JobSpec, store: JobStore, cfg: Config, resume: bool) -> int:
    from talos.bench import ModalBench
    from talos.loop import Loop
    from talos.package import build_package
    from talos.agentic import attach_agentic
    provider = make_provider(cfg.provider, cfg.model, api_key=resolve_api_key(cfg), api_base=cfg.api_base)
    bench = ModalBench()
    state = store.load() if resume else JobState.fresh(Spend(started_at=time.time()))
    if not resume:
        store.save(state)
    spec_cls = CHALLENGES[spec.challenge]
    hardware = spec_cls.gpu or f"cpu{spec_cls.cpu}"

    def on_event(kind, data):
        line = f"[{time.strftime('%H:%M:%S')}] it={state.iteration} {kind} " + \
               " ".join(f"{k}={str(v)[:60]}" for k, v in data.items())
        print(line, flush=True)

    loop = Loop(spec, state, store, provider, bench, template_rs="", on_event=on_event)
    if cfg.mode == "agentic":
        attach_agentic(loop, cfg.provider, cfg.model)
    signal.signal(signal.SIGINT, lambda *_: loop.request_stop())
    try:
        if state.baseline is None:
            loop.measure_baseline(BASELINE_CACHE, hardware)
        else:
            from talos.mainnet import fetch_template
            loop.template_rs = fetch_template(spec.challenge)
        final = loop.run()
    except Exception as e:  # noqa: BLE001 - report, package what we have, exit non-zero
        state.status, state.stop_reason = "failed", str(e)
        store.save(state)
        final = state
    pkg = build_package(spec, final, store)
    print(f"\nStatus: {final.status} ({final.stop_reason})")
    print(f"LLM spend: ${final.spend.llm_usd:.2f}   Modal spend (estimated): ${final.spend.modal_usd:.2f}")
    print(f"Package: {pkg}")
    return 0 if final.status == "won" else 1


def cmd_run(args, ask) -> int:
    root = Path.cwd()
    try:
        cfg = load(root)
    except ConfigError as e:
        print(str(e), file=sys.stderr)
        return 2
    if args.resume:
        store = JobStore(root / "runs" / args.resume)
        return execute_job(store.read_spec(), store, cfg, resume=True)
    challenge = args.challenge or ask(f"Challenge ({', '.join(CHALLENGES)})", "vehicle_routing")
    if challenge not in CHALLENGES:
        print(f"unknown challenge {challenge!r}", file=sys.stderr)
        return 2
    direction = args.direction
    if args.direction_file:
        direction = Path(args.direction_file).read_text()
    if not direction:
        direction = ask("Direction for the agent (what to explore)")
    metered = cfg.provider not in ("claude-cli", "codex-cli")
    if not args.yes and args.budget_usd is None and args.budget_hours is None \
            and args.budget_iterations is None:
        if metered:
            args.budget_usd = float(ask("LLM budget in USD", "20"))
        else:
            args.budget_iterations = int(ask("Iteration budget", "50"))
        args.budget_hours = float(ask("Wall-clock budget in hours", "4"))
    try:
        budget = _budget_from_args(args)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2
    try:
        info = fetch_challenge_info(challenge)
    except MainnetError as e:
        print(f"mainnet unreachable: {e}", file=sys.stderr)
        return 1
    rand_hash = new_rand_hash()
    training, holdout = draw_nonce_sets(info.tracks, rand_hash)
    job_id = time.strftime("%Y%m%d-%H%M%S") + f"-{challenge}"
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
    from talos.bench import ModalBench
    d = Path(args.dir)
    files = {str(p.relative_to(d)): p.read_text() for p in d.rglob("*")
             if p.is_file() and p.suffix in (".rs", ".cu")}
    r = ModalBench().compile(args.challenge, files)
    print(r.output[-4000:])
    return 0 if r.ok else 1


def cmd_status(args, ask) -> int:
    root = Path.cwd() / "runs"
    for job in sorted(root.glob("*/state.json")):
        st = json.loads(job.read_text())
        print(f"{job.parent.name}: {st['status']} it={st['iteration']} "
              f"llm=${st['spend']['llm_usd']:.2f} modal=${st['spend']['modal_usd']:.2f}")
    return 0


def main(argv=None, ask=default_ask) -> int:
    p = argparse.ArgumentParser(prog="talos")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup")
    r = sub.add_parser("run")
    r.add_argument("--challenge")
    r.add_argument("--direction")
    r.add_argument("--direction-file")
    r.add_argument("--budget-usd", type=float)
    r.add_argument("--budget-hours", type=float)
    r.add_argument("--budget-iterations", type=int)
    r.add_argument("--budget-modal-usd", type=float)
    r.add_argument("--resume")
    r.add_argument("--yes", action="store_true")
    c = sub.add_parser("compile")
    c.add_argument("--challenge", required=True)
    c.add_argument("--dir", default="algorithm")
    sub.add_parser("status")
    args = p.parse_args(argv)
    return {"setup": cmd_setup, "run": cmd_run, "compile": cmd_compile, "status": cmd_status}[args.cmd](args, ask)


if __name__ == "__main__":
    sys.exit(main())
