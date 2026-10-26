"""`talos doctor`: free health checks, and an optional confirmed `--full` that compiles and
scores one baseline on the configured backend. Every dependency is reached through the `cli`
and `mainnet` modules so tests can replace it."""
from __future__ import annotations

import http.client
import json
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from talos import cli, local_transport, mainnet
from talos.bench import CPU_USD_PER_CORE_SECOND, GPU_USD_PER_SECOND, MEM_USD_PER_GIB_SECOND
from talos.c3_bench import GBP_PER_HOUR
from talos.challenges import (CHALLENGES, DEV_IMAGE_TAG, MODAL_GPUS, c3_hardware_options,
                              gpu_slug)
from talos.config import ENV_KEYS, Config, ConfigError, load, resolve_api_key, resolve_c3_api_key
from talos.wizard import confirm

OK, WARN, FAIL, SKIP = "ok", "warn", "FAIL", "skip"
MODAL_APP_NAME = "talos-bench"
CLI_BINARIES = {"claude-cli": "claude", "codex-cli": "codex"}
# Assumed wall seconds of one --full run, for the cost estimate only. Not measured.
FULL_ASSUMED_S = {"modal_cpu": 600, "modal_gpu": 1800, "c3_cpu": 1200, "c3_gpu": 2400}
FREE_NAMES = ("config", "credential", "provider", "backend", "mainnet", "images")


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str = ""


def modal_function_names() -> list[str]:
    """Every function `modal_app/talos_bench.py::register` deploys."""
    names = []
    for name, spec in CHALLENGES.items():
        suffixes = [f"_{gpu_slug(g)}" for g in MODAL_GPUS] if spec.is_gpu else [""]
        for suffix in suffixes:
            names += [f"compile_{name}{suffix}", f"score_batch_{name}{suffix}"]
    return names + [f"probe_{gpu_slug(g)}" for g in MODAL_GPUS]


def modal_lookup(app: str, name: str) -> bool:
    """True when the deployed app has the function. Only NotFoundError means missing; an auth
    or network error propagates."""
    import modal
    import modal.exception
    try:
        modal.Function.from_name(app, name).hydrate()
    except modal.exception.NotFoundError:
        return False
    return True


def check_config(root: Path) -> tuple[Check, Config | None]:
    try:
        cfg = load(root)
    except ConfigError as e:
        msg = str(e)
        return Check("config", FAIL, msg if "talos setup" in msg
                     else f"{msg}; run `talos setup`"), None
    except (ValueError, KeyError, TypeError) as e:
        return Check("config", FAIL, f"talos.config.json is unreadable ({e!r}); run "
                                     f"`talos setup`"), None
    return Check("config", OK, f"{cfg.provider} {cfg.model}, backend {cfg.backend}"), cfg


def check_credential(cfg: Config) -> Check:
    if cfg.provider in CLI_BINARIES:
        binary = CLI_BINARIES[cfg.provider]
        if shutil.which(binary) is None:
            return Check("credential", FAIL, f"the `{binary}` CLI is not on PATH; install it "
                                             f"and log in, or run `talos setup` for another "
                                             f"provider")
        return Check("credential", OK, f"`{binary}` found")
    if cfg.provider == "fake":
        return Check("credential", OK, "fake provider, no key")
    if resolve_api_key(cfg) is None:
        env = ENV_KEYS.get(cfg.provider, "TALOS_CUSTOM_API_KEY")
        return Check("credential", FAIL, f"no API key for {cfg.provider}; set {env} or run "
                                         f"`talos setup`")
    return Check("credential", OK, "API key found")


def check_provider(cfg: Config) -> Check:
    try:
        p = cli.make_provider(cfg.provider, cfg.model, api_key=resolve_api_key(cfg),
                              api_base=cfg.api_base)
    except ValueError as e:
        return Check("provider", FAIL, f"{e}; run `talos setup`")
    err = cli.validate_provider(p)
    if err:
        return Check("provider", FAIL, f"{err}; run `talos setup`")
    return Check("provider", OK, "one test call answered")


def _check_modal() -> Check:
    names = modal_function_names()
    try:
        missing = [n for n in names if not modal_lookup(MODAL_APP_NAME, n)]
    except Exception as e:  # import, token or network: any of them means no usable deploy
        return Check("backend", FAIL, f"Modal lookup failed: {type(e).__name__}: {e}; run "
                                      f"`talos setup` to set the token and deploy")
    if missing:
        shown = ", ".join(missing[:3]) + (", ..." if len(missing) > 3 else "")
        return Check("backend", FAIL, f"{len(missing)} of {len(names)} Modal functions in "
                                      f"{MODAL_APP_NAME} are missing ({shown}); run "
                                      f"`talos setup` to deploy")
    return Check("backend", OK, f"deployed ({len(names)} functions)")


def check_backend(cfg: Config) -> Check:
    if cfg.backend == "modal":
        return _check_modal()
    if cfg.backend == "c3":
        try:
            balance = cli.check_c3(api_key=resolve_c3_api_key(cfg))
        except ConfigError as e:
            return Check("backend", FAIL, str(e))
        if balance == 0.0:
            return Check("backend", WARN, "could not read the balance, or it is £0.00; "
                                          "c3 topup")
        if balance < 1.0:
            return Check("backend", WARN, f"£{balance:.2f} left; c3 topup")
        return Check("backend", OK, f"£{balance:.2f}")
    if cfg.backend == "local":
        try:
            info = cli.check_local()
        except ConfigError as e:
            return Check("backend", FAIL, str(e))
        gpu = "yes" if "nvidia" in info.runtimes else "no"
        return Check("backend", OK, f"{info.ncpu} CPUs, {info.mem_total_gib} GiB, "
                                    f"NVIDIA runtime: {gpu}")
    return Check("backend", FAIL, f"unknown backend {cfg.backend!r}; run `talos setup`")


def check_mainnet() -> Check:
    try:
        infos = {i.name: i for i in mainnet.fetch_challenges()}
    except (mainnet.MainnetError, ValueError, OSError, http.client.HTTPException) as e:
        # ValueError: a captive portal's HTML page is not JSON. OSError: a connection dropped
        # on a path that does not wrap it. Doctor diagnoses the network, so none may escape.
        return Check("mainnet", FAIL, f"unreachable: {e}; check your network")
    drift = []
    for name, spec in CHALLENGES.items():
        info = infos.get(name)
        if info is None:
            drift.append(f"{name} not on mainnet")
        elif info.id != spec.id or info.is_gpu != spec.is_gpu:
            drift.append(f"{name}: mainnet says {info.id}/{info.is_gpu}, Talos has "
                         f"{spec.id}/{spec.is_gpu}")
    if drift:
        return Check("mainnet", FAIL, f"challenge table drift: {'; '.join(drift)}; update "
                                      f"talos/challenges.py")
    return Check("mainnet", OK, f"{len(CHALLENGES)} challenges match")


def check_images() -> Check:
    names = list(CHALLENGES)
    with ThreadPoolExecutor(max_workers=len(names)) as pool:
        found = list(pool.map(cli.image_available, names))
    missing = [n for n, ok in zip(names, found) if not ok]
    if missing:
        return Check("images", FAIL, f"dev image {DEV_IMAGE_TAG} not found on GHCR (or GHCR "
                                     f"unreachable) for {', '.join(missing)}; run "
                                     f"`talos check-updates`")
    return Check("images", OK, f"dev image {DEV_IMAGE_TAG} on GHCR for {len(names)} challenges")


def free_checks(root: Path) -> tuple[list[Check], Config | None]:
    config, cfg = check_config(root)
    checks = [config]
    if cfg is None:
        checks += [Check(n, SKIP, "no config") for n in ("credential", "provider", "backend")]
    else:
        cred = check_credential(cfg)
        checks.append(cred)
        checks.append(check_provider(cfg) if cred.status != FAIL
                      else Check("provider", SKIP, "credential check failed"))
        checks.append(check_backend(cfg))
    checks += [check_mainnet(), check_images()]
    return checks, cfg


def _minutes(seconds: int) -> str:
    return f"{seconds // 60} min"


def full_cost_text(backend: str, challenge: str) -> str:
    spec = CHALLENGES[challenge]
    if backend == "local":
        return ("no compute charge; the first run per challenge pulls a 13 GB image and does a "
                "warm-up build")
    if backend == "c3" and challenge == "knapsack":
        return "MEASURED 2026-09-15: £0.02 and 12 min 20 s for one knapsack job on C3"
    kind = "gpu" if spec.is_gpu else "cpu"
    if backend == "modal":
        seconds = FULL_ASSUMED_S[f"modal_{kind}"]
        if spec.is_gpu:
            costs = [seconds * GPU_USD_PER_SECOND[g] for g in MODAL_GPUS]
        else:
            costs = [seconds * (spec.cpu * CPU_USD_PER_CORE_SECOND
                                + spec.memory_mib / 1024 * MEM_USD_PER_GIB_SECOND)]
        unit = "$"
    elif backend == "c3":
        seconds = FULL_ASSUMED_S[f"c3_{kind}"]
        costs = [GBP_PER_HOUR[o] * seconds / 3600 for o in c3_hardware_options(spec)]
        unit = "£"
    else:
        raise ValueError(f"unknown backend {backend!r}")
    lo, hi = min(costs), max(costs)
    figure = f"{unit}{lo:.2f}" if f"{lo:.2f}" == f"{hi:.2f}" else f"{unit}{lo:.2f} to {unit}{hi:.2f}"
    text = (f"not measured for {challenge}; ESTIMATE (unverified): {figure} on {backend}, "
            f"assuming {_minutes(seconds)} of {'GPU' if spec.is_gpu else 'CPU'} time")
    if spec.is_gpu:
        text += "; GPU nonces can take many minutes each"
    return text


def full_check(cfg: Config, challenge: str) -> Check:
    """Compile and score mainnet's top algorithm for `challenge` on 2 training nonces of its
    first track, with no held-out set."""
    from talos.bench import EvalRequest, PendingJobStore
    from talos.diagnostics import first_error
    from talos.nonces import draw_nonce_sets, new_rand_hash

    spec = CHALLENGES[challenge]
    try:
        info = mainnet.fetch_challenge_info(challenge)
        top = mainnet.top_algorithm(challenge)
        if top is None:
            return Check("full", FAIL, f"no adopted compiled algorithm for {challenge} on "
                                       f"mainnet; try another --challenge")
        algo = top[0]
        files = mainnet.fetch_algorithm_files(challenge, algo)
        tr, ho = draw_nonce_sets(info.tracks[:1], new_rand_hash(), training_count=2,
                                 holdout_count=0)
        total = sum(ns.count for ns in tr)
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            local = None
            try:
                if cfg.backend == "local":
                    # stdout belongs to the text lines or, with --json, to the JSON alone
                    local_transport.prepare(challenge, log=_to_stderr)
                    local = cli.local_settings(cfg)
                bench = cli.make_bench(
                    cfg.backend, run_dir, PendingJobStore.memory(),
                    c3_api_key=resolve_c3_api_key(cfg) if cfg.backend == "c3" else None,
                    local=local)
                mark = bench.cost_mark()
                bench.select_hardware(challenge)
                r = bench.evaluate(EvalRequest(challenge, files, tr, ho, info.max_fuel, None,
                                               spec.beat))
                spend = bench.cost_usd_since(mark)
            finally:
                if cfg.backend == "local":
                    cli.remove_compile_containers(run_dir)
    except Exception as e:  # a diagnostic: report what broke rather than a traceback
        return Check("full", FAIL, f"{type(e).__name__}: {e}; run `talos doctor` without "
                                   f"--full to check the setup")
    k = sum(1 for x in r.training if x.ok)
    parts = [f"{algo} on {cfg.backend}",
             "compile ok" if r.compile.ok else f"compile failed: {first_error(r.compile.output)}",
             f"{k}/{total} nonces ok", f"compute spend (estimated) ${spend:.2f}"]
    status = OK if r.compile.ok and k > 0 else FAIL
    return Check("full", status, ", ".join(parts))


def _to_stderr(msg: str) -> None:
    print(msg, file=sys.stderr)


def _line(c: Check) -> str:
    if c.status == OK:
        return f"{c.name}: ok" + (f" ({c.detail})" if c.detail else "")
    return f"{c.name}: {c.status}: {c.detail}"


def run(root: Path, *, full: bool, challenge: str, yes: bool, json_out: bool,
        ask: Callable) -> int:
    checks, cfg = free_checks(root)
    if not json_out:
        for c in checks:
            print(_line(c))
    full_failed = False
    if full:
        if any(c.status == FAIL for c in checks) or cfg is None:
            fc = Check("full", SKIP, "a free check failed")
        elif json_out and not yes:
            fc = Check("full", SKIP, "needs --yes")
        else:
            print(f"--full compiles and scores mainnet's top {challenge} algorithm on "
                  f"{cfg.backend}, 2 nonces: {full_cost_text(cfg.backend, challenge)}",
                  file=sys.stderr if json_out else sys.stdout)
            if not yes and not confirm(ask, "Spend this?", False):
                fc = Check("full", SKIP, "declined")
            else:
                fc = full_check(cfg, challenge)
        checks.append(fc)
        full_failed = fc.status != OK
        if not json_out:
            print(_line(fc))
    rc = 1 if full_failed or any(c.status == FAIL for c in checks) else 0
    if json_out:
        print(json.dumps({"checks": [asdict(c) for c in checks], "ok": rc == 0}))
    return rc
