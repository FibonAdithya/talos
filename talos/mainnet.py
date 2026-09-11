"""Read-only mainnet and monorepo access. Every function takes its HTTP getter as a
parameter so tests never touch the network."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

from talos.challenges import MONOREPO_REF

MAINNET_API = "https://mainnet-api.tig.foundation"
GH_REPO = "tig-foundation/tig-monorepo"
GH_API = f"https://api.github.com/repos/{GH_REPO}"
GH_RAW = f"https://raw.githubusercontent.com/{GH_REPO}"
HTTP_TIMEOUT = 15
UA = "talos-tig"


class MainnetError(RuntimeError):
    pass


def _get(url: str, accept: str) -> bytes:
    req = urllib.request.Request(url, headers={"Accept": accept, "User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        raise MainnetError(f"HTTP {e.code} fetching {url}") from None
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise MainnetError(f"network error fetching {url}: {e}") from None


def _get_json(url: str):
    return json.loads(_get(url, "application/json"))


def _get_text(url: str) -> str:
    return _get(url, "text/plain").decode("utf-8")


@dataclass(frozen=True)
class ChallengeInfo:
    id: str
    name: str
    is_gpu: bool
    tracks: list[str]
    max_fuel: int


def _block_id(get_json) -> str:
    return get_json(f"{MAINNET_API}/get-block")["block"]["id"]


def fetch_challenge_info(name: str, get_json=_get_json) -> ChallengeInfo:
    block_id = _block_id(get_json)
    resp = get_json(f"{MAINNET_API}/get-challenges?block_id={block_id}")
    for c in resp["challenges"]:
        cfg = c.get("config") or {}
        if cfg.get("name") != name:
            continue
        tracks = sorted((cfg.get("active_tracks") or {}).keys())
        if not tracks:
            raise MainnetError(f"challenge {name} has no active tracks on mainnet")
        return ChallengeInfo(id=c["id"], name=name, is_gpu=cfg.get("type") == "gpu",
                             tracks=tracks, max_fuel=int(cfg["max_fuel_budget"]))
    raise MainnetError(f"challenge {name!r} not found on mainnet")


def top_algorithm(name: str, get_json=_get_json) -> tuple[str, int] | None:
    """(algorithm_name, adoption) of the highest-adoption compiled algorithm, or None."""
    block_id = _block_id(get_json)
    challenges = get_json(f"{MAINNET_API}/get-challenges?block_id={block_id}")
    algos = get_json(f"{MAINNET_API}/get-algorithms?block_id={block_id}")
    cid = next((c["id"] for c in challenges["challenges"]
                if (c.get("config") or {}).get("name") == name), None)
    if cid is None:
        raise MainnetError(f"challenge {name!r} not found on mainnet")
    compiled = {b["algorithm_id"]: bool((b.get("details") or {}).get("compile_success"))
                for b in algos.get("binarys", [])}
    best: tuple[str, int] | None = None
    for algo in algos.get("codes", []):
        details = algo.get("details") or {}
        if details.get("challenge_id") != cid or not compiled.get(algo["id"]):
            continue
        try:
            adoption = int((algo.get("block_data") or {}).get("adoption") or 0)
        except (TypeError, ValueError):
            adoption = 0
        algo_name = details.get("name")
        if adoption > 0 and algo_name and (best is None or adoption > best[1]):
            best = (algo_name, adoption)
    return best


def _walk(path: str, ref: str, get_json) -> list[str]:
    entries = get_json(f"{GH_API}/contents/{path}?ref={ref}")
    if isinstance(entries, dict):  # a single file, not a directory
        return [entries["path"]]
    out: list[str] = []
    for e in entries:
        if e["type"] == "dir":
            out.extend(_walk(e["path"], ref, get_json))
        elif e["type"] == "file":
            out.append(e["path"])
    return out


def fetch_algorithm_files(name: str, algorithm: str, get_text=_get_text,
                          get_json=_get_json) -> dict[str, str]:
    """{relative_path: contents}. A single-file algorithm comes back as {"mod.rs": ...}."""
    ref = f"{name}/{algorithm}"
    base_dir = f"tig-algorithms/src/{name}/{algorithm}"
    try:
        paths = _walk(base_dir, ref, get_json)
    except MainnetError:
        single = f"tig-algorithms/src/{name}/{algorithm}.rs"
        return {"mod.rs": get_text(f"{GH_RAW}/{ref}/{single}")}
    files: dict[str, str] = {}
    for p in paths:
        rel = p[len(base_dir) + 1:] if p.startswith(base_dir + "/") else p.rsplit("/", 1)[-1]
        files[rel] = get_text(f"{GH_RAW}/{ref}/{p}")
    if not files:
        raise MainnetError(f"no files found for {name}/{algorithm}")
    return files


def fetch_template(name: str, get_text=_get_text) -> str:
    return get_text(f"{GH_RAW}/{MONOREPO_REF}/tig-algorithms/src/{name}/template.rs")
