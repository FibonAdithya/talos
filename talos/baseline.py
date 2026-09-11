"""Resolve the mainnet top algorithm, compile and score it once, cache the result."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from talos import mainnet as _mainnet
from talos.challenges import MONOREPO_REF
from talos.state import BaselineRecord, _atomic_write
from talos.types import NonceResult, NonceSet


class BaselineError(RuntimeError):
    pass


def cache_key(challenge: str, monorepo_ref: str, name: str, training: list[NonceSet],
              holdout: list[NonceSet], fuel: int, hardware_class: str) -> str:
    h = hashlib.sha256()
    payload = {"challenge": challenge, "ref": monorepo_ref, "name": name, "fuel": fuel,
               "hw": hardware_class,
               "training": [(n.track, n.rand_hash, n.start, n.count) for n in training],
               "holdout": [(n.track, n.rand_hash, n.start, n.count) for n in holdout]}
    h.update(json.dumps(payload, sort_keys=True).encode())
    return h.hexdigest()[:24]


def resolve_baseline(challenge: str, training: list[NonceSet], holdout: list[NonceSet],
                     fuel: int, bench, cache_dir: Path, hardware_class: str,
                     mainnet=_mainnet, log=lambda msg: None) -> tuple[BaselineRecord, str]:
    top = mainnet.top_algorithm(challenge)
    if top is None:
        raise BaselineError(f"no adopted, compiled algorithm found on mainnet for {challenge}")
    name, adoption = top
    template = mainnet.fetch_template(challenge)
    key = cache_key(challenge, MONOREPO_REF, name, training, holdout, fuel, hardware_class)
    cache_file = Path(cache_dir) / challenge / f"{key}.json"
    if cache_file.exists():
        try:
            rec = BaselineRecord.from_dict(json.loads(cache_file.read_text()))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            log(f"baseline {name}: cache file {cache_file} is corrupt; re-measuring")
        else:
            log(f"baseline {name}: cache hit {key}")
            return rec, template
    files = mainnet.fetch_algorithm_files(challenge, name)
    log(f"baseline {name} (adoption {adoption}): compiling {len(files)} file(s)")
    c = bench.compile(challenge, files)
    if not c.ok:
        raise BaselineError(f"baseline {name} failed to compile; likely dev-image drift at "
                            f"{MONOREPO_REF}.\n{c.output[-4000:]}")
    log("baseline: scoring training nonces")
    tr: list[NonceResult] = bench.score(challenge, c.artifact_id, training, fuel)
    log("baseline: scoring held-out nonces")
    ho: list[NonceResult] = bench.score(challenge, c.artifact_id, holdout, fuel)
    rec = BaselineRecord(name=name, adoption=adoption, artifact_id=c.artifact_id, files=files,
                         training=tr, holdout=ho)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(cache_file, json.dumps(rec.to_dict(), indent=1))
    return rec, template
