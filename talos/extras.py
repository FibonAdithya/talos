"""Optional dependencies. `modal` and `anthropic` are extras, so a missing one has to become a
message that names the install command before anything is written or spent."""
from __future__ import annotations

import importlib.util

from talos.config import ConfigError

EXTRAS = {"modal": "modal", "anthropic": "anthropic"}  # extra name -> import name


def missing(extra: str) -> bool:
    return importlib.util.find_spec(EXTRAS[extra]) is None


def install_hint(extra: str) -> str:
    # MEASURED 2026-10-27: on a tool installed without the extra, this form adds it with no
    # --force; installing a different extra afterwards replaces the earlier one. So the command
    # names the missing extra first, then every other extra that is installed now. pip keeps
    # what is installed, so the clone form names only the missing one.
    names = [extra] + [e for e in EXTRAS if e != extra and not missing(e)]
    return (f"uv tool install --python 3.12 'talos-tig[{','.join(names)}]' "
            f"(in a clone: pip install -e '.[{extra}]')")


def require(extra: str, why: str) -> None:
    if missing(extra):
        raise ConfigError(f"{why} needs the {extra} extra: {install_hint(extra)}")
