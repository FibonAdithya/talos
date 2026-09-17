"""How Talos names the external CLIs it starts."""
import os
import shutil  # noqa: F401

_NT = os.name == "nt"


def argv0(name: str) -> str:
    return name
