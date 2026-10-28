"""The root files of a `--start template` job: TIG's empty template as mod.rs, plus, on a GPU
challenge, its template.cu as algorithm.cu (the build needs at least one .cu file in the
algorithm directory, `talos/inside.py`). Loop, package and report all build the root here."""
from __future__ import annotations

from pathlib import Path

from talos.challenges import CHALLENGES

# Any *.cu name compiles (the build globs); a name other than the challenge's own kernels.cu
# keeps the two apart in prompts. Stored jobs read iteration files by these names, so a rename
# would break them.
ROOT_CU = "algorithm.cu"


def template_root(challenge: str, template_rs: str, template_cu: str | None) -> dict[str, str]:
    if not CHALLENGES[challenge].is_gpu:
        return {"mod.rs": template_rs}
    if template_cu is None:
        raise ValueError(f"{challenge} is a GPU challenge: its template root needs template.cu")
    return {"mod.rs": template_rs, ROOT_CU: template_cu}


def root_names(challenge: str) -> list[str]:
    return sorted(["mod.rs", ROOT_CU] if CHALLENGES[challenge].is_gpu else ["mod.rs"])


def read_template_root(challenge_dir: Path, challenge: str) -> dict[str, str] | None:
    """The root from runs/<job>/challenge/, or None when a file it needs is missing."""
    challenge_dir = Path(challenge_dir)
    needed = {"mod.rs": "template.rs"}
    if CHALLENGES[challenge].is_gpu:
        needed[ROOT_CU] = "template.cu"
    root = {}
    for name, stored in needed.items():
        path = challenge_dir / stored
        if not path.is_file():
            return None
        root[name] = path.read_bytes().decode("utf-8", errors="replace")
    return root
