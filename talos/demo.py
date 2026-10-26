"""`talos demo`: a scripted run of the real loop, in-process, with no account, network or spend.

The baseline is `fn solve() { let k = 1; }` and training quality is `100 + k - 1` per nonce, so
k=1 is baseline parity. The script walks through the outcomes a real job can have, in order:

1. an edit that does not compile, then a compile fix that restores the baseline (no gain)
2. an edit that makes things worse (k=0): two iterations in a row with no improvement
3. an edit that wins on training (k=3) but loses on the held-out nonces: a false positive
4. an edit that wins on training and held-out (k=5): the job is won
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from talos.nonces import HOLDOUT_START

BROKEN = " BROKEN"
_K = re.compile(r"let k = (\d+);")

# One entry per edit index: the hypothesis shown for it, and the k the edit writes (None for
# the edit that breaks the build).
_STEPS = {
    1: ("Inline the value check",
        "A first edit that does not compile; the compile-fix round repairs it.", None),
    2: ("Lower k to 0",
        "A change that makes quality worse, so it is not kept.", 0),
    3: ("Raise k to 3",
        "Better on the training nonces, but it will not hold up on the held-out nonces.", 3),
    4: ("Raise k to 5",
        "Better on the training nonces and on the held-out nonces.", 5),
}

PACKAGE_FILES: dict[str, str] = {
    "README.md": "what the job found and how to read the rest of the package",
    "scores.md": "training and held-out scores of the best candidate against the baseline",
    "hypotheses.md": "every hypothesis the LLM proposed, with its outcome",
    "evidence_draft.md": "a draft of the evidence text for a TIG submission",
    "lineage.json": "the search tree: every scored candidate and its parent, as JSON",
    "lineage.md": "the search tree as a readable table",
    "diff_vs_baseline.patch": "the best candidate's changes against the mainnet baseline",
    "diff_vs_template.patch": "the best candidate's changes against the challenge template",
    "hyperparameters.json": "the per-track hyperparameters the job scored with",
    "report.html": "the whole run as one page (talos report writes it again)",
    "submission": "the code to submit, with a README template to fill out",
}


def _k(text: str) -> int:
    return int(_K.search(text).group(1))


class DemoScript:
    """The scripted LLM: `(system, user) -> str`, answering by prompt kind. It counts edit
    prompts, so the same run always produces the same sequence."""

    def __init__(self) -> None:
        self.edits = 0

    def __call__(self, system: str, user: str) -> str:
        if '"strategy_tag" (one of' in system:  # hypothesis prompt, for the next edit
            title, description, _ = _STEPS.get(self.edits + 1, ("Raise k by one",
                                                                "One more step up.", None))
            return json.dumps({"title": title, "description": description,
                               "design_family": "demo", "strategy_tag": "local_search"})
        if "You are fixing a Rust compile error" in system:
            k = _k(user)
            return (f"<<<<<<< SEARCH mod.rs\nlet k = {k};{BROKEN}\n=======\nlet k = {k};\n"
                    ">>>>>>> REPLACE\n")
        if "You are editing a Rust solver" in system:
            self.edits += 1
            k = _k(user)
            _, _, new_k = _STEPS.get(self.edits, ("", "", k + 1))
            new = f"let k = {k};{BROKEN}" if new_k is None else f"let k = {new_k};"
            return f"<<<<<<< SEARCH mod.rs\nlet k = {k};\n=======\n{new}\n>>>>>>> REPLACE\n"
        if "You distill one reusable lesson" in system:
            return "LESSON: a change has to hold up on nonces it was not tuned on."
        return "NONE"  # dead-code fix, edit repair: no usable edit, so the loop moves on


def demo_compile_ok(files: dict[str, str]) -> bool:
    return "BROKEN" not in files["mod.rs"]


def demo_scores(challenge: str, files: dict[str, str], ns) -> list[int]:
    k = _k(files["mod.rs"])
    if k == 3 and ns.start >= HOLDOUT_START:
        return [98 for _ in ns.nonces()]  # step 3 does worse on the held-out nonces
    return [100 + k - 1 for _ in ns.nonces()]


def describe_package(pkg: Path) -> list[str]:
    if not pkg.is_dir():
        return []
    return [f"  {p.name}  - {PACKAGE_FILES.get(p.name, '(no description)')}"
            for p in sorted(pkg.iterdir(), key=lambda p: p.name)]
