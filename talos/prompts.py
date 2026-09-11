"""Prompt builders and parsers. Pure functions over PromptContext; no I/O beyond reading the
packaged Rust rules once."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from importlib import resources

STRATEGY_TAGS = ["construction", "local_search", "metaheuristic", "constraint_relaxation",
                 "decomposition", "hybrid", "data_structure", "parameter_tuning"]

SEARCH_REPLACE_FORMAT = """\
Respond ONLY with one or more edit blocks in exactly this format:

<<<<<<< SEARCH path/to/file.rs
<lines to find, copied exactly>
=======
<replacement lines>
>>>>>>> REPLACE

Rules: the SEARCH text must match the current file exactly once; keep each block small;
emit several blocks for several places; never emit whole-file rewrites; never touch files
other than the algorithm files shown."""


def _rust_rules() -> str:
    return resources.files("talos.data").joinpath("rust_rules.md").read_text()


@dataclass
class PromptContext:
    challenge: str
    template_rs: str
    direction: str
    tacit: str
    files: dict[str, str]
    baseline_name: str
    best_delta: float
    failed_hypotheses: list[dict] = field(default_factory=list)
    forced_tag: str | None = None
    is_gpu: bool = False


def _files_block(files: dict[str, str]) -> str:
    return "\n\n".join(f"--- {name} ---\n{text}" for name, text in sorted(files.items()))


def hypothesis_prompts(ctx: PromptContext) -> tuple[str, str]:
    system = (
        f"You are a research engineer improving a Rust solver for the TIG challenge "
        f"\"{ctx.challenge}\". The goal is to beat the current mainnet state of the art on "
        f"TIG's own benchmark: higher verifier quality per nonce under a fixed fuel budget, "
        f"across every active track.\n\n"
        f"The solver must keep this contract (template.rs):\n```rust\n{ctx.template_rs}\n```\n\n"
        f"Propose ONE specific change. Reply with a JSON object with keys \"title\" "
        f"(short), \"description\" (what to change and why it should raise quality), and "
        f"\"strategy_tag\" (one of: {', '.join(STRATEGY_TAGS)}). No other text."
    )
    parts = [f"Baseline to beat: mainnet algorithm \"{ctx.baseline_name}\". "
             f"Your current best is {ctx.best_delta:+.3%} relative to it on the training nonces.",
             f"Direction from the user:\n{ctx.direction}"]
    if ctx.tacit.strip():
        parts.append(f"Tacit knowledge (lessons so far):\n{ctx.tacit}")
    if ctx.failed_hypotheses:
        lines = "\n".join(f"- {h.get('title', '')} [{h.get('outcome', '')}]"
                          for h in ctx.failed_hypotheses)
        parts.append("Already tried against this exact code and did NOT help; do not repeat:\n"
                     + lines)
    if ctx.forced_tag:
        parts.append(f"You have stagnated. Your strategy_tag MUST be \"{ctx.forced_tag}\" "
                     f"this time; change the approach, not the constants.")
    parts.append("Current algorithm source:\n" + _files_block(ctx.files))
    return system, "\n\n".join(parts)


def edit_prompts(ctx: PromptContext, hypothesis: dict) -> tuple[str, str]:
    system = (
        f"You are editing a Rust solver for the TIG challenge \"{ctx.challenge}\".\n\n"
        f"{SEARCH_REPLACE_FORMAT}\n\n{_rust_rules()}"
    )
    user = (f"Implement this hypothesis:\nTitle: {hypothesis['title']}\n"
            f"Description: {hypothesis['description']}\n\n"
            f"Current algorithm source files:\n{_files_block(ctx.files)}")
    return system, user


def compile_fix_prompts(ctx: PromptContext, files: dict[str, str],
                        compiler_output: str) -> tuple[str, str]:
    system = (f"You are fixing a Rust compile error in a TIG \"{ctx.challenge}\" solver.\n\n"
              f"{SEARCH_REPLACE_FORMAT}\n\n{_rust_rules()}")
    user = (f"The build failed with:\n```\n{compiler_output[-6000:]}\n```\n\n"
            f"Current files:\n{_files_block(files)}\n\nEmit edit blocks that fix the build "
            f"without abandoning the intended change.")
    return system, user


def edit_repair_prompts(ctx: PromptContext, files: dict[str, str],
                        misses_text: str) -> tuple[str, str]:
    system = (f"You are repairing edit blocks for a TIG \"{ctx.challenge}\" solver.\n\n"
              f"{SEARCH_REPLACE_FORMAT}")
    user = (f"These blocks did not match the files exactly once:\n{misses_text}\n\n"
            f"Current files:\n{_files_block(files)}\n\nRe-emit only the failed blocks with "
            f"SEARCH text copied exactly from the files above.")
    return system, user


def distill_prompts(ctx: PromptContext, failed: list[dict]) -> tuple[str, str]:
    system = ("You distill one reusable lesson from failed optimisation attempts. Reply with "
              "exactly one line starting with \"LESSON: \" or the single word NONE.")
    lines = "\n".join(f"- {h.get('title', '')}: {h.get('description', '')} "
                      f"[{h.get('outcome', '')}]" for h in failed)
    user = (f"Challenge: {ctx.challenge}. Direction: {ctx.direction}\n\nFailed attempts:\n"
            f"{lines}\n\nWhat general lesson, independent of these exact constants, should "
            f"guide the next attempts?")
    return system, user


_JSON_RE = re.compile(r"\{.*?\}", re.S)


def parse_hypothesis(text: str) -> dict:
    for m in _JSON_RE.finditer(text):
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if "title" in d and "description" in d:
            tag = d.get("strategy_tag")
            return {"title": str(d["title"]), "description": str(d["description"]),
                    "strategy_tag": tag if tag in STRATEGY_TAGS else "hybrid"}
    raise ValueError("no hypothesis JSON object found in response")


def parse_distillation(text: str) -> str | None:
    for line in text.splitlines():
        if line.strip().startswith("LESSON:"):
            lesson = line.split("LESSON:", 1)[1].strip()
            return lesson or None
    return None
