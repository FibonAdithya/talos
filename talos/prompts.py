"""Prompt builders and parsers. Pure functions over PromptContext; no I/O beyond reading the
packaged Rust rules once."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources

from talos.diagnostics import relevant

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


@lru_cache(maxsize=1)
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
    track: str | None = None
    guard_tracks: list[str] = field(default_factory=list)
    hyperparameters: dict[str, dict | None] | None = None


def _files_block(files: dict[str, str]) -> str:
    return "\n\n".join(f"--- {name} ---\n{text}" for name, text in sorted(files.items()))


def focus_sentence(ctx: PromptContext) -> str:
    """The target sentence for a focused job; empty when the job optimises every track."""
    if ctx.track is None:
        return ""
    guards = ", ".join(ctx.guard_tracks) or "none"
    return (f"Optimise for track \"{ctx.track}\" only. The other active tracks ({guards}) are "
            f"re-scored as a regression guard when a candidate wins, and none of them may get "
            f"worse: confine changes to the code path that serves \"{ctx.track}\".")


def hyperparameters_block(ctx: PromptContext) -> str:
    """What every run passes to solve_challenge, so edits keep those keys readable. Empty when the
    job runs without hyperparameters. A focused job shows only its own track."""
    if ctx.hyperparameters is None:
        return ""
    shown = [ctx.track] if ctx.track else sorted(ctx.hyperparameters)
    lines = []
    for track in shown:
        hp = ctx.hyperparameters.get(track)
        value = ("none (solve_challenge receives None)" if hp is None
                 else json.dumps(hp, sort_keys=True, separators=(",", ":")))
        lines.append(f"track {track}: {value}")
    text = ("Hyperparameters: every run of this code, the baseline and your candidate alike, "
            "passes these values to solve_challenge as `hyperparameters`, per track. They came "
            "from the best mainnet benchmark of this algorithm. Keep every key readable: you may "
            "add keys, but do not rename or remove existing ones.\n" + "\n".join(lines))
    if ctx.track and len(ctx.hyperparameters) > 1:
        text += "\nThe guard tracks run with their own values too."
    return text


def describe_attempt(h: dict) -> str:
    """One line per failed attempt, with what the run measured: a title plus the outcome told
    the model nothing about a +0.14% track, a -0.37% track and a 14x runtime."""
    line = f"- {h.get('title', '')} [{h.get('outcome', '')}]"
    if "mean_rel_delta" in h:
        line += (f" mean {h['mean_rel_delta']:+.2%}, worst track {h.get('worst_track', '?')} "
                 f"{h.get('worst_rel_delta', 0.0):+.2%}")
        if "runtime_ratio" in h:
            line += f", runtime {h['runtime_ratio']:.1f}x baseline"
    if h.get("error"):
        line += f": {str(h['error'])[:200]}"
    return line


def hypothesis_prompts(ctx: PromptContext) -> tuple[str, str]:
    scope = f"on track \"{ctx.track}\"" if ctx.track else "across every active track"
    focus = (focus_sentence(ctx) + "\n\n") if ctx.track else ""
    system = (
        f"You are a research engineer improving a Rust solver for the TIG challenge "
        f"\"{ctx.challenge}\". The goal is to beat the current mainnet state of the art on "
        f"TIG's own benchmark: higher verifier quality per nonce under a fixed fuel budget, "
        f"{scope}.\n\n{focus}"
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
        lines = "\n".join(describe_attempt(h) for h in ctx.failed_hypotheses)
        parts.append("Already tried against this exact code and did NOT help; do not repeat:\n"
                     + lines)
    if ctx.forced_tag:
        parts.append(f"You have stagnated. Your strategy_tag MUST be \"{ctx.forced_tag}\" "
                     f"this time; change the approach, not the constants.")
    hp = hyperparameters_block(ctx)
    if hp:
        parts.append(hp)
    parts.append("Current algorithm source:\n" + _files_block(ctx.files))
    return system, "\n\n".join(parts)


def edit_prompts(ctx: PromptContext, hypothesis: dict) -> tuple[str, str]:
    system = (
        f"You are editing a Rust solver for the TIG challenge \"{ctx.challenge}\".\n\n"
        f"{SEARCH_REPLACE_FORMAT}\n\n{_rust_rules()}"
    )
    focus = focus_sentence(ctx)
    hp = hyperparameters_block(ctx)
    user = (f"Implement this hypothesis:\nTitle: {hypothesis['title']}\n"
            f"Description: {hypothesis['description']}\n\n"
            + (focus + "\n\n" if focus else "")
            + (hp + "\n\n" if hp else "")
            + f"Current algorithm source files:\n{_files_block(ctx.files)}")
    return system, user


def compile_fix_prompts(ctx: PromptContext, files: dict[str, str],
                        compiler_output: str) -> tuple[str, str]:
    system = (f"You are fixing a Rust compile error in a TIG \"{ctx.challenge}\" solver.\n\n"
              f"{SEARCH_REPLACE_FORMAT}\n\n{_rust_rules()}")
    # 12000 holds the eight errors of run 20260916-095103 iteration 5; 6000 showed the last few.
    user = (f"The build failed with:\n```\n{relevant(compiler_output)[-12000:]}\n```\n\n"
            f"{_file_names_line(files)}\n\n"
            f"Current files:\n{_files_block(files)}\n\nEmit edit blocks that fix the build "
            f"without abandoning the intended change.")
    return system, user


def dead_code_fix_prompts(ctx: PromptContext, files: dict[str, str],
                          names: list[str]) -> tuple[str, str]:
    """The build succeeded but the functions the edit added are never called, so the change is
    not on the solve path. Same contract as compile_fix_prompts."""
    system = (f"You are completing an unfinished edit to a Rust solver for the TIG "
              f"\"{ctx.challenge}\" solver.\n\n{SEARCH_REPLACE_FORMAT}\n\n{_rust_rules()}")
    listed = "\n".join(f"- {n}" for n in names)
    user = (f"The build succeeded, but these functions the edit added are never called, so the "
            f"change cannot affect the score:\n{listed}\n\n{_file_names_line(files)}\n\n"
            f"Current files:\n{_files_block(files)}\n\nEmit edit blocks that call them from "
            f"the solve path as the hypothesis intended (or remove them if the hypothesis is "
            f"already implemented without them).")
    return system, user


def _file_names_line(files: dict[str, str]) -> str:
    return ("The only files you may edit, and the names to use in SEARCH headers: "
            + ", ".join(sorted(files)) + ". Other algorithms in the build output are not yours.")


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
    lines = "\n".join(f"{describe_attempt(h)}\n  {h.get('description', '')}" for h in failed)
    user = (f"Challenge: {ctx.challenge}. Direction: {ctx.direction}\n\nFailed attempts:\n"
            f"{lines}\n\nWhat general lesson, independent of these exact constants, should "
            f"guide the next attempts?")
    return system, user


def parse_hypothesis(text: str) -> dict:
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            d, _ = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            continue
        if isinstance(d, dict) and "title" in d and "description" in d:
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
