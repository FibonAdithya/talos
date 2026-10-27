"""`talos guide` and the drift check that ties the agent guide to the real parser (#71)."""

import re
from pathlib import Path

import talos
from talos import cli

GUIDE = Path(talos.__file__).parent / "data" / "agent_guide.md"

_FENCE = re.compile(r"^```[^\n]*\n(.*?)^```", re.M | re.S)
_SPAN = re.compile(r"`([^`\n]+)`")
# `talos`, any global flags (`--workspace DIR`, `--workspace=DIR`), then the subcommand.
_COMMAND = re.compile(r"(?<![\w/.-])talos ((?:--[a-z][a-z-]*(?:=\S+| [^\s-]\S*)? )*)"
                      r"([a-z][a-z-]*)")
# Where the talos command's own arguments end: a pipe, a command separator or a redirect.
_END = re.compile(r"\||;|&&|>|(?<!&)&(?!&)")


def _code_lines(text):
    """Each line of code in `text`: fenced lines with `\\` continuations joined, then the
    inline spans of the prose outside the fences."""
    lines = []
    for block in _FENCE.findall(text):
        lines.extend(re.sub(r"\\\n\s*", " ", block).splitlines())
    prose = _FENCE.sub("", text)
    return lines, _SPAN.findall(prose)


def _subparsers(parser):
    return next(a for a in parser._actions if a.__class__.__name__ == "_SubParsersAction").choices


def unknown_references(text, parser):
    subs = _subparsers(parser)
    every_flag = set(parser._option_string_actions).union(
        *(p._option_string_actions for p in subs.values()))
    unknown = []
    fenced, spans = _code_lines(text)
    for line in fenced + spans:
        commands = list(_COMMAND.finditer(line))
        for i, m in enumerate(commands):
            for flag in re.findall(r"--[a-z][a-z-]*", m.group(1)):
                if flag not in parser._option_string_actions:
                    unknown.append(f"talos {flag}")
            word = m.group(2)
            if word not in subs:
                unknown.append(f"talos {word}")
                continue
            rest = line[m.end():commands[i + 1].start() if i + 1 < len(commands) else None]
            end = _END.search(rest)
            rest = rest[:end.start()] if end else rest
            for flag in re.findall(r"(?<!\S)--[a-z][a-z-]*", rest):
                if flag not in subs[word]._option_string_actions:
                    unknown.append(f"talos {word} {flag}")
    # A span of bare flags with no `talos` before them, such as `--budget-iterations 5`: each
    # must be a global option or an option of some subcommand. A wildcard such as `--add-*` names a family.
    for span in spans:
        if span.startswith("--") and "*" not in span:
            for flag in re.findall(r"(?<!\S)--[a-z][a-z-]*", span):
                if flag not in every_flag:
                    unknown.append(flag)
    return unknown


def test_checker_flags_unknown_subcommands_and_flags():
    text = ("Run `talos shw` first, or pass `--add-hourz 2`.\n"
            "Propose `--budget-iterations 5 --budget-hourz 1`.\n"
            "```\ntalos run --budget-dollars 3 --yes\n```\n")
    assert sorted(unknown_references(text, cli.build_parser())) == [
        "--add-hourz", "--budget-hourz", "talos run --budget-dollars", "talos shw"]


def test_checker_follows_line_continuations():
    text = "```\ntalos run --challenge knapsack \\\n  --budget-dollars 3\n```\n"
    assert unknown_references(text, cli.build_parser()) == ["talos run --budget-dollars"]


def test_checker_reads_global_flags_before_the_command():
    # mutation: a pattern that needs the command right after `talos` skips these lines
    text = ("Run `talos --workspace . run --budget-dollars 1`, then\n"
            "`talos --workspaces . setup` and `talos --workspace=~/w status --json`.\n")
    assert sorted(unknown_references(text, cli.build_parser())) == [
        "talos --workspaces", "talos run --budget-dollars"]


def test_checker_ignores_flags_after_a_pipe_and_wildcards():
    text = "Use `talos status --json | jq --raw` and never an `--add-*` flag unasked.\n"
    assert unknown_references(text, cli.build_parser()) == []


def test_guide_names_only_real_commands_and_flags():
    assert unknown_references(GUIDE.read_text(encoding="utf-8"), cli.build_parser()) == []


def test_guide_keeps_its_spending_and_stop_rules():
    # A content guard only: it catches a deleted section, not wrong advice inside one.
    text = GUIDE.read_text(encoding="utf-8")
    for needle in ("talos run", "talos doctor --full", "kill -INT"):
        assert needle in text, needle


def test_talos_guide_prints_the_packaged_file(capsys):
    assert cli.main(["guide"]) == 0
    assert capsys.readouterr().out == GUIDE.read_text(encoding="utf-8")
