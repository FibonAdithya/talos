"""Fixed-choice prompts for the setup and run wizards: numbered menus that re-prompt on a typo."""
from __future__ import annotations

import sys
from typing import Callable, Sequence

from talos.config import ConfigError

CHOICE_TRIES = 3


def menu_text(title: str, options: Sequence[tuple[str, str]], default: str) -> str:
    width = max(len(value) for value, _ in options)
    lines = [f"{title}:"]
    for i, (value, explanation) in enumerate(options, 1):
        line = f"  {i}) {value:<{width}}  - {explanation}" if explanation else f"  {i}) {value}"
        if value == default:
            line += " (default)"
        lines.append(line)
    lines.append("Choose")
    return "\n".join(lines)


def ask_choice(ask: Callable, title: str, options: Sequence[tuple[str, str]], default: str,
               tries: int = CHOICE_TRIES) -> str:
    """One of the option values. A number is an index first; a name matches case-insensitively;
    a blank answer is the default."""
    values = [value for value, _ in options]
    default_index = values.index(default) + 1
    text = menu_text(title, options, default)
    for _ in range(tries):
        try:
            raw = ask(text, str(default_index))
        except EOFError:
            raise ConfigError(f"{title}: no answer (input closed)") from None
        answer = raw.strip()
        if answer == "":
            return default
        if answer.isascii() and answer.isdigit() and 1 <= int(answer) <= len(values):
            return values[int(answer) - 1]
        for value in values:
            if value.lower() == answer.lower():
                return value
        print(f"not one of the choices: {raw!r}", file=sys.stderr)
    raise ConfigError(f"{title}: no valid choice after {tries} tries")


def confirm(ask: Callable, prompt: str, default: bool, tries: int = CHOICE_TRIES) -> bool:
    """A yes/no question. Closed input and unreadable answers mean no: nothing starts or
    spends on a guess."""
    for _ in range(tries):
        try:
            raw = ask(prompt, "y" if default else "n")
        except EOFError:
            return False
        answer = raw.strip().lower()
        if answer == "":
            return default
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("please answer y or n", file=sys.stderr)
    return False
