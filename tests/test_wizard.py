import pytest

from talos.config import ConfigError
from talos.wizard import ask_choice, confirm


def scripted(answers):
    """An ask that returns the queued answers and records each prompt. A blank answer is
    returned as "", as the wizard tests' raw ask does."""
    it = iter(answers)
    prompts = []

    def ask(prompt, default=None, secret=False):
        prompts.append(prompt)
        try:
            v = next(it)
        except StopIteration:
            raise AssertionError(f"unexpected prompt: {prompt}")
        if isinstance(v, BaseException):
            raise v
        return v
    ask.prompts = prompts
    return ask


def test_number_name_and_blank_select():
    # mutation: the default marker dropped, or matching made case-sensitive
    options = [("modal", "a"), ("c3", "b"), ("local", "c")]
    ask = scripted(["2", "LOCAL", ""])
    assert ask_choice(ask, "Backend", options, "modal") == "c3"
    assert ask_choice(ask, "Backend", options, "modal") == "local"
    assert ask_choice(ask, "Backend", options, "modal") == "modal"
    assert "  1) modal  - a (default)" in ask.prompts[0]
    assert ask.prompts[0].endswith("Choose")
    assert ask.prompts[0].startswith("Backend:\n")


def test_empty_explanation_has_no_dangling_dash():
    # mutation: the " - " always appended, leaving "  2) n=1  - " for an option with no text
    options = [("n=1", ""), ("n=2", "")]
    ask = scripted(["1"])
    ask_choice(ask, "Nonces", options, "n=2")
    lines = ask.prompts[0].split("\n")
    assert lines[1] == "  1) n=1"
    assert lines[2] == "  2) n=2 (default)"


def test_typo_reprompts_then_accepts(capsys):
    # mutation: the first answer accepted unchecked
    ask = scripted(["agentc", "agentic"])
    options = [("single-shot", ""), ("agentic", "")]
    assert ask_choice(ask, "Mode", options, "single-shot") == "agentic"
    assert len(ask.prompts) == 2
    assert "not one of the choices: 'agentc'" in capsys.readouterr().err


def test_out_of_range_number_is_invalid():
    # mutation: options[int(raw)-1] without a bounds check ("-1" would pick the last option)
    options = [("a", ""), ("b", ""), ("c", "")]
    with pytest.raises(ConfigError):
        ask_choice(scripted(["9", "0", "-1"]), "T", options, "a")


def test_three_typos_raise():
    # mutation: an unbounded loop
    ask = scripted(["x", "y", "z", "a"])
    with pytest.raises(ConfigError, match="no valid choice after 3 tries"):
        ask_choice(ask, "T", [("a", "")], "a")
    assert len(ask.prompts) == 3


def test_eof_at_menu_raises_config_error():
    # mutation: letting EOFError escape, or treating it as the default
    with pytest.raises(ConfigError, match="input closed"):
        ask_choice(scripted([EOFError()]), "T", [("a", "")], "a")


def test_track_value_that_looks_like_a_number():
    # mutation: values matched before indexes
    options = [("all", ""), ("n=1", ""), ("2", "")]
    assert ask_choice(scripted(["n=1"]), "T", options, "all") == "n=1"
    assert ask_choice(scripted(["3"]), "T", options, "all") == "2"
    # "2" is both the index of "n=1" and the name of the third option: the index wins
    assert ask_choice(scripted(["2"]), "T", options, "all") == "n=1"


def test_confirm():
    # mutation: EOF falls back to the default
    assert confirm(scripted([""]), "Go?", True) is True
    assert confirm(scripted(["N"]), "Go?", True) is False
    ask = scripted(["maybe", "yes"])
    assert confirm(ask, "Go?", False) is True
    assert len(ask.prompts) == 2
    assert confirm(scripted([EOFError()]), "Go?", True) is False
    ask = scripted(["a", "b", "c", "y"])
    assert confirm(ask, "Go?", True) is False
    assert len(ask.prompts) == 3
