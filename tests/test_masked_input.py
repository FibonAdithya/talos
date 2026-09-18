import faulthandler
import io
import os
import select
import threading

import pytest

from talos import cli, masked_input
from talos.masked_input import ask_secret, read_masked


def _feed(text: str):
    """A getch that hands out `text` one character at a time, then "" as a closed stdin does."""
    chars = iter(text)
    return lambda: next(chars, "")


class _Tty(io.StringIO):
    def isatty(self):
        return True


def _settings(fd):
    """The terminal's settings without PENDIN. macOS sets that bit itself when canonical mode
    comes back with input still queued; it is driver state, not something a caller restores."""
    import termios

    attrs = termios.tcgetattr(fd)
    attrs[3] &= ~termios.PENDIN
    return attrs


@pytest.fixture
def pty_pair():
    """A real terminal pair, with the master drained the way a terminal emulator drains it. On
    macOS the echo of typed input queues as the slave's output, and restoring the mode with
    TCSADRAIN waits for that queue to empty; with nobody reading the master it waits for ever.
    A read that finds no input blocks too, and a signal handler cannot run while the main thread
    is inside the call, so the guard is faulthandler's watchdog thread: it prints where the
    thread is stuck and exits, instead of leaving a hung CI job."""
    pytest.importorskip("termios")
    import pty

    master, slave = pty.openpty()
    stop = threading.Event()

    def drain():
        while not stop.is_set():
            readable, _, _ = select.select([master], [], [], 0.05)
            if readable:
                try:
                    os.read(master, 4096)
                except OSError:
                    return

    drainer = threading.Thread(target=drain, daemon=True)
    drainer.start()
    faulthandler.dump_traceback_later(20, exit=True)
    try:
        yield master, slave
    finally:
        faulthandler.cancel_dump_traceback_later()
        stop.set()
        drainer.join()
        os.close(master)
        os.close(slave)


def test_a_pasted_secret_shows_one_star_per_character_and_is_returned_whole():
    out = io.StringIO()
    assert read_masked("API key: ", _feed("sk-ant-12345\n"), out) == "sk-ant-12345"
    assert out.getvalue() == "API key: " + "*" * 12 + "\n"


def test_the_prompt_and_every_star_are_flushed_before_the_next_key_is_waited_for():
    # stdout on a terminal is line buffered: an unflushed star would only appear at Enter
    class LineBuffered(io.StringIO):
        visible = ""

        def flush(self):
            self.visible = self.getvalue()

    out, seen, chars = LineBuffered(), [], iter("ab\n")

    def getch():
        seen.append(out.visible)
        return next(chars)

    assert read_masked("k: ", getch, out) == "ab"
    assert seen == ["k: ", "k: *", "k: **"]


def test_windows_enter_is_a_carriage_return():
    out = io.StringIO()
    assert read_masked("k: ", _feed("abc\rzzz"), out) == "abc"
    assert out.getvalue() == "k: ***\n"


@pytest.mark.parametrize("key", ["\x7f", "\b"])
def test_backspace_removes_one_character_and_erases_one_star(key):
    out = io.StringIO()
    assert read_masked("k: ", _feed(f"abcd{key}{key}e\n"), out) == "abe"
    assert out.getvalue() == "k: ****" + "\b \b" * 2 + "*\n"


def test_backspace_on_an_empty_line_does_not_eat_the_prompt():
    out = io.StringIO()
    assert read_masked("k: ", _feed("\x7f\x7fab\n"), out) == "ab"
    assert out.getvalue() == "k: **\n"


def test_escape_sequences_and_control_characters_are_neither_kept_nor_starred():
    # a bracketed paste wraps the text in ESC[200~ … ESC[201~; an arrow key is ESC[A or ESC O A
    out = io.StringIO()
    typed = "\x1b[200~se\x1b[Acr\x1bOAe\tt\x00\x1b[201~\n"
    assert read_masked("k: ", _feed(typed), out) == "secret"
    assert out.getvalue() == "k: ******\n"


def test_back_to_back_escape_sequences_are_all_dropped():
    out = io.StringIO()
    assert read_masked("k: ", _feed("\x1b[200~\x1b[A\x1bOBab\x1b[201~\x1b[C\n"), out) == "ab"
    assert out.getvalue() == "k: **\n"


def test_a_character_typed_right_after_a_bare_escape_is_kept():
    out = io.StringIO()
    assert read_masked("k: ", _feed("a\x1bb\n"), out) == "ab"
    assert out.getvalue() == "k: **\n"


def test_the_secret_is_never_written_to_the_terminal():
    out = io.StringIO()
    read_masked("k: ", _feed("hunter2\n"), out)
    shown = out.getvalue()
    assert not set("hunter2") & set(shown.removeprefix("k: "))


def test_ctrl_c_raises_keyboard_interrupt_and_ends_the_line():
    out = io.StringIO()
    with pytest.raises(KeyboardInterrupt):
        read_masked("k: ", _feed("ab\x03cd\n"), out)
    assert out.getvalue() == "k: **\n"


def test_ctrl_d_on_an_empty_line_is_eof_but_is_ignored_after_input():
    with pytest.raises(EOFError):
        read_masked("k: ", _feed("\x04abc\n"), io.StringIO())
    assert read_masked("k: ", _feed("ab\x04c\n"), io.StringIO()) == "abc"


def test_a_closed_stdin_is_eof_rather_than_an_endless_loop():
    with pytest.raises(EOFError):
        read_masked("k: ", _feed("abc"), io.StringIO())


@pytest.mark.parametrize("stdin_tty, out_tty", [(False, True), (True, False), (False, False)])
def test_without_a_terminal_on_both_ends_ask_secret_falls_back_to_getpass(monkeypatch, stdin_tty,
                                                                          out_tty):
    seen = []
    monkeypatch.setattr(masked_input.getpass, "getpass", lambda p: seen.append(p) or "piped-key")
    stdin = (_Tty if stdin_tty else io.StringIO)("typed\n")
    out = (_Tty if out_tty else io.StringIO)()
    assert ask_secret("API key: ", stdin=stdin, out=out) == "piped-key"
    assert seen == ["API key: "]
    assert out.getvalue() == ""


def test_on_a_posix_terminal_echo_is_off_while_reading_and_restored_after(monkeypatch, pty_pair):
    import termios

    monkeypatch.setattr(masked_input.getpass, "getpass",
                        lambda p: pytest.fail("a terminal must not fall back to getpass"))
    master, slave = pty_pair
    before = _settings(slave)
    assert before[3] & termios.ECHO and before[3] & termios.ICANON
    lflags = []

    class Out(_Tty):
        def write(self, s):
            lflags.append(termios.tcgetattr(slave)[3])
            return super().write(s)

    out = Out()
    os.write(master, b"sk-live\n")  # pasted before the prompt appears; it must not be discarded
    with os.fdopen(slave, "r", encoding="utf-8", closefd=False) as stdin:
        assert ask_secret("API key: ", stdin=stdin, out=out) == "sk-live"
    assert out.getvalue() == "API key: *******\n"
    assert lflags and not any(f & (termios.ECHO | termios.ICANON) for f in lflags)
    assert all(f & termios.ISIG for f in lflags)  # Ctrl-C still interrupts
    assert _settings(slave) == before


def test_the_terminal_is_restored_when_the_read_raises(pty_pair):
    class Broken(_Tty):
        def write(self, s):
            if s == "*":
                raise OSError("terminal went away")
            return super().write(s)

    master, slave = pty_pair
    before = _settings(slave)
    os.write(master, b"x\n")
    with os.fdopen(slave, "r", encoding="utf-8", closefd=False) as stdin:
        with pytest.raises(OSError, match="terminal went away"):
            ask_secret("k: ", stdin=stdin, out=Broken())
    assert _settings(slave) == before


def test_the_windows_reader_swallows_both_halves_of_a_special_key():
    keys = iter(["a", "\x00", "H", "\xe0", "K", "b"])
    getch = masked_input.windows_getch(lambda: next(keys))
    assert [getch(), getch()] == ["a", "b"]


def test_default_ask_masks_a_secret_and_strips_it(monkeypatch):
    asked = []
    monkeypatch.setattr(cli, "ask_secret", lambda p: asked.append(p) or "  sk-123 \n")
    monkeypatch.setattr("builtins.input", lambda p: pytest.fail("a secret must not use input()"))
    assert cli.default_ask("API key", secret=True) == "sk-123"
    assert asked == ["API key: "]


def test_default_ask_shows_a_secret_prompt_its_default_and_falls_back_to_it(monkeypatch):
    asked = []
    monkeypatch.setattr(cli, "ask_secret", lambda p: asked.append(p) or "")
    assert cli.default_ask("Token", default="keep-current", secret=True) == "keep-current"
    assert asked == ["Token [keep-current]: "]


def test_default_ask_still_echoes_a_plain_answer_and_applies_the_default(monkeypatch):
    monkeypatch.setattr(cli, "ask_secret", lambda p: pytest.fail("not a secret"))
    monkeypatch.setattr("builtins.input", lambda p: "")
    assert cli.default_ask("Provider", default="anthropic") == "anthropic"
