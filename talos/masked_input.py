"""Reads a secret from the terminal and shows one `*` per character, so a pasted key is visibly
there. `getpass` shows nothing, and its `echo_char` needs Python 3.14. Without a terminal on both
stdin and stdout (a pipe, CI) there is nothing to draw on and `getpass` does the read."""
from __future__ import annotations

import contextlib
import getpass
import os
import sys

_ENTER = ("\r", "\n")  # a POSIX terminal sends \n (ICRNL stays on), the Windows console \r
_BACKSPACE = ("\x7f", "\b")
_CTRL_C, _CTRL_D, _ESC = "\x03", "\x04", "\x1b"


def read_masked(prompt: str, getch, out) -> str:
    """`getch` returns one character per call and "" once stdin is closed."""

    def show(s: str) -> None:
        out.write(s)
        out.flush()

    show(prompt)
    chars: list[str] = []
    while True:
        ch = getch()
        while ch == _ESC:  # a sequence can be followed directly by another
            ch = _after_escape(getch)
        if ch in _ENTER:
            show("\n")
            return "".join(chars)
        if ch == _CTRL_C:  # Windows only: a POSIX terminal keeps ISIG and sends SIGINT instead
            show("\n")
            raise KeyboardInterrupt
        if ch == "" or (ch == _CTRL_D and not chars):
            show("\n")
            raise EOFError
        if ch in _BACKSPACE:
            if chars:
                chars.pop()
                show("\b \b")
        elif ch.isprintable():
            chars.append(ch)
            show("*")


def _after_escape(getch) -> str:
    """Consumes the rest of an escape sequence and returns the first character that is not part
    of one. Some terminals wrap a paste in ESC[200~ … ESC[201~, and dropping only the ESC would
    leave `[200~` in the secret."""
    ch = getch()
    if ch == "[":  # CSI: parameter and intermediate bytes, then one final byte in @ … ~
        while (ch := getch()) and not "@" <= ch <= "~":
            pass
        return getch() if ch else ""
    if ch == "O":  # SS3: exactly one more character
        return getch() if getch() else ""
    return ch


def windows_getch(getwch):
    """The console reports an arrow or function key as \\x00 or \\xe0 followed by a key code."""
    def getch() -> str:
        while (ch := getwch()) in ("\x00", "\xe0"):
            getwch()
        return ch
    return getch


@contextlib.contextmanager
def _cbreak(fd: int):
    import termios
    import tty

    old = termios.tcgetattr(fd)
    try:
        # no echo, no line buffering; Ctrl-C and Ctrl-Z stay signals. TCSADRAIN, not the
        # default TCSAFLUSH, which throws away a key pasted before the prompt appeared
        tty.setcbreak(fd, termios.TCSADRAIN)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _isatty(stream) -> bool:
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError):  # no stream at all (pythonw), or a closed one
        return False


def ask_secret(prompt: str, stdin=None, out=None) -> str:
    stdin = sys.stdin if stdin is None else stdin
    out = sys.stdout if out is None else out
    if not (_isatty(stdin) and _isatty(out)):
        return getpass.getpass(prompt)
    if os.name == "nt":
        import msvcrt

        return read_masked(prompt, windows_getch(msvcrt.getwch), out)
    with _cbreak(stdin.fileno()):
        return read_masked(prompt, lambda: stdin.read(1), out)
