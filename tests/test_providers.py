import io
import json
import types
import urllib.error
from pathlib import Path

import httpx2
import pytest

from talos.providers import (ProviderAuthError, ProviderError, ProviderRateLimited,
                             make_provider, validate_provider)
from talos.providers.fake import FakeProvider
from talos.providers.openai_compat import OpenAICompat
from talos.providers.pricing import estimate_cost
from talos.providers.claude_cli import ClaudeCli
from talos.providers.codex_cli import CodexCli, list_codex_models
from talos.types import Usage


def test_pricing_known_and_unknown():
    # mutation: swapping input/output rates changes 1M-in vs 1M-out asymmetry
    assert estimate_cost("claude-opus-5", Usage(1_000_000, 0)) == pytest.approx(5.0)
    assert estimate_cost("claude-opus-5", Usage(0, 1_000_000)) == pytest.approx(25.0)
    assert estimate_cost("some/unknown-model", Usage(10, 10)) is None


def test_fake_provider_scripted_and_metered():
    p = FakeProvider(["first", "second"])
    a = p.complete("sys", "u1")
    b = p.complete("sys", "u2")
    # mutation: not advancing self._i would return "first" for both calls, and skipping
    # the calls.append would let a caller run unlogged prompts through a "recording" fake
    assert (a.text, b.text) == ("first", "second") and p.calls == [("sys", "u1"), ("sys", "u2")]
    assert a.usage.cost_usd == pytest.approx(0.01) and p.metered


def test_openai_compat_request_shape_and_usage():
    seen = {}
    def post(url, body, headers):
        seen.update(url=url, body=body, headers=headers)
        return {"choices": [{"message": {"content": "hello"}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 3}}
    p = OpenAICompat(api_base="https://api.openai.com/v1", api_key="k", model="gpt-5", post=post)
    c = p.complete("SYS", "USER")
    assert c.text == "hello" and c.usage.input_tokens == 12 and c.usage.output_tokens == 3
    assert seen["url"] == "https://api.openai.com/v1/chat/completions"
    assert seen["headers"]["Authorization"] == "Bearer k"
    # mutation: swapping prompt_tokens/completion_tokens in the Usage mapping, or swapping
    # the system/user message order, would silently mis-report cost and mis-prompt the model
    assert seen["body"]["messages"][0] == {"role": "system", "content": "SYS"}
    assert seen["body"]["messages"][1] == {"role": "user", "content": "USER"}


def test_openai_compat_maps_http_errors():
    # mutation: treating 401 as retryable would loop forever on a bad key
    from talos.providers.openai_compat import HTTPError
    def post401(url, body, headers):
        raise HTTPError(401, "bad key")
    def post429(url, body, headers):
        raise HTTPError(429, "slow down")
    def post503(url, body, headers):
        raise HTTPError(503, "down for maintenance")
    with pytest.raises(ProviderAuthError):
        OpenAICompat("https://x/v1", "k", "m", post=post401).complete("s", "u")
    with pytest.raises(ProviderRateLimited):
        OpenAICompat("https://x/v1", "k", "m", post=post429).complete("s", "u")
    # mutation: mapping 5xx to ProviderError makes one transient server error kill the run
    with pytest.raises(ProviderRateLimited):
        OpenAICompat("https://x/v1", "k", "m", post=post503).complete("s", "u")


def test_post_json_handles_non_json_and_http_error(monkeypatch):
    from talos.providers.openai_compat import HTTPError, _post_json

    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"<html>oops</html>"

    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=None: FakeResp())
    # mutation: an uncaught JSONDecodeError escapes the provider error hierarchy and crashes the loop
    with pytest.raises(ProviderError):
        _post_json("https://x/v1/chat/completions", {}, {})

    def raise_503(req, timeout=None):
        raise urllib.error.HTTPError("https://x/v1/chat/completions", 503, "down", {},
                                     io.BytesIO(b"down"))

    monkeypatch.setattr("urllib.request.urlopen", raise_503)
    with pytest.raises(HTTPError) as ei:
        _post_json("https://x/v1/chat/completions", {}, {})
    assert ei.value.status == 503


def test_claude_cli_parses_json_result_and_cost():
    def run(cmd, input=None, **kw):
        # mutation: shutil.which() in argv[0] makes this machine-dependent
        assert cmd[:2] == ["claude", "-p"] and "--output-format" in cmd and "json" in cmd
        assert "--system-prompt-file" in cmd and input == "USER"
        class R:
            returncode = 0
            stdout = json.dumps({"result": "the code", "total_cost_usd": 0.42,
                                 "usage": {"input_tokens": 5, "output_tokens": 7}})
            stderr = ""
        return R()
    p = ClaudeCli(model="claude-opus-5", run=run)
    c = p.complete("SYS", "USER")
    assert c.text == "the code" and c.usage.cost_usd == 0.42 and not p.metered


def test_codex_cli_reads_last_message_file(tmp_path):
    def run(cmd, **kw):
        # mutation: shutil.which() in argv[0] makes this machine-dependent
        assert cmd[:2] == ["codex", "exec"] and "-m" in cmd
        out = cmd[cmd.index("-o") + 1]
        open(out, "w").write("edited code")

        class R:
            returncode = 0
            stdout = ""
            stderr = ""
        return R()
    p = CodexCli(model="gpt-5-codex", run=run)
    c = p.complete("SYS", "USER")
    assert c.text == "edited code" and not p.metered


def test_google_provider_header_key_and_usage():
    from talos.providers import ProviderAuthError
    from talos.providers.google import GoogleProvider
    from talos.providers.openai_compat import HTTPError

    seen = {}
    def post(url, body, headers):
        seen.update(url=url, body=body, headers=headers)
        return {"candidates": [{"content": {"parts": [{"text": "hi "}, {"text": "there"}]}}],
                "usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 3}}
    p = GoogleProvider(model="gemini-2.5-pro", api_key="secret-key", post=post)
    c = p.complete("SYS", "USER")
    assert seen["url"] == ("https://generativelanguage.googleapis.com/v1beta/models/"
                           "gemini-2.5-pro:generateContent")
    assert "key=" not in seen["url"]
    assert seen["headers"]["x-goog-api-key"] == "secret-key"
    assert seen["body"]["system_instruction"]["parts"][0]["text"] == "SYS"
    assert seen["body"]["contents"][0]["parts"][0]["text"] == "USER"
    assert c.text == "hi there"
    assert c.usage.input_tokens == 7 and c.usage.output_tokens == 3

    def post401(url, body, headers):
        raise HTTPError(401, "bad key")
    # mutation: putting the key in the URL leaks it into error messages and proxy logs
    with pytest.raises(ProviderAuthError):
        GoogleProvider(model="gemini-2.5-pro", api_key="secret-key", post=post401).complete("s", "u")


def test_make_provider_kinds_and_validate():
    fp = make_provider("fake", "m")
    # mutation: not raising ValueError for an unknown kind would return None and defer the
    # failure to a later, harder-to-diagnose AttributeError when .complete() is called
    assert isinstance(fp, FakeProvider) and validate_provider(fp) is None
    with pytest.raises(ValueError):
        make_provider("nope", "m")


def test_anthropic_provider_maps_errors_and_prices_usage():
    import anthropic

    from talos.providers.anthropic_provider import AnthropicProvider

    class Block:
        type = "text"
        text = "done"

    class Msg:
        stop_reason = "end_turn"
        content = [Block()]
        usage = types.SimpleNamespace(input_tokens=1_000_000, output_tokens=0)

    class Stream:
        def __init__(self, exc):
            self.exc = exc

        def __enter__(self):
            if self.exc:
                raise self.exc
            return self

        def __exit__(self, *a):
            return False

        def get_final_message(self):
            return Msg()

    def client(exc=None):
        return types.SimpleNamespace(messages=types.SimpleNamespace(stream=lambda **kw: Stream(exc)))

    p = AnthropicProvider(model="claude-opus-5", api_key="k", client=client())
    c = p.complete("s", "u")
    assert c.text == "done" and c.usage.cost_usd == pytest.approx(5.0) and p.metered
    # mutation: mapping RateLimitError to ProviderError makes the loop fail instead of wait
    resp = httpx2.Response(429, request=httpx2.Request("POST", "https://x"))
    err = anthropic.RateLimitError("slow", response=resp, body=None)
    with pytest.raises(ProviderRateLimited):
        AnthropicProvider("claude-opus-5", "k", client=client(err)).complete("s", "u")
    resp = httpx2.Response(401, request=httpx2.Request("POST", "https://x"))
    err = anthropic.AuthenticationError("bad", response=resp, body=None)
    with pytest.raises(ProviderAuthError):
        AnthropicProvider("claude-opus-5", "k", client=client(err)).complete("s", "u")


def _codex_catalog(*entries):
    return json.dumps({"models": [
        {"slug": s, "visibility": v, "priority": p} for s, v, p in entries]})


def _run_returning(stdout, returncode=0):
    def run(cmd, **kw):
        assert cmd == ["codex", "debug", "models"]
        return types.SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")
    return run


def test_codex_list_models_keeps_visible_entries_in_priority_order():
    # mutation: dropping the visibility filter leaks gpt-reserve; dropping the sort returns
    # catalog order, which puts gpt-5.5 first here
    run = _run_returning(_codex_catalog(("gpt-5.5", "list", 12), ("gpt-reserve", "hide", 3),
                                        ("gpt-5.6-sol", "list", 4)))
    assert list_codex_models(run=run) == ["gpt-5.6-sol", "gpt-5.5"]


@pytest.mark.parametrize("run", [
    _run_returning("", returncode=1),
    _run_returning("not json"),
    _run_returning(json.dumps({"models": "oops"})),
    lambda cmd, **kw: (_ for _ in ()).throw(FileNotFoundError("codex")),
])
def test_codex_list_models_is_empty_when_the_catalog_is_unavailable(run):
    # mutation: letting any of these raise breaks `talos setup` on a machine without codex
    assert list_codex_models(run=run) == []


def test_codex_cli_sends_the_prompt_on_stdin_not_argv(tmp_path):
    # Linux caps one argv element at 128 KiB (MAX_ARG_STRLEN); a knapsack prompt carrying five
    # ~100 KB track files failed with "[Errno 7] Argument list too long" on 2026-09-16.
    # mutation: putting the prompt back in argv makes the longest element ~200 KB
    seen = {}
    def run(cmd, **kw):
        seen["cmd"], seen["input"] = cmd, kw.get("input")
        Path(cmd[cmd.index("-o") + 1]).write_text("ok")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    big = "x" * 200_000
    CodexCli(model="gpt-5.5", run=run).complete("SYS", big)
    assert max(len(a) for a in seen["cmd"]) < 131072
    assert seen["cmd"][-1] == "-" and "SYS" in seen["input"] and big in seen["input"]
