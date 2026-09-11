import json
import types

import httpx
import pytest

from talos.providers import (ProviderAuthError, ProviderRateLimited, make_provider,
                             validate_provider)
from talos.providers.fake import FakeProvider
from talos.providers.openai_compat import OpenAICompat
from talos.providers.pricing import estimate_cost
from talos.providers.claude_cli import ClaudeCli
from talos.providers.codex_cli import CodexCli
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
    assert seen["body"]["messages"][0] == {"role": "system", "content": "SYS"}
    assert seen["body"]["messages"][1] == {"role": "user", "content": "USER"}


def test_openai_compat_maps_http_errors():
    # mutation: treating 401 as retryable would loop forever on a bad key
    from talos.providers.openai_compat import HTTPError
    def post401(url, body, headers):
        raise HTTPError(401, "bad key")
    def post429(url, body, headers):
        raise HTTPError(429, "slow down")
    with pytest.raises(ProviderAuthError):
        OpenAICompat("https://x/v1", "k", "m", post=post401).complete("s", "u")
    with pytest.raises(ProviderRateLimited):
        OpenAICompat("https://x/v1", "k", "m", post=post429).complete("s", "u")


def test_claude_cli_parses_json_result_and_cost():
    def run(cmd, input=None, **kw):
        # mutation: shutil.which() in argv[0] makes this machine-dependent
        assert cmd[:2] == ["claude", "-p"] and "--output-format" in cmd and "json" in cmd
        assert "--system-prompt" in cmd and input == "USER"
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


def test_make_provider_kinds_and_validate():
    fp = make_provider("fake", "m")
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
    resp = httpx.Response(429, request=httpx.Request("POST", "https://x"))
    err = anthropic.RateLimitError("slow", response=resp, body=None)
    with pytest.raises(ProviderRateLimited):
        AnthropicProvider("claude-opus-5", "k", client=client(err)).complete("s", "u")
    resp = httpx.Response(401, request=httpx.Request("POST", "https://x"))
    err = anthropic.AuthenticationError("bad", response=resp, body=None)
    with pytest.raises(ProviderAuthError):
        AnthropicProvider("claude-opus-5", "k", client=client(err)).complete("s", "u")
