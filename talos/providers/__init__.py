"""Provider protocol and factory. Every backend returns a Completion with measured usage."""
from __future__ import annotations

from typing import Protocol

from talos.types import Completion


class ProviderError(RuntimeError):
    pass


class ProviderAuthError(ProviderError):
    """Bad or missing credential, billing failure. Never retried."""


class ProviderRateLimited(ProviderError):
    """429 or equivalent. The loop waits and retries."""


class Provider(Protocol):
    name: str
    metered: bool

    def complete(self, system: str, user: str) -> Completion: ...


DEFAULT_MODELS = {
    "anthropic": "claude-opus-5",
    "openai": "gpt-5",
    "google": "gemini-2.5-pro",
    "openrouter": "anthropic/claude-opus-5",
    "custom": "",
    "claude-cli": "claude-opus-5",
    "codex-cli": "gpt-5.5",
    "fake": "fake",
}

KINDS = tuple(DEFAULT_MODELS)


def make_provider(kind: str, model: str, api_key: str | None = None,
                  api_base: str | None = None) -> Provider:
    if kind == "anthropic":
        from talos.providers.anthropic_provider import AnthropicProvider
        return AnthropicProvider(model=model, api_key=api_key)
    if kind in ("openai", "openrouter", "custom"):
        from talos.providers.openai_compat import OpenAICompat
        base = api_base or {"openai": "https://api.openai.com/v1",
                            "openrouter": "https://openrouter.ai/api/v1"}.get(kind)
        if not base:
            raise ValueError("custom provider needs api_base")
        return OpenAICompat(api_base=base, api_key=api_key or "", model=model)
    if kind == "google":
        from talos.providers.google import GoogleProvider
        return GoogleProvider(model=model, api_key=api_key or "")
    if kind == "claude-cli":
        from talos.providers.claude_cli import ClaudeCli
        return ClaudeCli(model=model)
    if kind == "codex-cli":
        from talos.providers.codex_cli import CodexCli
        return CodexCli(model=model)
    if kind == "fake":
        from talos.providers.fake import FakeProvider
        return FakeProvider(lambda s, u: "<<<<<<< SEARCH\nx\n=======\ny\n>>>>>>> REPLACE")
    raise ValueError(f"unknown provider kind {kind!r}; choose one of {KINDS}")


def validate_provider(p: Provider) -> str | None:
    """One tiny call. None on success, else a message naming what to fix."""
    try:
        c = p.complete("Reply with the single word OK.", "Say OK.")
    except ProviderAuthError as e:
        return f"credential rejected: {e}"
    except ProviderError as e:
        return f"provider error: {e}"
    return None if c.text.strip() else "provider returned an empty reply"
