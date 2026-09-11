"""Anthropic backend via the official SDK."""
from __future__ import annotations

from talos.providers import ProviderAuthError, ProviderError, ProviderRateLimited
from talos.providers.pricing import estimate_cost
from talos.types import Completion, Usage


class AnthropicProvider:
    metered = True

    def __init__(self, model: str, api_key: str | None, effort: str = "high",
                 max_tokens: int = 32000, client=None):
        import anthropic
        self.name = "anthropic"
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self._client = client or anthropic.Anthropic(api_key=api_key)

    def complete(self, system: str, user: str) -> Completion:
        import anthropic
        try:
            with self._client.messages.stream(
                model=self.model, max_tokens=self.max_tokens, system=system,
                thinking={"type": "adaptive"}, output_config={"effort": self.effort},
                messages=[{"role": "user", "content": user}],
            ) as stream:
                msg = stream.get_final_message()
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as e:
            raise ProviderAuthError(str(e)) from None
        except anthropic.RateLimitError as e:
            raise ProviderRateLimited(str(e)) from None
        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                raise ProviderRateLimited(f"server error {e.status_code}") from None
            raise ProviderError(str(e)) from None
        except anthropic.APIConnectionError as e:
            raise ProviderRateLimited(f"connection error: {e}") from None
        if msg.stop_reason == "refusal":
            raise ProviderError("model refused the request")
        text = "".join(b.text for b in msg.content if b.type == "text")
        usage = Usage(msg.usage.input_tokens, msg.usage.output_tokens)
        usage.cost_usd = estimate_cost(self.model, usage)
        return Completion(text=text, usage=usage)
