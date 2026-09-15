"""Gemini generateContent over urllib."""
from __future__ import annotations

from talos.providers import ProviderError
from talos.providers.openai_compat import HTTPError, _map_http, _post_json
from talos.providers.pricing import estimate_cost
from talos.types import Completion, Usage

BASE = "https://generativelanguage.googleapis.com/v1beta/models"


class GoogleProvider:
    metered = True

    def __init__(self, model: str, api_key: str, post=_post_json):
        self.name = "google"
        self.model = model
        self.api_key = api_key
        self._post = post

    def complete(self, system: str, user: str) -> Completion:
        body = {"system_instruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}]}
        url = f"{BASE}/{self.model}:generateContent"
        try:
            resp = self._post(url, body, {"x-goog-api-key": self.api_key})  # never in the URL
        except HTTPError as e:
            raise _map_http(e) from None
        try:
            text = "".join(p.get("text", "") for p in resp["candidates"][0]["content"]["parts"])
        except (KeyError, IndexError, TypeError):
            raise ProviderError(f"unexpected response shape: {str(resp)[:300]}") from None
        u = resp.get("usageMetadata") or {}
        usage = Usage(int(u.get("promptTokenCount", 0)), int(u.get("candidatesTokenCount", 0)))
        usage.cost_usd = estimate_cost(self.model, usage)
        return Completion(text=text, usage=usage)
