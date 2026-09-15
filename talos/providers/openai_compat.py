"""OpenAI-compatible chat completions over urllib. Covers OpenAI, OpenRouter, DeepSeek-style
endpoints and any local server. `post` is injectable for tests."""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from talos.providers import ProviderAuthError, ProviderError, ProviderRateLimited
from talos.providers.pricing import estimate_cost
from talos.types import Completion, Usage


class HTTPError(Exception):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:300]}")
        self.status = status
        self.body = body


def _post_json(url: str, body: dict, headers: dict) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            raw = r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        raise HTTPError(e.code, e.read().decode("utf-8", "replace")) from None
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise ProviderRateLimited(f"network error: {e}") from None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        raise ProviderError(f"non-JSON response from {url}: {raw[:200]}") from None


def _map_http(e: HTTPError) -> ProviderError:
    if e.status in (401, 403, 402):
        return ProviderAuthError(str(e))
    if e.status == 429 or e.status >= 500:
        return ProviderRateLimited(str(e))
    return ProviderError(str(e))


class OpenAICompat:
    metered = True

    def __init__(self, api_base: str, api_key: str, model: str, post=_post_json,
                 max_tokens: int = 16000):
        self.name = f"openai-compat:{api_base}"
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self._post = post
        self.max_tokens = max_tokens

    def complete(self, system: str, user: str) -> Completion:
        body = {"model": self.model,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}],
                "max_completion_tokens": self.max_tokens}
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            resp = self._post(f"{self.api_base}/chat/completions", body, headers)
        except HTTPError as e:
            if e.status == 400 and "max_completion_tokens" in e.body:
                body["max_tokens"] = body.pop("max_completion_tokens")
                try:
                    resp = self._post(f"{self.api_base}/chat/completions", body, headers)
                except HTTPError as e2:
                    raise _map_http(e2) from None
            else:
                raise _map_http(e) from None
        try:
            text = resp["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            raise ProviderError(f"unexpected response shape: {str(resp)[:300]}") from None
        u = resp.get("usage") or {}
        usage = Usage(int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0)))
        usage.cost_usd = estimate_cost(self.model, usage)
        return Completion(text=text, usage=usage)
