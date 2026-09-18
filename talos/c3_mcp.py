"""C3 over its hosted MCP endpoint. One JSON-RPC client plus the transport C3Bench calls.

Wire facts measured 2026-09-17 (spec docs/ai/specs/2026-09-17-c3-mcp-transport-design.md §2):
Cloudflare rejects the default urllib User-Agent with 403 error 1010, every request including
`initialize` needs the key, responses are application/json, and no session id is issued.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Callable

from talos import __version__
from talos.bench import _redact
from talos.c3_bench import C3CommandError

MCP_URL = "https://api.cthree.cloud/mcp"
PROTOCOL_VERSION = "2025-06-18"


class McpAuthError(C3CommandError):
    """The key was rejected (401/403). Setup reports this as a key problem, not an outage."""


class _SessionGone(C3CommandError):
    """404 on a request that carried a session id: the server dropped the session."""


def _urllib_post(url: str, headers: dict, body: bytes, timeout_s: int):
    req = urllib.request.Request(url, body, headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


class McpClient:
    def __init__(self, api_key: str, url: str = MCP_URL, post: Callable | None = None,
                 timeout_s: int = 60):
        self._key = api_key
        self._url = url
        self._post = post or _urllib_post
        self._timeout_s = timeout_s
        self._session: str | None = None
        self._ready = False
        self._id = 0

    def _headers(self) -> dict:
        h = {"User-Agent": f"talos/{__version__}", "Content-Type": "application/json",
             "Accept": "application/json, text/event-stream",
             "MCP-Protocol-Version": PROTOCOL_VERSION,
             "Authorization": f"Bearer {self._key}"}
        if self._session:
            h["Mcp-Session-Id"] = self._session
        return h

    def _send(self, method: str, params: dict | None, notify: bool = False):
        self._id += 1
        body: dict = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            body["params"] = params
        if not notify:
            body["id"] = self._id
        try:
            status, headers, raw = self._post(self._url, self._headers(),
                                              json.dumps(body).encode(), self._timeout_s)
        except Exception as e:  # OSError, URLError, anything the injected post raises
            raise C3CommandError(f"c3 {method} could not be sent: "
                                 f"{_redact(str(e))[:200]}") from None
        # urllib capitalises names ("Mcp-session-id") and HTTP/2 front ends lower-case them
        headers = {str(k).lower(): v for k, v in (headers or {}).items()}
        if status in (401, 403):
            raise McpAuthError("the C3 API key was rejected (check `c3 apikey list`)")
        if status == 404 and self._session:
            raise _SessionGone(f"c3 {method} failed with HTTP 404: the session is gone")
        if status >= 400:
            raise C3CommandError(f"c3 {method} failed with HTTP {status}")
        self._session = headers.get("mcp-session-id") or self._session
        if notify:
            return None
        return self._result(method, headers.get("content-type", ""), raw, self._id)

    def _result(self, method: str, content_type: str, raw: bytes, id_: int) -> dict:
        text = raw.decode("utf-8", "replace")
        docs = []
        if "text/event-stream" in content_type:
            for line in text.splitlines():
                if line.startswith("data:"):
                    try:
                        docs.append(json.loads(line[5:].strip()))
                    except ValueError:
                        continue
        else:
            try:
                docs = [json.loads(text)]
            except ValueError:
                raise C3CommandError(f"c3 {method} returned no JSON") from None
        docs = [d for d in docs if isinstance(d, dict)]
        doc = next((d for d in docs if d.get("id") == id_), docs[-1] if docs else None)
        if doc is None:
            raise C3CommandError(f"c3 {method} returned no reply for this request")
        if doc.get("error"):
            msg = str(doc["error"].get("message", doc["error"]))
            raise C3CommandError(f"c3 {method} failed: {_redact(msg)[:300]}")
        return doc.get("result") or {}

    def _ensure_ready(self) -> None:
        if self._ready:
            return
        self._send("initialize", {"protocolVersion": PROTOCOL_VERSION, "capabilities": {},
                                  "clientInfo": {"name": "talos", "version": __version__}})
        self._send("notifications/initialized", None, notify=True)
        self._ready = True

    def tool(self, name: str, arguments: dict) -> dict:
        """Calls one tool and returns its result document."""
        self._ensure_ready()
        params = {"name": name, "arguments": arguments}
        try:
            res = self._send("tools/call", params) or {}
        except _SessionGone:
            # spec §4.3: start a new session once. A second 404 raises as a C3CommandError.
            self._session, self._ready = None, False
            self._ensure_ready()
            res = self._send("tools/call", params) or {}
        text = " ".join(c.get("text", "") for c in res.get("content", [])
                        if isinstance(c, dict))
        if res.get("isError"):
            raise C3CommandError(f"c3 {name} failed: {_redact(text)[:300]}")
        if res.get("structuredContent") is not None:
            return res["structuredContent"]
        try:
            return json.loads(text)
        except ValueError:
            return {"text": text}
