import json

import pytest

from talos import __version__
from talos.c3_bench import C3CommandError
from talos.c3_mcp import McpAuthError, McpClient

KEY = "c3_key_" + "a" * 20


def rpc(id_, result):
    return json.dumps({"jsonrpc": "2.0", "id": id_, "result": result}).encode()


class FakePost:
    """Scripted transport. `replies` maps JSON-RPC method -> (status, headers, body-builder)."""

    def __init__(self, replies=None):
        self.calls = []
        self.replies = replies or {}

    def __call__(self, url, headers, body, timeout_s):
        doc = json.loads(body)
        self.calls.append((url, headers, doc))
        method = doc.get("method")
        if method in self.replies:
            return self.replies[method](doc)
        if method == "initialize":
            return 200, {"Content-Type": "application/json"}, rpc(doc["id"], {
                "protocolVersion": "2025-06-18", "serverInfo": {"name": "c3", "version": "1.0.0"}})
        if method == "notifications/initialized":
            return 202, {}, b""
        return 200, {"Content-Type": "application/json"}, rpc(doc["id"], {
            "structuredContent": {"ok": True, "tool": doc["params"]["name"]}})


def test_tool_call_sends_auth_and_user_agent_and_returns_structured_content():
    post = FakePost()
    out = McpClient(KEY, post=post).tool("whoami", {})
    assert out == {"ok": True, "tool": "whoami"}
    # mutation: a missing User-Agent is a Cloudflare 403; a missing key is a 401
    for _url, headers, _doc in post.calls:
        assert headers["User-Agent"] == f"talos/{__version__}"
        assert headers["Authorization"] == f"Bearer {KEY}"
        assert "application/json" in headers["Accept"]
    # mutation: initialize skipped, or repeated per call
    methods = [doc["method"] for _u, _h, doc in post.calls]
    assert methods == ["initialize", "notifications/initialized", "tools/call"]


def test_initialize_runs_once_across_calls():
    post = FakePost()
    c = McpClient(KEY, post=post)
    c.tool("whoami", {})
    c.tool("balance", {})
    # mutation: re-initialising per call doubles every request
    assert [doc["method"] for _u, _h, doc in post.calls].count("initialize") == 1


def test_session_id_is_echoed_when_the_server_issues_one():
    def init(doc):
        return 200, {"Content-Type": "application/json", "Mcp-Session-Id": "sess-1"}, rpc(
            doc["id"], {"protocolVersion": "2025-06-18"})

    post = FakePost({"initialize": init})
    McpClient(KEY, post=post).tool("whoami", {})
    # mutation: dropping the session header breaks a stateful server (today's server sends none)
    assert post.calls[-1][1]["Mcp-Session-Id"] == "sess-1"


def test_event_stream_response_is_parsed_by_request_id():
    def call(doc):
        body = (b"event: message\ndata: " + rpc(99, {"structuredContent": {"wrong": True}})
                + b"\n\nevent: message\ndata: " + rpc(doc["id"], {"structuredContent": {"n": 7}})
                + b"\n\n")
        return 200, {"Content-Type": "text/event-stream"}, body

    out = McpClient(KEY, post=FakePost({"tools/call": call})).tool("get_job", {"job_id": "j"})
    # mutation: taking the first data: line returns another request's reply
    assert out == {"n": 7}


def test_content_text_json_is_used_when_there_is_no_structured_content():
    def call(doc):
        return 200, {"Content-Type": "application/json"}, rpc(
            doc["id"], {"content": [{"type": "text", "text": json.dumps({"balance_gbp": 1.5})}]})

    out = McpClient(KEY, post=FakePost({"tools/call": call})).tool("balance", {})
    # mutation: reading structuredContent only raises KeyError on a server that omits it
    assert out == {"balance_gbp": 1.5}


def test_tool_error_and_rpc_error_raise_c3commanderror():
    def is_error(doc):
        return 200, {"Content-Type": "application/json"}, rpc(doc["id"], {
            "isError": True,
            "content": [{"type": "text", "text": json.dumps(
                {"error": {"code": "NOT_FOUND", "message": "artifact not found: x"}})}]})

    with pytest.raises(C3CommandError) as ei:
        McpClient(KEY, post=FakePost({"tools/call": is_error})).tool("read_artifact", {})
    assert "NOT_FOUND" in str(ei.value)  # mutation: isError treated as success

    def rpc_error(doc):
        return 200, {"Content-Type": "application/json"}, json.dumps(
            {"jsonrpc": "2.0", "id": doc["id"],
             "error": {"code": -32602, "message": "bad params"}}).encode()

    with pytest.raises(C3CommandError):
        McpClient(KEY, post=FakePost({"tools/call": rpc_error})).tool("deploy", {})


@pytest.mark.parametrize("status", [401, 403])
def test_auth_failures_raise_mcpautherror_without_the_key(status):
    def call(doc):
        return status, {"Content-Type": "application/json"}, json.dumps(
            {"error": {"code": "UNAUTHORIZED", "message": f"send Bearer {KEY}"}}).encode()

    with pytest.raises(McpAuthError) as ei:
        McpClient(KEY, post=FakePost({"initialize": call})).tool("whoami", {})
    # mutation: a 401 reported as a transient failure makes setup retry a revoked key;
    # echoing the server's message leaks the key into the wizard's output
    assert KEY not in str(ei.value) and "c3_key_" not in str(ei.value)


def test_transport_exception_becomes_c3commanderror_without_the_key():
    def boom(doc):
        raise OSError(f"connection reset while sending Bearer {KEY}")

    with pytest.raises(C3CommandError) as ei:
        McpClient(KEY, post=FakePost({"initialize": boom})).tool("whoami", {})
    assert KEY not in str(ei.value)  # mutation: str(e) passed through unredacted


def test_a_404_for_a_stale_session_starts_a_new_session_once():
    state = {"inits": 0, "calls": 0}

    def init(doc):
        state["inits"] += 1
        return 200, {"Content-Type": "application/json",
                     "Mcp-Session-Id": f"sess-{state['inits']}"}, rpc(doc["id"], {})

    def first_call_is_gone(doc):
        state["calls"] += 1
        if state["calls"] == 1:
            return 404, {}, b""
        return 200, {"Content-Type": "application/json"}, rpc(
            doc["id"], {"structuredContent": {"ok": True}})

    post = FakePost({"initialize": init, "tools/call": first_call_is_gone})
    # mutation: a 404 raised straight through pauses the run on a session the server dropped
    assert McpClient(KEY, post=post).tool("whoami", {}) == {"ok": True}
    assert state["inits"] == 2 and post.calls[-1][1]["Mcp-Session-Id"] == "sess-2"

    post2 = FakePost({"initialize": init, "tools/call": lambda doc: (404, {}, b"")})
    with pytest.raises(C3CommandError):
        McpClient(KEY, post=post2).tool("whoami", {})
    # mutation: retrying without a bound loops for ever against a server that always says 404
    assert [d["method"] for _u, _h, d in post2.calls].count("tools/call") == 2


def test_response_headers_are_read_whatever_their_case():
    def init(doc):
        return 200, {"content-type": "application/json", "mcp-session-id": "sess-9"}, rpc(
            doc["id"], {})

    def call(doc):
        return 200, {"content-type": "text/event-stream"}, (
            b"data: " + rpc(doc["id"], {"structuredContent": {"n": 1}}) + b"\n\n")

    post = FakePost({"initialize": init, "tools/call": call})
    # mutation: a case-sensitive lookup misses the lower-case names an HTTP/2 front end sends
    assert McpClient(KEY, post=post).tool("whoami", {}) == {"n": 1}
    assert post.calls[-1][1]["Mcp-Session-Id"] == "sess-9"
