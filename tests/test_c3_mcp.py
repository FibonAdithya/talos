import hashlib
import json

import pytest

from talos import __version__
from talos.c3_bench import C3CommandError
from talos.c3_jobdir import job_settings, write_job_dir
from talos.c3_mcp import McpAuthError, McpClient, McpTransport

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


class FakeClient:
    """Records tool calls; returns scripted documents."""

    def __init__(self, docs):
        self.docs = docs
        self.calls = []

    def tool(self, name, arguments):
        self.calls.append((name, arguments))
        doc = self.docs[name]
        return doc(arguments) if callable(doc) else doc


def job_dir_for(tmp_path):
    from tests.test_c3_jobdir import req
    # The directory is deliberately not named after the purpose: the settings come from `.c3`.
    return write_job_dir(tmp_path / "job", req(), "3")


def test_deploy_sends_the_files_with_job_sh_executable_and_the_c3_settings(tmp_path):
    c = FakeClient({"deploy": {"job_id": "job_5"}})
    d = job_dir_for(tmp_path)
    assert McpTransport("k", client=c).deploy(d) == "job_5"
    name, args = c.calls[0]
    assert name == "deploy"
    files = {f["path"]: f for f in args["files"]}
    # mutation: no executable flag means C3 cannot run job.sh from a Windows upload
    assert files["job.sh"]["executable"] is True
    assert all(f.get("executable") is not True for p, f in files.items() if p != "job.sh")
    # mutation: shipping .c3 duplicates the settings and can carry an api_key upstream
    assert ".c3" not in files
    assert "payload.json" in files and "talos/c3_job.py" in files
    assert files["job.sh"]["content"].startswith("#!/bin/bash")
    # mutation: a hard-coded profile breaks invariant 1
    for k, v in job_settings("knapsack", "3", args["walltime_seconds"]).items():
        assert args[k] == v


def test_deploy_reads_the_job_id_under_either_key(tmp_path):
    d = job_dir_for(tmp_path)
    for doc in ({"job_id": "job_7"}, {"id": "job_7"}):
        c = FakeClient({"deploy": doc})
        assert McpTransport("k", client=c).deploy(d) == "job_7"
    with pytest.raises(C3CommandError):  # mutation: KeyError escapes as a crash, not a retry
        McpTransport("k", client=FakeClient({"deploy": {"nothing": 1}})).deploy(d)


def test_status_reads_the_top_level_status_upper_cased():
    c = FakeClient({"get_job": {"status": "running", "current_activity":
                                {"event_type": "JOB_SUCCEEDED"}}})
    # mutation: reading current_activity.event_type reports SUCCEEDED for a running job
    assert McpTransport("k", client=c).status("job_1") == "RUNNING"
    assert c.calls[0] == ("get_job", {"job_id": "job_1"})


def test_cancel_is_best_effort():
    def boom(args):
        raise C3CommandError("already terminal")

    # mutation: letting this raise turns a stop request into a crash
    McpTransport("k", client=FakeClient({"cancel_job": boom})).cancel("job_1")


def test_fetch_writes_inline_content_and_checks_its_sha256(tmp_path):
    body = '{"compile": {"ok": true}}'
    doc = {"inline": True, "encoding": "utf8", "content": body, "size_bytes": len(body),
           "sha256": hashlib.sha256(body.encode()).hexdigest()}
    c = FakeClient({"read_artifact": doc})
    dest = tmp_path / "artifacts" / "results.json"
    assert McpTransport("k", client=c).fetch("job_1", "results.json", dest) is True
    assert dest.read_text(encoding="utf-8") == body
    # mutation: a name sent without the artifacts/ prefix is NOT_FOUND on every job
    assert c.calls[0] == ("read_artifact", {"job_id": "job_1", "path": "artifacts/results.json"})


def test_fetch_rejects_content_that_does_not_match_its_hash(tmp_path):
    doc = {"inline": True, "encoding": "utf8", "content": "truncated",
           "sha256": hashlib.sha256(b"the whole file").hexdigest()}
    t = McpTransport("k", client=FakeClient({"read_artifact": doc}))
    with pytest.raises(C3CommandError):  # mutation: no hash check lets a truncated file score
        t.fetch("job_1", "results.json", tmp_path / "results.json")
    assert not (tmp_path / "results.json").exists()


def test_fetch_downloads_when_the_artifact_is_too_big_for_inline(tmp_path):
    body = b"x" * 2_000_000
    doc = {"inline": False, "size_bytes": len(body), "download_url": "https://storage/x",
           "sha256": hashlib.sha256(body).hexdigest()}
    seen = {}

    def fetch_url(url, timeout_s):
        seen["url"] = url
        return body

    t = McpTransport("k", client=FakeClient({"read_artifact": doc}), fetch_url=fetch_url)
    dest = tmp_path / "build.log"
    assert t.fetch("job_1", "build.log", dest) is True
    # mutation: inline-only fetch silently truncates a build log over 1 MiB
    assert seen["url"] == "https://storage/x" and dest.read_bytes() == body


def test_fetch_decodes_base64_content(tmp_path):
    import base64
    raw = b"\x00\x01binary"
    doc = {"inline": True, "encoding": "base64", "content": base64.b64encode(raw).decode(),
           "sha256": hashlib.sha256(raw).hexdigest()}
    t = McpTransport("k", client=FakeClient({"read_artifact": doc}))
    dest = tmp_path / "blob"
    assert t.fetch("job_1", "blob", dest) is True
    # mutation: writing the base64 text corrupts the file
    assert dest.read_bytes() == raw


def test_fetch_returns_false_for_a_missing_artifact_but_raises_on_other_errors(tmp_path):
    def missing(args):
        raise C3CommandError("c3 read_artifact failed: NOT_FOUND artifact not found: x")

    def auth(args):
        raise McpAuthError("the C3 API key was rejected")

    def server_error(args):
        raise C3CommandError("c3 read_artifact failed with HTTP 500")

    t = McpTransport("k", client=FakeClient({"read_artifact": missing}))
    # mutation: raising here turns "job produced no results" into an infrastructure error
    assert t.fetch("job_1", "results.json", tmp_path / "r.json") is False
    t2 = McpTransport("k", client=FakeClient({"read_artifact": auth}))
    with pytest.raises(McpAuthError):  # mutation: swallowing every error hides a revoked key
        t2.fetch("job_1", "results.json", tmp_path / "r.json")
    t3 = McpTransport("k", client=FakeClient({"read_artifact": server_error}))
    # mutation: treating every C3CommandError as "not found" hides a server outage as a no-op
    with pytest.raises(C3CommandError):
        t3.fetch("job_1", "results.json", tmp_path / "r.json")


def test_balance_and_whoami_read_the_structured_fields():
    c = FakeClient({"balance": {"balance_gbp": 9.32, "low_balance": True},
                    "whoami": {"email": "x@example.com", "user_id": "u1"}})
    t = McpTransport("k", client=c)
    assert t.balance_gbp() == 9.32  # mutation: parsing text finds no number here
    assert t.whoami()["user_id"] == "u1"
    c2 = FakeClient({"balance": {"tier": "free"}})
    # mutation: returning 0.0 invents a balance the server never reported
    assert McpTransport("k", client=c2).balance_gbp() is None
