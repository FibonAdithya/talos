# C3 MCP Transport Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When a C3 API key is configured, Talos runs every C3 operation over C3's hosted MCP endpoint instead of the `c3` CLI, so no CLI install is needed (which is what blocks Windows users) and `job.sh` is uploaded already marked executable.

**Architecture:** The five `c3` calls in `talos/c3_bench.py` and the two in `talos/cli.py::check_c3` move behind a `C3Transport` protocol with two implementations: `CliTransport` (today's subprocess code, moved unchanged) and `McpTransport` (JSON-RPC over HTTPS with the standard library). `C3Bench` keeps all policy — request hashing, reattach, poll-failure tolerance, pending timeout, cancel, resubmission, cost estimates — and calls the transport for I/O only. A key selects MCP; no key selects the CLI.

**Tech Stack:** Python 3.10+, standard library only (`urllib.request`, `json`, `hashlib`, `base64`), pytest.

**Spec:** `docs/ai/specs/2026-09-17-c3-mcp-transport-design.md` — read it before Task 1; §2 holds the measured wire facts every task depends on.

## Global Constraints

- **Python floor 3.10** (`pyproject.toml: requires-python = ">=3.10"`). No `match`, no `typing.Self`, no 3.11+ syntax. `X | None` is fine (every touched module has `from __future__ import annotations`).
- **No new dependencies.** Standard library only. The MCP Python SDK is out of scope.
- **Line length 100.** `ruff` is configured with `select = ["E4","E7","E9","F"]`, which does **not** include E501, so ruff will not catch a long line. Check with: `python -c "import sys;[print(f'{p}:{i}') for p in sys.argv[1:] for i,l in enumerate(open(p,encoding='utf-8'),1) if len(l.rstrip(chr(10)))>100]" <files>`
- **Text IO:** every `read_text`/`write_text`/`open` in `talos/` must pass `encoding="utf-8"`, and every write `newline="\n"`. `tests/test_portability.py::test_text_io_names_its_encoding_and_line_ending` fails otherwise.
- **Every MCP request** sends `User-Agent: talos/<__version__>`. Without it Cloudflare returns `403 error 1010` (MEASURED, spec §2). `talos/__init__.py::__version__` is `"0.1.0"`.
- **Secrets:** the API key must never appear in an exception message, a log line, a timeline event, `state.json`, or a pending record. `talos/bench.py::_redact` is the one place that scrubs message text.
- **Invariant 1 (AGENTS.md):** baseline and candidate must be scored on the same hardware class, fuel and nonces. The hardware profile, image and walltime sent to MCP must equal what `.c3` names. Task 3 enforces this with one shared function.
- **No test may touch the network or the real `c3` binary.** Inject `post` (MCP) or `run` (CLI). Per the project's C3 test-safety note, stub the transport before mutation-checking anything that could deploy.
- **The gate** is `make check`, which runs ruff, pytest and `agentify check .`. `agentify` needs Python 3.11+, but the repo `.venv` is 3.10, so run the gate from a scratch venv:
  `uv venv --python 3.12 /tmp/v312 && uv pip install --python /tmp/v312/bin/python -r requirements-dev.txt -e . && make check PYTHON=/tmp/v312/bin/python`
- **Mutation-check every new test** before committing its task: break the code under test, confirm that test fails, restore, confirm it passes. Run with `PYTHONDONTWRITEBYTECODE=1` and `-p no:cacheprovider`; stale bytecode has produced false "MISSED" results here before.
- **Baseline:** `369 passed, 2 deselected` at `main` (87bc891), MEASURED 2026-09-17.

---

## File Structure

| File | Responsibility |
|---|---|
| `talos/c3_mcp.py` | **create.** `McpClient` (JSON-RPC/SSE over HTTPS, auth, errors) and `McpTransport` (the six transport methods). |
| `talos/c3_transport.py` | **create.** `C3Transport` protocol, `CliTransport` (moved from `C3Bench`), `make_transport(api_key, run)`. |
| `talos/c3_bench.py` | **modify.** Delete `_c3`, `_deploy`, `_status`, `_cancel`, `_pull`; call the transport instead. Keep every policy method. |
| `talos/c3_jobdir.py` | **modify.** Add `job_settings(...)`, the single source of the `.c3` values; `c3_config_text` renders it. |
| `talos/cli.py` | **modify.** `check_c3` goes through a transport; fix the CLI-missing message. |
| `tests/test_c3_mcp.py` | **create.** `McpClient` and `McpTransport` unit tests with an injected `post`, plus one local `http.server` test. |
| `tests/test_c3_transport.py` | **create.** Transport selection and `CliTransport` behaviour. |
| `tests/test_c3_bench.py` | **modify.** Existing `FakeC3` tests keep working through `CliTransport`; add a fake transport used for the policy tests. |
| `tests/test_c3_jobdir.py` | **modify.** `job_settings` matches the rendered `.c3`. |
| `tests/test_cli.py` | **modify.** `check_c3` over both transports; message wording. |
| `tests/test_live.py` | **modify.** The live C3 test runs over MCP when a key is present. |
| `README.md` | **modify.** C3 row: with a key nothing to install; without one, the CLI. |

---

### Task 1: `McpClient` — one authenticated JSON-RPC call

**Files:**
- Create: `talos/c3_mcp.py`
- Create: `tests/test_c3_mcp.py`

**Interfaces:**
- Consumes: `talos.c3_bench.C3CommandError`, `talos.bench._redact`, `talos.__version__`.
- Produces:
  - `class McpAuthError(C3CommandError)` — bad or revoked key (HTTP 401/403).
  - `McpClient(api_key: str, url: str = "https://api.cthree.cloud/mcp", post: Callable | None = None, timeout_s: int = 60)`
  - `McpClient.tool(name: str, arguments: dict) -> dict` — returns the tool's result document (`structuredContent`, else parsed `content[].text`). Raises `C3CommandError` on any failure.
  - The injected `post(url: str, headers: dict, body: bytes, timeout_s: int) -> tuple[int, dict, bytes]` (status, headers, body).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_c3_mcp.py
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
```

- [ ] **Step 2: Run the tests and watch them fail**

Run: `/tmp/v312/bin/python -m pytest -q -p no:cacheprovider tests/test_c3_mcp.py`
Expected: collection error, `ModuleNotFoundError: No module named 'talos.c3_mcp'`.

- [ ] **Step 3: Write `talos/c3_mcp.py` (client only)**

```python
"""C3 over its hosted MCP endpoint. One JSON-RPC client plus the transport C3Bench calls.

Wire facts measured 2026-09-17 (spec docs/ai/specs/2026-09-17-c3-mcp-transport-design.md §2):
Cloudflare rejects the default urllib User-Agent with 403 error 1010, every request including
`initialize` needs the key, responses are application/json, and no session id is issued.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Callable

from talos import __version__
from talos.bench import _redact
from talos.c3_bench import C3CommandError

MCP_URL = "https://api.cthree.cloud/mcp"
PROTOCOL_VERSION = "2025-06-18"
_KEY_RE = re.compile(r"c3_key_[A-Za-z0-9_-]+")


def _scrub(text: str) -> str:
    """No key and no rand hash in any message this module raises."""
    return _KEY_RE.sub("c3_key_<redacted>", _redact(text))


class McpAuthError(C3CommandError):
    """The key was rejected (401/403). Setup reports this as a key problem, not an outage."""


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
                                 f"{_scrub(str(e))[:200]}") from None
        if status in (401, 403):
            raise McpAuthError("the C3 API key was rejected (check `c3 apikey list`)")
        if status >= 400:
            raise C3CommandError(f"c3 {method} failed with HTTP {status}")
        self._session = headers.get("Mcp-Session-Id") or self._session
        if notify:
            return None
        return self._result(method, headers.get("Content-Type", ""), raw, self._id)

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
        doc = next((d for d in docs if d.get("id") == id_), docs[-1] if docs else None)
        if doc is None:
            raise C3CommandError(f"c3 {method} returned no reply for this request")
        if doc.get("error"):
            msg = str(doc["error"].get("message", doc["error"]))
            raise C3CommandError(f"c3 {method} failed: {_scrub(msg)[:300]}")
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
        res = self._send("tools/call", {"name": name, "arguments": arguments}) or {}
        text = " ".join(c.get("text", "") for c in res.get("content", [])
                        if isinstance(c, dict))
        if res.get("isError"):
            raise C3CommandError(f"c3 {name} failed: {_scrub(text)[:300]}")
        if res.get("structuredContent") is not None:
            return res["structuredContent"]
        try:
            return json.loads(text)
        except ValueError:
            return {"text": text}
```

- [ ] **Step 4: Run the tests and watch them pass**

Run: `/tmp/v312/bin/python -m pytest -q -p no:cacheprovider tests/test_c3_mcp.py`
Expected: 9 passed.

- [ ] **Step 5: Mutation-check every new test**

For each, edit `talos/c3_mcp.py`, run the one test, confirm it fails, restore:

| Mutation | Test that must fail |
|---|---|
| drop `"User-Agent"` from `_headers` | `test_tool_call_sends_auth_and_user_agent...` |
| set `self._ready = False` at the end of `_ensure_ready` | `test_initialize_runs_once_across_calls` |
| drop the `Mcp-Session-Id` line from `_headers` | `test_session_id_is_echoed...` |
| in `_result`, use `docs[0]` instead of matching `id_` | `test_event_stream_response_is_parsed_by_request_id` |
| in `tool`, return `res["structuredContent"]` unconditionally | `test_content_text_json_is_used...` |
| in `tool`, delete the `if res.get("isError")` branch | `test_tool_error_and_rpc_error_raise_c3commanderror` |
| raise `C3CommandError(text)` instead of `McpAuthError(...)` for 401/403 | both `test_auth_failures_...` cases |
| in `_send`'s `except`, use `str(e)` instead of `_scrub(str(e))` | `test_transport_exception_becomes_c3commanderror_without_the_key` |

- [ ] **Step 6: Commit**

```bash
git add talos/c3_mcp.py tests/test_c3_mcp.py
git commit -m "feat: an MCP JSON-RPC client for C3, keyed and redacted"
```

---

### Task 2: `C3Transport` protocol and `CliTransport`

Pure refactor: the CLI code moves out of `C3Bench` behind an interface, with behaviour unchanged. `tests/test_c3_bench.py` must pass untouched at the end of this task — that is the regression proof.

**Files:**
- Create: `talos/c3_transport.py`
- Create: `tests/test_c3_transport.py`
- Modify: `talos/c3_bench.py` (remove `_c3`, `_deploy`, `_status`, `_cancel`, `_pull`; add `self._t`)

**Interfaces:**
- Produces:
  - `class C3Transport(Protocol)` with `whoami() -> dict`, `balance_gbp() -> float | None`, `deploy(job_dir: Path) -> str`, `status(job_id: str) -> str`, `cancel(job_id: str) -> None`, `fetch(job_id: str, name: str, dest: Path) -> bool`.
  - `CliTransport(run=subprocess.run, api_key: str | None = None)` — same behaviour as today's `C3Bench._c3` and friends, including `c3_env`, the 600 s default timeout, the 60 s timeouts for `squeue`/`cancel`, `parse_json_stdout`, and the two-attempt pull.
  - `make_transport(api_key: str | None, run=subprocess.run) -> C3Transport` — `McpTransport` when `api_key` is set (added in Task 4; until then `CliTransport`), else `CliTransport`.
  - `CliTransport.fetch(job_id, name, dest)` runs `c3 pull <job_id> --json` once per call, finds `name` under the pulled directory (`artifacts/` first, then the directory itself), copies it to `dest`, and returns `False` when absent.
- Consumes: `talos.c3_bench.C3CommandError`, `parse_json_stdout`, `c3_env`, `talos.executables.argv0`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_c3_transport.py
import json
import types
from pathlib import Path

import pytest

from talos.c3_bench import C3CommandError
from talos.c3_transport import CliTransport, make_transport


def runner(script):
    """script: (cmd_word) -> (returncode, stdout). Records calls."""
    calls = []

    def run(cmd, **kw):
        calls.append((cmd, kw))
        rc, out = script(cmd[1], kw)
        return types.SimpleNamespace(returncode=rc, stdout=out, stderr="" if rc == 0 else out)
    return run, calls


def test_cli_transport_deploy_returns_the_job_id_from_the_job_dir(tmp_path):
    run, calls = runner(lambda word, kw: (0, "Warning: experimental\n" + json.dumps({"id": "job_9"})))
    assert CliTransport(run=run).deploy(tmp_path) == "job_9"
    # mutation: deploying outside the job dir uploads the wrong workspace
    assert calls[0][1]["cwd"] == str(tmp_path)
    assert calls[0][0][:2] == ["c3", "deploy"]


def test_cli_transport_status_picks_this_job_and_raises_when_absent():
    rows = [{"job_id": "job_1", "status": "running"}, {"job_id": "job_2", "status": "PENDING"}]
    run, _ = runner(lambda word, kw: (0, json.dumps(rows)))
    # mutation: returning the first row reports another job's status; not upper-casing breaks
    # the ACTIVE/TERMINAL comparisons in C3Bench._wait
    assert CliTransport(run=run).status("job_1") == "RUNNING"
    with pytest.raises(C3CommandError):
        CliTransport(run=run).status("job_absent")


def test_cli_transport_fetch_copies_the_named_artifact_and_reports_absence(tmp_path):
    def script(word, kw):
        d = Path(kw["cwd"]) / "job_1" / "artifacts"
        d.mkdir(parents=True, exist_ok=True)
        (d / "results.json").write_text('{"compile": {}}', encoding="utf-8", newline="\n")
        return 0, json.dumps({"jobs": [{"job_id": "job_1", "directory": str(d.parent)}]})

    run, _ = runner(script)
    t = CliTransport(run=run)
    dest = tmp_path / "out" / "results.json"
    assert t.fetch("job_1", "results.json", dest) is True
    assert json.loads(dest.read_text(encoding="utf-8")) == {"compile": {}}
    # mutation: returning True for a file the job never wrote makes _collect read a stale path
    assert t.fetch("job_1", "build.log", tmp_path / "out" / "build.log") is False


def test_cli_transport_balance_parses_the_printed_amount_and_survives_garbage():
    run, _ = runner(lambda word, kw: (0, "Credit balance: £12.34 (free tier)"))
    assert CliTransport(run=run).balance_gbp() == 12.34
    run, _ = runner(lambda word, kw: (0, "no balance here"))
    # mutation: returning 0.0 invents a number the CLI never printed
    assert CliTransport(run=run).balance_gbp() is None


def test_cli_transport_missing_binary_is_a_c3commanderror():
    def run(cmd, **kw):
        raise FileNotFoundError(2, "No such file or directory: 'c3'")

    with pytest.raises(C3CommandError):
        CliTransport(run=run).whoami()


def test_make_transport_uses_the_cli_without_a_key_and_mcp_with_one():
    from talos.c3_mcp import McpTransport
    assert isinstance(make_transport(None), CliTransport)
    # mutation: ignoring the key keeps the CLI requirement on Windows
    assert isinstance(make_transport("c3_key_" + "a" * 20), McpTransport)
```

- [ ] **Step 2: Run them and watch them fail**

Run: `/tmp/v312/bin/python -m pytest -q -p no:cacheprovider tests/test_c3_transport.py`
Expected: `ModuleNotFoundError: No module named 'talos.c3_transport'`. The `make_transport` test also fails on `McpTransport` until Task 4; leave it failing and note it — Task 4 Step 4 is where it must pass. (If you prefer a green suite at every commit, mark only that one test `@pytest.mark.xfail(reason="McpTransport lands in Task 4", strict=True)` and remove the marker in Task 4.)

- [ ] **Step 3: Write `talos/c3_transport.py`**

Move the bodies of `C3Bench._c3`, `_deploy`, `_status`, `_cancel` and `_pull` here unchanged except for the signature changes below. Take `c3_env` and `parse_json_stdout` from `talos.c3_bench` (import them; do not copy them).

```python
"""How Talos reaches C3: the `c3` CLI, or the hosted MCP endpoint when a key is configured."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Protocol

from talos.c3_bench import C3CommandError, c3_env, parse_json_stdout
from talos.executables import argv0


class C3Transport(Protocol):
    def whoami(self) -> dict: ...
    def balance_gbp(self) -> float | None: ...
    def deploy(self, job_dir: Path) -> str: ...
    def status(self, job_id: str) -> str: ...
    def cancel(self, job_id: str) -> None: ...
    def fetch(self, job_id: str, name: str, dest: Path) -> bool: ...


class CliTransport:
    """Subprocess `c3`. Uses the `c3 login` session, or C3_API_KEY when a key is given."""

    name = "cli"

    def __init__(self, run=subprocess.run, api_key: str | None = None):
        self._run = run
        self._env = c3_env(api_key)

    def _c3(self, *args: str, cwd: Path | None = None, timeout: int = 600) -> str:
        # body moved verbatim from C3Bench._c3, with argv0("c3") as argv[0]
        ...

    def whoami(self) -> dict:
        return {"text": self._c3("whoami", timeout=60)}

    def balance_gbp(self) -> float | None:
        import re
        m = re.search(r"Credit balance:\s*£([0-9.]+)", self._c3("balance", timeout=60))
        return float(m.group(1)) if m else None

    def deploy(self, job_dir: Path) -> str:
        return parse_json_stdout(self._c3("deploy", "--json", cwd=job_dir))["id"]

    def status(self, job_id: str) -> str:
        for row in parse_json_stdout(self._c3("squeue", "--json", timeout=60)):
            if row.get("job_id") == job_id:
                return str(row.get("status", "UNKNOWN")).upper()
        raise C3CommandError(f"job {job_id} not listed by squeue")

    def cancel(self, job_id: str) -> None:
        try:
            self._c3("cancel", job_id, timeout=60)
        except C3CommandError:
            pass  # best effort: the job may already be terminal

    def fetch(self, job_id: str, name: str, dest: Path) -> bool:
        """`c3 pull` writes into cwd; copy the one file out. Keeps today's two-attempt pull."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        pull_root = dest.parent
        for attempt in range(2):
            try:
                doc = parse_json_stdout(self._c3("pull", job_id, "--json", cwd=pull_root))
            except (C3CommandError, ValueError):
                if attempt == 1:
                    raise
                continue
            jobs = doc.get("jobs") or []
            d = Path(jobs[0]["directory"]) if jobs and jobs[0].get("directory") else pull_root / job_id
            if not d.is_absolute():
                d = pull_root / d
            for cand in (d / "artifacts" / name, d / name):
                if cand.exists():
                    if cand != dest:
                        shutil.copy2(cand, dest)
                    return True
            return False
        return False


def make_transport(api_key: str | None, run=subprocess.run) -> C3Transport:
    if api_key:
        from talos.c3_mcp import McpTransport
        return McpTransport(api_key)
    return CliTransport(run=run)
```

- [ ] **Step 4: Rewire `C3Bench` to the transport**

In `talos/c3_bench.py`: delete `_c3`, `_deploy`, `_status`, `_cancel`, `_pull`. Add to `__init__`:

```python
        from talos.c3_transport import CliTransport, make_transport
        self._t = transport or (CliTransport(run=run, api_key=api_key) if run is not subprocess.run
                                else make_transport(api_key, run=run))
```

with a new keyword-only parameter `transport: C3Transport | None = None`. An injected `run` keeps meaning "drive the CLI", so every existing `FakeC3` test keeps working; a caller with a key and no injected `run` gets MCP.

Replace the call sites:

| was | becomes |
|---|---|
| `self._deploy(job_dir)` | `self._t.deploy(job_dir)`, wrapped in the same `try` that raises `BenchUnavailable(f"C3 deploy failed: …")` |
| `self._status(job_id)` | `self._t.status(job_id)` |
| `self._cancel(job_id)` | `self._t.cancel(job_id)` |
| `artifacts = self._pull(job_id, job_dir)` | the block below |

```python
        artifacts = job_dir / job_id / "artifacts"
        try:
            have_results = self._t.fetch(job_id, "results.json", artifacts / "results.json")
            self._t.fetch(job_id, "build.log", artifacts / "build.log")
        except (C3CommandError, ValueError, KeyError, IndexError) as e:
            raise BenchUnavailable(
                f"C3 pull failed for {job_id}: {_redact(str(e))[:300]}") from None
        results = artifacts / "results.json"
        if not have_results:
            ...  # the existing not-exists branch, unchanged
```

- [ ] **Step 5: Run the C3 suites**

Run: `/tmp/v312/bin/python -m pytest -q -p no:cacheprovider tests/test_c3_bench.py tests/test_c3_transport.py tests/test_cli.py`
Expected: `test_c3_bench.py` and `test_cli.py` fully pass (unchanged files — this is the refactor's regression proof); in `test_c3_transport.py` only `test_make_transport_...` fails, on `McpTransport`.

- [ ] **Step 6: Mutation-check the new tests**

| Mutation | Test that must fail |
|---|---|
| in `CliTransport.deploy`, drop `cwd=job_dir` | `test_cli_transport_deploy_returns_the_job_id_from_the_job_dir` |
| in `status`, `return rows[0]["status"].upper()` | `test_cli_transport_status_picks_this_job...` |
| in `status`, drop `.upper()` | same test |
| in `fetch`, `return True` after the loop | `test_cli_transport_fetch_copies_the_named_artifact...` |
| in `balance_gbp`, `return 0.0` when the regex misses | `test_cli_transport_balance_parses...` |
| in `_c3`, drop the `except OSError` branch | `test_cli_transport_missing_binary_is_a_c3commanderror` |

- [ ] **Step 7: Commit**

```bash
git add talos/c3_transport.py talos/c3_bench.py tests/test_c3_transport.py
git commit -m "refactor: put the c3 CLI behind a C3Transport seam"
```

---

### Task 3: One source for the `.c3` settings

`.c3` and the MCP `deploy` arguments must never disagree: a different hardware profile on one path would score baseline and candidate on different hardware (invariant 1).

**Files:**
- Modify: `talos/c3_jobdir.py` (add `job_settings`, render `c3_config_text` from it)
- Modify: `tests/test_c3_jobdir.py`

**Interfaces:**
- Produces: `job_settings(challenge: str, purpose: str, seconds: int) -> dict` with exactly the keys `project`, `job_name`, `script`, `hardware`, `walltime_seconds`, `docker_image`, `docker_requires_accelerator`. `walltime_seconds` is an `int`; the others are `str`. These are the MCP `deploy` argument names (MEASURED from `tools/list`, spec §8 Q3).
- `c3_config_text(challenge, purpose, seconds)` keeps its signature and output, rendered from `job_settings`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_c3_jobdir.py
def test_job_settings_and_the_c3_file_agree():
    from talos import c3_jobdir
    s = c3_jobdir.job_settings("knapsack", "3", 1380)
    text = c3_jobdir.c3_config_text("knapsack", "3", 1380)
    assert set(s) == {"project", "job_name", "script", "hardware", "walltime_seconds",
                      "docker_image", "docker_requires_accelerator"}
    # mutation: a hardware profile hard-coded on either path scores the baseline and the
    # candidate on different hardware (AGENTS.md invariant 1)
    assert f"hardware: {s['hardware']}\n" in text
    assert f"  image: {s['docker_image']}\n" in text
    assert f"  requires_accelerator: {s['docker_requires_accelerator']}\n" in text
    assert f"job_name: {s['job_name']}\n" in text and f"script: {s['script']}\n" in text
    assert f"project: {s['project']}\n" in text
    assert s["walltime_seconds"] == 1380 and 'time: "00:23:00"' in text
    g = c3_jobdir.job_settings("hypergraph", "1", 60)
    assert g["docker_requires_accelerator"] == "cuda" and g["hardware"] == "l40"
```

- [ ] **Step 2: Run it and watch it fail**

Run: `/tmp/v312/bin/python -m pytest -q -p no:cacheprovider tests/test_c3_jobdir.py -k job_settings`
Expected: FAIL, `AttributeError: module 'talos.c3_jobdir' has no attribute 'job_settings'`.

- [ ] **Step 3: Implement**

```python
def job_settings(challenge: str, purpose: str, seconds: int) -> dict:
    """The job's settings, named as C3's MCP `deploy` tool names them. `.c3` renders these same
    values, so the two submission paths cannot disagree about hardware, image or time limit."""
    spec = CHALLENGES[challenge]
    return {"project": "talos", "job_name": f"talos-{challenge}-{purpose}", "script": "job.sh",
            "hardware": c3_profile(spec), "walltime_seconds": seconds,
            "docker_image": c3_image(challenge),
            "docker_requires_accelerator": "cuda" if spec.is_gpu else "none"}


def c3_config_text(challenge: str, purpose: str, seconds: int) -> str:
    s = job_settings(challenge, purpose, seconds)
    return (f"project: {s['project']}\njob_name: {s['job_name']}\nscript: {s['script']}\n"
            f"hardware: {s['hardware']}\ntime: \"{hhmmss(s['walltime_seconds'])}\"\n"
            f"docker:\n  image: {s['docker_image']}\n"
            f"  requires_accelerator: {s['docker_requires_accelerator']}\n")
```

- [ ] **Step 4: Run the jobdir suite**

Run: `/tmp/v312/bin/python -m pytest -q -p no:cacheprovider tests/test_c3_jobdir.py`
Expected: all pass, including the existing exact-text test `test_c3_config_text_is_exact` (byte-for-byte output must not change).

- [ ] **Step 5: Mutation-check**

| Mutation | Test that must fail |
|---|---|
| `"hardware": "l40"` hard-coded in `job_settings` | `test_job_settings_and_the_c3_file_agree` (and the exact-text test) |
| `"docker_requires_accelerator": "none"` always | same test, GPU half |

- [ ] **Step 6: Commit**

```bash
git add talos/c3_jobdir.py tests/test_c3_jobdir.py
git commit -m "refactor: one source for the .c3 settings and the MCP deploy arguments"
```

---

### Task 4: `McpTransport`

**Files:**
- Modify: `talos/c3_mcp.py` (add `McpTransport`)
- Modify: `tests/test_c3_mcp.py`
- Modify: `tests/test_c3_transport.py` (remove the xfail marker if you added one)

**Interfaces:**
- Produces: `McpTransport(api_key: str, client: McpClient | None = None, fetch_url: Callable | None = None)` implementing `C3Transport`. `fetch_url(url: str, timeout_s: int) -> bytes` is the download-link fetcher, injected for tests.
- Consumes: `talos.c3_jobdir.job_settings`, `talos.c3_jobdir.JOB_MODULES` layout on disk (it reads the job dir `write_job_dir` produced).

- [ ] **Step 1: Write the failing tests**

```python
# appended to tests/test_c3_mcp.py
import hashlib

from talos.c3_jobdir import job_settings, write_job_dir
from talos.c3_mcp import McpTransport


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

    t = McpTransport("k", client=FakeClient({"read_artifact": missing}))
    # mutation: raising here turns "job produced no results" into an infrastructure error
    assert t.fetch("job_1", "results.json", tmp_path / "r.json") is False
    t2 = McpTransport("k", client=FakeClient({"read_artifact": auth}))
    with pytest.raises(McpAuthError):  # mutation: swallowing every error hides a revoked key
        t2.fetch("job_1", "results.json", tmp_path / "r.json")


def test_balance_and_whoami_read_the_structured_fields():
    c = FakeClient({"balance": {"balance_gbp": 9.32, "low_balance": True},
                    "whoami": {"email": "x@example.com", "user_id": "u1"}})
    t = McpTransport("k", client=c)
    assert t.balance_gbp() == 9.32  # mutation: parsing text finds no number here
    assert t.whoami()["user_id"] == "u1"
    c2 = FakeClient({"balance": {"tier": "free"}})
    # mutation: returning 0.0 invents a balance the server never reported
    assert McpTransport("k", client=c2).balance_gbp() is None
```

- [ ] **Step 2: Run them and watch them fail**

Run: `/tmp/v312/bin/python -m pytest -q -p no:cacheprovider tests/test_c3_mcp.py`
Expected: the new tests fail with `ImportError: cannot import name 'McpTransport'`.

- [ ] **Step 3: Implement `McpTransport`**

```python
class McpTransport:
    """C3 over the hosted MCP endpoint. No `c3` binary is needed."""

    name = "mcp"

    def __init__(self, api_key: str, client: McpClient | None = None,
                 fetch_url: Callable | None = None, timeout_s: int = 60):
        self._c = client or McpClient(api_key, timeout_s=timeout_s)
        self._fetch_url = fetch_url or _download
        self._timeout_s = timeout_s

    def whoami(self) -> dict:
        return self._c.tool("whoami", {})

    def balance_gbp(self) -> float | None:
        v = self._c.tool("balance", {}).get("balance_gbp")
        return float(v) if isinstance(v, (int, float)) else None

    def deploy(self, job_dir: Path) -> str:
        from talos.c3_jobdir import job_settings
        job_dir = Path(job_dir)
        files = []
        for p in sorted(job_dir.rglob("*")):
            if not p.is_file() or p.name == ".c3":
                continue
            rel = p.relative_to(job_dir).as_posix()
            entry = {"path": rel, "content": p.read_text(encoding="utf-8")}
            if rel == "job.sh":
                entry["executable"] = True  # mode 0755; Windows cannot set it on disk
            files.append(entry)
        challenge, purpose, seconds = _job_dir_settings_inputs(job_dir)
        args = {**job_settings(challenge, purpose, seconds), "files": files}
        doc = self._c.tool("deploy", args)
        job_id = doc.get("job_id") or doc.get("id")
        if not job_id:
            raise C3CommandError("c3 deploy returned no job id")
        return str(job_id)

    def status(self, job_id: str) -> str:
        doc = self._c.tool("get_job", {"job_id": job_id})
        st = doc.get("status")
        if not st:
            raise C3CommandError(f"c3 get_job returned no status for {job_id}")
        return str(st).upper()

    def cancel(self, job_id: str) -> None:
        try:
            self._c.tool("cancel_job", {"job_id": job_id})
        except C3CommandError:
            pass  # best effort: the job may already be terminal

    def fetch(self, job_id: str, name: str, dest: Path) -> bool:
        path = name if "/" in name else f"artifacts/{name}"
        try:
            doc = self._c.tool("read_artifact", {"job_id": job_id, "path": path})
        except McpAuthError:
            raise
        except C3CommandError as e:
            if "NOT_FOUND" in str(e):
                return False
            raise
        if doc.get("inline"):
            content = doc.get("content") or ""
            raw = (base64.b64decode(content) if doc.get("encoding") == "base64"
                   else content.encode("utf-8"))
        else:
            url = doc.get("download_url")
            if not url:
                raise C3CommandError(f"c3 read_artifact gave neither content nor a URL for {path}")
            raw = self._fetch_url(url, self._timeout_s)
        want = doc.get("sha256")
        if want and hashlib.sha256(raw).hexdigest() != want:
            # the GPU-box lesson: a matching hash is the only proof of a complete transfer
            raise C3CommandError(f"c3 artifact {path} did not match its sha256")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)
        return True
```

Add the two helpers:

```python
def _download(url: str, timeout_s: int) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": f"talos/{__version__}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            return r.read()
    except Exception as e:
        raise C3CommandError(f"artifact download failed: {_scrub(str(e))[:200]}") from None


def _job_dir_settings_inputs(job_dir: Path) -> tuple[str, str, int]:
    """Recover (challenge, purpose, walltime seconds) from the job dir write_job_dir produced:
    the challenge from payload.json, the purpose from the directory name, and the walltime from
    the `.c3` the same function rendered, so both paths use one number."""
    payload = json.loads((job_dir / "payload.json").read_text(encoding="utf-8"))
    text = (job_dir / ".c3").read_text(encoding="utf-8")
    m = re.search(r'^time:\s*"(\d+):(\d\d):(\d\d)"', text, re.M)
    if not m:
        raise C3CommandError("job dir has no time: in .c3")
    h, mi, s = (int(x) for x in m.groups())
    return payload["challenge"], job_dir.name, h * 3600 + mi * 60 + s
```

Also add the imports this needs at the top of the module: `base64`, `hashlib`, `from pathlib import Path`.

- [ ] **Step 4: Run the tests**

Run: `/tmp/v312/bin/python -m pytest -q -p no:cacheprovider tests/test_c3_mcp.py tests/test_c3_transport.py`
Expected: all pass, `test_make_transport_uses_the_cli_without_a_key_and_mcp_with_one` included.

- [ ] **Step 5: Mutation-check**

| Mutation | Test that must fail |
|---|---|
| drop `entry["executable"] = True` | `test_deploy_sends_the_files_with_job_sh_executable...` |
| set `executable` on every file | same test |
| include `.c3` in `files` | same test |
| `"hardware": "l40"` in the `args` dict | same test (settings loop) |
| in `status`, read `doc["current_activity"]["event_type"]` | `test_status_reads_the_top_level_status_upper_cased` |
| in `cancel`, remove the `try` | `test_cancel_is_best_effort` |
| skip the `sha256` comparison in `fetch` | `test_fetch_rejects_content_that_does_not_match_its_hash` |
| in `fetch`, treat every `C3CommandError` as `return False` | `test_fetch_returns_false_for_a_missing_artifact...` |
| in `fetch`, drop the `download_url` branch | `test_fetch_downloads_when_the_artifact_is_too_big_for_inline` |
| in `fetch`, write `content` without base64 decoding | `test_fetch_decodes_base64_content` |
| in `balance_gbp`, `return 0.0` on a missing field | `test_balance_and_whoami_read_the_structured_fields` |

- [ ] **Step 6: Commit**

```bash
git add talos/c3_mcp.py tests/test_c3_mcp.py tests/test_c3_transport.py
git commit -m "feat: McpTransport - deploy, status, cancel and hash-checked artifact fetch"
```

---

### Task 5: One real HTTP round trip, in-process

The `post` function is injected everywhere above, so nothing has yet proved the real `urllib` path builds correct requests. This task pins it against a local server. No network.

**Files:**
- Modify: `tests/test_c3_mcp.py`

- [ ] **Step 1: Write the failing test**

```python
def test_real_urllib_post_round_trip_against_a_local_server():
    """The only test that exercises _urllib_post. Binds 127.0.0.1 on a free port."""
    import json as _json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            seen.append((dict(self.headers), _json.loads(body)))
            doc = _json.loads(body)
            if doc.get("method") == "notifications/initialized":
                self.send_response(202)
                self.end_headers()
                return
            payload = _json.dumps({"jsonrpc": "2.0", "id": doc["id"],
                                   "result": {"structuredContent": {"balance_gbp": 3.5}}})
            out = f"event: message\ndata: {payload}\n\n".encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        client = McpClient(KEY, url=f"http://127.0.0.1:{srv.server_port}/mcp", timeout_s=10)
        assert client.tool("balance", {}) == {"balance_gbp": 3.5}
    finally:
        srv.shutdown()
        srv.server_close()
        t.join(timeout=5)
    headers, _doc = seen[-1]
    # mutation: header names that only the fake honoured (Cloudflare needs the real User-Agent)
    assert headers["Authorization"] == f"Bearer {KEY}"
    assert headers["User-Agent"] == f"talos/{__version__}"
    assert "text/event-stream" in headers["Accept"]
```

- [ ] **Step 2: Run it**

Run: `/tmp/v312/bin/python -m pytest -q -p no:cacheprovider tests/test_c3_mcp.py -k round_trip`
Expected: it passes if `_urllib_post` and the SSE parser are right. If it fails, fix `talos/c3_mcp.py`, not the test — this is the test that catches a header the fake accepted but a real server does not see.

- [ ] **Step 3: Mutation-check**

| Mutation | Test that must fail |
|---|---|
| in `_urllib_post`, pass `headers={}` | this test |
| in `_result`, ignore `text/event-stream` | this test |

- [ ] **Step 4: Commit**

```bash
git add tests/test_c3_mcp.py
git commit -m "test: a real urllib round trip against a local MCP server"
```

---

### Task 6: Setup, messages, README

**Files:**
- Modify: `talos/cli.py` (`check_c3`, the CLI-missing message, `make_bench` unchanged)
- Modify: `tests/test_cli.py`
- Modify: `README.md`

**Interfaces:**
- `check_c3(run=None, api_key: str | None = None, transport=None) -> float` keeps returning the balance in GBP and keeps printing the low-balance warning. With a key it goes through `McpTransport`; without one through `CliTransport`. `transport` is for tests.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cli.py
def test_check_c3_with_a_key_uses_mcp_and_never_shells_out(monkeypatch, capsys):
    calls = []

    class T:
        name = "mcp"

        def whoami(self):
            calls.append("whoami")
            return {"user_id": "u1"}

        def balance_gbp(self):
            calls.append("balance")
            return 0.5

    def no_subprocess(*a, **kw):
        raise AssertionError("check_c3 must not run a subprocess when a key is configured")

    monkeypatch.setattr(cli.subprocess, "run", no_subprocess)
    made = {}
    monkeypatch.setattr(cli, "make_transport",
                        lambda api_key, run=None: made.setdefault("key", api_key) or T())
    assert cli.check_c3(api_key="c3_key_" + "a" * 20) == 0.5
    assert calls == ["whoami", "balance"]  # mutation: skipping whoami accepts a revoked key
    assert made["key"] == "c3_key_" + "a" * 20
    # mutation: dropping the warning hides a balance that cannot pay for the next job
    assert "low" in capsys.readouterr().err.lower()


def test_check_c3_reports_a_rejected_key_as_a_key_problem(monkeypatch):
    from talos.c3_mcp import McpAuthError

    class T:
        name = "mcp"

        def whoami(self):
            raise McpAuthError("the C3 API key was rejected")

        def balance_gbp(self):
            return 1.0

    monkeypatch.setattr(cli, "make_transport", lambda api_key, run=None: T())
    with pytest.raises(ConfigError) as ei:
        cli.check_c3(api_key="c3_key_" + "a" * 20)
    msg = str(ei.value)
    # mutation: telling a key user to run `c3 login` sends them down the wrong path
    assert "apikey" in msg and "c3 login" not in msg


def test_check_c3_without_a_key_still_asks_the_cli_to_log_in(monkeypatch):
    class T:
        name = "cli"

        def whoami(self):
            raise C3CommandError("c3 whoami could not be run: [Errno 2] no such file")

        def balance_gbp(self):
            return 1.0

    monkeypatch.setattr(cli, "make_transport", lambda api_key, run=None: T())
    with pytest.raises(ConfigError) as ei:
        cli.check_c3()
    # mutation: a CLI user with no session gets no instruction at all
    assert "c3 login" in str(ei.value)


def test_check_c3_unreadable_balance_warns_and_returns_zero(monkeypatch, capsys):
    class T:
        name = "mcp"

        def whoami(self):
            return {}

        def balance_gbp(self):
            return None

    monkeypatch.setattr(cli, "make_transport", lambda api_key, run=None: T())
    # mutation: returning a number the server never reported ("£0.00 is low")
    assert cli.check_c3(api_key="k") == 0.0
    assert "could not read the C3 balance" in capsys.readouterr().err
```

Keep the existing `test_setup_*` C3 tests working: they drive `cli.subprocess` with a fake and pass no key, so they must continue to exercise `CliTransport`.

- [ ] **Step 2: Run them and watch them fail**

Run: `/tmp/v312/bin/python -m pytest -q -p no:cacheprovider tests/test_cli.py -k check_c3`
Expected: FAIL — `cli` has no attribute `make_transport`.

- [ ] **Step 3: Rewrite `check_c3`**

```python
def check_c3(run=None, api_key: str | None = None, transport=None) -> float:
    """Confirms C3 is reachable and authenticated — over MCP when a key is configured, over the
    `c3` CLI otherwise — and returns the credit balance in GBP. 0.0 means "could not read it"."""
    from talos.c3_mcp import McpAuthError
    t = transport or make_transport(api_key, run=run or subprocess.run)
    try:
        t.whoami()
    except McpAuthError as e:
        raise ConfigError(f"C3 rejected the API key: {e}; check `c3 apikey list` and "
                          f"run `talos setup` again") from None
    except C3CommandError as e:
        fix = ("check the C3 API key (`c3 apikey list`)" if (api_key or os.environ.get("C3_API_KEY"))
               else "install the `c3` CLI and run `c3 login`, or give setup a C3 API key "
                    "(`c3 apikey create`),")
        raise ConfigError(f"C3 login check failed: {e}; {fix} and retry") from None
    try:
        balance = t.balance_gbp()
    except C3CommandError as e:
        print(f"warning: could not read the C3 balance: {e}", file=sys.stderr)
        return 0.0
    if balance is None:
        print("warning: could not read the C3 balance", file=sys.stderr)
        return 0.0
    if balance < 1.0:
        print(f"warning: C3 credit balance is low (£{balance:.2f}); run `c3 topup`",
              file=sys.stderr)
    return balance
```

Import `make_transport` and `C3CommandError` at the top of `talos/cli.py`. Delete the old inner `c3()` helper, the `re.search` over `Credit balance`, and the `ConfigError("the c3 CLI is not on PATH; install it and run `c3 login`")` branch — `CliTransport` now raises `C3CommandError` for a missing binary and the message above covers both cases.

- [ ] **Step 4: Run the CLI suite**

Run: `/tmp/v312/bin/python -m pytest -q -p no:cacheprovider tests/test_cli.py`
Expected: all pass. If an existing setup test asserted the old wording, update that assertion — the new message must still name `c3 login` for a keyless user (that is the test above).

- [ ] **Step 5: Update the README**

In the backend table row for `c3`, replace the "What you need before setup" cell with:

```
With a C3 API key (`c3 apikey create`): nothing to install — Talos talks to C3 over HTTPS. Without a key: the `c3` CLI ([cthree.cloud](https://cthree.cloud)) installed and logged in with `c3 login`. Either way, credit on the account; top up with `c3 topup`.
```

In "Compute backends in detail", add one paragraph:

```
On the C3 backend Talos uses C3's hosted MCP endpoint (`https://api.cthree.cloud/mcp`) when a C3
API key is configured, and the `c3` CLI when it is not. The key path needs no C3 install, which is
what makes the C3 backend usable on Windows, and it uploads `job.sh` already marked executable.
Both paths submit the same job directory, which is still written to `runs/<job_id>/c3/<n>/`.
```

Then run the docs gate, which checks every reference in the authoritative docs resolves:
`/tmp/v312/bin/python -m pytest -q -p no:cacheprovider tests/test_docs_references.py tests/test_contract.py`

- [ ] **Step 6: Mutation-check**

| Mutation | Test that must fail |
|---|---|
| in `check_c3`, skip the `whoami` call | `test_check_c3_with_a_key_uses_mcp_and_never_shells_out` |
| catch `McpAuthError` in the same branch as `C3CommandError` | `test_check_c3_reports_a_rejected_key_as_a_key_problem` |
| `return 0.0` when `balance_gbp()` is `None`, without the warning | `test_check_c3_unreadable_balance_warns_and_returns_zero` |
| make the keyless message say `c3 apikey list` | `test_check_c3_without_a_key_still_asks_the_cli_to_log_in` |

- [ ] **Step 7: Commit**

```bash
git add talos/cli.py tests/test_cli.py README.md
git commit -m "feat: setup checks C3 over MCP when a key is configured"
```

---

### Task 7: Gate, live test, and the spec's status

**Files:**
- Modify: `tests/test_live.py`
- Modify: `docs/ai/specs/2026-09-17-c3-mcp-transport-design.md` (§8 answers only)

- [ ] **Step 1: Point the live C3 test at the configured key**

```python
# tests/test_live.py, inside test_c3_knapsack_job, replacing `b = C3Bench(tmp_path)`
    # With a C3 API key in .talos/secrets.json or C3_API_KEY this runs over MCP; with no key it
    # runs over the `c3` CLI and needs `c3 login`. Print which, so the run's evidence says so.
    from talos.config import load, resolve_c3_api_key
    key = resolve_c3_api_key(load(Path.cwd()))
    b = C3Bench(tmp_path, api_key=key)
    print({"transport": b._t.name, "keyed": bool(key)})
```

Update its docstring: "Needs either a C3 API key or `c3 login`, and about £0.05 of credit; takes about 20 minutes."

- [ ] **Step 2: Run the whole gate**

```bash
make check PYTHON=/tmp/v312/bin/python > /tmp/c3mcp-gate.log 2>&1; echo $?; tail -5 /tmp/c3mcp-gate.log
```

Expected: ruff clean, `agentify` `level 1: PASS`, and pytest at or above `369 passed` plus the new tests (about 30 more cases), `2 deselected`. Read the log file, not the tail alone.

- [ ] **Step 3: Check the line length by hand**

```bash
/tmp/v312/bin/python -c "import sys;[print(f'{p}:{i}') for p in sys.argv[1:] for i,l in enumerate(open(p,encoding='utf-8'),1) if len(l.rstrip(chr(10)))>100]" talos/c3_mcp.py talos/c3_transport.py talos/c3_bench.py talos/c3_jobdir.py talos/cli.py tests/test_c3_mcp.py tests/test_c3_transport.py
```

Expected: no output. ruff does not enforce E501 here.

- [ ] **Step 4: Confirm no key can leak**

```bash
grep -rn "c3_key_" talos/ | grep -v "c3_key_<redacted>\|c3_key_\[A-Za-z0-9\|apikey"
```

Expected: no line that interpolates a key into a message. Then, with a key configured, run the
unit suite and grep the output for `c3_key_`: `/tmp/v312/bin/python -m pytest -q -m "not live" 2>&1 | grep -c c3_key_` should print `0` (the test key `c3_key_aaa…` appears only inside assertions, which do not fire).

- [ ] **Step 5: Run the live job (needs the user's go-ahead — it spends credit)**

```bash
TALOS_LIVE_BACKEND=c3 /tmp/v312/bin/python -m pytest -m live tests/test_live.py -k c3 -s 2>&1 | tee /tmp/c3mcp-live.log
```

This settles spec §8 Q8 and Q9: that a job deployed over MCP runs `job.sh` from an inline upload, that polling at 20 s draws no `429`, and that `cancel_job` works. ESTIMATE: about 12 minutes of `cpu-d3-4vcpu-16gb` and roughly £0.03.

Expected: the printed `{"transport": "mcp", "keyed": true}`, `r.compile.ok`, two training and two holdout results.

If the job fails at `job.sh`, read `c3 logs <job-id>` (or `job_logs`) before changing anything: the likely cause is the `executable` flag not being honoured, in which case the fallback is to prefix the script with a shell (`script: "bash job.sh"` is not valid; instead upload `job.sh` and set `script: "job.sh"` with `executable: true` — if that fails, report it rather than guessing).

- [ ] **Step 6: Record the result in the spec and commit**

Replace §8's Q8 and Q9 rows with what the live run measured, including the job id and the observed cost, and change the status line at the top from "draft" to "implemented <date>, PR #<n>".

```bash
git add tests/test_live.py docs/ai/specs/2026-09-17-c3-mcp-transport-design.md
git commit -m "test: the live C3 job runs over MCP when a key is configured"
```

---

## Self-Review

**1. Spec coverage**

| Spec section | Task |
|---|---|
| §4.1 transport seam, six methods | Task 2 (protocol, `CliTransport`), Task 4 (`McpTransport`) |
| §4.1 `fetch` with sha256 and `NOT_FOUND` → `False` | Task 4 |
| §4.2 key selects MCP | Task 2 (`make_transport`), Task 6 (setup) |
| §4.3 wire details: user agent, auth, initialize once, session id, SSE, `structuredContent`, error mapping, redaction | Task 1, Task 5 |
| §4.4 deploy payload: files, `executable`, settings as arguments, no `.c3` | Task 3 (one source), Task 4 (payload) |
| §4.5 resume across transports | no code needed (job ids are shared, MEASURED); the live test in Task 7 exercises a keyed submit and collect |
| §4.6 setup, messages, README | Task 6 |
| §5 test table | every row appears in a task's tests: transport choice (T2), headers (T1, T5), no key in messages (T1, T7 Step 4), JSON and SSE (T1, T5), `isError` (T1), 401/403 (T1, T6), `executable` (T4), hardware equals `.c3` (T3, T4), inline/URL/missing artifact (T4), over 1 MiB (T4), sha256 (T4), top-level status (T4), policy tests over both transports (T2 Step 5 keeps `test_c3_bench.py` green through `CliTransport`; the keyed path is covered by T4's unit tests and T7's live run) |
| §6 live verification | Task 7 |
| §7 out of scope: no `CliTransport` removal, no OAuth, no `prepare_upload` | respected; nothing in the plan touches them |

**2. Placeholder scan.** One deliberate ellipsis remains: `CliTransport._c3`'s body in Task 2 Step 3, which is "moved verbatim from `C3Bench._c3`" — the source is in the repo at `talos/c3_bench.py`, and copying it is the point. Everything else is literal code.

**3. Type consistency.** `C3Transport`'s six method names and signatures are identical in Task 2 (protocol and `CliTransport`), Task 4 (`McpTransport`), Task 6 (`check_c3`'s fakes) and Task 7 (`b._t.name`). `job_settings`'s seven keys in Task 3 are exactly the `deploy` argument names used in Task 4. `McpClient.tool(name, arguments)` is the only client method the transport calls, and the fakes in Tasks 4 and 6 implement that one signature. `fetch(job_id, name, dest) -> bool` is called with plain names (`"results.json"`, `"build.log"`) in Task 2's `C3Bench` rewiring, and `McpTransport.fetch` adds the `artifacts/` prefix itself.

## Risks

- **`deploy` with inline `files` is unmeasured.** Every other MCP call is measured (spec §2), but not this one, and it is the one that costs money. Task 7 Step 5 is where it is proved. If the tool rejects the payload, the fallback inside this design is `prepare_upload` (spec §7 lists it as out of scope, so that would be a new spec).
- **No output schemas.** C3 publishes no `outputSchema`, so field names (`job_id`, `status`, `balance_gbp`, `inline`, `sha256`, `download_url`) are observed, not contracted. Task 4 accepts `job_id` or `id`, and every reader raises `C3CommandError` rather than `KeyError` when a field is missing, so a C3 change surfaces as a paused run with a clear message.
- **Terminal statuses other than `SUCCEEDED` were never seen over MCP.** `C3Bench._wait` already tolerates an unrecognised status up to `poll_failures_max` and then raises `BenchUnavailable`, so a spelling difference pauses the run instead of scoring wrongly.
