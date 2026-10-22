"""Valency Bond through the official MCP SDK, Streamable HTTP and OAuth PKCE."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import tempfile
import threading
from urllib.parse import parse_qs, urlsplit
import webbrowser

from rsi.hpc.core import locked

ENDPOINT = "https://mcp.valency.io"
DEFAULT_AUTH = Path(__file__).resolve().parents[2] / ".valency" / "oauth.json"
REDIRECT = "http://127.0.0.1:8765/callback"


class LoginRequired(RuntimeError):
    """Interactive consent is required; unattended calls never open a browser."""


class OAuthFiles:
    """SDK TokenStorage protocol; credentials are separate from run evidence."""

    def __init__(self, path):
        self.path = Path(path).resolve()

    def _read(self):
        if not self.path.exists():
            return {"endpoint": ENDPOINT}
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if data.get("endpoint") != ENDPOINT:
            raise ValueError("OAuth credential endpoint binding changed")
        return data

    def _save(self, key, value):
        data = self._read()
        data[key] = value.model_dump(mode="json", exclude_none=True)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd, temporary = tempfile.mkstemp(dir=self.path.parent, prefix=".oauth-")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(data, stream)
            os.replace(temporary, self.path)
        finally:
            Path(temporary).unlink(missing_ok=True)

    async def get_tokens(self):
        from mcp.shared.auth import OAuthToken
        data = self._read().get("tokens")
        return OAuthToken.model_validate(data) if data else None

    async def set_tokens(self, tokens):
        self._save("tokens", tokens)

    async def get_client_info(self):
        from mcp.shared.auth import OAuthClientInformationFull
        data = self._read().get("client")
        return OAuthClientInformationFull.model_validate(data) if data else None

    async def set_client_info(self, client_info):
        self._save("client", client_info)


@asynccontextmanager
async def session(auth_path=DEFAULT_AUTH, *, interactive=False, timeout=60):
    """No sampling, filesystem resources or shell access is granted to the server."""
    import httpx
    from mcp import ClientSession
    from mcp.client.auth import OAuthClientProvider
    from mcp.client.streamable_http import streamable_http_client
    from mcp.shared.auth import OAuthClientMetadata

    storage = OAuthFiles(auth_path)
    if not interactive and not await storage.get_tokens():
        raise LoginRequired("Run python -m rsi.literature login first")
    server = thread = None
    event, received = threading.Event(), {}
    expected_state = None

    async def redirect(url):
        nonlocal expected_state
        if not interactive:
            raise LoginRequired("Valency authorization expired; run the login command again")
        expected_state = parse_qs(urlsplit(url).query).get("state", [None])[0]
        if not expected_state:
            raise RuntimeError("OAuth authorization request omitted state")
        print("Authorize MOFi in your browser. If it does not open, visit:\n" + url, flush=True)
        webbrowser.open(url)

    async def callback():
        # The server thread handles the browser; the async task remains cancellable.
        while not event.is_set():
            await asyncio.sleep(0.1)
        if "error" in received:
            raise LoginRequired("Valency consent was denied")
        return received["code"], received["state"]

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlsplit(self.path)
            query = parse_qs(parsed.query)
            state = query.get("state", [None])[0]
            valid = (parsed.path == "/callback" and expected_state is not None
                     and state == expected_state and not event.is_set()
                     and ("code" in query or "error" in query))
            self.send_response(200 if valid else 400)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(b"Return to the MOFi terminal." if valid else b"Invalid OAuth callback.")
            if valid:
                received.update({key: value[0] for key, value in query.items()})
                event.set()

        def log_message(self, *_):
            pass  # Never log callback authorization codes.

    storage.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    # Serialize token refresh and registration across local clients.
    with locked(storage.path, timeout=2, stale_seconds=600):
        try:
            if interactive:
                server = HTTPServer(("127.0.0.1", 8765), Handler)
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
            auth = OAuthClientProvider(
                server_url=ENDPOINT,
                client_metadata=OAuthClientMetadata(
                    client_name="MOFi literature harness", redirect_uris=[REDIRECT],
                    grant_types=["authorization_code", "refresh_token"], response_types=["code"],
                    token_endpoint_auth_method="none", scope="email profile offline_access"),
                storage=storage, redirect_handler=redirect, callback_handler=callback,
                timeout=timeout)
            async with asyncio.timeout(timeout):
                async with httpx.AsyncClient(auth=auth, timeout=timeout, follow_redirects=False) as http:
                    async with streamable_http_client(ENDPOINT, http_client=http) as streams:
                        async with ClientSession(streams[0], streams[1],
                                                 read_timeout_seconds=timedelta(seconds=timeout)) as remote:
                            await remote.initialize()
                            yield remote
        finally:
            if server:
                server.shutdown()
                server.server_close()
            if thread:
                thread.join(timeout=1)


async def discover(auth_path=DEFAULT_AUTH, *, interactive=False):
    tools, cursor, seen = [], None, set()
    async with session(auth_path, interactive=interactive, timeout=300 if interactive else 60) as remote:
        while True:
            page = await remote.list_tools(cursor=cursor)
            tools.extend(tool.model_dump(mode="json", exclude_none=True) for tool in page.tools)
            cursor = page.nextCursor
            if not cursor:
                break
            if cursor in seen or len(tools) > 1000:
                raise ValueError("Invalid or excessive tool pagination")
            seen.add(cursor)
    names = [tool["name"] for tool in tools]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate remote tool names")
    return {"schema_version": 1, "endpoint": ENDPOINT, "tools": tools}


async def invoke(name, arguments, auth_path, timeout):
    async with session(auth_path, timeout=timeout) as remote:
        result = await remote.call_tool(name, arguments)
        return result.model_dump(mode="json", exclude_none=True)
