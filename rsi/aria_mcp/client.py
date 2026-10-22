"""Official MCP SDK transport for the pinned, separately installed ARIA server."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from importlib.metadata import distribution, version
import json
import os
from pathlib import Path
import sys
from urllib.parse import urlsplit

from rsi.hpc.core import sha256

REPOSITORY = "https://github.com/yicao-elina/ARIA"
REVISION = "cff9080f57360b1e89509adb3f298824eeed809d"
TOOLS = {"predict", "design", "explain", "diagnose"}
CONTROLLER_FIELDS = {"kg", "model", "mode"}


def installation():
    """Require the admitted VCS install, and fingerprint code/data actually used."""
    package = distribution("aria-materials")
    direct = json.loads(package.read_text("direct_url.json") or "{}")
    url = urlsplit(direct.get("url", ""))
    if (url.hostname != "github.com" or url.path.rstrip("/").removesuffix(".git") != "/yicao-elina/ARIA"
            or direct.get("vcs_info", {}).get("commit_id") != REVISION):
        raise ValueError("Install ARIA at the revision pinned in pyproject.toml: uv sync --extra aria")
    root = Path(package.locate_file("aria")).resolve()
    if not (root / "mcp/server.py").is_file():
        raise ValueError("Installed ARIA MCP server is missing")
    files = {str(path.relative_to(root)): sha256(path)
             for path in sorted(root.rglob("*")) if path.is_file() and path.suffix in {".py", ".json"}}
    return root, {"repository": REPOSITORY, "revision": REVISION, "version": package.version,
                  "source_files": files, "mcp_version": version("mcp"),
                  "python": str(Path(sys.executable).resolve())}


async def exchange(action, *, timeout, stderr_path, tool=None, arguments=None):
    """One fresh server per operation. Cancellation invokes SDK stdio cleanup.

    The deadline bounds waiting for MCP, not work inside an external Ollama daemon.
    SDK process shutdown can take additional time after the deadline.
    """
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    env = {"PYTHONUTF8": "1", "TOKENIZERS_PARALLELISM": "false",
           "OLLAMA_HOST": "http://127.0.0.1:11434",
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    # The MCP SDK inherits only its documented basic process environment. Do not
    # forward API keys, Valency credentials or arbitrary caller environment values.
    for key in ("HF_HOME", "HF_HUB_CACHE", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "CUDA_VISIBLE_DEVICES"):
        if key in os.environ:
            env[key] = os.environ[key]
    params = StdioServerParameters(command=sys.executable, args=["-m", "aria.mcp.server"],
                                   env=env, cwd=str(Path(stderr_path).parent))
    stderr_path = Path(stderr_path)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stderr_path.open("x", encoding="utf-8") as errlog:
        async with asyncio.timeout(timeout):
            async with stdio_client(params, errlog=errlog) as streams:
                async with ClientSession(*streams, read_timeout_seconds=timedelta(seconds=timeout)) as remote:
                    initialized = await remote.initialize()
                    if action == "discover":
                        tools, cursor, seen = [], None, set()
                        while True:
                            page = await remote.list_tools(cursor=cursor)
                            tools.extend(item.model_dump(mode="json", exclude_none=True) for item in page.tools)
                            if len(tools) > 64:
                                raise ValueError("Excessive ARIA tool catalog")
                            cursor = page.nextCursor
                            if not cursor:
                                break
                            if cursor in seen:
                                raise ValueError("Repeated tool catalog cursor")
                            seen.add(cursor)
                        return {"server_info": initialized.serverInfo.model_dump(mode="json"), "tools": tools}
                    if action != "call" or tool not in TOOLS:
                        raise ValueError("Unsupported ARIA operation")
                    result = await remote.call_tool(tool, arguments)
                    return result.model_dump(mode="json", exclude_none=True)


def payload(raw):
    """ARIA returns application errors inside successful MCP result envelopes."""
    if raw.get("isError"):
        return None, "mcp_error"
    data = raw.get("structuredContent")
    if data is None:
        blocks = [item["text"] for item in raw.get("content", []) if item.get("type") == "text"]
        if len(blocks) != 1:
            return None, "invalid_output"
        try:
            data = json.loads(blocks[0])
        except (ValueError, TypeError):
            return None, "invalid_output"
    if not isinstance(data, dict):
        return None, "invalid_output"
    # FastMCP wraps parameterized Dict[str, Any] return values in a result
    # envelope. ARIA's four upstream tools use that annotation. Keep raw MCP
    # evidence unchanged, but inspect the contained application payload.
    if set(data) == {"result"}:
        data = data["result"]
        if not isinstance(data, dict):
            return None, "invalid_output"
    if data.get("error"):
        return data, "aria_error"
    return data, "completed"
