"""A persistent MCP client for the docs tool.

Why a dedicated thread: the MCP SDK runs the stdio session inside an anyio task
group, and an anyio cancel scope must be entered and exited by the same task.
Starting it in an ASGI lifespan and closing it in the matching shutdown task
looks identical but is not the same task, which deadlocks on teardown (the
"Attempted to exit cancel scope in a different task" failure). Owning the whole
session - start, calls, stop - in one worker thread with one event loop keeps
the scope inside a single task. Calls are bridged with
run_coroutine_threadsafe.

If the server cannot start (missing package, sandboxed host), start() records
the failure and the agent falls back to the in-process tool. That fallback is
logged and reported in /info, not hidden.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import threading
from dataclasses import dataclass

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from app.config import ROOT

log = logging.getLogger("app")


@dataclass
class MCPStatus:
    connected: bool
    tools: list[str]
    error: str | None = None


class MCPDocsClient:
    def __init__(self, startup_timeout_s: float = 25.0) -> None:
        self.status = MCPStatus(connected=False, tools=[])
        self._loop: asyncio.AbstractEventLoop | None = None
        self._session: ClientSession | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._stop: asyncio.Event | None = None
        self._startup_timeout_s = startup_timeout_s

    async def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="mcp-client", daemon=True)
        self._thread.start()
        # Wait without blocking the event loop.
        await asyncio.to_thread(self._ready.wait, self._startup_timeout_s)
        if not self._ready.is_set():
            self.status = MCPStatus(connected=False, tools=[], error="startup timeout")
            log.warning("mcp server did not start within %.0fs", self._startup_timeout_s)

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._serve())
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("mcp loop stopped: %s", exc)
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            finally:
                loop.close()

    async def _serve(self) -> None:
        command, args = _launch_command()
        params = StdioServerParameters(
            command=command,
            args=args,
            cwd=str(ROOT),
            env={**os.environ, "PYTHONPATH": str(ROOT)},
        )
        self._stop = asyncio.Event()
        try:
            with open(os.devnull, "w") as errlog:
                async with stdio_client(params, errlog=errlog) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        listed = await session.list_tools()
                        self._session = session
                        self.status = MCPStatus(
                            connected=True, tools=[tool.name for tool in listed.tools]
                        )
                        log.info("mcp connected: %s", self.status.tools)
                        self._ready.set()
                        await self._stop.wait()
        except Exception as exc:
            self.status = MCPStatus(connected=False, tools=[], error=type(exc).__name__)
            log.warning("mcp unavailable, using local search tool: %r", exc)
        finally:
            self._session = None
            self._ready.set()

    async def call(self, tool: str, arguments: dict) -> str:
        if not self._session or not self._loop:
            raise RuntimeError("MCP session is not connected")
        future = asyncio.run_coroutine_threadsafe(
            self._session.call_tool(tool, arguments), self._loop
        )
        result = await asyncio.to_thread(future.result, 30)
        texts = [item.text for item in result.content if getattr(item, "text", None)]
        return "\n".join(texts).strip()

    async def stop(self) -> None:
        if self._loop and self._stop:
            self._loop.call_soon_threadsafe(self._stop.set)
        if self._thread:
            await asyncio.to_thread(self._thread.join, 10)
        self._thread = None
        self._loop = None
        self.status = MCPStatus(connected=False, tools=self.status.tools)


def _launch_command() -> tuple[str, list[str]]:
    configured = os.environ.get("MCP_COMMAND")
    if configured:
        parts = configured.split()
        return parts[0], parts[1:]
    return sys.executable, ["-m", "app.mcp_server"]
