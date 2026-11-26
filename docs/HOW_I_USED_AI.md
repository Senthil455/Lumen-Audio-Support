# How I used AI

I used an AI coding assistant for most of the code in this repository: the
retrieval layer, the FastAPI service, the agent loop, the evaluation harness and
the README. I designed the scope, the knowledge base, the evaluation questions
and the nine failure modes, and I ran everything that produced a number here.

The useful part is not "it wrote code". It is the two places it was confidently
wrong and how I found out.

## Case 1: the MCP client that hung, and the wrong reason it hung

I asked for an MCP client that starts at application startup and is reused by
every request. The assistant produced a version that opened the MCP session
with an `AsyncExitStack` inside the FastAPI lifespan, and both `/ask` and
`/agent` returned 500.

The error was:

```
RuntimeError: Attempted to exit cancel scope in a different task than it was entered in
```

The assistant's first explanation was that the server process must be dying, and
it suggested restarting the subprocess per request. That would have been a
performance disaster and it was wrong: the server was healthy. Running a
minimal reproduction showed the session worked when started and stopped inside
one coroutine, and deadlocked when the exit happened in the lifespan shutdown
task. The MCP SDK runs the stdio session in an anyio task group, and an anyio
cancel scope must be entered and exited by the same task. Two different ASGI
tasks are not the same task.

The fix is the one in `app/agent/mcp_client.py`: the client owns a worker thread
with its own event loop, the session lives entirely inside one task there, and
calls are bridged with `run_coroutine_threadsafe`. I would not have found this
by reading the assistant's explanation; I found it by reproducing with a
20-line script and reading the traceback carefully.

## Case 2: a hang that was hiding an `AttributeError`

While debugging the above, the agent eval timed out after 180 seconds with no
output at all. The assistant proposed three causes - the model provider, the
retriever, the subprocess - all of them wrong. What was actually happening:

1. `Agent.run` referenced `self.llm_enabled`, which does not exist (it is
   `self.settings.llm_enabled`). The assistant wrote that line and never ran it.
2. The exception propagated out of `asyncio.run`, which then tried to cancel
   the still-open MCP task group and deadlocked during shutdown.

So the hang was a teardown symptom and the real bug was a one-word attribute
error. `faulthandler.dump_traceback_later(20)` showed the process parked in
`asyncio.runners._cancel_all_tasks`, which is what pointed at the second half.
The lesson I will keep: an assistant's confident cause is a hypothesis, and
"it hangs" needs a stack dump before it needs a theory.

## Two smaller ones worth naming

* The assistant assumed `from mcp.server.fastmcp import FastMCP`. The installed
  SDK is 2.x, where it is `mcp.server.mcpserver.MCPServer`. The import error
  said so directly, which is a good reminder to read the error instead of
  patching the import until something works.
* `slowapi` with `headers_enabled=True` requires the endpoint to accept a
  `response: Response` parameter. Without it every request 500s with
  `Exception: parameter 'response' must be an instance of starlette.responses.Response`.
  The assistant's first draft omitted it because the documentation example did.

## What I stopped trusting it on

* **Numbers.** Every metric in `eval/RESULTS.md` was produced by running the
  scripts in this repository. I did not accept a single figure it suggested.
* **"This should work."** The MCP client and the app both "should have worked"
  and did not. Anything touching async teardown, subprocesses or third-party
  SDK versions gets run immediately.
* **Its reading of its own failures.** In both cases above the first diagnosis
  was wrong, and the wrong diagnosis would have produced a worse design (spawn
  a process per request; add retries).

## What it was genuinely good at

* Boilerplate that has one correct shape: Pydantic models, Dockerfiles, the
  JSON log formatter, the pytest fixtures.
* Turning an error message into a focused reproduction.
* Writing the boring half of the evaluation harness so I could spend my time on
  the questions and on deciding what "correct" means.

The failure analysis in the README is mine. The assistant never predicted any of
those five failures; they came from running the evaluation and reading the
outputs.
