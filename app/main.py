"""FastAPI entrypoint.

    GET  /health          liveness, exactly {"status": "ok"}
    GET  /info            what is running: mode, model, corpus, tools, budget
    POST /ask             one question, grounded answer plus citations
    POST /agent           the tool-using agent, returns a trace id
    GET  /traces/{id}     the steps of one agent run (no question or answer text)
    GET  /stats           in-process counters for the demo dashboard
"""

from __future__ import annotations

import asyncio
import threading
import time

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded

from app.agent.loop import Agent, AgentOutcome
from app.agent.mcp_client import MCPDocsClient
from app.budget import BudgetExceeded
from app.config import settings
from app.llm import LLMError
from app.logging_setup import caller_id, log_event
from app.models import (
    AgentResponse,
    AgentStep,
    Answer,
    HealthResponse,
    InfoResponse,
    Question,
    Usage,
)
from app.pipeline import RAGPipeline
from app.ratelimit import limiter, rate_limit_handler


class Counters:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.data = {"ask": 0, "agent": 0, "refused": 0, "errors": 0, "rate_limited": 0}

    def bump(self, key: str) -> None:
        with self._lock:
            self.data[key] = self.data.get(key, 0) + 1

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self.data)


counters = Counters()


async def _lifespan(app: FastAPI):
    corpus_pipeline = RAGPipeline()
    app.state.pipeline = corpus_pipeline
    mcp = MCPDocsClient()
    if settings.mcp_enabled:
        await mcp.start()
    app.state.mcp = mcp
    app.state.agent = Agent(corpus=corpus_pipeline.corpus, mcp=mcp)
    log_event(
        "startup",
        mode=corpus_pipeline.mode,
        model=corpus_pipeline.llm.model,
        chunks=len(corpus_pipeline.corpus),
        documents=len(corpus_pipeline.corpus.docs),
        mcp_connected=mcp.status.connected,
    )
    try:
        yield
    finally:
        await mcp.stop()
        log_event("shutdown")


app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="Grounded support answers with citations, a tool-using agent and traces.",
    lifespan=_lifespan,
)
app.state.limiter = limiter
app.state.counters = counters
app.add_exception_handler(RateLimitExceeded, rate_limit_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def observe(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    latency = int((time.perf_counter() - started) * 1000)
    if response.status_code == 429:
        counters.bump("rate_limited")
    log_event(
        "request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        latency_ms=latency,
        caller=caller_id(request),
    )
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@app.get("/info", response_model=InfoResponse)
def info(request: Request) -> InfoResponse:
    pipeline: RAGPipeline = request.app.state.pipeline
    mcp: MCPDocsClient = request.app.state.mcp
    return InfoResponse(
        app=settings.app_name,
        version=settings.version,
        mode=pipeline.mode,
        model=pipeline.llm.model,
        documents=len(pipeline.corpus.docs),
        chunks=len(pipeline.corpus),
        retriever=pipeline.retriever_name,
        tools=["search_docs", "lookup_order", "escalate_to_human"],
        mcp_connected=mcp.status.connected,
        budget=pipeline.budget.snapshot(),
    )


@app.post("/ask", response_model=Answer)
@limiter.limit(settings.rate_limit)
async def ask(question: Question, request: Request, response: Response) -> Answer:
    pipeline: RAGPipeline = request.app.state.pipeline
    caller = caller_id(request)
    counters.bump("ask")
    try:
        result = await asyncio.to_thread(pipeline.ask, question.text)
    except BudgetExceeded as exc:
        counters.bump("refused")
        log_event("ask", ok=False, reason="budget", caller=caller, q_len=len(question.text))
        raise HTTPException(status_code=429, detail=f"Daily budget exhausted: {exc}") from exc
    except LLMError as exc:
        counters.bump("errors")
        log_event("ask", ok=False, reason="llm_error", caller=caller, q_len=len(question.text))
        raise HTTPException(status_code=502, detail="Model provider unavailable") from exc

    if result.refused:
        counters.bump("refused")
    log_event(
        "ask",
        ok=not result.refused,
        latency_ms=result.latency_ms,
        tokens=result.input_tokens + result.output_tokens,
        cost_usd=round(result.cost_usd, 6),
        q_len=len(question.text),
        topic=result.topic,
        refused=result.refused,
        sources=len(result.sources),
        caller=caller,
    )
    return Answer(
        request_id=caller,
        answer=result.answer,
        sources=result.sources,
        refused=result.refused,
        refusal_reason=result.refusal_reason,
        topic=result.topic,
        topic_confidence=result.topic_confidence,
        usage=Usage(
            latency_ms=result.latency_ms,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cost_usd=result.cost_usd,
            model=result.model,
            mode=result.mode,
        ),
    )


@app.post("/agent", response_model=AgentResponse)
@limiter.limit(settings.agent_rate_limit)
async def agent_endpoint(
    question: Question, request: Request, response: Response
) -> AgentResponse:
    agent: Agent = request.app.state.agent
    caller = caller_id(request)
    counters.bump("agent")
    try:
        outcome: AgentOutcome = await agent.run(question.text)
    except BudgetExceeded as exc:
        counters.bump("refused")
        log_event("agent", ok=False, reason="budget", caller=caller, q_len=len(question.text))
        raise HTTPException(status_code=429, detail=f"Daily budget exhausted: {exc}") from exc
    except LLMError as exc:
        counters.bump("errors")
        log_event("agent", ok=False, reason="llm_error", caller=caller)
        raise HTTPException(status_code=502, detail="Model provider unavailable") from exc

    if outcome.refused:
        counters.bump("refused")
    log_event(
        "agent",
        ok=not outcome.refused,
        latency_ms=outcome.latency_ms,
        tokens=outcome.input_tokens + outcome.output_tokens,
        cost_usd=round(outcome.cost_usd, 6),
        q_len=len(question.text),
        steps=len(outcome.steps),
        tools=outcome.tool_calls,
        refused=outcome.refused,
        trace_id=outcome.trace_id,
        caller=caller,
    )
    return AgentResponse(
        request_id=caller,
        answer=outcome.answer,
        sources=outcome.sources,
        refused=outcome.refused,
        refusal_reason=outcome.refusal_reason,
        topic=outcome.topic,
        topic_confidence=outcome.topic_confidence,
        usage=Usage(
            latency_ms=outcome.latency_ms,
            input_tokens=outcome.input_tokens,
            output_tokens=outcome.output_tokens,
            cost_usd=outcome.cost_usd,
            model=outcome.model,
            mode=outcome.mode,
        ),
        trace_id=outcome.trace_id,
        steps=[AgentStep(**step.model_dump()) for step in outcome.steps],
    )


@app.get("/traces/{trace_id}")
def get_trace(trace_id: str) -> JSONResponse:
    record = Agent.read_trace(trace_id)
    if record is None:
        raise HTTPException(status_code=404, detail="trace not found")
    return JSONResponse(record)


@app.get("/stats")
def stats(request: Request) -> JSONResponse:
    pipeline: RAGPipeline = request.app.state.pipeline
    return JSONResponse(
        {
            "counters": counters.snapshot(),
            "budget": pipeline.budget.snapshot(),
            "mode": pipeline.mode,
        }
    )


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception) -> JSONResponse:
    counters.bump("errors")
    log_event("error", path=request.url.path, error=type(exc).__name__)
    # Never leak internals or provider messages to the caller.
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
