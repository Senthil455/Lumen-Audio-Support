"""Request and response contracts.

Validation lives here so it cannot be forgotten in an endpoint. The lesson's
limits are enforced by the model: 3 to 500 characters, and a whitespace-only
question is rejected even though it is technically long enough.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

MIN_QUESTION_CHARS = 3
MAX_QUESTION_CHARS = 500


class Question(BaseModel):
    text: str = Field(
        ...,
        min_length=MIN_QUESTION_CHARS,
        max_length=MAX_QUESTION_CHARS,
        description="The customer's question, plain text.",
        examples=["How long does a refund take?"],
    )

    @field_validator("text")
    @classmethod
    def not_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if len(cleaned) < MIN_QUESTION_CHARS:
            raise ValueError(f"question must be at least {MIN_QUESTION_CHARS} characters")
        return cleaned


class Source(BaseModel):
    id: str
    doc_id: str
    citation: str
    snippet: str
    score: float


class Usage(BaseModel):
    latency_ms: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    model: str
    mode: Literal["llm", "offline-extractive"]


class Answer(BaseModel):
    request_id: str
    answer: str
    sources: list[Source]
    refused: bool
    refusal_reason: str | None = None
    topic: str | None = None
    topic_confidence: float | None = None
    usage: Usage
    trace_id: str | None = None


class AgentStep(BaseModel):
    step: int
    action: str
    tool: str | None = None
    args_summary: str | None = None
    observation: str | None = None
    ok: bool = True


class AgentResponse(Answer):
    steps: list[AgentStep] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class InfoResponse(BaseModel):
    app: str
    version: str
    mode: str
    model: str
    documents: int
    chunks: int
    retriever: str
    tools: list[str]
    mcp_connected: bool
    budget: dict
