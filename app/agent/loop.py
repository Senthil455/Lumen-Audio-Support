"""The agent loop.

The model, when there is one, chooses the next tool from a JSON protocol - no
provider-specific function-calling payloads, so swapping Z.ai for OpenAI or a
local model is an environment change. Without a key, a deterministic policy
does the same job: look up the order if the question names one, search the
docs, escalate when asked, then answer from what was gathered.

Either way the run produces a trace with every step, and the trace contains no
question text, no answer text and no document bodies.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from app.agent.mcp_client import MCPDocsClient
from app.agent.tools import TOOL_SCHEMAS, ToolRegistry, ToolResult, describe_args
from app.budget import BudgetExceeded, BudgetLedger
from app.config import ROOT, Settings, settings
from app.llm import LLMError, SYSTEM_PROMPT, build_client, estimate_tokens
from app.models import AgentStep, Source
from app.pipeline import REFUSAL_TEXT, _snippet
from app.retrieval import Corpus
from app.retrieval.text import tokenize
from app.topic_router import TopicRouter

TRACE_PATH = settings.log_dir / "traces.jsonl"
ORDER_ID = re.compile(r"LA-\d{5}", re.I)
ESCALATION_WORDS = ("human", "person", "manager", "agent", "complaint", "escalate")

PLANNER_PROMPT = """You are the planner for Lumen Audio customer support.
You have these tools:
{tools}

Work in steps. On each turn reply with ONE json object and nothing else.
To call a tool:  {{"action": "tool", "tool": "<name>", "args": {{...}}}}
When you can answer: {{"action": "final", "answer": "<text for the customer>"}}

Rules:
- Use lookup_order when the question mentions an order id.
- Use search_docs before answering any policy question.
- If the context and order data do not answer the question, use
  escalate_to_human or say you cannot find it.
- Never invent order statuses, dates, fees or ticket numbers.
- Answer in under 120 words, quoting exact numbers from the observations."""


@dataclass
class AgentOutcome:
    answer: str
    sources: list[Source]
    steps: list[AgentStep]
    refused: bool
    refusal_reason: str | None
    trace_id: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    model: str
    mode: str
    topic: str | None
    topic_confidence: float
    tool_calls: list[str] = field(default_factory=list)


class Agent:
    def __init__(
        self,
        config: Settings | None = None,
        corpus: Corpus | None = None,
        mcp: MCPDocsClient | None = None,
    ) -> None:
        self.settings = config or settings
        self.corpus = corpus or Corpus()
        self.tools = ToolRegistry(self.corpus)
        self.router = TopicRouter()
        self.llm = build_client()
        self.mcp = mcp
        self.budget = BudgetLedger(limit_usd=self.settings.daily_budget_usd)

    @property
    def mode(self) -> str:
        return "llm" if self.settings.llm_enabled else "offline-extractive"

    async def run(self, question: str) -> AgentOutcome:
        started = time.perf_counter()
        trace_id = uuid.uuid4().hex[:12]
        topic, confidence = self.router.classify(question)

        steps: list[AgentStep] = []
        calls: list[str] = []
        observations: list[tuple[str, ToolResult]] = []
        input_tokens = output_tokens = 0
        cost = 0.0
        answer = ""
        refused = False
        refusal_reason: str | None = None

        for step_number in range(1, self.settings.max_agent_steps + 1):
            if self.settings.llm_enabled:
                decision, tokens_in, tokens_out, step_cost = await self._plan_with_llm(
                    question, observations, calls
                )
                input_tokens += tokens_in
                output_tokens += tokens_out
                cost += step_cost
            else:
                decision = self._plan_offline(question, calls, observations)

            if decision.get("action") == "final":
                answer = str(decision.get("answer", "")).strip()
                steps.append(AgentStep(step=step_number, action="final"))
                break

            tool = str(decision.get("tool", ""))
            args = decision.get("args") or {}
            if tool not in {schema["name"] for schema in TOOL_SCHEMAS}:
                steps.append(
                    AgentStep(
                        step=step_number,
                        action="tool",
                        tool=tool or "(none)",
                        ok=False,
                        observation="unknown tool, ignored",
                    )
                )
                calls.append(f"{tool}?")
                continue

            result = await self._execute(tool, args)
            observations.append((tool, result))
            calls.append(tool)
            steps.append(
                AgentStep(
                    step=step_number,
                    action="tool",
                    tool=tool,
                    args_summary=describe_args(tool, args),
                    observation=self._safe_observation(tool, result),
                    ok=result.ok,
                )
            )
        else:
            steps.append(AgentStep(step=len(steps) + 1, action="final", observation="step limit reached"))

        if not answer:
            answer, refused, refusal_reason = self._compose_offline(question, observations)
        elif not answer.strip():
            answer, refused, refusal_reason = self._compose_offline(question, observations)
        if answer.strip().upper() == "NOT_IN_DOCS":
            answer, refused, refusal_reason = REFUSAL_TEXT, True, "not_in_documents"

        sources = self._sources_from(observations)
        latency_ms = int((time.perf_counter() - started) * 1000)
        outcome = AgentOutcome(
            answer=answer,
            sources=sources,
            steps=steps,
            refused=refused,
            refusal_reason=refusal_reason,
            trace_id=trace_id,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=round(cost, 6),
            model=self.llm.model,
            mode=self.mode,
            topic=topic,
            topic_confidence=confidence,
            tool_calls=calls,
        )
        self._write_trace(question, outcome)
        return outcome

    # -- planning --------------------------------------------------------
    async def _plan_with_llm(self, question, observations, calls):
        transcript = [f"CUSTOMER: {question}"]
        for tool, result in observations:
            transcript.append(f"OBSERVATION from {tool}:\n{result.observation[:1200]}")
        transcript.append("What is your next step? Reply with one json object.")
        user = "\n\n".join(transcript)
        tools_json = json.dumps(TOOL_SCHEMAS, indent=2)
        system = PLANNER_PROMPT.format(tools=tools_json)

        self.budget.check(
            estimate_tokens(system + user) / 1000 * self.settings.price_input_per_1k
        )
        result = await asyncio.to_thread(self.llm.complete, system, user, 400)
        self.budget.charge(result.cost_usd)
        decision = _parse_json(result.text)
        if decision is None:
            # One malformed plan should not fail the request: answer from what
            # has already been gathered.
            decision = {"action": "final", "answer": ""}
        return decision, result.input_tokens, result.output_tokens, result.cost_usd

    def _plan_offline(self, question, calls, observations) -> dict:
        order_ids = ORDER_ID.findall(question)
        if order_ids and "lookup_order" not in calls:
            return {"action": "tool", "tool": "lookup_order", "args": {"order_id": order_ids[0].upper()}}
        if "search_docs" not in calls:
            return {"action": "tool", "tool": "search_docs", "args": {"query": question, "k": 4}}
        if any(word in question.lower() for word in ESCALATION_WORDS) and "escalate_to_human" not in calls:
            return {
                "action": "tool",
                "tool": "escalate_to_human",
                "args": {"summary": question},
            }
        return {"action": "final", "answer": ""}

    # -- tools -----------------------------------------------------------
    async def _execute(self, tool: str, args: dict) -> ToolResult:
        if (
            tool == "search_docs"
            and self.mcp is not None
            and self.mcp.status.connected
        ):
            try:
                text = await self.mcp.call(
                    "search_docs",
                    {"query": str(args.get("query", "")), "k": int(args.get("k", 4))},
                )
                hits = _parse_mcp_hits(text)
                return ToolResult(
                    ok=True, observation=text, data={"transport": "mcp", "hits": hits}
                )
            except Exception as exc:
                # Fall back rather than fail the customer's request.
                result = await asyncio.to_thread(self.tools.execute, tool, args)
                result.observation = f"{result.observation}\n(mcp failed: {type(exc).__name__}, used local tool)"
                return result
        return await asyncio.to_thread(self.tools.execute, tool, args)

    @staticmethod
    def _safe_observation(tool: str, result: ToolResult) -> str:
        if tool == "search_docs":
            hits = result.data.get("hits", [])
            top = ", ".join(hit["citation"] for hit in hits[:2]) or "none"
            return f"{len(hits)} passages, top: {top}"
        if tool == "lookup_order":
            order = result.data.get("order") or {}
            return f"order found, status={order.get('status', 'unknown')}" if order else "order not found"
        if tool == "escalate_to_human":
            return "ticket created" if result.ok else "ticket failed"
        return "ok" if result.ok else "failed"

    # -- answering -------------------------------------------------------
    def _compose_offline(self, question: str, observations) -> tuple[str, bool, str | None]:
        order_text = ""
        doc_text = ""
        ticket_text = ""
        has_docs = False

        for tool, result in observations:
            if tool == "lookup_order" and result.ok:
                order = result.data["order"]
                status = order["status"]
                if order.get("delivered") and status == "delivered":
                    status = f"delivered on {order['delivered']}"
                order_text = f"Order {order['order_id']} is {status}. {order['notes'].rstrip('.')}."
            elif tool == "search_docs" and result.ok:
                hits = result.data.get("hits", [])
                if hits:
                    has_docs = True
                    doc_text = _best_sentences(question, [hit["text"] for hit in hits])
            elif tool == "escalate_to_human" and result.ok:
                ticket_text = result.observation

        if not order_text and not has_docs and not ticket_text:
            return REFUSAL_TEXT, True, "not_in_documents"

        parts = [part for part in (order_text, doc_text, ticket_text) if part]
        answer = " ".join(parts).strip()
        if not answer:
            return REFUSAL_TEXT, True, "not_in_documents"
        return answer, False, None

    def _sources_from(self, observations) -> list[Source]:
        sources: list[Source] = []
        for tool, result in observations:
            if tool != "search_docs":
                continue
            for hit in result.data.get("hits", [])[:3]:
                if any(source.doc_id == hit["id"].split("#")[0] for source in sources):
                    continue
                sources.append(
                    Source(
                        id=hit["id"],
                        doc_id=hit["id"].split("#")[0],
                        citation=hit["citation"],
                        snippet=_snippet(hit["text"]),
                        score=hit["score"],
                    )
                )
                if len(sources) == 3:
                    return sources
        return sources

    # -- traces ----------------------------------------------------------
    def _write_trace(self, question: str, outcome: AgentOutcome) -> None:
        TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "trace_id": outcome.trace_id,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "mode": outcome.mode,
            "topic": outcome.topic,
            "tool_calls": outcome.tool_calls,
            "steps": [step.model_dump() for step in outcome.steps],
            "refused": outcome.refused,
            "refusal_reason": outcome.refusal_reason,
            "sources": [source.id for source in outcome.sources],
            "latency_ms": outcome.latency_ms,
            "input_tokens": outcome.input_tokens,
            "output_tokens": outcome.output_tokens,
            "cost_usd": outcome.cost_usd,
            # Sizes, never content.
            "question_chars": len(question),
            "answer_chars": len(outcome.answer),
        }
        with TRACE_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    @staticmethod
    def read_trace(trace_id: str) -> dict | None:
        if not TRACE_PATH.exists():
            return None
        for line in TRACE_PATH.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("trace_id") == trace_id:
                return record
        return None


MCP_HIT = re.compile(
    r"^\[(\d+)\] (\S+) \| (.+?) \(score ([\d.]+)\)\n(.+?)(?=\n\n\[|\Z)",
    re.S | re.M,
)


def _parse_mcp_hits(text: str) -> list[dict]:
    """Turn the MCP server's passage format back into structured hits."""
    hits = []
    for _, chunk_id, citation, score, body in MCP_HIT.findall(text):
        hits.append(
            {
                "id": chunk_id,
                "citation": citation,
                "score": float(score),
                "text": body.strip(),
            }
        )
    return hits


def _best_sentences(question: str, texts: list[str], limit: int = 2) -> str:
    """Sentence selection for the offline path, shared by ask and agent.

    Take the best-matching sentence, then the sentence right after it in the
    same passage: policy answers are written as a short run of consecutive
    sentences ("start a claim ..." then "enter the serial number ..."), so the
    neighbour carries the step that keyword overlap alone misses.
    """
    query_terms = set(tokenize(question))
    per_text: list[list[str]] = []
    for text in texts:
        per_text.append(
            [
                part.strip()
                for part in re.split(r"(?<=[.!?])\s+|\n+", text)
                if len(part.strip()) > 30
            ]
        )

    scored = []
    for text_index, sentences in enumerate(per_text):
        for sentence_index, sentence in enumerate(sentences):
            overlap = len(query_terms & set(tokenize(sentence)))
            if overlap >= 2:
                scored.append((overlap, text_index, sentence_index))
    if not scored:
        return ""
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    _, best_text, best_sentence = scored[0]

    chosen: list[str] = []
    sentences = per_text[best_text]
    for offset in range(0, limit):
        index = best_sentence + offset
        if index < len(sentences):
            chosen.append(sentences[index])
    for _, text_index, sentence_index in scored[1:]:
        if len(chosen) >= limit:
            break
        sentence = per_text[text_index][sentence_index]
        if sentence not in chosen:
            chosen.append(sentence)
    return re.sub(r"\s+", " ", " ".join(chosen)).strip()


def _parse_json(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?|\n?```$", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


__all__ = ["Agent", "AgentOutcome", "BudgetExceeded", "LLMError", "ROOT", "SYSTEM_PROMPT", "Path"]
