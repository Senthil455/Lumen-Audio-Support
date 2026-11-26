"""The ask pipeline: route, retrieve, ground, answer, cite, account.

Every step produces something observable - a topic, retrieved chunk ids and
scores, token counts, cost, a refusal reason - because the operational question
is never "did it work" but "which step failed".
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from app.budget import BudgetLedger
from app.config import Settings, settings
from app.llm import NOT_IN_DOCS, SYSTEM_PROMPT, build_client
from app.models import Source
from app.retrieval import DEFAULT_RETRIEVER, Corpus, build_retriever
from app.retrieval.text import tokenize
from app.topic_router import TopicRouter

REFUSAL_TEXT = (
    "I could not find that in the Lumen Audio help centre, so I will not guess. "
    "I can help with refunds, returns, warranty claims, shipping, payments, "
    "your account, promotions and product questions - or I can pass this to a "
    "human agent."
)
SNIPPET_CHARS = 240


@dataclass
class AskResult:
    answer: str
    sources: list[Source]
    refused: bool
    refusal_reason: str | None
    topic: str | None
    topic_confidence: float
    latency_ms: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    model: str
    mode: str
    retrieved: list[dict] = field(default_factory=list)


def _snippet(text: str) -> str:
    flat = re.sub(r"\s+", " ", text).strip()
    return flat if len(flat) <= SNIPPET_CHARS else flat[: SNIPPET_CHARS - 1].rstrip() + "..."


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", text))


def _cited_sources(answer: str, hits) -> list[Source]:
    """Keep the chunks the answer actually came from, not everything retrieved."""
    answer_terms = set(tokenize(answer))
    answer_numbers = _numbers(answer)
    sources: list[Source] = []
    for hit in hits:
        chunk_terms = set(tokenize(hit.chunk.text))
        shared_numbers = answer_numbers & _numbers(hit.chunk.text)
        # A number match is strong evidence; the LLM copied a figure from here.
        if len(answer_terms & chunk_terms) >= 4 or shared_numbers:
            sources.append(
                Source(
                    id=hit.chunk.id,
                    doc_id=hit.chunk.doc_id,
                    citation=hit.chunk.citation,
                    snippet=_snippet(hit.chunk.text),
                    score=round(hit.score, 4),
                )
            )
        if len(sources) == 3:
            break
    if not sources and hits:
        # Still be honest about the best candidate rather than showing nothing.
        hit = hits[0]
        sources.append(
            Source(
                id=hit.chunk.id,
                doc_id=hit.chunk.doc_id,
                citation=hit.chunk.citation,
                snippet=_snippet(hit.chunk.text),
                score=round(hit.score, 4),
            )
        )
    return sources


class RAGPipeline:
    def __init__(self, config: Settings | None = None, corpus: Corpus | None = None) -> None:
        self.settings = config or settings
        self.corpus = corpus or Corpus()
        self.retriever_name = getattr(self.settings, "retriever", DEFAULT_RETRIEVER)
        self.retriever = build_retriever(self.retriever_name, self.corpus)
        self.router = TopicRouter()
        self.llm = build_client()
        self.budget = BudgetLedger(limit_usd=self.settings.daily_budget_usd)

    @property
    def mode(self) -> str:
        return "llm" if self.settings.llm_enabled else "offline-extractive"

    def retrieve(self, question: str, k: int | None = None):
        return self.retriever.search(question, k=k or self.settings.top_k)

    def build_prompt(self, question: str, hits) -> str:
        blocks = []
        for index, hit in enumerate(hits, start=1):
            blocks.append(
                f"[{index}] {hit.chunk.citation}\n{hit.chunk.text}"
            )
        return (
            f"QUESTION: {question}\n\n"
            "CONTEXT:\n" + "\n\n".join(blocks) + "\n\n"
            "Answer from the context. If it is not there, reply NOT_IN_DOCS."
        )

    def ask(self, question: str) -> AskResult:
        started = time.perf_counter()
        topic, topic_confidence = self.router.classify(question)

        if self.router.blocks(topic, topic_confidence):
            return self._refuse(topic, topic_confidence, started, "out_of_scope")

        hits = self.retrieve(question)
        if not hits:
            return self._refuse(topic, topic_confidence, started, "no_retrieval_hit")

        prompt = self.build_prompt(question, hits)
        estimated = (len(prompt) // 4) / 1000 * self.settings.price_input_per_1k
        # Raises BudgetExceeded; the endpoint maps it to 429.
        self.budget.check(estimated)

        result = self.llm.complete(SYSTEM_PROMPT, prompt)
        self.budget.charge(result.cost_usd)

        latency_ms = int((time.perf_counter() - started) * 1000)
        retrieved = [
            {"id": h.chunk.id, "doc_id": h.chunk.doc_id, "score": round(h.score, 4)}
            for h in hits
        ]

        if result.refused or result.text.strip().upper() == NOT_IN_DOCS:
            return AskResult(
                answer=REFUSAL_TEXT,
                sources=[],
                refused=True,
                refusal_reason="not_in_documents",
                topic=topic,
                topic_confidence=topic_confidence,
                latency_ms=latency_ms,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                cost_usd=result.cost_usd,
                model=result.model,
                mode=self.mode,
                retrieved=retrieved,
            )

        return AskResult(
            answer=result.text,
            sources=_cited_sources(result.text, hits),
            refused=False,
            refusal_reason=None,
            topic=topic,
            topic_confidence=topic_confidence,
            latency_ms=latency_ms,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cost_usd=result.cost_usd,
            model=result.model,
            mode=self.mode,
            retrieved=retrieved,
        )

    def _refuse(self, topic, confidence, started: float, reason: str) -> AskResult:
        return AskResult(
            answer=REFUSAL_TEXT,
            sources=[],
            refused=True,
            refusal_reason=reason,
            topic=topic,
            topic_confidence=confidence,
            latency_ms=int((time.perf_counter() - started) * 1000),
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
            model=self.llm.model,
            mode=self.mode,
        )
