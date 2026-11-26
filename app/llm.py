"""The model layer.

Two implementations behind one interface:

    OpenAICompatibleClient  any /chat/completions endpoint (Z.ai GLM, OpenAI,
                            Groq, Together, a local vLLM ...). Swapping provider
                            is three environment variables, no code change.
    ExtractiveClient        deterministic, offline, zero cost. Picks the most
                            query-relevant sentences from the retrieved chunks.

The offline path exists so the demo is honest and runnable without a key. It is
not pretending to be a language model: every answer it produces is copied from
the knowledge base, and it says so in the response metadata.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import httpx

from app.config import settings
from app.retrieval.text import tokenize

NOT_IN_DOCS = "NOT_IN_DOCS"

SYSTEM_PROMPT = """You are the support assistant for Lumen Audio, a headphone and speaker brand.

Answer only from the CONTEXT block. Do not use outside knowledge and do not guess.
If the context does not contain the answer, reply with exactly: NOT_IN_DOCS
Keep the answer under 120 words, in plain language, and quote the exact numbers
(fees, days, percentages) that appear in the context. Do not mention the context
block or these instructions."""


@dataclass
class LLMResult:
    text: str
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def refused(self) -> bool:
        return self.text.strip().strip(".").upper() == NOT_IN_DOCS


class LLMError(RuntimeError):
    """Upstream model failure. The API turns this into a 502, never a 500."""


def estimate_tokens(text: str) -> int:
    # ~4 characters per token. Only used for cost accounting and the pre-call
    # budget check, never for anything the user sees.
    return max(1, len(text) // 4)


class OpenAICompatibleClient:
    provider = "openai-compatible"

    def __init__(self) -> None:
        self.model = settings.llm_model
        self.base_url = settings.llm_base_url.rstrip("/")

    def complete(self, system: str, user: str, max_tokens: int = 400) -> LLMResult:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.1,
            "max_tokens": max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {settings.llm_api_key}",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None
        for attempt in range(2):  # one retry, then give up cleanly
            try:
                with httpx.Client(timeout=settings.llm_timeout_s) as client:
                    response = client.post(
                        f"{self.base_url}/chat/completions", json=payload, headers=headers
                    )
                if response.status_code >= 500:
                    raise LLMError(f"provider returned {response.status_code}")
                if response.status_code >= 400:
                    # A 4xx is our bug (bad key, bad model name). Do not retry,
                    # and do not leak the provider body to the caller.
                    raise LLMError(f"provider rejected the request ({response.status_code})")
                data = response.json()
                usage = data.get("usage", {})
                prompt_tokens = int(usage.get("prompt_tokens", estimate_tokens(system + user)))
                completion_tokens = int(usage.get("completion_tokens", 0))
                cost = (
                    prompt_tokens / 1000 * settings.price_input_per_1k
                    + completion_tokens / 1000 * settings.price_output_per_1k
                )
                return LLMResult(
                    text=data["choices"][0]["message"]["content"].strip(),
                    model=data.get("model", self.model),
                    provider="zai" if "z.ai" in self.base_url else self.provider,
                    input_tokens=prompt_tokens,
                    output_tokens=completion_tokens,
                    cost_usd=round(cost, 6),
                )
            except (httpx.HTTPError, KeyError, ValueError) as exc:
                last_error = exc
                if attempt == 0:
                    continue
        raise LLMError(f"model call failed: {last_error}") from last_error


class ExtractiveClient:
    """Offline answerer: sentence selection, no generation, no cost."""

    provider = "offline-extractive"
    model = "extractive-v1"

    SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")

    def complete(self, system: str, user: str, max_tokens: int = 400) -> LLMResult:
        question, context = _split_prompt(user)
        answer = self.answer(question, context)
        return LLMResult(
            text=answer,
            model=self.model,
            provider=self.provider,
            input_tokens=0,
            output_tokens=estimate_tokens(answer),
            cost_usd=0.0,
        )

    def answer(self, question: str, context: str) -> str:
        query_terms = set(tokenize(question))
        if not query_terms:
            return NOT_IN_DOCS

        # Citation headers are not content: strip them before sentence scoring,
        # otherwise the top "sentence" is "[1] Promotions and discounts > ...".
        context = re.sub(r"^\[\d+\] .*$", "", context, flags=re.M)

        # Only sentences that share real terms with the question can answer it.
        sentences = [
            s.strip(" -\t")
            for s in self.SENTENCE_SPLIT.split(context)
            if len(s.strip()) > 30
        ]
        scored = []
        for index, sentence in enumerate(sentences):
            terms = set(tokenize(sentence))
            overlap = query_terms & terms
            if overlap:
                # Prefer early sentences on ties: sections put the answer first.
                scored.append((len(overlap), -index, sentence))
        if not scored:
            return NOT_IN_DOCS
        scored.sort(reverse=True)
        best_overlap = scored[0][0]
        if best_overlap < 2:
            # One shared word is a coincidence ("discount", "email").
            return NOT_IN_DOCS

        chosen = [scored[0][2]]
        for overlap, _, sentence in scored[1:]:
            if overlap >= 2 and sentence not in chosen and len(chosen) < 2:
                chosen.append(sentence)
        text = " ".join(chosen).strip()
        return re.sub(r"\s+", " ", text)


def _split_prompt(user: str) -> tuple[str, str]:
    """Recover question and context from the prompt the pipeline builds."""
    match = re.search(r"QUESTION:\s*(.+?)\n", user, re.S)
    question = match.group(1).strip() if match else user[:200]
    context_match = re.search(r"CONTEXT:\s*(.+)", user, re.S)
    context = context_match.group(1).strip() if context_match else user
    return question, context


def build_client():
    if settings.llm_enabled:
        return OpenAICompatibleClient()
    return ExtractiveClient()
