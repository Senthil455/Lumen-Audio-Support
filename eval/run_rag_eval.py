"""Evaluate /ask against the 15 held-out questions.

Four things are checked per question, because "the answer looked fine" is not a
measurement:
  1. retrieval - did any gold document appear in the chunks used?
  2. answer    - does the answer contain an accepted fact?
  3. refusal   - for unanswerable questions, did it refuse instead of guessing?
  4. grounding - are all numbers in the answer present in the cited context?

Run:  python eval/run_rag_eval.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.pipeline import RAGPipeline
from app.retrieval import Corpus
from topic_classifier.common import ROOT, save_json

QUESTIONS = ROOT / "data" / "eval" / "questions.jsonl"
RESULTS_DIR = ROOT / "eval" / "results"


def numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", text))


def main() -> None:
    questions = [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines() if line.strip()]
    pipeline = RAGPipeline()
    corpus = pipeline.corpus

    rows = []
    for question in questions:
        result = pipeline.ask(question["question"])
        retrieved_docs = {item["doc_id"] for item in result.retrieved}
        # Grounding is checked against everything the model was given, not just
        # the snippets that ended up cited.
        context = " ".join(corpus.get(item["id"]).text for item in result.retrieved)
        context_numbers = numbers(context)
        answer_numbers = numbers(result.answer)

        expected_refusal = question["type"] == "unanswerable"
        keyword_ok = (
            any(k.lower() in result.answer.lower() for k in question["must_contain"])
            if question["must_contain"]
            else None
        )
        rows.append(
            {
                "id": question["id"],
                "type": question["type"],
                "question": question["question"],
                "answer": result.answer,
                "refused": result.refused,
                "refusal_reason": result.refusal_reason,
                "gold_docs": question["gold_docs"],
                "retrieved_docs": sorted(retrieved_docs),
                "retrieval_hit": bool(retrieved_docs & set(question["gold_docs"]))
                if question["gold_docs"]
                else None,
                "keyword_ok": keyword_ok,
                "refusal_ok": result.refused == expected_refusal,
                "ungrounded_numbers": sorted(answer_numbers - context_numbers),
                "sources": [source.citation for source in result.sources],
                "latency_ms": result.latency_ms,
                "cost_usd": result.cost_usd,
            }
        )

    answerable = [row for row in rows if row["type"] != "unanswerable"]
    unanswerable = [row for row in rows if row["type"] == "unanswerable"]
    summary = {
        "mode": pipeline.mode,
        "model": pipeline.llm.model,
        "retriever": pipeline.retriever_name,
        "questions": len(rows),
        "retrieval_hit_rate": round(
            sum(1 for row in answerable if row["retrieval_hit"]) / len(answerable), 3
        ),
        "answer_fact_rate": round(
            sum(1 for row in answerable if row["keyword_ok"]) / len(answerable), 3
        ),
        "answer_fact_count": f"{sum(1 for r in answerable if r['keyword_ok'])}/{len(answerable)}",
        "refusal_accuracy": round(sum(1 for row in rows if row["refusal_ok"]) / len(rows), 3),
        "refusal_count": f"{sum(1 for r in rows if r['refusal_ok'])}/{len(rows)}",
        "unanswerable_refused": f"{sum(1 for r in unanswerable if r['refused'])}/{len(unanswerable)}",
        "ungrounded_answers": sum(1 for row in rows if row["ungrounded_numbers"]),
        "avg_latency_ms": round(sum(row["latency_ms"] for row in rows) / len(rows)),
        "total_cost_usd": round(sum(row["cost_usd"] for row in rows), 6),
    }

    payload = {"summary": summary, "rows": rows}
    save_json(RESULTS_DIR / "rag.json", payload)

    print(f"\nmode={summary['mode']} model={summary['model']} retriever={summary['retriever']}")
    print(
        f"retrieval hit {summary['retrieval_hit_rate']:.3f} | "
        f"answer fact {summary['answer_fact_rate']:.3f} ({summary['answer_fact_count']}) | "
        f"refusal {summary['refusal_accuracy']:.3f} ({summary['refusal_count']}) | "
        f"ungrounded {summary['ungrounded_answers']} | "
        f"avg {summary['avg_latency_ms']} ms"
    )
    print("\nfailures:")
    for row in rows:
        problems = []
        if row["type"] != "unanswerable" and not row["keyword_ok"]:
            problems.append("answer missing expected fact")
        if not row["refusal_ok"]:
            problems.append("wrong refusal behaviour")
        if row["ungrounded_numbers"]:
            problems.append(f"numbers not in context: {row['ungrounded_numbers']}")
        if problems:
            print(f"  {row['id']}: {'; '.join(problems)}")
            print(f"     -> {row['answer'][:160]}")


if __name__ == "__main__":
    main()
