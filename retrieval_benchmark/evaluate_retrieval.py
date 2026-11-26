"""Week 3 - retrieval evaluation.

Compares BM25, TF-IDF, LSA and a BM25+TF-IDF hybrid on the same chunk set and
the same 15 held-out questions. Reports recall@1/3/5 and MRR over the 13
answerable questions, and prints the top-1 scores for answerable versus
unanswerable questions so the refusal threshold is chosen from data rather
than guessed.

Run:  python retrieval_benchmark/evaluate_retrieval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.retrieval import Corpus, RETRIEVERS
from topic_classifier.common import ROOT, save_json

RESULTS = ROOT / "eval" / "results"


def load_questions() -> list[dict]:
    path = ROOT / "data" / "eval" / "questions.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate(retriever, questions: list[dict]) -> dict:
    answerable = [q for q in questions if q["gold_docs"]]
    per_question = []
    reciprocal_ranks = []
    hits_at = {1: 0, 3: 0, 5: 0}

    for question in answerable:
        results = retriever.search(question["question"], k=10)
        gold = set(question["gold_docs"])
        rank = None
        found_docs = []
        for position, hit in enumerate(results, start=1):
            found_docs.append(hit.chunk.doc_id)
            if hit.chunk.doc_id in gold:
                rank = position
                break
        if rank is not None:
            reciprocal_ranks.append(1.0 / rank)
            for k in hits_at:
                if rank <= k:
                    hits_at[k] += 1
        else:
            reciprocal_ranks.append(0.0)
        per_question.append(
            {
                "id": question["id"],
                "type": question["type"],
                "rank_of_first_gold_chunk": rank,
                "retrieved_docs_top3": found_docs[:3],
                "gold_docs": sorted(gold),
            }
        )

    # Where do answerable and unanswerable questions sit on the score axis?
    answerable_scores = []
    for question in answerable:
        results = retriever.search(question["question"], k=1)
        answerable_scores.append(results[0].score if results else 0.0)
    unanswerable_scores = []
    for question in questions:
        if question["gold_docs"]:
            continue
        results = retriever.search(question["question"], k=1)
        unanswerable_scores.append(results[0].score if results else 0.0)

    total = len(answerable)
    return {
        "recall@1": round(hits_at[1] / total, 3),
        "recall@3": round(hits_at[3] / total, 3),
        "recall@5": round(hits_at[5] / total, 3),
        "mrr": round(sum(reciprocal_ranks) / total, 3),
        "hits@5": f"{hits_at[5]}/{total}",
        "answerable_top1_score": {
            "min": round(min(answerable_scores), 3) if answerable_scores else None,
            "median": round(sorted(answerable_scores)[len(answerable_scores) // 2], 3)
            if answerable_scores
            else None,
        },
        "unanswerable_top1_score": {
            "max": round(max(unanswerable_scores), 3) if unanswerable_scores else None,
            "median": round(sorted(unanswerable_scores)[len(unanswerable_scores) // 2], 3)
            if unanswerable_scores
            else None,
        },
        "per_question": per_question,
    }


def main() -> None:
    questions = load_questions()
    corpus = Corpus()
    print(f"corpus: {len(corpus)} chunks from {len(corpus.docs)} documents\n")

    payload: dict = {
        "corpus": {
            "chunks": len(corpus),
            "documents": len(corpus.docs),
            "avg_chunk_chars": round(
                sum(len(c.text) for c in corpus.chunks) / len(corpus.chunks)
            ),
        },
        "questions": len(questions),
        "retrievers": {},
    }

    header = f"{'retriever':8s} {'R@1':>6s} {'R@3':>6s} {'R@5':>6s} {'MRR':>6s}"
    print(header)
    print("-" * len(header))
    for name in RETRIEVERS:
        retriever = RETRIEVERS[name](corpus)
        metrics = evaluate(retriever, questions)
        payload["retrievers"][name] = metrics
        print(
            f"{name:8s} {metrics['recall@1']:6.3f} {metrics['recall@3']:6.3f} "
            f"{metrics['recall@5']:6.3f} {metrics['mrr']:6.3f}"
        )

    best = max(payload["retrievers"], key=lambda n: payload["retrievers"][n]["mrr"])
    best_mrr = payload["retrievers"][best]["mrr"]
    # The retrievers tie on this corpus. Ship the hybrid, because it keeps both
    # a lexical and a vector signal and therefore degrades more gracefully on
    # wording the evaluation did not include.
    payload["shipped"] = (
        "hybrid" if payload["retrievers"]["hybrid"]["mrr"] == best_mrr else best
    )
    payload["tied_at_top"] = sorted(
        name
        for name, metrics in payload["retrievers"].items()
        if metrics["mrr"] == best_mrr
    )
    print(f"\nbest MRR: {best} ({best_mrr:.3f}); shipped: {payload['shipped']}")

    print("\nMisses at rank 1 (hybrid):")
    for item in payload["retrievers"]["hybrid"]["per_question"]:
        if item["rank_of_first_gold_chunk"] != 1:
            print(
                f"  {item['id']} rank={item['rank_of_first_gold_chunk']} "
                f"gold={item['gold_docs']} got={item['retrieved_docs_top3']}"
            )

    print("\nTop-1 score by answerability (hybrid):")
    for key in ("answerable_top1_score", "unanswerable_top1_score"):
        print(f"  {key}: {payload['retrievers']['hybrid'][key]}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    save_json(RESULTS / "retrieval.json", payload)


if __name__ == "__main__":
    main()
