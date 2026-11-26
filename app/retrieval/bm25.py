"""A small BM25 (Okapi) retriever - no external search service needed."""

from __future__ import annotations

import math
from collections import Counter

from app.retrieval.chunking import Chunk
from app.retrieval.text import tokenize


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.docs = [tokenize(chunk.text + " " + chunk.section + " " + chunk.doc_title) for chunk in chunks]
        self.term_freqs = [Counter(doc) for doc in self.docs]
        self.lengths = [len(doc) for doc in self.docs]
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        self.doc_freq: Counter[str] = Counter()
        for tf in self.term_freqs:
            self.doc_freq.update(tf.keys())
        self.n = len(chunks)

    def idf(self, term: str) -> float:
        df = self.doc_freq.get(term, 0)
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def scores(self, query: str) -> list[float]:
        terms = tokenize(query)
        if not terms or not self.n:
            return [0.0] * self.n
        out = []
        for i in range(self.n):
            score = 0.0
            tf = self.term_freqs[i]
            norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / (self.avg_len or 1.0))
            for term in terms:
                freq = tf.get(term, 0)
                if freq:
                    score += self.idf(term) * (freq * (self.k1 + 1)) / (freq + norm)
            out.append(score)
        return out
