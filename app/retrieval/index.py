"""Three retrievers over the same chunks, so the Week 3 comparison is fair."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.retrieval.bm25 import BM25Index
from app.retrieval.chunking import Chunk, Corpus
from app.retrieval.text import tokenize


@dataclass
class Hit:
    chunk: Chunk
    score: float


class BM25Retriever:
    name = "bm25"

    def __init__(self, corpus: Corpus) -> None:
        self.corpus = corpus
        self.index = BM25Index(corpus.chunks)

    def search(self, query: str, k: int = 5) -> list[Hit]:
        scores = self.index.scores(query)
        order = np.argsort(scores)[::-1][:k]
        return [Hit(self.corpus.chunks[i], float(scores[i])) for i in order if scores[i] > 0]


class TfidfRetriever:
    name = "tfidf"

    def __init__(self, corpus: Corpus) -> None:
        self.corpus = corpus
        self.vectorizer = TfidfVectorizer(analyzer=tokenize, min_df=1, sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(
            [f"{c.doc_title} {c.section} {c.text}" for c in corpus.chunks]
        )

    def search(self, query: str, k: int = 5) -> list[Hit]:
        q = self.vectorizer.transform([query])
        scores = cosine_similarity(q, self.matrix).ravel()
        order = np.argsort(scores)[::-1][:k]
        return [Hit(self.corpus.chunks[i], float(scores[i])) for i in order if scores[i] > 0]


class LsaRetriever:
    """TF-IDF projected to 64 latent dimensions: cheap 'embedding' baseline."""

    name = "lsa"

    def __init__(self, corpus: Corpus, dims: int = 64) -> None:
        self.corpus = corpus
        self.vectorizer = TfidfVectorizer(analyzer=tokenize, min_df=1, sublinear_tf=True)
        matrix = self.vectorizer.fit_transform(
            [f"{c.doc_title} {c.section} {c.text}" for c in corpus.chunks]
        )
        dims = max(2, min(dims, matrix.shape[1] - 1, matrix.shape[0] - 1))
        self.svd = TruncatedSVD(n_components=dims, random_state=7)
        self.matrix = self.svd.fit_transform(matrix)

    def search(self, query: str, k: int = 5) -> list[Hit]:
        q = self.svd.transform(self.vectorizer.transform([query]))
        denom = np.linalg.norm(self.matrix, axis=1) * (np.linalg.norm(q) or 1.0)
        scores = (self.matrix @ q.ravel()) / np.where(denom == 0, 1.0, denom)
        order = np.argsort(scores)[::-1][:k]
        return [Hit(self.corpus.chunks[i], float(scores[i])) for i in order if scores[i] > 0]


class HybridRetriever:
    """BM25 and TF-IDF, both normalised, summed. The shipped retriever."""

    name = "hybrid"

    def __init__(self, corpus: Corpus) -> None:
        self.corpus = corpus
        self.bm25 = BM25Retriever(corpus)
        self.tfidf = TfidfRetriever(corpus)

    def search(self, query: str, k: int = 5) -> list[Hit]:
        bm = {hit.chunk.id: hit.score for hit in self.bm25.search(query, k=len(self.corpus))}
        tf = {hit.chunk.id: hit.score for hit in self.tfidf.search(query, k=len(self.corpus))}
        bm_max = max(bm.values(), default=0.0) or 1.0
        tf_max = max(tf.values(), default=0.0) or 1.0
        combined = []
        for chunk in self.corpus.chunks:
            score = 0.6 * bm.get(chunk.id, 0.0) / bm_max + 0.4 * tf.get(chunk.id, 0.0) / tf_max
            if score > 0:
                combined.append(Hit(chunk, round(score, 4)))
        combined.sort(key=lambda hit: -hit.score)
        return combined[:k]


RETRIEVERS = {
    "bm25": BM25Retriever,
    "tfidf": TfidfRetriever,
    "lsa": LsaRetriever,
    "hybrid": HybridRetriever,
}

DEFAULT_RETRIEVER = "hybrid"


def build_retriever(name: str = DEFAULT_RETRIEVER, corpus: Corpus | None = None):
    if name not in RETRIEVERS:
        raise ValueError(f"Unknown retriever {name!r}. Choose from {sorted(RETRIEVERS)}")
    return RETRIEVERS[name](corpus or Corpus())
