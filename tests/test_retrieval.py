from __future__ import annotations

from app.retrieval import Corpus, build_retriever
from app.retrieval.chunking import chunk_markdown


def test_chunker_splits_on_headings():
    raw = "# Doc\nIntro text here.\n## One\nFirst section body is long enough.\n## Two\nSecond section body."
    chunks = chunk_markdown("doc", raw)
    assert [c.section for c in chunks] == ["", "One", "Two"]
    assert chunks[1].id.startswith("doc#one-")
    assert chunks[1].citation == "Doc > One"


def test_every_retriever_finds_the_refund_window():
    corpus = Corpus()
    for name in ("bm25", "tfidf", "lsa", "hybrid"):
        hits = build_retriever(name, corpus).search("opened item return window", k=3)
        assert hits, name
        assert any("refund" in hit.chunk.doc_id for hit in hits), (
            name,
            [hit.chunk.doc_id for hit in hits],
        )


def test_corpus_covers_every_document():
    corpus = Corpus()
    assert len(corpus.docs) == 9
    assert all(chunk.text.strip() for chunk in corpus.chunks)


def test_retrieval_returns_nothing_for_gibberish():
    hits = build_retriever("hybrid").search("zzzq qqqv", k=3)
    assert hits == []
