"""Tokenisation shared by every retriever, so comparisons are fair."""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can",
    "could", "did", "do", "does", "for", "from", "get", "got", "had", "has",
    "have", "how", "i", "if", "in", "into", "is", "it", "its", "just", "me",
    "my", "no", "not", "of", "on", "or", "our", "out", "should", "so", "than",
    "that", "the", "their", "them", "then", "there", "these", "they", "this",
    "to", "up", "us", "was", "we", "were", "what", "when", "where", "which",
    "who", "why", "will", "with", "would", "you", "your", "im", "ive", "want",
}


def stem(word: str) -> str:
    """A deliberately small suffix stripper.

    Enough to match "refund"/"refunds" and "ship"/"shipping" without dragging in
    NLTK. It is approximate by design - errors here are part of the retrieval
    failure analysis.
    """
    if len(word) > 6 and word.endswith("ing"):
        return word[:-3]
    if len(word) > 5 and word.endswith("ed"):
        return word[:-2]
    if len(word) > 4 and word.endswith("es") and not word.endswith("ses"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def tokenize(text: str, keep_stopwords: bool = False) -> list[str]:
    words = _TOKEN.findall(text.lower())
    if not keep_stopwords:
        words = [w for w in words if w not in STOPWORDS]
    return [stem(w) for w in words]
