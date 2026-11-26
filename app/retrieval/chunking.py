"""Chunk the markdown knowledge base into retrievable units.

Deliberately simple, and deliberately naive in one place: a section longer than
MAX_CHARS is cut on line boundaries, which can split a markdown table in the
middle. That behaviour is documented as a known failure in the README and is
visible in the product-spec questions in the evaluation. Making the chunker
table-aware is the next thing to build, not something to hide.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = ROOT / "data" / "docs"

MAX_CHARS = 420


@dataclass
class Chunk:
    id: str
    doc_id: str
    doc_title: str
    section: str
    text: str
    ordinal: int

    @property
    def citation(self) -> str:
        if self.section:
            return f"{self.doc_title} > {self.section}"
        return self.doc_title

    def as_dict(self) -> dict:
        data = asdict(self)
        data["citation"] = self.citation
        return data


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "section"


def chunk_markdown(doc_id: str, raw: str) -> list[Chunk]:
    lines = raw.splitlines()
    doc_title = doc_id.replace("-", " ").title()
    chunks: list[Chunk] = []
    section = ""
    buf: list[str] = []

    def flush() -> None:
        text = "\n".join(buf).strip()
        if not text:
            return
        ordinal = len(chunks) + 1
        chunks.append(
            Chunk(
                id=f"{doc_id}#{_slug(section or 'intro')}-{ordinal}",
                doc_id=doc_id,
                doc_title=doc_title,
                section=section,
                text=text,
                ordinal=ordinal,
            )
        )

    for line in lines:
        if line.startswith("# ") and not line.startswith("## "):
            doc_title = line[2:].strip()
            continue
        if line.startswith("## "):
            flush()
            buf = []
            section = line[3:].strip()
            continue
        buf.append(line)
        # Naive length-based cut: this is where a table can be torn in two.
        if sum(len(part) + 1 for part in buf) >= MAX_CHARS:
            flush()
            buf = []
    flush()
    return chunks


def load_chunks(docs_dir: Path | None = None) -> list[Chunk]:
    directory = docs_dir or DOCS_DIR
    chunks: list[Chunk] = []
    for path in sorted(directory.glob("*.md")):
        chunks.extend(chunk_markdown(path.stem, path.read_text(encoding="utf-8")))
    return chunks


class Corpus:
    """The chunk set plus lookups used by retrievers, tools and evaluations."""

    def __init__(self, chunks: list[Chunk] | None = None) -> None:
        self.chunks = chunks if chunks is not None else load_chunks()
        if not self.chunks:
            raise RuntimeError("Knowledge base is empty: no markdown files found")
        self.by_id = {chunk.id: chunk for chunk in self.chunks}
        self.docs = sorted({chunk.doc_id for chunk in self.chunks})

    def get(self, chunk_id: str) -> Chunk:
        return self.by_id[chunk_id]

    def __len__(self) -> int:
        return len(self.chunks)
