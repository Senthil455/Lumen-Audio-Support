"""MCP server exposing the knowledge-base search tool over stdio.

This is the "one tool delivered through MCP" requirement. The server owns the
retrieval index; the API process does not import it for this path. Running it
as a separate process is what makes the boundary real rather than decorative:

    python -m app.mcp_server

The client in app/agent/mcp_client.py speaks the protocol to it.
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from app.retrieval import Corpus, build_retriever

server = MCPServer(
    "lumen-audio-docs",
    instructions=(
        "Knowledge base tools for Lumen Audio customer support. "
        "search_docs returns policy and product passages with citations."
    ),
)

_corpus: Corpus | None = None
_retriever = None


def _get_retriever():
    global _corpus, _retriever
    if _retriever is None:
        _corpus = Corpus()
        _retriever = build_retriever("hybrid", _corpus)
    return _retriever


@server.tool()
def search_docs(query: str, k: int = 4) -> str:
    """Search Lumen Audio policy, shipping, warranty and product documents.

    Args:
        query: What to look for, for example "refund window for opened items".
        k: How many passages to return, 1 to 6.
    """
    hits = _get_retriever().search(query, k=max(1, min(int(k), 6)))
    if not hits:
        return "No matching documents in the help centre."
    blocks = []
    for index, hit in enumerate(hits, start=1):
        text = " ".join(hit.chunk.text.split())
        # The id is included so the client can attribute citations without a
        # second round trip. Format is parsed in app/agent/loop.py.
        blocks.append(
            f"[{index}] {hit.chunk.id} | {hit.chunk.citation} "
            f"(score {hit.score:.2f})\n{text}"
        )
    return "\n\n".join(blocks)


@server.tool()
def knowledge_base_stats() -> str:
    """Report how many documents and passages the knowledge base contains."""
    corpus = _get_retriever().corpus
    return f"{len(corpus.docs)} documents, {len(corpus.chunks)} passages: {', '.join(corpus.docs)}"


def main() -> None:
    server.run()


if __name__ == "__main__":
    main()
