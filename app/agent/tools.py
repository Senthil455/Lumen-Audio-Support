"""The three tools the agent can call.

    search_docs         the knowledge base, also delivered over MCP
    lookup_order        the order database (fake, personal data already masked)
    escalate_to_human   creates a ticket and returns its reference

Each tool returns a short text observation for the model plus structured data
for the trace. Tools never raise: a failure becomes an observation the agent
has to handle, which is what a real tool failure looks like.
"""

from __future__ import annotations

import json
import random
import string
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from app.config import ROOT, settings
from app.retrieval import Corpus, build_retriever

ORDERS_PATH = ROOT / "data" / "orders.json"
TICKETS_PATH = settings.runtime_dir / "tickets.jsonl"
MAX_OBSERVATION_CHARS = 1800

TOOL_SCHEMAS = [
    {
        "name": "search_docs",
        "description": (
            "Search the Lumen Audio help centre (refunds, returns, warranty, "
            "shipping, payments, account, promotions, products, support). "
            "Use for any policy question."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search terms"},
                "k": {"type": "integer", "default": 4, "minimum": 1, "maximum": 6},
            },
            "required": ["query"],
        },
    },
    {
        "name": "lookup_order",
        "description": (
            "Look up a Lumen Audio order by its id, for example LA-10023. "
            "Returns status, dates, items, refund eligibility and warranty."
        ),
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    },
    {
        "name": "escalate_to_human",
        "description": (
            "Create a support ticket for a human agent. Use when the customer "
            "asks to speak to a person, or the issue cannot be resolved from the "
            "policy and order data."
        ),
        "parameters": {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
        },
    },
]


@dataclass
class ToolResult:
    ok: bool
    observation: str
    data: dict = field(default_factory=dict)


class ToolRegistry:
    def __init__(self, corpus: Corpus | None = None) -> None:
        self.corpus = corpus or Corpus()
        self.retriever = build_retriever("hybrid", self.corpus)
        self.orders = json.loads(ORDERS_PATH.read_text(encoding="utf-8"))
        TICKETS_PATH.parent.mkdir(parents=True, exist_ok=True)

    # -- tools -----------------------------------------------------------
    def search_docs(self, query: str, k: int = 4) -> ToolResult:
        hits = self.retriever.search(query, k=max(1, min(int(k), 6)))
        if not hits:
            return ToolResult(
                ok=True,
                observation="No matching documents in the help centre.",
                data={"hits": []},
            )
        lines = []
        payload = []
        for index, hit in enumerate(hits, start=1):
            text = " ".join(hit.chunk.text.split())
            lines.append(f"[{index}] {hit.chunk.citation} (score {hit.score:.2f})\n{text}")
            payload.append(
                {
                    "id": hit.chunk.id,
                    "citation": hit.chunk.citation,
                    "score": round(hit.score, 4),
                    "text": text,
                }
            )
        observation = "\n\n".join(lines)[:MAX_OBSERVATION_CHARS]
        return ToolResult(ok=True, observation=observation, data={"hits": payload})

    def lookup_order(self, order_id: str) -> ToolResult:
        key = (order_id or "").strip().upper()
        order = self.orders.get(key)
        if not order:
            known = ", ".join(sorted(self.orders))
            return ToolResult(
                ok=False,
                observation=f"No order found with id {key or '(missing)'}. Known demo ids: {known}.",
                data={"found": False},
            )
        items = ", ".join(f"{i['qty']}x {i['name']}" for i in order["items"])
        observation = (
            f"Order {order['order_id']} for {order['customer']} ({order['email_masked']}).\n"
            f"Status: {order['status']}. Placed {order['placed']}."
            + (f" Delivered {order['delivered']}." if order.get("delivered") else "")
            + f"\nCarrier: {order['carrier'] or 'not assigned'}"
            + (f", tracking {order['tracking']}" if order.get("tracking") else "")
            + f"\nItems: {items}. Total {order['total_usd']:.2f} USD.\n"
            f"Payment: {order['payment_method']}.\n"
            f"Refund eligible until: {order['refund_eligible_until'] or 'not yet delivered'}.\n"
            f"Warranty: {order['warranty']}.\nNotes: {order['notes']}"
        )
        return ToolResult(ok=True, observation=observation, data={"order": order})

    def escalate_to_human(self, summary: str) -> ToolResult:
        reference = "LA-" + "".join(random.choices(string.digits, k=6))
        TICKETS_PATH.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ticket": reference,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            # The summary is stored for the human agent, never logged or returned
            # in traces. Traces only keep the length of the summary.
            "summary": summary[:500],
            "summary_chars": len(summary),
            "sla": "first response within 4 business hours",
        }
        with TICKETS_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        return ToolResult(
            ok=True,
            observation=(
                f"Ticket {reference} created. A support specialist will reply within "
                "4 business hours, Monday to Friday 9am-6pm Eastern Time. "
                f"Reference: {reference}."
            ),
            data={"ticket": reference, "sla": record["sla"]},
        )

    # -- dispatch --------------------------------------------------------
    def execute(self, name: str, args: dict) -> ToolResult:
        handler = {
            "search_docs": self.search_docs,
            "lookup_order": self.lookup_order,
            "escalate_to_human": self.escalate_to_human,
        }.get(name)
        if handler is None:
            return ToolResult(ok=False, observation=f"Unknown tool {name!r}.")
        try:
            return handler(**args)
        except TypeError as exc:
            return ToolResult(ok=False, observation=f"Bad arguments for {name}: {exc}")
        except Exception as exc:  # a tool failure is data, not a crash
            return ToolResult(ok=False, observation=f"Tool {name} failed: {type(exc).__name__}")


def describe_args(name: str, args: dict) -> str:
    """A trace-safe summary: argument names and sizes, never the values."""
    if not args:
        return ""
    parts = []
    for key, value in args.items():
        if isinstance(value, str):
            parts.append(f"{key}={len(value)} chars")
        else:
            parts.append(f"{key}={value!r}")
    return ", ".join(parts)


__all__ = [
    "TOOL_SCHEMAS",
    "TICKETS_PATH",
    "Path",
    "ToolRegistry",
    "ToolResult",
    "describe_args",
]
