"""Evaluate the agent on 10 end-to-end cases.

Checks the tool sequence (did it actually look the order up?), the answer facts,
and the refusal behaviour. Tool choice is checked separately from answer quality
because "right answer, wrong path" is a coincidence, not a working agent.

Run:  python eval/run_agent_eval.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.loop import Agent
from app.agent.mcp_client import MCPDocsClient
from week1_baseline.common import ROOT, save_json

CASES = ROOT / "data" / "eval" / "agent_cases.json"
RESULTS_DIR = ROOT / "eval" / "results"


async def main() -> None:
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    mcp = MCPDocsClient()
    await mcp.start()
    agent = Agent(mcp=mcp)

    rows = []
    for case in cases:
        outcome = await agent.run(case["text"])
        answer_lower = outcome.answer.lower()
        tools_ok = all(tool in outcome.tool_calls for tool in case["tools_must_include"])
        facts_ok = (
            any(k.lower() in answer_lower for k in case["answer_contains"])
            if case["answer_contains"]
            else None
        )
        refusal_ok = outcome.refused == case["expect_refusal"]
        rows.append(
            {
                "id": case["id"],
                "text": case["text"],
                "answer": outcome.answer,
                "tools_called": outcome.tool_calls,
                "expected_tools": case["tools_must_include"],
                "tools_ok": tools_ok,
                "facts_ok": facts_ok,
                "refusal_ok": refusal_ok,
                "refused": outcome.refused,
                "sources": [source.citation for source in outcome.sources],
                "steps": [step.model_dump() for step in outcome.steps],
                "trace_id": outcome.trace_id,
                "latency_ms": outcome.latency_ms,
                "cost_usd": outcome.cost_usd,
                "note": case["note"],
            }
        )

    mcp_connected = mcp.status.connected
    await mcp.stop()

    passed = [
        row
        for row in rows
        if row["tools_ok"] and row["facts_ok"] is not False and row["refusal_ok"]
    ]
    summary = {
        "mode": agent.mode,
        "model": agent.llm.model,
        "mcp_connected": mcp_connected,
        "cases": len(rows),
        "passed": len(passed),
        "pass_rate": round(len(passed) / len(rows), 3),
        "tool_selection_accuracy": round(
            sum(1 for row in rows if row["tools_ok"]) / len(rows), 3
        ),
        "fact_accuracy": round(
            sum(1 for row in rows if row["facts_ok"] is not False) / len(rows), 3
        ),
        "refusal_accuracy": round(
            sum(1 for row in rows if row["refusal_ok"]) / len(rows), 3
        ),
        "avg_latency_ms": round(sum(row["latency_ms"] for row in rows) / len(rows)),
        "avg_steps": round(sum(len(row["steps"]) for row in rows) / len(rows), 2),
        "total_cost_usd": round(sum(row["cost_usd"] for row in rows), 6),
    }
    save_json(RESULTS_DIR / "agent.json", {"summary": summary, "rows": rows})

    print(
        f"\nmode={summary['mode']} mcp={summary['mcp_connected']} "
        f"passed {summary['passed']}/{summary['cases']} "
        f"tools {summary['tool_selection_accuracy']:.3f} "
        f"facts {summary['fact_accuracy']:.3f} "
        f"refusal {summary['refusal_accuracy']:.3f} "
        f"avg {summary['avg_latency_ms']} ms / {summary['avg_steps']} steps"
    )
    print("\nfailures:")
    for row in rows:
        if row in passed:
            continue
        reasons = []
        if not row["tools_ok"]:
            reasons.append(f"tools {row['tools_called']} missing {row['expected_tools']}")
        if row["facts_ok"] is False:
            reasons.append("answer missing expected fact")
        if not row["refusal_ok"]:
            reasons.append("wrong refusal behaviour")
        print(f"  {row['id']}: {'; '.join(reasons)}")
        print(f"     -> {row['answer'][:200]}")


if __name__ == "__main__":
    asyncio.run(main())
