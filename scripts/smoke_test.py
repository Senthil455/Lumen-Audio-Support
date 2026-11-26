"""Smoke test a running deployment - the deployed URL, not localhost.

The lesson's rule: a green deploy is not evidence. This hits the live service
the way a customer would and fails loudly if the contract is broken.

    python scripts/smoke_test.py https://your-service.up.railway.app

Checks: health, info, answer with citations, validation, rate limiting (429)
and the agent with its trace. Rate limiting uses a spoofed X-Forwarded-For, so
only run it when the service trusts the proxy header.
"""

from __future__ import annotations

import sys
import time

import httpx

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")
TIMEOUT = 60


def report(name: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}")
    return ok


def main() -> int:
    results = []
    with httpx.Client(base_url=BASE, timeout=TIMEOUT) as client:
        response = client.get("/health")
        results.append(
            report("health", response.status_code == 200 and response.json() == {"status": "ok"}, str(response.json()))
        )

        response = client.get("/info")
        info = response.json() if response.status_code == 200 else {}
        results.append(
            report(
                "info",
                response.status_code == 200 and info.get("documents", 0) > 0,
                f"mode={info.get('mode')} docs={info.get('documents')} mcp={info.get('mcp_connected')}",
            )
        )

        started = time.perf_counter()
        response = client.post("/ask", json={"text": "How long does a refund take after you receive the return?"})
        elapsed = int((time.perf_counter() - started) * 1000)
        body = response.json() if response.status_code == 200 else {}
        results.append(
            report(
                "ask returns answer with citations",
                response.status_code == 200 and bool(body.get("answer")) and bool(body.get("sources")),
                f"{elapsed} ms, {len(body.get('sources', []))} sources",
            )
        )

        short = client.post("/ask", json={"text": "hi"})
        long = client.post("/ask", json={"text": "x" * 501})
        results.append(report("validation 422 (short and long)", short.status_code == 422 and long.status_code == 422))

        headers = {"X-Forwarded-For": "198.51.100.77"}
        statuses = [
            client.post("/ask", json={"text": "What is the warranty on the Buds Lite?"}, headers=headers).status_code
            for _ in range(7)
        ]
        results.append(
            report("rate limit returns 429", 429 in statuses, f"statuses={statuses}")
        )

        response = client.post(
            "/agent", json={"text": "Where is order LA-10031?"}, headers={"X-Forwarded-For": "198.51.100.78"}
        )
        agent_body = response.json() if response.status_code == 200 else {}
        tools = [step.get("tool") for step in agent_body.get("steps", []) if step.get("tool")]
        results.append(
            report(
                "agent uses a tool and answers",
                response.status_code == 200 and bool(tools) and "in transit" in agent_body.get("answer", "").lower(),
                f"tools={tools}",
            )
        )

        trace_id = agent_body.get("trace_id")
        trace = client.get(f"/traces/{trace_id}") if trace_id else None
        results.append(
            report(
                "trace is retrievable and free of PII",
                bool(trace)
                and trace.status_code == 200
                and "LA-10031" not in trace.text
                and "example.com" not in trace.text,
                f"trace={trace_id}",
            )
        )

    passed = sum(results)
    print(f"\n{passed}/{len(results)} checks passed against {BASE}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
