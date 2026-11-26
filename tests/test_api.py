from __future__ import annotations

import json
from pathlib import Path

from app.budget import BudgetExceeded, BudgetLedger
from app.config import settings


def test_health_is_exactly_ok(client, fresh_ip):
    response = client.get("/health", headers=fresh_ip())
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_info_reports_a_working_corpus(client, fresh_ip):
    body = client.get("/info", headers=fresh_ip()).json()
    assert body["documents"] >= 9
    assert body["chunks"] > 20
    assert body["mode"] == "offline-extractive"
    assert "search_docs" in body["tools"]


def test_ask_returns_answer_and_citations(client, fresh_ip):
    response = client.post(
        "/ask",
        json={"text": "How long do I have to return an opened item?"},
        headers=fresh_ip(),
    )
    assert response.status_code == 200
    body = response.json()
    assert "14" in body["answer"]
    assert body["sources"], "a grounded answer must cite at least one source"
    assert body["sources"][0]["citation"]
    assert body["usage"]["mode"] == "offline-extractive"
    assert body["usage"]["latency_ms"] >= 0


def test_short_question_is_rejected(client, fresh_ip):
    response = client.post("/ask", json={"text": "hi"}, headers=fresh_ip())
    assert response.status_code == 422


def test_long_question_is_rejected(client, fresh_ip):
    response = client.post("/ask", json={"text": "x" * 501}, headers=fresh_ip())
    assert response.status_code == 422


def test_whitespace_only_question_is_rejected(client, fresh_ip):
    response = client.post("/ask", json={"text": "     "}, headers=fresh_ip())
    assert response.status_code == 422


def test_unanswerable_question_refuses_without_sources(client, fresh_ip):
    response = client.post(
        "/ask", json={"text": "What is the CEO's personal phone number?"}, headers=fresh_ip()
    )
    assert response.status_code == 200
    body = response.json()
    assert body["refused"] is True
    assert body["sources"] == []


def test_rate_limit_returns_429_for_a_single_caller(client, fresh_ip):
    headers = {"X-Forwarded-For": "203.0.113.9"}
    statuses = [
        client.post("/ask", json={"text": "What is the refund policy?"}, headers=headers).status_code
        for _ in range(7)
    ]
    assert 429 in statuses, statuses
    assert statuses.count(200) == 5, statuses
    limited = client.post("/ask", json={"text": "What is the refund policy?"}, headers=headers)
    assert limited.status_code == 429
    assert "limit" in limited.json()


def test_agent_uses_tools_and_writes_a_trace(client, fresh_ip):
    response = client.post(
        "/agent",
        json={"text": "Where is my order LA-10031?"},
        headers=fresh_ip(),
    )
    assert response.status_code == 200
    body = response.json()
    assert "lookup_order" in [step["tool"] for step in body["steps"] if step["tool"]]
    assert "in transit" in body["answer"].lower()
    trace_id = body["trace_id"]
    assert trace_id

    trace = client.get(f"/traces/{trace_id}", headers=fresh_ip())
    assert trace.status_code == 200
    record = trace.json()
    assert record["trace_id"] == trace_id
    assert record["tool_calls"]
    # Traces are safe to share: no question text, no customer fields, no
    # order identifiers. Derived state such as the status is kept for debugging.
    assert "Where is my order" not in trace.text
    assert "LA-10031" not in trace.text
    assert "R. Mensah" not in trace.text
    assert "example.com" not in trace.text


def test_unknown_trace_is_404(client, fresh_ip):
    assert client.get("/traces/doesnotexist", headers=fresh_ip()).status_code == 404


def test_logs_never_contain_the_question(client, fresh_ip):
    secret_question = "My email is jane.doe@example.com and my refund is late"
    response = client.post("/ask", json={"text": secret_question}, headers=fresh_ip())
    assert response.status_code in (200, 429)

    log_file = Path(settings.log_dir) / "app.jsonl"
    contents = log_file.read_text(encoding="utf-8")
    assert "jane.doe@example.com" not in contents
    assert "my refund is late" not in contents.lower()
    assert "q_len" in contents


def test_budget_cap_blocks_the_next_call():
    ledger = BudgetLedger(limit_usd=0.01)
    ledger.charge(0.01)
    try:
        ledger.check(0.01)
    except BudgetExceeded:
        pass
    else:  # pragma: no cover
        raise AssertionError("budget should have refused the call")


def test_budget_snapshot_maths():
    ledger = BudgetLedger(limit_usd=0.5)
    ledger.charge(0.125)
    snapshot = ledger.snapshot()
    assert snapshot["spent_usd"] == 0.125
    assert snapshot["remaining_usd"] == 0.375
    assert snapshot["calls"] == 1


def test_traces_file_has_no_document_bodies():
    trace_path = Path(settings.log_dir) / "traces.jsonl"
    if not trace_path.exists():
        return
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        assert set(record) & {"question", "answer", "text"} == set()
        assert record["question_chars"] > 0
