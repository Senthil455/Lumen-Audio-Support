"""Streamlit UI for the support copilot.

Talk to the deployed API, not to the local one: set API_URL to the Railway URL.
The app shows the answer, the citations it used, the agent's steps and the
per-request cost, because those are the things that make a demo believable.
"""

from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT = 45

st.set_page_config(page_title="Lumen Audio Support Copilot", page_icon="🎧", layout="centered")


@st.cache_data(ttl=30)
def api_info() -> dict | None:
    try:
        response = requests.get(f"{API_URL}/info", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def ask(endpoint: str, question: str) -> tuple[dict | None, str | None]:
    try:
        response = requests.post(
            f"{API_URL}/{endpoint}", json={"text": question}, timeout=TIMEOUT
        )
    except requests.RequestException as exc:
        return None, f"Cannot reach the API: {exc.__class__.__name__}"
    if response.status_code == 422:
        return None, "Question must be between 3 and 500 characters."
    if response.status_code == 429:
        detail = response.json().get("detail", "Rate limit reached.")
        return None, f"{detail} Try again in a minute."
    if response.status_code == 502:
        return None, "The model provider is unavailable. Please retry."
    if not response.ok:
        return None, f"Request failed with HTTP {response.status_code}."
    return response.json(), None


info = api_info()

st.title("Lumen Audio support copilot")
st.caption(
    "Answers are grounded in a nine-document help centre. Every answer shows the "
    "passages it came from, and refuses when the policy does not cover the question."
)

if info:
    mode_label = "live model" if info["mode"] == "llm" else "offline extractive"
    left, middle, right = st.columns(3)
    left.metric("Mode", mode_label)
    middle.metric("Knowledge base", f"{info['documents']} docs")
    right.metric("Daily budget left", f"${info['budget']['remaining_usd']:.2f}")
    if not info["mcp_connected"]:
        st.warning("MCP server is not connected - the agent is using the local search tool.")
else:
    st.error(f"API not reachable at {API_URL}. Start it with `uvicorn app.main:app`.")

examples = [
    "How long do I have to return an opened item?",
    "My LA-10023 parcel arrived damaged. What are my options?",
    "What is the warranty on the Buds Lite?",
    "Do you offer a military discount?",
]

st.write("**Try one:**")
cols = st.columns(2)
for index, example in enumerate(examples):
    if cols[index % 2].button(example, use_container_width=True):
        st.session_state["question"] = example

with st.form("ask_form"):
    question = st.text_area(
        "Your question",
        value=st.session_state.get("question", ""),
        height=90,
        max_chars=500,
        placeholder="e.g. Can I return the Buds Lite after 30 days?",
    )
    col_ask, col_agent = st.columns(2)
    asked = col_ask.form_submit_button("Ask", type="primary", use_container_width=True)
    agented = col_agent.form_submit_button("Ask the agent", use_container_width=True)

if asked or agented:
    if len(question.strip()) < 3:
        st.warning("Please type a question of at least 3 characters.")
    else:
        endpoint = "agent" if agented else "ask"
        with st.spinner("Looking through the help centre..."):
            data, error = ask(endpoint, question.strip())
        if error:
            st.error(error)
        else:
            st.session_state["result"] = data
            st.session_state["question"] = question.strip()

# Shareable link: /?q=<question>&mode=agent runs the query on load.
params = st.query_params
if "q" in params and "result" not in st.session_state:
    prefilled = str(params["q"]).strip()
    if len(prefilled) >= 3:
        with st.spinner("Looking through the help centre..."):
            data, error = ask(str(params.get("mode", "ask")), prefilled)
        if data:
            st.session_state["result"] = data
            st.session_state["question"] = prefilled
        elif error:
            st.error(error)

result = st.session_state.get("result")
if result:
    if result["refused"]:
        st.warning(result["answer"])
        st.caption(f"Refused: {result['refusal_reason']}")
    else:
        st.markdown("### Answer")
        st.write(result["answer"])

    if result.get("sources"):
        st.markdown("### Sources")
        for source in result["sources"]:
            with st.expander(f"{source['citation']}  (score {source['score']:.2f})"):
                st.write(source["snippet"])
                st.caption(source["id"])

    if result.get("steps"):
        st.markdown("### Agent steps")
        for step in result["steps"]:
            if step["action"] == "tool":
                icon = "✅" if step["ok"] else "⚠️"
                st.write(f"{icon} **{step['tool']}** - {step['observation'] or ''}")
                if step.get("args_summary"):
                    st.caption(step["args_summary"])
            else:
                st.write("🏁 final answer")
        if result.get("trace_id"):
            st.caption(f"Trace `{result['trace_id']}` - GET {API_URL}/traces/{result['trace_id']}")

    usage = result["usage"]
    st.caption(
        f"{usage['model']} · {usage['mode']} · {usage['latency_ms']} ms · "
        f"{usage['input_tokens'] + usage['output_tokens']} tokens · "
        f"${usage['cost_usd']:.6f}"
    )

with st.sidebar:
    st.header("About this demo")
    st.write(
        "A RAG service with per-caller rate limits, a daily cost cap, structured "
        "logs and an agent that can search docs, look up an order and escalate "
        "to a human. One tool, `search_docs`, is served over MCP."
    )
    st.write(
        "Without an LLM key the service runs a deterministic extractive fallback, "
        "so the demo costs nothing and never calls out."
    )
    if st.button("Clear"):
        st.session_state.pop("result", None)
        st.session_state.pop("question", None)
        st.rerun()
