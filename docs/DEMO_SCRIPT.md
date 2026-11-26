# Two-minute demo script

Record with the API and UI already running. Keep the cursor moving: the demo is
the product, not the slides.

| Time | On screen | Say |
| --- | --- | --- |
| 0:00-0:15 | README top, then the Streamlit UI | "Lumen Audio sells headphones. This is the support copilot: it answers from the help centre, cites the passage, and refuses when the answer is not there. It is one FastAPI service plus a Streamlit UI, and it runs with zero API cost." |
| 0:15-0:35 | Click the refund example, press **Ask** | "Opened items: 14 days, 15% restocking fee. Three sources, ranked, with the exact policy text under each one. The model never sees a document it did not retrieve." |
| 0:35-0:55 | Click the military-discount example | "This is the part that matters. There is no military discount in the docs, so it refuses and offers a route out instead of inventing a policy. A support bot that guesses is worse than no bot." |
| 0:55-1:20 | Agent tab: "My LA-10023 parcel arrived damaged" | "The agent picks tools: it looks up the order, then searches the docs. 48-hour damage window, photo, replacement. The steps panel and the trace id are the receipts." |
| 1:20-1:40 | Terminal: `python eval/run_agent_eval.py`, then `eval/RESULTS.md` | "Ten agent cases, nine pass. Tool choice is 10 out of 10. The one failure is a question whose answer is split between the returns policy and the warranty - the agent retrieves both and uses one." |
| 1:40-2:00 | `/stats` or the usage line, then README limitations | "Rate limit is five a minute per caller, the daily cap is fifty cents, and logs carry latency, tokens, cost and a hashed caller id - never the question. What I would change next: a table-aware chunker and a generated refusal decision, because retrieval scores cannot tell an unanswerable question from an answerable one." |

## Before recording

* `uvicorn app.main:app --port 8000` and `API_URL=http://127.0.0.1:8000 streamlit run ui/streamlit_app.py`.
* Run `python scripts/smoke_test.py http://127.0.0.1:8000` once so you know the
  demo path is working, then close the terminal you ran it in.
* Clear the UI with the sidebar **Clear** button so the first answer is fresh.
* If you demo the live deployment, use `scripts/smoke_test.py <deployed-url>`
  and the deployed UI, never localhost.

## The two sentences to get right

1. "It refuses when the answer is not in the documents" - the differentiator.
2. "Nine out of ten, and here is the one that fails" - the credibility.
