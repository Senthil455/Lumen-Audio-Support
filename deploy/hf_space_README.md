---
title: Lumen Audio Support Copilot
emoji: 🎧
colorFrom: indigo
colorTo: blue
sdk: streamlit
sdk_version: 1.67.0
app_file: ui/streamlit_app.py
pinned: false
---

# Lumen Audio Support Copilot - UI

Streamlit front end for the FastAPI service in this repository. It calls the
deployed API; it does not host the model or the knowledge base.

## Setting it up as a Space

1. Create a Streamlit Space and upload `ui/streamlit_app.py` (keep the `ui/`
   folder, or change `app_file` to match).
2. In the Space settings, add a secret named `API_URL` with the Railway URL of
   the API, for example `https://lumen-support-api.up.railway.app`.
3. Add `requirements.txt` containing:

   ```
   streamlit==1.67.0
   requests==2.32.5
   ```

The app shows the API mode, the answer, its citations, the agent steps and the
per-request cost. If the API is unreachable it says so instead of pretending.
