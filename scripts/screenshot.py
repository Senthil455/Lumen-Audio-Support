"""Capture the UI with Chrome over the DevTools protocol.

Streamlit is a websocket-driven SPA, so `chrome --screenshot` fires before the
page has rendered and saves the loading skeleton. This waits for real text to
appear, then captures. Dev-only helper; not used by the app or the deploy.

    python scripts/screenshot.py "http://localhost:8501/?q=..." docs/screenshot.png
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request

from websockets.sync.client import connect

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    "/usr/bin/google-chrome",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
]
PORT = 9222


def chrome_path() -> str:
    for path in CHROME_CANDIDATES:
        if os.path.exists(path):
            return path
    raise SystemExit("Chrome not found; pass a URL to open manually instead.")


def main() -> None:
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501/"
    out = sys.argv[2] if len(sys.argv) > 2 else "docs/screenshot.png"
    wait_for = sys.argv[3] if len(sys.argv) > 3 else "Sources"

    profile = tempfile.mkdtemp(prefix="cdp-profile-")
    chrome = subprocess.Popen(
        [
            chrome_path(),
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            f"--remote-debugging-port={PORT}",
            "--remote-allow-origins=*",
            f"--user-data-dir={profile}",
            "--window-size=1280,1800",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json/version", timeout=1):
                    break
            except Exception:
                time.sleep(0.3)
        else:
            raise SystemExit("Chrome DevTools did not start")

        request = urllib.request.Request(
            f"http://127.0.0.1:{PORT}/json/new?{urllib.parse.quote(url, safe='')}",
            method="PUT",
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            target = json.load(response)

        with connect(target["webSocketDebuggerUrl"], max_size=40 * 1024 * 1024) as ws:
            counter = {"id": 0}

            def send(method: str, params: dict | None = None) -> dict:
                counter["id"] += 1
                ws.send(json.dumps({"id": counter["id"], "method": method, "params": params or {}}))
                while True:
                    message = json.loads(ws.recv())
                    if message.get("id") == counter["id"]:
                        return message

            send("Page.enable")
            # Streamlit fills the viewport and scrolls internally, so a tall
            # emulated viewport is what actually shows the whole answer.
            send(
                "Emulation.setDeviceMetricsOverride",
                {
                    "width": 1280,
                    "height": 2200,
                    "deviceScaleFactor": 1,
                    "mobile": False,
                },
            )
            send("Page.navigate", {"url": url})

            ready = False
            for _ in range(90):
                time.sleep(0.5)
                text = send(
                    "Runtime.evaluate",
                    {
                        "expression": (
                            "JSON.stringify({text: document.body.innerText,"
                            " skeletons: document.querySelectorAll('[data-testid=\\\"stSkeleton\\\"]').length})"
                        ),
                        "returnByValue": True,
                    },
                )
                raw = text.get("result", {}).get("result", {}).get("value") or "{}"
                state = json.loads(raw)
                body = state.get("text", "")
                if wait_for in body and "Looking through" not in body and state.get("skeletons", 0) == 0:
                    ready = True
                    break
            print("rendered content found:", ready)

            metrics = send("Page.getLayoutMetrics")
            size = metrics.get("result", {}).get("cssContentSize", {}) or {}
            height = min(int(size.get("height", 1800)), 1600)
            shot = send(
                "Page.captureScreenshot",
                {
                    "format": "png",
                    "captureBeyondViewport": True,
                    "clip": {
                        "x": 0,
                        "y": 0,
                        "width": int(size.get("width", 1280)),
                        "height": height,
                        "scale": 1,
                    },
                },
            )
            data = shot.get("result", {}).get("data")
            if not data:
                raise SystemExit("screenshot failed")
            with open(out, "wb") as fh:
                fh.write(base64.b64decode(data))
            print(f"wrote {out} ({os.path.getsize(out)} bytes)")
    finally:
        chrome.terminate()


if __name__ == "__main__":
    main()
