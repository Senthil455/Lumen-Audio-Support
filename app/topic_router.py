"""The Week 1 baseline classifier, used at runtime.

It does two jobs in the shipped pipeline:
  * tags the response with the topic, which the UI shows and logs aggregate;
  * refuses obviously off-topic questions before spending a model call.

It is a weak signal on purpose - 90% on hand-written questions - so a single
"out of scope" prediction never blocks a request unless the probability is high.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib

from app.config import ROOT

ARTIFACT = ROOT / "artifacts" / "topic_model.joblib"
OUT_OF_SCOPE = "out_of_scope"
MIN_BLOCK_PROBABILITY = 0.60

log = logging.getLogger("app")


class TopicRouter:
    def __init__(self, artifact: Path = ARTIFACT) -> None:
        self.available = False
        self.pipeline = None
        self.labels: list[str] = []
        if artifact.exists():
            try:
                bundle = joblib.load(artifact)
                self.pipeline = bundle["pipeline"]
                self.labels = bundle.get("labels", [])
                self.available = True
            except Exception as exc:  # a broken artifact must not take the API down
                log.warning("topic router unavailable: %s", exc)

    def classify(self, text: str) -> tuple[str | None, float]:
        if not self.available or self.pipeline is None:
            return None, 0.0
        probabilities = self.pipeline.predict_proba([text])[0]
        classes = list(self.pipeline.classes_)
        best = int(probabilities.argmax())
        return str(classes[best]), round(float(probabilities[best]), 3)

    def blocks(self, topic: str | None, confidence: float) -> bool:
        return topic == OUT_OF_SCOPE and confidence >= MIN_BLOCK_PROBABILITY
