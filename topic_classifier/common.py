"""Shared helpers for the Week 1 baseline and the Week 2 comparison."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
ARTIFACTS = ROOT / "artifacts"

LABELS = [
    "account",
    "payments",
    "products",
    "promotions",
    "refunds",
    "returns",
    "shipping",
    "support",
    "warranty",
    "out_of_scope",
]


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_split(name: str) -> tuple[list[str], list[str]]:
    rows = load_jsonl(DERIVED / name)
    return [r["text"] for r in rows], [r["label"] for r in rows]


def load_handwritten() -> tuple[list[str], list[str]]:
    rows = load_jsonl(ROOT / "data" / "eval" / "topic_test.jsonl")
    return [r["text"] for r in rows], [r["label"] for r in rows]


def summarise(y_true, y_pred, labels: list[str]) -> dict:
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    per_class = {
        label: {
            "precision": round(report[label]["precision"], 3),
            "recall": round(report[label]["recall"], 3),
            "f1": round(report[label]["f1-score"], 3),
            "support": int(report[label]["support"]),
        }
        for label in labels
    }
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    # Report the mistakes a human would care about, not the whole matrix.
    confusions = []
    for i, actual in enumerate(labels):
        for j, predicted in enumerate(labels):
            if i != j and matrix[i][j]:
                confusions.append(
                    {"actual": actual, "predicted": predicted, "count": int(matrix[i][j])}
                )
    confusions.sort(key=lambda c: -c["count"])
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 3),
        "macro_f1": round(float(report["macro avg"]["f1-score"]), 3),
        "per_class": per_class,
        "top_confusions": confusions[:5],
        "n": len(y_true),
    }


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")
