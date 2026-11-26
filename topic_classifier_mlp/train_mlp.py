"""Week 2 - the deep learning comparison.

Week 2 asked for a neural model next to the baseline. The honest answer for a
dataset this small (480 templated rows) is that a transformer would memorise
the templates, so the comparison uses a small multi-layer perceptron over the
same TF-IDF features. Same inputs, same split, different model - that is the
only way the comparison means anything.

Outcome on the hand-written test set is stored in artifacts/metrics_week2.json.
The script also saves the MLP so the winner can be shipped.

Run:  python topic_classifier_mlp/train_mlp.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from topic_classifier.common import (
    ARTIFACTS,
    LABELS,
    load_handwritten,
    load_split,
    save_json,
    summarise,
)


def build_mlp() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                    stop_words="english",
                ),
            ),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(128, 64),
                    activation="relu",
                    alpha=1e-4,
                    learning_rate_init=3e-3,
                    batch_size=32,
                    max_iter=800,
                    early_stopping=True,
                    n_iter_no_change=25,
                    random_state=7,
                ),
            ),
        ]
    )


def main() -> None:
    x_train, y_train = load_split("train.jsonl")
    x_dev, y_dev = load_split("dev.jsonl")
    x_test, y_test = load_handwritten()

    model = build_mlp()
    model.fit(x_train, y_train)

    dev_metrics = summarise(y_dev, model.predict(x_dev), LABELS)
    test_metrics = summarise(y_test, model.predict(x_test), LABELS)

    baseline_path = ARTIFACTS / "metrics_week1.json"
    baseline_dev = baseline_acc = None
    if baseline_path.exists():
        import json

        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        baseline_dev = baseline["headline"]["dev_accuracy"]
        baseline_acc = baseline["headline"]["handwritten_accuracy"]

    payload = {
        "model": "tfidf_word_1_2 + MLP(128, 64, relu, early_stopping)",
        "train_rows": len(x_train),
        "eval": {"template_dev": dev_metrics, "handwritten_40": test_metrics},
        "headline": {
            "dev_accuracy": dev_metrics["accuracy"],
            "handwritten_accuracy": test_metrics["accuracy"],
            "handwritten_macro_f1": test_metrics["macro_f1"],
            "baseline_dev_accuracy": baseline_dev,
            "baseline_handwritten_accuracy": baseline_acc,
        },
        "verdict": None,
    }
    payload["verdict"] = (
        "MLP wins"
        if baseline_acc is not None and test_metrics["accuracy"] > baseline_acc
        else "Baseline wins or ties"
    )
    save_json(ARTIFACTS / "metrics_week2.json", payload)

    ARTIFACTS.mkdir(exist_ok=True)
    mlp_path = ARTIFACTS / "topic_model_mlp.joblib"
    joblib.dump({"pipeline": model, "labels": LABELS}, mlp_path)
    print(f"wrote {mlp_path.relative_to(ARTIFACTS.parent)}")

    print(
        f"\nWeek 2 MLP      dev(acc)={dev_metrics['accuracy']:.3f}  "
        f"handwritten(acc)={test_metrics['accuracy']:.3f}  "
        f"macro_f1={test_metrics['macro_f1']:.3f}"
    )
    if baseline_acc is not None:
        print(f"Baseline       dev(acc)={baseline_dev:.3f}  handwritten(acc)={baseline_acc:.3f}")
        print(f"Verdict: {payload['verdict']} on the hand-written set.")
    print(f"MLP train accuracy after {model.named_steps['mlp'].n_iter_} epochs: "
          f"{accuracy_score(y_train, model.predict(x_train)):.3f}")


if __name__ == "__main__":
    main()
