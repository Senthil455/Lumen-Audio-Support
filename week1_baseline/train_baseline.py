"""Week 1 - the baseline: TF-IDF + logistic regression for topic routing.

Why this exists: before any model API or agent, the question "which part of the
docs is this about, and is it in scope at all?" was answered by a cheap
classifier. It is kept in the repository because the progression from a
scikit-learn baseline to a deployed agent is the thing being assessed.

Run:  python week1_baseline/build_dataset.py
      python week1_baseline/train_baseline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from week1_baseline.common import (
    ARTIFACTS,
    LABELS,
    load_handwritten,
    load_split,
    save_json,
    summarise,
)


def build_pipeline() -> Pipeline:
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
            ("clf", LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")),
        ]
    )


def main() -> None:
    x_train, y_train = load_split("train.jsonl")
    x_dev, y_dev = load_split("dev.jsonl")
    x_test, y_test = load_handwritten()

    model = build_pipeline()
    model.fit(x_train, y_train)

    dev_metrics = summarise(y_dev, model.predict(x_dev), LABELS)
    test_metrics = summarise(y_test, model.predict(x_test), LABELS)

    payload = {
        "model": "tfidf_word_1_2 + logistic_regression(C=4)",
        "train_rows": len(x_train),
        "eval": {
            "template_dev": dev_metrics,
            "handwritten_40": test_metrics,
        },
        "headline": {
            "dev_accuracy": dev_metrics["accuracy"],
            "handwritten_accuracy": test_metrics["accuracy"],
            "handwritten_macro_f1": test_metrics["macro_f1"],
        },
    }
    save_json(ARTIFACTS / "metrics_week1.json", payload)

    # The baseline is the model that ships: Week 2's MLP did not beat it on the
    # hand-written set, so the simpler, cheaper model is the production router.
    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump({"pipeline": model, "labels": LABELS}, ARTIFACTS / "topic_model.joblib")
    print(f"wrote {(ARTIFACTS / 'topic_model.joblib').relative_to(ARTIFACTS.parent)}")

    print(
        f"\nWeek 1 baseline  dev(acc)={dev_metrics['accuracy']:.3f}  "
        f"handwritten(acc)={test_metrics['accuracy']:.3f}  "
        f"macro_f1={test_metrics['macro_f1']:.3f}"
    )
    for mistake in test_metrics["top_confusions"]:
        print(
            f"  confused: {mistake['actual']} -> {mistake['predicted']} "
            f"x{mistake['count']}"
        )


if __name__ == "__main__":
    main()
