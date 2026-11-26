"""Week 1 data work: build the topic-routing dataset.

The production assistant needs to know which part of the knowledge base a
question is about, and whether it is out of scope at all. Week 1 solved this
with a small supervised classifier, and the same idea is still used in the
shipped pipeline (app/topic_router.py) to tag each request.

This script writes two files under data/derived/:

    train.jsonl   480 template questions, 48 per label (synthetic)
    dev.jsonl     120 template questions from a different template family

The hand-written questions in data/eval/topic_test.jsonl are NEVER used for
training. They are the honest check: templated questions are easy, a person
phrasing the same question is not.

Run:  python week1_baseline/build_dataset.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "derived"
random.seed(7)

# One phrase bank per label. Kept short on purpose - the point of Week 1 is a
# cheap baseline, not data collection.
PHRASES = {
    "refunds": [
        "refunds", "the refund policy", "getting my money back",
        "a refund to my card", "the restocking fee", "refund timing",
    ],
    "returns": [
        "returns", "the RMA process", "sending an item back",
        "returning opened ear tips", "the 30-day return window",
        "return shipping labels",
    ],
    "warranty": [
        "the warranty", "a warranty claim", "the serial number",
        "out-of-warranty repairs", "battery capacity coverage",
        "warranty in another country",
    ],
    "shipping": [
        "shipping", "express delivery", "same-day delivery",
        "international shipping", "the delivery cut-off", "tracking my parcel",
    ],
    "payments": [
        "payments", "Klarna instalments", "the authorisation hold",
        "a declined card", "currency conversion", "when my card is charged",
    ],
    "account": [
        "my account", "two-factor authentication", "resetting my password",
        "changing my email address", "deleting my account", "exporting my data",
    ],
    "promotions": [
        "discounts", "the student discount", "stacking coupons",
        "the referral credit", "price adjustments", "the Black Friday sale",
    ],
    "products": [
        "the product range", "Buds Pro specifications", "the Mini Speaker",
        "the Studio headphones", "product prices", "ear cushion replacements",
    ],
    "support": [
        "contacting support", "support opening hours", "the phone number",
        "escalating a ticket", "response times", "ticket references",
    ],
    "out_of_scope": [
        "laptops", "phone plans", "the weather", "stock prices",
        "writing code", "a recipe",
    ],
}

TRAIN_STEMS = [
    "what is the policy on {}",
    "how long does {} take",
    "can you explain {}",
    "I have a question about {}",
    "tell me about {}",
    "how does {} work",
    "what are the rules for {}",
    "is there a limit on {}",
]

DEV_STEMS = [
    "I need help with {}",
    "question about {}",
    "where can I read about {}",
    "what should I know about {}",
    "how do I handle {}",
]


def build(stems: list[str], per_label: int) -> list[dict]:
    rows: list[dict] = []
    for label, phrases in PHRASES.items():
        combos = [(s, p) for s in stems for p in phrases]
        random.shuffle(combos)
        for stem, phrase in combos[:per_label]:
            rows.append({"text": stem.format(phrase), "label": label})
    random.shuffle(rows)
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train = build(TRAIN_STEMS, per_label=48)
    dev = build(DEV_STEMS, per_label=12)
    for name, rows in (("train.jsonl", train), ("dev.jsonl", dev)):
        path = OUT_DIR / name
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        print(f"wrote {len(rows):4d} rows -> {path.relative_to(ROOT)}")

    counts: dict[str, int] = {}
    for row in train:
        counts[row["label"]] = counts.get(row["label"], 0) + 1
    print("train label distribution:", dict(sorted(counts.items())))


if __name__ == "__main__":
    main()
