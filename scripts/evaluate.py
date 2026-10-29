"""Offline translation-quality evaluation.

Two complementary signals:

1. BLEU (sacreBLEU) against reference translations written by a human.
2. A model-based judge that flags defects and returns a 1-5 score.

Run: python scripts/evaluate.py            (uses the configured model)
     python scripts/evaluate.py --offline  (no model calls, BLEU only)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from translator.service import build_default_service  # noqa: E402

DATASET = ROOT / "examples" / "evaluation_set.json"


def load_dataset() -> list[dict]:
    return json.loads(DATASET.read_text(encoding="utf-8"))


def bleu(hypotheses: list[str], references: list[str]) -> float:
    import sacrebleu

    return sacrebleu.corpus_bleu(hypotheses, [references]).score


def judge_score(review: str) -> int:
    """Map a free-text critique to a coarse 1-5 score.

    The judge is instructed to answer "No issues found." for a clean output and
    to list defects otherwise. We convert the number of defect bullets into a
    score so the evaluation is machine-readable.
    """

    if re.search(r"no issues", review, re.IGNORECASE):
        return 5
    bullets = len(re.findall(r"^\s*[-*•]\s+", review, re.MULTILINE))
    return max(1, 5 - bullets)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="Skip model calls.")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    dataset = load_dataset()
    if args.limit:
        dataset = dataset[: args.limit]

    if args.offline:
        print("Offline mode: no hypotheses available without a model. "
              "Run without --offline to generate and score translations.")
        return 0

    service = build_default_service()
    hypotheses, references, rows = [], [], []
    for item in dataset:
        result = service.translate(item["source"], item["target_lang"], item["source_lang"])
        hypotheses.append(result.text)
        references.append(item["reference"])
        score = judge_score(
            service.review(item["source"], result.text, item["source_lang"], item["target_lang"])
        )
        rows.append(
            {
                "source": item["source"],
                "hypothesis": result.text,
                "reference": item["reference"],
                "judge_score": score,
            }
        )
        print(f"{item['source_lang']}->{item['target_lang']}  judge={score}/5  {result.text[:70]}")

    corpus = bleu(hypotheses, references)
    average_judge = sum(r["judge_score"] for r in rows) / len(rows)
    print(f"\nBLEU: {corpus:.2f}")
    print(f"Average judge score: {average_judge:.2f}/5")

    report = {
        "bleu": corpus,
        "average_judge_score": average_judge,
        "samples": len(rows),
        "rows": rows,
    }
    out = ROOT / "docs" / "evaluation_results.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report written to {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
