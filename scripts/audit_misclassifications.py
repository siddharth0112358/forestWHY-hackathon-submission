#!/usr/bin/env python3
"""Diagnose remaining misclassifications: synonym-fixable vs model-wrong.

For every fine-tune sample where (a) the binary deforestation prediction
disagreed with truth, or (b) the predicted driver did not match the truth's
extracted driver, this script:

  1. Pulls the model's actual prose from the stored `raw.reasoning`.
  2. Searches the prose for keywords that would indicate the *correct* answer.
  3. Buckets the failure as:
       - SYNONYM_MISS  : model said the right thing in words my normaliser
                         didn't catch — fixable by extending vocabulary.
       - MODEL_WRONG   : model's prose contradicts the truth (interpretation
                         error, not extraction).
       - AMBIGUOUS     : neither — model's prose is unclear or mixed.

Usage:
    uv run python scripts/audit_misclassifications.py evals/<timestamp>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from forestwhy.evaluator import (  # noqa: E402
    _CLASS_SYNONYMS,
    _DRIVER_SYNONYMS,
    _conclusion_text,
    _count_synonym_hits,
)
from evaluate import PRED_TO_TRUTH_LABEL  # noqa: E402

# Map the v2 truth label back to our canonical class space so we can search
# its synonyms in the prose.
TRUTH_TO_CANON = {v: k for k, v in PRED_TO_TRUTH_LABEL.items()}


def _strong_signal(text: str, vocab_class: str, vocab) -> int:
    for cls, syns in vocab:
        if cls == vocab_class:
            return _count_synonym_hits(text, syns)
    return 0


def _bucket_class(prose: str, predicted: str | None, truth_canon: str) -> str:
    """Was the binary or 6-way miss a synonym issue or a real model error?"""
    text = prose.lower()
    conclusion = _conclusion_text(text)
    truth_hits = _strong_signal(conclusion, truth_canon, _CLASS_SYNONYMS)
    pred_hits = _strong_signal(conclusion, predicted or "", _CLASS_SYNONYMS) if predicted else 0
    if truth_hits >= 2 and pred_hits < truth_hits:
        return "SYNONYM_MISS"
    if pred_hits >= 1 and truth_hits == 0:
        return "MODEL_WRONG"
    if truth_hits == 0 and pred_hits == 0:
        return "AMBIGUOUS"
    return "MODEL_WRONG" if pred_hits > truth_hits else "SYNONYM_MISS"


def _bucket_driver(prose: str, predicted_driver: str | None, truth_driver: str) -> str:
    text = prose.lower()
    conclusion = _conclusion_text(text)
    truth_hits = _strong_signal(conclusion, truth_driver, _DRIVER_SYNONYMS)
    pred_hits = _strong_signal(conclusion, predicted_driver or "", _DRIVER_SYNONYMS) if predicted_driver else 0
    if truth_hits >= 2 and (predicted_driver != truth_driver):
        return "SYNONYM_MISS"
    if truth_hits == 0:
        return "MODEL_WRONG" if pred_hits >= 1 else "AMBIGUOUS"
    return "SYNONYM_MISS" if pred_hits < truth_hits else "MODEL_WRONG"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("eval_dir", type=Path)
    p.add_argument("--limit", type=int, default=0,
                   help="Print at most N example diagnostics (default: all).")
    args = p.parse_args()

    src = args.eval_dir / "results.json"
    if not src.exists():
        print(f"Not found: {src}", file=sys.stderr)
        sys.exit(1)
    data = json.loads(src.read_text())
    samples = data["samples"]

    binary_buckets = {"SYNONYM_MISS": 0, "MODEL_WRONG": 0, "AMBIGUOUS": 0}
    driver_buckets = {"SYNONYM_MISS": 0, "MODEL_WRONG": 0, "AMBIGUOUS": 0}
    binary_examples: list[dict] = []
    driver_examples: list[dict] = []

    for s in samples:
        truth = s["truth"]
        pred = s.get("finetuned") or {}
        truth_class = truth.get("change_class")
        truth_canon = TRUTH_TO_CANON.get(truth_class, "ambiguous")
        truth_def = truth.get("deforestation", False)

        # Pull model prose
        raw = pred.get("raw") or {}
        prose = ""
        if isinstance(raw, dict):
            prose = raw.get("reasoning") or pred.get("reasoning", "") or ""

        # Binary deforestation check
        pred_class = pred.get("change_class")
        pred_def = pred_class == "deforestation"
        if pred_def != truth_def:
            bucket = _bucket_class(prose, pred_class, truth_canon)
            binary_buckets[bucket] += 1
            binary_examples.append({
                "id": s.get("id"), "region": s.get("region"),
                "truth": truth_class, "pred": pred_class,
                "bucket": bucket, "prose_tail": _conclusion_text(prose)[:300],
            })

        # Driver check (only meaningful when truth has a driver and the model thinks deforestation)
        truth_driver = truth.get("driver_hypothesis") or "unknown"
        pred_driver = pred.get("driver_hypothesis") or "unknown"
        if truth_driver != "unknown" and pred_driver != truth_driver:
            bucket = _bucket_driver(prose, pred_driver, truth_driver)
            driver_buckets[bucket] += 1
            driver_examples.append({
                "id": s.get("id"), "region": s.get("region"),
                "truth_driver": truth_driver, "pred_driver": pred_driver,
                "bucket": bucket, "prose_tail": _conclusion_text(prose)[:300],
            })

    n = len(samples)
    print(f"Audited {n} samples\n")

    def _pct(buckets):
        total = sum(buckets.values())
        if total == 0:
            return "(no errors)"
        return ", ".join(f"{k}: {v} ({v/total:.0%})" for k, v in buckets.items())

    print(f"=== Binary deforestation errors ({sum(binary_buckets.values())} of {n}) ===")
    print(_pct(binary_buckets))
    print(f"\n=== Driver errors ({sum(driver_buckets.values())} of {n}) ===")
    print(_pct(driver_buckets))

    def _show(label, examples):
        print(f"\n--- {label} examples ---")
        per_bucket: dict[str, list[dict]] = {"SYNONYM_MISS": [], "MODEL_WRONG": [], "AMBIGUOUS": []}
        for e in examples:
            per_bucket[e["bucket"]].append(e)
        for bucket, items in per_bucket.items():
            if not items:
                continue
            print(f"\n  [{bucket}]  {len(items)} cases")
            limit = args.limit if args.limit else len(items)
            for e in items[:limit]:
                print(f"    #{e['id']} {e['region']}")
                if "truth_driver" in e:
                    print(f"      truth_driver={e['truth_driver']!r}  pred={e['pred_driver']!r}")
                else:
                    print(f"      truth={e['truth']!r}  pred={e['pred']!r}")
                tail = re.sub(r"\s+", " ", e["prose_tail"]).strip()
                print(f"      …{tail[:280]}")

    _show("Binary deforestation", binary_examples)
    _show("Driver", driver_examples)


if __name__ == "__main__":
    main()
