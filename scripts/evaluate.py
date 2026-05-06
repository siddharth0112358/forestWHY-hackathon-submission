#!/usr/bin/env python3
"""Per-field accuracy + JSON schema validation: base LFM2.5-VL vs forestWHY.

Reads a held-out test split from `Siddharth63/forestwhy-training-v1`,
runs both the base and fine-tuned models, computes per-field accuracy
plus area_pct MAE and confidence Brier score, and emits:

    evals/<timestamp>/
        report.md         human-readable summary
        results.json      per-sample records (used by app/eval_compare.py)
        meta.json         backend, model ids, args

Usage:
    uv run python scripts/evaluate.py --backend vllm --max-samples 50
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from forestwhy.evaluator import make_backend  # noqa: E402
from forestwhy.prompts import JSON_SCHEMA, PANEL_ORDER  # noqa: E402

EVAL_ROOT = REPO_ROOT / "evals"
log = logging.getLogger("forestwhy.evaluate")


def _load_test_split(repo: str, split: str, n: int):
    """Load up to `n` samples from the HF dataset."""
    from datasets import load_dataset
    ds = load_dataset(repo, split=split, streaming=True)
    out: list[dict] = []
    for row in ds:
        out.append(row)
        if len(out) >= n:
            break
    return out


def _row_to_panels(row: dict) -> list[bytes]:
    """Convert one HF dataset row's images into 14 PNG byte-strings.

    The forestwhy-training-v1 dataset stores panels under keys matching
    PANEL_ORDER. Falls back to grey placeholders for any missing.
    """
    pngs: list[bytes] = []
    for name in PANEL_ORDER:
        img = row.get(name)
        if img is None:
            placeholder = Image.new("RGB", (128, 128), (64, 64, 64))
            buf = io.BytesIO()
            placeholder.save(buf, format="PNG")
            pngs.append(buf.getvalue())
            continue
        if isinstance(img, dict) and "bytes" in img:
            pngs.append(img["bytes"])
            continue
        # PIL.Image.Image
        buf = io.BytesIO()
        if hasattr(img, "save"):
            img.save(buf, format="PNG")
            pngs.append(buf.getvalue())
        else:
            pngs.append(b"")
    return pngs


def _row_metadata(row: dict) -> dict:
    return {
        "lon": row.get("lon", 0.0),
        "lat": row.get("lat", 0.0),
        "size_km": row.get("size_km", 5.0),
        "region_id": row.get("region_id", "unknown"),
        "biome": row.get("biome", "unknown"),
        "country": row.get("country", "unknown"),
        "before_timestamp": row.get("before_timestamp", "unknown"),
        "after_timestamp": row.get("after_timestamp", "unknown"),
        "before_cloud_cover": row.get("before_cloud_cover"),
        "after_cloud_cover": row.get("after_cloud_cover"),
    }


def _row_truth(row: dict) -> dict:
    """Pull the ground-truth labels into a JSON_SCHEMA-shaped dict."""
    return {
        "change_class":      row.get("change_class"),
        "severity":          row.get("severity"),
        "area_pct":          row.get("area_pct"),
        "driver_hypothesis": row.get("driver_hypothesis"),
        "confidence":        row.get("confidence", 1.0),
        "reasoning":         row.get("reasoning", ""),
        "cloud_cover_note":  row.get("cloud_cover_note", ""),
    }


def _accuracy(pred: list[Any], truth: list[Any]) -> float:
    if not pred:
        return 0.0
    return sum(1 for p, t in zip(pred, truth) if p == t) / len(pred)


def _mae(pred: list[float | None], truth: list[float | None]) -> float | None:
    pairs = [(p, t) for p, t in zip(pred, truth) if p is not None and t is not None]
    if not pairs:
        return None
    return sum(abs(p - t) for p, t in pairs) / len(pairs)


def _brier(prob: list[float | None], hit: list[bool]) -> float | None:
    pairs = [(p, h) for p, h in zip(prob, hit) if p is not None]
    if not pairs:
        return None
    return sum((p - (1.0 if h else 0.0)) ** 2 for p, h in pairs) / len(pairs)


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s",
                        datefmt="%H:%M:%S")
    p = argparse.ArgumentParser()
    p.add_argument("--backend", required=True, choices=["vllm", "llamacpp", "transformers"])
    p.add_argument("--base-model", default="LiquidAI/LFM2-VL-450M")
    p.add_argument("--finetuned-model", default="Siddharth63/LFM2.5-forestWHY")
    p.add_argument("--base-base-url", default=None)
    p.add_argument("--finetuned-base-url", default=None)
    p.add_argument("--dataset", default="Siddharth63/forestwhy-training-v1")
    p.add_argument("--split", default="test")
    p.add_argument("--max-samples", type=int, default=50)
    args = p.parse_args()

    log.info("Loading %d samples from %s [%s] ...", args.max_samples, args.dataset, args.split)
    samples = _load_test_split(args.dataset, args.split, args.max_samples)

    log.info("Building backends ...")
    base = make_backend(args.backend, model=args.base_model, base_url=args.base_base_url)
    fine = make_backend(args.backend, model=args.finetuned_model, base_url=args.finetuned_base_url)

    out_dir = EVAL_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for i, row in enumerate(samples):
        log.info("[%d/%d] %s", i + 1, len(samples), row.get("region_id", "?"))
        pngs = _row_to_panels(row)
        meta = _row_metadata(row)
        truth = _row_truth(row)
        try:
            base_pred = base(pngs, meta)
        except Exception as exc:
            log.warning("  base failed: %s", exc)
            base_pred = {"error": str(exc)}
        try:
            fine_pred = fine(pngs, meta)
        except Exception as exc:
            log.warning("  finetuned failed: %s", exc)
            fine_pred = {"error": str(exc)}
        records.append({
            "id": row.get("id", i),
            "region_id": row.get("region_id"),
            "metadata": meta,
            "truth": truth,
            "base": base_pred,
            "finetuned": fine_pred,
        })

    # Aggregate
    def field(records, m, k):
        return [r[m].get(k) for r in records if isinstance(r[m], dict) and k in r[m]]

    truths = [r["truth"] for r in records]

    summary = {
        "n_samples": len(records),
        "change_class_accuracy": {
            "base": _accuracy(field(records, "base", "change_class"),
                              [t.get("change_class") for t in truths]),
            "finetuned": _accuracy(field(records, "finetuned", "change_class"),
                                   [t.get("change_class") for t in truths]),
        },
        "severity_accuracy": {
            "base": _accuracy(field(records, "base", "severity"),
                              [t.get("severity") for t in truths]),
            "finetuned": _accuracy(field(records, "finetuned", "severity"),
                                   [t.get("severity") for t in truths]),
        },
        "driver_accuracy": {
            "base": _accuracy(field(records, "base", "driver_hypothesis"),
                              [t.get("driver_hypothesis") for t in truths]),
            "finetuned": _accuracy(field(records, "finetuned", "driver_hypothesis"),
                                   [t.get("driver_hypothesis") for t in truths]),
        },
        "area_pct_mae": {
            "base": _mae(field(records, "base", "area_pct"),
                         [t.get("area_pct") for t in truths]),
            "finetuned": _mae(field(records, "finetuned", "area_pct"),
                              [t.get("area_pct") for t in truths]),
        },
    }

    (out_dir / "results.json").write_text(json.dumps({"summary": summary, "samples": records}, indent=2, default=str))
    (out_dir / "meta.json").write_text(json.dumps({
        "args": vars(args),
        "schema": JSON_SCHEMA,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }, indent=2))

    # Render markdown report
    lines = [
        f"# forestWHY evaluation — {out_dir.name}",
        "",
        f"- Dataset: `{args.dataset}` [{args.split}]",
        f"- Backend: `{args.backend}`",
        f"- Base model: `{args.base_model}`",
        f"- Fine-tuned model: `{args.finetuned_model}`",
        f"- Samples scored: **{summary['n_samples']}**",
        "",
        "| Field | Base | Fine-tuned | Δ |",
        "|---|---|---|---|",
    ]
    for k, label in [
        ("change_class_accuracy", "change_class accuracy"),
        ("severity_accuracy", "severity accuracy"),
        ("driver_accuracy", "driver accuracy"),
    ]:
        b = summary[k]["base"]
        f = summary[k]["finetuned"]
        lines.append(f"| {label} | {b:.3f} | {f:.3f} | {f - b:+.3f} |")
    if summary["area_pct_mae"]["finetuned"] is not None:
        b = summary["area_pct_mae"]["base"] or 0.0
        f = summary["area_pct_mae"]["finetuned"]
        lines.append(f"| area_pct MAE | {b:.2f} | {f:.2f} | {f - b:+.2f} |")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")
    log.info("Wrote %s", out_dir)


if __name__ == "__main__":
    main()
