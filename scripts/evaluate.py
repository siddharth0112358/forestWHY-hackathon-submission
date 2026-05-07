#!/usr/bin/env python3
"""Per-field accuracy: base LFM2.5-VL-1.6B vs Siddharth63/LFM2.5-forestWHY.

Reads a balanced 50/50 (deforestation / no-deforestation) subsample from
`Siddharth63/forestwhy-training-v2` and runs both models on the same panels.
Reports change-class accuracy, binary deforestation accuracy, area_pct MAE,
and severity accuracy. Writes:

    evals/<timestamp>/
        report.md         human-readable summary table
        results.json      per-sample records (used by app/eval_compare.py)
        meta.json         args + dataset/model identifiers

Note: the LFM2.5-forestWHY fine-tune was trained on a *private* curated set
`Siddharth63/forestwhy-combined-v1` (mix of v1 + curated driver examples).
We evaluate against `forestwhy-training-v2`, an out-of-distribution public
test set, to give an honest estimate of generalisation.

Usage:
    # Both models behind the same backend (e.g. one CUDA host with vLLM):
    uv run python scripts/evaluate.py \\
        --backend vllm --base-url https://<host>/v1 \\
        --max-samples 300

    # Mixed: fine-tune via local llama-server, base via remote vLLM:
    uv run python scripts/evaluate.py \\
        --finetuned-backend llamacpp --finetuned-base-url http://localhost:8080/v1 \\
        --base-backend vllm        --base-base-url https://<host>/v1 \\
        --max-samples 300

    # Fine-tune only (skip base comparison):
    uv run python scripts/evaluate.py \\
        --finetuned-backend llamacpp --finetuned-base-url http://localhost:8080/v1 \\
        --skip-base --max-samples 50
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from forestwhy.evaluator import make_backend, model_name  # noqa: E402
from forestwhy.prompts import JSON_SCHEMA, PANEL_ORDER  # noqa: E402

EVAL_ROOT = REPO_ROOT / "evals"
log = logging.getLogger("forestwhy.evaluate")


# ─────────────────────────────────────────────────────────────────────────────
# v2 dataset adapters
# ─────────────────────────────────────────────────────────────────────────────

# Map our canonical PANEL_ORDER (used at training time + by predict.py) to the
# image column names in `Siddharth63/forestwhy-training-v2`.
V2_IMAGE_KEYS: dict[str, str] = {
    "rgb_before":         "img_rgb_before",
    "rgb_after":          "img_rgb_after",
    "nir_fc_before":      "img_nir_false_color_before",
    "nir_fc_after":       "img_nir_false_color_after",
    "swir_before":        "img_swir_composite_before",
    "swir_after":         "img_swir_composite_after",
    "delta_ndvi":         "img_delta_ndvi",
    "delta_nbr":          "img_delta_nbr",
    "attention_multi":    "img_attention_multi",
    "embedding_change":   "img_embedding_change",
    "cropa_roads":        "img_cropa_roads",
    "delta_attn_role":    "img_delta_attn_role",
    "head_disagreement":  "img_head_disagreement",
    "pca_semantic":       "img_pca_semantic",
}

# v2 dataset label vocabulary (different from the model's emitted classes):
#   active_front_anthropogenic  ~46% — this is the deforestation class
#   recovery_or_afforestation
#   stable_forest_intact
#   non_forest_negative
#   stable_forest_managed
#   fire_or_disturbance
TRUTH_DEFORESTATION_LABEL = "active_front_anthropogenic"

# Model's emitted change_class is mapped onto the v2 truth label space here, so
# accuracy is measured on the same vocabulary.
PRED_TO_TRUTH_LABEL: dict[str, str] = {
    "deforestation":     "active_front_anthropogenic",
    "fire_disturbance":  "fire_or_disturbance",
    "afforestation":     "recovery_or_afforestation",
    "stable_forest":     "stable_forest_intact",
    "stable_non_forest": "non_forest_negative",
    "ambiguous":         "ambiguous",
}


def _row_to_panels(row: dict) -> list[bytes]:
    """Convert a v2 row's images into 14 PNG byte-strings in PANEL_ORDER."""
    pngs: list[bytes] = []
    for name in PANEL_ORDER:
        col = V2_IMAGE_KEYS.get(name, name)
        img = row.get(col)
        if img is None:
            buf = io.BytesIO()
            Image.new("RGB", (128, 128), (64, 64, 64)).save(buf, format="PNG")
            pngs.append(buf.getvalue())
            continue
        if isinstance(img, dict) and "bytes" in img:
            pngs.append(img["bytes"])
            continue
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
        "size_km": 5.0,
        "region_id": row.get("region", "unknown"),
        "biome": "tropical",
        "country": row.get("region", "unknown"),
        "before_timestamp": f"{row.get('year_before', 'unknown')}-07-15T00:00:00Z",
        "after_timestamp":  f"{row.get('year_after',  'unknown')}-07-15T00:00:00Z",
        "before_cloud_cover": None,
        "after_cloud_cover": None,
    }


def _row_truth(row: dict) -> dict:
    return {
        "change_class":      row.get("label"),
        "severity":          row.get("change_magnitude"),
        "area_pct":          row.get("affected_area_percent"),
        "driver_hypothesis": _coerce_driver(row),
        "deforestation":     row.get("label") == TRUTH_DEFORESTATION_LABEL,
        "reasoning":         row.get("assistant_text", ""),
    }


def _coerce_driver(row: dict) -> str:
    """Try to extract a coarse driver from v2's free-text fields."""
    text = " ".join(filter(None, [
        row.get("step8_driver_synthesis"),
        row.get("causal_mechanism"),
        row.get("alternative_drivers"),
    ])).lower()
    for k in (
        "agricultural_clearing", "agriculture", "soy", "cattle", "ranch", "pasture",
        "logging", "selective", "timber",
        "mining", "gold", "ore",
        "fire", "burn", "wildfire",
        "flood", "river",
        "plantation", "palm", "rubber",
        "natural_regrowth", "regrowth", "secondary",
        "road", "infrastructure",
    ):
        if k in text:
            for canonical, keys in {
                "agricultural_clearing": ["agricultural_clearing", "agriculture", "soy", "cattle", "ranch", "pasture"],
                "logging_road":          ["logging", "selective", "timber", "road"],
                "mining":                ["mining", "gold", "ore"],
                "fire":                  ["fire", "burn", "wildfire"],
                "flood":                 ["flood", "river"],
                "plantation":            ["plantation", "palm", "rubber"],
                "natural_regrowth":      ["natural_regrowth", "regrowth", "secondary"],
            }.items():
                if k in keys:
                    return canonical
    return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Balanced sampling
# ─────────────────────────────────────────────────────────────────────────────

def _balanced_sample(repo: str, split: str, n_per_class: int, seed: int = 42) -> list[dict]:
    """Pull n_per_class rows of `deforestation` and n_per_class rows of the rest.

    Iterates the dataset sequentially without shuffle (`.shuffle(buffer_size=...)`
    on a streaming dataset triggers repeated re-downloads of the same parquet
    shard on some HF storage backends). The dataset is heterogeneous enough
    that any contiguous window contains both classes.
    """
    from datasets import load_dataset
    ds = load_dataset(repo, split=split, streaming=True)
    deforestation: list[dict] = []
    other: list[dict] = []
    target = n_per_class
    seen = 0
    for row in ds:
        seen += 1
        label = row.get("label")
        if label == TRUTH_DEFORESTATION_LABEL and len(deforestation) < target:
            deforestation.append(row)
        elif label is not None and label != TRUTH_DEFORESTATION_LABEL and len(other) < target:
            other.append(row)
        if len(deforestation) >= target and len(other) >= target:
            break
        if seen % 200 == 0:
            log.info("  scanned %d rows (deforestation=%d/%d, other=%d/%d)",
                     seen, len(deforestation), target, len(other), target)
    rng = random.Random(seed)
    out = deforestation + other
    rng.shuffle(out)
    log.info("Sampled %d deforestation + %d other = %d total (scanned %d rows)",
             len(deforestation), len(other), len(out), seen)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def _accuracy(pred: Iterable, truth: Iterable) -> float:
    pairs = [(p, t) for p, t in zip(pred, truth) if p is not None and t is not None]
    if not pairs:
        return 0.0
    return sum(1 for p, t in pairs if p == t) / len(pairs)


def _mae(pred: Iterable, truth: Iterable) -> float | None:
    pairs = [(p, t) for p, t in zip(pred, truth) if p is not None and t is not None]
    if not pairs:
        return None
    return sum(abs(float(p) - float(t)) for p, t in pairs) / len(pairs)


def _binary_deforestation_pred(pred: dict) -> bool | None:
    cc = pred.get("change_class")
    if cc is None:
        return None
    return cc == "deforestation"


def _mapped_change_class(pred: dict) -> Optional[str]:
    cc = pred.get("change_class")
    if cc is None:
        return None
    return PRED_TO_TRUTH_LABEL.get(cc, cc)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s  %(message)s",
                        datefmt="%H:%M:%S")
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="Siddharth63/forestwhy-training-v2")
    p.add_argument("--split", default="train",
                   help="v2 has only a `train` split; we subsample.")
    p.add_argument("--max-samples", type=int, default=300,
                   help="Total samples (split 50/50 between deforestation and not).")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--base-model", default="LiquidAI/LFM2.5-VL-1.6B")
    p.add_argument("--finetuned-model", default="Siddharth63/LFM2.5-forestWHY")

    # One backend covers both unless overridden:
    p.add_argument("--backend", default=None,
                   choices=["vllm", "llamacpp", "transformers"],
                   help="Default backend for both base and fine-tuned (override per-side below).")
    p.add_argument("--base-url", default=None)

    p.add_argument("--base-backend", default=None,
                   choices=["vllm", "llamacpp", "transformers"])
    p.add_argument("--base-base-url", default=None)
    p.add_argument("--finetuned-backend", default=None,
                   choices=["vllm", "llamacpp", "transformers"])
    p.add_argument("--finetuned-base-url", default=None)

    p.add_argument("--skip-base", action="store_true",
                   help="Skip the base-model side (fine-tune-only eval).")
    args = p.parse_args()

    if args.max_samples < 2 or args.max_samples % 2 != 0:
        log.warning("--max-samples should be even and ≥ 2; rounding up.")
        args.max_samples = max(2, args.max_samples + (args.max_samples % 2))
    n_per_class = args.max_samples // 2

    # ------------------- sample -----------------------------------------------
    log.info("Loading %d balanced samples (50/50 deforestation/other) from %s [%s] ...",
             args.max_samples, args.dataset, args.split)
    samples = _balanced_sample(args.dataset, args.split, n_per_class, seed=args.seed)
    if not samples:
        log.error("No samples loaded — check dataset access.")
        sys.exit(1)

    # ------------------- backends ---------------------------------------------
    base_backend = args.base_backend or args.backend
    base_url_for_base = args.base_base_url or args.base_url
    fine_backend = args.finetuned_backend or args.backend
    base_url_for_fine = args.finetuned_base_url or args.base_url

    if not args.skip_base and not base_backend:
        log.error("Base side has no backend. Pass --backend or --base-backend, or use --skip-base.")
        sys.exit(1)
    if not fine_backend:
        log.error("Fine-tuned side has no backend. Pass --backend or --finetuned-backend.")
        sys.exit(1)

    log.info("Building backends ...")
    fine = make_backend(fine_backend, model=args.finetuned_model, base_url=base_url_for_fine)
    if args.skip_base:
        base = None
        log.info("--skip-base: only the fine-tuned side will be scored.")
    else:
        base = make_backend(base_backend, model=args.base_model, base_url=base_url_for_base)

    # ------------------- score -------------------------------------------------
    out_dir = EVAL_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Writing results to %s", out_dir)

    records: list[dict] = []
    incremental_path = out_dir / "results_partial.json"
    for i, row in enumerate(samples):
        log.info("[%d/%d] label=%s region=%s",
                 i + 1, len(samples), row.get("label"), row.get("region"))
        pngs = _row_to_panels(row)
        meta = _row_metadata(row)
        truth = _row_truth(row)

        base_pred: dict
        if base is None:
            base_pred = {"skipped": True}
        else:
            try:
                base_pred = base(pngs, meta)
            except Exception as exc:
                log.warning("  base failed: %s", exc)
                base_pred = {"error": str(exc)}

        try:
            fine_pred = fine(pngs, meta)
        except Exception as exc:
            log.warning("  fine-tuned failed: %s", exc)
            fine_pred = {"error": str(exc)}

        records.append({
            "id": row.get("pair_id", i),
            "region": row.get("region"),
            "year_before": row.get("year_before"),
            "year_after": row.get("year_after"),
            "metadata": meta,
            "truth": truth,
            "base": base_pred,
            "finetuned": fine_pred,
        })

        # Incremental write so partial runs survive process kills / hangs.
        try:
            incremental_path.write_text(
                json.dumps({"samples": records}, indent=2, default=str)
            )
        except Exception as exc:
            log.warning("  incremental write failed: %s", exc)

    # ------------------- aggregate --------------------------------------------
    truths = [r["truth"] for r in records]

    def col(side: str, k: str):
        return [r[side].get(k) if isinstance(r[side], dict) else None for r in records]

    summary: dict[str, Any] = {
        "n_samples": len(records),
        "n_deforestation_truth": sum(1 for t in truths if t["deforestation"]),
        "n_other_truth": sum(1 for t in truths if not t["deforestation"]),
    }

    metrics_rows = [
        ("change_class accuracy (mapped)", "change_class",
         lambda side: _accuracy(
             [_mapped_change_class(r[side]) if isinstance(r[side], dict) else None
              for r in records],
             [t["change_class"] for t in truths])),
        ("binary deforestation acc.",  "deforestation",
         lambda side: _accuracy(
             [_binary_deforestation_pred(r[side]) if isinstance(r[side], dict) else None
              for r in records],
             [t["deforestation"] for t in truths])),
        ("driver accuracy",            "driver_hypothesis",
         lambda side: _accuracy(col(side, "driver_hypothesis"),
                                [t["driver_hypothesis"] for t in truths])),
        ("area_pct MAE (lower=better)", "area_pct",
         lambda side: _mae(col(side, "area_pct"), [t["area_pct"] for t in truths])),
    ]
    for label, _key, fn in metrics_rows:
        summary[label] = {
            "base":      None if base is None else fn("base"),
            "finetuned": fn("finetuned"),
        }

    # ------------------- persist ----------------------------------------------
    (out_dir / "results.json").write_text(
        json.dumps({"summary": summary, "samples": records}, indent=2, default=str)
    )
    (out_dir / "meta.json").write_text(json.dumps({
        "args": vars(args),
        "schema": JSON_SCHEMA,
        "panel_order": list(PANEL_ORDER),
        "v2_image_keys": V2_IMAGE_KEYS,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }, indent=2))

    # Markdown report
    lines = [
        f"# forestWHY evaluation — {out_dir.name}",
        "",
        f"- Dataset: `{args.dataset}` [{args.split}]",
        f"- Sampling: 50/50 deforestation / other ({summary['n_deforestation_truth']}/{summary['n_other_truth']} actual)",
        f"- Base model: `{args.base_model}` ({base_backend or 'skipped'})",
        f"- Fine-tuned: `{args.finetuned_model}` ({fine_backend})",
        f"- Samples scored: **{summary['n_samples']}**",
        "",
        "| Metric | Base | Fine-tuned | Δ |",
        "|---|---|---|---|",
    ]
    for label, _, _ in metrics_rows:
        b = summary[label]["base"]
        f = summary[label]["finetuned"]
        if b is None and f is None:
            continue
        b_s = "—" if b is None else (f"{b:.3f}" if isinstance(b, float) else str(b))
        f_s = "—" if f is None else (f"{f:.3f}" if isinstance(f, float) else str(f))
        d_s = ""
        if isinstance(b, (int, float)) and isinstance(f, (int, float)):
            d_s = f"{f - b:+.3f}"
        lines.append(f"| {label} | {b_s} | {f_s} | {d_s} |")

    (out_dir / "report.md").write_text("\n".join(lines) + "\n")
    log.info("Wrote %s", out_dir)


if __name__ == "__main__":
    main()
