#!/usr/bin/env python3
"""Convert `Siddharth63/forestwhy-training-v1` to leap-finetune JSONL + image dir.

Output format (one JSON line per sample):
    {
      "messages": [{role: "system", content: "..."}, {role: "user", content: "..."}],
      "images": ["images/<id>/rgb_before.png", ..., "images/<id>/pca_semantic.png"],
      "completion": <ground-truth JSON serialised>
    }

Usage:
    uv run python scripts/prepare_dataset.py --out_dir ./dataset_out --split train
    uv run python scripts/prepare_dataset.py --out_dir ./dataset_out --split val
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from forestwhy.prompts import PANEL_ORDER, SYSTEM_PROMPT, render_user_prompt  # noqa: E402

log = logging.getLogger("forestwhy.prepare_dataset")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s",
                        datefmt="%H:%M:%S")
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="Siddharth63/forestwhy-training-v1")
    p.add_argument("--split", default="train")
    p.add_argument("--out-dir", default="./dataset_out")
    p.add_argument("--max-samples", type=int, default=0, help="0 = all")
    args = p.parse_args()

    out = Path(args.out_dir)
    images_dir = out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out / f"{args.split}.jsonl"

    from datasets import load_dataset
    log.info("Streaming %s [%s] ...", args.dataset, args.split)
    ds = load_dataset(args.dataset, split=args.split, streaming=True)

    n = 0
    with jsonl_path.open("w") as f:
        for row in ds:
            sample_id = str(row.get("id", n))
            sample_dir = images_dir / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)
            image_paths: list[str] = []
            for name in PANEL_ORDER:
                img = row.get(name)
                if img is None:
                    log.warning("  sample %s missing panel %s; padding grey", sample_id, name)
                    img = Image.new("RGB", (128, 128), (64, 64, 64))
                if isinstance(img, dict) and "bytes" in img:
                    img = Image.open(io.BytesIO(img["bytes"]))
                target = sample_dir / f"{name}.png"
                if hasattr(img, "save"):
                    img.save(target)
                image_paths.append(str(target.relative_to(out)))

            metadata = {
                "lon": row.get("lon", 0.0), "lat": row.get("lat", 0.0),
                "size_km": row.get("size_km", 5.0),
                "region_id": row.get("region_id", "unknown"),
                "biome": row.get("biome", "unknown"),
                "country": row.get("country", "unknown"),
                "before_timestamp": row.get("before_timestamp", "unknown"),
                "after_timestamp": row.get("after_timestamp", "unknown"),
                "before_cloud_cover": row.get("before_cloud_cover"),
                "after_cloud_cover": row.get("after_cloud_cover"),
            }
            user_text = render_user_prompt(metadata)
            completion = json.dumps({
                "change_class":      row.get("change_class"),
                "severity":          row.get("severity"),
                "area_pct":          row.get("area_pct"),
                "driver_hypothesis": row.get("driver_hypothesis"),
                "confidence":        row.get("confidence", 1.0),
                "reasoning":         row.get("reasoning", ""),
                "cloud_cover_note":  row.get("cloud_cover_note", ""),
            }, separators=(",", ":"))

            record = {
                "id": sample_id,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_text},
                ],
                "images": image_paths,
                "completion": completion,
            }
            f.write(json.dumps(record) + "\n")
            n += 1
            if args.max_samples and n >= args.max_samples:
                break

    log.info("Wrote %d records to %s", n, jsonl_path)


if __name__ == "__main__":
    main()
