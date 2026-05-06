#!/usr/bin/env python3
"""Re-run the VLM on existing predictions, reusing saved panels.

For each row in predictions.db, load the 14 PNG panels from db_images/<id>/
and call the active backend (vLLM or llama-server) with the current prompt
and max_tokens settings. Updates the row in place — no SimSat fetch, no
JEPA inference.

Use this when you've changed `prompts.py`, the normaliser, or the backend's
generation settings (e.g. max_tokens) and want to refresh existing rows
without paying the full pipeline cost.

Usage:
    uv run python scripts/reinfer_existing.py --backend llamacpp --base-url http://localhost:8080/v1
    uv run python scripts/reinfer_existing.py --backend llamacpp --base-url http://localhost:8080/v1 --ids 1 5 12
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from forestwhy.evaluator import make_backend, model_name  # noqa: E402
from forestwhy.prompts import PANEL_ORDER  # noqa: E402

DB_PATH = REPO_ROOT / "predictions.db"
log = logging.getLogger("forestwhy.reinfer_existing")


def _row_metadata(row: sqlite3.Row) -> dict:
    return {
        "lon": row["lon"], "lat": row["lat"],
        "size_km": row["size_km"],
        "region_id": row["region_id"], "biome": "tropical", "country": "n/a",
        "before_timestamp": row["before_timestamp"],
        "after_timestamp": row["after_timestamp"],
        "before_cloud_cover": row["before_cloud_cover"],
        "after_cloud_cover": row["after_cloud_cover"],
    }


def _load_panels(panels_dir: Path) -> list[bytes]:
    pngs: list[bytes] = []
    for name in PANEL_ORDER:
        path = panels_dir / f"{name}.png"
        if not path.exists():
            raise RuntimeError(f"missing panel {path}")
        pngs.append(path.read_bytes())
    return pngs


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s  %(message)s",
                        datefmt="%H:%M:%S")
    p = argparse.ArgumentParser()
    p.add_argument("--backend", required=True,
                   choices=["vllm", "llamacpp", "transformers"])
    p.add_argument("--model", default=None)
    p.add_argument("--base-url", default=None)
    p.add_argument("--ids", nargs="*", type=int, default=None,
                   help="Only refresh these row ids; default = all rows.")
    args = p.parse_args()

    if not DB_PATH.exists():
        log.error("DB not found: %s", DB_PATH)
        sys.exit(1)

    predict = make_backend(args.backend, model=args.model, base_url=args.base_url)
    label = model_name(args.backend, args.model)

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    sql = "SELECT * FROM predictions"
    if args.ids:
        sql += f" WHERE id IN ({','.join('?' * len(args.ids))})"
    sql += " ORDER BY id"
    rows = list(conn.execute(sql, args.ids or ()))
    log.info("Re-inferring %d row(s) ...", len(rows))

    successes = 0
    failures = 0
    for r in rows:
        if not r["panels_dir"]:
            log.warning("[%d %s] no panels_dir, skipping", r["id"], r["region_id"])
            continue
        panels_dir = REPO_ROOT / r["panels_dir"]
        if not panels_dir.is_dir():
            log.warning("[%d %s] panels_dir missing on disk: %s", r["id"], r["region_id"], panels_dir)
            continue

        try:
            pngs = _load_panels(panels_dir)
        except Exception as exc:
            log.warning("[%d %s] panels load failed: %s", r["id"], r["region_id"], exc)
            failures += 1
            continue

        meta = _row_metadata(r)
        log.info("[%d %s] -> %s ...", r["id"], r["region_id"], label)
        try:
            pred = predict(pngs, meta)
        except Exception as exc:
            log.warning("[%d %s] inference failed: %s", r["id"], r["region_id"], exc)
            failures += 1
            continue

        conn.execute(
            """UPDATE predictions
               SET change_class      = ?,
                   severity          = ?,
                   area_pct          = ?,
                   driver_hypothesis = ?,
                   confidence        = ?,
                   reasoning         = ?,
                   cloud_cover_note  = ?,
                   raw_response      = ?,
                   model             = ?,
                   created_at        = ?
               WHERE id = ?""",
            (
                pred.get("change_class"),
                pred.get("severity"),
                pred.get("area_pct"),
                pred.get("driver_hypothesis"),
                pred.get("confidence"),
                pred.get("reasoning"),
                pred.get("cloud_cover_note"),
                json.dumps(pred, default=str),
                label,
                datetime.now(timezone.utc).isoformat(),
                r["id"],
            ),
        )
        conn.commit()
        successes += 1
        log.info("    class=%s severity=%s area=%s driver=%s",
                 pred.get("change_class"), pred.get("severity"),
                 pred.get("area_pct"), pred.get("driver_hypothesis"))

    conn.close()
    log.info("Done. %d updated, %d failed.", successes, failures)


if __name__ == "__main__":
    main()
