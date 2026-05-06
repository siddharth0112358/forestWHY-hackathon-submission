#!/usr/bin/env python3
"""Generate historical predictions for evaluation and dashboard seeding.

For each location and each year-pair in the requested range, fetches a
13-band Sentinel-2 before/after pair from SimSat's historical endpoint,
runs the panel pipeline + VLM, and writes one row per pair.

Usage:
    uv run python scripts/backfill.py --backend vllm --years 2020 2024
    uv run python scripts/backfill.py --backend stub --years 2022 2024 --location amazon_acre
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from forestwhy.db import init_db  # noqa: E402
from forestwhy.evaluator import make_backend, model_name  # noqa: E402
from forestwhy.jepa import load_jepa_encoder  # noqa: E402
from forestwhy.live import DEFAULT_BASE_URL, fetch_13band_at_with_walkback  # noqa: E402
from forestwhy.locations import LOCATIONS, LOCATIONS_BY_ID  # noqa: E402
from forestwhy.pipeline import score_tile  # noqa: E402

DB_PATH = REPO_ROOT / "predictions.db"
IMAGES_ROOT = REPO_ROOT / "db_images"
log = logging.getLogger("forestwhy.backfill")


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s",
                        datefmt="%H:%M:%S")
    p = argparse.ArgumentParser()
    p.add_argument("--backend", required=True, choices=["vllm", "llamacpp", "transformers", "stub"])
    p.add_argument("--model", default=None)
    p.add_argument("--base-url", default=None)
    p.add_argument("--location", default=None, choices=list(LOCATIONS_BY_ID))
    p.add_argument("--years", nargs=2, type=int, metavar=("BEFORE", "AFTER"), default=[2020, 2024])
    p.add_argument("--size-km", type=float, default=5.0)
    p.add_argument("--simsat-url", default=DEFAULT_BASE_URL)
    p.add_argument("--device", default="auto")
    p.add_argument("--no-jepa", action="store_true")
    args = p.parse_args()

    locs = [LOCATIONS_BY_ID[args.location]] if args.location else list(LOCATIONS)
    log.info("Backfilling %d location(s), years %d -> %d", len(locs), args.years[0], args.years[1])

    conn = init_db(DB_PATH)
    encoder = None
    enc_device = "cpu"
    if not args.no_jepa:
        encoder = load_jepa_encoder(device=args.device)
        enc_device = next(encoder.parameters()).device.type

    predict = make_backend(args.backend, model=args.model, base_url=args.base_url)
    label = model_name(args.backend, args.model)

    before_target = datetime(args.years[0], 7, 15, tzinfo=timezone.utc)
    after_target = datetime(args.years[1], 7, 15, tzinfo=timezone.utc)

    completed = 0
    for loc in locs:
        log.info("== %s (%.4f, %.4f)", loc.id, loc.lon, loc.lat)
        try:
            before_arr, before_meta = fetch_13band_at_with_walkback(
                lon=loc.lon, lat=loc.lat, target_ts=before_target,
                size_km=args.size_km, base_url=args.simsat_url,
            )
            after_arr, after_meta = fetch_13band_at_with_walkback(
                lon=loc.lon, lat=loc.lat, target_ts=after_target,
                size_km=args.size_km, base_url=args.simsat_url,
            )
        except Exception as exc:
            log.warning("  fetch failed for %s: %s", loc.id, exc)
            continue

        try:
            row_id = score_tile(
                conn=conn, images_root=IMAGES_ROOT,
                encoder=encoder, encoder_device=enc_device,
                predict=predict, backend_name=args.backend, model_label=label,
                lon=loc.lon, lat=loc.lat, size_km=args.size_km,
                region_id=loc.id, biome=loc.biome, country=loc.country,
                before=before_arr, after=after_arr,
                before_timestamp=before_meta.get("achieved_timestamp")
                                  or before_meta.get("datetime")
                                  or before_target.strftime("%Y-%m-%dT%H:%M:%SZ"),
                after_timestamp=after_meta.get("achieved_timestamp")
                                 or after_meta.get("datetime")
                                 or after_target.strftime("%Y-%m-%dT%H:%M:%SZ"),
                before_cloud_cover=before_meta.get("cloud_cover"),
                after_cloud_cover=after_meta.get("cloud_cover"),
                source="backfill",
            )
            log.info("  row=%d", row_id)
            completed += 1
        except Exception as exc:
            log.exception("  scoring failed for %s: %s", loc.id, exc)

    log.info("Backfill complete: %d rows.", completed)


if __name__ == "__main__":
    main()
