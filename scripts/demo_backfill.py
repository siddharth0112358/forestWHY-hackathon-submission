#!/usr/bin/env python3
"""Curated demo run: 8 known-active deforestation hotspots + 2 stable-forest contrasts.

Hand-picked geographic + driver diversity for the hackathon demo:
  Active   :  Acre, Pará, Madre de Dios (mining), Borneo, Madagascar, Chocó,
              Cambodia (rubber), Sumatra
  Stable   :  Yasuní NP (Ecuador), Manu NP (Peru)

The two protected-area pairs serve as visual contrast — the model should
land on `stable_forest` for both.

Usage:
    uv run python scripts/demo_backfill.py
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

from forestwhy.db import init_db  # noqa: E402
from forestwhy.evaluator import make_backend, model_name  # noqa: E402
from forestwhy.jepa import load_jepa_encoder  # noqa: E402
from forestwhy.live import DEFAULT_BASE_URL, fetch_13band_at_with_walkback  # noqa: E402
from forestwhy.locations import LOCATIONS_BY_ID  # noqa: E402
from forestwhy.pipeline import score_tile  # noqa: E402

DB_PATH = REPO_ROOT / "predictions.db"
IMAGES_ROOT = REPO_ROOT / "db_images"
log = logging.getLogger("forestwhy.demo_backfill")


# (region_id, lon, lat, biome, country, expected, label)
DEMO_LOCATIONS = [
    # Active deforestation — variety of drivers and biomes
    ("amazon_acre",         -68.4000,  -9.1000, "amazon",    "Brazil",    "deforestation", "Acre arc-of-deforestation, BR-364 frontier"),
    ("amazon_para",         -53.0000,  -3.5000, "amazon",    "Brazil",    "deforestation", "Pará Trans-Amazon Highway"),
    ("amazon_madre_dios",   -71.0000, -12.5000, "amazon",    "Peru",      "deforestation", "Madre de Dios artisanal gold mining"),
    ("amazon_rondonia",     -62.0000, -10.7000, "amazon",    "Brazil",    "deforestation", "Rondônia fishbone settlement"),
    ("borneo_kalimantan",   113.0000,  -1.5000, "borneo",    "Indonesia", "deforestation", "Central Kalimantan oil palm conversion"),
    ("madagascar_east",      48.5000, -18.5000, "madagascar","Madagascar","deforestation", "Eastern rainforest tavy slash-and-burn"),
    ("choco_colombia",      -76.5000,   5.5000, "choco",     "Colombia",  "deforestation", "Chocó Pacific small-scale mining"),
    ("sumatra_riau",        102.5000,  -1.0000, "sumatra",   "Indonesia", "deforestation", "Riau peatland pulp plantations"),
    # Stable-forest contrasts — known protected areas, expected to land on stable_forest
    ("yasuni_ecuador",      -76.4000,  -1.0000, "amazon",    "Ecuador",   "stable_forest", "Yasuní National Park (protected primary forest)"),
    ("manu_peru",           -71.5000, -12.0000, "amazon",    "Peru",      "stable_forest", "Manu National Park (protected primary forest)"),
]


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s  %(message)s",
                        datefmt="%H:%M:%S")

    log.info("Curated demo backfill: %d locations (8 active + 2 stable)", len(DEMO_LOCATIONS))
    encoder = load_jepa_encoder(device="auto")
    enc_device = next(encoder.parameters()).device.type

    predict = make_backend("llamacpp",
                           model="LFM2.5-forestWHY",
                           base_url="http://localhost:8080/v1")
    label = model_name("llamacpp", "Siddharth63/LFM2.5-forestWHY")

    before_target = datetime(2020, 7, 15, tzinfo=timezone.utc)
    after_target  = datetime(2024, 7, 15, tzinfo=timezone.utc)

    conn = init_db(DB_PATH)

    for i, (region_id, lon, lat, biome, country, expected, descr) in enumerate(DEMO_LOCATIONS, 1):
        log.info("== [%d/%d] %s — %s (expected: %s)", i, len(DEMO_LOCATIONS), region_id, descr, expected)
        try:
            before, b_meta = fetch_13band_at_with_walkback(
                lon=lon, lat=lat, target_ts=before_target,
                size_km=5.0, max_cloud_cover_pct=50.0,
            )
            after, a_meta = fetch_13band_at_with_walkback(
                lon=lon, lat=lat, target_ts=after_target,
                size_km=5.0, max_cloud_cover_pct=50.0,
            )
        except Exception as exc:
            log.warning("    fetch failed: %s", exc)
            continue

        try:
            row_id = score_tile(
                conn=conn, images_root=IMAGES_ROOT,
                encoder=encoder, encoder_device=enc_device,
                predict=predict, backend_name="llamacpp", model_label=label,
                lon=lon, lat=lat, size_km=5.0,
                region_id=region_id, biome=biome, country=country,
                before=before, after=after,
                before_timestamp=b_meta.get("achieved_timestamp")
                                  or before_target.strftime("%Y-%m-%dT%H:%M:%SZ"),
                after_timestamp=a_meta.get("achieved_timestamp")
                                 or after_target.strftime("%Y-%m-%dT%H:%M:%SZ"),
                before_cloud_cover=b_meta.get("cloud_cover"),
                after_cloud_cover=a_meta.get("cloud_cover"),
                source="demo",
            )
            log.info("    -> row=%d  cc=%s/%s",
                     row_id,
                     f"{b_meta.get('cloud_cover'):.1f}%" if b_meta.get('cloud_cover') is not None else '-',
                     f"{a_meta.get('cloud_cover'):.1f}%" if a_meta.get('cloud_cover') is not None else '-')
        except Exception as exc:
            log.exception("    scoring failed: %s", exc)

    log.info("Demo backfill done.")


if __name__ == "__main__":
    main()
