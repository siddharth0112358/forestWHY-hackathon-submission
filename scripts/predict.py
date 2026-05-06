#!/usr/bin/env python3
"""Watch loop: poll SimSat, run JEPA + spectral panels, call the VLM, save to SQLite.

Runs until interrupted with Ctrl+C. The loop polls SimSat for the satellite's
current position and, when the satellite is over one of the 18 watched
deforestation hotspots, fetches a (before, after) Sentinel-2 13-band pair,
generates 14 panels (8 spectral + 6 JEPA), calls the fine-tuned LFM2.5-VL
via vLLM (or llama.cpp), and persists the structured prediction to SQLite.

Usage:
    uv run python scripts/predict.py --backend vllm
    uv run python scripts/predict.py --backend vllm --location amazon_acre
    uv run python scripts/predict.py --backend llamacpp --base-url http://localhost:8080/v1
    uv run python scripts/predict.py --smoke-test --backend stub
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from forestwhy.evaluator import make_backend, model_name  # noqa: E402
from forestwhy.db import init_db  # noqa: E402
from forestwhy.jepa import S2Encoder, load_jepa_encoder  # noqa: E402
from forestwhy.live import (  # noqa: E402
    DEFAULT_BASE_URL,
    fetch_13band_at_with_walkback,
    fetch_13band_current,
    get_current_position,
)
from forestwhy.llama_server import (  # noqa: E402
    DEFAULT_GGUF_QUANT,
    DEFAULT_GGUF_REPO,
    ensure_gguf,
    start_llama_server,
    stop_server,
    wait_for_server,
)
from forestwhy.locations import LOCATIONS, LOCATIONS_BY_ID, Location  # noqa: E402
from forestwhy.pipeline import now_iso, score_tile, synthesize_tile  # noqa: E402
from forestwhy.regions import find_tile, haversine_km  # noqa: E402

DB_PATH = REPO_ROOT / "predictions.db"
IMAGES_ROOT = REPO_ROOT / "db_images"

log = logging.getLogger("forestwhy.predict")


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="forestWHY watch loop")
    p.add_argument("--backend", default="stub",
                   choices=["vllm", "llamacpp", "transformers", "stub"],
                   help="VLM inference backend (required for live runs; "
                        "ignored by --smoke-test which uses --with-gguf or stub).")
    p.add_argument("--model", default=None,
                   help="VLM model id (HF repo). Defaults to $VLM_MODEL.")
    p.add_argument("--base-url", default=None,
                   help="OpenAI-compatible endpoint for vllm/llamacpp.")
    p.add_argument("--location", default=None, choices=list(LOCATIONS_BY_ID),
                   help="Watch only this hotspot. Default: all 18.")
    p.add_argument("--size-km", type=float, default=5.0,
                   help="Tile edge in km.")
    p.add_argument("--interval", type=float, default=60.0,
                   help="Seconds between satellite-position polls.")
    p.add_argument("--min-distance-km", type=float, default=None,
                   help="Skip if last processed point is closer than this. Default: --size-km.")
    p.add_argument("--lookback-days", type=int, default=365,
                   help="Days back for the 'before' tile.")
    p.add_argument("--device", default="auto",
                   help="JEPA encoder device: auto|cpu|cuda|mps.")
    p.add_argument("--simsat-url", default=DEFAULT_BASE_URL,
                   help="SimSat base URL.")
    p.add_argument("--no-jepa", action="store_true",
                   help="Skip JEPA encoder loading; use placeholder JEPA panels.")
    p.add_argument("--smoke-test", action="store_true",
                   help="Run one synthetic iteration without SimSat. Pair with "
                        "--with-gguf for a real end-to-end VLM call.")
    p.add_argument("--with-gguf", action="store_true",
                   help="Smoke test only: auto-download the GGUF + mmproj from "
                        "HF, launch llama-server locally, score one tile, tear down.")
    p.add_argument("--gguf-repo", default=DEFAULT_GGUF_REPO,
                   help="HF repo for the GGUF pair when --with-gguf is set.")
    p.add_argument("--gguf-quant", default=DEFAULT_GGUF_QUANT,
                   help="Quant suffix, e.g. Q4_K_M (731 MB), Q5_K_M, Q8_0 (1.25 GB).")
    p.add_argument("--llama-port", type=int, default=8080,
                   help="Port for the auto-launched llama-server.")
    p.add_argument("--max-iters", type=int, default=0,
                   help="Stop after N successful predictions (0 = forever).")
    args = p.parse_args()
    if args.min_distance_km is None:
        args.min_distance_km = args.size_km
    return args


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test path
# ─────────────────────────────────────────────────────────────────────────────

def _smoke_test(args: argparse.Namespace) -> int:
    """Synthesise a tile, run JEPA + spectral panels, optionally call a real
    GGUF-backed VLM, write one DB row.

    Three modes, in order of fidelity:
      1. `--smoke-test` (default)              JEPA + stub backend.
      2. `--smoke-test --no-jepa`              placeholder JEPA + stub backend.
      3. `--smoke-test --with-gguf`            JEPA + real GGUF VLM via llama-server.
    """
    log.info("Smoke test: synthesising a tile and writing one row to %s", DB_PATH)
    conn = init_db(DB_PATH)
    encoder = None
    enc_device = "cpu"
    if not args.no_jepa:
        try:
            encoder = load_jepa_encoder(device=args.device)
            enc_device = next(encoder.parameters()).device.type
        except Exception as exc:
            log.warning("Could not load JEPA encoder for smoke test (%s); using placeholders.", exc)

    server_proc = None
    backend_label = "stub"
    if args.with_gguf:
        try:
            model_path, mmproj_path = ensure_gguf(repo=args.gguf_repo, quant=args.gguf_quant)
            server_proc = start_llama_server(model_path, mmproj_path, port=args.llama_port)
            wait_for_server(port=args.llama_port)
            backend_label = "llamacpp"
            predict = make_backend(
                "llamacpp",
                model=args.model or f"{args.gguf_repo}:{args.gguf_quant}",
                base_url=f"http://127.0.0.1:{args.llama_port}/v1",
            )
        except Exception:
            if server_proc is not None:
                stop_server(server_proc)
            raise
    else:
        if args.backend != "stub":
            log.info("Smoke test without --with-gguf uses backend=stub.")
        predict = make_backend("stub")

    label = model_name(backend_label, args.model or args.gguf_repo if args.with_gguf else args.model)

    loc = LOCATIONS[0]
    before = synthesize_tile(seed=1)
    after = synthesize_tile(seed=2)
    try:
        row_id = score_tile(
            conn=conn, images_root=IMAGES_ROOT,
            encoder=encoder, encoder_device=enc_device,
            predict=predict, backend_name=backend_label, model_label=label,
            lon=loc.lon, lat=loc.lat, size_km=args.size_km,
            region_id=loc.id, biome=loc.biome, country=loc.country,
            before=before, after=after,
            before_timestamp=(datetime.now(timezone.utc) - timedelta(days=365)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            after_timestamp=now_iso(),
            before_cloud_cover=10.0, after_cloud_cover=12.0,
            source="smoke-test" if not args.with_gguf else "smoke-test-gguf",
        )
        log.info("Smoke test wrote row id=%d to %s", row_id, DB_PATH)
        return row_id
    finally:
        if server_proc is not None:
            log.info("Stopping llama-server ...")
            stop_server(server_proc)


# ─────────────────────────────────────────────────────────────────────────────
# Live watch loop
# ─────────────────────────────────────────────────────────────────────────────

def _build_location_list(arg: str | None) -> list[Location]:
    if arg:
        return [LOCATIONS_BY_ID[arg]]
    return list(LOCATIONS)


def _watch(args: argparse.Namespace) -> None:
    conn = init_db(DB_PATH)

    if args.no_jepa:
        log.warning("--no-jepa: JEPA panels will be placeholder grey squares.")
        encoder: S2Encoder | None = None
        enc_device = "cpu"
    else:
        encoder = load_jepa_encoder(device=args.device)
        enc_device = next(encoder.parameters()).device.type

    predict = make_backend(args.backend, model=args.model, base_url=args.base_url)
    label = model_name(args.backend, args.model)

    locs = _build_location_list(args.location)
    tiles = [(loc.lon, loc.lat) for loc in locs]
    by_coords = {(loc.lon, loc.lat): loc for loc in locs}

    log.info(
        "Watching %d location(s)  size_km=%.1f  interval=%.0fs  backend=%s  model=%s",
        len(locs), args.size_km, args.interval, args.backend, label,
    )

    last_lon: float | None = None
    last_lat: float | None = None
    successes = 0

    while True:
        try:
            sat_lon, sat_lat = get_current_position(args.simsat_url)
        except requests.ConnectionError:
            log.warning("SimSat not reachable at %s. Run: cd ../SimSat && docker compose up -d", args.simsat_url)
            time.sleep(args.interval)
            continue
        except Exception as exc:
            log.warning("Position fetch failed: %s", exc)
            time.sleep(args.interval)
            continue

        ts = now_iso()
        hit = find_tile(sat_lon, sat_lat, tiles, args.size_km)
        if hit is None:
            log.info("[%s] sat lon=%.4f lat=%.4f outside watched tiles", ts[:19], sat_lon, sat_lat)
            time.sleep(args.interval)
            continue

        tile_lon, tile_lat = hit
        loc = by_coords[(tile_lon, tile_lat)]

        if last_lon is not None and last_lat is not None:
            d = haversine_km(last_lon, last_lat, tile_lon, tile_lat)
            if d < args.min_distance_km:
                time.sleep(args.interval)
                continue

        log.info("[%s] hit %s lon=%.4f lat=%.4f  fetching ...", ts[:19], loc.id, tile_lon, tile_lat)

        try:
            after_arr, after_meta = fetch_13band_current(size_km=args.size_km, base_url=args.simsat_url)
        except (requests.HTTPError, requests.Timeout, requests.ConnectionError, RuntimeError) as exc:
            log.warning("  after-tile fetch failed (%s); skipping", exc)
            time.sleep(args.interval)
            continue

        target_before = datetime.now(timezone.utc) - timedelta(days=args.lookback_days)
        try:
            before_arr, before_meta = fetch_13band_at_with_walkback(
                lon=tile_lon, lat=tile_lat, target_ts=target_before,
                size_km=args.size_km, base_url=args.simsat_url,
            )
        except RuntimeError as exc:
            log.warning("  before-tile walkback failed (%s); skipping", exc)
            time.sleep(args.interval)
            continue

        try:
            row_id = score_tile(
                conn=conn, images_root=IMAGES_ROOT,
                encoder=encoder, encoder_device=enc_device,
                predict=predict, backend_name=args.backend, model_label=label,
                lon=tile_lon, lat=tile_lat, size_km=args.size_km,
                region_id=loc.id, biome=loc.biome, country=loc.country,
                before=before_arr, after=after_arr,
                before_timestamp=before_meta.get("achieved_timestamp")
                                  or before_meta.get("datetime")
                                  or target_before.strftime("%Y-%m-%dT%H:%M:%SZ"),
                after_timestamp=after_meta.get("datetime") or ts,
                before_cloud_cover=before_meta.get("cloud_cover"),
                after_cloud_cover=after_meta.get("cloud_cover"),
                source="simsat-live",
            )
        except Exception as exc:
            log.exception("  scoring failed (%s); skipping", exc)
            time.sleep(args.interval)
            continue

        last_lon, last_lat = tile_lon, tile_lat
        successes += 1
        log.info(
            "  row=%d total=%d",
            row_id, successes,
        )
        if args.max_iters and successes >= args.max_iters:
            log.info("Reached --max-iters=%d, stopping.", args.max_iters)
            return

        time.sleep(args.interval)


# ─────────────────────────────────────────────────────────────────────────────
# Entry
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s",
                        datefmt="%H:%M:%S")
    args = _parse_args()

    if args.smoke_test:
        _smoke_test(args)
        return

    try:
        _watch(args)
    except KeyboardInterrupt:
        log.info("Stopped by user.")


if __name__ == "__main__":
    main()
