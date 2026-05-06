#!/usr/bin/env python3
"""Build a small annotated evaluation set with a strong frontier model.

Mirrors `wildfire-prevention/scripts/generate_samples.py`. For each watched
location we fetch a temporal pair from SimSat, build the 14 panels, and ask
Anthropic Claude to generate the ground-truth JSON. This is for held-out
*evaluation* only — full training was already done upstream and produced
`Siddharth63/forestwhy-training-v1`.

Usage:
    uv run python scripts/generate_samples.py --years 2020 2024 --n-per-location 3
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from forestwhy.jepa import load_jepa_encoder  # noqa: E402
from forestwhy.live import DEFAULT_BASE_URL, fetch_13band_at_with_walkback  # noqa: E402
from forestwhy.locations import LOCATIONS, LOCATIONS_BY_ID  # noqa: E402
from forestwhy.pipeline import build_14_panels, panels_to_pngs  # noqa: E402
from forestwhy.prompts import (  # noqa: E402
    JSON_SCHEMA, PANEL_ORDER, SYSTEM_PROMPT, render_user_prompt,
)

SAMPLES_ROOT = REPO_ROOT / "samples"
log = logging.getLogger("forestwhy.generate_samples")


def _annotate_with_claude(panel_pngs: list[bytes], metadata: dict, model: str) -> dict:
    """Use Claude as the ground-truth annotator."""
    import anthropic
    client = anthropic.Anthropic()
    content: list[dict] = []
    for png in panel_pngs:
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png",
                       "data": base64.b64encode(png).decode("ascii")},
        })
    content.append({"type": "text", "text": render_user_prompt(metadata) + "\n\nReturn the JSON only."})
    rsp = client.messages.create(
        model=model,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
    )
    text = "".join(block.text for block in rsp.content if hasattr(block, "text"))
    text = text.strip()
    if text.startswith("```"):
        text = text.lstrip("`").lstrip("json").strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    return json.loads(text)


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s",
                        datefmt="%H:%M:%S")
    p = argparse.ArgumentParser()
    p.add_argument("--location", default=None, choices=list(LOCATIONS_BY_ID))
    p.add_argument("--years", nargs=2, type=int, metavar=("BEFORE", "AFTER"), default=[2020, 2024])
    p.add_argument("--n-per-location", type=int, default=1)
    p.add_argument("--simsat-url", default=DEFAULT_BASE_URL)
    p.add_argument("--anthropic-model", default="claude-opus-4-5-20251001")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        log.error("ANTHROPIC_API_KEY not set in env (.env or shell). Aborting.")
        sys.exit(1)

    locs = [LOCATIONS_BY_ID[args.location]] if args.location else list(LOCATIONS)
    log.info("Generating %d sample(s) per location across %d location(s).",
             args.n_per_location, len(locs))

    encoder = load_jepa_encoder(device=args.device)
    enc_device = next(encoder.parameters()).device.type

    out_root = SAMPLES_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_root.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    before_target = datetime(args.years[0], 7, 15, tzinfo=timezone.utc)
    after_target = datetime(args.years[1], 7, 15, tzinfo=timezone.utc)

    for loc in locs:
        for k in range(args.n_per_location):
            sample_id = f"{loc.id}__{k}"
            log.info("== %s", sample_id)
            try:
                before, before_meta = fetch_13band_at_with_walkback(
                    lon=loc.lon, lat=loc.lat, target_ts=before_target,
                    base_url=args.simsat_url,
                )
                after, after_meta = fetch_13band_at_with_walkback(
                    lon=loc.lon, lat=loc.lat, target_ts=after_target,
                    base_url=args.simsat_url,
                )
            except Exception as exc:
                log.warning("  fetch failed: %s", exc)
                continue

            panels = build_14_panels(before, after, encoder, device=enc_device)
            sample_dir = out_root / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)
            for name in PANEL_ORDER:
                panels[name].save(sample_dir / f"{name}.png")

            metadata = {
                "lon": loc.lon, "lat": loc.lat, "size_km": 5.0,
                "region_id": loc.id, "biome": loc.biome, "country": loc.country,
                "before_timestamp": before_meta.get("achieved_timestamp")
                                     or before_target.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "after_timestamp": after_meta.get("achieved_timestamp")
                                    or after_target.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "before_cloud_cover": before_meta.get("cloud_cover"),
                "after_cloud_cover": after_meta.get("cloud_cover"),
            }
            try:
                truth = _annotate_with_claude(
                    panels_to_pngs(panels), metadata, args.anthropic_model,
                )
            except Exception as exc:
                log.warning("  Claude annotation failed: %s", exc)
                continue

            (sample_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
            (sample_dir / "truth.json").write_text(json.dumps(truth, indent=2))
            manifest.append({"id": sample_id, "metadata": metadata, "truth": truth})
            log.info("  -> %s  class=%s  severity=%s",
                     sample_dir, truth.get("change_class"), truth.get("severity"))

    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("Wrote %d samples to %s", len(manifest), out_root)


if __name__ == "__main__":
    main()
