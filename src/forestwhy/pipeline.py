"""Per-tile pipeline glue — used by both `predict.py` and `backfill.py`.

Single source of truth for: turning a (before, after) 13-band pair into 14
PNG panels in PANEL_ORDER, calling the VLM backend, and persisting the
result + artefacts to disk.
"""

from __future__ import annotations

import io
import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from .db import insert_prediction, update_artifact_paths
from .evaluator import PredictFn
from .jepa import S2Encoder, make_jepa_panels
from .prompts import PANEL_ORDER
from .spectral import make_spectral_panels

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Panel preparation
# ─────────────────────────────────────────────────────────────────────────────

def _resize_13band(arr: np.ndarray, target: int = 64) -> np.ndarray:
    """Resize a (13, H, W) tile to (13, target, target) with bilinear interp.

    Sentinel-2 native is 10 m/pixel → ~100 px per km. SimSat may return any
    resolution; the JEPA encoder requires exactly 64×64.
    """
    if arr.shape[1] == target and arr.shape[2] == target:
        return arr.astype(np.float32, copy=False)
    out = np.zeros((arr.shape[0], target, target), dtype=np.float32)
    for c in range(arr.shape[0]):
        img = Image.fromarray(arr[c].astype(np.float32))
        img = img.resize((target, target), Image.BILINEAR)
        out[c] = np.asarray(img, dtype=np.float32)
    return out


def build_14_panels(
    before: np.ndarray,
    after: np.ndarray,
    encoder: Optional[S2Encoder],
    device: str = "cpu",
    panel_size: int = 128,
) -> dict[str, Image.Image]:
    """Generate the 14 panels in PANEL_ORDER.

    Inputs: (13, H, W) float32 arrays, values roughly in [0, 1].

    If `encoder` is None we degrade gracefully — JEPA panels are filled with
    grey placeholders so the dashboard still renders. This branch only fires
    when JEPA loading fails; predict.py logs a warning.
    """
    before_64 = _resize_13band(before, 64)
    after_64 = _resize_13band(after, 64)

    spectral = make_spectral_panels(before_64, after_64, size=panel_size)
    if encoder is not None:
        jepa = make_jepa_panels(before_64, after_64, encoder, device=device, size=panel_size)
    else:
        log.warning("JEPA encoder unavailable; filling JEPA panels with placeholders")
        placeholder = Image.new("RGB", (panel_size, panel_size), color=(64, 64, 64))
        jepa = {
            "delta_ndvi": spectral["delta_ndvi"],
            "delta_nbr": spectral["delta_nbr"],
            "attention_multi": placeholder,
            "embedding_change": placeholder,
            "cropa_roads": placeholder,
            "delta_attn_role": placeholder,
            "head_disagreement": placeholder,
            "pca_semantic": placeholder,
        }

    panels = {**spectral, **jepa}
    missing = [k for k in PANEL_ORDER if k not in panels]
    if missing:
        raise RuntimeError(f"Panel pipeline missing keys: {missing}")
    return {k: panels[k] for k in PANEL_ORDER}


def panels_to_pngs(panels: dict[str, Image.Image]) -> list[bytes]:
    out: list[bytes] = []
    for name in PANEL_ORDER:
        buf = io.BytesIO()
        panels[name].save(buf, format="PNG")
        out.append(buf.getvalue())
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_panels(
    panels: dict[str, Image.Image],
    images_root: Path,
    row_id: int,
) -> tuple[str, list[dict[str, str]], str, str]:
    """Save all 14 panels under `images_root / str(row_id) / <name>.png`.

    Returns (panels_dir_relpath, manifest, rgb_before_relpath, rgb_after_relpath).
    Paths are stored relative to images_root.parent (the repo root) so they
    survive moving the SQLite file independently.
    """
    target = images_root / str(row_id)
    target.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, str]] = []
    for name in PANEL_ORDER:
        rel = f"{images_root.name}/{row_id}/{name}.png"
        path = images_root.parent / rel
        panels[name].save(path)
        manifest.append({"name": name, "path": rel})

    rgb_before = f"{images_root.name}/{row_id}/rgb_before.png"
    rgb_after = f"{images_root.name}/{row_id}/rgb_after.png"
    return f"{images_root.name}/{row_id}", manifest, rgb_before, rgb_after


# ─────────────────────────────────────────────────────────────────────────────
# Top-level entry: one tile -> one DB row
# ─────────────────────────────────────────────────────────────────────────────

def score_tile(
    *,
    conn: sqlite3.Connection,
    images_root: Path,
    encoder: Optional[S2Encoder],
    encoder_device: str,
    predict: PredictFn,
    backend_name: str,
    model_label: str,
    lon: float,
    lat: float,
    size_km: float,
    region_id: Optional[str],
    biome: Optional[str],
    country: Optional[str],
    before: np.ndarray,
    after: np.ndarray,
    before_timestamp: str,
    after_timestamp: str,
    before_cloud_cover: Optional[float],
    after_cloud_cover: Optional[float],
    source: str,
) -> int:
    """Build panels, call VLM, write DB row + panel files. Returns row id."""
    panels = build_14_panels(before, after, encoder, device=encoder_device)
    panel_pngs = panels_to_pngs(panels)

    metadata = {
        "lon": lon, "lat": lat, "size_km": size_km, "region_id": region_id,
        "biome": biome, "country": country,
        "before_timestamp": before_timestamp, "after_timestamp": after_timestamp,
        "before_cloud_cover": before_cloud_cover, "after_cloud_cover": after_cloud_cover,
    }

    prediction = predict(panel_pngs, metadata)

    row_id = insert_prediction(
        conn,
        lon=lon, lat=lat,
        timestamp=after_timestamp, size_km=size_km,
        source=source, model=model_label, region_id=region_id,
        before_timestamp=before_timestamp, after_timestamp=after_timestamp,
        before_cloud_cover=before_cloud_cover, after_cloud_cover=after_cloud_cover,
        prediction=prediction,
    )

    panels_dir, manifest, rgb_before_path, rgb_after_path = save_panels(panels, images_root, row_id)
    update_artifact_paths(
        conn, row_id,
        panels_dir=panels_dir,
        panels_manifest=manifest,
        rgb_before_path=rgb_before_path,
        rgb_after_path=rgb_after_path,
    )
    return row_id


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def synthesize_tile(seed: int = 0) -> np.ndarray:
    """Synthesise a (13, 64, 64) tile for smoke testing — deterministic."""
    rng = np.random.default_rng(seed)
    arr = rng.uniform(0.05, 0.45, size=(13, 64, 64)).astype(np.float32)
    return arr


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
