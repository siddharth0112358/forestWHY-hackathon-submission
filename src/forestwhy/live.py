"""SimSat HTTP client.

SimSat (https://github.com/DPhi-Space/SimSat) exposes the simulated satellite
position and Sentinel-2 imagery via REST. This module wraps the two endpoints
predict.py needs:

    GET  /data/current/position           -> (lon, lat) and timestamp
    GET  /data/current/image/sentinel     -> live 13-band Sentinel-2 tile
    GET  /data/image/sentinel             -> historical 13-band tile by lon/lat/timestamp

The Sentinel-2 endpoint supports two band-naming conventions, and SimSat
docs are ambiguous about which the active build accepts. `_probe_bands`
discovers the working spelling on first use and caches it for the process.

Run `python -m forestwhy.live --probe` to print the discovered convention
without starting the watch loop.
"""

from __future__ import annotations

import argparse
import io
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import requests
from PIL import Image

log = logging.getLogger(__name__)


DEFAULT_BASE_URL = os.environ.get("SIMSAT_BASE_URL", "http://localhost:9005")

# 13 Sentinel-2 bands in the order forestWHY expects.
BANDS_SENTINEL_HUB = [
    "B1", "B2", "B3", "B4", "B5", "B6", "B7",
    "B8", "B8A", "B9", "B10", "B11", "B12",
]

# Descriptive band aliases used by some SimSat builds.
# Mapping is best-effort: SimSat may not expose every Sentinel-2 band by
# descriptive name. When the descriptive form is active, predict.py will
# fall back to NIR + RGB + SWIR composites only and JEPA panel generation
# is skipped (a clear error is raised early).
BANDS_DESCRIPTIVE = [
    "coastal", "blue", "green", "red", "rededge1", "rededge2", "rededge3",
    "nir08", "narrow_nir", "watervapor", "cirrus", "swir16", "swir22",
]


_BAND_NAMES_CACHE: Optional[list[str]] = None


# ─────────────────────────────────────────────────────────────────────────────
# Position
# ─────────────────────────────────────────────────────────────────────────────

def get_current_position(base_url: str = DEFAULT_BASE_URL, timeout: float = 10.0) -> tuple[float, float]:
    """Return (lon, lat) of the simulated satellite right now.

    Raises requests exceptions if SimSat is unreachable — caller decides
    whether to retry.
    """
    r = requests.get(f"{base_url}/data/current/position", timeout=timeout)
    r.raise_for_status()
    payload = r.json()
    lon, lat, _alt = payload["lon-lat-alt"]
    return float(lon), float(lat)


def get_current_position_full(base_url: str = DEFAULT_BASE_URL, timeout: float = 10.0) -> dict:
    r = requests.get(f"{base_url}/data/current/position", timeout=timeout)
    r.raise_for_status()
    return r.json()


# ─────────────────────────────────────────────────────────────────────────────
# Band-name probe
# ─────────────────────────────────────────────────────────────────────────────

def _probe_bands(base_url: str, timeout: float = 10.0) -> list[str]:
    """Return the band-name convention the running SimSat accepts."""
    global _BAND_NAMES_CACHE
    if _BAND_NAMES_CACHE is not None:
        return _BAND_NAMES_CACHE

    for candidate, label in [
        (BANDS_SENTINEL_HUB, "sentinel-hub"),
        (BANDS_DESCRIPTIVE, "descriptive"),
    ]:
        try:
            r = requests.get(
                f"{base_url}/data/current/image/sentinel",
                params={
                    "spectral_bands": ",".join(candidate[:3]),
                    "size_km": 1.0,
                    "return_type": "png",
                },
                timeout=timeout,
            )
            if r.ok:
                log.info("SimSat band-name convention: %s", label)
                _BAND_NAMES_CACHE = candidate
                return candidate
        except requests.RequestException as exc:
            log.debug("Probe failed for %s: %s", label, exc)
            continue

    raise RuntimeError(
        f"Could not determine SimSat band naming via {base_url}. "
        f"Check that SimSat docker is running (docker compose up -d) and "
        f"the API is healthy."
    )


# ─────────────────────────────────────────────────────────────────────────────
# 13-band tile fetch
# ─────────────────────────────────────────────────────────────────────────────

def _decode_image_response(content: bytes) -> np.ndarray:
    """Decode a SimSat image payload into an array.

    Tries .npy first (typical for return_type=array), then PNG. PNG decode
    yields a (H, W, C) uint8 array which we transpose to (C, H, W) and
    normalise to [0, 1].
    """
    # Try NumPy .npy
    try:
        arr = np.load(io.BytesIO(content), allow_pickle=False)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        # Normalise to [0, 1] if values look like 0-10000 (Sentinel-2 raw DN)
        if arr.max() > 5.0:
            arr = arr / 10000.0
        # Transpose to (C, H, W) if needed
        if arr.ndim == 3 and arr.shape[-1] in (3, 13) and arr.shape[0] != arr.shape[-1]:
            arr = np.transpose(arr, (2, 0, 1))
        return arr
    except Exception:
        pass

    # Fall back to PNG decode
    img = Image.open(io.BytesIO(content)).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))   # (C, H, W)


def fetch_13band_current(
    size_km: float = 5.0,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 30.0,
) -> tuple[np.ndarray, dict]:
    """Fetch the satellite's current 13-band Sentinel-2 tile."""
    bands = _probe_bands(base_url, timeout)
    return _fetch_sentinel(
        endpoint="/data/current/image/sentinel",
        params={
            "spectral_bands": ",".join(bands),
            "size_km": size_km,
            "return_type": "array",
        },
        base_url=base_url,
        timeout=timeout,
    )


def fetch_13band_at(
    lon: float,
    lat: float,
    timestamp: str,
    size_km: float = 5.0,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 30.0,
    window_seconds: int = 864000,
) -> tuple[np.ndarray, dict]:
    """Fetch a 13-band Sentinel-2 tile for a specific (lon, lat, timestamp)."""
    bands = _probe_bands(base_url, timeout)
    return _fetch_sentinel(
        endpoint="/data/image/sentinel",
        params={
            "lon": lon,
            "lat": lat,
            "timestamp": timestamp,
            "spectral_bands": ",".join(bands),
            "size_km": size_km,
            "return_type": "array",
            "window_seconds": window_seconds,
        },
        base_url=base_url,
        timeout=timeout,
    )


def fetch_13band_at_with_walkback(
    lon: float,
    lat: float,
    target_ts: datetime,
    size_km: float = 5.0,
    base_url: str = DEFAULT_BASE_URL,
    max_walkback_days: int = 90,
    stride_days: int = 14,
    timeout: float = 30.0,
) -> tuple[np.ndarray, dict]:
    """Try the target timestamp; if SimSat reports no coverage, walk back in
    14-day strides up to 90 days before raising."""
    deltas = [0] + list(range(stride_days, max_walkback_days + 1, stride_days))
    last_exc: Optional[Exception] = None
    for d in deltas:
        ts = (target_ts - timedelta(days=d)).strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            arr, meta = fetch_13band_at(
                lon=lon, lat=lat, timestamp=ts,
                size_km=size_km, base_url=base_url, timeout=timeout,
            )
            if arr.size > 0:
                meta["achieved_timestamp"] = ts
                return arr, meta
        except (requests.HTTPError, requests.Timeout, RuntimeError) as exc:
            last_exc = exc
            continue
    raise RuntimeError(
        f"No Sentinel-2 coverage at lon={lon} lat={lat} within "
        f"{max_walkback_days} days of {target_ts.isoformat()}. "
        f"Last error: {last_exc}"
    )


def _fetch_sentinel(
    endpoint: str,
    params: dict,
    base_url: str,
    timeout: float,
) -> tuple[np.ndarray, dict]:
    r = requests.get(f"{base_url}{endpoint}", params=params, timeout=timeout)
    r.raise_for_status()

    # SimSat returns image bytes in the body; metadata may be in headers or
    # JSON body. Detect by content-type.
    ctype = r.headers.get("content-type", "")
    if "json" in ctype:
        payload = r.json()
        meta = {k: v for k, v in payload.items() if k != "data"}
        if "data" not in payload or not payload.get("image_available", True):
            raise RuntimeError(f"No image available: {payload}")
        # The payload may carry a base64 string under "data"; decode.
        import base64
        data = base64.b64decode(payload["data"])
        arr = _decode_image_response(data)
        return arr, meta

    # Binary response: parse metadata from headers if present.
    meta: dict = {
        "content_type": ctype,
        "cloud_cover": _safe_float(r.headers.get("X-Cloud-Cover")),
        "datetime": r.headers.get("X-Datetime"),
        "source": r.headers.get("X-Source"),
    }
    arr = _decode_image_response(r.content)
    return arr, meta


def _safe_float(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    try:
        return float(s)
    except ValueError:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# CLI: probe / spot-check
# ─────────────────────────────────────────────────────────────────────────────

def _main() -> None:
    parser = argparse.ArgumentParser(description="SimSat client probe")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--probe", action="store_true", help="Probe band-name convention")
    parser.add_argument("--position", action="store_true", help="Print current satellite position")
    parser.add_argument("--current", action="store_true", help="Fetch current 13-band tile")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.probe:
        bands = _probe_bands(args.base_url)
        print("Active band convention:", bands)
        return

    if args.position:
        lon, lat = get_current_position(args.base_url)
        print(f"lon={lon:.4f} lat={lat:.4f}")
        return

    if args.current:
        arr, meta = fetch_13band_current(base_url=args.base_url)
        print(f"shape={arr.shape} dtype={arr.dtype} meta={meta}")
        return

    parser.print_help()


if __name__ == "__main__":
    _main()
