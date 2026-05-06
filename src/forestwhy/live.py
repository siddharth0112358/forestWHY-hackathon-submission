"""SimSat HTTP client.

SimSat (https://github.com/DPhi-Space/SimSat) exposes the simulated satellite
position and Sentinel-2 imagery via REST. The two endpoints predict.py needs:

    GET  /data/current/position           -> (lon, lat) and timestamp
    GET  /data/current/image/sentinel     -> live tile, current sat pos
    GET  /data/image/sentinel             -> historical tile by lon/lat/timestamp

SimSat uses **descriptive band names** (red, green, blue, nir, nir08, nir09,
swir16, swir22, rededge1/2/3, coastal). It serves Sentinel-2 L2A from the AWS
Open Data tier, which retains 12 of the 13 spectral bands — B10 (cirrus) is
dropped during L2A processing. We fetch the 12 available bands, then assemble
a (13, H, W) tensor in the JEPA encoder's expected channel order, zero-padding
B10. The JEPA encoder weights for B10 are negligible (B10 is mostly used for
atmospheric correction in L1C and not informative for change detection).

Run `python -m forestwhy.live --probe` to confirm SimSat is reachable and
the API is responding.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional, Sequence

import numpy as np
import requests
from PIL import Image

log = logging.getLogger(__name__)


DEFAULT_BASE_URL = os.environ.get("SIMSAT_BASE_URL", "http://localhost:9005")


# ─────────────────────────────────────────────────────────────────────────────
# Band vocabulary
# ─────────────────────────────────────────────────────────────────────────────

# Descriptive names that SimSat's Sentinel-2 L2A provider exposes. Order here
# is arbitrary — it just has to be sent verbatim to the API.
SIMSAT_BANDS_AVAILABLE: tuple[str, ...] = (
    "coastal", "blue", "green", "red",
    "rededge1", "rededge2", "rededge3",
    "nir", "nir08", "nir09",
    "swir16", "swir22",
)

# Maps SimSat descriptive band -> index in the JEPA's 13-channel input order
# (B1, B2, ..., B12, with B8A at index 8). B10 (index 10) is not retained in
# Sentinel-2 L2A and gets zero-padded by `_remap_to_13band`.
SIMSAT_TO_S2_INDEX: dict[str, int] = {
    "coastal":   0,   # B1
    "blue":      1,   # B2
    "green":     2,   # B3
    "red":       3,   # B4
    "rededge1":  4,   # B5
    "rededge2":  5,   # B6
    "rededge3":  6,   # B7
    "nir":       7,   # B8
    "nir08":     8,   # B8A
    "nir09":     9,   # B9
    # B10 (index 10) — no SimSat name, zero-padded
    "swir16":   11,   # B11
    "swir22":   12,   # B12
}

# Minimal probe set — just enough to confirm the API is reachable.
PROBE_BANDS: tuple[str, ...] = ("red", "green", "blue")


# ─────────────────────────────────────────────────────────────────────────────
# Position
# ─────────────────────────────────────────────────────────────────────────────

def get_current_position(base_url: str = DEFAULT_BASE_URL, timeout: float = 10.0) -> tuple[float, float]:
    """Return (lon, lat) of the simulated satellite right now."""
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
# Connectivity probe
# ─────────────────────────────────────────────────────────────────────────────

def probe(base_url: str = DEFAULT_BASE_URL, timeout: float = 10.0) -> dict:
    """Return diagnostic info about a running SimSat. Raises RuntimeError on failure."""
    info: dict = {"base_url": base_url}
    try:
        info["position"] = get_current_position_full(base_url, timeout=timeout)
    except requests.RequestException as exc:
        raise RuntimeError(
            f"SimSat /data/current/position not reachable at {base_url}: {exc}\n"
            f"  - Is SimSat running? `docker ps` should show fakesat-sim and fakesat-dashboard.\n"
            f"  - Did fakesat-sim start? `docker logs fakesat-sim` to check.\n"
            f"  - SimSat needs MAPBOX_ACCESS_TOKEN set (any non-empty string) to boot.\n"
        ) from exc

    # Hit the sentinel endpoint with a 1 km tile and the 3-band probe set.
    r = requests.get(
        f"{base_url}/data/current/image/sentinel",
        params=[("spectral_bands", b) for b in PROBE_BANDS]
        + [("size_km", 1.0), ("return_type", "png")],
        timeout=timeout,
    )
    if not r.ok:
        raise RuntimeError(
            f"SimSat /data/current/image/sentinel returned {r.status_code}: {r.text[:200]}"
        )
    info["sentinel_status"] = r.status_code
    info["sentinel_metadata"] = _parse_sentinel_metadata_header(r.headers)
    info["available_bands"] = list(SIMSAT_BANDS_AVAILABLE)
    return info


# ─────────────────────────────────────────────────────────────────────────────
# Tile fetch
# ─────────────────────────────────────────────────────────────────────────────

def fetch_13band_current(
    size_km: float = 5.0,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 60.0,
    bands: Sequence[str] = SIMSAT_BANDS_AVAILABLE,
) -> tuple[np.ndarray, dict]:
    """Fetch the satellite's current Sentinel-2 tile, normalised to (13, H, W) for JEPA."""
    return _fetch_sentinel_array(
        endpoint="/data/current/image/sentinel",
        base_url=base_url,
        timeout=timeout,
        params=_sentinel_params(bands, size_km),
    )


def fetch_13band_at(
    lon: float,
    lat: float,
    timestamp: str,
    size_km: float = 5.0,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 60.0,
    window_seconds: int = 864_000,
    bands: Sequence[str] = SIMSAT_BANDS_AVAILABLE,
) -> tuple[np.ndarray, dict]:
    """Fetch a Sentinel-2 tile for a specific (lon, lat, timestamp), normalised to (13, H, W)."""
    return _fetch_sentinel_array(
        endpoint="/data/image/sentinel",
        base_url=base_url,
        timeout=timeout,
        params=_sentinel_params(bands, size_km, lon=lon, lat=lat,
                                timestamp=timestamp, window_seconds=window_seconds),
    )


def fetch_13band_at_with_walkback(
    lon: float,
    lat: float,
    target_ts: datetime,
    size_km: float = 5.0,
    base_url: str = DEFAULT_BASE_URL,
    max_walkback_days: int = 90,
    stride_days: int = 14,
    timeout: float = 60.0,
    bands: Sequence[str] = SIMSAT_BANDS_AVAILABLE,
) -> tuple[np.ndarray, dict]:
    """Try the target timestamp; on failure, walk back in 14-day strides up to 90 days."""
    deltas = [0] + list(range(stride_days, max_walkback_days + 1, stride_days))
    last_exc: Optional[Exception] = None
    for d in deltas:
        ts = (target_ts - timedelta(days=d)).strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            arr, meta = fetch_13band_at(
                lon=lon, lat=lat, timestamp=ts,
                size_km=size_km, base_url=base_url, timeout=timeout, bands=bands,
            )
            if arr.size > 0 and meta.get("image_available", True):
                meta["achieved_timestamp"] = ts
                return arr, meta
        except (requests.HTTPError, requests.Timeout, RuntimeError) as exc:
            last_exc = exc
            continue
    raise RuntimeError(
        f"No Sentinel-2 coverage at lon={lon} lat={lat} within "
        f"{max_walkback_days} days of {target_ts.isoformat()}. Last error: {last_exc}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# HTTP helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sentinel_params(
    bands: Sequence[str],
    size_km: float,
    *,
    lon: Optional[float] = None,
    lat: Optional[float] = None,
    timestamp: Optional[str] = None,
    window_seconds: Optional[int] = None,
) -> list[tuple[str, str | float]]:
    """Build the query params as a list of tuples so each band is a separate
    `?spectral_bands=` entry — matches FastAPI's `List[str] = Query(...)` parsing."""
    params: list[tuple[str, str | float]] = [("spectral_bands", b) for b in bands]
    params.append(("size_km", size_km))
    params.append(("return_type", "array"))
    if lon is not None:
        params.append(("lon", lon))
    if lat is not None:
        params.append(("lat", lat))
    if timestamp is not None:
        params.append(("timestamp", timestamp))
    if window_seconds is not None:
        params.append(("window_seconds", window_seconds))
    return params


def _parse_sentinel_metadata_header(headers) -> dict:
    raw = headers.get("sentinel_metadata") or headers.get("Sentinel-Metadata")
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}


def _fetch_sentinel_array(
    endpoint: str,
    base_url: str,
    timeout: float,
    params: list[tuple[str, str | float]],
) -> tuple[np.ndarray, dict]:
    """Hit a SimSat sentinel endpoint with return_type=array, decode, remap to (13, H, W).

    Response shape (both endpoints):
        {
          "image": { "metadata": {"shape": [...], "dtype": "...", "bands": [...]},
                     "image": "<base64-encoded raw bytes>" },
          "sentinel_metadata": {"image_available": ..., "source": ..., ...}
        }
    """
    r = requests.get(f"{base_url}{endpoint}", params=params, timeout=timeout)
    r.raise_for_status()

    payload = r.json()
    sentinel_meta = payload.get("sentinel_metadata", {}) or {}
    inner = payload.get("image") or {}
    inner_meta = inner.get("metadata") if isinstance(inner, dict) else {}

    meta: dict = {
        "image_available": sentinel_meta.get("image_available", True),
        "source": sentinel_meta.get("source"),
        "footprint": sentinel_meta.get("footprint"),
        "cloud_cover": sentinel_meta.get("cloud_cover"),
        "datetime": sentinel_meta.get("datetime"),
        "spectral_bands": (inner_meta or {}).get("bands", []),
        "shape_raw": (inner_meta or {}).get("shape"),
        "dtype_raw": (inner_meta or {}).get("dtype"),
    }
    if not meta["image_available"]:
        raise RuntimeError(f"SimSat reported no image available: {meta}")

    arr = _decode_array_payload(inner)
    arr = _remap_to_13band(arr, meta["spectral_bands"])
    return arr, meta


def _decode_array_payload(inner: dict) -> np.ndarray:
    """Decode SimSat's inner payload {image: <base64>, metadata: {shape, dtype, bands}}."""
    img_b64 = inner.get("image") if isinstance(inner, dict) else None
    if not isinstance(img_b64, str):
        raise RuntimeError(
            f"SimSat array payload missing string 'image' field; got "
            f"{type(img_b64).__name__}: keys={list(inner.keys()) if isinstance(inner, dict) else inner}"
        )
    md = inner.get("metadata") or {}
    shape = tuple(md.get("shape", ()))
    dtype = np.dtype(md.get("dtype", "uint16"))
    raw = base64.b64decode(img_b64)
    arr = np.frombuffer(raw, dtype=dtype)
    if shape:
        arr = arr.reshape(shape)
    return arr.astype(np.float32, copy=False)


def _remap_to_13band(arr: np.ndarray, requested_bands: list[str]) -> np.ndarray:
    """Reorder a (C, H, W) tensor of arbitrary band order into JEPA's 13-channel order.

    Bands not in SIMSAT_TO_S2_INDEX (notably B10/cirrus, which SimSat doesn't
    expose) are zero-padded. Sentinel-2 reflectance values are in 0–10000 raw
    DN; we normalise to [0, 1] for the JEPA encoder.
    """
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise RuntimeError(f"Expected (C, H, W) array, got shape {arr.shape}")
    C, H, W = arr.shape
    if C != len(requested_bands):
        raise RuntimeError(
            f"Band count mismatch: array has {C} channels but {len(requested_bands)} band names"
        )
    out = np.zeros((13, H, W), dtype=np.float32)
    for ch, name in enumerate(requested_bands):
        idx = SIMSAT_TO_S2_INDEX.get(name)
        if idx is None:
            log.debug("Skipping unmapped band %s", name)
            continue
        out[idx] = arr[ch]

    # Sentinel-2 L2A reflectance is in raw DN (0-10000). Detect and normalise.
    if out.max() > 5.0:
        out = out / 10000.0
    return np.clip(out, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _main() -> None:
    parser = argparse.ArgumentParser(description="SimSat client probe")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--probe", action="store_true",
                        help="Confirm SimSat is reachable and report band info")
    parser.add_argument("--position", action="store_true",
                        help="Print current satellite position")
    parser.add_argument("--current", action="store_true",
                        help="Fetch current 13-band tile and print shape + metadata")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.probe:
        info = probe(args.base_url)
        print(json.dumps(info, indent=2, default=str))
        return

    if args.position:
        lon, lat = get_current_position(args.base_url)
        print(f"lon={lon:.4f} lat={lat:.4f}")
        return

    if args.current:
        arr, meta = fetch_13band_current(base_url=args.base_url)
        print(f"shape={arr.shape} dtype={arr.dtype}")
        print(f"min={arr.min():.4f} max={arr.max():.4f} mean={arr.mean():.4f}")
        print("metadata:", json.dumps(meta, default=str, indent=2))
        return

    parser.print_help()


if __name__ == "__main__":
    _main()
