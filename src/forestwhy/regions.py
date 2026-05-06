"""Tile geometry helpers — match a satellite position to a watched location.

A tile is a square area of side `size_km` centered on (lon, lat). We treat
1° latitude ≈ 111 km and 1° longitude ≈ 111 km × cos(lat). This is good
enough for the 5 km tiles we use; a sub-kilometre approximation error is
within SimSat's positional jitter.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional


KM_PER_DEG_LAT = 111.0


def _km_per_deg_lon(lat_deg: float) -> float:
    return max(0.1, 111.0 * math.cos(math.radians(lat_deg)))


def tile_contains(
    sat_lon: float, sat_lat: float,
    tile_lon: float, tile_lat: float,
    size_km: float,
) -> bool:
    """True if the satellite point falls inside the tile box of side `size_km`."""
    half = size_km / 2.0
    dlat_km = abs(sat_lat - tile_lat) * KM_PER_DEG_LAT
    dlon_km = abs(sat_lon - tile_lon) * _km_per_deg_lon(tile_lat)
    return dlat_km <= half and dlon_km <= half


def find_tile(
    sat_lon: float, sat_lat: float,
    tiles: Iterable[tuple[float, float]],
    size_km: float,
) -> Optional[tuple[float, float]]:
    """Return the first tile centre containing (sat_lon, sat_lat), else None."""
    for lon, lat in tiles:
        if tile_contains(sat_lon, sat_lat, lon, lat, size_km):
            return (lon, lat)
    return None


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in km."""
    R = 6371.0
    dlon = math.radians(lon2 - lon1)
    dlat = math.radians(lat2 - lat1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.asin(math.sqrt(a))
