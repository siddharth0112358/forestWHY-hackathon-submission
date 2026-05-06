"""SQLite store for forestWHY predictions.

Schema mirrors wildfire-prevention's bookkeeping columns (so dashboard ports
mechanically) and replaces the wildfire-specific booleans with deforestation
scalars + before/after pair metadata.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional


SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    lon                 REAL    NOT NULL,
    lat                 REAL    NOT NULL,
    timestamp           TEXT    NOT NULL,
    size_km             REAL    NOT NULL,
    source              TEXT    NOT NULL,
    model               TEXT    NOT NULL,
    created_at          TEXT    NOT NULL,
    region_id           TEXT,

    before_timestamp    TEXT    NOT NULL,
    after_timestamp     TEXT    NOT NULL,
    before_cloud_cover  REAL,
    after_cloud_cover   REAL,

    change_class        TEXT,
    severity            TEXT,
    area_pct            REAL,
    driver_hypothesis   TEXT,
    confidence          REAL,
    reasoning           TEXT,
    cloud_cover_note    TEXT,

    panels_dir          TEXT,
    panels_manifest     TEXT,
    rgb_before_path     TEXT,
    rgb_after_path      TEXT,
    raw_response        TEXT
);

CREATE INDEX IF NOT EXISTS idx_predictions_region_created
    ON predictions(region_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_change_class
    ON predictions(change_class);
"""


def init_db(path: str | Path) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def insert_prediction(
    conn: sqlite3.Connection,
    *,
    lon: float,
    lat: float,
    timestamp: str,
    size_km: float,
    source: str,
    model: str,
    region_id: Optional[str],
    before_timestamp: str,
    after_timestamp: str,
    before_cloud_cover: Optional[float],
    after_cloud_cover: Optional[float],
    prediction: Mapping[str, Any],
    panels_dir: Optional[str] = None,
    panels_manifest: Optional[list[dict[str, str]]] = None,
    rgb_before_path: Optional[str] = None,
    rgb_after_path: Optional[str] = None,
) -> int:
    """Insert one prediction row. Returns the row id."""
    created_at = datetime.now(timezone.utc).isoformat()
    cur = conn.execute(
        """
        INSERT INTO predictions (
            lon, lat, timestamp, size_km, source, model, created_at, region_id,
            before_timestamp, after_timestamp, before_cloud_cover, after_cloud_cover,
            change_class, severity, area_pct, driver_hypothesis,
            confidence, reasoning, cloud_cover_note,
            panels_dir, panels_manifest, rgb_before_path, rgb_after_path,
            raw_response
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?,
            ?, ?, ?, ?,
            ?
        )
        """,
        (
            lon, lat, timestamp, size_km, source, model, created_at, region_id,
            before_timestamp, after_timestamp, before_cloud_cover, after_cloud_cover,
            prediction.get("change_class"),
            prediction.get("severity"),
            prediction.get("area_pct"),
            prediction.get("driver_hypothesis"),
            prediction.get("confidence"),
            prediction.get("reasoning"),
            prediction.get("cloud_cover_note"),
            panels_dir,
            json.dumps(panels_manifest) if panels_manifest is not None else None,
            rgb_before_path,
            rgb_after_path,
            json.dumps(dict(prediction)),
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def fetch_recent(conn: sqlite3.Connection, n: int = 50) -> list[sqlite3.Row]:
    return list(conn.execute(
        "SELECT * FROM predictions ORDER BY created_at DESC LIMIT ?", (n,)
    ))


def fetch_all(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return list(conn.execute(
        "SELECT * FROM predictions ORDER BY created_at DESC"
    ))


def fetch_by_id(conn: sqlite3.Connection, row_id: int) -> Optional[sqlite3.Row]:
    rows = list(conn.execute(
        "SELECT * FROM predictions WHERE id = ?", (row_id,)
    ))
    return rows[0] if rows else None


def update_artifact_paths(
    conn: sqlite3.Connection,
    row_id: int,
    *,
    panels_dir: Optional[str] = None,
    panels_manifest: Optional[list[dict[str, str]]] = None,
    rgb_before_path: Optional[str] = None,
    rgb_after_path: Optional[str] = None,
) -> None:
    conn.execute(
        """
        UPDATE predictions
        SET panels_dir       = COALESCE(?, panels_dir),
            panels_manifest  = COALESCE(?, panels_manifest),
            rgb_before_path  = COALESCE(?, rgb_before_path),
            rgb_after_path   = COALESCE(?, rgb_after_path)
        WHERE id = ?
        """,
        (
            panels_dir,
            json.dumps(panels_manifest) if panels_manifest is not None else None,
            rgb_before_path,
            rgb_after_path,
            row_id,
        ),
    )
    conn.commit()
