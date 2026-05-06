"""Streamlit dashboard — map of hotspots + 14-panel viewer for any prediction.

Run:
    uv run streamlit run app/app.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from forestwhy.locations import LOCATIONS  # noqa: E402
from forestwhy.prompts import PANEL_ORDER  # noqa: E402

DB_PATH = REPO_ROOT / "predictions.db"

st.set_page_config(page_title="forestWHY", layout="wide")
st.title("forestWHY — On-orbit Deforestation Detection")
st.caption("Live predictions from LFM2.5-forestWHY over Sentinel-2 imagery (SimSat).")


@st.cache_data(ttl=10.0)
def _load_predictions(db_path: str) -> pd.DataFrame:
    if not Path(db_path).exists():
        return pd.DataFrame()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = list(conn.execute("SELECT * FROM predictions ORDER BY created_at DESC"))
    conn.close()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([dict(r) for r in rows])


df = _load_predictions(str(DB_PATH))

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar filters
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Filters")
    if df.empty:
        st.info(
            "No predictions yet. Run `uv run python scripts/predict.py --smoke-test --backend stub` "
            "to seed a test row, or start the live watch loop."
        )
    region_filter = st.multiselect(
        "Region",
        sorted(df["region_id"].dropna().unique()) if not df.empty else [],
    )
    class_filter = st.multiselect(
        "Change class",
        sorted(df["change_class"].dropna().unique()) if not df.empty else [],
    )
    severity_filter = st.multiselect(
        "Severity",
        ["none", "low", "medium", "high"],
    )
    driver_filter = st.multiselect(
        "Driver",
        sorted(df["driver_hypothesis"].dropna().unique()) if not df.empty else [],
    )
    exclude_clouds = st.checkbox("Exclude high cloud cover (>60%)", value=True)


filtered = df.copy()
if not filtered.empty:
    if region_filter:
        filtered = filtered[filtered["region_id"].isin(region_filter)]
    if class_filter:
        filtered = filtered[filtered["change_class"].isin(class_filter)]
    if severity_filter:
        filtered = filtered[filtered["severity"].isin(severity_filter)]
    if driver_filter:
        filtered = filtered[filtered["driver_hypothesis"].isin(driver_filter)]
    if exclude_clouds:
        cc = filtered["after_cloud_cover"].fillna(0)
        filtered = filtered[cc <= 60.0]

# ─────────────────────────────────────────────────────────────────────────────
# Map
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("Watched hotspots and recent predictions")

hotspot_df = pd.DataFrame([{
    "id": loc.id, "lon": loc.lon, "lat": loc.lat,
    "biome": loc.biome, "country": loc.country, "description": loc.description,
} for loc in LOCATIONS])

layers = [
    pdk.Layer(
        "ScatterplotLayer", data=hotspot_df,
        get_position=["lon", "lat"], get_radius=80_000,
        get_fill_color=[120, 200, 120, 80], pickable=True,
    ),
]

if not filtered.empty:
    color_map = {
        "deforestation":     [220, 50, 47, 200],
        "fire_disturbance":  [203, 75, 22, 200],
        "afforestation":     [133, 153, 0, 200],
        "stable_forest":     [38, 139, 210, 200],
        "stable_non_forest": [101, 123, 131, 180],
        "ambiguous":         [181, 137, 0, 200],
    }
    plot_df = filtered.copy()
    plot_df["color"] = plot_df["change_class"].map(
        lambda c: color_map.get(c, [100, 100, 100, 180])
    )
    layers.append(
        pdk.Layer(
            "ScatterplotLayer", data=plot_df,
            get_position=["lon", "lat"], get_radius=30_000,
            get_fill_color="color", pickable=True,
        )
    )

st.pydeck_chart(pdk.Deck(
    initial_view_state=pdk.ViewState(latitude=0, longitude=-30, zoom=1.2),
    layers=layers, map_style=None,
    tooltip={"text": "{id}\n{change_class} ({severity})\nconfidence={confidence}"},
))

# ─────────────────────────────────────────────────────────────────────────────
# Predictions table + detail view
# ─────────────────────────────────────────────────────────────────────────────

st.subheader(f"Predictions ({len(filtered)} of {len(df)})")
if filtered.empty:
    st.write("No rows match the current filters.")
    st.stop()

table_cols = [
    "id", "created_at", "region_id", "change_class", "severity",
    "driver_hypothesis", "area_pct", "confidence",
    "before_timestamp", "after_timestamp",
    "before_cloud_cover", "after_cloud_cover",
    "source", "model",
]
present = [c for c in table_cols if c in filtered.columns]
st.dataframe(filtered[present], use_container_width=True, height=300)

selected = st.selectbox(
    "Inspect prediction id",
    options=filtered["id"].tolist(),
    format_func=lambda i: f"#{i}  {filtered.loc[filtered['id'] == i, 'region_id'].iloc[0]}  "
                          f"{filtered.loc[filtered['id'] == i, 'change_class'].iloc[0]}",
)

row = filtered.loc[filtered["id"] == selected].iloc[0]

c1, c2 = st.columns([2, 1])
with c1:
    st.markdown(f"**Tile:** {row['region_id']} ({row['lon']:.4f}, {row['lat']:.4f}) "
                f"size {row['size_km']} km")
    st.markdown(f"**Window:** {row['before_timestamp']} → {row['after_timestamp']}")
    cc_before = row.get("before_cloud_cover")
    cc_after = row.get("after_cloud_cover")
    if pd.notna(cc_before) or pd.notna(cc_after):
        st.markdown(f"**Cloud cover:** before={cc_before}  after={cc_after}")
with c2:
    st.metric("change_class", row["change_class"] or "—")
    st.metric("severity", row["severity"] or "—")
    st.metric("driver", row["driver_hypothesis"] or "—")
    st.metric("confidence", f"{row['confidence']:.2f}" if pd.notna(row["confidence"]) else "—")
    if pd.notna(row.get("area_pct")):
        st.metric("area_pct", f"{row['area_pct']:.1f}%")

if row.get("reasoning"):
    st.markdown("**Reasoning**")
    st.write(row["reasoning"])
if row.get("cloud_cover_note"):
    st.caption(f"Cloud-cover note: {row['cloud_cover_note']}")

# ─────────────────────────────────────────────────────────────────────────────
# 14-panel grid
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("14-panel input")
manifest_raw = row.get("panels_manifest")
manifest: list[dict] = []
if manifest_raw:
    try:
        manifest = json.loads(manifest_raw)
    except Exception:
        manifest = []

if not manifest:
    st.info("Panel artefacts not found on disk for this row.")
else:
    panel_index = {m["name"]: m["path"] for m in manifest}
    cols_per_row = 4
    for i in range(0, len(PANEL_ORDER), cols_per_row):
        row_cols = st.columns(cols_per_row)
        for j, name in enumerate(PANEL_ORDER[i:i + cols_per_row]):
            rel = panel_index.get(name)
            if rel is None:
                continue
            full = REPO_ROOT / rel
            if not full.exists():
                row_cols[j].caption(f"{name}: missing")
                continue
            row_cols[j].image(str(full), caption=f"{i + j + 1}. {name}", width="stretch")

with st.expander("Raw VLM JSON"):
    raw = row.get("raw_response")
    if raw:
        try:
            st.json(json.loads(raw))
        except Exception:
            st.code(raw)
