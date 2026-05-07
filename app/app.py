"""Streamlit dashboard — map of hotspots + 14-panel viewer for any prediction.

Run:
    uv run streamlit run app/app.py
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import folium
import pandas as pd
import pydeck as pdk
import requests
import streamlit as st
from dotenv import load_dotenv
from streamlit_folium import st_folium

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
load_dotenv(REPO_ROOT / ".env")

from forestwhy.locations import LOCATIONS, LOCATIONS_BY_ID  # noqa: E402
from forestwhy.prompts import PANEL_ORDER  # noqa: E402

DB_PATH = REPO_ROOT / "predictions.db"
IMAGES_ROOT = REPO_ROOT / "db_images"

st.set_page_config(page_title="forestWHY", layout="wide", page_icon="🛰️")

# ─────────────────────────────────────────────────────────────────────────────
# Global styling
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
:root {
  --fw-bg-card:        rgba(255,255,255,0.03);
  --fw-bg-card-strong: rgba(255,255,255,0.06);
  --fw-border:         rgba(255,255,255,0.10);
  --fw-text-dim:       rgba(255,255,255,0.62);
  --fw-accent:         #4ec9b0;
}

/* Let Streamlit size the container naturally — fixed max-width breaks at high
   browser zoom levels (sidebar takes its width, main area pushed off-screen).
   Allow horizontal scroll on the body as a final safety net. */
.block-container { padding-top: 2rem; padding-bottom: 3rem; }
html, body { overflow-x: auto; }
.main, .stApp { min-width: 0; }
[data-testid="stVerticalBlock"] { min-width: 0; }
[data-testid="stHorizontalBlock"] { min-width: 0; flex-wrap: wrap; }

/* Title */
h1 { font-weight: 700 !important; letter-spacing: -0.5px; }

/* Section headings */
.fw-section {
    font-size: 14px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--fw-text-dim);
    margin: 28px 0 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--fw-border);
}

/* Hero card */
.fw-hero {
    background: linear-gradient(135deg,
        rgba(78,201,176,0.06),
        rgba(255,255,255,0.02));
    border: 1px solid var(--fw-border);
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 14px;
}
.fw-hero-title {
    font-size: 18px;
    font-weight: 600;
    letter-spacing: -0.2px;
    margin-bottom: 10px;
}
.fw-hero-meta {
    color: var(--fw-text-dim);
    font-size: 13px;
    line-height: 1.6;
    font-feature-settings: "tnum";
}
.fw-hero-meta code {
    background: var(--fw-bg-card-strong);
    padding: 1px 6px;
    border-radius: 4px;
    font-size: 12px;
}

/* Badges */
.fw-badge {
    display: inline-block;
    padding: 4px 11px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.6px;
    text-transform: uppercase;
    margin-right: 6px;
    margin-bottom: 4px;
    border: 1px solid transparent;
}
.fw-cls-deforestation     { background: rgba(220,50,47,0.18);  color: #f38680; border-color: rgba(220,50,47,0.55); }
.fw-cls-fire_disturbance  { background: rgba(203,75,22,0.18);  color: #ed9456; border-color: rgba(203,75,22,0.55); }
.fw-cls-afforestation     { background: rgba(133,153,0,0.20);  color: #b9c952; border-color: rgba(133,153,0,0.55); }
.fw-cls-stable_forest     { background: rgba(38,139,210,0.20); color: #6bb1e0; border-color: rgba(38,139,210,0.55); }
.fw-cls-stable_non_forest { background: rgba(101,123,131,0.25);color: #9eb1b9; border-color: rgba(101,123,131,0.55); }
.fw-cls-ambiguous         { background: rgba(181,137,0,0.18);  color: #d8b454; border-color: rgba(181,137,0,0.55); }
.fw-cls-                  { background: rgba(120,120,120,0.18);color: #aaa;    border-color: rgba(120,120,120,0.40); }

.fw-sev-high   { background: rgba(220,50,47,0.16);   color: #f38680; }
.fw-sev-medium { background: rgba(255,165,0,0.16);   color: #ffbd66; }
.fw-sev-low    { background: rgba(133,153,0,0.16);   color: #b9c952; }
.fw-sev-none   { background: rgba(120,120,120,0.16); color: #aaa;    }

.fw-driver { background: rgba(78,201,176,0.12); color: #6fcfba; border-color: rgba(78,201,176,0.40); }

/* Reasoning card */
.fw-reasoning {
    background: var(--fw-bg-card);
    border: 1px solid var(--fw-border);
    border-radius: 10px;
    padding: 18px 22px;
    max-height: 460px;
    overflow-y: auto;
    font-size: 14.5px;
    line-height: 1.7;
    font-family: -apple-system, system-ui, "SF Pro Text", "Segoe UI", sans-serif;
    color: rgba(255,255,255,0.88);
}
.fw-reasoning h2 {
    font-size: 14px !important;
    font-weight: 600;
    color: var(--fw-accent);
    margin: 16px 0 6px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}
.fw-reasoning h2:first-child { margin-top: 0; }
.fw-reasoning strong { color: rgba(255,255,255,0.95); }
.fw-reasoning::-webkit-scrollbar { width: 8px; }
.fw-reasoning::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 4px; }

/* Metric tweaks */
[data-testid="stMetric"] {
    background: var(--fw-bg-card);
    border: 1px solid var(--fw-border);
    padding: 14px 16px;
    border-radius: 10px;
}
[data-testid="stMetricLabel"] {
    color: var(--fw-text-dim);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-size: 11px !important;
}
[data-testid="stMetricValue"] {
    font-size: 20px !important;
    font-weight: 600;
}

/* Panel grid captions */
.fw-panel-cap {
    color: var(--fw-text-dim);
    font-size: 11px;
    margin-top: 4px;
    margin-bottom: 8px;
    text-align: center;
    letter-spacing: 0.3px;
}

/* Sidebar polish */
section[data-testid="stSidebar"] { border-right: 1px solid var(--fw-border); }
section[data-testid="stSidebar"] h2 { font-size: 16px !important; margin-top: 12px; }

/* Buttons */
button[data-testid="baseButton-secondaryFormSubmit"],
button[data-testid="stBaseButton-secondaryFormSubmit"] {
    font-weight: 600 !important;
    letter-spacing: 0.4px;
}

/* Map container */
[data-testid="stDeckGlJsonChart"] {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid var(--fw-border);
}
</style>
""", unsafe_allow_html=True)

st.title("forestWHY")
st.caption("On-orbit deforestation detection from Sentinel-2 imagery, powered by LFM2.5-forestWHY + I-JEPA.")


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


# ─────────────────────────────────────────────────────────────────────────────
# Live inference: backend probe + cached encoder
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading JEPA encoder (1.2 GB, ~5 s on first run)...")
def _get_encoder():
    """Cache the JEPA encoder across reruns. Streamlit holds it in process memory."""
    from forestwhy.jepa import load_jepa_encoder
    enc = load_jepa_encoder(device="auto")
    device = next(enc.parameters()).device.type
    return enc, device


def _probe_backend(name: str, base_url: str, timeout: float = 1.5) -> Optional[dict]:
    try:
        r = requests.get(f"{base_url}/models", timeout=timeout)
        if not r.ok:
            return None
        payload = r.json()
        items = payload.get("data") or payload.get("models") or []
        model_id = items[0].get("id") if items and isinstance(items[0], dict) else None
        return {"backend": name, "base_url": base_url, "model": model_id or "unknown"}
    except (requests.RequestException, ValueError):
        return None


def _detect_backend() -> Optional[dict]:
    """Probe the two known endpoints. vLLM (port 8000) wins over llama-server (8080)."""
    for name, url in [
        ("vllm",     os.environ.get("VLM_BASE_URL", "http://localhost:8000/v1")),
        ("llamacpp", os.environ.get("LLAMA_BASE_URL", "http://localhost:8080/v1")),
    ]:
        info = _probe_backend(name, url)
        if info is not None:
            return info
    return None


def _run_inference(
    *, lon: float, lat: float, region_id: str, biome: str, country: str,
    year_before: int, year_after: int, size_km: float, max_cloud_cover: Optional[float],
    backend_info: dict,
) -> int:
    """Fetch a temporal pair from SimSat, score it, write a DB row. Returns row id."""
    from forestwhy.db import init_db
    from forestwhy.evaluator import make_backend, model_name
    from forestwhy.live import fetch_13band_at_with_walkback
    from forestwhy.pipeline import score_tile

    encoder, enc_device = _get_encoder()
    predict = make_backend(
        backend_info["backend"],
        model=backend_info["model"],
        base_url=backend_info["base_url"],
    )
    label = model_name(backend_info["backend"], backend_info["model"])

    before_target = datetime(year_before, 7, 15, tzinfo=timezone.utc)
    after_target = datetime(year_after, 7, 15, tzinfo=timezone.utc)

    before_arr, before_meta = fetch_13band_at_with_walkback(
        lon=lon, lat=lat, target_ts=before_target,
        size_km=size_km, max_cloud_cover_pct=max_cloud_cover,
    )
    after_arr, after_meta = fetch_13band_at_with_walkback(
        lon=lon, lat=lat, target_ts=after_target,
        size_km=size_km, max_cloud_cover_pct=max_cloud_cover,
    )

    conn = init_db(DB_PATH)
    row_id = score_tile(
        conn=conn, images_root=IMAGES_ROOT,
        encoder=encoder, encoder_device=enc_device,
        predict=predict, backend_name=backend_info["backend"], model_label=label,
        lon=lon, lat=lat, size_km=size_km,
        region_id=region_id, biome=biome, country=country,
        before=before_arr, after=after_arr,
        before_timestamp=before_meta.get("achieved_timestamp")
                          or before_target.strftime("%Y-%m-%dT%H:%M:%SZ"),
        after_timestamp=after_meta.get("achieved_timestamp")
                         or after_target.strftime("%Y-%m-%dT%H:%M:%SZ"),
        before_cloud_cover=before_meta.get("cloud_cover"),
        after_cloud_cover=after_meta.get("cloud_cover"),
        source="dashboard",
    )
    conn.close()
    return row_id


df = _load_predictions(str(DB_PATH))

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar filters
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Run new inference")
    backend_info = _detect_backend()
    if backend_info:
        st.success(
            f"**{backend_info['backend']}** at `{backend_info['base_url']}`  \n"
            f"model: `{backend_info['model']}`"
        )
    else:
        st.warning(
            "No backend running. Start one in another terminal:\n\n"
            "**vLLM** (GPU) → `vllm serve Siddharth63/LFM2.5-forestWHY --port 8000`\n\n"
            "**llama-server** (laptop) → see `QUICKSTART.md` §3d."
        )

    # Radio is OUTSIDE the form so the conditional UI updates on click.
    # (Streamlit forms batch all widget changes until submit — radio inside
    #  a form would never reveal the lat/lon inputs.)
    mode = st.radio(
        "Location",
        ["Named hotspot", "Custom lat/lon"],
        horizontal=True,
        key="loc_mode",
    )

    # In Custom-lat/lon mode, the user picks coordinates by clicking the
    # folium picker (or by typing). The picker has to live OUTSIDE the form,
    # because folium's click → rerun → number_input update flow doesn't
    # work inside Streamlit forms.
    if mode == "Custom lat/lon":
        st.caption("Click the map below to drop a pin, or type coordinates.")
        if "picker_lat" not in st.session_state:
            st.session_state.picker_lat = -9.1
            st.session_state.picker_lon = -68.4

        _picker_map = folium.Map(
            location=[st.session_state.picker_lat, st.session_state.picker_lon],
            zoom_start=2,
            tiles="cartodbpositron",
        )
        folium.Marker(
            [st.session_state.picker_lat, st.session_state.picker_lon],
            tooltip=f"({st.session_state.picker_lon:.3f}, {st.session_state.picker_lat:.3f})",
            icon=folium.Icon(color="red", icon="crosshairs", prefix="fa"),
        ).add_to(_picker_map)
        # Optional: show the 18 watched hotspots as small grey dots so the user
        # has reference geography.
        for _h in LOCATIONS:
            folium.CircleMarker(
                [_h.lat, _h.lon], radius=3,
                color="#7aa57a", fill=True, fill_opacity=0.7, weight=1,
                tooltip=_h.id,
            ).add_to(_picker_map)
        _click = st_folium(
            _picker_map,
            height=260,
            width=None,
            returned_objects=["last_clicked"],
            key="custom_picker",
        )
        if _click and _click.get("last_clicked"):
            new_lat = float(_click["last_clicked"]["lat"])
            new_lon = float(_click["last_clicked"]["lng"])
            if (abs(new_lat - st.session_state.picker_lat) > 1e-6
                    or abs(new_lon - st.session_state.picker_lon) > 1e-6):
                st.session_state.picker_lat = new_lat
                st.session_state.picker_lon = new_lon
                st.rerun()

    with st.form("run_inference", clear_on_submit=False):
        if mode == "Named hotspot":
            loc_id = st.selectbox("Hotspot", list(LOCATIONS_BY_ID))
            _loc = LOCATIONS_BY_ID[loc_id]
            in_lon, in_lat = _loc.lon, _loc.lat
            in_biome, in_country = _loc.biome, _loc.country
            in_region = _loc.id
            st.caption(f"{_loc.country} · {_loc.biome} · ({_loc.lon:.4f}, {_loc.lat:.4f})")
        else:
            c1, c2 = st.columns(2)
            in_lon = c1.number_input(
                "Longitude", value=float(st.session_state.picker_lon),
                min_value=-180.0, max_value=180.0, format="%.4f",
                key="custom_lon",
            )
            in_lat = c2.number_input(
                "Latitude", value=float(st.session_state.picker_lat),
                min_value=-90.0, max_value=90.0, format="%.4f",
                key="custom_lat",
            )
            in_region = st.text_input(
                "Region label",
                value=f"custom_{in_lon:.2f}_{in_lat:.2f}",
                key="custom_region",
            )
            in_biome = "custom"
            in_country = "custom"

        c1, c2 = st.columns(2)
        year_before = c1.number_input("Before year", min_value=2017, max_value=2026, value=2020, step=1)
        year_after  = c2.number_input("After year",  min_value=2017, max_value=2026, value=2024, step=1)
        size_km = st.slider("Tile size (km)", 1.0, 10.0, 5.0, 0.5)
        max_cc  = st.slider("Max cloud cover (%) for walkback", 0, 100, 50, 5)

        submitted = st.form_submit_button(
            "🛰️  Run inference",
            disabled=(backend_info is None),
            use_container_width=True,
        )

    if submitted and backend_info:
        try:
            with st.spinner(f"Fetching tiles + scoring (~{15 if backend_info['backend']=='vllm' else 25} s)..."):
                new_id = _run_inference(
                    lon=in_lon, lat=in_lat, region_id=in_region,
                    biome=in_biome, country=in_country,
                    year_before=int(year_before), year_after=int(year_after),
                    size_km=float(size_km),
                    max_cloud_cover=(None if max_cc >= 100 else float(max_cc)),
                    backend_info=backend_info,
                )
            st.success(f"Wrote row #{new_id}. Refreshing dashboard ...")
            st.cache_data.clear()      # invalidate _load_predictions
            st.rerun()
        except Exception as exc:
            st.error(f"Inference failed: {exc}")

    st.divider()
    st.header("Filters")
    if df.empty:
        st.info(
            "No predictions yet. Use **Run new inference** above, or run `predict.py` from the CLI."
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

st.markdown('<div class="fw-section">Map · 18 watched hotspots and recent predictions</div>',
            unsafe_allow_html=True)

hotspot_df = pd.DataFrame([{
    "id": loc.id, "lon": loc.lon, "lat": loc.lat,
    "biome": loc.biome, "country": loc.country, "description": loc.description,
} for loc in LOCATIONS])

layers = [
    # Watched hotspots — outlined ring, always visible at any zoom.
    pdk.Layer(
        "ScatterplotLayer", data=hotspot_df,
        get_position=["lon", "lat"],
        get_radius=80_000,
        radius_units="meters",
        radius_min_pixels=7,
        radius_max_pixels=18,
        get_fill_color=[120, 200, 120, 90],
        get_line_color=[60, 180, 90, 240],
        line_width_min_pixels=2,
        stroked=True,
        filled=True,
        pickable=True,
    ),
]

if not filtered.empty:
    color_map = {
        "deforestation":     [220, 50, 47, 230],
        "fire_disturbance":  [203, 75, 22, 230],
        "afforestation":     [133, 153, 0, 230],
        "stable_forest":     [38, 139, 210, 230],
        "stable_non_forest": [101, 123, 131, 220],
        "ambiguous":         [181, 137, 0, 230],
    }
    plot_df = filtered.copy()
    plot_df["color"] = plot_df["change_class"].map(
        lambda c: color_map.get(c, [100, 100, 100, 200])
    )
    layers.append(
        pdk.Layer(
            "ScatterplotLayer", data=plot_df,
            get_position=["lon", "lat"],
            get_radius=30_000,
            radius_units="meters",
            radius_min_pixels=8,
            radius_max_pixels=18,
            get_fill_color="color",
            get_line_color=[255, 255, 255, 200],
            line_width_min_pixels=1.5,
            stroked=True,
            filled=True,
            pickable=True,
        )
    )

# Constrain interaction so the world doesn't pan into multiple wrapped copies.
# Allow zoom (scroll/pinch) but NOT drag-pan, and lock pitch/bearing.
_view = pdk.View(
    type="MapView",
    controller={"dragPan": False, "doubleClickZoom": False,
                "scrollZoom": True, "touchZoom": True,
                "dragRotate": False, "keyboard": False},
)

st.pydeck_chart(
    pdk.Deck(
        views=[_view],
        initial_view_state=pdk.ViewState(
            latitude=10, longitude=20, zoom=1.4,
            pitch=0, bearing=0, max_zoom=8, min_zoom=1.4,
        ),
        layers=layers,
        # CARTO Voyager: colourful land tones + soft blue oceans, no Mapbox token required.
        map_style="https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",
        tooltip={"text": "{id}\n{change_class} ({severity})\nconfidence={confidence}"},
    ),
    height=620,   # ~2:1 aspect for typical viewport — one world copy fits.
)

# ─────────────────────────────────────────────────────────────────────────────
# Predictions table + detail view
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    f'<div class="fw-section">Predictions · showing {len(filtered)} of {len(df)} total</div>',
    unsafe_allow_html=True,
)
if filtered.empty:
    st.info("No rows match the current filters. Adjust filters in the sidebar or run a new inference.")
    st.stop()

table_cols = [
    "id", "created_at", "region_id", "change_class", "severity",
    "driver_hypothesis", "area_pct", "confidence",
    "before_timestamp", "after_timestamp",
    "before_cloud_cover", "after_cloud_cover",
    "source", "model",
]
present = [c for c in table_cols if c in filtered.columns]
table_df = filtered[present].reset_index(drop=True)

st.caption("Click a row to inspect the model's reasoning + 14 panels for that prediction.")
table_event = st.dataframe(
    table_df,
    use_container_width=True,
    height=320,
    on_select="rerun",
    selection_mode="single-row",
    hide_index=True,
    key="predictions_table",
)

# Resolve selected row → DB id, defaulting to the most recent (top) row.
_selected_rows = []
if table_event is not None and hasattr(table_event, "selection"):
    _selected_rows = list(getattr(table_event.selection, "rows", []) or [])

if _selected_rows:
    selected = int(table_df.iloc[_selected_rows[0]]["id"])
else:
    selected = int(table_df.iloc[0]["id"])

row = filtered.loc[filtered["id"] == selected].iloc[0]


# ─────────────────────────────────────────────────────────────────────────────
# Hero card
# ─────────────────────────────────────────────────────────────────────────────

import html as _html

def _safe_str(v, default: str = "") -> str:
    """Return v as a string, treating None / NaN / 'None' / 'nan' as the default."""
    if v is None:
        return default
    if isinstance(v, float) and pd.isna(v):
        return default
    s = str(v).strip()
    if s.lower() in ("none", "nan", ""):
        return default
    return s


def _fmt_cc(v) -> str:
    return f"{float(v):.1f}%" if pd.notna(v) else "n/a"


def _badge(label: str, css_class: str) -> str:
    return f'<span class="fw-badge {css_class}">{_html.escape(str(label))}</span>'


cls = _safe_str(row["change_class"])
sev = _safe_str(row["severity"], "none")
drv = _safe_str(row["driver_hypothesis"])

badges = [_badge((cls or "—").replace("_", " "), f"fw-cls-{cls}")]
badges.append(_badge(f"{sev} severity", f"fw-sev-{sev}"))
if drv and drv != "unknown":
    badges.append(_badge(drv.replace("_", " "), "fw-driver fw-badge"))

hero = f"""
<div class="fw-hero">
    <div class="fw-hero-title">{' '.join(badges)}</div>
    <div class="fw-hero-meta">
        <code>{_html.escape(row['region_id'])}</code> ·
        ({row['lon']:.4f}, {row['lat']:.4f}) ·
        {row['size_km']:.1f} km tile ·
        <code>{_html.escape(str(row.get('model') or '—'))}</code>
        <br>
        <strong>{row['before_timestamp']}</strong> → <strong>{row['after_timestamp']}</strong> ·
        cloud cover: before {_fmt_cc(row.get('before_cloud_cover'))}, after {_fmt_cc(row.get('after_cloud_cover'))}
    </div>
</div>
"""
st.markdown(hero, unsafe_allow_html=True)


# Quick metrics row
m1, m2, m3, m4 = st.columns(4)
m1.metric("Change class", _safe_str(row["change_class"], "—").replace("_", " "))
m2.metric("Severity",     _safe_str(row["severity"], "—").title())
m3.metric("Area affected", f"{row['area_pct']:.1f}%" if pd.notna(row.get("area_pct")) else "—")
m4.metric("Confidence",
          f"{row['confidence']:.2f}" if pd.notna(row["confidence"]) else "—")


# ─────────────────────────────────────────────────────────────────────────────
# 14-panel grid (7 cols × 2 rows, fixed pixel size)
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="fw-section">14-panel input</div>', unsafe_allow_html=True)

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
    cols_per_row = 7
    panel_size_px = 180
    for i in range(0, len(PANEL_ORDER), cols_per_row):
        row_cols = st.columns(cols_per_row, gap="small")
        for j, name in enumerate(PANEL_ORDER[i:i + cols_per_row]):
            rel = panel_index.get(name)
            if rel is None:
                continue
            full = REPO_ROOT / rel
            with row_cols[j]:
                if not full.exists():
                    st.caption(f"{name}: missing")
                    continue
                st.image(str(full), width=panel_size_px)
                st.markdown(
                    f'<div class="fw-panel-cap">{i + j + 1}. {name.replace("_", " ")}</div>',
                    unsafe_allow_html=True,
                )


# ─────────────────────────────────────────────────────────────────────────────
# Reasoning — scrollable, readable
# ─────────────────────────────────────────────────────────────────────────────

if row.get("reasoning"):
    st.markdown('<div class="fw-section">Model reasoning (10-step protocol)</div>',
                unsafe_allow_html=True)

    def _md_lite_to_html(text: str) -> str:
        """Tiny converter for the model's '## Step N:' markdown."""
        out: list[str] = []
        for raw_line in text.split("\n"):
            line = raw_line.rstrip()
            if not line:
                out.append("<br>")
                continue
            if line.startswith("### "):
                out.append(f"<h3>{_html.escape(line[4:])}</h3>")
            elif line.startswith("## "):
                out.append(f"<h2>{_html.escape(line[3:])}</h2>")
            elif line.startswith("# "):
                out.append(f"<h2>{_html.escape(line[2:])}</h2>")
            else:
                escaped = _html.escape(line)
                # Bold **...**
                import re as _re_local
                escaped = _re_local.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
                out.append(f"<p style='margin: 4px 0;'>{escaped}</p>")
        return "\n".join(out)

    st.markdown(
        f'<div class="fw-reasoning">{_md_lite_to_html(row["reasoning"])}</div>',
        unsafe_allow_html=True,
    )

if row.get("cloud_cover_note"):
    st.caption(f"☁️  Cloud-cover note from model: {row['cloud_cover_note']}")
