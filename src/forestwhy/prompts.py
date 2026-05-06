"""Prompt templates and JSON schema for the forestWHY VLM.

Single source of truth — referenced by `evaluator.py`, `annotator.py`, and the
README. The fine-tuned `Siddharth63/LFM2.5-forestWHY` model expects exactly
14 images in the order given by `PANEL_ORDER`.
"""

from __future__ import annotations


PANEL_ORDER: tuple[str, ...] = (
    "rgb_before",          # 1
    "rgb_after",           # 2
    "nir_fc_before",       # 3  B8-B4-B3
    "nir_fc_after",        # 4
    "swir_before",         # 5  B12-B8-B4
    "swir_after",          # 6
    "delta_ndvi",          # 7
    "delta_nbr",           # 8
    "attention_multi",     # 9   JEPA
    "embedding_change",    # 10  JEPA
    "cropa_roads",         # 11  JEPA
    "delta_attn_role",     # 12  JEPA
    "head_disagreement",   # 13  JEPA
    "pca_semantic",        # 14  JEPA
)


SYSTEM_PROMPT = (
    "You are forestWHY, a remote-sensing analyst that classifies forest-cover change "
    "in a 5 km Sentinel-2 tile. You receive 14 images: panels 1–8 are spectral "
    "composites and difference maps; panels 9–14 are differential features from an "
    "I-JEPA ViT-L/8 encoder pretrained on global Sentinel-2. Use spectral panels for "
    "what changed and JEPA panels for where attention shifted and which patches "
    "semantically transitioned. Reply with one JSON object that conforms exactly to "
    "the schema. Do not output prose outside the JSON."
)


USER_PROMPT_TEMPLATE = """\
Tile centre: lon={lon:.4f}, lat={lat:.4f}, size={size_km} km.
Before timestamp (UTC): {before_ts}    cloud cover: {before_cc}
After timestamp  (UTC): {after_ts}    cloud cover: {after_cc}
Region id: {region_id}    biome: {biome}    country: {country}

Image  1: true-colour RGB before.
Image  2: true-colour RGB after.
Image  3: NIR false colour (B8 B4 B3) before — vegetation vigour.
Image  4: NIR false colour after.
Image  5: SWIR composite (B12 B8 B4) before — burn scars and bare soil.
Image  6: SWIR composite after.
Image  7: Delta NDVI (B8 - B4 normalised), red = vegetation loss.
Image  8: Delta NBR  (B8 - B11 normalised), red = biomass loss.
Image  9: JEPA multi-scale attention (R fine, G mid, B landscape).
Image 10: JEPA embedding cosine distance before -> after, red = semantic class flipped.
Image 11: JEPA cross-patch correlation, hot = linear structures (roads, logging tracks).
Image 12: JEPA delta attention role, red = patches that gained scene importance.
Image 13: JEPA head disagreement, hot = ambiguous land cover.
Image 14: JEPA PCA semantic clusters, left half = before, right half = after.

Return the deforestation analysis JSON for this tile.
"""


JSON_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "change_class", "severity", "area_pct",
        "driver_hypothesis", "confidence", "reasoning", "cloud_cover_note",
    ],
    "properties": {
        "change_class": {
            "type": "string",
            "enum": [
                "deforestation",
                "afforestation",
                "fire_disturbance",
                "stable_forest",
                "stable_non_forest",
                "ambiguous",
            ],
        },
        "severity": {
            "type": "string",
            "enum": ["none", "low", "medium", "high"],
        },
        "area_pct": {"type": "number", "minimum": 0, "maximum": 100},
        "driver_hypothesis": {
            "type": "string",
            "enum": [
                "agricultural_clearing",
                "logging_road",
                "mining",
                "fire",
                "flood",
                "plantation",
                "natural_regrowth",
                "unknown",
            ],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reasoning":  {"type": "string", "maxLength": 600},
        "cloud_cover_note": {"type": "string", "maxLength": 200},
    },
}


def render_user_prompt(metadata: dict) -> str:
    """Format USER_PROMPT_TEMPLATE with safe defaults for missing fields."""
    return USER_PROMPT_TEMPLATE.format(
        lon=metadata.get("lon", 0.0),
        lat=metadata.get("lat", 0.0),
        size_km=metadata.get("size_km", 5.0),
        before_ts=metadata.get("before_timestamp", "unknown"),
        after_ts=metadata.get("after_timestamp", "unknown"),
        before_cc=_fmt_cc(metadata.get("before_cloud_cover")),
        after_cc=_fmt_cc(metadata.get("after_cloud_cover")),
        region_id=metadata.get("region_id", "unknown"),
        biome=metadata.get("biome", "unknown"),
        country=metadata.get("country", "unknown"),
    )


def _fmt_cc(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.0f}%"
