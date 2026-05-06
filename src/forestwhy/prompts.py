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


# Verbatim from the LFM2.5-forestWHY model card, so the fine-tuned weights see
# the same prompt at inference as during training.
SYSTEM_PROMPT = (
    "You are an expert remote sensing analyst and tropical ecologist "
    "specializing in forest cover change detection from Sentinel-2 satellite "
    "imagery. You have access to both raw spectral data AND outputs from a "
    "trained I-JEPA Vision Transformer encoder (ViT-L/8, 24 layers, trained "
    "on 1.57M Sentinel-2 patches).\n\n"
    "For each observation you receive 14 image panels:\n"
    "SPECTRAL (1-8): RGB before/after, NIR before/after, SWIR before/after, "
    "DELTA_NDVI, DELTA_NBR\n"
    "JEPA ENCODER (9-14): Multi-scale attention, Embedding change, CroPA roads, "
    "Delta attention role, Head disagreement, PCA semantic clusters\n\n"
    "Critical rules:\n"
    "- Provide detailed 10-step reasoning citing specific panel numbers\n"
    "- JEPA panels supersede spectral panels for ambiguous cases\n"
    "- Panel 10 embedding change is more reliable than DELTA_NDVI for degradation detection\n"
    "- Panel 11 CroPA road presence is the strongest predictor of continued deforestation\n"
    "- Panel 14 PCA cluster split is ground truth for genuine land cover transition\n"
    "- Write detailed prose for each step - minimum 4 sentences per step"
)


# Verbatim USER_PROMPT_TEMPLATE structure from the model card. We compute the
# DELTA_NDVI / DELTA_NBR magnitudes ourselves (they are present in the spectral
# panels by construction) and pass region/year metadata through.
USER_PROMPT_TEMPLATE = """\
Analyze this Sentinel-2 satellite observation showing land cover change.

Location: {region_id}, {lat} N {lon} E
Period: {before_ts} to {after_ts} | Patch: 64x64 px at ~19m/px (~1.2km x 1.2km)
Biome: {biome}    country: {country}
Cloud cover: before={before_cc}  after={after_cc}

Panels (14 total):
  1 RGB before                2 RGB after
  3 NIR false-colour before   4 NIR false-colour after
  5 SWIR composite before     6 SWIR composite after
  7 DELTA_NDVI                8 DELTA_NBR
  9 JEPA multi-scale attention (R fine, G mid, B landscape)
 10 JEPA embedding change (cosine distance, red = semantic class flipped)
 11 JEPA CroPA roads (hot = linear structures: roads, logging tracks)
 12 JEPA delta attention role (red = gained scene importance)
 13 JEPA head disagreement (hot = ambiguous land cover)
 14 JEPA PCA semantic clusters (left = before, right = after)

Follow the 10-step reasoning protocol. Use JEPA panels to go beyond
what spectral indices alone can detect.
"""


# The fine-tune was trained to produce free-form prose with a 10-step
# reasoning chain, not structured JSON. We accept any object the model returns,
# preserve it verbatim in `raw_response`, and use the normaliser below to
# extract dashboard-friendly fields. This keeps the door open for either
# structured-output fine-tunes in the future or natural-prose models today.
DASHBOARD_FIELDS: tuple[str, ...] = (
    "change_class",        # deforestation | afforestation | fire_disturbance | stable_forest | stable_non_forest | ambiguous
    "severity",            # none | low | medium | high
    "area_pct",            # 0-100
    "driver_hypothesis",   # agricultural_clearing | logging_road | mining | fire | flood | plantation | natural_regrowth | unknown
    "confidence",          # 0-1
    "reasoning",           # short prose summary
    "cloud_cover_note",    # free text
)

JSON_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": True,
    "properties": {
        "change_class": {"type": "string"},
        "severity":     {"type": "string"},
        "area_pct":     {"type": "number"},
        "driver_hypothesis": {"type": "string"},
        "confidence":   {"type": ["number", "string"]},
        "reasoning":    {"type": "string"},
        "cloud_cover_note": {"type": "string"},
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
