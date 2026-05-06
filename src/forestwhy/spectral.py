"""Spectral panel generators — 8 panels paired with the 6 JEPA panels.

The fine-tuned LFM2.5-forestWHY VLM expects exactly 14 panels per call. This
module produces panels 1-8 (spectral side); `jepa.make_jepa_panels` produces
the remaining 6 plus two spectral diffs (delta_ndvi, delta_nbr). The 8
spectral panels in `make_spectral_panels` are:

    rgb_before     — true colour B4-B3-B2 before
    rgb_after      — true colour after
    nir_fc_before  — B8-B4-B3 NIR false colour before (vegetation vigour)
    nir_fc_after   — NIR false colour after
    swir_before    — B12-B8-B4 SWIR composite before (burn scars, bare soil)
    swir_after     — SWIR composite after
    delta_ndvi     — ΔNDVI heatmap (re-uses jepa.make_jepa_panels output if
                      caller wants exact alignment; we recompute here for the
                      spectral-only fallback path)
    delta_nbr      — ΔNBR heatmap

Inputs to public functions are 13-band float32 arrays of shape (13, H, W) in
[0, 1]; outputs are PIL.Image.Image at the requested `size`.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from PIL import Image

from .jepa import BAND_IDX, colorise, patch_to_band_dict, to_rgb_image


def _ndvi(p: np.ndarray) -> np.ndarray:
    return (p[BAND_IDX["B8"]] - p[BAND_IDX["B4"]]) / (
        p[BAND_IDX["B8"]] + p[BAND_IDX["B4"]] + 1e-6
    )


def _nbr(p: np.ndarray) -> np.ndarray:
    return (p[BAND_IDX["B8"]] - p[BAND_IDX["B11"]]) / (
        p[BAND_IDX["B8"]] + p[BAND_IDX["B11"]] + 1e-6
    )


def make_spectral_panels(
    before: np.ndarray,
    after: np.ndarray,
    size: int = 128,
) -> Dict[str, Image.Image]:
    """Build the 8 spectral panels for one before/after pair.

    Returned keys (PANEL_ORDER 1-8 in src/forestwhy/prompts.py):
        rgb_before, rgb_after, nir_fc_before, nir_fc_after,
        swir_before, swir_after, delta_ndvi, delta_nbr
    """
    b_bands = patch_to_band_dict(before)
    a_bands = patch_to_band_dict(after)

    panels: Dict[str, Image.Image] = {
        "rgb_before":    to_rgb_image(b_bands["B4"], b_bands["B3"], b_bands["B2"], size),
        "rgb_after":     to_rgb_image(a_bands["B4"], a_bands["B3"], a_bands["B2"], size),
        "nir_fc_before": to_rgb_image(b_bands["B8"], b_bands["B4"], b_bands["B3"], size),
        "nir_fc_after":  to_rgb_image(a_bands["B8"], a_bands["B4"], a_bands["B3"], size),
        "swir_before":   to_rgb_image(b_bands["B12"], b_bands["B8"], b_bands["B4"], size),
        "swir_after":    to_rgb_image(a_bands["B12"], a_bands["B8"], a_bands["B4"], size),
        "delta_ndvi":    colorise(_ndvi(after) - _ndvi(before), "RdYlGn", -0.5, 0.5, size),
        "delta_nbr":     colorise(_nbr(after) - _nbr(before), "RdYlGn", -0.5, 0.5, size),
    }
    return panels
