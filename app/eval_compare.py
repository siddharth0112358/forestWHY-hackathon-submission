"""Side-by-side viewer: base LFM2.5-VL vs fine-tuned forestWHY for one tile.

Reads `evals/<run>/results.json` from `scripts/evaluate.py`. Pick a sample
id and the app shows the 14 panels plus both models' predicted JSON
side-by-side, highlighting fields where they disagree.

Run:
    uv run streamlit run app/eval_compare.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from forestwhy.prompts import PANEL_ORDER  # noqa: E402

EVALS_ROOT = REPO_ROOT / "evals"

st.set_page_config(page_title="forestWHY — eval", layout="wide")
st.title("forestWHY — base vs fine-tuned comparison")

if not EVALS_ROOT.exists():
    st.warning(f"No `evals/` directory yet. Run `uv run python scripts/evaluate.py --backend vllm` first.")
    st.stop()

runs = sorted([p for p in EVALS_ROOT.iterdir() if p.is_dir()], reverse=True)
if not runs:
    st.warning("No eval runs in `evals/`.")
    st.stop()

run_dir = st.sidebar.selectbox("Eval run", runs, format_func=lambda p: p.name)
results_path = run_dir / "results.json"
if not results_path.exists():
    st.error(f"{results_path} missing.")
    st.stop()

with open(results_path) as f:
    results = json.load(f)

samples = results.get("samples", [])
if not samples:
    st.info("This run has no per-sample records.")
    st.stop()

idx = st.sidebar.slider("Sample", 0, len(samples) - 1, 0)
sample = samples[idx]

st.subheader(f"Sample {sample.get('id', idx)}  —  {sample.get('region_id', '')}")

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Base model**")
    st.json(sample.get("base", {}))
with c2:
    st.markdown("**Fine-tuned model**")
    st.json(sample.get("finetuned", {}))

with st.expander("Ground truth"):
    st.json(sample.get("truth", {}))

panel_dir = sample.get("panels_dir")
if panel_dir:
    full_dir = REPO_ROOT / panel_dir
    cols_per_row = 4
    for i in range(0, len(PANEL_ORDER), cols_per_row):
        row_cols = st.columns(cols_per_row)
        for j, name in enumerate(PANEL_ORDER[i:i + cols_per_row]):
            path = full_dir / f"{name}.png"
            if path.exists():
                row_cols[j].image(str(path), caption=f"{i + j + 1}. {name}", width="stretch")
