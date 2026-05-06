#!/usr/bin/env python3
"""Pre-warm the HuggingFace cache.

Pulls (a) the JEPA Sentinel-2 ViT-L/8 encoder used to build the 6 differential
panels and (b) the fine-tuned LFM2.5-forestWHY VLM weights so vLLM can start
without an additional download. Idempotent — re-running hits the cache.

Usage:
    uv run python scripts/download_weights.py
    uv run python scripts/download_weights.py --vlm-only
    uv run python scripts/download_weights.py --jepa-only
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

log = logging.getLogger("forestwhy.download_weights")


def _download_jepa() -> None:
    from huggingface_hub import hf_hub_download
    repo = os.environ.get("JEPA_HF_REPO", "Siddharth63/forestWHY-JEPA-vitl")
    fn = os.environ.get("JEPA_HF_FILENAME", "s2_ijepa_gee_vitl_full_encoder_final.pt")
    log.info("Downloading JEPA encoder %s/%s ...", repo, fn)
    path = hf_hub_download(repo_id=repo, filename=fn)
    log.info("  -> %s", path)


def _download_vlm() -> None:
    from huggingface_hub import snapshot_download
    repo = os.environ.get("VLM_MODEL", "Siddharth63/LFM2.5-forestWHY")
    log.info("Downloading VLM snapshot %s (~17 GB; this can take a while) ...", repo)
    path = snapshot_download(repo_id=repo)
    log.info("  -> %s", path)


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s",
                        datefmt="%H:%M:%S")
    p = argparse.ArgumentParser()
    p.add_argument("--jepa-only", action="store_true")
    p.add_argument("--vlm-only", action="store_true")
    args = p.parse_args()

    try:
        if not args.vlm_only:
            _download_jepa()
        if not args.jepa_only:
            _download_vlm()
    except Exception as exc:
        log.error("Download failed: %s", exc)
        log.error("If a repo is private, run `huggingface-cli login` first or set HF_TOKEN.")
        sys.exit(1)


if __name__ == "__main__":
    main()
