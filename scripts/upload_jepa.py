#!/usr/bin/env python3
"""Upload the I-JEPA Sentinel-2 ViT-L/8 encoder to a HuggingFace repo.

One-off helper. After running this, `forestwhy.jepa.load_jepa_encoder` will
auto-download the weights from the configured repo (no JEPA_CKPT needed).

Usage:
    uv run python scripts/upload_jepa.py \\
        --src /Users/admin/Desktop/forestWHY/forestwhy-sub/forestWHY_jepa/s2_ijepa_gee_vitl_full_encoder_final.pt \\
        --repo Siddharth63/forestWHY-JEPA-vitl
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

log = logging.getLogger("forestwhy.upload_jepa")


MODEL_CARD = """\
---
license: mit
tags:
  - sentinel-2
  - i-jepa
  - remote-sensing
  - earth-observation
  - vision-transformer
library_name: pytorch
---

# forestWHY JEPA — Sentinel-2 I-JEPA ViT-L/8 encoder

A 302 M parameter Vision Transformer (ViT-Large/8) pretrained with
[I-JEPA](https://arxiv.org/abs/2301.08243) on global Sentinel-2 imagery.
Used by [forestWHY](https://huggingface.co/Siddharth63/LFM2.5-forestWHY) to
produce six differential attention/embedding panels from a temporal pair of
13-band Sentinel-2 tiles.

## Architecture

| Field        | Value                          |
|--------------|--------------------------------|
| Backbone     | ViT-Large/8                    |
| Embed dim    | 1024                           |
| Depth        | 24 transformer blocks          |
| Heads        | 16                             |
| Patch size   | 8 × 8                          |
| Input        | 13-band Sentinel-2, 64 × 64    |
| Pretraining  | I-JEPA (no contrastive heads)  |

Input bands (channel order): `B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B10, B11, B12`
(Sentinel-Hub naming).

## Training data

Two-stage pretraining:

1. **BigEarthNet** — 590 K labelled Sentinel-2 patches, used for the initial
   I-JEPA warm-up.
2. **Google Earth Engine** — ~ 330 K locations × 8 years (2017–2024) = 1.57 M
   temporal Sentinel-2 patches, sampled to over-represent forest-loss
   pixels using Hansen Global Forest Change as a prior.

## Files

- `s2_ijepa_gee_vitl_full_encoder_final.pt` — encoder-only checkpoint (1.1 GB).
  Loadable via the `S2Encoder` class in
  [forestwhy/cookbook/src/forestwhy/jepa.py](https://github.com/Liquid4All/cookbook).

## Usage

```python
from huggingface_hub import hf_hub_download
import torch

# Drop the S2Encoder class from forestwhy.jepa into your project, then:
from forestwhy.jepa import S2Encoder

ckpt_path = hf_hub_download(
    repo_id="Siddharth63/forestWHY-JEPA-vitl",
    filename="s2_ijepa_gee_vitl_full_encoder_final.pt",
)
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
encoder = S2Encoder(embed_dim=1024, depth=24, num_heads=16)
encoder.load_state_dict(ckpt["encoder"])
encoder.eval()
```

For the full panel-generation pipeline (six differential panels per
temporal pair), see
[forestwhy.jepa.make_jepa_panels](https://github.com/Liquid4All/cookbook).

## License

MIT. The training data is © Copernicus / ESA (Sentinel-2) and the European
Space Agency under the Copernicus open licence.

## Citation

Built for the Liquid AI hackathon LFM track.
"""


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s  %(message)s",
                        datefmt="%H:%M:%S")
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="Path to the .pt encoder file.")
    p.add_argument("--repo", default="Siddharth63/forestWHY-JEPA-vitl")
    p.add_argument("--private", action="store_true")
    p.add_argument("--filename", default="s2_ijepa_gee_vitl_full_encoder_final.pt")
    args = p.parse_args()

    src = Path(args.src).resolve()
    if not src.exists():
        log.error("Source not found: %s", src)
        sys.exit(1)

    from huggingface_hub import HfApi, create_repo

    log.info("Creating/ensuring repo %s (private=%s)", args.repo, args.private)
    create_repo(args.repo, repo_type="model", exist_ok=True, private=args.private)

    api = HfApi()

    log.info("Uploading model card README ...")
    api.upload_file(
        path_or_fileobj=MODEL_CARD.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="model",
        commit_message="Add model card",
    )

    log.info("Uploading %s (%.1f GB) ... this can take several minutes",
             src.name, src.stat().st_size / (1024 ** 3))
    api.upload_file(
        path_or_fileobj=str(src),
        path_in_repo=args.filename,
        repo_id=args.repo,
        repo_type="model",
        commit_message="Add I-JEPA Sentinel-2 ViT-L/8 encoder",
    )

    log.info("Done. https://huggingface.co/%s", args.repo)


if __name__ == "__main__":
    main()
