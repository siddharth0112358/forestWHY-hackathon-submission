#!/usr/bin/env python3
"""Upload a forestwhy GGUF pair to a HuggingFace Hub repo.

Usage:
    uv run python scripts/push_gguf_to_hf.py \\
        --src ./forestwhy-gguf \\
        --repo Siddharth63/LFM2.5-forestWHY-GGUF
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

log = logging.getLogger("forestwhy.push_gguf")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s",
                        datefmt="%H:%M:%S")
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="Directory containing forestwhy-*.gguf and forestwhy-mmproj-*.gguf")
    p.add_argument("--repo", required=True, help="HF repo id, e.g. Siddharth63/LFM2.5-forestWHY-GGUF")
    p.add_argument("--private", action="store_true")
    args = p.parse_args()

    src = Path(args.src)
    if not src.is_dir():
        log.error("Not a directory: %s", src)
        sys.exit(1)
    files = sorted(src.glob("*.gguf"))
    if not files:
        log.error("No .gguf files in %s", src)
        sys.exit(1)

    from huggingface_hub import HfApi, create_repo
    api = HfApi()
    create_repo(args.repo, exist_ok=True, private=args.private)
    for f in files:
        log.info("Uploading %s ...", f.name)
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=f.name,
            repo_id=args.repo,
        )
    log.info("Done. https://huggingface.co/%s", args.repo)


if __name__ == "__main__":
    main()
