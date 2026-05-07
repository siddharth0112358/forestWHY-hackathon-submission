#!/usr/bin/env python3
"""Merge a forestWHY LoRA checkpoint into the LFM2.5-VL base and push to HF.

Run this after `scripts/finetune_lfm2vl.py` completes. It loads
`LiquidAI/LFM2.5-VL-1.6B`, applies the saved PEFT adapter, calls
`merge_and_unload()`, saves the merged weights locally, and (optionally)
pushes them to a Hugging Face Hub repository.

Authentication
--------------
Set `HF_TOKEN` in the environment so the push has write permission::

    export HF_TOKEN=hf_xxx

Usage
-----
::

    python scripts/merge_lora_and_push.py \\
        --base-model LiquidAI/LFM2.5-VL-1.6B \\
        --adapter   forestwhy_lfm_finetuned/checkpoint-15213 \\
        --save-path ./merged_model \\
        --hub-repo  Siddharth63/LFM2.5-forestWHY
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

log = logging.getLogger("forestwhy.merge_push")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base-model", default="LiquidAI/LFM2.5-VL-1.6B")
    p.add_argument("--adapter", required=True,
                   help="Path to a PEFT adapter checkpoint, e.g. forestwhy_lfm_finetuned/checkpoint-15213")
    p.add_argument("--save-path", default="./merged_model",
                   help="Local directory to write the merged HF snapshot.")
    p.add_argument("--hub-repo", default=None,
                   help="If set, push the merged model + processor to this HF repo.")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--private", action="store_true",
                   help="Create the hub repo as private (default: public).")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()

    adapter = Path(args.adapter)
    if not adapter.exists():
        log.error("Adapter path not found: %s", adapter)
        sys.exit(1)

    if args.hub_repo and not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
        log.warning("HF_TOKEN is not set; the push step will fail. Set `export HF_TOKEN=hf_xxx`.")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dtype = {"float16": torch.float16,
             "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.dtype]

    log.info("Loading base %s in %s ...", args.base_model, args.dtype)
    base = AutoModelForImageTextToText.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)

    log.info("Applying adapter from %s ...", adapter)
    model = PeftModel.from_pretrained(base, str(adapter), torch_dtype=dtype)

    log.info("Merging adapter into base weights ...")
    model = model.merge_and_unload()
    model.eval()

    save_path = Path(args.save_path).resolve()
    save_path.mkdir(parents=True, exist_ok=True)
    log.info("Saving merged snapshot to %s ...", save_path)
    model.save_pretrained(str(save_path))
    processor.save_pretrained(str(save_path))

    if args.hub_repo:
        from huggingface_hub import create_repo

        log.info("Ensuring hub repo %s exists (private=%s) ...", args.hub_repo, args.private)
        create_repo(args.hub_repo, exist_ok=True, private=args.private)

        log.info("Pushing model to %s ...", args.hub_repo)
        model.push_to_hub(args.hub_repo)
        log.info("Pushing processor to %s ...", args.hub_repo)
        processor.push_to_hub(args.hub_repo)
        log.info("Done. https://huggingface.co/%s", args.hub_repo)
    else:
        log.info("Done. (No --hub-repo given, skipped push.)")


if __name__ == "__main__":
    sys.exit(main())
