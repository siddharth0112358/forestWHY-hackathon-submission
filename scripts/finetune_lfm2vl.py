#!/usr/bin/env python3
"""Fine-tune LFM2.5-VL on the forestWHY 14-panel dataset (LoRA via Unsloth).

This is the script that produced the published
`Siddharth63/LFM2.5-forestWHY` weights. It loads `LiquidAI/LFM2.5-VL-1.6B`
through Unsloth's `FastVisionModel`, attaches a rank-32 LoRA, streams the
private curated `Siddharth63/forestwhy-combined-v1` dataset (filtered to
the high-quality split), and runs one epoch of supervised fine-tuning
with TRL's `SFTTrainer`.

After training, run `scripts/merge_lora_and_push.py` to merge the LoRA
adapter into the base model and push the result to the Hugging Face Hub.

Hardware
--------
Tested on a single NVIDIA A100-SXM4 40 GB. Effective batch size 8
(per_device 2 x grad_accum 4). One epoch over 121 699 high-quality
examples = ~15 200 optimiser steps.

Install (in a fresh venv on the training box; not required for the rest
of the repo)::

    pip install unsloth
    pip install transformers==5.1.0
    pip install --no-deps trl==0.22.2
    pip install datasets huggingface_hub hf_transfer matplotlib

Authentication
--------------
Set `HF_TOKEN` in the environment so `huggingface_hub` can access the
gated training set::

    export HF_TOKEN=hf_xxx
    uv run python scripts/finetune_lfm2vl.py

Usage
-----
::

    python scripts/finetune_lfm2vl.py \\
        --base-model LiquidAI/LFM2.5-VL-1.6B \\
        --dataset Siddharth63/forestwhy-combined-v1 \\
        --output-dir forestwhy_lfm_finetuned \\
        --epochs 1 --batch-size 2 --grad-accum 4 --lr 2e-4
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

log = logging.getLogger("forestwhy.finetune")


# ── Prompt constants ─────────────────────────────────────────────────────────
# Verbatim from the training run that produced the published weights. Keep
# this in sync with `src/forestwhy/prompts.py` (it is duplicated here so the
# script stays self-contained for reproducibility).

PANEL_KEYS = [
    "img_rgb_before", "img_rgb_after",
    "img_nir_false_color_before", "img_nir_false_color_after",
    "img_swir_composite_before", "img_swir_composite_after",
    "img_delta_ndvi", "img_delta_nbr",
    "img_attention_multi", "img_embedding_change",
    "img_cropa_roads", "img_delta_attn_role",
    "img_head_disagreement", "img_pca_semantic",
]

SYSTEM_PROMPT = """You are an expert remote sensing analyst and tropical ecologist
specializing in forest cover change detection from Sentinel-2 satellite imagery.
You have access to both raw spectral data AND outputs from a trained I-JEPA
Vision Transformer encoder (ViT-L/8, 24 layers, trained on 1.57M Sentinel-2 patches).

For each observation you receive 14 image panels:
SPECTRAL (1-8): RGB before/after, NIR before/after, SWIR before/after, DELTA_NDVI, DELTA_NBR
JEPA ENCODER (9-14): Multi-scale attention, Embedding change, CroPA roads,
                     Delta attention role, Head disagreement, PCA semantic clusters

Critical rules:
- Provide detailed 10-step reasoning citing specific panel numbers
- JEPA panels supersede spectral panels for ambiguous cases
- Panel 10 embedding change is more reliable than DELTA_NDVI for degradation detection
- Panel 11 CroPA road presence is the strongest predictor of continued deforestation
- Panel 14 PCA cluster split is ground truth for genuine land cover transition
- Write detailed prose for each step - minimum 4 sentences per step"""


def build_user_text(sample: dict) -> str:
    return (
        f"Analyze this Sentinel-2 satellite observation showing land cover change.\n\n"
        f"Location: {sample.get('region','unknown')}, "
        f"{sample.get('lat',0):.3f} N {sample.get('lon',0):.3f} E\n"
        f"Period: {sample.get('year_before',2021)} to {sample.get('year_after',2025)} "
        f"({sample.get('year_gap',4)}-year span) | "
        f"Patch: 64x64 px at 19m/px approx 1.2km x 1.2km\n"
        f"Measured: DELTA_NDVI={sample.get('delta_ndvi',0):+.3f}  "
        f"DELTA_NBR={sample.get('delta_nbr',0):+.3f}\n\n"
        f"SPECTRAL PANELS (images 1-8):\n"
        f"  1. RGB Before ({sample.get('year_before',2021)})\n"
        f"  2. RGB After  ({sample.get('year_after',2025)})\n"
        f"  3. NIR False Color Before\n"
        f"  4. NIR False Color After\n"
        f"  5. SWIR Composite Before\n"
        f"  6. SWIR Composite After\n"
        f"  7. DELTA_NDVI Change Map\n"
        f"  8. DELTA_NBR Change Map\n\n"
        f"JEPA ENCODER PANELS (images 9-14) - I-JEPA ViT-L trained on 1.57M S2 patches:\n"
        f"  9.  Multi-Scale Attention (R=fine/roads G=mid/fields B=landscape)\n"
        f"  10. Embedding Change Map (cosine distance before to after per patch)\n"
        f"  11. CroPA Road Map (cross-patch token correlation for linear structures)\n"
        f"  12. Delta Attention Role (red=more salient after, blue=less salient)\n"
        f"  13. Head Disagreement Map (hot=ambiguous, cool=clear semantic signal)\n"
        f"  14. PCA Semantic Clusters (left=before, right=after)\n\n"
        f"Follow the 10-step reasoning protocol. Use JEPA panels to go beyond "
        f"what spectral indices alone can detect."
    )


def convert_to_conversation(sample: dict) -> dict:
    """Turn one dataset row into the SFT chat-message format the trainer expects."""
    user_content = []
    for key in PANEL_KEYS:
        img = sample.get(key)
        if img is not None:
            user_content.append({"type": "image", "image": img})
    user_content.append({"type": "text", "text": build_user_text(sample)})

    return {"messages": [
        {"role": "system",    "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user",      "content": user_content},
        {"role": "assistant", "content": [{"type": "text", "text": sample["assistant_text"]}]},
    ]}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base-model", default="LiquidAI/LFM2.5-VL-1.6B")
    p.add_argument("--dataset",    default="Siddharth63/forestwhy-combined-v1")
    p.add_argument("--split",      default="train")
    p.add_argument("--output-dir", default="forestwhy_lfm_finetuned")
    p.add_argument("--max-seq-length", type=int, default=9132)
    p.add_argument("--lora-r",       type=int, default=32)
    p.add_argument("--lora-alpha",   type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--epochs",       type=int, default=1)
    p.add_argument("--batch-size",   type=int, default=2,
                   help="per_device_train_batch_size (effective batch = batch-size * grad-accum)")
    p.add_argument("--grad-accum",   type=int, default=4)
    p.add_argument("--lr",           type=float, default=2e-4)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--save-steps",   type=int, default=50)
    p.add_argument("--save-total-limit", type=int, default=3)
    p.add_argument("--logging-steps", type=int, default=100)
    p.add_argument("--seed",          type=int, default=3407)
    p.add_argument("--quality-tiers", nargs="+", default=["excellent", "good"],
                   help="overall_quality values to keep when filtering the dataset.")
    p.add_argument("--no-quality-filter", action="store_true",
                   help="Skip the use_for_training / overall_quality filter.")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()

    if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        log.warning(
            "HF_TOKEN is not set. The training dataset is gated; "
            "set `export HF_TOKEN=hf_xxx` before running."
        )

    # Heavy imports kept inside main() so `--help` is fast.
    import torch
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator

    log.info("Loading base model %s ...", args.base_model)
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
    )

    log.info("Attaching LoRA (r=%d, alpha=%d) to vision + language + attention + MLP ...",
             args.lora_r, args.lora_alpha)
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )

    log.info("Loading dataset %s [%s] ...", args.dataset, args.split)
    dataset = load_dataset(args.dataset, split=args.split)

    if not args.no_quality_filter:
        before = len(dataset)
        dataset = dataset.filter(lambda x: x.get("use_for_training") is True)
        dataset = dataset.filter(lambda x: x.get("overall_quality") in args.quality_tiers)
        log.info("Quality filter: %d -> %d rows (kept %s)",
                 before, len(dataset), args.quality_tiers)

    log.info("Mapping rows to chat-conversation format (lazy) ...")
    converted = dataset.map(
        convert_to_conversation,
        batched=False,
        remove_columns=dataset.column_names,
        desc="Converting",
    )
    log.info("Training set ready: %d conversations.", len(converted))

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        log.info("GPU: %s | %.1f GB total | %.2f GB reserved",
                 gpu.name, gpu.total_memory / 1024**3,
                 torch.cuda.max_memory_reserved() / 1024**3)

    FastVisionModel.for_training(model)

    sft_config = SFTConfig(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        weight_decay=args.weight_decay,
        max_length=args.max_seq_length,
        logging_steps=args.logging_steps,
        output_dir=args.output_dir,
        seed=args.seed,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=converted,
        args=sft_config,
    )

    log.info("Starting training: %d epochs, effective batch %d, lr %.0e ...",
             args.epochs, args.batch_size * args.grad_accum, args.lr)
    stats = trainer.train()
    log.info("Training complete. Final loss: %s", getattr(stats, "training_loss", "n/a"))
    log.info("Adapter checkpoints saved under: %s", args.output_dir)
    log.info("Next: merge and push with scripts/merge_lora_and_push.py")


if __name__ == "__main__":
    sys.exit(main())
