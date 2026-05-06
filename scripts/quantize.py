#!/usr/bin/env python3
"""Convert the fine-tuned LFM2.5-forestWHY checkpoint to GGUF for llama.cpp.

Produces a Q8_0 quantised backbone + F16 multimodal projector pair, mirroring
wildfire-prevention's quantisation. Requires llama.cpp's convert tooling on
the path (`llama.cpp/convert_hf_to_gguf.py`).

Usage:
    uv run python scripts/quantize.py \\
        --src $HF_HOME/hub/models--Siddharth63--LFM2.5-forestWHY \\
        --out ./forestwhy-gguf \\
        --llama-cpp ../llama.cpp
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

log = logging.getLogger("forestwhy.quantize")


def _run(cmd: list[str]) -> None:
    log.info("$ %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s",
                        datefmt="%H:%M:%S")
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="Local HF snapshot directory of LFM2.5-forestWHY.")
    p.add_argument("--out", default="./forestwhy-gguf", help="Output directory.")
    p.add_argument("--llama-cpp", required=True, help="Path to a built llama.cpp checkout.")
    p.add_argument("--quant", default="Q8_0", help="Quantisation type for the backbone.")
    args = p.parse_args()

    src = Path(args.src).resolve()
    out = Path(args.out).resolve()
    llama_cpp = Path(args.llama_cpp).resolve()
    out.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        log.error("Source not found: %s", src)
        sys.exit(1)

    convert_hf = llama_cpp / "convert_hf_to_gguf.py"
    quantize_bin = llama_cpp / "build" / "bin" / "llama-quantize"
    mmproj_bin = llama_cpp / "build" / "bin" / "llama-mtmd-cli"  # used to verify only

    if not convert_hf.exists():
        log.error("convert_hf_to_gguf.py not found at %s. Clone llama.cpp and run `cmake -B build && cmake --build build -j`.", convert_hf)
        sys.exit(1)

    base_gguf = out / "forestwhy-base-f16.gguf"
    quant_gguf = out / f"forestwhy-{args.quant}.gguf"
    mmproj_gguf = out / "forestwhy-mmproj-f16.gguf"

    # 1) Backbone: HF -> F16 GGUF -> quantised GGUF
    _run([
        sys.executable, str(convert_hf),
        str(src), "--outtype", "f16", "--outfile", str(base_gguf),
    ])
    _run([str(quantize_bin), str(base_gguf), str(quant_gguf), args.quant])

    # 2) MMProj: HF -> F16 GGUF (vision tower only)
    _run([
        sys.executable, str(convert_hf),
        str(src), "--mmproj", "--outtype", "f16", "--outfile", str(mmproj_gguf),
    ])

    log.info("GGUF artefacts:")
    log.info("  backbone (quantised): %s", quant_gguf)
    log.info("  mmproj (vision):      %s", mmproj_gguf)
    log.info("Run with: %s -m %s --mmproj %s --port 8080",
             mmproj_bin, quant_gguf, mmproj_gguf)


if __name__ == "__main__":
    main()
