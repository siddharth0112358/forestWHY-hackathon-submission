"""Lifecycle helpers for an OpenAI-compatible llama-server fronting a GGUF.

Auto-downloads the LFM2.5-forestWHY GGUF (backbone + mmproj) from
HuggingFace on first use, launches `llama-server` as a subprocess, waits
for the OpenAI-compatible endpoint to come up, and tears it down on exit.

The wildfire-prevention example follows the same pattern with `start_llama_server`
/ `wait_for_server` / `stop_server`.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

import requests

log = logging.getLogger(__name__)


DEFAULT_GGUF_REPO = os.environ.get("GGUF_HF_REPO", "Siddharth63/LFM2.5-forestWHY-GGUF")
DEFAULT_GGUF_QUANT = os.environ.get("GGUF_QUANT", "Q8_0")
DEFAULT_MMPROJ_FILE = os.environ.get(
    "GGUF_MMPROJ_FILENAME", "LFM2.5-forestWHY.BF16-mmproj.gguf"
)


def gguf_filename(quant: str = DEFAULT_GGUF_QUANT) -> str:
    return f"LFM2.5-forestWHY.{quant}.gguf"


# ─────────────────────────────────────────────────────────────────────────────
# Download
# ─────────────────────────────────────────────────────────────────────────────

def ensure_gguf(
    repo: str = DEFAULT_GGUF_REPO,
    quant: str = DEFAULT_GGUF_QUANT,
    mmproj_filename: str = DEFAULT_MMPROJ_FILE,
) -> tuple[Path, Path]:
    """Download the GGUF backbone + mmproj if not already cached. Returns (model, mmproj)."""
    from huggingface_hub import hf_hub_download

    model_fn = gguf_filename(quant)
    log.info("Resolving GGUF %s/%s ...", repo, model_fn)
    model_path = Path(hf_hub_download(repo_id=repo, filename=model_fn))
    log.info("  -> %s", model_path)

    log.info("Resolving mmproj %s/%s ...", repo, mmproj_filename)
    mmproj_path = Path(hf_hub_download(repo_id=repo, filename=mmproj_filename))
    log.info("  -> %s", mmproj_path)

    return model_path, mmproj_path


# ─────────────────────────────────────────────────────────────────────────────
# Server lifecycle
# ─────────────────────────────────────────────────────────────────────────────

def llama_server_path() -> str:
    """Return absolute path to the llama-server binary, or raise."""
    binary = shutil.which("llama-server")
    if not binary:
        raise RuntimeError(
            "llama-server not found on PATH. Install with one of:\n"
            "  brew install llama.cpp        # macOS\n"
            "  apt install llama.cpp         # Debian/Ubuntu (if available)\n"
            "  or build from https://github.com/ggerganov/llama.cpp\n"
        )
    return binary


def start_llama_server(
    model_path: Path,
    mmproj_path: Path,
    port: int = 8080,
    ctx_size: int = 8192,
    n_gpu_layers: int = -1,
    extra_args: Optional[list[str]] = None,
) -> subprocess.Popen[bytes]:
    """Spawn llama-server hosting the given GGUF + vision projector.

    `n_gpu_layers=-1` offloads all layers to Metal/CUDA when available;
    on a pure-CPU build it's silently ignored.
    """
    binary = llama_server_path()
    cmd = [
        binary,
        "-m", str(model_path),
        "--mmproj", str(mmproj_path),
        "--port", str(port),
        "--ctx-size", str(ctx_size),
        "--n-gpu-layers", str(n_gpu_layers),
        "--jinja",
        "-fa", "on",
    ]
    if extra_args:
        cmd.extend(extra_args)
    log.info("Starting llama-server on :%d ...", port)
    log.debug("  %s", " ".join(cmd))
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def wait_for_server(port: int = 8080, timeout: float = 120.0, poll: float = 1.0) -> None:
    """Block until llama-server's /v1/models endpoint returns 200, or raise."""
    url = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = requests.get(url, timeout=2.0)
            if r.ok:
                return
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(poll)
    raise TimeoutError(
        f"llama-server did not become ready on :{port} within {timeout:.0f}s"
    )


def stop_server(proc: subprocess.Popen[bytes]) -> None:
    """Send SIGTERM, fall back to SIGKILL after 10 s."""
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
