"""Backend abstraction for the deforestation VLM.

`PredictFn` takes 14 PNG byte-strings (in PANEL_ORDER) plus a metadata dict
and returns a JSON dict conforming to `prompts.JSON_SCHEMA`. Three concrete
factories are provided:

    vllm_backend         — OpenAI-compatible HTTP (vLLM running LFM2.5-forestWHY)
    llamacpp_backend     — OpenAI-compatible HTTP (llama-server with GGUF)
    transformers_backend — local AutoModelForImageTextToText, no server

All backends produce identical JSON. Validation against the schema happens
inside the factory; on schema failure we re-raise as `BackendError` so the
watch loop can decide whether to skip the row or retry.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from typing import Any, Callable, Optional, Protocol, Sequence

import jsonschema

from .prompts import JSON_SCHEMA, SYSTEM_PROMPT, render_user_prompt

log = logging.getLogger(__name__)


class PredictFn(Protocol):
    def __call__(self, panel_pngs: Sequence[bytes], metadata: dict) -> dict: ...


class BackendError(RuntimeError):
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_messages(panel_pngs: Sequence[bytes], metadata: dict) -> list[dict]:
    user_text = render_user_prompt(metadata)
    content: list[dict] = [{"type": "text", "text": user_text}]
    for png in panel_pngs:
        b64 = base64.b64encode(png).decode("ascii")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def _validate(payload: dict) -> dict:
    try:
        jsonschema.validate(payload, JSON_SCHEMA)
    except jsonschema.ValidationError as exc:
        raise BackendError(f"VLM response did not match JSON_SCHEMA: {exc.message}") from exc
    return payload


def _parse_response(text: str) -> dict:
    text = text.strip()
    # Strip markdown fences if the model wrapped the JSON
    if text.startswith("```"):
        text = text.lstrip("`").lstrip("json").strip()
        if text.endswith("```"):
            text = text[: -3].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise BackendError(f"VLM response was not valid JSON: {exc.msg}\n--\n{text[:300]}") from exc


# ─────────────────────────────────────────────────────────────────────────────
# vLLM backend
# ─────────────────────────────────────────────────────────────────────────────

def vllm_backend(
    base_url: str = "http://localhost:8000/v1",
    model: str = "Siddharth63/LFM2.5-forestWHY",
    timeout: float = 180.0,
    max_tokens: int = 1024,
    temperature: float = 0.2,
) -> PredictFn:
    """Return a PredictFn that calls a running vLLM server."""
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=os.environ.get("VLLM_API_KEY", "EMPTY"), timeout=timeout)

    def predict(panel_pngs: Sequence[bytes], metadata: dict) -> dict:
        messages = _build_messages(panel_pngs, metadata)
        # Try strict json_schema first; fall back to json_object then plain.
        for response_format in (
            {"type": "json_schema", "json_schema": {"name": "forestwhy", "schema": JSON_SCHEMA, "strict": True}},
            {"type": "json_object"},
            None,
        ):
            try:
                kwargs: dict[str, Any] = dict(
                    model=model, messages=messages,
                    temperature=temperature, max_tokens=max_tokens,
                )
                if response_format is not None:
                    kwargs["response_format"] = response_format
                rsp = client.chat.completions.create(**kwargs)
                text = rsp.choices[0].message.content or ""
                return _validate(_parse_response(text))
            except Exception as exc:
                log.warning("vLLM call with response_format=%s failed: %s", response_format, exc)
                continue
        raise BackendError("vLLM backend exhausted all response_format fallbacks")

    return predict


# ─────────────────────────────────────────────────────────────────────────────
# llama.cpp backend
# ─────────────────────────────────────────────────────────────────────────────

def llamacpp_backend(
    base_url: str = "http://localhost:8080/v1",
    model: str = "forestwhy",
    timeout: float = 240.0,
    max_tokens: int = 1024,
    temperature: float = 0.2,
) -> PredictFn:
    """Return a PredictFn that calls a llama-server (mmproj + GGUF)."""
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=os.environ.get("LLAMA_API_KEY", "EMPTY"), timeout=timeout)

    def predict(panel_pngs: Sequence[bytes], metadata: dict) -> dict:
        messages = _build_messages(panel_pngs, metadata)
        rsp = client.chat.completions.create(
            model=model, messages=messages,
            temperature=temperature, max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        text = rsp.choices[0].message.content or ""
        return _validate(_parse_response(text))

    return predict


# ─────────────────────────────────────────────────────────────────────────────
# Transformers backend (no server)
# ─────────────────────────────────────────────────────────────────────────────

def transformers_backend(
    model_id: str = "Siddharth63/LFM2.5-forestWHY",
    device: str = "auto",
    max_new_tokens: int = 1024,
) -> PredictFn:
    """Return a PredictFn that runs LFM2.5-forestWHY locally via transformers.

    Slow on CPU. Useful for laptop demos and judges without a GPU server.
    """
    import io as _io

    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch, "backends") and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    log.info("Loading %s on %s ...", model_id, device)
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, device_map=device,
        dtype=torch.bfloat16 if device != "cpu" else torch.float32,
    )

    def predict(panel_pngs: Sequence[bytes], metadata: dict) -> dict:
        images = [Image.open(_io.BytesIO(p)).convert("RGB") for p in panel_pngs]
        user_text = render_user_prompt(metadata)
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                *[{"type": "image", "image": img} for img in images],
                {"type": "text", "text": user_text},
            ]},
        ]
        inputs = processor.apply_chat_template(
            conversation, add_generation_prompt=True,
            return_tensors="pt", return_dict=True, tokenize=True,
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.2, do_sample=False)
        text = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return _validate(_parse_response(text))

    return predict


# ─────────────────────────────────────────────────────────────────────────────
# Stub backend — used by predict.py --smoke-test
# ─────────────────────────────────────────────────────────────────────────────

def stub_backend() -> PredictFn:
    """Return a deterministic PredictFn that does no I/O."""
    def predict(panel_pngs: Sequence[bytes], metadata: dict) -> dict:
        return {
            "change_class": "deforestation",
            "severity": "medium",
            "area_pct": 12.5,
            "driver_hypothesis": "agricultural_clearing",
            "confidence": 0.42,
            "reasoning": (
                f"Stub prediction for tile lon={metadata.get('lon', 0):.4f} "
                f"lat={metadata.get('lat', 0):.4f}. The watch loop and panel "
                f"pipeline are working end-to-end; no real model was queried."
            ),
            "cloud_cover_note": "n/a (smoke test)",
        }
    return predict


# ─────────────────────────────────────────────────────────────────────────────
# Factory dispatcher (used by both predict.py and evaluate.py)
# ─────────────────────────────────────────────────────────────────────────────

BackendFactory = Callable[..., PredictFn]


def make_backend(
    name: str,
    *,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> PredictFn:
    name = name.lower()
    if name == "vllm":
        return vllm_backend(
            base_url=base_url or os.environ.get("VLM_BASE_URL", "http://localhost:8000/v1"),
            model=model or os.environ.get("VLM_MODEL", "Siddharth63/LFM2.5-forestWHY"),
        )
    if name == "llamacpp":
        return llamacpp_backend(
            base_url=base_url or os.environ.get("LLAMA_BASE_URL", "http://localhost:8080/v1"),
            model=model or "forestwhy",
        )
    if name == "transformers":
        return transformers_backend(
            model_id=model or os.environ.get("VLM_MODEL", "Siddharth63/LFM2.5-forestWHY"),
        )
    if name == "stub":
        return stub_backend()
    raise ValueError(f"Unknown backend: {name}")


def model_name(backend: str, model: Optional[str]) -> str:
    if backend == "stub":
        return "stub"
    return model or os.environ.get("VLM_MODEL", "Siddharth63/LFM2.5-forestWHY")
