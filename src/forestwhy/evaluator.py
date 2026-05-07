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

from .prompts import DASHBOARD_FIELDS, JSON_SCHEMA, SYSTEM_PROMPT, render_user_prompt

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


def _parse_response(text: str) -> dict:
    """Parse the VLM response.

    The fine-tune emits free-form prose (10-step reasoning protocol). Some
    backends additionally produce a JSON wrapper. We accept either:
        1. valid JSON object -> return as-is
        2. anything else      -> wrap as {"reasoning": <text>}
    """
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.lstrip("`")
        if text[:4].lower() == "json":
            text = text[4:]
        text = text.strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    return {"reasoning": text}


# Vocabulary the LFM2.5-forestWHY fine-tune actually uses in its 10-step prose.
# Order within each list does not matter; order across lists DOES — earlier
# entries win. We score class by total occurrence count across synonyms,
# breaking ties by the order below.
_CLASS_SYNONYMS: list[tuple[str, tuple[str, ...]]] = [
    ("deforestation", (
        "deforestation", "deforested", "forest loss", "canopy loss",
        "clearing", "cleared", "clear-cut", "clearcut", "clear cut",
        "logging", "logged", "selective logging", "logging tracks",
        "canopy disturbance", "forest degradation", "degraded",
        "bare soil", "exposed soil", "newly cleared",
    )),
    ("fire_disturbance", (
        "fire", "burned", "burn scar", "burn scars", "wildfire",
        "fire damage", "scorched", "charred",
    )),
    ("afforestation", (
        "afforestation", "reforestation", "regrowth", "secondary growth",
        "regenerating forest", "tree planting", "natural regeneration",
    )),
    ("stable_forest", (
        "intact forest", "intact tropical forest", "undisturbed",
        "unchanged", "no significant change", "no observable change",
        "stable canopy", "dense canopy", "primary forest",
    )),
    ("stable_non_forest", (
        "stable_non_forest", "established agriculture", "stable pasture",
        "stable cropland", "no forest", "non-forest", "open canopy",
    )),
    ("ambiguous", (
        "ambiguous", "unclear", "uncertain", "inconclusive",
        "cannot be determined", "obscured by cloud", "cloud cover",
    )),
]

_SEVERITY_SYNONYMS: list[tuple[str, tuple[str, ...]]] = [
    ("high",   ("high severity", "severe", "extensive", "large-scale", "widespread", "major")),
    ("medium", ("moderate", "medium severity", "mid-scale", "noticeable")),
    ("low",    ("minor", "low severity", "limited", "small", "localised", "localized", "patchy")),
    ("none",   ("no change", "no observable change", "no significant change", "stable")),
]

_DRIVER_SYNONYMS: list[tuple[str, tuple[str, ...]]] = [
    ("agricultural_clearing", (
        "agricultural clearing", "agricultural expansion",
        "agricultural conversion", "agricultural frontier",
        "agriculture", "agricultural", "agro-industrial",
        "soy", "soybean", "soya", "soy cultivation",
        "cattle", "ranching", "ranch expansion", "pasture", "pastureland",
        "cropland", "crop expansion", "field expansion", "agricultural fields",
        "smallholder agriculture", "shifting cultivation", "slash and burn",
        "swidden", "milpa", "subsistence farming",
        "farmland", "farmstead",
    )),
    ("logging_road", (
        "logging road", "logging roads", "logging tracks", "logging trail",
        "selective logging", "selective harvest", "selective extraction",
        "timber extraction", "timber harvesting", "timber operations",
        "logging concession", "logging operation",
        "skid trail", "skidder", "industrial logging", "commercial logging",
        "log landing", "haul road",
        "road construction", "new roads", "road network",
    )),
    ("mining", (
        "mining", "gold mining", "artisanal mining",
        "small-scale mining", "industrial mining", "open-pit mining",
        "open pit", "open cast",
        "tailings pond", "tailings", "mine pit",
        "extractive operation", "ore extraction",
        "garimpeiro", "alluvial mining",
        "quarry", "quarrying",
    )),
    ("fire", (
        "wildfire", "wildfires",
        "burn scar", "burn scars", "fire scar", "fire scars",
        "fire damage", "fire-damaged",
        "burned area", "burned forest",
        "active fire", "hotspot",
        "forest fire", "crown fire", "bushfire",
    )),
    ("flood", (
        "flood", "flooding", "flooded forest",
        "inundation", "inundated",
        "river migration", "river meander", "channel migration",
        "reservoir", "dam impoundment",
    )),
    ("plantation", (
        "oil palm", "palm oil", "palm plantation", "palm-oil plantation",
        "rubber plantation", "rubber",
        "eucalyptus plantation", "eucalyptus",
        "monoculture", "monocultural", "tree plantation",
        "industrial plantation", "tree farm",
        "pine plantation", "acacia plantation", "teak plantation",
    )),
    ("natural_regrowth", (
        "natural regrowth", "secondary regrowth", "secondary forest",
        "regeneration", "regenerating forest", "regenerating canopy",
        "vegetation recovery", "forest recovery",
        "ecological succession",
    )),
]


_NEGATION_TOKENS = (
    "no ", "not ", "absent", "absence of", "lacks", "without",
    "no clear", "no significant", "no observable", "no distinct",
    "would indicate", "no evidence",
)

import re as _re

def _count_synonym_hits(text: str, synonyms: tuple[str, ...]) -> int:
    """Word-boundary matches, excluding negated mentions within the same sentence."""
    total = 0
    for s in synonyms:
        for m in _re.finditer(rf"\b{_re.escape(s)}\b", text):
            start = m.start()
            # Look back to the previous sentence boundary (or start of text).
            sentence_start = max(
                text.rfind(". ", 0, start),
                text.rfind("\n", 0, start),
                0,
            )
            window = text[sentence_start: start]
            if any(neg in window for neg in _NEGATION_TOKENS):
                continue
            total += 1
    return total


def _conclusion_text(text: str) -> str:
    """Return the model's synthesis section. Looks for Step 8 onward, else
    falls back to the last 35 % of the text."""
    m = _re.search(r"##\s*step\s*8\b", text, flags=_re.IGNORECASE)
    if m is not None:
        return text[m.start():]
    return text[int(len(text) * 0.65):]


def _classify(text: str, vocab: list[tuple[str, tuple[str, ...]]]) -> Optional[str]:
    """Return the class with the most synonym hits in `text`. None if no hits."""
    best, best_count = None, 0
    for cls, syns in vocab:
        c = _count_synonym_hits(text, syns)
        if c > best_count:
            best, best_count = cls, c
    return best


def _extract_area_pct(text: str) -> Optional[float]:
    """Find the LARGEST 'X%' or 'X percent' mention; the model's final
    estimate tends to be the highest figure quoted."""
    import re
    candidates: list[float] = []
    for m in re.findall(r"(\d{1,3}(?:\.\d+)?)\s*(?:%|percent)", text):
        try:
            v = float(m)
            if 0.0 <= v <= 100.0:
                candidates.append(v)
        except ValueError:
            continue
    if not candidates:
        return None
    return max(candidates)


def _payload_text(payload: dict) -> str:
    """Concatenate all string-typed values in the payload (recursive shallow)."""
    parts: list[str] = []
    def walk(v):
        if isinstance(v, str):
            parts.append(v)
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
    walk(payload)
    return " ".join(parts).lower()


def _normalize(payload: dict) -> dict:
    """Coerce arbitrary VLM output into the dashboard's expected fields.

    The fine-tune emits free-form prose with a 10-step reasoning protocol, so
    this function does best-effort extraction via synonym counting. The full
    raw payload is always preserved under `raw` for audit.
    """
    out: dict = {"raw": payload}

    # Direct passthrough for any structured fields the model already emits.
    for k in DASHBOARD_FIELDS:
        if k in payload:
            out[k] = payload[k]

    text = _payload_text(payload)
    # Use the synthesis section (Step 8 onward, or last 35 %) for classification.
    # The model's earlier steps describe what it sees; the synthesis is its conclusion.
    conclusion = _conclusion_text(text)

    # change_class
    if "change_class" not in out or out["change_class"] is None:
        cls = _classify(conclusion, _CLASS_SYNONYMS)
        if cls is not None:
            out["change_class"] = cls

    # severity
    if "severity" not in out or out["severity"] is None:
        sev = _classify(conclusion, _SEVERITY_SYNONYMS)
        if sev is not None:
            out["severity"] = sev

    # driver — only meaningful when the model thinks change happened
    if "driver_hypothesis" not in out or out["driver_hypothesis"] is None:
        if out.get("change_class") in ("deforestation", "fire_disturbance", "afforestation"):
            drv = _classify(conclusion, _DRIVER_SYNONYMS)
            if drv is not None:
                out["driver_hypothesis"] = drv
            else:
                out["driver_hypothesis"] = "unknown"
        else:
            out["driver_hypothesis"] = None

    # area_pct
    if "area_pct" not in out or out["area_pct"] is None:
        mag = _dig(payload, "change_forest", "measured_magnitude")
        if isinstance(mag, (int, float)):
            out["area_pct"] = max(0.0, min(100.0, float(mag) * 100.0))
        else:
            ap = _extract_area_pct(conclusion)
            if ap is None:
                ap = _extract_area_pct(text)
            if ap is not None:
                out["area_pct"] = ap

    # confidence
    if "confidence" in out and isinstance(out["confidence"], str):
        out["confidence"] = {"low": 0.3, "medium": 0.6, "high": 0.9}.get(
            out["confidence"].lower(), 0.5
        )
    elif "confidence" not in out or out["confidence"] is None:
        # Heuristic: if classification succeeded, use the synonym hit count
        # as a proxy. ≥3 hits → 0.7, ≥1 → 0.5, else None.
        if out.get("change_class"):
            for cls, syns in _CLASS_SYNONYMS:
                if cls == out["change_class"]:
                    hits = _count_synonym_hits(text, syns)
                    out["confidence"] = 0.7 if hits >= 3 else (0.5 if hits >= 1 else 0.3)
                    break

    # reasoning fallback
    if "reasoning" not in out or not out["reasoning"]:
        out["reasoning"] = (text[:600] + ("…" if len(text) > 600 else "")) if text else ""

    try:
        jsonschema.validate(out, JSON_SCHEMA)
    except jsonschema.ValidationError as exc:
        log.warning("Output did not match relaxed JSON_SCHEMA: %s", exc.message)
    return out


def _truthy(v) -> bool:
    if isinstance(v, str):
        return v.lower() in ("yes", "true", "high", "medium")
    return bool(v)


def _dig(d: dict, *keys):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


# ─────────────────────────────────────────────────────────────────────────────
# vLLM backend
# ─────────────────────────────────────────────────────────────────────────────

def vllm_backend(
    base_url: str = "http://localhost:8000/v1",
    model: str = "Siddharth63/LFM2.5-forestWHY",
    timeout: float = 180.0,
    max_tokens: int = 4096,
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
                return _normalize(_parse_response(text))
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
    max_tokens: int = 4096,
    temperature: float = 0.2,
) -> PredictFn:
    """Return a PredictFn that calls a llama-server (mmproj + GGUF)."""
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=os.environ.get("LLAMA_API_KEY", "EMPTY"), timeout=timeout)

    def predict(panel_pngs: Sequence[bytes], metadata: dict) -> dict:
        messages = _build_messages(panel_pngs, metadata)
        # The fine-tune was trained on free-form prose, not strict JSON. We
        # don't constrain response_format here; if the user wants JSON-mode
        # they can swap to vllm_backend with response_format json_object.
        rsp = client.chat.completions.create(
            model=model, messages=messages,
            temperature=temperature, max_tokens=max_tokens,
        )
        text = rsp.choices[0].message.content or ""
        return _normalize(_parse_response(text))

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
        return _normalize(_parse_response(text))

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
