# forestWHY — On-orbit Deforestation Detection with LFM2.5-VL

> Liquid AI hackathon **LFM track** submission. Built around SimSat
> Sentinel-2 imagery, a Sentinel-2 I-JEPA encoder, and a fine-tuned
> [`Siddharth63/LFM2.5-forestWHY`](https://huggingface.co/Siddharth63/LFM2.5-forestWHY)
> trained on
> [`Siddharth63/forestwhy-training-v1`](https://huggingface.co/datasets/Siddharth63/forestwhy-training-v1).

forestWHY runs deforestation analysis aboard a simulated satellite. For each
5 km tile it sees, the system fetches a temporal pair of Sentinel-2 13-band
acquisitions (≈ 1 year apart), runs both through an **I-JEPA ViT-L/8** encoder
to produce six differential attention/embedding panels, pairs those with
eight spectral panels (RGB, NIR-FC, SWIR, ΔNDVI, ΔNBR), and asks
**LFM2.5-forestWHY** — a 2 B fine-tune of LFM2-VL — to classify the change.
Only the resulting JSON is downlinked. No raw imagery leaves orbit.

## Documentation

- [**QUICKSTART.md**](QUICKSTART.md) — step-by-step instructions to install
  and run, including a fully local end-to-end smoke test that needs **no
  GPU and no SimSat** (just `uv sync` + `brew install llama.cpp`).
- [**ARCHITECTURE.md**](ARCHITECTURE.md) — design rationale, why the
  14-panel approach beats spectral-only and generic VLMs, what is
  genuinely innovative, and the evaluation methodology.

## Hosted artefacts

| Artefact                                | Where                                                                                                |
|-----------------------------------------|------------------------------------------------------------------------------------------------------|
| Source                                  | https://github.com/siddharth0112358/forestWHY-hackathon-submission                                   |
| Sentinel-2 I-JEPA ViT-L/8 encoder (1.21 GB) | https://huggingface.co/Siddharth63/forestWHY-JEPA-vitl                                           |
| Fine-tuned VLM (full precision)         | https://huggingface.co/Siddharth63/LFM2.5-forestWHY                                                   |
| Fine-tuned VLM (GGUF Q4_K_M / Q5_K_M / Q8_0) | https://huggingface.co/Siddharth63/LFM2.5-forestWHY-GGUF                                         |
| Training dataset (≈ 150 K Sentinel-2 pairs) | https://huggingface.co/datasets/Siddharth63/forestwhy-training-v1                                |

## Repository layout

```
.
├── README.md                          # this file
├── QUICKSTART.md                      # how to run, step-by-step
├── ARCHITECTURE.md                    # design rationale + innovation
├── pyproject.toml                     # uv-managed Python deps
├── uv.lock                            # locked dependency graph
├── .env.example                       # all environment variables documented
├── .gitignore
│
├── app/
│   ├── app.py                         # Streamlit dashboard: pydeck map + 14-panel viewer
│   └── eval_compare.py                # base vs fine-tuned side-by-side comparator
│
├── assets/                            # diagrams, screenshots, demo GIFs
│
├── configs/
│   └── forestwhy_finetune_modal.yaml  # leap-finetune Modal H100 config
│
├── scripts/
│   ├── predict.py                     # main watch loop (--smoke-test or live SimSat)
│   ├── backfill.py                    # historical (location, before, after) iteration
│   ├── download_weights.py            # pre-warm HF cache (JEPA + VLM)
│   ├── upload_jepa.py                 # one-off: push the encoder .pt to HF Hub
│   ├── generate_samples.py            # eval-set labeller (uses Anthropic for ground truth)
│   ├── check_samples.py               # validate generated samples conform to schema
│   ├── evaluate.py                    # base vs fine-tuned per-field accuracy
│   ├── prepare_dataset.py             # HF dataset → leap-finetune JSONL
│   ├── quantize.py                    # safetensors → GGUF Q4_K_M / Q8_0 + mmproj
│   └── push_gguf_to_hf.py             # upload GGUF pair to HF Hub
│
└── src/forestwhy/
    ├── __init__.py
    ├── db.py                          # SQLite schema + insert/fetch helpers
    ├── live.py                        # SimSat HTTP client (position, 13-band, walkback)
    ├── locations.py                   # 18 watched deforestation hotspots
    ├── regions.py                     # tile geometry (haversine, find_tile)
    ├── spectral.py                    # 8 spectral panel generators
    ├── jepa.py                        # vendored S2Encoder + 6 JEPA panels + HF auto-download
    ├── llama_server.py                # subprocess lifecycle for llama-server + GGUF auto-pull
    ├── pipeline.py                    # tile → 14 panels → VLM → SQLite glue
    ├── prompts.py                     # SYSTEM_PROMPT (training-time verbatim) + dashboard schema
    ├── annotator.py                   # (planned) Anthropic ground-truth labeller helpers
    └── evaluator.py                   # PredictFn protocol + vllm/llamacpp/transformers/stub backends
```

## What's in each top-level directory

- **`app/`** — Streamlit. The map + 14-panel viewer is the demo surface;
  the comparator app is for the fine-tuning evaluation.
- **`scripts/`** — every CLI entry point. `predict.py` is the one judges
  invoke; the rest are operational tooling (eval, dataset prep,
  quantisation, HF upload).
- **`src/forestwhy/`** — the library. Importable as `forestwhy.*`,
  installed as a package by `uv sync`. The two heaviest files are
  `jepa.py` (vendored ViT-L/8 + 6 differential panels) and `pipeline.py`
  (the per-tile glue).
- **`configs/`** — fine-tuning specs only. No runtime config (everything
  else lives in `.env`).

## Want to run it?

Pick the fidelity that matches your environment:

| You have                      | Mode                        | First command                                        |
|-------------------------------|-----------------------------|------------------------------------------------------|
| Just a laptop                 | Stub smoke test             | `uv run python scripts/predict.py --smoke-test`      |
| Laptop + `brew install llama.cpp` | **End-to-end with the real fine-tune** | `uv run python scripts/predict.py --smoke-test --with-gguf` |
| Laptop + Docker + a GPU host  | Full live run               | follow [QUICKSTART.md §3](QUICKSTART.md#mode-3--live-run-against-simsat--a-hosted-vllm) |

See [QUICKSTART.md](QUICKSTART.md) for full instructions, troubleshooting,
and the smoke-test checklist.

## License & credits

MIT. Built for the Liquid AI hackathon LFM track. Repository structure
inspired by the upstream
[`Liquid4All/cookbook/examples/wildfire-prevention`](https://github.com/Liquid4All/cookbook/tree/main/examples/wildfire-prevention).
Sentinel-2 imagery © Copernicus / ESA. SimSat by
[DPhi Space](https://github.com/DPhi-Space/SimSat).
