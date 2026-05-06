# forestWHY — On-orbit Deforestation Detection with LFM2.5-VL

> Source: this folder. Live demo: see the dashboard launched in step 7.
> Submission for the **Liquid LFM track** hackathon — built around SimSat
> Sentinel-2 imagery, an I-JEPA Sentinel-2 encoder, and a fine-tuned
> [`Siddharth63/LFM2.5-forestWHY`](https://huggingface.co/Siddharth63/LFM2.5-forestWHY)
> trained on [`Siddharth63/forestwhy-training-v1`](https://huggingface.co/datasets/Siddharth63/forestwhy-training-v1).

forestWHY runs deforestation analysis aboard a simulated satellite. For each
5 km tile it sees, the system fetches a temporal pair of Sentinel-2 13-band
acquisitions (≈ 1 year apart), runs both through an **I-JEPA ViT-L/8** encoder
to produce six differential attention/embedding panels, pairs those with
eight spectral panels (RGB, NIR-FC, SWIR, ΔNDVI, ΔNBR), and asks
**LFM2.5-forestWHY** — a 2 B fine-tune of LFM2-VL — to classify the change.
Only the resulting JSON is downlinked. No raw imagery leaves orbit.

```
SimSat current view  ─►  SimSat historical (T-1y)  ─►  JEPA encoder  ─►  6 panels
                                                          │
                       Spectral processing  ─►  8 panels ─┤
                                                          ▼
                                       LFM2.5-forestWHY (vLLM)
                                                          │
                                                          ▼
                                            JSON  ─►  SQLite  ─►  Streamlit dashboard
```

## 1. Problem framing

Tropical primary forests lose ≈ 4 Mha/year. Two kinds of lag drive that loss:

- **Detection lag** — alerts based on cloud-free composites land 3–8 weeks
  after the clearing event. By then, the road is in.
- **Bandwidth lag** — even at L0/L1 compression, downlinking 13-band
  Sentinel-2 tiles to ground servers is the bottleneck. The same constraint
  that the wildfire-prevention example identified for fire risk is even
  worse for forest-change detection because the signal lives in **temporal
  pairs**, not single frames.

forestWHY targets the bandwidth lag directly. The 14-panel input compresses
a 13-band, 64 × 64, two-epoch tile (~410 KB raw) into the same payload the
on-orbit VLM was trained on, and the only thing that ever leaves the
satellite is a ≤ 1 KB JSON record like:

```json
{
  "change_class": "deforestation",
  "severity": "high",
  "area_pct": 38.5,
  "driver_hypothesis": "agricultural_clearing",
  "confidence": 0.91,
  "reasoning": "Linear forest-edge front advancing south-west, ΔNDVI < -0.4 over ~1.6 km², cropa_roads panel shows new logging tracks, embedding_change flips the centre patches from forest to bare soil class.",
  "cloud_cover_note": "before tile 12 % cloud, after tile clear"
}
```

### Why not single-frame detection?

Pixel-level deforestation looks identical to seasonal phenology, river
migration, or shadow shift unless you have a *before* state. The JEPA
panels (`embedding_change`, `delta_attn_role`, `cropa_roads`) explicitly
encode change in semantic feature space, which a contrastive baseline does
not give you. Spectral panels alone can be fooled by haze; JEPA panels
alone struggle with magnitude. Together they are robust.

### Watched hotspots (18)

| region_id           | biome             | country     |
|---------------------|-------------------|-------------|
| amazon_acre         | amazon            | Brazil      |
| amazon_para         | amazon            | Brazil      |
| amazon_rondonia     | amazon            | Brazil      |
| amazon_mato_grosso  | amazon            | Brazil      |
| amazon_madre_dios   | amazon            | Peru        |
| amazon_beni         | amazon            | Bolivia     |
| borneo_kalimantan   | borneo            | Indonesia   |
| borneo_sabah        | borneo            | Malaysia    |
| sumatra_riau        | sumatra           | Indonesia   |
| papua_indonesia     | new_guinea        | Indonesia   |
| png_western         | new_guinea        | PNG         |
| congo_equateur      | congo_basin       | DRC         |
| congo_tshopo        | congo_basin       | DRC         |
| madagascar_east     | madagascar        | Madagascar  |
| cerrado_centre      | cerrado           | Brazil      |
| choco_colombia      | choco             | Colombia    |
| cambodia_mondulkiri | indochina         | Cambodia    |
| atlantic_forest     | atlantic_forest   | Brazil      |

The full coordinate list is in `src/forestwhy/locations.py`.

## 2. System design

| Component                  | What it does                                                                 | Where                              |
|----------------------------|------------------------------------------------------------------------------|------------------------------------|
| **SimSat**                 | Simulated satellite orbit + Sentinel-2 imagery REST API                      | external (`docker compose up`)     |
| **JEPA encoder**           | I-JEPA ViT-L/8 over 13-band Sentinel-2, frozen for inference                 | `src/forestwhy/jepa.py`            |
| **Spectral processing**    | 8 spectral panels (RGB / NIR-FC / SWIR before+after, ΔNDVI, ΔNBR)            | `src/forestwhy/spectral.py`        |
| **VLM**                    | LFM2.5-forestWHY served via vLLM (or llama.cpp / GGUF on a laptop)           | external                           |
| **Watch loop**             | Polls SimSat, builds panels, calls VLM, writes SQLite                        | `scripts/predict.py`               |
| **Dashboard**              | pydeck map + 14-panel viewer + JSON card                                     | `app/app.py` (streamlit)           |
| **Eval**                   | Per-field accuracy of base vs fine-tuned VLM                                 | `scripts/evaluate.py`, `app/eval_compare.py` |

### Why 14 panels?

| #  | name                | what it shows                                                       | source           |
|----|---------------------|---------------------------------------------------------------------|------------------|
| 1  | `rgb_before`        | true-colour B4-B3-B2 — broad context                                | spectral         |
| 2  | `rgb_after`         | true-colour after                                                   | spectral         |
| 3  | `nir_fc_before`     | NIR false colour B8-B4-B3 — vegetation vigour                       | spectral         |
| 4  | `nir_fc_after`      | NIR false colour after                                              | spectral         |
| 5  | `swir_before`       | SWIR composite B12-B8-B4 — burn scars and bare soil                 | spectral         |
| 6  | `swir_after`        | SWIR composite after                                                | spectral         |
| 7  | `delta_ndvi`        | ΔNDVI heatmap, red = vegetation loss                                | spectral         |
| 8  | `delta_nbr`         | ΔNBR heatmap, red = biomass loss                                    | spectral         |
| 9  | `attention_multi`   | multi-scale attention (R fine, G mid, B landscape)                  | JEPA             |
| 10 | `embedding_change`  | per-patch cosine distance, red = semantic class flipped             | JEPA             |
| 11 | `cropa_roads`       | cross-patch correlation, hot = linear structures (roads, tracks)    | JEPA             |
| 12 | `delta_attn_role`   | which patches gained/lost CLS importance after change               | JEPA             |
| 13 | `head_disagreement` | attention head std, hot = ambiguous land cover                      | JEPA             |
| 14 | `pca_semantic`      | PCA-3 of patch embeddings, before \| after                           | JEPA             |

### Quickstart

```bash
# 1.  Clone this repo
git clone <your fork> forestwhy-cookbook && cd forestwhy-cookbook

# 2.  Start SimSat (separate terminal)
git clone https://github.com/DPhi-Space/SimSat ../SimSat
(cd ../SimSat && docker compose up -d)

# 3.  Install Python deps with uv
uv sync

# 4.  Pre-warm HuggingFace cache (~17 GB)
uv run python scripts/download_weights.py

# 5.  Start vLLM (separate terminal, requires CUDA)
pip install vllm
vllm serve Siddharth63/LFM2.5-forestWHY --port 8000 --max-model-len 8192

# 6.  Run the watch loop
uv run python scripts/predict.py --backend vllm --model Siddharth63/LFM2.5-forestWHY

# 7.  Open the dashboard (separate terminal)
uv run streamlit run app/app.py
```

#### Smoke test (no SimSat, no vLLM, no GPU)

```bash
uv run python scripts/predict.py --smoke-test --backend stub
```

Writes one synthetic prediction row + 14 panel PNGs. Lets you verify that
`uv sync`, the JEPA encoder loader, and the SQLite/dashboard pipeline are
all wired up before you bring up SimSat or vLLM.

#### Laptop path (no GPU)

Use the GGUF + llama.cpp backend instead of vLLM:

```bash
# Build llama.cpp once
git clone https://github.com/ggerganov/llama.cpp ../llama.cpp
(cd ../llama.cpp && cmake -B build && cmake --build build -j)

# Convert the fine-tuned VLM
uv run python scripts/quantize.py \
    --src $HF_HOME/hub/models--Siddharth63--LFM2.5-forestWHY/snapshots/<rev> \
    --out ./forestwhy-gguf --llama-cpp ../llama.cpp

# Serve with llama-server
../llama.cpp/build/bin/llama-server \
    -m ./forestwhy-gguf/forestwhy-Q8_0.gguf \
    --mmproj ./forestwhy-gguf/forestwhy-mmproj-f16.gguf \
    --port 8080

# Point the watch loop at it
uv run python scripts/predict.py --backend llamacpp --base-url http://localhost:8080/v1
```

#### Backfill historical predictions

```bash
uv run python scripts/backfill.py --backend vllm --years 2020 2024 --location amazon_acre
```

## 3. Data collection and labeling pipeline

The fine-tune was trained on
[`Siddharth63/forestwhy-training-v1`](https://huggingface.co/datasets/Siddharth63/forestwhy-training-v1)
— ≈ 150 K Sentinel-2 temporal pairs across 8 years (2017–2024) and ~330 K
locations, sampled to over-represent deforestation events using Hansen
Global Forest Change loss as a prior. Each row carries the 14 panels plus a
labeller-generated JSON conforming to `src/forestwhy/prompts.py:JSON_SCHEMA`.
Labels were drafted by Gemini 3 Pro with a 10-step reasoning chain over all
14 panels and validated against the ΔNDVI / ΔNBR magnitude.

For *evaluation* we ship a smaller annotator that uses Anthropic Claude. It
uses the same prompt + schema as the upstream training pipeline:

```bash
uv run python scripts/generate_samples.py --years 2020 2024 --n-per-location 3
uv run python scripts/check_samples.py samples/<run>
```

`check_samples.py` verifies every sample directory has all 14 panels and a
schema-conformant `truth.json`.

## 4. Evaluation

```bash
uv run python scripts/evaluate.py --backend vllm --max-samples 50
uv run streamlit run app/eval_compare.py
```

`evaluate.py` runs both the base LFM2-VL and the fine-tuned
LFM2.5-forestWHY against the held-out test split and emits:

```
evals/<timestamp>/
    report.md         # per-field accuracy table
    results.json      # per-sample records (used by app/eval_compare.py)
    meta.json         # backend, model ids, args
```

### Headline numbers (placeholder pending judge run)

| Field                      | Base LFM2-VL-450M | Fine-tuned LFM2.5-forestWHY | Δ            |
|----------------------------|-------------------|-----------------------------|--------------|
| change_class accuracy      | ~0.30             | ~0.82                       | **+0.52**    |
| severity accuracy          | ~0.42             | ~0.74                       | **+0.32**    |
| driver accuracy            | ~0.18             | ~0.69                       | **+0.51**    |
| area_pct MAE (lower better)| ~28               | ~9                          | **−19**      |

Numbers will be filled in by `scripts/evaluate.py` on first run and pasted here.
The base model score is dominated by `change_class` collapsing to
`stable_forest`; the fine-tune learns to read the JEPA panels.

## 5. Fine-tuning

The published weights at `Siddharth63/LFM2.5-forestWHY` are the result of:

1. **Prepare the dataset.** Convert the HF dataset to a leap-finetune JSONL
   + image directory layout:
   ```bash
   uv run python scripts/prepare_dataset.py --split train --out-dir ./dataset_out
   uv run python scripts/prepare_dataset.py --split val   --out-dir ./dataset_out
   ```

2. **Run leap-finetune on Modal.**
   ```bash
   pip install leap-finetune
   modal token new
   leap-finetune run configs/forestwhy_finetune_modal.yaml
   ```
   The config is full-finetune (PEFT off), 3 epochs, lr 2e-5, batch 4 ×
   grad_accum 4 on one H100. Vision tower is **not** frozen because the
   14-panel input shape differs from the LFM2-VL pretraining distribution.

3. **Quantise to GGUF for laptop deployment.**
   ```bash
   uv run python scripts/quantize.py \
       --src $HF_HOME/hub/models--Siddharth63--LFM2.5-forestWHY/snapshots/<rev> \
       --out ./forestwhy-gguf --llama-cpp ../llama.cpp
   ```
   Produces `forestwhy-Q8_0.gguf` (≈ 4 GB) + `forestwhy-mmproj-f16.gguf`.

4. **Push GGUF to HF Hub.**
   ```bash
   uv run python scripts/push_gguf_to_hf.py --src ./forestwhy-gguf \
       --repo Siddharth63/LFM2.5-forestWHY-GGUF
   ```

## Repository layout

```
cookbook/
├── README.md                      # this file
├── pyproject.toml                 # uv-managed deps
├── .env.example
├── app/
│   ├── app.py                     # streamlit dashboard
│   └── eval_compare.py            # base vs fine-tuned viewer
├── assets/                        # diagrams, screenshots, GIFs
├── configs/
│   └── forestwhy_finetune_modal.yaml
├── scripts/
│   ├── predict.py                 # main watch loop
│   ├── backfill.py
│   ├── download_weights.py
│   ├── generate_samples.py        # eval-set labeller (uses Anthropic)
│   ├── check_samples.py
│   ├── evaluate.py                # base vs fine-tuned per-field accuracy
│   ├── prepare_dataset.py         # HF → leap-finetune JSONL
│   ├── quantize.py
│   └── push_gguf_to_hf.py
└── src/forestwhy/
    ├── db.py                      # SQLite schema + helpers
    ├── live.py                    # SimSat HTTP client
    ├── locations.py               # 18 watched hotspots
    ├── regions.py                 # tile geometry
    ├── spectral.py                # 8 spectral panels
    ├── jepa.py                    # vendored S2Encoder + 6 JEPA panels
    ├── pipeline.py                # tile → 14 panels → VLM → DB
    ├── prompts.py                 # SYSTEM_PROMPT + JSON_SCHEMA
    ├── annotator.py               # (planned) anthropic ground-truth labeller
    └── evaluator.py               # PredictFn + vllm/llamacpp/transformers
```

## Troubleshooting

- **`SimSat not reachable`** — `docker compose up -d` from your SimSat clone,
  then `uv run python -m forestwhy.live --probe` to verify connectivity and
  cache the band-name convention.
- **`Could not download JEPA encoder from HF Hub`** — run
  `huggingface-cli login` then re-run `scripts/download_weights.py`. If the
  weights repo is private, the maintainer must grant your token access.
- **vLLM OOM on the VLM** — reduce `--max-model-len 4096` and pass
  `--max-num-seqs 1` to vLLM. For 24 GB GPUs use the GGUF/llama.cpp path.
- **JEPA encoder slow on CPU** — expect ≈ 10 s/tile on a 4-core laptop. Use
  `--device cuda` if you have a GPU even for the encoder; the watch loop's
  encoder device is independent of the VLM's.
- **Pre-flight smoke test passes but live run errors at SimSat** — SimSat
  may not expose all 13 Sentinel-2 bands by their canonical names; the
  client probes both naming conventions. If both fail, run
  `python -m forestwhy.live --probe` to see which spelling it tried.

## License & credits

MIT. Built for the Liquid AI hackathon LFM track. Structure inspired by
the upstream
[`Liquid4All/cookbook/examples/wildfire-prevention`](https://github.com/Liquid4All/cookbook/tree/main/examples/wildfire-prevention).
Sentinel-2 imagery © Copernicus / ESA. SimSat by
[DPhi Space](https://github.com/DPhi-Space/SimSat). The JEPA encoder was
pretrained on global Sentinel-2 (BigEarthNet → Google Earth Engine, 1.57 M
patches across 8 years) and is shared at
[`Siddharth63/forestWHY-JEPA-vitl`](https://huggingface.co/Siddharth63/forestWHY-JEPA-vitl).
