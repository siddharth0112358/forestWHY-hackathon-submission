# forestWHY — Quickstart

This is the step-by-step guide to run the project. For the architecture and
the design rationale, see [ARCHITECTURE.md](ARCHITECTURE.md). For the
high-level project description and repo layout, see [README.md](README.md).

There are **three** fidelities at which you can run forestWHY, listed below
from cheapest to most realistic. Pick the one that matches the resources
you have:

| Mode | What runs | Hardware | Time-to-first-row |
|---|---|---|---|
| 1. **Stub smoke test** | JEPA panels + dummy VLM output | any laptop | ~30 s |
| 2. **GGUF smoke test** | JEPA panels + real fine-tuned VLM (Q4_K_M GGUF) | any laptop | ~3 min first run, ~15 s after |
| 3. **Live SimSat run** | Full pipeline against simulated satellite + vLLM | local Docker + remote GPU | varies |

Mode 2 is the **judge-friendly default**: it works on a Mac with no GPU,
no SimSat, no remote services, and exercises the actual fine-tuned model.

## 0. Prerequisites

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- For Mode 2 only: `llama-server` on PATH
  - macOS: `brew install llama.cpp`
  - Linux: build from source — see the [Mode 2 troubleshooting](#troubleshooting) section
- For Mode 3 only: Docker (for SimSat) + a remote CUDA host running vLLM

## 1. Clone and install

```bash
git clone https://github.com/siddharth0112358/forestWHY-hackathon-submission.git
cd forestWHY-hackathon-submission
uv sync
```

This creates a `.venv/` with PyTorch, transformers, streamlit, and ~80
other deps. Takes 2–3 min on a fresh checkout.

(Optional) copy environment defaults — only needed if you want to override
the auto-pulled HF repos:

```bash
cp .env.example .env
```

The defaults already point at:
- JEPA encoder → `Siddharth63/forestWHY-JEPA-vitl`
- VLM (vLLM serving) → `Siddharth63/LFM2.5-forestWHY`
- VLM (GGUF, Q4_K_M) → `Siddharth63/LFM2.5-forestWHY-GGUF`

## Mode 1 — Stub smoke test (fastest)

Verifies the JEPA encoder loads and the 14-panel pipeline + SQLite + dashboard
all work. Uses a stub VLM (no model call). Good first sanity check.

```bash
uv run python scripts/predict.py --smoke-test --backend stub
uv run streamlit run app/app.py
```

What you see:
- Console: JEPA encoder downloads (1.21 GB) on first run, loads in ~5 s on Apple Silicon (MPS) or CPU.
- `predictions.db` gets one row, `db_images/1/` gets 14 PNG panels.
- Dashboard opens at http://localhost:8501 with the row plotted on a world map.

Stop the dashboard with Ctrl+C in its terminal.

## Mode 2 — End-to-end with the real fine-tuned VLM (no GPU needed)

Runs the same pipeline as Mode 1 but **calls the real LFM2.5-forestWHY** via
a locally-launched `llama-server` on the GGUF weights. The Q4_K_M quant is
731 MB; the BF16 mmproj projector is ~200 MB. First run downloads both;
subsequent runs hit the HF cache.

```bash
# One-time install
brew install llama.cpp        # macOS
# (Linux: build llama.cpp from source — see troubleshooting below)

# Run
uv run python scripts/predict.py --smoke-test --with-gguf
```

`--with-gguf` makes `predict.py`:
1. Resolve the GGUF + mmproj from `Siddharth63/LFM2.5-forestWHY-GGUF` (HF cache).
2. Launch `llama-server` on port 8080 with `--mmproj`.
3. Wait for the OpenAI-compatible API to respond.
4. Synthesise a tile, run JEPA + spectral panels, call the VLM.
5. Tear down `llama-server`.

Open the dashboard:

```bash
uv run streamlit run app/app.py
```

You should see the model's full 10-step reasoning under "Raw VLM JSON". On
synthetic input the reasoning will note the noise / lack of land-cover
features — that's correct. On real Sentinel-2 imagery from SimSat (Mode 3),
it produces specific deforestation/fire/clearing analysis.

**Override the quant** if you want a higher-quality but bigger weight:

```bash
uv run python scripts/predict.py --smoke-test --with-gguf --gguf-quant Q8_0   # 1.25 GB
```

## Mode 3 — Live run against SimSat + a hosted vLLM

Adds the simulated satellite and a remote/hosted GPU running vLLM. This is
the production-shaped flow.

### 3a. Start SimSat (separate terminal)

```bash
git clone https://github.com/DPhi-Space/SimSat ../SimSat
cd ../SimSat
# SimSat's `sim` container imports a Mapbox provider at boot and crashes
# without MAPBOX_ACCESS_TOKEN. We do not call the Mapbox endpoint, so any
# non-empty string is fine. To use the real Mapbox endpoint, get a free
# token at https://account.mapbox.com/access-tokens and set it here.
MAPBOX_ACCESS_TOKEN=dummy_token_for_sentinel_only docker compose up -d
```

Once both containers report healthy (`docker ps` should show
`fakesat-dashboard` on :8000 and `fakesat-sim` on :9005), start the orbit
(it boots paused at lon=0, lat=0):

```bash
curl -X POST http://localhost:8000/api/commands/ \
    -H "Content-Type: application/json" \
    -d '{"command":"start","step_size_seconds":20,"replay_speed":50}'
```

Or open http://localhost:8000 and click the **Start** button on the
dashboard.

Verify connectivity:

```bash
cd /path/to/forestWHY-hackathon-submission
uv run python -m forestwhy.live --probe         # API healthy + band info
uv run python -m forestwhy.live --position      # current sat lon/lat
uv run python -m forestwhy.live --current       # fetch (13, H, W) tile (over land only)
```

The probe reports the 12 spectral bands SimSat exposes from Sentinel-2
L2A: `coastal, blue, green, red, rededge1, rededge2, rededge3, nir,
nir08, nir09, swir16, swir22`. Sentinel-2 L2A drops B10/cirrus during
processing; `live.py` zero-pads that channel so the JEPA encoder still
sees a `(13, H, W)` input.

### 3b. Start vLLM on a CUDA host

On any rented GPU (Modal, RunPod, Lambda, AWS, etc.):

```bash
pip install vllm
vllm serve Siddharth63/LFM2.5-forestWHY \
    --port 8000 --max-model-len 8192
```

Expose port 8000 to your laptop (ngrok, ssh tunnel, or a public IP):

```bash
ngrok http 8000     # one option
```

### 3c. Run the watch loop

```bash
uv run python scripts/predict.py \
    --backend vllm \
    --base-url https://<your-vllm-host>/v1 \
    --interval 60
```

The loop polls SimSat every 60 s, fetches before/after Sentinel-2 13-band
pairs whenever the satellite passes one of the 18 watched hotspots, runs
the full panel pipeline + remote VLM, and writes one row per scoring event.

### 3d. (Alternative) Live run with the local GGUF

If you'd rather skip the GPU, the laptop GGUF backend works against live
SimSat too:

```bash
# Terminal A: launch llama-server manually
llama-server \
    -m ~/.cache/huggingface/hub/models--Siddharth63--LFM2.5-forestWHY-GGUF/snapshots/*/LFM2.5-forestWHY.Q4_K_M.gguf \
    --mmproj ~/.cache/huggingface/hub/models--Siddharth63--LFM2.5-forestWHY-GGUF/snapshots/*/LFM2.5-forestWHY.BF16-mmproj.gguf \
    --port 8080 --jinja -fa on

# Terminal B: predict.py against it
uv run python scripts/predict.py --backend llamacpp --base-url http://localhost:8080/v1
```

### 3e. Backfill historical tiles (optional, for the dashboard demo)

```bash
uv run python scripts/backfill.py \
    --backend vllm \
    --base-url https://<your-vllm-host>/v1 \
    --years 2020 2024
```

Iterates the 18 hotspots comparing 2020 vs 2024 — populates the dashboard
with ~18 historical predictions in one shot. Takes ~20 min on an H100.

## 4. The dashboard

```bash
uv run streamlit run app/app.py
```

Sidebar filters by region, change_class, severity, driver, cloud cover.
Click any row to see the 14-panel grid + the full VLM response.

For the base-vs-fine-tune comparator (after running `evaluate.py`):

```bash
uv run streamlit run app/eval_compare.py
```

## 5. Evaluation

```bash
uv run python scripts/evaluate.py --backend vllm --max-samples 50 \
    --base-base-url https://<base-vllm>/v1 \
    --finetuned-base-url https://<finetuned-vllm>/v1
```

Outputs `evals/<timestamp>/`:
- `report.md` — per-field accuracy table (paste into ARCHITECTURE.md results section)
- `results.json` — per-sample records (used by `app/eval_compare.py`)

## Troubleshooting

**`llama-server` not found on PATH**
- macOS: `brew install llama.cpp`
- Linux: clone llama.cpp, `cmake -B build && cmake --build build -j`, then `sudo install build/bin/llama-server /usr/local/bin/`

**JEPA encoder fails to download (HF 401)**
- Run `hf auth login` and provide a read token.
- The repo `Siddharth63/forestWHY-JEPA-vitl` is public, so this should not normally happen.

**`SimSat not reachable at http://localhost:9005`**
- `docker ps` and check that **both** `fakesat-dashboard` and `fakesat-sim` containers are up.
- If `fakesat-sim` exited, `docker logs fakesat-sim | tail -20`. The most common cause is missing `MAPBOX_ACCESS_TOKEN`. Fix:
  ```bash
  cd ../SimSat
  MAPBOX_ACCESS_TOKEN=dummy_token_for_sentinel_only docker compose up -d --force-recreate sim
  ```
- After both containers are healthy, the orbit simulator must be **manually started** via the dashboard UI or `POST /api/commands/ {"command":"start"}` — see step 3a above. Until then `lon-lat-alt` returns `[0,0,0]`.

**`SimSat reported no image available` over open ocean**
- Correct behaviour. Sentinel-2 doesn't cover open oceans. Use `predict.py` against named land hotspots (`--location amazon_acre`) or wait for the orbit to cross land.

**Probe shows only 12 bands instead of 13**
- Correct. Sentinel-2 L2A doesn't retain B10 (cirrus); SimSat ships L2A. `live.py` zero-pads B10 in the JEPA input. The JEPA encoder's B10 weights contribute negligibly to the change-detection panels.

**vLLM out-of-memory on the VLM**
- Reduce `--max-model-len 4096` and pass `--max-num-seqs 1` to vLLM.
- For 24 GB GPUs use the GGUF path instead.

**MPS (Apple Silicon) backend warnings during JEPA load**
- `ggml_metal: tensor API disabled for pre-M5 and pre-A19 devices` is informational, not an error.
- The encoder still runs on Metal. Expect ~3–5 s per tile on M1/M2/M3.

**Schema validation warnings in the predict.py output**
- The fine-tune emits free-form prose with a 10-step reasoning protocol, not strict JSON. The evaluator's `_normalize` step extracts dashboard fields where possible and stores the full prose under `raw_response`. The warnings are expected and harmless.

**First-run downloads take a long time**
- Total cold-cache download: ~2 GB (JEPA 1.2 GB + GGUF 0.7 GB + mmproj 0.2 GB).
- Run `uv run python scripts/download_weights.py` once at setup time to pre-warm.

## Smoke-test checklist (for judges)

The repo passes if **all four** of these complete without manual edits:

```bash
git clone https://github.com/siddharth0112358/forestWHY-hackathon-submission.git
cd forestWHY-hackathon-submission
uv sync                                                         # 1. install
uv run python scripts/predict.py --smoke-test --backend stub    # 2. JEPA + stub
brew install llama.cpp                                          # 3. install llama-server
uv run python scripts/predict.py --smoke-test --with-gguf       # 4. real VLM end-to-end
uv run streamlit run app/app.py                                 # 5. visual confirmation
```

If step 4 prints `Smoke test wrote row id=1` and step 5 shows the row on
the map with 14 panels and a 10-step reasoning prose under the JSON
expander, the system is fully operational.
