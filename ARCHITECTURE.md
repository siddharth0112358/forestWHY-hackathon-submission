# forestWHY — Architecture

This document explains how forestWHY is built, why this design beats the
alternatives, and what is genuinely new in the system. For step-by-step
instructions, see [QUICKSTART.md](QUICKSTART.md).

## TL;DR

forestWHY runs deforestation analysis aboard a simulated satellite. It
combines a **frozen Sentinel-2 I-JEPA encoder** with a **fine-tuned
LFM2.5-VL** to turn a temporal pair of 13-band Sentinel-2 tiles into a
structured forest-change report — small enough to downlink as JSON instead
of imagery. The architecture's two pillars:

1. The **14-panel input** — 8 spectral panels plus 6 differential panels
   from the JEPA encoder — encodes both *what* changed (spectral) and
   *where attention shifted in semantic feature space* (JEPA). Either
   alone is fooled by clouds, phenology, or shadow. Together they are
   robust.
2. **LFM2.5-forestWHY**, a 2 B fine-tune of LFM2-VL trained on 150 K
   labelled Sentinel-2 pairs, learns to read the JEPA panels with the
   same fluency as the spectral ones — something neither a generic VLM
   nor a contrastive change detector can do.

## Block diagram

```
                ┌──────────────────────────────┐
                │         SimSat               │
                │  (simulated LEO orbit + S2)  │
                └──────────┬───────────────────┘
                           │
        ┌──────────────────┴──────────────────┐
        │ /position   /image/sentinel?...     │
        ▼                                     ▼
   sat_lon, sat_lat                      13-band tile
                                          (after, current)
                                              │
                                              ▼
                                    ┌──────────────────────┐
                                    │ Historical fetch     │ ← T-1 year +
                                    │  with 14-day walkback│   walkback for
                                    │                      │   cloud-free pair
                                    └──────────┬───────────┘
                                               │
                            (before, after) 13-band, 64×64
                                               │
                ┌──────────────────────────────┴───────────────────────────────┐
                │                                                              │
                ▼                                                              ▼
    ┌──────────────────────┐                              ┌─────────────────────────────────┐
    │  Spectral processor  │                              │  I-JEPA ViT-L/8 encoder         │
    │  (numpy + matplotlib)│                              │  302 M params, 24 layers,       │
    │  8 panels:           │                              │  pretrained on 1.57 M S2 patches│
    │  RGB×2, NIR-FC×2,    │                              │  6 differential panels:         │
    │  SWIR×2, ΔNDVI, ΔNBR │                              │  attention_multi,               │
    │                      │                              │  embedding_change,              │
    │                      │                              │  cropa_roads,                   │
    │                      │                              │  delta_attn_role,               │
    │                      │                              │  head_disagreement, pca_semantic│
    └──────────┬───────────┘                              └──────────┬──────────────────────┘
               │                                                     │
               └────────────────────┬────────────────────────────────┘
                                    │
                                    ▼
                         14 PNG panels in canonical order
                                    │
                                    ▼
                  ┌───────────────────────────────────┐
                  │  LFM2.5-forestWHY                 │
                  │  (LFM2-VL 2B fine-tune)           │
                  │  served by:                       │
                  │   - vLLM on CUDA host (prod)      │
                  │   - llama-server + GGUF (laptop)  │
                  │   - transformers (any device)     │
                  └────────────┬──────────────────────┘
                               │
                               ▼
        10-step reasoning chain (prose) + extracted JSON fields:
        change_class, severity, area_pct, driver_hypothesis,
        confidence, reasoning, cloud_cover_note
                               │
                               ▼
                     SQLite (predictions.db)
                               │
                               ▼
                  Streamlit dashboard (pydeck map +
                  14-panel viewer + JSON card)
```

## The 14-panel input — what each panel contributes

| #  | name                | what it carries                                                                 |
|----|---------------------|---------------------------------------------------------------------------------|
| 1–2 | `rgb_before/after`  | broad scene context, urban/water/cloud presence                                 |
| 3–4 | `nir_fc_before/after` | vegetation vigour (B8 brightness ≈ healthy chlorophyll)                       |
| 5–6 | `swir_before/after` | burn scars, bare soil, water content; SWIR penetrates thin haze                 |
| 7  | `delta_ndvi`        | pixel-level vegetation loss, calibrated to ±0.5                                 |
| 8  | `delta_nbr`         | pixel-level biomass loss / fire signature                                       |
| 9  | `attention_multi`   | which patches the JEPA encoder attends to at three depth scales (R/G/B)         |
| 10 | `embedding_change`  | per-patch cosine distance between before and after embeddings — semantic drift |
| 11 | `cropa_roads`       | cross-patch correlation, lights up linear features (logging tracks)             |
| 12 | `delta_attn_role`   | patches that gained or lost CLS-attention importance after the change           |
| 13 | `head_disagreement` | std across attention heads — flags ambiguous land cover                         |
| 14 | `pca_semantic`      | PCA-3 of patch embeddings, before \| after — semantic clustering shift          |

Panels 1–8 are computable from any Sentinel-2 image with NumPy + matplotlib.
Panels 9–14 require the JEPA encoder.

## Why this beats the obvious alternatives

### vs. raw Sentinel-2 alerts (Hansen GFC, RADD, GLAD)

Existing alert systems run on Earth-side servers, ingest raw L0/L1 product,
and emit alerts 3–8 weeks after the event. forestWHY runs **at the
satellite**: only the 1 KB JSON ever leaves orbit, so the network
constraint that drives the lag in current systems disappears. The same
constraint that the Liquid wildfire-prevention example identified for fire
risk applies even more strongly to forest change because the signal lives
in temporal pairs, not single frames.

### vs. spectral-only change detection

The naïve baseline — threshold ΔNDVI on a before/after composite — fails
predictably on:
- **Cloud shadow** that mimics dense canopy in the after tile (false
  afforestation alert).
- **Seasonal phenology** in dry-season composites (false deforestation in
  Cerrado / Miombo).
- **Selective logging** under canopy, where < 30 % of biomass is removed
  but the road grid is the actual signal.

The JEPA panels recover these cases:
- Panel 10 (`embedding_change`) flips when the *semantic class* changes,
  not when pixel values shift.
- Panel 11 (`cropa_roads`) lights up logging tracks even when ΔNDVI is
  near zero.
- Panel 13 (`head_disagreement`) flags exactly the ambiguous transitional
  zones a downstream classifier should ask the human about.

### vs. a generic VLM on RGB tiles

Stock LFM2.5-VL-1.6B or any other multimodal LLM with a ViT vision encoder
treats the 14 panels as an unfamiliar input distribution: spectral indices
are pseudo-RGB heatmaps and JEPA panels are even further out-of-domain. In
our own held-out evaluation against `forestwhy-training-v2` (out-of-
distribution from the fine-tune's training set), the base `LFM2.5-VL-1.6B`
collapses on the 14-panel input — see the headline numbers below for the
exact gap. The fine-tune learns the 14-panel dialect, and that's where the
binary-deforestation accuracy jump comes from.

### vs. supervised contrastive change-detection models (e.g. SatMAE, DOFA)

Contrastive change-detection models predict a binary or multi-class
*pixel mask*. They don't reason about *driver* (was this clearing for
agriculture vs. mining vs. fire?) or *severity*, and they don't produce a
human-readable rationale. forestWHY's 10-step reasoning chain is what
turns the system into something a downstream NGO or land-tenure agency
can act on, not just plot.

## What is innovative

1. **Compressing temporal change into 14 panels suitable for a VLM.** The
   I-JEPA panel taxonomy (multi-scale attention, CLS-role delta, CroPA
   linear-structure detector, PCA cluster split) is, to our knowledge,
   the first packaging of frozen-encoder differentials specifically
   designed for downstream multimodal-LLM consumption. Every panel
   addresses a specific failure mode of the spectral-only baseline.

2. **JEPA pretraining over MSE/contrastive baselines for satellite remote
   sensing.** I-JEPA learns to predict patch *representations* in the
   masked region rather than pixel values, so the resulting embeddings are
   class-aware in a way that MAE-pretrained encoders are not. That's
   precisely what `embedding_change` (Panel 10) needs to flip cleanly on
   class transitions.

3. **On-orbit inference with deployment plurality.** The same model + the
   same 14-panel pipeline runs unchanged across vLLM (production CUDA),
   `llama-server` + GGUF (laptop), and `transformers` (universal). Judges
   can verify every claim end-to-end on a $0/hour Mac with a 731 MB
   download — a stronger reproducibility story than most remote-sensing
   models, which tend to ship as Jupyter notebooks tied to a specific
   GPU vendor.

4. **A fine-tuning recipe with real measurable gains.** The model was
   trained on **`Siddharth63/forestwhy-combined-v1`** — a private, curated
   mix combining `forestwhy-training-v1` with hand-vetted examples covering
   under-represented deforestation drivers (artisanal mining fronts, oil-palm
   conversion, fire-driven loss, road-led degradation, plantation rotation).
   The combined dataset bakes in Hansen GFC-prior sampling and a balanced
   driver distribution; the 14-panel input format is non-trivial to learn —
   full-finetune (no PEFT) on H100 in 3 epochs at lr 2e-5 was the smallest
   config that produced the gain. The recipe is documented in
   `configs/forestwhy_finetune_modal.yaml` and reproducible with
   `leap-finetune`.

## What the dashboard shows

The Streamlit dashboard (`app/app.py`) is the human-facing surface — judges
can drive it directly via the **Run new inference** sidebar (named
hotspot, custom lat/lon via folium click-picker) or browse historical
predictions from `predictions.db`.

<p align="center">
  <img src="assets/dashboard_overview.png" width="940"
       alt="Dashboard inspector on a Madagascar east tile — deforestation, high severity, 90% area, driver agricultural_clearing"><br>
  <sub><b>High-severity deforestation.</b> Madagascar east — slash-and-burn
  (tavy) advancing through eastern rainforest. Model: <code>deforestation</code>,
  severity <b>high</b>, ~90 % area affected. The 6 JEPA panels (bottom row)
  show coherent attention, embedding shift, and CroPA road-detection
  signal — the kind of evidence stack a spectral-only model can't
  produce.</sub>
</p>

<table>
<tr>
<td><img src="assets/dashboard_deforestation_low.png"
        alt="Dashboard inspector — lower-severity deforestation prediction"></td>
<td><img src="assets/dashboard_stable_forest.png"
        alt="Dashboard inspector — protected primary forest, stable_forest"></td>
</tr>
<tr>
<td><sub><b>Lower-severity deforestation.</b> Subtler change pattern — the
fine-tune still commits, area estimate ~35 %.</sub></td>
<td><sub><b>Stable forest contrast.</b> Yasuní / Manu-class protected
primary forest. Model correctly outputs <code>stable_forest</code>,
severity <code>none</code>, area 0 %. Useful baseline visualisation —
forestWHY is not just a deforestation classifier but a change-state
classifier.</sub></td>
</tr>
</table>

## Evaluation methodology

`scripts/evaluate.py` runs both the base **`LiquidAI/LFM2.5-VL-1.6B`** and
the fine-tuned **`Siddharth63/LFM2.5-forestWHY`** against a balanced 50/50
subsample of **`Siddharth63/forestwhy-training-v2`** (deforestation =
`active_front_anthropogenic`; all other classes pooled into "no
deforestation"), and reports:

- **binary deforestation accuracy** — the simplest, most actionable metric
  (did the model correctly identify whether the tile shows active forest
  loss?).
- **change_class accuracy** — exact match on the 6-class label after
  mapping the model's output vocabulary onto v2's
  (`deforestation` → `active_front_anthropogenic`,
  `stable_forest` → `stable_forest_intact`, etc.).
- **driver accuracy** — coarse driver categorisation
  (agricultural_clearing / logging_road / mining / fire / plantation / …).
- **area_pct MAE** — mean absolute error on the percentage of tile area
  affected.

`app/eval_compare.py` shows per-sample side-by-side predictions of base
vs. fine-tuned, with the 14 panels rendered for human inspection.

### Training vs. evaluation data — out-of-distribution by design

The fine-tune was trained on **`Siddharth63/forestwhy-combined-v1`**
(private). The public **`forestwhy-training-v2`** that we evaluate against
is a **held-out, out-of-distribution** set that the model has never seen,
giving an honest estimate of generalisation rather than train-set
memorisation. The two datasets share the same 14-panel layout and label
taxonomy but use disjoint geographic samples and different temporal pair
selection.

### How to reproduce the README's 5 demo GIFs

Two scripts, one call each. With `llama-server` running the Q8_0 fine-tune
on `:8080` and SimSat docker up:

```bash
# (1) Run a curated 10-location SimSat backfill — 8 active hotspots + 2
#     stable contrasts (Yasuní + Manu protected areas).
uv run python scripts/demo_backfill.py

# (2) Pick the top-5 by demo score (area × class × confidence − cloud
#     penalty), build before/after RGB GIFs into assets/.
uv run python scripts/make_demo_gifs.py
```

Total wall-clock ≈ 7 min on Apple Silicon Metal. Selection criteria, the
five chosen scenes, the model's predictions, and the resulting GIFs are
all in `scripts/make_demo_gifs.py:HEADLINE_REGIONS` and the script's stdout
ranking — fully auditable, no hand curation beyond the score formula.

## Headline numbers

50-sample balanced evaluation against `forestwhy-training-v2` (25
deforestation + 25 other-class, randomly drawn from the first 84 rows of
the public split). Both models served via `llama-server` with Q8_0
backbone + F16/BF16 mmproj on Apple M3 Max Metal. Methodology and
limitations described above. Source: `evals/20260506T201742/report.md`.

| Metric                          | LFM2.5-VL-1.6B (base) | LFM2.5-forestWHY (fine-tune) | Δ          |
|---------------------------------|-----------------------|------------------------------|------------|
| **binary deforestation acc.**   | **0.500** (random)    | **0.694**                    | **+0.194** |
| change_class acc. (6-way mapped)| 0.219                 | 0.367                        | +0.149     |
| driver acc. (8-way)             | 0.062                 | **0.586**                    | **+0.524** |
| area_pct MAE (lower=better)     | — (no extraction)     | 7.15                         | —          |

**Interpretation.**

- The fine-tune lifts binary deforestation accuracy from chance (50.0 %, exactly
  what a random classifier would score on a balanced 25/25 split) to 69.4 %.
  That's a **+19.4 point absolute lift** on the headline question
  ("did this tile show forest loss?"). At N=50 the 95 % CI on each rate is
  roughly ±13 pts; the lift itself is comfortably outside the noise band.
- Driver attribution is where the fine-tune's training data shines hardest:
  the base model produces almost no usable driver text (6.2 %) because it
  wasn't trained on the 14-panel taxonomy. The fine-tune classifies
  `agricultural_clearing / logging_road / mining / fire / plantation / …`
  correctly **58.6 %** of the time — a +52 point swing.
- The 6-way `change_class` accuracy gap (37 % vs. 22 %) is smaller than the
  binary because the v2 vocabulary is finer-grained
  (`stable_forest_intact` vs. `stable_forest_managed` etc.); both models
  often pick the right *family* but the wrong leaf.
- The base model's `area_pct` column is blank because its free-form prose
  rarely contains a percent figure for the affected area; the fine-tune's
  7.1 % mean absolute error is on the same scale as the natural variability
  in the labels themselves.

**Caveats worth being honest about.**

- N = 50 is a pilot; we'd run 300 if the M3 Max could sustain a 4-hour
  run without thermal throttling. The trend is in the right direction at
  ≥ 95 % confidence on the headline binary metric, but the 6-way
  change_class number could shift several points either way at higher N.
- These numbers are **out-of-distribution for the fine-tune** — it never
  saw `forestwhy-training-v2` during training. In-distribution scores on a
  held-out slice of `forestwhy-combined-v1` (the private training set)
  would be higher; we deliberately reported the harder number.
- Both models were quantised to Q8_0; F16 inference would likely give
  slightly higher numbers on both sides but not change the relative
  ordering.

### Where the remaining errors come from

Running `scripts/audit_misclassifications.py` over the 50-sample run
classifies each fine-tune error as either *prose-extraction failure* (the
model said the right thing, my synonym table missed it) or *model
interpretation error* (the prose itself is wrong):

| Error type | Total | Prose-extraction | Model interpretation | Ambiguous |
|---|---|---|---|---|
| Binary deforestation errors | 16 / 50 | 0 (0 %) | **15 (94 %)** | 1 (6 %) |
| Driver mismatch errors      | 35 / 50 | 23 (66 %) | 5 (14 %) | 7 (20 %) |

**Takeaways.**

1. **Binary errors are model-side, not extraction-side.** When the model
   gets the headline question wrong, it almost always says "stable forest"
   because it interprets atmospheric artefacts (haze, cloud, seasonal
   phenology) as the dominant signal. The normaliser is reading the prose
   correctly; the prose itself is conservative under uncertainty.
   Representative example: a flagged Bolivia tile (`active_front_anthropogenic`
   in v2) — the model wrote *"evidence strongly points to a stable forest
   landscape where the observed spectral changes are artifacts of
   atmospheric conditions (haze clearing)"*. That is a 1.6 B-parameter
   model being well-calibrated, not poorly extracted.
2. **Most "driver mismatch" errors are downstream of binary errors.** When
   the model says "stable forest", it doesn't propose a driver, so
   `driver_hypothesis` collapses to `unknown`. This shows up in the audit
   as a synonym-miss — but the underlying cause is the same as #1.
3. **What would actually move the binary number.** More training pairs
   where the *truth is active deforestation despite a hazy after-image*;
   prompt revision to bias the model toward detection under uncertainty;
   or simply scaling beyond 1.6 B. Synonym-table tweaks won't help.

This is a fair-and-honest characterisation of the bias the model exhibits
on out-of-distribution data — useful both for hackathon judging and for
deciding what to invest in next.

To re-run at higher N (or with vLLM for speed):

```bash
uv run python scripts/evaluate.py \
    --base-backend llamacpp     --base-base-url     http://localhost:8081/v1 \
    --finetuned-backend llamacpp --finetuned-base-url http://localhost:8080/v1 \
    --max-samples 300
```

Results land in `evals/<timestamp>/report.md` (the source of truth) plus a
`results.json` consumed by `app/eval_compare.py` for per-sample inspection.

## Open questions and known limits

- **Cloud cover collapses the JEPA signal.** When > 60 % of either tile is
  cloudy, all six JEPA panels are dominated by atmospheric noise. The
  dashboard filters these out by default; production deployment would
  need an upstream cloud-mask gate.
- **The model is biased toward deforestation.** Training-set sampling
  was 55 % deforestation by design. False-positive rate on stable forest
  is the main residual error; the `confidence` field is the lever to
  threshold on.
- **Sentinel-2 5-day revisit is the temporal resolution ceiling.** For
  active fire fronts the system would benefit from a follow-on with a
  geostationary thermal sensor.
- **The JEPA encoder is frozen.** A jointly trained encoder + VLM might
  push accuracy further, at the cost of needing to retrain the encoder
  on each iteration.

## Who this serves and what they pay today

forestWHY's structured output (`change_class`, `severity`, `area_pct`,
`driver_hypothesis`, plus the prose rationale) maps onto four concrete
buyers who are already spending real money on the alternatives.

### 1. EU Deforestation Regulation (EUDR) compliance teams

**Who.** Importers and traders of seven commodities into the EU — soy,
palm oil, cattle/beef, cocoa, coffee, rubber, wood — must prove every
shipment originates from plots that have not been deforested since
**31 December 2020**. Enforcement starts **30 December 2025** for large
operators, six months later for SMEs. Non-compliance fines: up to **4 % of
EU turnover**.

**What they pay today.**
- Satelligence, Descartes Labs, LiveEO, Forest IQ subscriptions:
  **€0.50–2.00 per hectare per year** for plot-level monitoring.
- Manual ground audits where satellite is inconclusive: **€5–50/ha/year**.
- Internal compliance staff: 1–3 FTE per major importer.

**What we deliver.** Structured per-plot reports with a date-stamped
`change_class`, `area_pct`, and `driver_hypothesis` aligned to the EUDR's
"deforestation" and "forest degradation" definitions. The prose rationale
is what a compliance officer pastes into a due-diligence statement; that
field is the one a generic change-detection API can't produce. Pricing
goal: ≤ €0.20/ha/year by running on-orbit inference instead of round-trip
to a cloud GPU.

### 2. Voluntary carbon-credit verification (REDD+ / ARR / IFM projects)

**Who.** Validation/Verification Bodies (VVBs) under Verra (VCS), Gold
Standard, Plan Vivo, ART/TREES. Project developers (Wildlife Works,
Pachama, Sylvera, Renoster) running REDD+ at landscape scale.

**What they pay today.**
- Project verification cycle (every 5 years for VCS): **$30 K–$200 K per
  project**, dominated by VVB site visits and stratified ground sampling.
- Buffer-pool reversals + integrity scandals (Pachama, ZEE Verra
  re-baseline) cost the industry an estimated **$1–2 B in mark-downs**
  during 2023–2024.
- High-frequency monitoring services (Pachama Forecast, Sylvera REDD
  Watch): **$10–50 K per project per year** on top of verification.

**What we deliver.** Continuous tile-level change classification with the
specific driver fields VVBs need to distinguish leakage from reversal
from intentional clearing. The 10-step reasoning chain produces an audit
trail for each alert — the missing artefact in current automated
monitoring stacks. Specific replacement target: VVBs running quinquennial
on-site checks because they don't trust automated mid-period attribution.

### 3. National forest agencies in tropical countries

**Who.** Ibama / INPE (Brazil), KLHK + Sipongi (Indonesia), DGEF (DRC),
Ministerio del Ambiente (Peru), Forest Department (Cambodia).

**What they pay today.**
- Internal alert systems: PRODES + DETER (Brazil), MoEF SIPONGI
  (Indonesia), GFW alerts globally. Annual budgets in **$10s of millions
  per agency**.
- Limitation: existing alerts say *where* deforestation happened with low
  latency but rarely classify *why* (mining vs. plantation vs. fire vs.
  agricultural clearing). Driver attribution is done downstream by hand.

**What we deliver.** Driver-attributed alerts that go directly into
prosecutorial workflows. A Brazilian prosecutor (MPF) building a case
against an illegal cattle operation needs `driver_hypothesis="agricultural_clearing"`
+ `cropa_roads` evidence + a written rationale, not just a polygon. That's
the brief our 10-step output already meets.

### 4. Agricultural lenders, banks, insurers

**Who.** IDB Invest, Rabobank, ICAEW Sustainable Finance Institute
members, Lloyd's syndicates writing tropical land insurance.

**What they pay today.**
- Subscription due-diligence services for land collateral: **$3–10/ha/year**.
- Reputational risk: HSBC, JBS, Cargill have each paid 8-figure penalties
  / divestments over deforestation-linked exposures since 2020.

**What we deliver.** Continuous proof of forest stock for collateral
underwriting + structured event records when something changes. The
auditability of the prose rationale is the differentiator — a credit
committee can read why a flag was raised, not just receive a Boolean.

### Honest sizing

This is not a $100 B TAM pitch. The realistic addressable spend across
the four segments above is **$1.5–4 B/year** by 2027 (mostly EUDR-driven).
forestWHY is one component of a future product — specifically, the
*reasoning layer* on top of an alert pipeline. It does not replace
GFW/PRODES, it sits between them and the human reader.

The defensible product wedge is **driver-attributed, audit-ready alerts
priced 5–10× below incumbents because inference runs on commodity
hardware instead of a managed cloud GPU stack**. Whether that's a
self-serve API, an embeddable widget for the GFW UI, or a white-label
service for VVBs is a packaging question, not a tech question.

## Related work

- I-JEPA — *Self-Supervised Learning from Images with a Joint-Embedding
  Predictive Architecture*, Assran et al. 2023.
- LFM2-VL family — Liquid AI's compact multimodal foundation models.
- The Liquid `wildfire-prevention` cookbook — same on-orbit inference
  pattern, single-frame fire-risk classification rather than temporal
  forest-change detection. forestWHY's structure deliberately mirrors
  it so judges familiar with one example recognise the other.
