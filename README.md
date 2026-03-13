# Landscape Signatures BYOL

**Creating quantifiable metrics of Baukultur with self-supervised visual place recognition.**

This pipeline learns compact, L2-normalized *landscape signatures* — high-dimensional embeddings that encode the architectural and urban character of a place — by applying **Bootstrap Your Own Latent (BYOL)** self-supervised learning to street-level imagery collected via the Mapillary API.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Pipeline Stages](#pipeline-stages)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Outputs](#outputs)
- [License](#license)

---

## Overview

*Baukultur* (building culture) captures the qualitative character of the built environment. This project makes that character **measurable** by training a model to produce consistent, geometry-aware image embeddings from geographically close street-level photo pairs.

Key design decisions:

| Decision | Rationale |
|---|---|
| **BYOL (no negatives)** | Avoids mode collapse without needing large batches of hard negatives |
| **Frozen DINOv3 ViT-H/16+ backbone** | Reuses 1.2 billion-image pretrained features; keeps VRAM manageable |
| **Dual LoRA adapters** (online + target) | Fine-tunes only ~1% of parameters while sharing one frozen backbone |
| **Geographic positive pairs** | Images 5–15 m apart share architectural context; no manual labels needed |
| **H3 hexagonal grid** | Systematic, gapless coverage of the study area (Switzerland, res-7 ≈ 50 m) |
| **Conservative augmentation** | Crops 70–100 % scale preserves façade and street detail critical for Baukultur |

---

## Architecture

```
                         ┌──────────────────────────────────────────────────────┐
                         │              Shared frozen backbone                  │
                         │         DINOv3  ViT-H/16+  (1280-d CLS token)       │
                         └──────────────────┬──────────────────┬───────────────┘
                                            │                  │
                          LoRA-online (Δθ)  │                  │  LoRA-target  (Δξ)
                                            ▼                  ▼
                                    ┌───────────┐      ┌───────────┐
                                    │ Projector │      │ Projector │
                                    │ (MLP 2048)│      │ (MLP 2048)│
                                    └─────┬─────┘      └─────┬─────┘
                                          │                  │
                                    ┌─────▼─────┐           z'  (stop-grad)
                                    │ Predictor │
                                    │ (MLP 2048)│
                                    └─────┬─────┘
                                          │
                                          q
                                          │
                              Loss = 2 − 2 · ⟨q, z'⟩        (symmetric)
```

**EMA update (target network):**
```
ξ  ←  τ · ξ  +  (1 − τ) · θ          τ cosine-annealed 0.996 → 1.0
```

---

## Pipeline Stages

### Stage 1 — Data Ingestion (`data/ingestion.py` + `orchestrator.py`)

1. **H3 grid generation** — Tessellates the study area into hexagonal cells (resolution 7, ≈ 50 m edge length).
2. **Mapillary scraping** — For each cell, queries the Mapillary GraphQL API for street-level images.
3. **Positive-pair mining** — Within each image sequence, selects pairs separated by 5–15 m.
4. **Async download** — Parallel downloads with configurable rate limits (default: 10 API requests, 50 concurrent image fetches).
5. **WebDataset packing** — Pairs are packed into sharded TAR files (default 1 GB / shard). Each sample contains `img1.jpg`, `img2.jpg`, and a `meta.json` with geographic metadata.

### Stage 2 — Training (`train.py`)

1. Loads positive-pair shards from the local cache.
2. Applies random crop (70–100 %), colour jitter, and optional grayscale augmentation independently to each view.
3. Runs the **online** forward pass (`encoder → projector → predictor`) and the **target** forward pass (`encoder → projector`, no gradient).
4. Computes symmetric BYOL loss and back-propagates through the online network only.
5. Updates target parameters via EMA after each gradient step.
6. Logs loss, embedding standard deviation, and EMA momentum τ to **Weights & Biases**.
7. Saves LoRA adapter weights and aggregator state at the end of each epoch.

### Stage 3 — Replay & Cleanup (`orchestrator.py`)

1. Trained shards are moved to a FIFO replay buffer (capped at 750 GB by default).
2. The orchestrator triggers a new training run once the cache reaches 90 % capacity.
3. Continues until all H3 cells in the study area have been processed.

---

## Installation

### Prerequisites

- CUDA-capable GPU (tested on CUDA 12.4)
- [Conda](https://docs.conda.io/en/latest/miniconda.html)
- Mapillary API access token

### Environment setup

```bash
conda env create -f environment.yml
conda activate landscape_signatures
```

Key dependencies installed by the environment:

| Package | Role |
|---|---|
| `torch` + `torchvision` | Deep learning framework (CUDA 12.4 build) |
| `peft` | LoRA adapter injection |
| `webdataset` | Streaming TAR-shard dataloading |
| `h3` | Hexagonal grid indexing |
| `aiohttp` | Async Mapillary downloads |
| `wandb` | Experiment tracking |
| `xformers` / `flash-attn` | Memory-efficient attention |

---

## Configuration

All hyperparameters are controlled by `config.yaml`. Key sections:

```yaml
data:
  grid_resolution: 7          # H3 resolution (≈ 50 m cells)
  shard_size_gb: 1.0          # Target TAR shard size
  geo_pair_radius_m: 10       # Max distance (m) between positive-pair images
  geo_pair_pool: 512          # Mining pool size for pair diversity
  image_dims: [4096, 2048]    # Raw download resolution

model:
  encoder: dinov3_vith16plus  # Frozen ViT backbone
  embed_dim: 1280             # CLS token dimensionality
  lora_r: 16                  # LoRA rank
  lora_alpha: 32              # LoRA scaling factor

byol:
  tau_start: 0.996            # Initial EMA momentum
  tau_end: 1.0                # Final EMA momentum
  hidden_dim: 2048            # Projector / predictor hidden size
  output_dim: 1280            # Signature dimensionality

training:
  micro_batch: 16             # Per-GPU batch size
  grad_accum_steps: 16        # Effective batch = micro_batch × grad_accum (256)
  lr: 5.0e-4                  # AdamW learning rate
  epochs: 10                  # Training epochs per orchestration cycle
  grad_clip: 1.0              # Gradient clipping norm

orchestration:
  max_cache_gb: 750           # Disk cap for the shard cache
  train_trigger_pct: 0.90     # Cache fill fraction that starts training
```

---

## Usage

### 1. Set your Mapillary API token

```bash
export MAPILLARY_TOKEN="<your_token>"
```

### 2. Run the full pipeline

```bash
python orchestrator.py
```

The orchestrator manages the full lifecycle:

```
ingest H3 cells → pack WebDataset shards → train BYOL → move shards to replay
      └─────────────────── repeat until grid exhausted ──────────────────────┘
```

### 3. Train only (from existing shards)

```bash
python train.py
```

---

## Outputs

| Path | Content |
|---|---|
| `models/weights/checkpoints/lora_latest/` | Trained LoRA adapter weights (online + target) |
| `models/weights/agg_online_latest.pt` | Online aggregator head (projector + predictor) |
| `models/weights/agg_target_latest.pt` | Target aggregator head (projector only) |
| `models/weights/optimizer_latest.pt` | AdamW optimizer state (for resumption) |
| W&B dashboard | Loss, embedding std, τ per step; average loss per epoch |

**Landscape signatures** are 1280-dimensional L2-normalized vectors extracted from the online network (encoder → projector, without the predictor) after training.

---

## License

[MIT](LICENSE) — Copyright 2026 JustinBooz
