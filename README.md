# Landscape_Signatures_BYOL
Creating quantifiable metrics of Baukultur with VPR
Landscape_Signatures_BYOL/
├── README.md                 # Minimal existing README
├── LICENSE                   # MIT License (Copyright 2026 JustinBooz)
├── config.yaml              # Master hyperparameter configuration
├── environment.yml          # Conda environment dependencies
├── orchestrator.py          # Pipeline orchestration & lifecycle management
├── train.py                 # BYOL training loop with EMA & gradient accumulation
├── models/
│   ├── encoder.py          # VisionEncoder with LoRA-adapted DINOv3
│   └── aggregator.py       # AggregatorHead: GeM pooling + Projector + Predictor
├── data/
│   ├── ingestion.py        # Mapillary API scraper + WebDataset packing
│   └── dataloader.py       # WebDataset-based training dataloader
└── .git/                    # Git repository


2. This pipeline specifically uses BYOL to extract Landscape Signatures: high-dimensional architectural and urban feature embeddings from street-level imagery.

3. What This Pipeline Does (Step-by-Step)
Phase 1: Data Ingestion (orchestrator.py → data/ingestion.py)

Geographic Coverage: Uses H3 hexagonal grid (resolution 7) covering Switzerland at ~50m resolution
Mapillary API Scraping: For each grid cell, queries Mapillary API for street-level images
Spatial Mining: Within each sequence, finds image pairs 5-15m apart (positive pairs for BYOL)
Batch Processing:
Async/parallel downloads with rate limiting (10 API requests, 50 concurrent downloads)
~200 image pairs per geographic cell
WebDataset Packing: Compresses pairs into sharded TAR files (1 GB per shard default)
Each shard contains: img1.jpg, img2.jpg, meta.json with location metadata

Phase 2: Training (train.py)

Encoder Setup:
Loads frozen DINOv3 ViT-H/16+ backbone (pretrained on 1.2B images)
Injects dual LoRA adapters ("online" + "target") for efficient fine-tuning
Only 2 adapters share VRAM; same backbone prevents memory doubling
BYOL Architecture:

Online Network: Encoder(LoRA-online) → Projector → Predictor → L2-normalized prediction
Target Network: Encoder(LoRA-target) → Projector → L2-normalized projection (no predictor, no grad)
Training Loop:

Loads positive image pairs from WebDataset shards
Forward pass with gentle augmentation (crop 70-100%, color jitter, grayscale)
Loss: Symmetric negative cosine similarity: L = 2 - 2*<q(z1), z'(x2)>
EMA Update: Target parameters updated with exponential moving average
τ cosine-annealed from 0.996 → 1.0 over all steps
Variance Monitoring: Tracks embedding std to detect representation collapse
Checkpointing: Saves LoRA adapters + aggregator weights every epoch

Configuration Highlights:

Effective batch size: 16 micro-batch × 16 gradient accumulation = 256 samples
Learning rate: 5e-4 (AdamW)
Training epochs: 10
Gradient clipping: 1.0
bfloat16 precision for efficiency
Phase 3: Cleanup & Replay (orchestrator.py)
FIFO Management: Trained shards moved to replay buffer (750 GB cap)
Continuous Learning: Enables replay of older data while ingesting new geographic regions
Termination: Pipeline completes when entire H3 grid is processed

4. Input/Output of the Pipeline
Input:

Geographic Grid: H3 coordinates covering Switzerland (resolution 7)
Street Imagery: ~4096×2048px thumb images from Mapillary (10K+ coverage)
API Access: Mapillary GraphQL API token

Output:

Landscape Signatures: 1280-dimensional L2-normalized embeddings per image pair
Checkpoints:
models/weights/checkpoints/lora_latest/ — Trained LoRA adapters
agg_online_latest.pt / agg_target_latest.pt — Aggregator weights
optimizer_latest.pt — Optimizer state for resumption
Training Metrics (W&B logging):
Per-step: loss, embedding_std, τ (EMA momentum)
Per-epoch: average loss
5. Key Parameters & Configurations
Category	Parameter	Value	Purpose
Data	Grid size	50m	Geographic resolution for data collection
Shard size	1.0 GB	WebDataset shard file size
Image dims	4096×2048	Input resolution (cropped to 518×518 in training)
Geo pair radius	10m	Max distance for positive pairs
Geo pair pool	512	Mining pool size for diversity
Model	Encoder	dinov3_vith16plus	DINOv3 ViT-Huge/16+ frozen backbone
Embed dim	1280	ViT-H output dimension
Patch size	16	Vision Transformer patch granularity
LoRA	r	16	LoRA rank (adapter bottleneck)
α (alpha)	32	LoRA scaling factor
Target modules	auto-detected	Query/Value linear layers in ViT blocks
BYOL	τ_start	0.996	EMA momentum initialization
τ_end	1.0	EMA momentum target
Hidden dim	2048	MLP projector/predictor hidden dimension
Output dim	1280	Final signature dimensionality
GeM exponent	3.0	Generalized mean pooling (not currently used)
Training	Batch size	16	Micro-batch
Grad accum	16	Accumulation steps (effective batch = 256)
Learning rate	5e-4	AdamW optimizer
Epochs	10	Total training epochs
Augmentation scale	0.7-1.0	Conservative crop to preserve architecture
Collapse threshold	0.015	Embedding std alert
Orchestration	Max cache	750 GB	Disk usage cap before training trigger
Training trigger	90% capacity	When to start training phase

6. Dependencies

Conda Environment (environment.yml):

YAML
Python 3.11
PyTorch + CUDA 12.4
torchvision
NumPy, Pandas, GeoPandas, Shapely
aiohttp, requests (async/sync HTTP)
webdataset (TAR-based datasets)
h3 (hexagonal indexing)
xformers, flash-attn (optimization)
wandb (experiment tracking)
Key Libraries:

PEFT (imported in encoder.py): For LoRA adapter injection
Torch.nn.utils.clip_grad_norm_: Gradient clipping
torch.utils.checkpoint: Gradient checkpointing for memory efficiency

License
MIT License (Copyright 2026 JustinBooz)
