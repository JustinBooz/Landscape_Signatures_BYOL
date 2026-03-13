"""
BYOL Training Loop — Landscape Signatures

Network:
  Online (θ): Encoder(LoRA) → AggregatorHead(online=True)  [GeM → Projector → Predictor]
  Target (ξ): Encoder(LoRA) → AggregatorHead(online=False) [GeM → Projector]     (no grad)

  The Target encoder SHARES the frozen ViT-H/16+ backbone with the Online encoder.
  Only the lightweight LoRA adapters are separate (swapped in/out during forward).
  This halves VRAM usage compared to a full deepcopy.

Loss:
  Symmetric negative cosine similarity:
  L = 2 - 2 * <q(z1), z'(x2)> / (||q(z1)|| * ||z'(x2)||)

EMA update (wrapped in @torch.no_grad()):
  ξ ← τ·ξ + (1-τ)·θ
  τ cosine-annealed from tau_start → tau_end over all training steps.
"""

import os
import glob
import math
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("WARNING: wandb not installed — telemetry disabled.")

from models.encoder import VisionEncoder
from models.aggregator import AggregatorHead
from data.dataloader import get_dataloader


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# EMA utilities
# ---------------------------------------------------------------------------

@torch.no_grad()
def ema_update_target(encoder, online_agg, target_agg, tau: float):
    """
    Update Target parameters via Exponential Moving Average.
    Now handles internal LoRA adapter synchronization.
    """
    # 1. Update LoRA adapters (Target ← τ·Target + (1-τ)·Online)
    # peft stores adapters as 'online' and 'target'
    online_params = {n: p for n, p in encoder.backbone.named_parameters() if ".online." in n}
    target_params = {n: p for n, p in encoder.backbone.named_parameters() if ".target." in n}

    for name_online, p_online in online_params.items():
        # Match target param by replacing .online. with .target.
        name_target = name_online.replace(".online.", ".target.")
        if name_target in target_params:
            target_params[name_target].mul_(tau).add_((1.0 - tau) * p_online.detach())

    # --- Aggregator EMA ---
    online_agg_dict = dict(online_agg.named_parameters())
    for name, p_target in target_agg.named_parameters():
        if name in online_agg_dict:
            p_online = online_agg_dict[name]
            p_target.data.mul_(tau).add_((1.0 - tau) * p_online.detach())



def cosine_tau(step: int, total_steps: int,
               tau_start: float, tau_end: float) -> float:
    """Cosine schedule for EMA momentum τ from tau_start → tau_end."""
    progress = step / max(total_steps, 1)
    return tau_end - (tau_end - tau_start) * (1.0 + math.cos(math.pi * progress)) / 2.0


# ---------------------------------------------------------------------------
# BYOL loss
# ---------------------------------------------------------------------------

def byol_loss(q: torch.Tensor, z_prime: torch.Tensor) -> torch.Tensor:
    """
    Negative Cosine Similarity loss (inputs assumed L2-normalised by AggregatorHead):
    L = 2 - 2 * dot(q, z')
    """
    return 2.0 - 2.0 * (q * z_prime.detach()).sum(dim=-1).mean()


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

CKPT_DIR = "models/weights/checkpoints"


def _ckpt(name: str) -> str:
    return os.path.join(CKPT_DIR, name)


def save_checkpoints(encoder, online_agg, target_agg, optimizer, epoch: int):
    """Save LoRA adapters + aggregator weights."""
    os.makedirs(CKPT_DIR, exist_ok=True)
    # Save all adapters (online and target are both in encoder.backbone)
    encoder.backbone.save_pretrained(_ckpt("lora_latest"))
    torch.save(online_agg.state_dict(),   _ckpt("agg_online_latest.pt"))
    torch.save(target_agg.state_dict(),   _ckpt("agg_target_latest.pt"))
    torch.save(optimizer.state_dict(),    _ckpt("optimizer_latest.pt"))
    with open(_ckpt("epoch.txt"), "w") as f:
        f.write(str(epoch))
    print(f"Checkpoints saved (epoch {epoch}).")


def load_checkpoints(encoder, online_agg, target_agg, optimizer, device):
    """Resume from latest checkpoints."""
    start_epoch = 0
    epoch_file = _ckpt("epoch.txt")
    if not os.path.exists(epoch_file):
        return start_epoch

    try:
        with open(epoch_file) as f:
            start_epoch = int(f.read().strip()) + 1
        print(f"Resuming from epoch {start_epoch} ...")
        
        # Load all adapters back into the shared backbone
        encoder.backbone.load_adapter(_ckpt("lora_latest"), "online")
        encoder.backbone.load_adapter(_ckpt("lora_latest"), "target")

        online_agg.load_state_dict(torch.load(_ckpt("agg_online_latest.pt"), map_location=device))
        target_agg.load_state_dict(torch.load(_ckpt("agg_target_latest.pt"), map_location=device))
        optimizer.load_state_dict(torch.load(_ckpt("optimizer_latest.pt"),   map_location=device))
    except Exception as e:
        print(f"Warning: checkpoint loading failed ({e}). Starting from scratch.")
        start_epoch = 0

    return start_epoch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Blackwell / bfloat16 performance flags
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    byol_cfg     = config["byol"]
    train_cfg    = config["training"]
    wandb_cfg    = config.get("wandb", {})

    accum_steps  = int(train_cfg.get("gradient_accumulation_steps", 1))
    use_ckpt     = train_cfg.get("gradient_checkpointing", False)

    # ------------------------------------------------------------------ wandb
    if WANDB_AVAILABLE:
        wandb.init(
            project=wandb_cfg.get("project", "baukultur_vpr"),
            entity=wandb_cfg.get("entity") or None,
            mode="offline" if wandb_cfg.get("offline", False) else "online",
            config={
                "lr":             train_cfg["learning_rate"],
                "batch_size":     train_cfg["batch_size"],
                "effective_batch": train_cfg["batch_size"] * accum_steps,
                "lora_r":         config["lora"]["r"],
                "lora_alpha":     config["lora"]["lora_alpha"],
                "tau_start":      byol_cfg["tau_start"],
                "tau_end":        byol_cfg["tau_end"],
                "gem_p_init":     byol_cfg["gem_p_init"],
                "output_dim":     byol_cfg["output_dim"],
                "grad_accum":     accum_steps,
                "grad_checkpointing": use_ckpt,
            },
        )

    # ------------------------------------------------------------------ models
    print("Initialising Encoder (ViT-H/16+ with Dual LoRA Adapters) ...")
    encoder = VisionEncoder(config).to(device)

    if use_ckpt:
        encoder.enable_gradient_checkpointing()

    embed_dim = config["model"]["encoder"]["embed_dim"]

    online_agg = AggregatorHead(
        embed_dim=embed_dim,
        hidden_dim=byol_cfg["hidden_dim"],
        out_dim=byol_cfg["output_dim"],
        online=True,
    ).to(device)

    target_agg = AggregatorHead(
        embed_dim=embed_dim,
        hidden_dim=byol_cfg["hidden_dim"],
        out_dim=byol_cfg["output_dim"],
        online=False,
    ).to(device)

    # NEW: Sync teacher to student at initialization to prevent Step 0 collapse
    target_agg.load_state_dict(online_agg.state_dict(), strict=False)

    for p in target_agg.parameters():
        p.requires_grad = False
    
    # PEFT SAFETY: Sever autograd for all target parameters in the backbone
    for n, p in encoder.backbone.named_parameters():
        if ".target." in n:
            p.requires_grad = False

    # ---------------------------------------------------------------- dataloader
    shards_dir = config["data"]["output_shards_dir"]

    if not os.path.exists(shards_dir):
        raise RuntimeError(
            f"Shards directory '{shards_dir}' not found. "
            "Run data/ingestion.py first to download data."
        )

    # Auto-discover all available .tar shards (sorted for reproducibility)
    shard_files = sorted(glob.glob(os.path.join(shards_dir, "dataset-*.tar")))
    if not shard_files:
        raise RuntimeError(f"No .tar shards found in '{shards_dir}'.")
    print(f"Found {len(shard_files)} shards in '{shards_dir}'")

    print("Connecting positive-pair WebDataset stream ...")
    dataloader = get_dataloader(
        shards_pattern=shard_files,
        config=config,
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg["num_workers"],
    )

    # ---------------------------------------------------------------- training
    trainable_params = list(encoder.backbone.parameters()) + list(online_agg.parameters())
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.05)),
    )

    start_epoch = load_checkpoints(encoder, online_agg, target_agg, optimizer, device)
    total_epochs = train_cfg["num_epochs"]
    steps_per_epoch = 2000  # Virtual epoch size for WebDataset
    total_steps = (total_epochs - start_epoch) * steps_per_epoch
    global_step = start_epoch * steps_per_epoch
    
    grad_clip = train_cfg.get("gradient_clip", 1.0)
    std_thresh = train_cfg.get("embedding_std_threshold", 0.015)

    print(f"\n--- Commencing BYOL Training on {device} ---")
    print(f"    Micro-batch: {train_cfg['batch_size']}  ×  Accum steps: {accum_steps}  =  Effective batch: {train_cfg['batch_size'] * accum_steps}")
    print(f"    Gradient checkpointing: {use_ckpt}")
    print(f"    Epochs {start_epoch}→{total_epochs}\n")

    data_iter = iter(dataloader)

    for epoch in range(start_epoch, total_epochs):
        encoder.train()
        online_agg.train()
        target_agg.eval()

        epoch_loss = 0.0
        loop = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{total_epochs}")
        
        for step in loop:
            optimizer.zero_grad()
            accum_loss = 0.0
            accum_std  = 0.0

            for accum_i in range(accum_steps):
                # ---- Data ----
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    batch = next(data_iter)
                
                imgs1, imgs2 = batch
                imgs1 = imgs1.to(device, non_blocking=True)
                imgs2 = imgs2.to(device, non_blocking=True)

                # ---- Forward pass (bfloat16 autocast) ----
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    # Online branch: Predicts the target's projection
                    q1 = online_agg(encoder(imgs1, use_target=False))
                    q2 = online_agg(encoder(imgs2, use_target=False))

                    # Target branch: No grad, instant adapter switch
                    with torch.no_grad():
                        z1 = target_agg(encoder(imgs1, use_target=True)).detach()
                        z2 = target_agg(encoder(imgs2, use_target=True)).detach()

                    # PEFT SAFETY: Reset adapter state immediately
                    encoder.backbone.set_adapter("online")

                    # Symmetric BYOL loss
                    loss = (byol_loss(q1, z2) + byol_loss(q2, z1)) * 0.5 / accum_steps
                    loss.backward()

                    # VARIANCE TRACKING: Mean STD across batch for normalized predictions
                    # Collapse if std -> 0
                    with torch.no_grad():
                        std1 = q1.std(dim=0).mean()
                        std2 = q2.std(dim=0).mean()
                        batch_std = (std1 + std2) * 0.5
                        accum_std += batch_std.item() / accum_steps

                accum_loss += loss.item()
                del imgs1, imgs2, q1, q2, z1, z2, loss

            # ---- Optimiser step ----
            torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
            optimizer.step()
            torch.cuda.empty_cache()

            # ---- EMA update ----
            tau = cosine_tau(global_step, total_steps, byol_cfg["tau_start"], byol_cfg["tau_end"])
            ema_update_target(encoder, online_agg, target_agg, tau)

            epoch_loss  += accum_loss
            global_step += 1
            
            # Representation Collapse Warning
            if accum_std < std_thresh:
                print(f"\n[WARNING] Representations Collapsing! (std={accum_std:.4f} < {std_thresh})", flush=True)

            loop.set_postfix({"loss": f"{accum_loss:.4f}", "std": f"{accum_std:.4f}", "τ": f"{tau:.4f}"})

            if WANDB_AVAILABLE:
                wandb.log({
                    "loss": accum_loss,
                    "embedding_std": accum_std,
                    "tau": tau,
                    "epoch": epoch,
                    "global_step": global_step,
                }, step=global_step)

        avg_loss = epoch_loss / steps_per_epoch
        print(f"Epoch {epoch+1} | Avg BYOL Loss: {avg_loss:.6f}")

        if WANDB_AVAILABLE:
            wandb.log({"epoch_loss": avg_loss, "epoch": epoch + 1})

        save_checkpoints(
            encoder, online_agg, target_agg, optimizer, epoch
        )


if __name__ == "__main__":
    main()
