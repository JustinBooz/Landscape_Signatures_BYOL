import os
import copy
import types
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("WARNING: peft not installed — LoRA adapters will not be injected.")

# DINOv3 implementation remains mostly stock, but we inject LoRA adapters.


# ---------------------------------------------------------------------------
# LoRA injection helpers
# ---------------------------------------------------------------------------

def _get_lora_target_modules(model):
    targets = set()
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        base = name.split(".")[-1]
        if base in ("q", "v", "query", "value", "qkv"):
            targets.add(base)
    if "q" in targets and "v" in targets:
        return ["q", "v"]
    if "query" in targets and "value" in targets:
        return ["query", "value"]
    return ["qkv"]

def inject_lora(backbone, r: int = 16, lora_alpha: int = 32, target_modules=None, adapter_name="default"):
    if not PEFT_AVAILABLE:
        print("peft not available — skipping LoRA injection.")
        return backbone

    if target_modules is None:
        target_modules = _get_lora_target_modules(backbone)

    print(f"Injecting LoRA adapters into: {target_modules}  (r={r}, α={lora_alpha})")
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
    )
    peft_model = get_peft_model(backbone, config, adapter_name=adapter_name)
    peft_model.print_trainable_parameters()
    return peft_model

# ---------------------------------------------------------------------------
# VisionEncoder
# ---------------------------------------------------------------------------

class VisionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        enc_cfg = config["model"]["encoder"]
        lora_cfg = config.get("lora", {})

        model_name = enc_cfg["name"]
        self.freeze = enc_cfg.get("freeze_weights", True)
        # Strict constraint for bfloat16 to avoid VRAM OOM
        self.dtype = torch.bfloat16

        local_weights_path = enc_cfg.get("local_weights_path")
        self.patch_size = enc_cfg.get("patch_size", 16)   # ViT-H/16+ default

        hub_repo = "facebookresearch/dinov3"
        hub_model = "dinov3_vit7b16" if "7b" in model_name.lower() else model_name

        if not local_weights_path or not os.path.exists(local_weights_path):
            raise FileNotFoundError(
                f"ViT-H/16+ weights not found at '{local_weights_path}'. "
            )
            
        print(f"Loading {hub_model} structure (pretrained=False) ...")
        backbone = torch.hub.load(hub_repo, hub_model, pretrained=False)
        print(f"Injecting weights from: {local_weights_path} ...")
        state_dict = torch.load(local_weights_path, map_location="cpu", weights_only=True)
        backbone.load_state_dict(state_dict)

        if self.freeze:
            for param in backbone.parameters():
                param.requires_grad = False
            backbone.eval()

        self.backbone = inject_lora(
            backbone,
            r=lora_cfg.get("r", 16),
            lora_alpha=lora_cfg.get("lora_alpha", 32),
            target_modules=lora_cfg.get("target_modules", None),
            adapter_name="online"
        )
        
        # Create 'target' copy using the SAME configuration (including discovered modules)
        if PEFT_AVAILABLE:
            online_config = self.backbone.peft_config["online"]
            self.backbone.add_adapter("target", online_config)
            # Initialize target with online weights initially
            self.backbone.set_adapter("target")
            # We'll rely on EMA update to sync them later
            self.backbone.set_adapter("online")
            
            # TASK 5: Sever Autograd for the Target LoRA
            for n, p in self.backbone.named_parameters():
                if ".target." in n:
                    p.requires_grad = False

        # DINOv3 already uses SDPA natively in its blocks.
        # We don't need to monkey-patch if we pass rope correctly.

        self.backbone = self.backbone.to(self.dtype)
        self._gradient_checkpointing = False

    def enable_gradient_checkpointing(self):
        self._gradient_checkpointing = True
        print("Gradient checkpointing ENABLED for VisionEncoder.")

    def forward(self, x: torch.Tensor, use_target: bool = False):
        if x.dtype != self.dtype:
            x = x.to(self.dtype)

        # Switch adapter context instantly in VRAM
        if PEFT_AVAILABLE:
            self.backbone.set_adapter("target" if use_target else "online")

        _base = self.backbone.model if hasattr(self.backbone, "model") else self.backbone

        # DINOv3 prepare_tokens_with_masks returns (tokens, (H, W))
        features, (H, W) = _base.prepare_tokens_with_masks(x)
        
        # Generate RoPE sincos for the current H, W
        rope_sincos = _base.rope_embed(H=H, W=W) if hasattr(_base, "rope_embed") else None

        for blk in _base.blocks:
            if self._gradient_checkpointing and self.training:
                features = activation_checkpoint(
                    blk, features, rope_sincos, use_reentrant=False
                )
            else:
                features = blk(features, rope_sincos)

        # DINOv3 prepends the [CLS] token at index 0.
        # features shape is [B, N_tokens, D]
        cls_token = features[:, 0, :]  # Extract only the CLS token -> [B, D]
        return cls_token
