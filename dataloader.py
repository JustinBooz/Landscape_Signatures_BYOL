"""
WebDataset-based dataloader using pre-mined JPEG pairs.
This optimizes training by moving the expensive spatial mining offline
and using compressed shards to prevent IO-asphyxiation.
"""

import torch
from torchvision.transforms import v2 as T
from torch.utils.data import IterableDataset, DataLoader
import webdataset as wds
import os
import glob

class GeoPairDataset(IterableDataset):
    def __init__(
        self,
        dataset: wds.WebDataset,
        config: dict,
    ):
        super().__init__()
        self.dataset = dataset
        
        train_cfg = config["training"]
        aug_scale = train_cfg.get("resized_crop_scale", [0.7, 1.0])
        cj_cfg = train_cfg.get("color_jitter", {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.05})

        # GENTLE AUGMENTATION: Essential to preserve "Baukultur" architectural signatures
        self.augment_pipeline = T.Compose([
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.RandomResizedCrop(size=(518, 518), scale=tuple(aug_scale), antialias=True),
            T.RandomHorizontalFlip(p=0.5), # Orientation generic for VPR unless one-way streets
            T.RandomApply([T.ColorJitter(**cj_cfg)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.count = 0

    def __iter__(self):
        """
        Yields (img1, img2) pairs directly from compressed JPEG shards.
        """
        # I/O OPTIMIZATION: Decode JPEGs on the fly in worker threads
        stream = self.dataset.decode("torchrgb8").to_tuple("img1.jpg", "img2.jpg")
        
        for img1, img2 in stream:
            try:
                # Augment
                view1 = self.augment_pipeline(img1)
                view2 = self.augment_pipeline(img2)

                self.count += 1
                if self.count % 100 == 0:
                    print(f"[Dataloader] Yielded {self.count} samples...", flush=True)

                yield view1, view2
            except Exception as e:
                if self.count < 5:
                    print(f"[Dataloader Error] {e}", flush=True)
                continue

def get_dataloader(
    shards_pattern: list or str,
    config: dict,
    batch_size: int = 32,
    num_workers: int = 8,
    **kwargs
) -> DataLoader:
    if isinstance(shards_pattern, list):
        pattern = shards_pattern
    else:
        shards_dir = os.path.dirname(shards_pattern)
        pattern = os.path.join(shards_dir, "*.tar")
    
    # Extract the buffer size from config, defaulting to 20000 if missing
    train_cfg = config.get("training", {})
    shuffle_buf = train_cfg.get("shuffle_buffer", 20000)
    
    # DECORRELATION: Explicit shuffle buffer before resampling
    dataset = wds.WebDataset(pattern, shardshuffle=True, resampled=True)
    dataset = dataset.shuffle(shuffle_buf) # Now dynamically sized!
    ds = GeoPairDataset(dataset, config)

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers, # Make sure this is high (e.g., 8 or 12)
        pin_memory=True,
        drop_last=True,
    )
