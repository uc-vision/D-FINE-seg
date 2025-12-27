from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader

from typing import Any
from d_fine.dataset.base import BaseDataset
from d_fine.dataset.mask_types import CompactMasks
from d_fine.utils import vis_one_box


def collate_fn(batch: list[Any], expand: bool = True) -> tuple[torch.Tensor | None, list[dict[str, Any]] | None, list[Path] | None]:
    if None in batch:
        return None, None, None

    images, targets, img_paths = [], [], []
    for item in batch:
        images.append(item[0])
        targets.append({
            "labels": item[1],
            "boxes": item[2],
            "masks": item[3], # CompactMasks
            "orig_size": item[5],
        })
        img_paths.append(item[4])

    if expand:
        for t in targets:
            t["masks"] = t["masks"].expand()

    return torch.stack(images, dim=0), targets, img_paths


def train_collate_fn(batch: list[Any], multiscale_prob: float) -> tuple[torch.Tensor, list[dict[str, Any]], list[Path]]:
    # Don't expand yet so we can interpolate efficiently
    images, targets, img_paths = collate_fn(batch, expand=False)

    if random.random() < multiscale_prob:
        offset = random.choice([-2, -1, 1, 2]) * 32
        new_h, new_w = images.shape[2] + offset, images.shape[3] + offset

        images = torch.nn.functional.interpolate(
            images, size=(new_h, new_w), mode="bilinear", align_corners=False
        )

        for t in targets:
            m = t["masks"]
            # Strict CompactMasks handling
            id_map_float = m.id_map.float()[None, None, ...]
            id_map_interp = torch.nn.functional.interpolate(id_map_float, size=(new_h, new_w), mode="nearest")
            t["masks"] = CompactMasks(
                id_map=id_map_interp.view(new_h, new_w).to(torch.uint16),
                indices=m.indices
            )

    # Final expansion to bitmasks for all targets
    for t in targets:
        t["masks"] = t["masks"].expand()
            
    return images, targets, img_paths


def log_debug_images_from_batch(
    images: torch.Tensor,
    targets: list[dict],
    dataset: BaseDataset,
    wandb_logger,
    num_images: int = 5,
) -> None:
    if wandb_logger is None or not dataset.debug_img_processing or images is None:
        return
    
    alpha = 0.4
    debug_imgs = []
    for i in range(min(images.shape[0], num_images)):
        img = dataset._img_config.denormalize(images[i]).transpose(1, 2, 0)
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8).copy()
        
        target = targets[i]
        masks = target.get("masks")
        if masks is not None and masks.numel() > 0:
            for mask in masks.cpu().numpy():
                if mask.max() == 0: continue
                overlay = np.zeros_like(img); overlay[:] = (0, 255, 0)
                m = mask.astype(bool)
                img[m] = cv2.addWeighted(img[m], 1 - alpha, overlay[m], alpha, 0)
        
        for box, cid in zip(target["boxes"].cpu().numpy().astype(int), target["labels"].cpu().numpy()):
            vis_one_box(img, box, cid, "gt", dataset.label_to_name)
        
        debug_imgs.append(img)
    
    if debug_imgs:
        wandb_logger.log({"train/debug_images": [wandb_logger.Image(img, caption=f"train_debug_{i}") for i, img in enumerate(debug_imgs)]})
