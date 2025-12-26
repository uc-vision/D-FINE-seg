from __future__ import annotations

import random

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from d_fine.config import Mode
from d_fine.dataset.base import BaseDataset
from d_fine.utils import seed_worker, vis_one_box


def build_dataloader_impl(
    dataset: BaseDataset,
    batch_size: int,
    num_workers: int,
    collate_fn,
    shuffle: bool = False,
    distributed: bool = False,
) -> tuple[DataLoader, object]:
    sampler = None
    shuffle_flag = shuffle

    if distributed:
        sampler = DistributedSampler(
            dataset, shuffle=(shuffle and dataset.mode == Mode.TRAIN), drop_last=False
        )
        shuffle_flag = False

    dataloader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "shuffle": shuffle_flag,
        "sampler": sampler,
        "collate_fn": collate_fn,
        "worker_init_fn": seed_worker,
        "pin_memory": True,
    }
    # prefetch_factor only works with num_workers > 0
    if num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = 4
    
    dataloader = DataLoader(**dataloader_kwargs)

    train_sampler = sampler if dataset.mode == Mode.TRAIN else None
    return dataloader, train_sampler


def collate_fn(batch: list) -> tuple[torch.Tensor, list[dict], list[Path]]:
    if None in batch:
        return None, None, None
    images = []
    targets = []
    img_paths = []

    for item in batch:
        target_dict = {
            "labels": item[1],
            "boxes": item[2],
            "masks": item[3],
            "orig_size": item[5],
        }
        images.append(item[0])
        targets.append(target_dict)
        img_paths.append(item[4])

    images = torch.stack(images, dim=0)
    return images, targets, img_paths


def train_collate_fn(batch: list, multiscale_prob: float) -> tuple[torch.Tensor, list[dict], list[Path]]:
    images, targets, img_paths = collate_fn(batch)

    if random.random() < multiscale_prob:
        offset = random.choice([-2, -1, 1, 2]) * 32
        new_h = images.shape[2] + offset
        new_w = images.shape[3] + offset

        images = torch.nn.functional.interpolate(
            images, size=(new_h, new_w), mode="bilinear", align_corners=False
        )

        for t in targets:
            m = t["masks"]
            if m.numel() == 0:
                continue
            m = m.unsqueeze(1).float()
            m = torch.nn.functional.interpolate(m, size=(new_h, new_w), mode="nearest")
            t["masks"] = (m.squeeze(1) > 0.5).to(torch.uint8)
    return images, targets, img_paths


def log_debug_images_from_batch(
    images: torch.Tensor,
    targets: list[dict],
    dataset: BaseDataset,
    wandb_logger,
    num_images: int = 5,
) -> None:
    """Generate and log debug images from a training batch.
    
    Args:
        images: Batch of images tensor of shape (B, C, H, W)
        targets: List of target dictionaries
        dataset: Dataset instance (for accessing config and label_to_name)
        wandb_logger: wandb module (or None if logging is disabled)
        num_images: Number of debug images to generate and log
    """
    if wandb_logger is None:
        return
    
    if not dataset.debug_img_processing:
        return
    
    if images is None:
        return
    
    debug_imgs = []
    for i in range(min(images.shape[0], num_images)):
        image = images[i]
        target = targets[i]
        
        image_np = dataset._img_config.denormalize(image)
        image_np = np.transpose(image_np, (1, 2, 0))
        image_np = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)
        image_np = np.ascontiguousarray(image_np)
        
        boxes = target["boxes"].cpu().numpy().astype(int)
        labels = target["labels"].cpu().numpy()
        masks = target.get("masks")
        
        if masks is not None and masks.numel() > 0:
            mnp = masks.cpu().numpy()
            for k in range(mnp.shape[0]):
                mask = mnp[k].astype(np.uint8)
                if mask.max() == 0:
                    continue
                overlay = np.zeros_like(image_np, dtype=np.uint8)
                overlay[:] = (0, 255, 0)
                mask_bool = mask.astype(bool)
                image_np[mask_bool] = cv2.addWeighted(
                    image_np[mask_bool], 0.6, overlay[mask_bool], 0.4, 0
                ) 
        
        for box, class_id in zip(boxes, labels):
            vis_one_box(image_np, box, class_id, mode="gt", label_to_name=dataset.label_to_name)
        
        debug_imgs.append(image_np)
    
    if debug_imgs:
        wandb_images = [
            wandb_logger.Image(img, caption=f"train_debug_{i}")
            for i, img in enumerate(debug_imgs)
        ]
        wandb_logger.log({"train/debug_images": wandb_images})

