from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch

from d_fine.config import MultiscaleConfig
from d_fine.dataset.dataset import Dataset, CompactMasks, ProcessedSample
from d_fine.validation.visualization import vis_one_box
from d_fine.dl.logging import Logger


def collate_fn(
  batch: list[ProcessedSample], expand: bool = True
) -> tuple[torch.Tensor, list[dict], list[tuple[Path, ...]]]:
  images, targets, img_paths = [], [], []
  for item in batch:
    images.append(item.image)
    targets.append(
      {
        "labels": item.labels,
        "boxes": item.boxes,
        "masks": item.masks,  # CompactMasks
        "orig_size": item.orig_size,
      }
    )
    img_paths.append(item.paths)

  if expand:
    for t in targets:
      t["masks"] = t["masks"].expand()

  return torch.stack(images, dim=0), targets, img_paths


def apply_multiscale(
  images: torch.Tensor, targets: list[dict], config: MultiscaleConfig
) -> torch.Tensor:
  if random.random() < config.prob:
    offset = random.choice(config.offset_range) * config.step_size
    new_h, new_w = images.shape[2] + offset, images.shape[3] + offset

    images = torch.nn.functional.interpolate(
      images, size=(new_h, new_w), mode="bilinear", align_corners=False
    )

    for t in targets:
      m = t["masks"]
      # Strict CompactMasks handling
      id_map_float = m.id_map.float()[None, None, ...]
      id_map_interp = torch.nn.functional.interpolate(
        id_map_float, size=(new_h, new_w), mode="nearest"
      )
      t["masks"] = CompactMasks(
        id_map=id_map_interp.view(new_h, new_w).to(torch.uint16), indices=m.indices
      )
  return images


def train_collate_fn(
  batch: list[ProcessedSample], multiscale_config: MultiscaleConfig
) -> tuple[torch.Tensor, list[dict], list[tuple[Path, ...]]]:
  # Don't expand yet so we can interpolate efficiently
  images, targets, img_paths = collate_fn(batch, expand=False)

  images = apply_multiscale(images, targets, multiscale_config)

  # Final expansion to bitmasks for all targets
  for t in targets:
    t["masks"] = t["masks"].expand()

  return images, targets, img_paths


def log_debug_images_from_batch(
  images: torch.Tensor,
  targets: list[dict],
  dataset: Dataset,
  results_logger: Logger,
  num_images: int,
  alpha: float = 0.4,
) -> None:
  debug_imgs = []
  for i in range(min(images.shape[0], num_images)):
    img = dataset.denormalize(images[i]).copy()

    target = targets[i]
    masks = target.get("masks")
    if masks is not None and masks.numel() > 0:
      for mask in masks.cpu().numpy():
        if mask.max() == 0:
          continue
        overlay = np.zeros_like(img)
        overlay[:] = (0, 255, 0)
        m = mask.astype(bool)
        img[m] = cv2.addWeighted(img[m], 1 - alpha, overlay[m], alpha, 0)

    for box, cid in zip(
      target["boxes"].cpu().numpy().astype(np.float32), target["labels"].cpu().numpy()
    ):
      vis_one_box(img, box, int(cid), "gt", dataset.label_to_name)

    debug_imgs.append(img)

  results_logger.log_images("train", "debug_images", debug_imgs)
