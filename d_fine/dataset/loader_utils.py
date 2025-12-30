from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from loguru import logger

from d_fine.config import MultiscaleConfig
from d_fine.dataset.dataset import Dataset, ProcessedSample


def collate_fn(batch: list[ProcessedSample]) -> tuple[torch.Tensor, list[dict], tuple]:
  images = torch.stack([s.image for s in batch])
  targets = [
    {
      "labels": s.labels,
      "boxes": s.boxes,
      "masks": s.masks,
      "paths": s.paths,
      "orig_size": s.orig_size,
    }
    for s in batch
  ]
  paths = tuple(p for s in batch for p in s.paths)
  return images, targets, paths


def train_collate_fn(
  batch: list[ProcessedSample], multiscale_config: Optional[MultiscaleConfig] = None
) -> tuple[torch.Tensor, list[dict], tuple]:
  images, targets, paths = collate_fn(batch)

  if multiscale_config and multiscale_config.prob > 0:
    import random

    if random.random() < multiscale_config.prob:
      offset = random.choice(multiscale_config.offset_range)
      size = images.shape[-1] + offset * multiscale_config.step_size
      images = F.interpolate(images, size=(size, size), mode="bilinear", align_corners=False)

  return images, targets, paths


def log_debug_images_from_batch(
  images: torch.Tensor,
  targets: list[dict],
  dataset: Dataset,
  logger_obj: Optional[any] = None,
  num_images: int = 5,
):
  """Log debug images from a batch to the logger."""
  from d_fine.validation.visualization import visualize_batch

  vis_images = visualize_batch(
    images[:num_images], targets[:num_images], dataset.label_to_name, dataset=dataset
  )

  if logger_obj:
    logger_obj.log_images("debug", "batch", vis_images)
