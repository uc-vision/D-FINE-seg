from __future__ import annotations

import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from loguru import logger

from d_fine.config import TrainConfig
from d_fine.validation.visualization import vis_one_box


def run_view_samples(train_config: TrainConfig, num_batches: int = 100):
  """Visualize training samples with augmentations applied."""

  loader_builder = train_config.dataset.create_loader(
    batch_size=train_config.batch_size,
    num_workers=min(train_config.num_workers, 1),  # Keep it simple for visualization
  )
  train_loader, _, _ = loader_builder.build_dataloaders(distributed=False)
  dataset = train_loader.dataset

  alpha = 0.4
  logger.info("Starting visualization. Press any key for next batch, 'q' to quit.")

  for batch_idx, (images, targets, _) in enumerate(train_loader):
    if batch_idx >= num_batches:
      break

    for i in range(images.shape[0]):
      img = dataset.denormalize(images[i]).copy()
      # Convert RGB to BGR for OpenCV
      img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

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
        vis_one_box(img, box, int(cid), mode="gt", label_to_name=dataset.label_to_name)

      cv2.imshow("Training Sample (Augmented)", img)
      key = cv2.waitKey(0) & 0xFF
      if key == ord("q"):
        cv2.destroyAllWindows()
        return

  cv2.destroyAllWindows()


def main(train_config: TrainConfig):
  """Entry point for viewing samples."""
  run_view_samples(train_config)
