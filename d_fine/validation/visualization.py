from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from numpy.typing import NDArray

from typing import TYPE_CHECKING

from ..core.types import ImageResult

if TYPE_CHECKING:
  from ..dataset.base import Dataset


def draw_mask(image: np.ndarray, mask: np.ndarray, color=(255, 0, 0), alpha=0.5) -> np.ndarray:
  """Draw a mask on an image. Default color is Red (RGB)."""
  mask = mask.astype(bool)
  image[mask] = image[mask] * (1 - alpha) + np.array(color) * alpha
  return image.astype(np.uint8)


def vis_one_box(
  img: np.ndarray,
  box: NDArray[np.float32],
  label: int,
  score: float | None = None,
  label_to_name: dict[int, str] | None = None,
  color: tuple[int, int, int] | None = None,
  mode: str = "gt",
) -> None:
  """Visualize a single bounding box. Default colors for RGB: GT=Green, Pred=Red."""
  if color is None:
    color = (0, 255, 0) if mode == "gt" else (255, 0, 0)

  x1, y1, x2, y2 = map(int, box)
  cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
  name = label_to_name[label] if label_to_name else str(label)
  txt = f"{name} {score:.2f}" if score is not None else name
  cv2.putText(img, txt, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def visualize(
  img: NDArray[np.uint8], result: ImageResult, label_to_name: dict[int, str], mode: str = "pred"
) -> NDArray[np.uint8]:
  """Visualize detection and segmentation results on an image."""
  img = img.copy()

  # Draw masks first
  for m in result.masks:
    img = draw_mask(img, m.to_full_mask(*img.shape[:2]).cpu().numpy())

  # Draw boxes and labels on top
  for box, label, score in zip(result.boxes, result.labels, result.scores):
    vis_one_box(
      img,
      box.cpu().numpy(),
      int(label.item()),
      mode=mode,
      label_to_name=label_to_name,
      score=float(score.item()) if mode == "pred" else None,
    )
  return img


def visualize_instances(
  img: NDArray[np.uint8], instances: list[InstanceMask], label_to_name: dict[int, str]
) -> NDArray[np.uint8]:
  """Visualize a list of InstanceMask objects on an image."""
  from d_fine.core.types import ImageResult

  res = ImageResult.from_instances(instances, img.shape[:2])
  return visualize(img, res, label_to_name)


def visualize_results(
  img: NDArray[np.uint8],
  gt: ImageResult | None,
  pred: ImageResult | None,
  label_to_name: dict[int, str],
) -> NDArray[np.uint8]:
  """Visualize both ground truth and predictions on an image."""
  img = img.copy()
  if gt is not None:
    img = visualize(img, gt, label_to_name, mode="gt")
  if pred is not None:
    img = visualize(img, pred, label_to_name, mode="pred")
  return img


def visualize_batch(
  images: torch.Tensor,
  gt: list[ImageResult],
  preds: list[ImageResult],
  label_to_name: dict[int, str],
  dataset: Dataset,
) -> list[NDArray[np.uint8]]:
  """Visualize a batch of image tensors with ground truth and predictions."""
  vis_images = []
  for i in range(images.shape[0]):
    img = dataset.denormalize(images[i])
    vis_images.append(visualize_results(img, gt[i], preds[i], label_to_name))
  return vis_images
