from __future__ import annotations
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING
from pathlib import Path

import cv2
import numpy as np
import torch

from d_fine.config import ImageConfig
from d_fine.core.box_utils import get_transform_matrix


def get_mosaic_coordinate(
  mosaic_index: int, xc: int, yc: int, w: int, h: int, target_size: tuple[int, int]
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
  """Compute coordinates for mosaic augmentation."""
  tw, th = target_size
  if mosaic_index == 0:
    x1, y1, x2, y2 = max(xc - w, 0), max(yc - h, 0), xc, yc
    small_coord = w - (x2 - x1), h - (y2 - y1), w, h
  elif mosaic_index == 1:
    x1, y1, x2, y2 = xc, max(yc - h, 0), min(xc + w, tw * 2), yc
    small_coord = 0, h - (y2 - y1), min(w, x2 - x1), h
  elif mosaic_index == 2:
    x1, y1, x2, y2 = max(xc - w, 0), yc, xc, min(th * 2, yc + h)
    small_coord = w - (x2 - x1), 0, w, min(y2 - y1, h)
  else:  # mosaic_index == 3
    x1, y1, x2, y2 = xc, yc, min(xc + w, tw * 2), min(th * 2, yc + h)
    small_coord = 0, 0, min(w, x2 - x1), min(y2 - y1, h)
  return (x1, y1, x2, y2), small_coord


@dataclass
class MosaicSource:
  """Per-image transformation information for mosaic."""

  scale: tuple[float, float]
  pad: tuple[int, int]
  large_coords: tuple[int, int, int, int]
  small_coords: tuple[int, int, int, int]
  canvas_size: tuple[int, int]

  def resize_image(self, img: np.ndarray, interpolation: int = cv2.INTER_AREA) -> np.ndarray:
    """Resize image using scale factors."""
    h, w = img.shape[:2]
    return cv2.resize(
      img, (int(w * self.scale[0]), int(h * self.scale[1])), interpolation=interpolation
    )

  def paste(self, img: np.ndarray, mosaic_img: np.ndarray) -> None:
    """Crop image according to small_coords and paste into mosaic_img at large_coords."""
    cropped = img[
      self.small_coords[1] : self.small_coords[3], self.small_coords[0] : self.small_coords[2]
    ]
    mosaic_img[
      self.large_coords[1] : self.large_coords[3], self.large_coords[0] : self.large_coords[2]
    ] = cropped


def _get_mosaic_params(
  samples: list, target_size: tuple[int, int], img_config: ImageConfig
) -> tuple[int, int, list[MosaicSource], tuple[Path, ...]]:
  tw, th = target_size
  yc = int(random.uniform(th * 0.6, th * 1.4))
  xc = int(random.uniform(tw * 0.6, tw * 1.4))
  canvas_w, canvas_h = 2 * tw, 2 * th

  per_image_info = []
  all_paths = []

  for i, sample in enumerate(samples):
    h, w = sample.image.shape[:2]
    sw, sh = (
      (min(th / h, tw / w),) * 2
      if img_config.keep_aspect
      else (tw / w, th / h)
    )

    (l_x1, l_y1, l_x2, l_y2), (s_x1, s_y1, s_x2, s_y2) = get_mosaic_coordinate(
      i, xc, yc, int(w * sw), int(h * sh), target_size
    )

    info = MosaicSource(
      scale=(sw, sh),
      pad=(l_x1 - s_x1, l_y1 - s_y1),
      large_coords=(l_x1, l_y1, l_x2, l_y2),
      small_coords=(s_x1, s_y1, s_x2, s_y2),
      canvas_size=(canvas_w, canvas_h),
    )
    per_image_info.append(info)
    all_paths.extend(sample.paths)

  return canvas_w, canvas_h, per_image_info, tuple(all_paths)
