from __future__ import annotations
from pathlib import Path
from typing import Annotated, Callable
import torch
import numpy as np
from pydantic import BaseModel
from ucvision_utility import torch_types, numpy_types
from ucvision_utility.array_types import Int64, Float32, UInt8

from d_fine.dataset.dataset import CompactMasks, ProcessedSample, ImageTransform


class DetectionSample(BaseModel, frozen=True):
  """Raw sample for object detection (image + boxes)."""

  image: Annotated[np.ndarray, numpy_types.Dynamic(UInt8, "H W 3")]
  targets: Annotated[np.ndarray, numpy_types.Dynamic(Float32, "N 5")]
  orig_size: Annotated[torch.Tensor, torch_types.Dynamic(Int64, "2")]
  paths: tuple[Path, ...]

  def apply_transform(self, transform: ImageTransform) -> ProcessedSample:
    n = len(self.targets)
    bboxes = np.hstack([self.targets[:, 1:], np.arange(n).reshape(-1, 1)])
    res = transform(image=self.image, bboxes=bboxes, class_labels=self.targets[:, 0])

    t_bboxes = np.array(res["bboxes"]).reshape(-1, 5)
    return ProcessedSample(
      image=res["image"],
      labels=torch.as_tensor(res["class_labels"], dtype=torch.int64),
      boxes=torch.as_tensor(t_bboxes[:, :4], dtype=torch.float32),
      masks=CompactMasks.empty(*res["image"].shape[1:]),
      paths=self.paths,
      orig_size=self.orig_size,
    )

  def warp_affine(self, M: np.ndarray, target_size: tuple[int, int]) -> "DetectionSample":
    from d_fine.core.box_utils import refine_boxes_from_affine
    import cv2

    image = self.image
    if (M != np.eye(3)).any():
      image = cv2.warpAffine(image, M[:2], dsize=target_size, borderValue=(114, 114, 114))
    targets, _, _ = refine_boxes_from_affine(self.targets, M)
    return DetectionSample(
      image=image,
      targets=targets,
      orig_size=torch.tensor([target_size[1], target_size[0]]),
      paths=self.paths,
    )
