from __future__ import annotations
from pathlib import Path
from typing import Annotated, Callable
import torch
import numpy as np
from pydantic import BaseModel
from ucvision_utility import torch_types, numpy_types
from ucvision_utility.array_types import UInt16, Int64, Float32, UInt8

from d_fine.dataset.dataset import CompactMasks, ProcessedSample, ImageTransform


class SegmentationSample(BaseModel, frozen=True):
  """Raw sample for instance segmentation (image + labels + masks)."""

  image: Annotated[np.ndarray, numpy_types.Dynamic(UInt8, "H W 3")]
  labels: Annotated[np.ndarray, numpy_types.Dynamic(Int64, "N")]
  orig_size: Annotated[torch.Tensor, torch_types.Dynamic(Int64, "2")]
  paths: tuple[Path, ...]
  id_map: Annotated[np.ndarray, numpy_types.Dynamic(UInt16, "H W")]

  def get_boxes(self) -> np.ndarray:
    """Compute tight bounding boxes from the ID map."""
    if self.labels.size == 0:
      return np.zeros((0, 4), dtype=np.float32)
    m = CompactMasks(
      id_map=torch.as_tensor(self.id_map, dtype=torch.uint16),
      indices=torch.arange(len(self.labels), dtype=torch.int64),
    )
    return m.get_boxes().cpu().numpy()

  def apply_transform(self, transform: ImageTransform) -> ProcessedSample:
    boxes = self.get_boxes()
    bboxes = np.hstack([boxes, np.arange(len(self.labels)).reshape(-1, 1)])
    res = transform(image=self.image, bboxes=bboxes, class_labels=self.labels, mask=self.id_map)

    t_bboxes = np.array(res["bboxes"]).reshape(-1, 5)
    t_mask = res["mask"]
    masks = CompactMasks(
      id_map=torch.as_tensor(t_mask, dtype=torch.uint16),
      indices=torch.as_tensor(t_bboxes[:, 4], dtype=torch.int64),
    )
    return ProcessedSample(
      image=res["image"],
      labels=torch.as_tensor(res["class_labels"], dtype=torch.int64),
      boxes=masks.get_boxes(),
      masks=masks,
      paths=self.paths,
      orig_size=self.orig_size,
    )

  def warp_affine(self, M: np.ndarray, target_size: tuple[int, int]) -> "SegmentationSample":
    import cv2

    image = self.image
    id_map = self.id_map
    if (M != np.eye(3)).any():
      image = cv2.warpAffine(image, M[:2], dsize=target_size, borderValue=(114, 114, 114))
      id_map = cv2.warpAffine(
        id_map, M[:2], dsize=target_size, borderValue=0, interpolation=cv2.INTER_NEAREST
      )

    if id_map.any():
      present = np.unique(id_map)
      present = present[present > 0]
      indices = present - 1
      labels = self.labels[indices]

      remapper = np.zeros(int(present.max() + 1), dtype=np.uint16)
      for i, val in enumerate(present):
        remapper[val] = i + 1
      id_map = remapper[id_map]

      return SegmentationSample(
        image=image,
        labels=labels,
        id_map=id_map,
        orig_size=torch.tensor([target_size[1], target_size[0]]),
        paths=self.paths,
      )

    return SegmentationSample(
      image=image,
      labels=np.zeros(0, dtype=np.int64),
      id_map=id_map,
      orig_size=torch.tensor([target_size[1], target_size[0]]),
      paths=self.paths,
    )
