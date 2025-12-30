from __future__ import annotations
from pathlib import Path
from typing import Annotated, Callable
import torch
import numpy as np
from pydantic import BaseModel
from ucvision_utility import torch_types, numpy_types
from ucvision_utility.array_types import Int64, UInt16, UInt8

from d_fine.dataset.dataset import CompactMasks, ProcessedSample


def remap_id_map(id_map: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """Filter labels for present instances and remap id_map to be contiguous."""
  present = np.unique(id_map)
  present = present[present > 0]
  if present.size == 0:
    return id_map, np.zeros(0, dtype=labels.dtype)

  indices = present - 1
  new_labels = labels[indices]

  remapper = np.zeros(int(present.max() + 1), dtype=np.uint16)
  remapper[present] = np.arange(1, len(present) + 1, dtype=np.uint16)

  return remapper[id_map], new_labels


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

  def apply_transform(self, transform: Callable) -> ProcessedSample:
    boxes = self.get_boxes()
    labels = self.labels
    indices = np.arange(len(labels))

    res = transform(
      image=self.image,
      bboxes=boxes,
      class_labels=labels,
      indices=indices,
      mask=self.id_map,
    )

    t_bboxes = np.array(res["bboxes"]).reshape(-1, 4)
    t_mask = res["mask"]
    masks = CompactMasks(
      id_map=torch.as_tensor(t_mask, dtype=torch.uint16),
      indices=torch.as_tensor(res["indices"], dtype=torch.int64),
    )
    return ProcessedSample(
      image=res["image"],
      labels=torch.as_tensor(res["class_labels"], dtype=torch.int64),
      boxes=torch.as_tensor(t_bboxes, dtype=torch.float32),
      masks=masks,
      paths=self.paths,
      orig_size=self.orig_size,
    )

  def warp_affine(self, M: np.ndarray, target_size: tuple[int, int]) -> "SegmentationSample":
    import cv2

    image = cv2.warpAffine(self.image, M[:2], dsize=target_size, borderValue=(114, 114, 114))
    id_map = cv2.warpAffine(
      self.id_map, M[:2], dsize=target_size, borderValue=0, flags=cv2.INTER_NEAREST
    )

    new_id_map, new_labels = remap_id_map(id_map, self.labels)

    return SegmentationSample(
      image=image,
      labels=new_labels,
      id_map=new_id_map,
      orig_size=torch.tensor([target_size[1], target_size[0]]),
      paths=self.paths,
    )
