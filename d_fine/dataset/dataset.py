from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Annotated, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset
from pydantic import BaseModel
from ucvision_utility import torch_types
from ucvision_utility.array_types import Int64, Float32, UInt16

from d_fine.config import Mode
from d_fine.core.types import ImageResult


type ToImageResult[T] = Callable[[T], ImageResult]
type ImageTransform = Callable[..., dict]


class CompactMasks(BaseModel, frozen=True):
  """Compact representation of instance masks using an ID map and index list."""

  id_map: Annotated[torch.Tensor, torch_types.Dynamic(UInt16, "H W")]
  indices: Annotated[torch.Tensor, torch_types.Dynamic(Int64, "N")]

  @staticmethod
  def empty(h: int, w: int, device: torch.device = torch.device("cpu")) -> "CompactMasks":
    return CompactMasks(
      id_map=torch.zeros((h, w), dtype=torch.uint16, device=device),
      indices=torch.zeros(0, dtype=torch.int64, device=device),
    )

  def expand(self) -> torch.Tensor:
    """Expand compact ID map representation to (N, H, W) bool tensor."""
    if self.indices.numel() == 0:
      return torch.zeros((0, *self.id_map.shape), dtype=torch.bool, device=self.id_map.device)
    return self.id_map.unsqueeze(0).to(torch.int32) == (self.indices.view(-1, 1, 1) + 1).to(
      torch.int32
    )

  def get_boxes(self) -> torch.Tensor:
    """Compute tight bounding boxes from the ID map."""
    if self.indices.numel() == 0:
      return torch.zeros((0, 4), dtype=torch.float32, device=self.id_map.device)

    import torchvision.ops

    masks = self.expand()
    mask_any = masks.flatten(1).any(dim=1)

    boxes = torch.zeros((len(self.indices), 4), dtype=torch.float32, device=self.id_map.device)
    if mask_any.any():
      boxes[mask_any] = torchvision.ops.masks_to_boxes(masks[mask_any])
    return boxes


class ProcessedSample(BaseModel, frozen=True):
  """Transformed and formatted sample ready for the model."""

  image: Annotated[torch.Tensor, torch_types.Dynamic(Float32, "3 H W")]
  labels: Annotated[torch.Tensor, torch_types.Dynamic(Int64, "N")]
  boxes: Annotated[torch.Tensor, torch_types.Dynamic(Float32, "N 4")]
  masks: CompactMasks
  paths: tuple[Path, ...]
  orig_size: Annotated[torch.Tensor, torch_types.Dynamic(Int64, "2")]


class Dataset[T](TorchDataset, ABC):
  @property
  @abstractmethod
  def mode(self) -> Mode:
    """Return the dataset mode: 'train', 'val', 'test', or 'bench'."""
    pass

  @abstractmethod
  def denormalize(self, image: torch.Tensor) -> np.ndarray:
    """Convert a normalized image tensor back to a uint8 RGB image."""
    pass

  @property
  @abstractmethod
  def label_to_name(self) -> dict[int, str]:
    """Return mapping from class ID to class name."""
    pass

  @property
  def num_classes(self) -> int:
    """Return the number of classes in the dataset."""
    return len(self.label_to_name)

  @abstractmethod
  def get_data(self, idx: int) -> T:
    """Load and return raw data for a given index."""
    pass


class Loader(ABC):
  @property
  @abstractmethod
  def label_to_name(self) -> dict[int, str]:
    """Return mapping from class ID to class name."""
    pass

  @property
  def num_classes(self) -> int:
    """Return the number of classes."""
    return len(self.label_to_name)

  @abstractmethod
  def build_dataloaders(
    self, distributed: bool = False
  ) -> tuple[
    DataLoader[ProcessedSample], DataLoader[ProcessedSample], DataLoader[ProcessedSample] | None
  ]:
    """Build and return data loaders for train, validation, and test splits."""
    pass
