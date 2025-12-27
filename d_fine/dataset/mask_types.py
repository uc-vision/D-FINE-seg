from __future__ import annotations
from typing import Annotated
import torch
import numpy as np
from pydantic import BaseModel
from ucvision_utility import torch_types, numpy_types
from ucvision_utility.array_types import UInt16, Int64, Float32, UInt8
from image_detection.annotation.coco import PolygonMasks

class CompactMasks(BaseModel, frozen=True):
    """Compact representation of instance masks using an ID map and index list."""
    id_map: Annotated[torch.Tensor, torch_types.Dynamic(UInt16, "H W")]
    indices: Annotated[torch.Tensor, torch_types.Dynamic(Int64, "N")]

    @staticmethod
    def empty(h: int, w: int, device: torch.device = torch.device("cpu")) -> CompactMasks:
        return CompactMasks(
            id_map=torch.zeros((h, w), dtype=torch.uint16, device=device),
            indices=torch.zeros(0, dtype=torch.int64, device=device)
        )

    def expand(self) -> torch.Tensor:
        """Expand compact ID map representation to (N, H, W) bool tensor."""
        if self.indices.numel() == 0:
            return torch.zeros((0, *self.id_map.shape), dtype=torch.bool, device=self.id_map.device)
        return self.id_map.unsqueeze(0).to(torch.int32) == (self.indices.view(-1, 1, 1) + 1).to(torch.int32)

class RawSample(BaseModel, frozen=True):
    """Raw sample data loaded from disk."""
    image: Annotated[np.ndarray, numpy_types.Dynamic(UInt8, "H W 3")]
    targets: Annotated[np.ndarray, numpy_types.Dynamic(Float32, "N 5")]
    orig_size: Annotated[torch.Tensor, torch_types.Dynamic(Int64, "2")]
    id_map: Annotated[np.ndarray, numpy_types.Dynamic(UInt16, "H W")]

    def prepare_for_transform(self) -> dict:
        """Extract data for Albumentations transform in a uniform way."""
        n = len(self.targets)
        # Uniformly handle empty/non-empty targets with numpy
        indices = np.arange(n).reshape(-1, 1)
        bboxes = np.hstack([self.targets[:, 1:], indices])
        return {
            "image": self.image,
            "bboxes": bboxes,
            "class_labels": self.targets[:, 0],
            "mask": self.id_map
        }

class ProcessedSample(BaseModel, frozen=True):
    """Transformed and formatted sample ready for the model."""
    image: Annotated[torch.Tensor, torch_types.Dynamic(Float32, "3 H W")]
    labels: Annotated[torch.Tensor, torch_types.Dynamic(Int64, "N")]
    boxes: Annotated[torch.Tensor, torch_types.Dynamic(Float32, "N 4")]
    masks: CompactMasks

    @classmethod
    def from_transform(cls, transformed: dict) -> ProcessedSample:
        """Create ProcessedSample from transform output without branching."""
        bboxes = np.array(transformed["bboxes"]).reshape(-1, 5)
        return cls(
            image=transformed["image"],
            labels=torch.as_tensor(transformed["class_labels"], dtype=torch.int64),
            boxes=torch.as_tensor(bboxes[:, :4], dtype=torch.float32),
            masks=CompactMasks(
                id_map=torch.as_tensor(transformed["mask"], dtype=torch.uint16),
                indices=torch.as_tensor(bboxes[:, 4], dtype=torch.int64)
            )
        )
