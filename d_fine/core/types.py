from __future__ import annotations
from dataclasses import dataclass, field
import torch
from image_detection.annotation.coco import InstanceMask

@dataclass
class ImageResult:
    labels: torch.Tensor  # [N] Int64
    boxes: torch.Tensor   # [N, 4] Float32 (abs xyxy)
    img_size: tuple[int, int]
    scores: torch.Tensor  # [N] Float32
    masks: list[InstanceMask] = field(default_factory=list)

    def to_torchmetrics_dict(self, is_gt: bool = False) -> dict[str, torch.Tensor | list[list[list[float]]]]:
        res: dict[str, torch.Tensor | list[list[list[float]]]] = {
            "boxes": self.boxes,
            "labels": self.labels,
        }
        if not is_gt:
            res["scores"] = self.scores
        
        if self.masks:
            # Provide masks as Polygons (list[list[float]]) for memory efficiency
            # torchmetrics/faster-coco-eval handles this directly in C++
            res["masks"] = [m.to_polygons() for m in self.masks]
            
        return res

    def filter(self, threshold: float) -> ImageResult:
        """Filter predictions by confidence threshold."""
        keep = self.scores >= threshold
        indices = keep.nonzero(as_tuple=True)[0]
        
        return ImageResult(
            labels=self.labels[keep],
            boxes=self.boxes[keep],
            img_size=self.img_size,
            scores=self.scores[keep],
            masks=[self.masks[i] for i in indices.tolist()] if self.masks else []
        )

    def to_device(self, device: torch.device) -> ImageResult:
        return ImageResult(
            labels=self.labels.to(device),
            boxes=self.boxes.to(device),
            img_size=self.img_size,
            scores=self.scores.to(device),
            masks=[m.to(device) for m in self.masks]
        )
