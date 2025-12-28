from __future__ import annotations
from dataclasses import dataclass, field
import torch
from lib_detection.annotation.coco import InstanceMask


@dataclass
class ImageResult:
  labels: torch.Tensor  # [N] Int64
  boxes: torch.Tensor  # [N, 4] Float32 (abs xyxy)
  img_size: tuple[int, int]
  scores: torch.Tensor  # [N] Float32
  masks: list[InstanceMask] = field(default_factory=list)

  @classmethod
  def from_instances(cls, instances: list[InstanceMask], img_size: tuple[int, int]) -> ImageResult:
    """Create ImageResult from a list of InstanceMask objects."""
    if not instances:
      return cls(
        labels=torch.zeros(0, dtype=torch.long),
        boxes=torch.zeros((0, 4)),
        img_size=img_size,
        scores=torch.zeros(0),
        masks=[],
      )

    labels = torch.tensor([inst.label for inst in instances], dtype=torch.long)
    scores = torch.tensor([inst.score for inst in instances], dtype=torch.float32)
    boxes = torch.stack(
      [
        torch.tensor(
          [inst.bounds.x1, inst.bounds.y1, inst.bounds.x2, inst.bounds.y2], dtype=torch.float32
        )
        for inst in instances
      ]
    )

    return cls(labels=labels, boxes=boxes, img_size=img_size, scores=scores, masks=instances)

  def to_torchmetrics_dict(
    self, is_gt: bool = False
  ) -> dict[str, torch.Tensor | list[list[list[float]]]]:
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
      masks=[self.masks[i] for i in indices.tolist()] if self.masks else [],
    )

  @classmethod
  def from_dict(cls, data: dict, conf_thresh: float | None = None) -> ImageResult:
    """Create ImageResult from a dictionary of results."""
    res = cls(
      labels=torch.as_tensor(data["labels"]),
      boxes=torch.as_tensor(data["boxes"]),
      img_size=data.get("img_size", (0, 0)),
      scores=torch.as_tensor(data["scores"]),
      masks=data.get("masks", []),
    )
    if conf_thresh is not None:
      res = res.filter(conf_thresh)
    return res

  def to_device(self, device: torch.device) -> ImageResult:
    return ImageResult(
      labels=self.labels.to(device),
      boxes=self.boxes.to(device),
      img_size=self.img_size,
      scores=self.scores.to(device),
      masks=[m.to(device) for m in self.masks],
    )

  def nms(self, iou_threshold: float = 0.5) -> ImageResult:
    """Apply non-maximum suppression to filter detections."""
    if len(self.labels) == 0:
      return self

    from torchvision.ops import nms

    # Perform NMS per class
    keep_indices = []
    unique_labels = self.labels.unique()

    for label in unique_labels:
      cls_mask = self.labels == label
      cls_indices = torch.where(cls_mask)[0]
      cls_boxes = self.boxes[cls_mask]
      cls_scores = self.scores[cls_mask]

      cls_keep = nms(cls_boxes, cls_scores, iou_threshold)
      keep_indices.append(cls_indices[cls_keep])

    keep = torch.cat(keep_indices)
    # Sort by score to keep things predictable
    keep = keep[self.scores[keep].argsort(descending=True)]

    return ImageResult(
      labels=self.labels[keep],
      boxes=self.boxes[keep],
      img_size=self.img_size,
      scores=self.scores[keep],
      masks=[self.masks[i] for i in keep.tolist()] if self.masks else [],
    )
