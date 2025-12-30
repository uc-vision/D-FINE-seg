from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from lib_detection.annotation.coco import CocoDataset, InstanceMask
from lib_detection.annotation.instance_mask import tight_bounds

from d_fine.core.types import ImageResult
from d_fine.dataset.dataset import CompactMasks, ProcessedSample
from d_fine.dataset.detection.sample import DetectionSample
from d_fine.dataset.segmentation.sample import SegmentationSample

from .confusion_matrix import ConfusionMatrix
from .metrics import PerClassMetrics


@dataclass(frozen=True)
class Match:
  pred_idx: int
  gt_idx: int
  iou: float


@dataclass(frozen=True)
class MatchingResult:
  matches: list[Match]
  unmatched_preds: list[int]
  unmatched_gt: list[int]


def update_metrics_with_matches(
  metrics: dict[int, PerClassMetrics],
  conf_matrix: ConfusionMatrix,
  pred_labels: torch.Tensor,
  gt_labels: torch.Tensor,
  result: MatchingResult,
) -> tuple[dict[int, PerClassMetrics], ConfusionMatrix]:
  """Update metric counters and confusion matrix based on matching results."""
  # Matches (TP)
  for m in result.matches:
    label = int(gt_labels[m.gt_idx])
    metrics[label] += PerClassMetrics.tp(m.iou)
    conf_matrix = conf_matrix.update(int(pred_labels[m.pred_idx]), label)

  # Unmatched predictions (FP)
  for p_idx in result.unmatched_preds:
    label = int(pred_labels[p_idx])
    metrics[label] += PerClassMetrics.fp(0.0)
    conf_matrix = conf_matrix.update(label, -1)  # -1 for background/no GT

  # Unmatched ground truths (FN)
  for g_idx in result.unmatched_gt:
    label = int(gt_labels[g_idx])
    metrics[label] += PerClassMetrics.fn(0.0)
    conf_matrix = conf_matrix.update(-1, label)

  return metrics, conf_matrix


def get_sorted_class_preds(results: list[ImageResult], label: int) -> list[tuple[float, int, int]]:
  """Get sorted predictions for a single class across all images."""
  preds = []
  for img_idx, res in enumerate(results):
    mask = res.labels == label
    if mask.any():
      scores = res.scores[mask].tolist()
      indices = torch.where(mask)[0].tolist()
      for idx, score in zip(indices, scores):
        preds.append((score, img_idx, idx))
  return sorted(preds, key=lambda x: x[0], reverse=True)


def get_class_indices(results: list[ImageResult], label: int) -> list[list[int]]:
  """Get indices of instances with a given label for each image."""
  return [torch.where(res.labels == label)[0].tolist() for res in results]


def find_best_matches(
  ious: np.ndarray, matched_gt: np.ndarray, thresholds: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
  """Find best GT match for each prediction at multiple IoU thresholds."""
  num_thresholds = len(thresholds)
  is_match = np.zeros(num_thresholds, dtype=bool)
  best_gt_idx = np.zeros(num_thresholds, dtype=int)

  if ious.size == 0:
    return is_match, best_gt_idx

  for t_idx, thr in enumerate(thresholds):
    # Filter out already matched GTs for this threshold
    valid_ious = ious.copy()
    valid_ious[matched_gt[t_idx]] = -1

    if (valid_ious >= thr).any():
      best_idx = valid_ious.argmax()
      is_match[t_idx] = True
      best_gt_idx[t_idx] = best_idx

  return is_match, best_gt_idx


def compute_ap_from_counts(tps: np.ndarray, fps: np.ndarray, num_gt: int) -> float:
  """Compute Average Precision from TP/FP counts."""
  if tps.size == 0:
    return 0.0

  tps_cum = np.cumsum(tps)
  fps_cum = np.cumsum(fps)

  recalls = tps_cum / num_gt
  precisions = tps_cum / (tps_cum + fps_cum + 1e-16)

  mrec = np.concatenate(([0.0], recalls, [1.0]))
  mpre = np.concatenate(([1.0], precisions, [0.0]))

  for i in range(len(mpre) - 2, -1, -1):
    mpre[i] = max(mpre[i], mpre[i + 1])

  indices = np.where(mrec[1:] != mrec[:-1])[0]
  ap = np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])
  return float(ap)


def processed_sample_to_instances(sample: ProcessedSample) -> list[InstanceMask]:
  """Convert a ProcessedSample to a list of InstanceMasks."""
  labels = sample.labels
  boxes = sample.boxes
  masks = sample.masks

  instances = []
  expanded = masks.expand()

  for i in range(len(labels)):
    label = int(labels[i])
    box = boxes[i].cpu().numpy()
    x1, y1, x2, y2 = map(int, box)

    mask = expanded[i] if expanded is not None and expanded.numel() > 0 else None

    if mask is not None and mask.any():
      tight_res = tight_bounds(mask)
      if tight_res:
        ox1, oy1, ox2, oy2 = tight_res
        instances.append(
          InstanceMask(mask=mask[oy1:oy2, ox1:ox2].cpu(), label=label, offset=(ox1, oy1), score=1.0)
        )
    else:
      w_box, h_box = max(1, x2 - x1), max(1, y2 - y1)
      instances.append(
        InstanceMask(
          mask=torch.ones((h_box, w_box), dtype=torch.bool), label=label, offset=(x1, y1), score=1.0
        )
      )
  return instances


def dataloader_target_to_instances(target: dict) -> list[InstanceMask]:
  """Convert a dataloader target dictionary to a list of InstanceMasks."""
  labels = target["labels"]
  boxes = target["boxes"]
  masks: CompactMasks = target["masks"]

  instances = []
  expanded = masks.expand()

  for i in range(len(labels)):
    label = int(labels[i])
    box = boxes[i].cpu().numpy()
    x1, y1, x2, y2 = map(int, box)

    mask = expanded[i] if expanded is not None and expanded.numel() > 0 else None

    if mask is not None and mask.any():
      tight_res = tight_bounds(mask)
      if tight_res:
        ox1, oy1, ox2, oy2 = tight_res
        instances.append(
          InstanceMask(mask=mask[oy1:oy2, ox1:ox2].cpu(), label=label, offset=(ox1, oy1), score=1.0)
        )
    else:
      w_box, h_box = max(1, x2 - x1), max(1, y2 - y1)
      instances.append(
        InstanceMask(
          mask=torch.ones((h_box, w_box), dtype=torch.bool), label=label, offset=(x1, y1), score=1.0
        )
      )
  return instances


def dataloader_target_to_image_result(target: dict) -> ImageResult:
  """Convert a dataloader target dictionary to an ImageResult."""
  labels = torch.as_tensor(target["labels"], dtype=torch.long)
  boxes = torch.as_tensor(target["boxes"], dtype=torch.float32)

  img_size_tensor = target.get("orig_size")
  img_size = tuple(img_size_tensor.tolist()) if img_size_tensor is not None else (0, 0)

  return ImageResult(
    labels=labels,
    boxes=boxes,
    img_size=img_size,
    scores=torch.ones_like(labels, dtype=torch.float32),
    masks=dataloader_target_to_instances(target),
  )


def coco_to_image_results(coco: CocoDataset) -> list[ImageResult]:
  """Convert COCO dataset annotations to a list of ImageResult objects."""
  results = []
  for img_info in coco.images:
    anns = coco.annotations_by_image_id.get(img_info.id, [])
    instances = [
      a.to_instance_mask(img_info.height, img_info.width, torch.device("cpu")).model_copy(
        update={"score": 1.0}
      )
      for a in anns
    ]
    results.append(ImageResult.from_instances(instances, (img_info.width, img_info.height)))
  return results


def detection_sample_to_image_result(sample: DetectionSample) -> ImageResult:
  """Convert DetectionSample to ImageResult."""
  w, h = sample.orig_size.tolist()
  return ImageResult(
    labels=torch.as_tensor(sample.targets[:, 0], dtype=torch.long),
    boxes=torch.as_tensor(sample.targets[:, 1:5], dtype=torch.float32),
    img_size=(w, h),
    scores=torch.ones(len(sample.targets), dtype=torch.float32),
    masks=[],
  )


def segmentation_sample_to_image_result(sample: SegmentationSample) -> ImageResult:
  """Convert SegmentationSample to ImageResult."""
  w, h = sample.orig_size.tolist()
  boxes = sample.get_boxes()

  masks = []
  for i, label in enumerate(sample.labels):
    mask_bool = sample.id_map == (i + 1)
    if not mask_bool.any():
      continue

    y, x = np.where(mask_bool)
    y1, x1, y2, x2 = y.min(), x.min(), y.max(), x.max()

    m = InstanceMask(
      mask=torch.from_numpy(mask_bool[y1 : y2 + 1, x1 : x2 + 1]),
      label=int(label),
      offset=(int(x1), int(y1)),
      score=1.0,
    )
    masks.append(m)

  return ImageResult(
    labels=torch.as_tensor(sample.labels, dtype=torch.long),
    boxes=torch.as_tensor(boxes, dtype=torch.float32),
    img_size=(w, h),
    scores=torch.ones(len(sample.labels), dtype=torch.float32),
    masks=masks,
  )
