from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
import numpy as np
import torch
from d_fine.core.types import ImageResult
from .metrics import PerClassMetrics
from .confusion_matrix import ConfusionMatrix
from lib_detection.annotation import InstanceMask, stack_boxes

if TYPE_CHECKING:
  from lib_detection.annotation.coco import CocoDataset
  from d_fine.dataset.detection.sample import DetectionSample
  from d_fine.dataset.segmentation.sample import SegmentationSample


def detection_sample_to_image_result(sample: DetectionSample) -> ImageResult:
  """Convert DetectionSample into ImageResult ground truth."""
  h, w = sample.image.shape[:2]
  return ImageResult(
    labels=torch.from_numpy(sample.targets[:, 0]).long(),
    boxes=torch.from_numpy(sample.targets[:, 1:]).float(),
    img_size=(h, w),
    scores=torch.ones(len(sample.targets)),
    masks=[],
  )


def segmentation_sample_to_image_result(sample: SegmentationSample) -> ImageResult:
  """Convert SegmentationSample into ImageResult ground truth."""
  h, w = sample.image.shape[:2]
  n = len(sample.labels)
  boxes = sample.get_boxes()
  masks = []
  if n > 0:
    for i in range(n):
      mask_bool = sample.id_map == (i + 1)
      if not mask_bool.any():
        continue
      label = sample.labels[i]
      x1, y1, x2, y2 = boxes[i]
      ix1, iy1, ix2, iy2 = max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
      if ix2 <= ix1 or iy2 <= iy1:
        continue
      masks.append(
        InstanceMask(
          mask=torch.from_numpy(mask_bool[iy1:iy2, ix1:ix2]).bool(),
          label=int(label),
          offset=(ix1, iy1),
          score=1.0,
        )
      )
  return ImageResult(
    labels=torch.from_numpy(sample.labels).long(),
    boxes=torch.from_numpy(boxes).float(),
    img_size=(h, w),
    scores=torch.ones(n),
    masks=masks,
  )


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


def get_class_indices(results: list[ImageResult], label: int) -> list[list[int]]:
  return [(r.labels == label).nonzero(as_tuple=True)[0].tolist() for r in results]


def get_sorted_class_preds(preds: list[ImageResult], label: int) -> list[tuple[float, int, int]]:
  return sorted(
    [
      (p.scores[idx].item(), img_idx, idx)
      for img_idx, p in enumerate(preds)
      for idx in (p.labels == label).nonzero(as_tuple=True)[0].tolist()
    ],
    key=lambda x: x[0],
    reverse=True,
  )


def find_best_matches(
  ious: np.ndarray, matched: np.ndarray, thresholds: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
  masked_ious = np.where(matched, -1.0, ious[None, :])
  best_gt_idx = np.argmax(masked_ious, axis=1)
  return masked_ious[np.arange(len(thresholds)), best_gt_idx] > thresholds, best_gt_idx


def compute_ap_from_counts(
  true_positives: np.ndarray, false_positives: np.ndarray, num_ground_truth: int
) -> float:
  if num_ground_truth == 0:
    return 0.0
  cum_tp = np.cumsum(true_positives)
  recalls = cum_tp / num_ground_truth
  precisions = cum_tp / (cum_tp + np.cumsum(false_positives) + 1e-16)
  recall_points = np.linspace(0, 1, 101)
  return float(np.mean([np.max(precisions[recalls >= r], initial=0.0) for r in recall_points]))


def update_metrics_with_matches(
  metrics: dict[int, PerClassMetrics],
  conf_matrix: ConfusionMatrix,
  pred_labels: torch.Tensor | np.ndarray,
  gt_labels: torch.Tensor | np.ndarray,
  result: MatchingResult,
) -> None:
  for m in result.matches:
    p_lbl, g_lbl = int(pred_labels[m.pred_idx]), int(gt_labels[m.gt_idx])
    conf_matrix.update(g_lbl, p_lbl)
    if p_lbl == g_lbl:
      metrics[g_lbl].add_tp(m.iou)
    else:
      metrics[g_lbl].add_fn()
      metrics[p_lbl].add_fp()
  for p_idx in result.unmatched_preds:
    p_lbl = int(pred_labels[p_idx])
    conf_matrix.update(None, p_lbl)
    metrics[p_lbl].add_fp()
  for g_idx in result.unmatched_gt:
    g_lbl = int(gt_labels[g_idx])
    conf_matrix.update(g_lbl, None)
    metrics[g_lbl].add_fn()


def coco_to_image_results(
  coco: CocoDataset, device: torch.device = torch.device("cpu")
) -> list[ImageResult]:
  results = []
  for img in coco.images:
    anns = coco.annotations_by_image_id.get(img.id, [])
    if not anns:
      continue
    masks = [ann.to_instance_mask(img.height, img.width, device) for ann in anns]
    results.append(
      ImageResult(
        labels=torch.tensor([ann.category_id for ann in anns], dtype=torch.int64),
        boxes=stack_boxes(masks),
        img_size=(img.height, img.width),
        scores=torch.ones(len(anns)),
        masks=masks,
      )
    )
  return results
