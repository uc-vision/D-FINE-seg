from __future__ import annotations
from collections import defaultdict
from collections.abc import Callable
import torch
import numpy as np
from torchvision.ops import box_iou
from .metrics import PerClassMetrics
from .confusion_matrix import ConfusionMatrix
from .utils import Match, MatchingResult, update_metrics_with_matches
from d_fine.core.types import ImageResult


def greedy_match(ious: torch.Tensor, iou_threshold: float) -> MatchingResult:
  """Performs greedy matching between predictions and ground truths based on IoU."""
  if ious.numel() == 0:
    return MatchingResult([], list(range(ious.shape[0])), list(range(ious.shape[1])))

  mask = ious >= iou_threshold
  p_idx, g_idx = torch.nonzero(mask, as_tuple=True)
  vals = ious[p_idx, g_idx]
  order = torch.argsort(-vals)

  pred_matched = np.zeros(ious.shape[0], dtype=bool)
  gt_matched = np.zeros(ious.shape[1], dtype=bool)
  matches: list[Match] = []

  for pi, gi in zip(p_idx[order].tolist(), g_idx[order].tolist()):
    if pred_matched[pi] or gt_matched[gi]:
      continue
    pred_matched[pi] = True
    gt_matched[gi] = True
    matches.append(Match(pi, gi, ious[pi, gi].item()))

  return MatchingResult(
    matches=matches,
    unmatched_preds=[i for i in range(ious.shape[0]) if not pred_matched[i]],
    unmatched_gt=[i for i in range(ious.shape[1]) if not gt_matched[i]],
  )


def box_iou_fn(preds: ImageResult, gt: ImageResult) -> torch.Tensor:
  """Compute IoU matrix between boxes of two ImageResults."""
  if not len(preds.boxes) or not len(gt.boxes):
    return torch.zeros((len(preds.boxes), len(gt.boxes)))
  return box_iou(preds.boxes, gt.boxes)


def mask_iou_fn(preds: ImageResult, gt: ImageResult) -> torch.Tensor:
  """Compute IoU matrix between masks of two ImageResults."""
  if not preds.masks or not gt.masks:
    return torch.zeros((len(preds.masks), len(gt.masks)))

  ious = torch.zeros((len(preds.masks), len(gt.masks)))
  for p_idx, pred_mask in enumerate(preds.masks):
    for g_idx, gt_mask in enumerate(gt.masks):
      ious[p_idx, g_idx] = pred_mask.iou(gt_mask)
  return ious


class GreedyMatcher:
  def __init__(self, iou_threshold: float):
    self.iou_threshold = iou_threshold

  def compute_metrics(
    self,
    preds: list[ImageResult],
    gts: list[ImageResult],
    conf_matrix: ConfusionMatrix,
    iou_fn: Callable[[ImageResult, ImageResult], torch.Tensor],
  ) -> dict[int, PerClassMetrics]:
    """Computes per-class metrics across a batch of images using greedy matching."""
    metrics: dict[int, PerClassMetrics] = defaultdict(PerClassMetrics)

    for pred, gt in zip(preds, gts):
      ious = iou_fn(pred, gt)
      result = greedy_match(ious, self.iou_threshold)
      update_metrics_with_matches(metrics, conf_matrix, pred.labels, gt.labels, result)

    return metrics
