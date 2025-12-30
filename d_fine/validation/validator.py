import logging
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from d_fine.core.types import ImageResult
from .metrics import (
  EvaluationMetrics,
  PerClassMetrics,
  ValidationConfig,
  CoreMetrics,
  Counts,
  APMetrics,
)
from .confusion_matrix import ConfusionMatrix
from .mask_ap import SparseMaskAP
from .matcher import GreedyMatcher, box_iou_fn, mask_iou_fn
from .plots import ValidationPlotter
from lib_detection.annotation import InstanceMask

# Suppress verbose output from faster_coco_eval
logging.getLogger("faster_coco_eval").setLevel(logging.WARNING)


class Validator:
  def __init__(
    self,
    gt: Sequence[ImageResult],
    preds: Sequence[ImageResult],
    config: ValidationConfig,
    mask_batch_size: int = 1000,
  ) -> None:
    """
    Args:
        gt: List of ground truth samples
        preds: List of predicted samples
        config: Validation configuration (thresholds, labels)
        mask_batch_size: Batch size for mask metric computation
    """
    self.gt = gt
    self.preds = preds
    self.config = config
    self.conf_thresh = config.conf_threshold
    self.iou_thresh = config.iou_threshold
    self.thresholds = np.arange(0.2, 1.0, 0.05)
    self.label_to_name = config.label_to_name
    self.mask_batch_size = mask_batch_size

    # Use faster_coco_eval backend for numpy 2.x compatibility
    self.torch_metric = MeanAveragePrecision(
      box_format="xyxy", iou_type="bbox", sync_on_compute=False, backend="faster_coco_eval"
    )
    self.torch_metric.warn_on_many_detections = False

    # Prepare data for torchmetrics (standard bbox)
    tm_gt = [g.to_torchmetrics_dict(is_gt=True) for g in gt]
    tm_preds = [p.to_torchmetrics_dict(is_gt=False) for p in preds]

    self.torch_metric.update(tm_preds, tm_gt)

    # Check if masks are present
    self.use_masks = any(p.masks for p in preds) and any(g.masks for g in gt)

    # Initialize components
    all_classes = sorted(list({lbl for r in gt + preds for lbl in r.labels.tolist()}))
    self.conf_matrix = ConfusionMatrix.empty(
      n_classes=len(all_classes),
      class_to_idx={cls_id: idx for idx, cls_id in enumerate(all_classes)},
    )
    self.sparse_mask_ap = SparseMaskAP()
    self.matcher = GreedyMatcher(iou_threshold=self.iou_thresh)
    self.plotter = ValidationPlotter(self.thresholds, self.label_to_name)

  def compute_metrics(self, extended=False, ignore_masks=False) -> EvaluationMetrics:
    self.torch_metrics = self.torch_metric.compute()

    metrics = self._compute_main_metrics(self.preds, ignore_masks=ignore_masks)

    bbox_ap = APMetrics(
      map_50=self.torch_metrics["map_50"].item(),
      map_75=self.torch_metrics["map_75"].item(),
      map_50_95=self.torch_metrics["map"].item(),
    )

    mask_ap = APMetrics()
    if self.use_masks and not ignore_masks:
      res = self.sparse_mask_ap.compute(self.gt, self.preds)
      mask_ap = APMetrics(
        map_50=res.map_50,
        map_75=res.map_75,
        map_50_95=res.map_50_95,
      )

    from dataclasses import replace

    return replace(metrics, bbox=bbox_ap, mask=mask_ap)

  def _compute_main_metrics(
    self, preds: list[ImageResult], ignore_masks=False
  ) -> EvaluationMetrics:
    self.metrics_per_class = self._compute_matches(preds, ignore_masks=ignore_masks)
    total = sum(self.metrics_per_class.values(), PerClassMetrics())

    return EvaluationMetrics(
      core=CoreMetrics(
        f1=total.f1,
        precision=total.precision,
        recall=total.recall,
        iou=total.avg_iou,
      ),
      counts=Counts(
        tps=total.tps,
        fps=total.fps,
        fns=total.fns,
      ),
      per_class=dict(self.metrics_per_class),
    )

  def _compute_matches(
    self, preds: list[ImageResult], ignore_masks: bool
  ) -> dict[int, PerClassMetrics]:
    if self.use_masks and not ignore_masks:
      metrics, self.conf_matrix = self.matcher.compute_metrics(
        preds, self.gt, self.conf_matrix, mask_iou_fn
      )
    else:
      metrics, self.conf_matrix = self.matcher.compute_metrics(
        preds, self.gt, self.conf_matrix, box_iou_fn
      )
    return metrics

  def save_plots(self, path_to_save: Path) -> None:
    self.plotter.save_all_plots(
      path_to_save,
      self.preds,
      lambda p: self._compute_main_metrics(p, ignore_masks=True),
      self.conf_matrix,
    )
