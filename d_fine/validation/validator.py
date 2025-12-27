import logging
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from d_fine.core.types import ImageResult
from .metrics import EvaluationMetrics, PerClassMetrics
from .confusion_matrix import ConfusionMatrix
from .mask_ap import SparseMaskAP
from .matcher import GreedyMatcher, box_iou_fn, mask_iou_fn
from .plots import ValidationPlotter
from image_detection.annotation import InstanceMask

# Suppress verbose output from faster_coco_eval
logging.getLogger("faster_coco_eval").setLevel(logging.WARNING)


class Validator:
    def __init__(
        self,
        gt: list[ImageResult],
        preds: list[ImageResult],
        label_to_name: dict[int, str],
        conf_thresh: float = 0.5,
        iou_thresh: float = 0.5,
        mask_batch_size: int = 1000,
    ) -> None:
        """
        Args:
            gt: List of ground truth samples
            preds: List of predicted samples
            label_to_name: Mapping from class ID to name
            conf_thresh: Confidence threshold for predictions
            iou_thresh: IoU threshold for matching
            mask_batch_size: Batch size for mask metric computation
        """
        self.gt = gt
        self.preds = preds
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.thresholds = np.arange(0.2, 1.0, 0.05)
        self.label_to_name = label_to_name
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
        self.conf_matrix = ConfusionMatrix(
            n_classes=len(all_classes),
            class_to_idx={cls_id: idx for idx, cls_id in enumerate(all_classes)}
        )
        self.sparse_mask_ap = SparseMaskAP()
        self.matcher = GreedyMatcher(iou_threshold=iou_thresh)
        self.plotter = ValidationPlotter(self.thresholds, label_to_name)

    def compute_metrics(self, extended=False, ignore_masks=False) -> EvaluationMetrics:
        self.torch_metrics = self.torch_metric.compute()

        metrics = self._compute_main_metrics(self.preds, ignore_masks=ignore_masks)
        metrics.map_50 = self.torch_metrics["map_50"].item()
        metrics.map_75 = self.torch_metrics["map_75"].item()
        metrics.map_50_95 = self.torch_metrics["map"].item()
        
        if self.use_masks and not ignore_masks:
            mask_ap = self.sparse_mask_ap.compute(self.gt, self.preds)
            metrics.map_50_mask = mask_ap.map_50
            metrics.map_75_mask = mask_ap.map_75
            metrics.map_50_95_mask = mask_ap.map_50_95

        return metrics

    def _compute_main_metrics(self, preds: list[ImageResult], ignore_masks=False) -> EvaluationMetrics:
        self.metrics_per_class = self._compute_matches(preds, ignore_masks=ignore_masks)
        total = sum(self.metrics_per_class.values(), PerClassMetrics())
        
        return EvaluationMetrics(
            f1=total.f1, precision=total.precision, recall=total.recall, iou=total.avg_iou,
            tps=total.tps, fps=total.fps, fns=total.fns,
            per_class=dict(self.metrics_per_class)
        )

    def _compute_matches(self, preds: list[ImageResult], ignore_masks: bool) -> dict[int, PerClassMetrics]:
        if self.use_masks and not ignore_masks:
            return self.matcher.compute_metrics(preds, self.gt, self.conf_matrix, mask_iou_fn)
        return self.matcher.compute_metrics(preds, self.gt, self.conf_matrix, box_iou_fn)

    def save_plots(self, path_to_save: str | Path) -> None:
        self.plotter.save_all_plots(
            Path(path_to_save), 
            self.preds, 
            lambda p: self._compute_main_metrics(p, ignore_masks=True),
            self.conf_matrix
        )

