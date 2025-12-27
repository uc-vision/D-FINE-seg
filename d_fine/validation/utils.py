from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
import numpy as np
import torch
from d_fine.core.types import ImageResult
from .metrics import PerClassMetrics
from .confusion_matrix import ConfusionMatrix
from image_detection.annotation import InstanceMask, stack_boxes

if TYPE_CHECKING:
    from image_detection.annotation.coco import CocoDataset

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
    """Get indices of instances matching a specific label for each image."""
    return [(r.labels == label).nonzero(as_tuple=True)[0].tolist() for r in results]

def get_sorted_class_preds(preds: list[ImageResult], label: int) -> list[tuple[float, int, int]]:
    """Gather and sort all predictions for a specific label across all images."""
    extracted = [
        (p.scores[idx].item(), img_idx, idx)
        for img_idx, p in enumerate(preds)
        for idx in (p.labels == label).nonzero(as_tuple=True)[0].tolist()
    ]
    return sorted(extracted, key=lambda x: x[0], reverse=True)

def find_best_matches(ious: np.ndarray, matched: np.ndarray, thresholds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Find best matching GT indices for a prediction across multiple thresholds.
    Args:
        ious: (num_gt,) array of IoUs for a single prediction
        matched: (num_thresholds, num_gt) boolean array of matched status
        thresholds: (num_thresholds,) array of IoU thresholds
    Returns:
        is_match: (num_thresholds,) boolean array
        best_gt_idx: (num_thresholds,) int array of best matching GT local indices
    """
    masked_ious = np.where(matched, -1.0, ious[None, :])
    best_gt_idx = np.argmax(masked_ious, axis=1)
    best_iou = masked_ious[np.arange(len(thresholds)), best_gt_idx]
    is_match = best_iou > thresholds
    return is_match, best_gt_idx

def compute_ap_from_counts(true_positives: np.ndarray, false_positives: np.ndarray, num_ground_truth: int) -> float:
    """Compute Average Precision using COCO 101-point interpolation."""
    if num_ground_truth == 0: return 0.0
    
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
    result: MatchingResult
) -> None:
    """Update metrics and confusion matrix from matching results."""
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

def coco_to_image_results(coco: CocoDataset, device: torch.device = torch.device("cpu")) -> list[ImageResult]:
    """Convert COCO annotations into a list of ImageResult objects."""
    results = []
    for img in coco.images:
        anns = coco.annotations_by_image_id.get(img.id, [])
        if not anns: continue
        
        masks = [ann.to_instance_mask(img.height, img.width, device) for ann in anns]
        results.append(ImageResult(
            labels=torch.tensor([ann.category_id for ann in anns], dtype=torch.int64),
            boxes=stack_boxes(masks),
            img_size=(img.height, img.width),
            scores=torch.ones(len(anns)),
            masks=masks
        ))
    return results
