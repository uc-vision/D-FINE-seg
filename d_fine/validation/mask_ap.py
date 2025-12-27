from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from image_detection.annotation import find_instance_overlaps
from d_fine.core.types import ImageResult
from .utils import (
    get_class_indices, 
    get_sorted_class_preds, 
    find_best_matches, 
    compute_ap_from_counts
)

@dataclass(frozen=True)
class MaskAPResult:
    map_50: float
    map_75: float
    map_50_95: float

def compute_img_ious(preds: ImageResult, gt: ImageResult) -> np.ndarray:
    """Pre-compute pairwise IoU matrix for a single image with box pre-filtering."""
    num_p, num_g = len(preds.labels), len(gt.labels)
    ious = np.zeros((num_p, num_g), dtype=np.float32)
    if num_p == 0 or num_g == 0:
        return ious
    
    box_ious = find_instance_overlaps(preds.masks, gt.masks)
    pred_indices, gt_indices = torch.nonzero(box_ious > 0, as_tuple=True)
    
    for p_idx, g_idx in zip(pred_indices.tolist(), gt_indices.tolist()):
        ious[p_idx, g_idx] = preds.masks[p_idx].iou(gt.masks[g_idx])
    return ious

class SparseMaskAP:
    def __init__(self, thresholds: np.ndarray | None = None):
        self.thresholds = thresholds if thresholds is not None else np.arange(0.5, 1.0, 0.05)

    def compute(self, gt: list[ImageResult], preds: list[ImageResult]) -> MaskAPResult:
        """Compute mask mAP using InstanceMask.iou with box-prefiltering."""
        all_labels = sorted(list({lbl for r in gt + preds for lbl in r.labels.tolist()}))
        all_ious = [compute_img_ious(p, g) for g, p in zip(gt, preds)]

        aps = [self._compute_class_ap(lbl, gt, preds, all_ious) for lbl in all_labels]
        results = [a for a in aps if a is not None]
        
        if not results:
            return MaskAPResult(0.0, 0.0, 0.0)
        
        results_arr = np.array(results)
        return MaskAPResult(
            map_50=float(np.mean(results_arr[:, 0])), 
            map_75=float(np.mean(results_arr[:, 5])) if results_arr.shape[1] > 5 else 0.0, 
            map_50_95=float(np.mean(results_arr))
        )

    def _compute_class_ap(self, label: int, gt: list[ImageResult], preds: list[ImageResult], all_ious: list[np.ndarray]) -> np.ndarray | None:
        gt_indices_per_img = get_class_indices(gt, label)
        num_ground_truth = sum(len(idx) for idx in gt_indices_per_img)
        if num_ground_truth == 0: return None

        class_preds = get_sorted_class_preds(preds, label)
        if not class_preds: return np.zeros(len(self.thresholds))

        num_thresholds, num_preds = len(self.thresholds), len(class_preds)
        img_gt_matched = [np.zeros((num_thresholds, len(idx)), dtype=bool) for idx in gt_indices_per_img]
        tps, fps = np.zeros((num_thresholds, num_preds)), np.zeros((num_thresholds, num_preds))
        
        for p_idx, (_, img_idx, p_mask_idx) in enumerate(class_preds):
            local_gt_indices = gt_indices_per_img[img_idx]
            if not local_gt_indices:
                fps[:, p_idx] = 1
                continue
            
            local_ious = all_ious[img_idx][p_mask_idx, local_gt_indices]
            is_match, best_gt_local_idx = find_best_matches(local_ious, img_gt_matched[img_idx], self.thresholds)
            
            t_match = np.where(is_match)[0]
            tps[t_match, p_idx] = 1
            img_gt_matched[img_idx][t_match, best_gt_local_idx[t_match]] = True
            fps[:, p_idx] = 1 - tps[:, p_idx]

        return np.array([compute_ap_from_counts(tps[t_idx], fps[t_idx], num_ground_truth) 
                        for t_idx in range(num_thresholds)])
