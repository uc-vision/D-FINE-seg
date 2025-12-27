from __future__ import annotations
import cv2
import numpy as np
from numpy.typing import NDArray

from ..utils import draw_mask, vis_one_box
from ..core.types import ImageResult


def visualize(
    img: NDArray[np.uint8],
    result: ImageResult,
    label_to_name: dict[int, str],
) -> NDArray[np.uint8]:
    """Visualize detection and segmentation results on an image."""
    img = img.copy()
    
    # Draw masks first
    for m in result.masks:
        img = draw_mask(img, m.to_full_mask(*img.shape[:2]).cpu().numpy())
        
    # Draw boxes and labels on top
    for box, label, score in zip(result.boxes, result.labels, result.scores):
        vis_one_box(
            img, 
            box.tolist(), 
            int(label.item()), 
            mode="pred", 
            label_to_name=label_to_name, 
            score=float(score.item())
        )
    return img
