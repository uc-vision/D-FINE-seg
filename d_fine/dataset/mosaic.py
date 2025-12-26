import random
from dataclasses import dataclass

import cv2
import numpy as np

from d_fine.config import ImageConfig, MosaicAugsConfig
from d_fine.utils import get_mosaic_coordinate, get_transform_matrix


@dataclass
class MosaicSource:
    """Per-image transformation information for mosaic."""
    scale: tuple[float, float]
    pad: tuple[int, int]
    large_coords: tuple[int, int, int, int]
    small_coords: tuple[int, int, int, int]
    canvas_size: tuple[int, int]
    
    def resize_image(self, img: np.ndarray, interpolation: int = cv2.INTER_AREA) -> np.ndarray:
        """Resize image using scale factors."""
        h, w = img.shape[:2]
        return cv2.resize(img, (int(w * self.scale[0]), int(h * self.scale[1])), interpolation=interpolation)
    
    def resize_mask(self, mask: np.ndarray, h: int, w: int) -> np.ndarray:
        """Resize mask using scale factors with nearest neighbor interpolation."""
        return cv2.resize(mask.astype(np.uint8), (int(w * self.scale[0]), int(h * self.scale[1])), interpolation=cv2.INTER_NEAREST)
    
    def transform_polygon(self, poly: np.ndarray) -> np.ndarray:
        """Transform polygon coordinates using scale and pad."""
        p = poly.copy()
        p[:, 0] = self.scale[0] * p[:, 0] + self.pad[0]
        p[:, 1] = self.scale[1] * p[:, 1] + self.pad[1]
        return p
    
    def create_canvas(self, dtype=np.uint8) -> np.ndarray:
        """Create zero-filled canvas."""
        return np.zeros((self.canvas_size[0], self.canvas_size[1]), dtype=dtype)
    
    def paste(self, img: np.ndarray, mosaic_img: np.ndarray) -> None:
        """Crop image according to small_coords and paste into mosaic_img at large_coords."""
        cropped = img[self.small_coords[1]:self.small_coords[3], self.small_coords[0]:self.small_coords[2]]
        mosaic_img[self.large_coords[1]:self.large_coords[3], self.large_coords[0]:self.large_coords[2]] = cropped
    
    def paste_mask(self, mask: np.ndarray, canvas: np.ndarray) -> None:
        """Crop mask according to small_coords and paste into canvas at large_coords."""
        cropped = mask[self.small_coords[1]:self.small_coords[3], self.small_coords[0]:self.small_coords[2]]
        canvas[self.large_coords[1]:self.large_coords[3], self.large_coords[0]:self.large_coords[2]] = cropped
    
    def transform_mask(self, mask: np.ndarray, h: int, w: int) -> np.ndarray:
        """Transform mask: resize and place on canvas."""
        mask_resized = self.resize_mask(mask, h, w)
        canvas = self.create_canvas()
        self.paste_mask(mask_resized, canvas)
        return canvas


@dataclass
class MosaicInfo:
    """Information needed to extract and transform mosaic components."""
    mosaic_targets: np.ndarray
    affine_matrix: np.ndarray
    indices: tuple[int, ...]
    xc: int
    yc: int
    per_image_info: tuple[MosaicSource, ...]


def load_mosaic(
    base_size: int,
    idx: int,
    get_data_fn,
    target_h: int,
    target_w: int,
    img_config: ImageConfig,
    mosaic_augs: MosaicAugsConfig,
) -> MosaicInfo:
    """Common mosaic loading logic shared between YOLO and COCO datasets.
    
    Args:
        base_size: Actual dataset size (number of samples)
        idx: Index of the primary image
        get_data_fn: Function to get data for an index (returns img, targets, orig_size, masks/polys)
        target_h: Target height
        target_w: Target width
        img_config: Image configuration (contains augmentation config)
    
    Returns:
        MosaicInfo containing all information needed to extract and transform mosaic components
    """
    mosaic_targets = []
    yc = int(random.uniform(target_h * 0.6, target_h * 1.4))
    xc = int(random.uniform(target_w * 0.6, target_w * 1.4))
    indices = [idx] + [random.randint(0, base_size - 1) for _ in range(3)]
    canvas_w, canvas_h = 2 * target_w, 2 * target_h

    per_image_info = []
    for i_mosaic, m_idx in enumerate(indices):
        img, targets, _, _ = get_data_fn(m_idx)
        (h, w, c) = img.shape[:3]

        if img_config.keep_ratio:
            scale_h = min(1.0 * target_h / h, 1.0 * target_w / w)
            scale_w = scale_h
        else:
            scale_h, scale_w = (1.0 * target_h / h, 1.0 * target_w / w)

        img_resized_h = int(h * scale_h)
        img_resized_w = int(w * scale_w)
        
        mosaic_img_temp = np.full((target_h * 2, target_w * 2, c), 114, dtype=np.uint8)
        (l_x1, l_y1, l_x2, l_y2), (s_x1, s_y1, s_x2, s_y2) = get_mosaic_coordinate(
            mosaic_img_temp, i_mosaic, xc, yc, img_resized_w, img_resized_h, target_h, target_w
        )

        padw, padh = l_x1 - s_x1, l_y1 - s_y1

        if targets.size > 0:
            targets = targets.copy()
            targets[:, 1] = scale_w * targets[:, 1] + padw
            targets[:, 2] = scale_h * targets[:, 2] + padh
            targets[:, 3] = scale_w * targets[:, 3] + padw
            targets[:, 4] = scale_h * targets[:, 4] + padh
        
        mosaic_targets.append(targets)
        
        per_image_info.append(MosaicSource(
            scale=(scale_w, scale_h),
            pad=(padw, padh),
            large_coords=(l_x1, l_y1, l_x2, l_y2),
            small_coords=(s_x1, s_y1, s_x2, s_y2),
            canvas_size=(canvas_h, canvas_w),
        ))

    if len(mosaic_targets):
        mosaic_targets = np.concatenate(mosaic_targets, 0)
        np.clip(mosaic_targets[:, 1], 0, canvas_w, out=mosaic_targets[:, 1])
        np.clip(mosaic_targets[:, 2], 0, canvas_h, out=mosaic_targets[:, 2])
        np.clip(mosaic_targets[:, 3], 0, canvas_w, out=mosaic_targets[:, 3])
        np.clip(mosaic_targets[:, 4], 0, canvas_h, out=mosaic_targets[:, 4])

    M, _ = get_transform_matrix(
        (target_h * 2, target_w * 2), (target_w, target_h),
        mosaic_augs.degrees, mosaic_augs.mosaic_scale,
        mosaic_augs.shear, mosaic_augs.translate
    )

    return MosaicInfo(
        mosaic_targets=mosaic_targets,
        affine_matrix=M,
        indices=tuple(indices),
        xc=xc,
        yc=yc,
        per_image_info=tuple(per_image_info),
    )

