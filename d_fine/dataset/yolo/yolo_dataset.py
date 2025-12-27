from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

from loguru import logger

from d_fine.core.dist_utils import is_main_process
from d_fine.dataset.base import BaseDataset
from d_fine.config import ImageConfig, Task
from d_fine.dataset.config import AugConfig
from d_fine.dataset.augmentations import init_augs
from d_fine.dataset.mosaic import MosaicSource, load_mosaic, MosaicInfo
from d_fine.utils import (
    abs_xyxy_to_norm_xywh,
    get_mosaic_coordinate,
    norm_poly_to_abs,
    norm_xywh_to_abs_xyxy,
    poly_abs_to_mask,
    random_affine,
    vis_one_box,
)


def parse_yolo_label_file(path: Path):
    """
    Supports both pure detection lines (5 cols) and YOLO-Seg lines (>=7 cols).
    Returns:
      boxes_norm: np.ndarray (N,5) -> [cls, xc, yc, w, h] in norm (float32)
      polys_norm: list[np.ndarray] -> each (K,2) normalized polygon (float32) or [] if none
    """
    boxes_norm = []
    polys_norm = []

    with open(path, "r") as f:
        for ln, raw in enumerate(f, 1):
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            cl = float(parts[0])

            nums = [float(x) for x in parts[1:]]
            if len(nums) == 4:
                boxes_norm.append([cl, *nums[:4]])
                polys_norm.append(np.empty((0, 2), dtype=np.float32))
            elif len(nums) >= 6:
                if len(nums) % 2 == 1:
                    nums = nums[:-1]
                    logger.warning(
                        f"Odd number of coordinates in segmentation annotation at {path}:{ln}: {s}. "
                        "Dropping the last value."
                    )
                poly = np.array(nums).reshape(-1, 2)
                polys_norm.append(poly)
                x_min, y_min = poly.min(axis=0)
                x_max, y_max = poly.max(axis=0)
                boxes_norm.append(
                    [cl, (x_min + x_max) / 2, (y_min + y_max) / 2, x_max - x_min, y_max - y_min]
                )
            else:
                raise ValueError(f"Invalid label line (wrong number of values) {path}:{ln}: {s}")

    if len(boxes_norm) == 0:
        return np.zeros((0, 5), dtype=np.float32), []
    boxes_norm = np.asarray(boxes_norm, dtype=np.float32)
    return boxes_norm, polys_norm


from d_fine.config import DatasetConfig, Mode
from d_fine.config import ImageConfig

if TYPE_CHECKING:
    from d_fine.dataset.yolo.yolo_loader import YoloLoader


class YoloDatasetConfig(DatasetConfig, frozen=True):
    """YOLO dataset configuration."""
    
    label_to_name: dict[int, str]  # Required for YOLO
    img_config: ImageConfig
    
    def create_dataset(
        self,
        root_path: Path,
        split: list[str],
        mode: Mode,
    ) -> YoloDataset:
        """Create a YoloDataset instance from this config."""
        return YoloDataset(self.model_copy(update={"mode": mode}), root_path, split)
    
    def create_loader(
        self,
        batch_size: int,
        num_workers: int,
    ) -> YoloLoader:
        """Create a YoloLoader instance from this config."""
        from d_fine.dataset.yolo.yolo_loader import YoloLoader
        
        return YoloLoader(
            cfg=self,
            batch_size=batch_size,
            num_workers=num_workers,
        )


class YoloDataset(BaseDataset):
    def __init__(
        self,
        config: YoloDatasetConfig,
        root_path: Path,
        split: list[str],
    ) -> None:
        self.config = config
        self.project_path = config.base_path
        self.root_path = root_path
        self.split = split
        self.debug_img_processing = config.debug_img_processing
        self.ignore_background = False
        self.return_masks = config.task == Task.SEGMENT
        self.cases_to_debug = 20
        self.examples_per_epoch = config.examples_per_epoch

        self._img_config = config.img_config
        
        self.transform = init_augs(self._img_config, config.mode, config)
        self.mosaic_transform = A.Compose([
            A.Normalize(mean=self._img_config.norm[0], std=self._img_config.norm[1]),
            ToTensorV2(),
        ])

        self.debug_img_path = config.debug_img_path
        self._target_h = self._img_config.target_h
        self._target_w = self._img_config.target_w

    @property
    def label_to_name(self) -> dict[int, str]:
        """Return mapping from class ID to class name."""
        return self.config.label_to_name

    @property
    def base_size(self) -> int:
        return len(self.split)

    def get_data(self, idx: int) -> tuple[np.ndarray, np.ndarray, torch.Tensor, list[np.ndarray]]:
        """Load data from YOLO format labels."""
        image_filename = self.split[idx]
        image_path = self.root_path / "images" / image_filename
        image = cv2.imread(str(image_path))
        assert image is not None, f"Image wasn't loaded: {image_path}"

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape
        orig_size = torch.tensor([height, width])

        targets = np.zeros((0, 5), dtype=np.float32)
        masks = []

        labels_path = self.root_path / "labels" / f"{Path(image_filename).stem}.txt"
        if labels_path.exists() and labels_path.stat().st_size > 1:
            boxes_norm, polys_norm = parse_yolo_label_file(labels_path)

            xyxy_abs = norm_xywh_to_abs_xyxy(boxes_norm[:, 1:5], height, width).astype(np.float32)
            targets = np.concatenate([boxes_norm[:, [0]], xyxy_abs], axis=1)
            
            polys_abs = [norm_poly_to_abs(p, height, width) for p in polys_norm]
            masks = [poly_abs_to_mask(p, height, width) for p in polys_abs]
        
        return image, targets, orig_size, masks

    @property
    def mode(self) -> Mode:
        return self.config.mode

    def has_mosaic(self) -> bool:
        return self.config.mosaic_augs.mosaic_prob > 0.0

    def close_mosaic(self):
        if is_main_process():
            logger.info("Closing mosaic")

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Path, torch.Tensor]:
        actual_idx = idx % self.base_size
        
        image_filename = self.split[actual_idx]
        if random.random() < self.config.mosaic_augs.mosaic_prob:
            image, labels, boxes, masks_t, orig_size = self._load_mosaic(actual_idx)
            image_path = Path("mosaic")
        else:
            image, targets, orig_size, masks = self.get_data(actual_idx)
            image_path = Path(image_filename)

            if self.ignore_background and np.all(targets == 0) and self.config.mode == Mode.TRAIN:
                return None

            if targets.shape[0]:
                box_heights = targets[:, 3] - targets[:, 1]
                box_widths = targets[:, 4] - targets[:, 2]
                keep = np.minimum(box_heights, box_widths) > 0
                targets = targets[keep]
                masks = [m for m, k in zip(masks, keep) if k]
            else:
                masks = []

            if self.return_masks:
                transformed = self.transform(
                    image=image, bboxes=targets[:, 1:], class_labels=targets[:, 0], masks=masks
                )
                transformed_masks = transformed.get("masks", [])
                if len(transformed_masks):
                    masks_t = torch.stack([m.squeeze().to(dtype=torch.uint8) for m in transformed_masks], dim=0)
                else:
                    masks_t = torch.zeros(
                        (0, transformed["image"].shape[1], transformed["image"].shape[2]),
                        dtype=torch.uint8,
                    )
            else:
                transformed = self.transform(
                    image=image, bboxes=targets[:, 1:], class_labels=targets[:, 0]
                )
                masks_t = torch.zeros(
                    (0, transformed["image"].shape[1], transformed["image"].shape[2]),
                    dtype=torch.uint8,
                )

            image = transformed["image"]
            boxes = torch.tensor(transformed["bboxes"], dtype=torch.float32)
            labels = torch.tensor(transformed["class_labels"], dtype=torch.int64)

        boxes = torch.tensor(
            abs_xyxy_to_norm_xywh(boxes, image.shape[1], image.shape[2]), dtype=torch.float32
        )
        return image, labels, boxes, masks_t, image_path, orig_size

    def __len__(self):
        if self.examples_per_epoch is not None and self.config.mode == Mode.TRAIN:
            return self.examples_per_epoch
        return len(self.split)

    def _load_mosaic(self, idx):
        mosaic_info = load_mosaic(
            self.base_size,
            idx,
            self.get_data,
            self._target_h,
            self._target_w,
            self._img_config,
            self.config.mosaic_augs,
        )
        
        mosaic_img = None
        mosaic_polygons = []
        for i_mosaic, (m_idx, info) in enumerate(zip(mosaic_info.indices, mosaic_info.per_image_info)):
            img, _, _, masks_polys = self.get_data(m_idx)
            (h, w, c) = img.shape[:3]
            
            img = info.resize_image(img)
            (h, w, c) = img.shape[:3]
            
            if mosaic_img is None:
                mosaic_img = np.full((self._target_h * 2, self._target_w * 2, c), 114, dtype=np.uint8)
            
            info.paste(img, mosaic_img)
            
            if len(masks_polys):
                # Placeholder for mask transformation in mosaic
                pass
        
        mosaic_img, mosaic_targets, _ = random_affine(
            mosaic_img,
            mosaic_info.mosaic_targets if len(mosaic_info.mosaic_targets) else np.zeros((0, 5), dtype=np.float32),
            [],
            target_size=(self._target_w, self._target_h),
            degrees=self.config.mosaic_augs.degrees,
            translate=self.config.mosaic_augs.translate,
            scales=self.config.mosaic_augs.mosaic_scale,
            shear=self.config.mosaic_augs.shear,
        )
        
        image = self.mosaic_transform(image=mosaic_img)["image"]
        labels = torch.tensor(mosaic_targets[:, 0], dtype=torch.int64)
        boxes = torch.tensor(mosaic_targets[:, 1:], dtype=torch.float32)
        masks_t = torch.zeros((0, self._target_h, self._target_w), dtype=torch.uint8)
        
        return image, labels, boxes, masks_t, torch.tensor([self._target_h, self._target_w])
