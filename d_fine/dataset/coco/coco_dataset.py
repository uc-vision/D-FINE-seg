from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2

from loguru import logger

from d_fine.core.dist_utils import is_main_process
from d_fine.config import DatasetConfig, Mode, Task
from d_fine.dataset.base import BaseDataset
from d_fine.dataset.augmentations import init_augs
from d_fine.config import ImageConfig

if TYPE_CHECKING:
    from d_fine.dataset.coco.coco_loader import CocoLoader
from d_fine.dataset.mosaic import MosaicSource, load_mosaic, MosaicInfo
from d_fine.utils import get_transform_matrix
from d_fine.utils import (
    abs_xyxy_to_norm_xywh,
    get_mosaic_coordinate,
    norm_xywh_to_abs_xyxy,
    random_affine,
    vis_one_box,
)


def instance_mask_to_full_image_mask(
    instance_mask,
    image_height: int,
    image_width: int,
) -> np.ndarray:
    """Convert InstanceMask to full-image mask."""
    mask_np = instance_mask.mask.cpu().numpy().astype(np.uint8)
    x_offset, y_offset = instance_mask.offset
    mask_h, mask_w = mask_np.shape
    
    full_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    
    y_start = max(0, y_offset)
    x_start = max(0, x_offset)
    y_end = min(y_offset + mask_h, image_height)
    x_end = min(x_offset + mask_w, image_width)
    
    mask_y_start = max(0, -y_offset)
    mask_x_start = max(0, -x_offset)
    mask_y_end = mask_y_start + (y_end - y_start)
    mask_x_end = mask_x_start + (x_end - x_start)
    
    full_mask[y_start:y_end, x_start:x_end] = mask_np[mask_y_start:mask_y_end, mask_x_start:mask_x_end]
    
    return full_mask


def load_coco_dataset_if_available(root_path: Path) -> tuple[object | None, dict[int, int] | None]:
    """Load COCO dataset if annotations file exists."""
    coco_json_path = root_path / "annotations" / "instances_default.json"
    if not coco_json_path.exists():
        return None, None
    
    try:
        from image_detection.annotation.coco import load_coco_json
        coco_dataset = load_coco_json(coco_json_path)
        sorted_categories = sorted(coco_dataset.categories, key=lambda c: c.id)
        category_to_class_id = {
            cat.id: idx for idx, cat in enumerate(sorted_categories)
        }
        logger.info(f"Using COCO format: Loaded dataset from {coco_json_path}")
        logger.info(f"COCO categories: {len(sorted_categories)} classes")
        return coco_dataset, category_to_class_id
    except ImportError:
        logger.warning("COCO JSON found but image-detection package not available")
        return None, None
    except Exception as e:
        logger.warning(f"Failed to load COCO dataset: {e}")
        return None, None


def parse_coco_annotations(
    coco_dataset,
    image_id: int,
    image_height: int,
    image_width: int,
    category_to_class_id: dict[int, int],
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Parse COCO annotations for a single image."""
    try:
        from image_detection.annotation.instance_mask import InstanceMask
    except ImportError:
        raise ImportError("image-detection package required for COCO support")
    
    boxes_norm = []
    masks = []
    
    annotations = coco_dataset.annotations_by_image_id.get(image_id, [])
    if not annotations:
        return np.zeros((0, 5), dtype=np.float32), []
    
    image = coco_dataset.images_by_id.get(image_id)
    if image is None:
        return np.zeros((0, 5), dtype=np.float32), []
    
    for ann in annotations:
        if ann.category_id not in category_to_class_id:
            continue
        
        class_id = category_to_class_id[ann.category_id]
        
        instance_mask = ann.to_instance_mask(image_height, image_width, torch.device("cpu"))
        full_mask = instance_mask_to_full_image_mask(instance_mask, image_height, image_width)
        masks.append(full_mask)
        
        rows = np.any(full_mask, axis=1)
        cols = np.any(full_mask, axis=0)
        if not rows.any() or not cols.any():
            continue
        
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        xc = (x_min + x_max + 1) / 2 / image_width
        yc = (y_min + y_max + 1) / 2 / image_height
        w = (x_max - x_min + 1) / image_width
        h = (y_max - y_min + 1) / image_height
        
        boxes_norm.append([class_id, xc, yc, w, h])
    
    if len(boxes_norm) == 0:
        return np.zeros((0, 5), dtype=np.float32), []
    
    return np.asarray(boxes_norm, dtype=np.float32), masks


class CocoDatasetConfig(DatasetConfig, frozen=True):
    """COCO dataset configuration."""
    
    # label_to_name is derived from COCO categories and saved separately
    img_config: ImageConfig
    
    def create_dataset(
        self,
        root_path: Path,
        split: pd.DataFrame,
        mode: Mode,
    ) -> CocoDataset:
        """Create a CocoDataset instance from this config."""
        config_with_mode = self.model_copy(update={"mode": mode})
        return CocoDataset(config_with_mode, root_path, split)
    
    def create_loader(
        self,
        batch_size: int,
        num_workers: int,
    ) -> CocoLoader:
        """Create a CocoLoader instance from this config."""
        from d_fine.dataset.coco.coco_loader import CocoLoader
        
        return CocoLoader(
            cfg=self,
            batch_size=batch_size,
            num_workers=num_workers,
        )


class CocoDataset(BaseDataset):
    def __init__(
        self,
        config: CocoDatasetConfig,
        root_path: Path,
        split: pd.DataFrame,
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
        
        self.coco_dataset, self.coco_category_to_class_id = load_coco_dataset_if_available(self.root_path)
        if self.coco_dataset is None:
            raise ValueError(f"COCO dataset not found at {root_path / 'annotations' / 'instances_default.json'}")
        
        # Derive label_to_name from COCO categories
        sorted_categories = sorted(self.coco_dataset.categories, key=lambda c: c.id)
        self._label_to_name = {
            self.coco_category_to_class_id[cat.id]: cat.name
            for cat in sorted_categories
        }

    @property
    def label_to_name(self) -> dict[int, str]:
        """Return mapping from class ID to class name."""
        return self._label_to_name

    @property
    def base_size(self) -> int:
        return len(self.split)

    def get_data(self, idx) -> tuple[np.ndarray, np.ndarray, torch.Tensor, list[np.ndarray]]:
        """Load data from COCO format annotations."""
        image_path = Path(self.split.iloc[idx].values[0])
        image = cv2.imread(str(self.root_path / "images" / f"{image_path}"))
        assert image is not None, f"Image wasn't loaded: {image_path}"

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape
        orig_size = torch.tensor([height, width])

        targets = np.zeros((0, 5), dtype=np.float32)
        masks = []

        image_filename = image_path.name
        coco_image = None
        for img in self.coco_dataset.images:
            if img.file_name == image_filename:
                coco_image = img
                break
        
        if coco_image is not None:
            boxes_norm, masks_list = parse_coco_annotations(
                self.coco_dataset,
                coco_image.id,
                height,
                width,
                self.coco_category_to_class_id,
            )
            
            # use_one_class functionality removed - always use original class IDs
            
            xyxy_abs = norm_xywh_to_abs_xyxy(boxes_norm[:, 1:5], height, width).astype(np.float32)
            targets = np.concatenate([boxes_norm[:, [0]], xyxy_abs], axis=1)
            masks = masks_list
        
        return image, targets, orig_size, masks


    def _transform_masks_for_mosaic(self, masks, info: MosaicSource, h: int, w: int):
        transformed = []
        for mask in masks:
            if mask.ndim == 2:
                transformed.append(info.transform_mask(mask, h, w))
            else:
                transformed.append(info.create_canvas())
        return transformed

    def _transform_masks_affine(self, masks, M):
        if not masks or (M == np.eye(3)).all():
            return [m[:self._target_h, :self._target_w] if m.shape[0] > self._target_h or m.shape[1] > self._target_w else m for m in masks]
        return [
            cv2.warpAffine(m.astype(np.uint8), M[:2], dsize=(self._target_w, self._target_h), borderValue=0).astype(bool)
            for m in masks
        ]

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
        mosaic_masks = []
        for i_mosaic, (m_idx, info) in enumerate(zip(mosaic_info.indices, mosaic_info.per_image_info)):
            img, _, _, masks_polys = self.get_data(m_idx)
            (h, w, c) = img.shape[:3]
            
            img = info.resize_image(img)
            (h, w, c) = img.shape[:3]
            
            if mosaic_img is None:
                mosaic_img = np.full((self._target_h * 2, self._target_w * 2, c), 114, dtype=np.uint8)
            
            info.paste(img, mosaic_img)
            
            if len(masks_polys):
                transformed = self._transform_masks_for_mosaic(masks_polys, info, h, w)
                mosaic_masks.extend(transformed)
        
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
        
        if mosaic_targets.shape[0]:
            box_heights = mosaic_targets[:, 3] - mosaic_targets[:, 1]
            box_widths = mosaic_targets[:, 4] - mosaic_targets[:, 2]
            keep = np.minimum(box_heights, box_widths) > 1
            mosaic_targets = mosaic_targets[keep]
            mosaic_masks = [m for m, k in zip(mosaic_masks, keep) if k]
        
        image = self.mosaic_transform(image=mosaic_img)["image"]
        labels = torch.tensor(mosaic_targets[:, 0], dtype=torch.int64)
        boxes = torch.tensor(mosaic_targets[:, 1:], dtype=torch.float32)
        
        if self.return_masks and len(mosaic_masks):
            transformed_masks = self._transform_masks_affine(mosaic_masks, mosaic_info.affine_matrix)
            masks_t = torch.stack([torch.from_numpy(m).to(dtype=torch.uint8) for m in transformed_masks], dim=0)
        else:
            masks_t = torch.zeros((0, self._target_h, self._target_w), dtype=torch.uint8)
        
        return image, labels, boxes, masks_t, torch.tensor([self._target_h, self._target_w])

    @property
    def mode(self) -> Mode:
        return self.config.mode

    def has_mosaic(self) -> bool:
        return self.config.mosaic_augs.mosaic_prob > 0.0

    def close_mosaic(self):
        # Note: mosaic_prob is in config, cannot be modified directly
        # This would need to be handled differently if dynamic disabling is needed
        if is_main_process():
            logger.info("Closing mosaic")

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        actual_idx = idx % self.base_size
        
        image_path = Path(self.split.iloc[actual_idx].values[0])
        if random.random() < self.config.mosaic_augs.mosaic_prob:
            image, labels, boxes, masks_t, orig_size = self._load_mosaic(actual_idx)
        else:
            image, targets, orig_size, masks = self.get_data(actual_idx)

            if self.ignore_background and np.all(targets == 0) and self.config.mode == "train":
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
        if self.examples_per_epoch is not None and self.config.mode == "train":
            return self.examples_per_epoch
        return len(self.split)


