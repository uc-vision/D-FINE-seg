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
from d_fine.dataset.mask_types import CompactMasks, RawSample, ProcessedSample
from image_detection.annotation.coco import load_coco_json

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


class CocoDatasetConfig(DatasetConfig, frozen=True):
    """COCO dataset configuration."""
    
    img_config: ImageConfig
    train_ann: str = "instances_train"
    val_ann: str = "instances_val"
    test_ann: str = "instances_test"
    
    def create_dataset(
        self,
        root_path: Path,
        annotation_file: Path,
        mode: Mode,
    ) -> CocoDataset:
        """Create a CocoDataset instance from this config."""
        return CocoDataset(self.model_copy(update={"mode": mode}), root_path, annotation_file)
    
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
        annotation_file: Path,
    ) -> None:
        self.config = config
        self.project_path = config.base_path
        self.root_path = root_path
        self.annotation_file = annotation_file
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
        
        self.coco_dataset = load_coco_json(self.annotation_file)
        
        from operator import attrgetter
        sorted_categories = sorted(self.coco_dataset.categories, key=attrgetter("id"))
        self.coco_category_to_class_id = {
            cat.id: idx for idx, cat in enumerate(sorted_categories)
        }
        
        self._label_to_name = {
            self.coco_category_to_class_id[cat.id]: cat.name
            for cat in sorted_categories
        }

    @property
    def mode(self) -> Mode:
        return self.config.mode

    @property
    def label_to_name(self) -> dict[int, str]:
        """Return mapping from class ID to class name."""
        return self._label_to_name

    @property
    def base_size(self) -> int:
        return len(self.coco_dataset.images)

    def _transform_and_format(self, sample: RawSample) -> ProcessedSample:
        """Consistently apply transformations and format outputs."""
        return ProcessedSample.from_transform(self.transform(**sample.prepare_for_transform()))

    def get_data(self, idx: int) -> RawSample:
        """Load data from COCO format annotations."""
        coco_image = self.coco_dataset.images[idx]
        image_path = self.root_path / "images" / coco_image.file_name
        image = cv2.imread(str(image_path))
        assert image is not None, f"Image wasn't loaded: {image_path}"

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        orig_size = torch.tensor([height, width])

        annotations = self.coco_dataset.annotations_by_image_id.get(coco_image.id, [])
        id_map = np.zeros((height, width), dtype=np.uint16)
        targets_list = []
        
        for i, ann in enumerate(annotations):
            if ann.category_id not in self.coco_category_to_class_id: continue
            
            if self.return_masks:
                inst = ann.to_instance_mask(height, width, torch.device("cpu"))
                x, y = inst.offset
                h_inst, w_inst = inst.mask.shape
                y1, y2 = max(0, y), min(height, y + h_inst)
                x1, x2 = max(0, x), min(width, x + w_inst)
                if y2 > y1 and x2 > x1:
                    m_cropped = inst.mask.cpu().numpy()[0:(y2-y1), 0:(x2-x1)]
                    id_map[y1:y2, x1:x2][m_cropped] = i + 1
            
            x, y, w, h = ann.bbox
            targets_list.append([self.coco_category_to_class_id[ann.category_id], x, y, x + w, y + h])
            
        return RawSample(
            image=image, 
            targets=np.array(targets_list, dtype=np.float32).reshape(-1, 5), 
            orig_size=orig_size, 
            id_map=id_map
        )

    def _load_mosaic(self, idx: int) -> ProcessedSample:
        mosaic_info = load_mosaic(
            self.base_size, idx,
            lambda i: (s := self.get_data(i), (s.image, s.targets, s.orig_size, s.id_map))[1],
            self._target_h, self._target_w, self._img_config, self.config.mosaic_augs,
        )
        
        mosaic_img = np.full((self._target_h * 2, self._target_w * 2, 3), 114, dtype=np.uint8)
        mosaic_id_map = np.zeros((self._target_h * 2, self._target_w * 2), dtype=np.uint16)
        
        inst_counter = 0
        for i_mosaic, (m_idx, info) in enumerate(zip(mosaic_info.indices, mosaic_info.per_image_info)):
            sample = self.get_data(m_idx)
            info.paste(info.resize_image(sample.image), mosaic_img)
            
            if self.return_masks and sample.id_map.any():
                m_warped = info.resize_image(sample.id_map, interpolation=cv2.INTER_NEAREST)
                mask = m_warped > 0
                mosaic_id_map[info.y1:info.y2, info.x1:info.x2][mask] = m_warped[mask] + inst_counter
            inst_counter += len(sample.targets)
        
        M = mosaic_info.affine_matrix
        if (M != np.eye(3)).any():
            mosaic_img = cv2.warpAffine(mosaic_img, M[:2], dsize=(self._target_w, self._target_h), borderValue=(114, 114, 114))
            if self.return_masks:
                mosaic_id_map = cv2.warpAffine(mosaic_id_map, M[:2], dsize=(self._target_w, self._target_h), borderValue=0, interpolation=cv2.INTER_NEAREST)
        
        targets = mosaic_info.mosaic_targets
        if len(targets) > 0:
            xy = np.ones((len(targets) * 4, 3), dtype=np.float32)
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(-1, 2)
            xy = xy @ M.T
            xy = xy[:, :2].reshape(-1, 8)
            new_boxes = np.stack([xy[:, [0, 2, 4, 6]].min(1), xy[:, [1, 3, 5, 7]].min(1), 
                                 xy[:, [0, 2, 4, 6]].max(1), xy[:, [1, 3, 5, 7]].max(1)], axis=1)
            
            from d_fine.utils import box_candidates
            keep = box_candidates(box1=targets[:, 1:5].T, box2=new_boxes.T, area_thr=0.1)
            targets, original_indices = targets[keep], np.where(keep)[0]
        else:
            targets, original_indices = np.zeros((0, 5)), np.zeros(0, dtype=np.int64)
        
        image = self.mosaic_transform(image=mosaic_img)["image"]
        return ProcessedSample(
            image=image, 
            labels=torch.as_tensor(targets[:, 0], dtype=torch.int64), 
            boxes=torch.as_tensor(targets[:, 1:], dtype=torch.float32), 
            masks=CompactMasks(id_map=torch.as_tensor(mosaic_id_map, dtype=torch.uint16), 
                               indices=torch.as_tensor(original_indices, dtype=torch.int64))
        )

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, CompactMasks, Path, torch.Tensor]:
        actual_idx = idx % self.base_size
        
        if random.random() < self.config.mosaic_augs.mosaic_prob:
            processed = self._load_mosaic(actual_idx)
            image_path, orig_size = Path("mosaic"), torch.tensor([self._target_h, self._target_w])
        else:
            sample = self.get_data(actual_idx)
            if self.ignore_background and len(sample.targets) == 0 and self.config.mode == "train":
                sample = sample.model_copy(update={"targets": np.zeros((0, 5))}) # Simplified

            processed = self._transform_and_format(sample)
            image_path, orig_size = Path(self.coco_dataset.images[actual_idx].file_name), sample.orig_size

        norm_boxes = torch.tensor(
            abs_xyxy_to_norm_xywh(processed.boxes.numpy(), processed.image.shape[2], processed.image.shape[1]), 
            dtype=torch.float32
        )
        return processed.image, processed.labels, norm_boxes, processed.masks, image_path, orig_size

    def __len__(self):
        if self.examples_per_epoch is not None and self.config.mode == "train":
            return self.examples_per_epoch
        return len(self.coco_dataset.images)
