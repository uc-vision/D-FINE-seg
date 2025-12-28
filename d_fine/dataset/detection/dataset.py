from __future__ import annotations
import random
from pathlib import Path

import numpy as np
import torch
from lib_detection.annotation.coco import load_coco_json

from d_fine.config import Mode, ImageConfig
from d_fine.dataset.dataset import Dataset, ProcessedSample
from d_fine.dataset.detection.sample import DetectionSample
from d_fine.dataset.coco.coco_dataset import CocoDatasetConfig, load_coco_sample
from d_fine.dataset.augmentations import init_augs
from d_fine.core.box_utils import get_transform_matrix
from d_fine.dataset.mosaic import _get_mosaic_params


def assemble_mosaic(
  samples: list[DetectionSample], target_h: int, target_w: int, img_config: ImageConfig
) -> DetectionSample:
  """Specialized mosaic assembly for object detection."""
  ch, cw, infos, paths = _get_mosaic_params(samples, target_h, target_w, img_config)

  mosaic_img = np.full((ch, cw, 3), 114, dtype=np.uint8)
  mosaic_targets = []
  for sample, info in zip(samples, infos):
    info.paste(info.resize_image(sample.image), mosaic_img)
    if sample.targets.size > 0:
      t = sample.targets.copy()
      t[:, 1:5] = t[:, 1:5] * [info.scale[0], info.scale[1], info.scale[0], info.scale[1]] + [
        info.pad[0],
        info.pad[1],
        info.pad[0],
        info.pad[1],
      ]
      mosaic_targets.append(t)

  res_targets = (
    np.concatenate(mosaic_targets, 0) if mosaic_targets else np.zeros((0, 5), dtype=np.float32)
  )
  res_targets[:, [1, 3]] = np.clip(res_targets[:, [1, 3]], 0, cw)
  res_targets[:, [2, 4]] = np.clip(res_targets[:, [2, 4]], 0, ch)

  m_cfg = img_config.mosaic_augs
  M, _ = get_transform_matrix(
    (ch, cw), (target_w, target_h), m_cfg.degrees, m_cfg.mosaic_scale, m_cfg.shear, m_cfg.translate
  )

  res = DetectionSample(
    image=mosaic_img, targets=res_targets, orig_size=torch.tensor([target_h, target_w]), paths=paths
  )
  return res.warp_affine(M, (target_w, target_h))


class DetectionDataset(Dataset[DetectionSample]):
  """Dataset for object detection (bounding boxes)."""

  def __init__(self, config: CocoDatasetConfig, annotation_file: Path) -> None:
    self.config = config
    self.coco = load_coco_json(annotation_file)
    self.cat_to_id = {cat.id: i for i, cat in enumerate(self.coco.categories)}
    self._label_to_name = {i: cat.name for i, cat in enumerate(self.coco.categories)}
    self.transform = init_augs(config.img_config, config.mode)
    self.mosaic_transform = init_augs(config.img_config, config.mode)

  @property
  def mode(self) -> Mode:
    return self.config.mode

  @property
  def label_to_name(self) -> dict[int, str]:
    return self._label_to_name

  def denormalize(self, image: torch.Tensor) -> np.ndarray:
    return self.config.img_config.denormalize(image)

  def __len__(self) -> int:
    if self.config.examples_per_epoch and self.mode == Mode.TRAIN:
      return self.config.examples_per_epoch
    return len(self.coco.images)

  def get_data(self, idx: int) -> DetectionSample:
    image, anns, path = load_coco_sample(self.coco, self.config.base_path, self.cat_to_id, idx)
    targets = np.array(
      [
        [self.cat_to_id[a.category_id]]
        + list(a.bbox[:2])
        + [a.bbox[0] + a.bbox[2], a.bbox[1] + a.bbox[3]]
        for a in anns
      ],
      dtype=np.float32,
    ).reshape(-1, 5)
    return DetectionSample(
      image=image, targets=targets, paths=(path,), orig_size=torch.tensor(image.shape[:2])
    )

  def __getitem__(self, idx: int) -> ProcessedSample:
    img_cfg = self.config.img_config
    if random.random() < img_cfg.mosaic_augs.mosaic_prob and self.mode == Mode.TRAIN:
      indices = [idx] + [random.randint(0, len(self.coco.images) - 1) for _ in range(3)]
      s = assemble_mosaic(
        [self.get_data(i) for i in indices], img_cfg.target_h, img_cfg.target_w, img_cfg
      )
      return s.apply_transform(self.mosaic_transform)
    else:
      s = self.get_data(idx)
      return s.apply_transform(self.transform)
