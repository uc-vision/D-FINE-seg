from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from lib_detection.annotation import draw_instance
from lib_detection.annotation.coco import load_coco_json
from lib_detection.bounds import Bounds

from d_fine.config import ImageConfig, Mode
from d_fine.core.box_utils import get_transform_matrix
from d_fine.dataset.augmentations import init_augs
from d_fine.dataset.coco.coco_dataset import CocoDatasetConfig, load_coco_sample
from d_fine.dataset.dataset import Dataset, ProcessedSample
from d_fine.dataset.mosaic import _get_mosaic_params
from d_fine.dataset.segmentation.sample import SegmentationSample


def assemble_mosaic(
  samples: list[SegmentationSample], target_size: tuple[int, int], img_config: ImageConfig
) -> SegmentationSample:
  """Specialized mosaic assembly for instance segmentation."""
  ch, cw, infos, paths = _get_mosaic_params(samples, target_size, img_config)
  tw, th = target_size

  mosaic_img = np.full((ch, cw, 3), 114, dtype=np.uint8)
  mosaic_id_map = np.zeros((ch, cw), dtype=np.uint16)
  mosaic_labels = []

  inst_counter = 0
  for sample, info in zip(samples, infos):
    info.paste(info.resize_image(sample.image), mosaic_img)
    if sample.id_map.any():
      m_warped = cv2.resize(
        sample.id_map,
        (int(sample.image.shape[1] * info.scale[0]), int(sample.image.shape[0] * info.scale[1])),
        interpolation=cv2.INTER_NEAREST,
      )
      cropped = m_warped[
        info.small_coords[1] : info.small_coords[3], info.small_coords[0] : info.small_coords[2]
      ]
      mask = cropped > 0
      mosaic_id_map[
        info.large_coords[1] : info.large_coords[3], info.large_coords[0] : info.large_coords[2]
      ][mask] = (cropped[mask] + inst_counter)
    mosaic_labels.append(sample.labels)
    inst_counter += len(sample.labels)

  m_cfg = img_config.mosaic_augs
  M, _ = get_transform_matrix(
    (ch, cw), (tw, th), m_cfg.degrees, m_cfg.mosaic_scale, m_cfg.shear, m_cfg.translate
  )

  res = SegmentationSample(
    image=mosaic_img,
    labels=np.concatenate(mosaic_labels),
    id_map=mosaic_id_map,
    orig_size=torch.tensor([th, tw]),
    paths=paths,
  )
  return res.warp_affine(M, (tw, th))


class SegmentationDataset(Dataset[SegmentationSample]):
  """Dataset for instance segmentation (masks)."""

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

  def get_data(self, idx: int) -> SegmentationSample:
    image, anns, path = load_coco_sample(self.coco, self.config.base_path, self.cat_to_id, idx)
    h, w = image.shape[:2]
    labels = np.array([self.cat_to_id[a.category_id] for a in anns], dtype=np.int64)
    id_map = torch.zeros((h, w), dtype=torch.int32)
    img_bounds = Bounds.from_size(w, h)
    for i, a in enumerate(anns):
      m = a.to_instance_mask(h, w, torch.device("cpu"))
      if m.bounds.intersects(img_bounds):
        draw_instance(id_map, m.crop(img_bounds), i + 1)
    return SegmentationSample(
      image=image,
      labels=labels,
      id_map=id_map.to(torch.uint16).numpy(),
      paths=(path,),
      orig_size=torch.tensor([image.shape[0], image.shape[1]]),
    )

  def __getitem__(self, idx: int) -> ProcessedSample:
    img_cfg = self.config.img_config
    if random.random() < img_cfg.mosaic_augs.mosaic_prob and self.mode == Mode.TRAIN:
      indices = [idx] + [random.randint(0, len(self.coco.images) - 1) for _ in range(3)]
      s = assemble_mosaic([self.get_data(i) for i in indices], img_cfg.target_size, img_cfg)
      return s.apply_transform(self.mosaic_transform)
    else:
      s = self.get_data(idx)
      return s.apply_transform(self.transform)
