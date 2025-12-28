from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from loguru import logger
from lib_detection.annotation.coco import CocoDataset as CocoFile

from d_fine.core.dist_utils import is_main_process
from d_fine.config import Mode, Task, DatasetConfig
from d_fine.dataset.dataset import Loader, Dataset, ProcessedSample
from d_fine.dataset.loader_utils import collate_fn, train_collate_fn
from d_fine.dataset.augmentations import init_augs
from d_fine.core.image_utils import load_image
from d_fine.core.utils import seed_worker


def load_coco_sample(
  coco: CocoFile, base_path: Path, cat_to_id: dict[int, int], idx: int
) -> tuple[np.ndarray, list, Path]:
  info = coco.images[idx % len(coco.images)]
  image = load_image(base_path / "images" / info.file_name)
  anns = [a for a in coco.annotations_by_image_id.get(info.id, []) if a.category_id in cat_to_id]
  return image, anns, Path(info.file_name)


class CocoDatasetConfig(DatasetConfig, frozen=True):
  base_path: Path
  train_ann: str = "instances_train"
  val_ann: str = "instances_val"
  test_ann: str = "instances_test"

  def create_dataset(self, annotation_file: Path, mode: Mode) -> Dataset:
    cfg = self.model_copy(update={"mode": mode})
    if self.task == Task.SEGMENT:
      from d_fine.dataset.segmentation.dataset import SegmentationDataset

      return SegmentationDataset(cfg, annotation_file)
    else:
      from d_fine.dataset.detection.dataset import DetectionDataset

      return DetectionDataset(cfg, annotation_file)

  def create_loader(self, batch_size: int, num_workers: int) -> "CocoLoader":
    return CocoLoader(cfg=self, batch_size=batch_size, num_workers=num_workers)


class CocoLoader(Loader):
  _dataset: Dataset

  def __init__(self, cfg: CocoDatasetConfig, batch_size: int, num_workers: int) -> None:
    self.cfg, self.batch_size, self.num_workers = cfg, batch_size, num_workers
    self._dataset = self.cfg.create_dataset(self._get_ann_path(Mode.TRAIN), Mode.TRAIN)

  @property
  def label_to_name(self) -> dict[int, str]:
    return self._dataset.label_to_name

  def _get_ann_path(self, m: Mode) -> Path:
    match m:
      case Mode.TRAIN:
        name = self.cfg.train_ann
      case Mode.VAL:
        name = self.cfg.val_ann
      case Mode.TEST:
        name = self.cfg.test_ann
    return (
      self.cfg.base_path / "annotations" / (f"{name}.json" if not name.endswith(".json") else name)
    )

  def _build(self, m: Mode, collate: partial, shuffle: bool, distributed: bool) -> DataLoader:
    p = self._get_ann_path(m)
    ds = self.cfg.create_dataset(p, m)
    sampler = DistributedSampler(ds, shuffle=shuffle, drop_last=False) if distributed else None
    return DataLoader(
      ds,
      self.batch_size,
      shuffle=shuffle if not distributed else False,
      sampler=sampler,
      num_workers=self.num_workers,
      collate_fn=collate,
      worker_init_fn=seed_worker,
      pin_memory=False,
      prefetch_factor=2 if self.num_workers > 0 else None,
    )

  def build_dataloaders(
    self, distributed: bool = False
  ) -> tuple[
    DataLoader[ProcessedSample], DataLoader[ProcessedSample], DataLoader[ProcessedSample] | None
  ]:
    train = self._build(
      Mode.TRAIN,
      partial(train_collate_fn, multiscale_config=self.cfg.multiscale),
      True,
      distributed,
    )
    val = self._build(Mode.VAL, partial(collate_fn), False, distributed)

    p_test = self._get_ann_path(Mode.TEST)
    test = (
      self._build(Mode.TEST, partial(collate_fn), False, distributed) if p_test.exists() else None
    )

    if is_main_process():
      logger.info(f"Images in train: {len(train.dataset)}, val: {len(val.dataset)}")
    return train, val, test
