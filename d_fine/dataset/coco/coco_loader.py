from __future__ import annotations

from functools import partial
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from loguru import logger

from d_fine.core.dist_utils import is_main_process
from d_fine.config import Mode
from d_fine.dataset.base import Loader
from d_fine.dataset.loader_utils import collate_fn, train_collate_fn
from d_fine.dataset.coco.coco_dataset import CocoDatasetConfig
from d_fine.utils import seed_worker


class CocoLoader(Loader):
    def __init__(
        self,
        cfg: CocoDatasetConfig,
        batch_size: int,
        num_workers: int,
    ) -> None:
        self.cfg, self.batch_size, self.num_workers = cfg, batch_size, num_workers
        self._dataset = None

    @property
    def label_to_name(self) -> dict[int, str]:
        if self._dataset is None:
            # Use training annotation by default to get all labels
            p = self.cfg.data_path / "annotations" / (f"{self.cfg.train_ann}.json" if not self.cfg.train_ann.endswith(".json") else self.cfg.train_ann)
            self._dataset = self.cfg.create_dataset(self.cfg.data_path, p, Mode.TRAIN)
        return self._dataset.label_to_name

    def build_dataloaders(
        self, distributed: bool = False
    ) -> tuple[DataLoader, DataLoader, DataLoader | None]:
        def get_ann_path(m: Mode) -> Path:
            name = {
                Mode.TRAIN: self.cfg.train_ann,
                Mode.VAL: self.cfg.val_ann,
                Mode.TEST: self.cfg.test_ann,
            }[m]
            # Handle both name only or full filename
            filename = f"{name}.json" if not name.endswith(".json") else name
            return self.cfg.data_path / "annotations" / filename
        
        def build(m, collate, shuffle):
            p = get_ann_path(m)
            if m == Mode.TEST and not p.exists(): return None
            ds = self.cfg.create_dataset(self.cfg.data_path, p, m)
            sampler = DistributedSampler(ds, shuffle=shuffle, drop_last=False) if distributed else None
            return DataLoader(
                ds, self.batch_size, shuffle=shuffle if not distributed else False, sampler=sampler,
                num_workers=self.num_workers, collate_fn=collate, worker_init_fn=seed_worker,
                pin_memory=False, prefetch_factor=2 if self.num_workers > 0 else None
            )

        train = build(Mode.TRAIN, partial(train_collate_fn, multiscale_prob=self.cfg.augs.multiscale_prob), True)
        val = build(Mode.VAL, collate_fn, False)
        test = build(Mode.TEST, collate_fn, False)

        if is_main_process():
            logger.info(f"Images in train: {len(train.dataset)}, val: {len(val.dataset)}")
        return train, val, test
