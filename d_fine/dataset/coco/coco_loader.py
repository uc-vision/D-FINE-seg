from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader

from loguru import logger

from d_fine.core.dist_utils import is_main_process
from d_fine.config import Mode
from d_fine.dataset.base import BaseLoader
from d_fine.dataset.loader_utils import build_dataloader_impl, collate_fn, train_collate_fn
from d_fine.dataset.utils import get_splits
from d_fine.dataset.coco.coco_dataset import CocoDatasetConfig


class CocoLoader(BaseLoader):
    def __init__(
        self,
        cfg: CocoDatasetConfig,
        batch_size: int,
        num_workers: int,
    ) -> None:
        self.cfg = cfg
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_path = cfg.data_path
        self.img_size = cfg.img_size
        self.splits = get_splits(cfg.data_path)
        self.multiscale_prob = cfg.augs.multiscale_prob
        self.train_sampler = None

    def build_dataloaders(
        self, distributed: bool = False
    ) -> tuple[DataLoader, DataLoader, DataLoader | None]:
        
        train_ds = self.cfg.create_dataset(
            root_path=self.root_path,
            split=self.splits["train"],
            mode=Mode.TRAIN,
        )
        val_ds = self.cfg.create_dataset(
            root_path=self.root_path,
            split=self.splits["val"],
            mode=Mode.VAL,
        )

        train_collate = lambda b: train_collate_fn(b, self.multiscale_prob)
        train_loader, self.train_sampler = build_dataloader_impl(
            train_ds, self.batch_size, self.num_workers, train_collate, shuffle=True, distributed=distributed
        )
        val_loader, _ = build_dataloader_impl(
            val_ds, self.batch_size, self.num_workers, collate_fn, shuffle=False, distributed=distributed
        )

        test_loader = None
        if len(self.splits["test"]):
            test_ds = self.cfg.create_dataset(
                root_path=self.root_path,
                split=self.splits["test"],
                mode=Mode.TEST,
            )
            test_loader, _ = build_dataloader_impl(
                test_ds, self.batch_size, self.num_workers, collate_fn, shuffle=False, distributed=distributed
            )

        if is_main_process():
            logger.info(
                f"Images in train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds) if len(self.splits['test']) else 0}"
            )
        return train_loader, val_loader, test_loader

