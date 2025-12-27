from __future__ import annotations

from functools import partial
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from loguru import logger

from d_fine.core.dist_utils import is_main_process
from d_fine.config import Mode
from d_fine.dataset.base import Loader
from d_fine.dataset.loader_utils import collate_fn, train_collate_fn
from d_fine.dataset.yolo.yolo_dataset import YoloDatasetConfig
from d_fine.utils import seed_worker


def get_splits(root_path: Path) -> dict[str, list[str]]:
    splits = {"train": [], "val": [], "test": []}
    for name in splits.keys():
        csv_path = root_path / f"{name}.csv"
        if csv_path.exists():
            with open(csv_path, "r") as f:
                splits[name] = [line.strip() for line in f if line.strip()]
    
    assert splits["train"] and splits["val"], "Train and Val splits must be present"
    return splits


class YoloLoader(Loader):
    def __init__(
        self,
        cfg: YoloDatasetConfig,
        batch_size: int,
        num_workers: int,
    ) -> None:
        self.cfg, self.batch_size, self.num_workers = cfg, batch_size, num_workers
        self.splits = get_splits(cfg.data_path)

    def build_dataloaders(
        self, distributed: bool = False
    ) -> tuple[DataLoader, DataLoader, DataLoader | None]:
        
        def build(m, collate, shuffle):
            s = self.splits[m.value]
            if m == Mode.TEST and not s: return None
            ds = self.cfg.create_dataset(self.cfg.data_path, s, m)
            sampler = DistributedSampler(ds, shuffle=shuffle, drop_last=False) if distributed else None
            return DataLoader(
                ds, self.batch_size, shuffle=shuffle if not distributed else False, sampler=sampler,
                num_workers=self.num_workers, collate_fn=collate, worker_init_fn=seed_worker,
                pin_memory=True, prefetch_factor=4 if self.num_workers > 0 else None
            )

        train = build(Mode.TRAIN, partial(train_collate_fn, multiscale_prob=self.cfg.augs.multiscale_prob), True)
        val = build(Mode.VAL, collate_fn, False)
        test = build(Mode.TEST, collate_fn, False)

        if is_main_process():
            logger.info(f"Images in train: {len(train.dataset)}, val: {len(val.dataset)}, test: {len(test.dataset) if test else 0}")
        return train, val, test
