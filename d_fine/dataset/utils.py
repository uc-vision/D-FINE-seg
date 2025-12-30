from d_fine.config import DatasetConfig, Mode
from d_fine.dataset.dataset import Loader


def get_loader_class(cfg: DatasetConfig) -> type[Loader]:
  from d_fine.dataset.coco.coco_dataset import CocoLoader

  return CocoLoader
