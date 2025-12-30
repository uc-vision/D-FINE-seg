from __future__ import annotations

import torch
from loguru import logger

from d_fine.config import TrainConfig, Task, ClassConfig
from d_fine.dataset import get_loader_class
from d_fine.core import build_model


class Trainer:
  def __init__(self, config: TrainConfig):
    self.config = config
    self.device = torch.device(config.device)

    # Ensure output directory exists
    self.config.paths.path_to_save.mkdir(parents=True, exist_ok=True)

    loader_class = get_loader_class(config.dataset)
    self.loader = loader_class(
      config.dataset, batch_size=config.batch_size, num_workers=config.num_workers
    )

    # Save configuration and class mapping
    self.config.save(self.config.paths.path_to_save / "config.json")
    class_config = ClassConfig(
      label_to_name=self.loader.label_to_name,
      conf_thresh=self.config.conf_thresh,
      iou_thresh=self.config.iou_thresh,
    )
    class_config.save(self.config.paths.path_to_save / "classes.json")

    self.train_loader, self.val_loader, self.test_loader = self.loader.build_dataloaders()
    
    self.model = build_model(
      model_name=config.model_name,
      num_classes=self.loader.num_classes,
      enable_mask_head=config.dataset.task == Task.SEGMENT,
      device=self.device,
      img_size=config.dataset.img_config.img_size,
      pretrained_model_path=config.pretrained_model_path,
    )

  def train(self):
    logger.info(f"Starting training for {self.config.epochs} epochs")
    # ... training loop ...


def run_training(config: TrainConfig):
  trainer = Trainer(config)
  trainer.train()
