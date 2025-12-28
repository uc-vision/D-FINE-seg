from __future__ import annotations

from typing import Any

import numpy as np
import wandb
import torch.nn as nn

from .base import Logger


class WandbLogger(Logger):
  """Logger for Weights & Biases."""

  def __init__(self, project: str, name: str, config: dict[str, Any] | None = None):
    self.current_epoch = 0
    from d_fine.core.dist_utils import get_rank

    if get_rank() == 0:
      wandb.init(project=project, name=name, config=config)

  def step(self, epoch: int) -> None:
    self.current_epoch = epoch

  def log_config(self, config: dict[str, Any]) -> None:
    from d_fine.core.dist_utils import get_rank

    if get_rank() == 0:
      wandb.config.update(config)

  def log_values(self, category: str, data: dict[str, float]) -> None:
    from d_fine.core.dist_utils import get_rank

    if get_rank() == 0:
      log_dict = {f"{category}/{k}": v for k, v in data.items()}
      log_dict["epoch"] = self.current_epoch
      wandb.log(log_dict)

  def log_value(self, category: str, name: str, value: float) -> None:
    from d_fine.core.dist_utils import get_rank

    if get_rank() == 0:
      wandb.log({f"{category}/{name}": value, "epoch": self.current_epoch})

  def log_images(
    self, category: str, name: str, images: list[np.ndarray], captions: list[str] | None = None
  ) -> None:
    from d_fine.core.dist_utils import get_rank

    if get_rank() == 0 and images:
      wandb_images = []
      for i, img in enumerate(images):
        caption = captions[i] if captions and i < len(captions) else f"{name}_{i}"
        wandb_images.append(wandb.Image(img, caption=caption))

      wandb.log({f"{category}/{name}": wandb_images, "epoch": self.current_epoch})

  def watch(self, model: nn.Module) -> None:
    from d_fine.core.dist_utils import get_rank

    if get_rank() == 0:
      wandb.watch(model)
