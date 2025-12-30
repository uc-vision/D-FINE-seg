from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch.nn as nn


class Logger(ABC):
  """Abstract base class for all loggers."""

  @abstractmethod
  def step(self, epoch: int) -> None:
    """Set current epoch/step."""
    pass

  @abstractmethod
  def log_config(self, config: dict[str, Any]) -> None:
    """Log configuration parameters."""
    pass

  @abstractmethod
  def log_values(self, category: str, data: dict[str, float]) -> None:
    """Log multiple values under a category."""
    pass

  @abstractmethod
  def log_value(self, category: str, name: str, value: float) -> None:
    """Log a single value under a category."""
    pass

  @abstractmethod
  def log_images(
    self, category: str, name: str, images: list[np.ndarray], captions: list[str] | None = None
  ) -> None:
    """Log multiple visual debug information."""
    pass

  @abstractmethod
  def watch(self, model: nn.Module) -> None:
    """Watch model for gradients and parameters."""
    pass


class NullLogger(Logger):
  """Logger that does nothing."""

  def step(self, epoch: int) -> None:
    pass

  def log_config(self, config: dict[str, Any]) -> None:
    pass

  def log_values(self, category: str, data: dict[str, float]) -> None:
    pass

  def log_value(self, category: str, name: str, value: float) -> None:
    pass

  def log_images(
    self, category: str, name: str, images: list[np.ndarray], captions: list[str] | None = None
  ) -> None:
    pass

  def watch(self, model: nn.Module) -> None:
    pass
