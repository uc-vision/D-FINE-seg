from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
from d_fine.core.types import ImageResult
from d_fine.config import EvaluationConfig, TrainConfig


class InferenceModel(ABC):
  """Abstract base class for all D-FINE inference models."""

  @abstractmethod
  def __call__(self, img: np.ndarray) -> ImageResult:
    """Run inference on a single RGB image.

    Args:
        img: RGB image array [H, W, 3]

    Returns:
        ImageResult object
    """
    pass

  @classmethod
  @abstractmethod
  def from_train_config(
    cls, train_config: TrainConfig, num_classes: int, model_path: Path
  ) -> "InferenceModel":
    """Factory method to create a model from a training configuration."""
    pass
