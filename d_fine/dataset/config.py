from __future__ import annotations

import numpy as np
import torch
from pydantic import BaseModel


class AugConfig(BaseModel, frozen=True):
    """Augmentation configuration for mosaic."""
    mosaic_prob: float
    mosaic_scale: tuple[float, float]
    degrees: float
    translate: float
    shear: float


class ImageConfig(BaseModel, frozen=True):
    """Image processing configuration."""
    img_size: tuple[int, int]
    norm_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)  # ImageNet mean
    norm_std: tuple[float, float, float] = (0.229, 0.224, 0.225)  # ImageNet std
    keep_ratio: bool

    @property
    def target_w(self) -> int:
        """Return target width."""
        return self.img_size[0]

    @property
    def target_h(self) -> int:
        """Return target height."""
        return self.img_size[1]

    @property
    def norm(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Return normalization parameters as (mean, std)."""
        return (self.norm_mean, self.norm_std)

    def normalize(self, image: np.ndarray) -> np.ndarray:
        mean = np.array(self.norm_mean).reshape(-1, 1, 1)
        std = np.array(self.norm_std).reshape(-1, 1, 1)
        return (image - mean) / std

    def denormalize(self, image: torch.Tensor) -> np.ndarray:
        mean = np.array(self.norm_mean).reshape(-1, 1, 1)
        std = np.array(self.norm_std).reshape(-1, 1, 1)
        image_np = image.cpu().numpy()
        return (image_np * std) + mean

