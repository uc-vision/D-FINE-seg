from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from d_fine.config import Mode


class BaseDataset(Dataset, ABC):
    @property
    @abstractmethod
    def mode(self) -> Mode:
        """Return the dataset mode: 'train', 'val', 'test', or 'bench'."""
        pass

    @property
    @abstractmethod
    def label_to_name(self) -> dict[int, str]:
        """Return mapping from class ID to class name.
        
        Returns:
            Dictionary mapping class IDs (int) to class names (str)
        """
        pass

    @property
    def num_classes(self) -> int:
        """Return the number of classes in the dataset.
        
        Returns:
            Number of classes
        """
        return len(self.label_to_name)

    @abstractmethod
    def get_data(
        self, idx: int
    ) -> tuple[np.ndarray, np.ndarray, torch.Tensor, list[np.ndarray]]:
        """Load and return raw data for a given index.
        
        Args:
            idx: Dataset index
            
        Returns:
            Tuple containing:
                - image: RGB image array of shape (H, W, 3), dtype uint8
                - targets: Bounding box array of shape (N, 5) with columns [class_id, x1, y1, x2, y2]
                  in absolute coordinates, dtype float32
                - orig_size: Tensor of shape (2,) containing [height, width], dtype int
                - masks: List of mask arrays, each of shape (H, W), dtype uint8
        """
        pass



class BaseLoader(ABC):
    @abstractmethod
    def build_dataloaders(
        self, distributed: bool = False
    ) -> tuple[DataLoader, DataLoader, DataLoader | None]:
        """Build and return data loaders for train, validation, and test splits.
        
        Args:
            distributed: Whether to use distributed sampling for training
            
        Returns:
            Tuple containing:
                - train_loader: DataLoader for training data
                - val_loader: DataLoader for validation data
                - test_loader: DataLoader for test data, or None if no test split exists
        """
        pass

