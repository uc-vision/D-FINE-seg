from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset

from d_fine.config import Mode


class Dataset(TorchDataset, ABC):
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
    def get_data(self, idx: int) -> RawSample:
        """Load and return raw data for a given index."""
        pass



class Loader(ABC):
    @property
    @abstractmethod
    def label_to_name(self) -> dict[int, str]:
        """Return mapping from class ID to class name."""
        pass

    @property
    def num_classes(self) -> int:
        """Return the number of classes."""
        return len(self.label_to_name)

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

