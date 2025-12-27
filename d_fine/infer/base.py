from abc import ABC, abstractmethod
import numpy as np
from d_fine.core.types import ImageResult
from d_fine.config import ModelConfig, TrainConfig

class InferenceModel(ABC):
    """Abstract base class for all D-FINE inference models."""
    
    def __init__(self, model_config: ModelConfig, model_path: str):
        self.model_config = model_config
        self.model_path = model_path
        self.input_size = (model_config.input_height, model_config.input_width)
        self.rect = model_config.rect
        self.half = model_config.half
        self.keep_ratio = model_config.keep_ratio
        self.np_dtype = model_config.np_dtype
        self.device = model_config.device

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
    def from_train_config(cls, train_config: TrainConfig) -> "InferenceModel":
        """Factory method to create a model from a training configuration."""
        pass
