"""Pydantic models for D-FINE training configuration."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import numpy as np
import torch
from pydantic import BaseModel, Field, field_validator


class Task(str, Enum):
    """Task type enum."""
    DETECT = "detect"
    SEGMENT = "segment"


class Mode(str, Enum):
    """Dataset mode enum."""
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    BENCH = "bench"


class PretrainedDataset(str, Enum):
    """Pretrained dataset enum."""
    COCO = "coco"
    OBJ2COCO = "obj2coco"


class Device(str, Enum):
    """Device enum."""
    CPU = "cpu"
    CUDA = "cuda"


class DecisionMetric(str, Enum):
    """Decision metric enum."""
    F1 = "f1"
    MAP_50 = "mAP_50"
    MAP = "mAP"


class MosaicAugsConfig(BaseModel, frozen=True):
    """Mosaic augmentation configuration."""
    mosaic_prob: float = 0.8
    no_mosaic_epochs: int = 5
    mosaic_scale: tuple[float, float] = (0.5, 1.5)
    degrees: float = 0.0
    translate: float = 0.2
    shear: float = 2.0


class AugsConfig(BaseModel, frozen=True):
    """General augmentation configuration."""
    
    class RotationAug(BaseModel, frozen=True):
        prob: float = 0.0
        degree: float = 10.0
    
    class BrightnessAug(BaseModel, frozen=True):
        prob: float = 0.02
        limit: float = 0.2
    
    class GammaAug(BaseModel, frozen=True):
        prob: float = 0.02
        limit: tuple[float, float] = (80, 120)
    
    class BlurAug(BaseModel, frozen=True):
        prob: float = 0.01
        limit: int = 3
    
    class NoiseAug(BaseModel, frozen=True):
        prob: float = 0.01
        std_range: tuple[float, float] = (0.1, 0.2)
    
    class HSVAug(BaseModel, frozen=True):
        prob: float = 0.0
        hue_shift_limit: float = 20.0
        sat_shift_limit: float = 30.0
        val_shift_limit: float = 20.0
    
    rotation: RotationAug = RotationAug()
    shift_limit: float = 0.1
    scale_limit: tuple[float, float] = (-0.2, 0.2)
    multiscale_prob: float = 0.0
    rotate_90_prob: float = 0.05
    left_right_flip: float = 0.3
    up_down_flip: float = 0.0
    hsv: HSVAug = HSVAug()
    to_gray_prob: float = 0.01
    blur: BlurAug = BlurAug()
    gamma: GammaAug = GammaAug()
    brightness: BrightnessAug = BrightnessAug()
    noise: NoiseAug = NoiseAug()
    coarse_dropout_prob: float = 0.0


class DDPConfig(BaseModel, frozen=True):
    """Distributed Data Parallel configuration."""
    enabled: bool = False
    n_gpus: int = 2


class PathConfig(BaseModel, frozen=True):
    """Path configuration."""
    
    base_path: Path
    path_to_save: Path
    infer_path: Path


class ExportConfig(BaseModel, frozen=True):
    """Export configuration."""
    half: bool = True
    max_batch_size: int = 1
    dynamic_input: bool = False
    ov_int8_max_drop: float = 0.01


class ImageConfig(BaseModel, frozen=True):
    """Image processing configuration."""
    img_size: tuple[int, int]
    norm_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)  # ImageNet mean
    norm_std: tuple[float, float, float] = (0.229, 0.224, 0.225)  # ImageNet std
    keep_ratio: bool
    use_crop: bool = False

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


class ModelConfig(BaseModel, frozen=True):
    """Common model inference configuration."""
    model_name: str
    n_outputs: int
    input_width: int
    input_height: int
    conf_thresh: float
    rect: bool
    half: bool
    keep_ratio: bool
    enable_mask_head: bool
    dtype: str = "float32"
    device: str
    num_top_queries: int = 300
    use_focal_loss: bool = True

    @property
    def np_dtype(self):
        import numpy as np
        return np.dtype(self.dtype).type

    @property
    def torch_dtype(self):
        import torch
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "int32": torch.int32,
            "int8": torch.int8,
        }
        return dtype_map.get(self.dtype, torch.float32)


class InferConfig(BaseModel, frozen=True):
    """Inference configuration."""
    to_crop: bool = True
    paddings: dict[str, float] = {"w": 0.05, "h": 0.05}


class DatasetConfig(BaseModel, frozen=True):
    """Base dataset configuration."""
    
    data_path: Path
    path_to_test_data: Path
    debug_img_processing: bool = False
    base_path: Path
    task: Task
    examples_per_epoch: int | None = None
    debug_img_path: Path | None = None
    mode: Mode = Mode.TRAIN
    mosaic_augs: MosaicAugsConfig = MosaicAugsConfig()
    augs: AugsConfig = AugsConfig()
    img_config: ImageConfig
    
    @property
    def img_size(self) -> tuple[int, int]:
        """Derive img_size from img_config."""
        return self.img_config.img_size
    
    @property
    def keep_ratio(self) -> bool:
        """Derive keep_ratio from img_config."""
        return self.img_config.keep_ratio


class TrainConfig(BaseModel, frozen=True):
    """Training configuration."""
    
    # Project and experiment names
    project_name: str
    exp_name: str
    
    # Paths
    paths: PathConfig
    
    # Dataset (will be instantiated)
    dataset: DatasetConfig
    
    # Export and inference configs
    export: ExportConfig = ExportConfig()
    infer: InferConfig = InferConfig()
    
    # Model name
    model_name: str = "s"
    
    # Model
    pretrained_dataset: PretrainedDataset = PretrainedDataset.COCO
    pretrained_model_path: Path | None = None
    
    # Training parameters
    use_wandb: bool = True
    device: Device = Device.CUDA
    conf_thresh: float = 0.5
    iou_thresh: float = 0.5
    epochs: int = 55
    batch_size: int = 8
    b_accum_steps: int = 1
    num_workers: int = 12
    mask_batch_size: int = 1000
    early_stopping: int = 0
    ignore_background_epochs: int = 0
    num_top_queries: int = 300
    use_focal_loss: bool = True
    
    # Image config
    img_size: tuple[int, int] = (640, 640)
    keep_ratio: bool = False
    to_visualize_eval: bool = True
    debug_img_processing: bool = True
    
    # Optimization
    amp_enabled: bool = True
    clip_max_norm: float = 0.1
    
    # Decision metrics
    decision_metrics: tuple[DecisionMetric, ...] = (DecisionMetric.F1, DecisionMetric.MAP_50)
    
    # DDP
    ddp: DDPConfig = DDPConfig()
    
    # Reproducibility
    seed: int = 42
    cudnn_fixed: bool = False
    
    # Learning rates (will be resolved from model_name)
    base_lr: float | None = None
    backbone_lr: float | None = None
    
    # Augmentations
    mosaic_augs: MosaicAugsConfig = MosaicAugsConfig()
    augs: AugsConfig = AugsConfig()
    
    # Optimizer
    cycler_pct_start: float = 0.1
    weight_decay: float = 0.000125
    betas: tuple[float, float] = (0.9, 0.999)
    label_smoothing: float = 0.0
    
    # EMA
    use_ema: bool = True
    ema_momentum: float = 0.9998

    def get_model_config(self, num_classes: int) -> ModelConfig:
        """Get common model inference configuration.
        
        Args:
            num_classes: Number of output classes from the dataset
        
        Returns:
            ModelConfig instance
        """
        return ModelConfig(
            model_name=self.model_name,
            n_outputs=num_classes,
            input_width=self.img_size[1],
            input_height=self.img_size[0],
            conf_thresh=self.conf_thresh,
            rect=self.export.dynamic_input,
            half=self.export.half,
            keep_ratio=self.keep_ratio,
            enable_mask_head=self.dataset.task == Task.SEGMENT,
            dtype="float32",
            device=str(self.device),
            num_top_queries=self.num_top_queries,
            use_focal_loss=self.use_focal_loss,
        )

    @classmethod
    def save(cls, config: TrainConfig, path: Path) -> None:
        """Save TrainConfig to JSON file."""
        with open(path, "w") as f:
            f.write(config.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: Path) -> TrainConfig:
        """Load TrainConfig from JSON file."""
        import json
        with open(path) as f:
            config_dict = json.load(f)
        return cls.model_validate(config_dict)

    @staticmethod
    def get_default_lrs(model_name: str) -> tuple[float, float]:
        """Get default learning rates for model_name.
        
        Args:
            model_name: Model size (n, s, m, l, x)
        
        Returns:
            Tuple of (base_lr, backbone_lr)
        """
        lrs = {
            "n": (0.0008, 0.0004),
            "s": (0.00025, 0.00006),
            "m": (0.00015, 0.00002),
            "l": (0.000125, 0.00000625),
            "x": (0.0001, 0.0000015),
        }
        if model_name not in lrs:
            raise ValueError(f"Unknown model_name: {model_name}. Must be one of {list(lrs.keys())}")
        return lrs[model_name]

    @classmethod
    def load_from_experiment(
        cls,
        base_path: Path,
        exp_name: str | None = None,
    ) -> TrainConfig:
        """Load TrainConfig from experiment directory.
        
        Args:
            base_path: Base project path
            exp_name: Experiment name (defaults to latest)
        
        Returns:
            TrainConfig instance
        """
        models_dir = base_path / "output" / "models"
        if not models_dir.exists():
            raise FileNotFoundError(f"Models directory not found: {models_dir}")
        
        def get_mtime(x):
            return x.stat().st_mtime

        experiments = sorted(
            [d for d in models_dir.iterdir() if d.is_dir()],
            key=get_mtime,
            reverse=True
        )
        if not experiments:
            raise FileNotFoundError(f"No experiments found in {models_dir}")
        
        if exp_name:
            exp_path = models_dir / exp_name
            if not exp_path.exists():
                available = [d.name for d in experiments]
                raise FileNotFoundError(
                    f"Experiment '{exp_name}' not found. Available experiments:\n  " + "\n  ".join(available)
                )
        else:
            exp_path = experiments[0]
        
        config_path = exp_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        return cls.load(config_path)
