"""Pydantic models for D-FINE training configuration."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import numpy as np
import torch
from pydantic import BaseModel, Field, field_validator

import json
from typing import Any, Callable
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
import importlib.util


def config_path():
  spec = importlib.util.find_spec("d_fine")
  if not (spec and spec.origin):
    raise RuntimeError("Could not find d_fine package")
  return Path(spec.origin).parent / "config"


def _get_hydra_config(
  project_name: str, exp_name: str, base_path: Path, overrides: list[str] | None = None
) -> Any:
  from operator import getitem

  if not OmegaConf.has_resolver("lookup"):
    OmegaConf.register_new_resolver("lookup", getitem, replace=True)

  config_dir = config_path()

  override_list = [
    f"project_name={project_name}",
    f"exp_name={exp_name}",
    f"base_path={base_path}",
    "dataset=coco",
  ] + (overrides or [])

  with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
    return compose(config_name="config", overrides=override_list)


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


class LoggerType(str, Enum):
  """Logger type enum."""

  WANDB = "wandb"
  NONE = "none"


class DecisionMetric(str, Enum):
  """Decision metric enum."""

  F1 = "f1"
  MAP_50 = "mAP_50"
  MAP = "mAP"


class ClassConfig(BaseModel, frozen=True):
  """Class configuration including label mappings and thresholds."""

  label_to_name: dict[int, str]
  conf_thresh: float | None = None
  iou_thresh: float | None = None

  @classmethod
  def load(cls, path: Path) -> "ClassConfig":
    """Load class config from JSON file."""
    import json

    if not path.exists():
      raise FileNotFoundError(f"class_config.json not found at {path}")
    with open(path) as f:
      data = json.load(f)

    # Handle legacy label_to_name.json format
    if "label_to_name" not in data:
      # Assume entire file is label_to_name mapping
      label_to_name = {int(k): v for k, v in data.items()}
      return cls(label_to_name=label_to_name)

    # Convert label_to_name keys from string to int (JSON keys are strings)
    label_to_name = {int(k): v for k, v in data["label_to_name"].items()}
    return cls(
      label_to_name=label_to_name,
      conf_thresh=data.get("conf_thresh"),
      iou_thresh=data.get("iou_thresh"),
    )

  def save(self, path: Path) -> None:
    """Save class config to JSON file."""
    import json

    data = {"label_to_name": self.label_to_name}
    if self.conf_thresh is not None:
      data["conf_thresh"] = self.conf_thresh
    if self.iou_thresh is not None:
      data["iou_thresh"] = self.iou_thresh

    with open(path, "w") as f:
      json.dump(data, f, indent=2)


class MultiscaleConfig(BaseModel, frozen=True):
  """Multiscale training configuration."""

  prob: float = 0.0
  offset_range: tuple[int, ...] = (-2, -1, 1, 2)
  step_size: int = 32


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

  class Brightness(BaseModel, frozen=True):
    prob: float = 0.02
    limit: float = 0.2

  class Gamma(BaseModel, frozen=True):
    prob: float = 0.02
    limit: tuple[float, float] = (80, 120)

  class Blur(BaseModel, frozen=True):
    prob: float = 0.01
    limit: int = 3

  class Noise(BaseModel, frozen=True):
    prob: float = 0.01
    std_range: tuple[float, float] = (0.1, 0.2)

  class HSV(BaseModel, frozen=True):
    prob: float = 0.0
    hue_shift_limit: float = 20.0
    sat_shift_limit: float = 30.0
    val_shift_limit: float = 20.0

  class Geometric(BaseModel, frozen=True):
    shift_limit: float = 0.1
    scale_limit: tuple[float, float] = (-0.2, 0.2)
    rotate_limit: float = 10.0

  geometric: Geometric = Geometric()
  rotate_90_prob: float = 0.05
  left_right_flip: float = 0.3
  up_down_flip: float = 0.0
  hsv: HSV = HSV()
  to_gray_prob: float = 0.01
  blur: Blur = Blur()
  gamma: Gamma = Gamma()
  brightness: Brightness = Brightness()
  noise: Noise = Noise()
  coarse_dropout_prob: float = 0.0


class ImageConfig(BaseModel, frozen=True):
  """Image processing configuration."""

  img_size: tuple[int, int]
  norm_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)  # ImageNet mean
  norm_std: tuple[float, float, float] = (0.229, 0.224, 0.225)  # ImageNet std
  keep_aspect: bool = False
  crop_size: tuple[int, int] | None = None
  augs: AugsConfig = AugsConfig()
  mosaic_augs: MosaicAugsConfig = MosaicAugsConfig()

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


class EvaluationConfig(BaseModel, frozen=True):
  """Common model inference configuration."""

  model_name: str
  num_classes: int
  img_config: ImageConfig
  conf_thresh: float
  rect: bool
  half: bool
  enable_mask_head: bool
  dtype: str = "float32"
  device: str
  num_top_queries: int = 300
  use_focal_loss: bool = True

  @property
  def n_outputs(self) -> int:
    return self.num_classes

  @property
  def input_width(self) -> int:
    return self.img_config.target_w

  @property
  def input_height(self) -> int:
    return self.img_config.target_h

  @property
  def keep_aspect(self) -> bool:
    return self.img_config.keep_aspect

  @property
  def np_dtype(self):
    import numpy as np

    return np.dtype(self.dtype).type

  @property
  def torch_dtype(self):
    dtype_map = {
      "float32": torch.float32,
      "float16": torch.float16,
      "int32": torch.int32,
      "int8": torch.int8,
    }
    return dtype_map.get(self.dtype, torch.float32)


class DatasetConfig(BaseModel, frozen=True):
  """Base dataset configuration."""

  task: Task
  mode: Mode = Mode.TRAIN
  examples_per_epoch: int | None = None
  img_config: ImageConfig
  multiscale: MultiscaleConfig = MultiscaleConfig()

  @property
  def img_size(self) -> tuple[int, int]:
    """Derive img_size from img_config."""
    return self.img_config.img_size

  @property
  def keep_aspect(self) -> bool:
    """Derive keep_aspect from img_config."""
    return self.img_config.keep_aspect


class DDPConfig(BaseModel, frozen=True):
  """Distributed Data Parallel configuration."""

  enabled: bool = False
  n_gpus: int = 2


class PathConfig(BaseModel, frozen=True):
  """Path configuration."""

  base_path: Path
  path_to_save: Path


class ExportConfig(BaseModel, frozen=True):
  """Export configuration."""

  half: bool = True
  max_batch_size: int = 1
  dynamic_input: bool = False
  ov_int8_max_drop: float = 0.01


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

  # Model name
  model_name: str = "s"

  # Model
  pretrained_dataset: PretrainedDataset = PretrainedDataset.COCO
  pretrained_model_path: Path | None = None

  # Training parameters
  device: Device = Device.CUDA
  conf_thresh: float = 0.5
  iou_thresh: float = 0.5
  epochs: int = 55
  batch_size: int = 8
  b_accum_steps: int = 1
  num_workers: int = 12
  mask_batch_size: int = 1000
  early_stopping: int = 0
  num_top_queries: int = 300
  use_focal_loss: bool = True

  # Visualization options
  visualize_eval: int | None = 5
  visualize_training: int | None = 5
  visualize_loader: int | None = 5

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

  # Optimizer
  cycler_pct_start: float = 0.1
  weight_decay: float = 0.000125
  betas: tuple[float, float] = (0.9, 0.999)
  label_smoothing: float = 0.0

  # EMA
  use_ema: bool = True
  ema_momentum: float = 0.9998

  # Logger
  logger: LoggerType = LoggerType.WANDB

  def get_evaluation_config(self, num_classes: int) -> EvaluationConfig:
    """Get common model inference configuration.

    Args:
        num_classes: Number of output classes from the dataset

    Returns:
        EvaluationConfig instance
    """
    return EvaluationConfig(
      model_name=self.model_name,
      num_classes=num_classes,
      img_config=self.dataset.img_config,
      conf_thresh=self.conf_thresh,
      rect=self.export.dynamic_input,
      half=self.export.half,
      enable_mask_head=self.dataset.task == Task.SEGMENT,
      dtype="float32",
      device=str(self.device),
      num_top_queries=self.num_top_queries,
      use_focal_loss=self.use_focal_loss,
    )

  @classmethod
  def from_hydra(
    cls,
    base_path: Path,
    preset: Path | None = None,
    overrides: list[str] | None = None,
    project_name: str = "d-fine",
    exp_name: str = "default",
  ) -> TrainConfig:
    """Load TrainConfig from Hydra configuration and optional preset.

    Args:
        base_path: Base dataset path (compulsory).
        preset: Optional full path to a preset YAML file.
        overrides: Optional Hydra overrides
        project_name: Project name for the configuration
        exp_name: Experiment name for the configuration

    Returns:
        Instantiated TrainConfig
    """
    if not base_path.exists():
      raise FileNotFoundError(f"Base path not found: {base_path}")

    cfg = _get_hydra_config(project_name, exp_name, base_path, overrides)

    if preset:
      if not preset.is_file():
        raise FileNotFoundError(f"Preset config file not found: {preset}")
      cfg = OmegaConf.merge(cfg, OmegaConf.load(preset))

    return instantiate(cfg.train, _convert_="all")

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

  @classmethod
  def load_from_experiment(cls, base_path: Path, exp_name: str | None = None) -> TrainConfig:
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
      [d for d in models_dir.iterdir() if d.is_dir()], key=get_mtime, reverse=True
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
