from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
import torch
from loguru import logger
from numpy.typing import NDArray
from openvino import Core

from d_fine.config import EvaluationConfig, TrainConfig
from d_fine.infer import utils as infer_utils
from d_fine.infer.base import InferenceModel
from d_fine.core.types import ImageResult


class OV_model(InferenceModel):
  def __init__(self, config: EvaluationConfig, model_path: Path, max_batch_size: int = 1):
    self.config = config
    self.model_path = model_path
    self.max_batch_size = max_batch_size

    self._load_model()

  @classmethod
  def from_train_config(cls, train_config: TrainConfig, max_batch_size: int = 1) -> OV_model:
    """Create an OV_model from a training configuration."""
    model_path = train_config.paths.path_to_save / "model.xml"
    config = train_config.get_evaluation_config(train_config.num_classes)
    return cls(config, model_path, max_batch_size=max_batch_size)

  def _load_model(self) -> None:
    core = Core()
    det_ov_model = core.read_model(self.model_path)

    device_upper = self.config.device.upper()
    if device_upper == "CUDA" and "GPU" in core.get_available_devices() and not self.config.rect:
      self.device = "GPU"
    elif device_upper in ["CPU", "GPU"]:
      self.device = device_upper
    else:
      self.device = "CPU"

    if self.device != "CPU":
      det_ov_model.reshape({"input": [1, 3, self.config.input_height, self.config.input_width]})

    inference_hint = "f16" if self.config.half else "f32"
    inference_mode = "CUMULATIVE_THROUGHPUT" if self.max_batch_size > 1 else "LATENCY"
    self.model = core.compile_model(
      det_ov_model,
      self.device,
      config={"PERFORMANCE_HINT": inference_mode, "INFERENCE_PRECISION_HINT": inference_hint},
    )
    logger.info(f"OpenVino running on {self.device}")

  def _prepare_input(
    self, img: NDArray[np.uint8]
  ) -> tuple[NDArray, tuple[int, int], tuple[int, int]]:
    """Preprocess a single image for the model."""
    tensor, orig_size = infer_utils.preprocess(
      img,
      (self.config.input_height, self.config.input_width),
      self.config.keep_aspect,
      self.config.rect,
      dtype=self.config.np_dtype,
    )
    processed_size = (tensor.shape[1], tensor.shape[2])
    return tensor.unsqueeze(0).numpy(), processed_size, orig_size

  def _predict(self, inputs: NDArray) -> list[NDArray]:
    return list(self.model(inputs).values())

  def __call__(self, img: NDArray[np.uint8]) -> ImageResult:
    """
    Run inference on a single RGB image.
    Returns an ImageResult.
    """
    inputs, processed_size, orig_size = self._prepare_input(img)
    outputs = self._predict(inputs)

    outputs_torch = {
      "pred_logits": torch.from_numpy(outputs[0]),
      "pred_boxes": torch.from_numpy(outputs[1]),
    }
    if len(outputs) == 3:
      outputs_torch["pred_masks"] = torch.from_numpy(outputs[2])

    results = infer_utils.postprocess_predictions(
      outputs=outputs_torch,
      orig_sizes=torch.as_tensor([orig_size], dtype=torch.float32),
      config=self.config,
      processed_size=processed_size,
    )
    return results[0]
