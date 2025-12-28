from __future__ import annotations

from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch
from numpy.typing import NDArray

from d_fine.config import EvaluationConfig, TrainConfig
from d_fine.infer import utils as infer_utils
from d_fine.infer.base import InferenceModel
from d_fine.core.types import ImageResult


class ONNX_model(InferenceModel):
  def __init__(self, config: EvaluationConfig, model_path: Path):
    self.config = config
    self.model_path = model_path
    self.device = config.device

    self._load_model()

  @classmethod
  def from_train_config(cls, train_config: TrainConfig) -> ONNX_model:
    """Create an ONNX_model from a training configuration."""
    model_path = train_config.paths.path_to_save / "model.onnx"
    config = train_config.get_evaluation_config(train_config.num_classes)
    # Force rect=False and half=False for ONNX if needed, but evaluation_config usually comes from YAML
    return cls(config, model_path)

  def _load_model(self) -> None:
    providers = ["CUDAExecutionProvider"] if self.device == "cuda" else ["CPUExecutionProvider"]
    provider_options = [{"cudnn_conv_algo_search": "DEFAULT"}] if self.device == "cuda" else [{}]
    self.model = ort.InferenceSession(
      str(self.model_path), providers=providers, provider_options=provider_options
    )
    print(f"ONNX model loaded: {self.model_path} on {self.device}")

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

  def _predict(self, inputs: NDArray) -> dict[str, NDArray]:
    ort_inputs = {self.model.get_inputs()[0].name: inputs.astype(self.config.np_dtype)}
    outs = self.model.run(None, ort_inputs)
    return {
      "pred_logits": outs[0],
      "pred_boxes": outs[1],
      "pred_masks": outs[2] if len(outs) > 2 else None,
    }

  def __call__(self, img: NDArray[np.uint8]) -> ImageResult:
    """
    Run inference on a single RGB image.
    Returns an ImageResult.
    """
    inputs, processed_size, orig_size = self._prepare_input(img)
    outputs = self._predict(inputs)

    outputs_torch = {
      "pred_logits": torch.from_numpy(outputs["pred_logits"]),
      "pred_boxes": torch.from_numpy(outputs["pred_boxes"]),
    }
    if outputs["pred_masks"] is not None:
      outputs_torch["pred_masks"] = torch.from_numpy(outputs["pred_masks"])

    results = infer_utils.postprocess_predictions(
      outputs=outputs_torch,
      orig_sizes=torch.as_tensor([orig_size], dtype=torch.float32),
      config=self.config,
      processed_size=processed_size,
    )
    return results[0]
