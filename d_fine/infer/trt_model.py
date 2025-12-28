from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
import tensorrt as trt
import torch
from numpy.typing import NDArray

from d_fine.config import EvaluationConfig, TrainConfig
from d_fine.infer import utils as infer_utils
from d_fine.infer.base import InferenceModel
from d_fine.core.types import ImageResult


class TRT_model(InferenceModel):
  def __init__(self, config: EvaluationConfig, model_path: Path) -> None:
    self.config = config
    self.model_path = model_path
    self.device = config.device

    self._load_model()

  @classmethod
  def from_train_config(cls, train_config: TrainConfig) -> TRT_model:
    """Create a TRT_model from a training configuration."""
    model_path = train_config.paths.path_to_save / "model.engine"
    config = train_config.get_evaluation_config(train_config.num_classes)
    # Force rect=False for TRT engine usually
    config = config.model_copy(update={"rect": False})
    return cls(config, model_path)

  def _load_model(self):
    self.TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(self.model_path, "rb") as f, trt.Runtime(self.TRT_LOGGER) as runtime:
      self.engine = runtime.deserialize_cuda_engine(f.read())
    self.context = self.engine.create_execution_context()

  @staticmethod
  def _torch_dtype_from_trt(trt_dtype):
    if trt_dtype == trt.float32:
      return torch.float32
    elif trt_dtype == trt.float16:
      return torch.float16
    elif trt_dtype == trt.int32:
      return torch.int32
    elif trt_dtype == trt.int8:
      return torch.int8
    else:
      raise TypeError(f"Unsupported TensorRT data type: {trt_dtype}")

  def _test_pred(self) -> None:
    random_image = np.random.randint(0, 255, size=(1100, 1000, 3), dtype=np.uint8)
    processed_inputs, processed_sizes, original_sizes = self._prepare_inputs(random_image)
    preds = self._predict(processed_inputs)
    self._postprocess(preds, processed_sizes, original_sizes)

  def _compute_nearest_size(self, shape, target_size, stride=32) -> tuple[int, int]:
    scale = target_size / max(shape)
    new_shape = [int(round(dim * scale)) for dim in shape]
    return [max(stride, int(np.ceil(dim / stride) * stride)) for dim in new_shape]

  def _preprocess(self, img: NDArray, stride: int = 32) -> torch.Tensor:
    input_size = (self.config.input_height, self.config.input_width)
    if not self.config.keep_aspect:
      img = cv2.resize(img, (input_size[1], input_size[0]), interpolation=cv2.INTER_AREA)
    elif self.config.rect:
      target_height, target_width = self._compute_nearest_size(img.shape[:2], max(*input_size))
      img = infer_utils.letterbox(img, (target_height, target_width), stride=stride, auto=False)[0]
    else:
      img = infer_utils.letterbox(img, (input_size[0], input_size[1]), stride=stride, auto=False)[0]

    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=self.config.np_dtype)
    img /= 255.0
    return torch.from_numpy(img)

  def _prepare_inputs(self, inputs):
    original_sizes = []
    processed_sizes = []
    input_size = (self.config.input_height, self.config.input_width)

    if isinstance(inputs, np.ndarray) and inputs.ndim == 3:
      processed_inputs = self._preprocess(inputs)[None]
      original_sizes.append((inputs.shape[0], inputs.shape[1]))
      processed_sizes.append(processed_inputs[0].shape[1:])
    elif isinstance(inputs, np.ndarray) and inputs.ndim == 4:
      processed_inputs = torch.zeros(
        (inputs.shape[0], 3, *input_size),
        dtype=torch.from_numpy(np.zeros(0, dtype=self.config.np_dtype)).dtype,
      )
      for idx, image in enumerate(inputs):
        processed_inputs[idx] = self._preprocess(image)
        original_sizes.append(image.shape[:2])
        processed_sizes.append(processed_inputs[idx].shape[1:])

    if self.device == "cuda":
      processed_inputs = processed_inputs.pin_memory().to(self.device, non_blocking=True)
    else:
      processed_inputs = processed_inputs.to(self.device)
    return processed_inputs, processed_sizes, original_sizes

  def _predict(self, img: torch.Tensor) -> list[torch.Tensor]:
    img = img.contiguous()
    batch_shape = tuple(img.shape)
    n_io = self.engine.num_io_tensors
    bindings: list[int] = [None] * n_io
    outputs: list[torch.Tensor] = []

    for i in range(n_io):
      name = self.engine.get_tensor_name(i)
      mode = self.engine.get_tensor_mode(name)
      dims = tuple(self.engine.get_tensor_shape(name))
      dt = self.engine.get_tensor_dtype(name)
      t_dt = self._torch_dtype_from_trt(dt)

      if mode == trt.TensorIOMode.INPUT:
        self.context.set_input_shape(name, batch_shape)
        bindings[i] = img.data_ptr()
      else:
        out_shape = (batch_shape[0],) + dims[1:]
        out = torch.empty(out_shape, dtype=t_dt, device=self.device)
        outputs.append(out)
        bindings[i] = out.data_ptr()

    self.context.execute_v2(bindings)
    return outputs

  def _postprocess(
    self,
    outputs: list[torch.Tensor],
    processed_sizes: list[tuple[int, int]],
    original_sizes: list[tuple[int, int]],
  ) -> list[ImageResult]:
    outputs_dict = {"pred_logits": outputs[0], "pred_boxes": outputs[1]}
    if len(outputs) > 2:
      outputs_dict["pred_masks"] = outputs[2]

    orig_sizes_tensor = torch.tensor(original_sizes, device=self.device, dtype=torch.float32)

    return infer_utils.postprocess_predictions(
      outputs=outputs_dict,
      orig_sizes=orig_sizes_tensor,
      config=self.config,
      processed_size=processed_sizes[0]
      if len(processed_sizes) > 0
      else (self.config.input_height, self.config.input_width),
    )

  def __call__(self, inputs: NDArray[np.uint8]) -> ImageResult:
    """Run inference on an RGB image."""
    processed_inputs, processed_sizes, original_sizes = self._prepare_inputs(inputs)
    preds = self._predict(processed_inputs)
    results = self._postprocess(preds, processed_sizes, original_sizes)
    return results[0]
