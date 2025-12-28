from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
import torch
from loguru import logger
from numpy.typing import NDArray

from d_fine.core.dfine import build_model
from d_fine.config import EvaluationConfig, TrainConfig
from d_fine.infer import utils as infer_utils
from d_fine.infer.base import InferenceModel
from d_fine.core.types import ImageResult


class Torch_model(InferenceModel):
  def __init__(
    self,
    config: EvaluationConfig,
    model_path: Path,
    use_nms: bool = False,
    device: torch.device | None = None,
  ):
    self.config = config
    self.model_path = model_path
    self.use_nms = use_nms
    self.device = device if device else torch.device(config.device)

    self._load_model()

  @classmethod
  def from_train_config(
    cls,
    train_config: TrainConfig,
    model_path: Path | None = None,
    use_nms: bool = False,
    device: torch.device | None = None,
  ) -> Torch_model:
    """Create a Torch_model from a training configuration."""
    if model_path is None:
      model_path = train_config.paths.path_to_save / "model.pt"

    config = train_config.get_evaluation_config(train_config.num_classes)
    return cls(config, model_path, use_nms=use_nms, device=device)

  def _load_model(self) -> None:
    self.model = build_model(
      self.config.model_name,
      self.config.n_outputs,
      self.config.enable_mask_head,
      self.device,
      img_size=None,
    )
    self.model.load_state_dict(
      torch.load(self.model_path, weights_only=True, map_location=torch.device("cpu")), strict=False
    )
    self.model.eval()
    self.model.to(self.device)
    logger.info(f"Torch model loaded on {self.device}")

  def _prepare_input(
    self, img: NDArray[np.uint8]
  ) -> tuple[torch.Tensor, tuple[int, int], tuple[int, int]]:
    """Preprocess a single image for the model."""
    tensor, orig_size = infer_utils.preprocess(
      img,
      (self.config.input_height, self.config.input_width),
      self.config.keep_aspect,
      self.config.rect,
      dtype=self.config.np_dtype,
    )
    processed_size = (tensor.shape[1], tensor.shape[2])
    tensor = tensor.unsqueeze(0).to(self.device)
    if self.device.type == "cuda":
      tensor = tensor.pin_memory()
    return tensor, processed_size, orig_size

  @torch.no_grad()
  def predict_batch(self, imgs: NDArray[np.uint8]) -> list[ImageResult]:
    """
    Run inference on a batch of RGB images.
    imgs: [B, H, W, 3] RGB image array.
    """
    batch_size = imgs.shape[0]
    tensors = []
    orig_sizes = []
    processed_sizes = []

    for i in range(batch_size):
      tensor, orig_size = infer_utils.preprocess(
        imgs[i],
        (self.config.input_height, self.config.input_width),
        self.config.keep_aspect,
        self.config.rect,
        dtype=self.config.np_dtype,
      )
      tensors.append(tensor)
      orig_sizes.append(orig_size)
      processed_sizes.append((tensor.shape[1], tensor.shape[2]))

    input_tensor = torch.stack(tensors).to(self.device)
    if self.device.type == "cuda":
      input_tensor = input_tensor.pin_memory()

    outputs = self.model(input_tensor)

    results = infer_utils.postprocess_predictions(
      outputs=outputs,
      orig_sizes=torch.as_tensor(orig_sizes, dtype=torch.float32, device=self.device),
      config=self.config,
      processed_size=processed_sizes[0],
    )

    if self.use_nms:
      results = [res.nms() for res in results]

    return results

  @torch.no_grad()
  def __call__(self, img: NDArray[np.uint8]) -> ImageResult:
    """
    Run inference on a single RGB image.
    Returns an ImageResult.
    """
    tensor, processed_size, orig_size = self._prepare_input(img)
    outputs = self.model(tensor)

    results = infer_utils.postprocess_predictions(
      outputs=outputs,
      orig_sizes=torch.as_tensor([orig_size], dtype=torch.float32, device=self.device),
      config=self.config,
      processed_size=processed_size,
    )

    res = results[0]
    if self.use_nms:
      res = res.nms()

    return res
