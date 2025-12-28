import pytest
import torch
import numpy as np
import cv2
from pathlib import Path
from d_fine.infer.torch_model import Torch_model
from d_fine.config import EvaluationConfig, TrainConfig, ImageConfig
from d_fine.core.types import ImageResult
from lib_detection.annotation import InstanceMask


def create_dummy_config(tmp_path: Path):
  model_path = tmp_path / "model.pt"
  # We'll need a real model or a mock one.
  # Since we can't easily build a full model without weights,
  # we'll mock the internal model if needed, but Torch_model expects a file.

  from d_fine.core.dfine import build_model

  model = build_model("n", 1, True, "cpu")
  torch.save(model.state_dict(), model_path)

  return EvaluationConfig(
    model_name="n",
    num_classes=1,
    img_config=ImageConfig(img_size=(640, 640), keep_aspect=False),
    conf_thresh=0.5,
    enable_mask_head=True,
    device="cpu",
    rect=False,
    half=False,
  ), model_path


def test_torch_model_inference(tmp_path):
  config, model_path = create_dummy_config(tmp_path)
  model = Torch_model(evaluation_config=config, model_path=str(model_path))

  img = np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8)
  res = model(img)

  assert isinstance(res, ImageResult)
  assert res.img_size == (480, 640)
  assert isinstance(res.labels, torch.Tensor)
  assert isinstance(res.boxes, torch.Tensor)
  assert isinstance(res.scores, torch.Tensor)
  assert isinstance(res.masks, list)

  if res.masks:
    assert isinstance(res.masks[0], InstanceMask)


def test_torch_model_nms(tmp_path):
  config, model_path = create_dummy_config(tmp_path)
  # Enable NMS
  model = Torch_model(evaluation_config=config, model_path=str(model_path), use_nms=True)

  img = np.random.randint(0, 255, size=(640, 640, 3), dtype=np.uint8)
  res = model(img)

  # NMS shouldn't crash and should return ImageResult
  assert isinstance(res, ImageResult)


def test_image_result_nms():
  # Test NMS logic independently
  labels = torch.tensor([0, 0], dtype=torch.int64)
  boxes = torch.tensor([[10, 10, 50, 50], [12, 12, 52, 52]], dtype=torch.float32)
  scores = torch.tensor([0.9, 0.8], dtype=torch.float32)
  img_size = (100, 100)

  res = ImageResult(labels=labels, boxes=boxes, scores=scores, img_size=img_size)
  nms_res = res.nms(iou_threshold=0.5)

  # Second box should be suppressed as it highly overlaps with the first one
  assert len(nms_res.labels) == 1
  assert nms_res.scores[0] == 0.9


def test_image_result_filter():
  labels = torch.tensor([0, 1], dtype=torch.int64)
  boxes = torch.tensor([[10, 10, 50, 50], [60, 60, 90, 90]], dtype=torch.float32)
  scores = torch.tensor([0.9, 0.4], dtype=torch.float32)

  res = ImageResult(labels=labels, boxes=boxes, scores=scores, img_size=(100, 100))
  filtered = res.filter(threshold=0.5)

  assert len(filtered.labels) == 1
  assert filtered.labels[0] == 0
  assert filtered.scores[0] == 0.9
