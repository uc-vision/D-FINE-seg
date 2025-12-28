import datetime
import random
import time
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from .dist_utils import is_main_process


def set_seeds(seed: int, cudnn_fixed: bool = False) -> None:
  """Set random seeds for reproducibility."""
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  if cudnn_fixed:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
  """Seed function for DataLoader workers."""
  worker_seed = torch.initial_seed() % 2**32
  np.random.seed(worker_seed)
  random.seed(worker_seed)


def setup_multi_processes() -> None:
  """Configure multi-processing settings."""
  import os

  if os.name != "nt":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"


def get_vram_usage() -> float:
  """Return current GPU memory usage percentage."""
  if torch.cuda.is_available():
    t = torch.cuda.get_device_properties(0).total_memory
    a = torch.cuda.memory_allocated(0)
    return round((a / t) * 100, 2)
  return 0.0


def calculate_remaining_time(
  one_epoch_time: float | None,
  epoch_start_time: float,
  epoch: int,
  num_epochs: int,
  cur_iter: int,
  num_batches: int,
) -> str:
  """Estimate remaining training time."""
  current_epoch_elapsed = time.time() - epoch_start_time
  if cur_iter > 0:
    estimated_total_epoch = (current_epoch_elapsed / cur_iter) * num_batches
  else:
    estimated_total_epoch = one_epoch_time if one_epoch_time is not None else 0

  remaining_epochs = num_epochs - epoch
  remaining_current = estimated_total_epoch - current_epoch_elapsed

  if one_epoch_time is not None:
    remaining_future = remaining_epochs * one_epoch_time
  else:
    remaining_future = remaining_epochs * estimated_total_epoch

  total_remaining = max(0, remaining_current + remaining_future)
  return str(datetime.timedelta(seconds=int(total_remaining)))


obj365_ids = [
  0,
  46,
  5,
  58,
  114,
  55,
  116,
  65,
  21,
  40,
  176,
  127,
  249,
  24,
  56,
  139,
  92,
  78,
  99,
  96,
  144,
  295,
  178,
  180,
  38,
  39,
  13,
  43,
  120,
  219,
  148,
  173,
  165,
  154,
  137,
  113,
  145,
  146,
  204,
  8,
  35,
  10,
  88,
  84,
  93,
  26,
  112,
  82,
  265,
  104,
  141,
  152,
  234,
  143,
  150,
  97,
  2,
  50,
  25,
  75,
  98,
  153,
  37,
  73,
  115,
  132,
  106,
  61,
  163,
  134,
  277,
  81,
  133,
  18,
  94,
  30,
  169,
  70,
  328,
  226,
]


def map_class_weights(cur_tensor, pretrain_tensor):
  """Map class weights from pretrain model to current model based on class IDs."""
  if pretrain_tensor.size() == cur_tensor.size():
    return pretrain_tensor

  adjusted_tensor = cur_tensor.clone()
  adjusted_tensor.requires_grad = False

  if pretrain_tensor.size() > cur_tensor.size():
    for coco_id, obj_id in enumerate(obj365_ids):
      adjusted_tensor[coco_id] = pretrain_tensor[obj_id + 1]
  else:
    for coco_id, obj_id in enumerate(obj365_ids):
      adjusted_tensor[obj_id + 1] = pretrain_tensor[coco_id]

  return adjusted_tensor


def adjust_head_parameters(cur_state_dict, pretrain_state_dict):
  """Adjust head parameters between datasets."""
  # List of parameters to adjust
  if (
    pretrain_state_dict["decoder.denoising_class_embed.weight"].size()
    != cur_state_dict["decoder.denoising_class_embed.weight"].size()
  ):
    del pretrain_state_dict["decoder.denoising_class_embed.weight"]

  head_param_names = ["decoder.enc_score_head.weight", "decoder.enc_score_head.bias"]
  for i in range(8):
    head_param_names.append(f"decoder.dec_score_head.{i}.weight")
    head_param_names.append(f"decoder.dec_score_head.{i}.bias")

  adjusted_params = []

  for param_name in head_param_names:
    if param_name in cur_state_dict and param_name in pretrain_state_dict:
      cur_tensor = cur_state_dict[param_name]
      pretrain_tensor = pretrain_state_dict[param_name]
      adjusted_tensor = map_class_weights(cur_tensor, pretrain_tensor)
      if adjusted_tensor is not None:
        pretrain_state_dict[param_name] = adjusted_tensor
        adjusted_params.append(param_name)
      # Size mismatch - parameter will be skipped

  return pretrain_state_dict


def matched_state(state: dict[str, torch.Tensor], params: dict[str, torch.Tensor]):
  missed_list = []
  unmatched_list = []
  matched_state = {}
  for k, v in state.items():
    if k in params:
      if v.shape == params[k].shape:
        matched_state[k] = params[k]
      else:
        unmatched_list.append(k)
    else:
      missed_list.append(k)

  return matched_state, {"missed": missed_list, "unmatched": unmatched_list}


def load_tuning_state(model, path: str | Path):
  """Load model for tuning and adjust mismatched head parameters"""
  path_str = str(path)
  if path_str.startswith("http"):
    state = torch.hub.load_state_dict_from_url(path_str, map_location="cpu")
  else:
    state = torch.load(path, map_location="cpu", weights_only=True)

  # Load the appropriate state dict
  if "ema" in state:
    pretrain_state_dict = state["ema"]["module"]
  elif "model" in state:
    pretrain_state_dict = state["model"]
  else:
    pretrain_state_dict = state

  # Adjust head parameters between datasets
  try:
    adjusted_state_dict = adjust_head_parameters(model.state_dict(), pretrain_state_dict)
    stat, infos = matched_state(model.state_dict(), adjusted_state_dict)
  except Exception:
    stat, infos = matched_state(model.state_dict(), pretrain_state_dict)

  model.load_state_dict(stat, strict=False)
  if is_main_process():
    logger.info(f"Pretrained weigts from {path}, {infos}")
  return model
