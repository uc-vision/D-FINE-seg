from __future__ import annotations

import time

import torch
from loguru import logger

from d_fine.config import TrainConfig
from d_fine.infer.torch_model import Torch_model


def run_benchmark(config: TrainConfig) -> None:
  """Run batching throughput benchmark."""
  logger.info(f"Running batching throughput benchmark for {config.exp_name}")

  classes_path = config.paths.path_to_save / "classes.json"
  if not classes_path.exists():
    raise FileNotFoundError(f"Classes configuration not found at {classes_path}")

  from d_fine.config import ClassConfig

  class_config_obj = ClassConfig.load(classes_path)
  num_classes = len(class_config_obj.label_to_name)

  torch_model = Torch_model.from_train_config(
    config,
    num_classes=num_classes,
    model_path=config.paths.path_to_save / "model.pt",
    device=torch.device(config.device),
  )
  device = torch_model.device

  # Benchmark parameters
  batch_sizes = [1, 2, 4, 8, 16, 32, 64]
  h, w = config.dataset.img_size[1], config.dataset.img_size[0]
  n_iters = 50

  print("\n" + "=" * 80)
  print(f"BATCHING THROUGHPUT BENCHMARK: {config.exp_name}")
  print("=" * 80)
  print(f"{'Batch Size':<12} | {'Model FPS':<15} | {'Latency/Img (ms)':<18} | {'Status':<15}")
  print("-" * 80)

  for bs in batch_sizes:
    try:
      # Use random data on device for pure model throughput
      dummy_batch = torch.randn(bs, 3, h, w).to(device)

      # Warmup
      for _ in range(10):
        _ = torch_model.model(dummy_batch)
      if device.type == "cuda":
        torch.cuda.synchronize()

      # Measure model throughput
      start = time.perf_counter()
      for _ in range(n_iters):
        _ = torch_model.model(dummy_batch)
      if device.type == "cuda":
        torch.cuda.synchronize()
      end = time.perf_counter()

      total_time = end - start
      fps = (bs * n_iters) / total_time
      latency_per_img = (total_time / (bs * n_iters)) * 1000

      print(f"{bs:<12} | {fps:<15.2f} | {latency_per_img:<18.2f} | {'OK':<15}")

    except torch.cuda.OutOfMemoryError:
      print(f"{bs:<12} | {'-':<15} | {'-':<18} | {'OOM':<15}")
      if device.type == "cuda":
        torch.cuda.empty_cache()
      break
    except Exception as e:
      print(f"{bs:<12} | {'-':<15} | {'-':<18} | {'Error':<15}")
      logger.error(f"Batch size {bs} failed: {e}")
      break

  print("=" * 80 + "\n")
