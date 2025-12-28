from __future__ import annotations

import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tabulate import tabulate
from tqdm import tqdm
from loguru import logger

from d_fine.config import TrainConfig
from d_fine.infer.torch_model import Torch_model


def run_batch_benchmark(
  train_config: TrainConfig, num_images: int = 512, batch_sizes: list[int] | None = None
) -> pd.DataFrame:
  """Benchmark model throughput for different batch sizes."""
  batch_sizes = batch_sizes or [1, 2, 4, 8, 16, 32]
  torch_model = Torch_model.from_train_config(train_config)

  # Get a sample image for realistic sizing
  loader = train_config.dataset.create_loader(batch_size=1, num_workers=0)
  _, val_loader, _ = loader.build_dataloaders(distributed=False)
  img = val_loader.dataset.get_data(0).image

  def bench_bs(bs: int) -> dict:
    batch_img = np.stack([img] * bs)
    iterations = max(1, num_images // bs)

    # Warmup
    for _ in range(3):
      _ = torch_model.predict_batch(batch_img)

    if torch.cuda.is_available():
      torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in tqdm(range(iterations), desc=f"Batch size {bs}", leave=False):
      _ = torch_model.predict_batch(batch_img)

    if torch.cuda.is_available():
      torch.cuda.synchronize()
    t1 = time.perf_counter()

    total_images = iterations * bs
    elapsed = t1 - t0

    return {
      "bs": bs,
      "throughput": total_images / elapsed,
      "latency_ms": (elapsed * 1000) / total_images,
    }

  results = [bench_bs(bs) for bs in batch_sizes]
  return pd.DataFrame(results)


def run_benchmark(train_config: TrainConfig) -> None:
  """Run batching throughput benchmark."""
  df = run_batch_benchmark(train_config)

  save_dir = train_config.paths.path_to_save
  save_dir.mkdir(parents=True, exist_ok=True)
  save_path = save_dir / "batch_benchmark.csv"
  df.to_csv(save_path, index=False)

  logger.info(f"Results saved to {save_path}")
  print("\n" + tabulate(df.round(2), headers="keys", tablefmt="pretty", showindex=False))
