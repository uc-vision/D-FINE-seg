from __future__ import annotations

import time
from pathlib import Path
from collections.abc import Iterator, Callable
import numpy as np
import pandas as pd
import torch
from loguru import logger
from tabulate import tabulate
from tqdm import tqdm

from d_fine.config import TrainConfig, Task
from d_fine.infer.base import InferenceModel
from d_fine.dataset.dataset import Dataset, ToImageResult
from d_fine.validation import Validator, ValidationConfig, EvaluationMetrics
from d_fine.validation.utils import (
  detection_sample_to_image_result,
  segmentation_sample_to_image_result,
)
from d_fine.core.types import ImageResult


def run_inference_iterator[T](
  model: InferenceModel, dataset: Dataset[T], converter: ToImageResult[T]
) -> Iterator[tuple[ImageResult, ImageResult, float]]:
  """Yields (prediction, ground_truth, latency_ms) for each sample."""
  for i in range(len(dataset)):
    sample = dataset.get_data(i)

    t0 = time.perf_counter()
    pred = model(sample.image)
    latency = (time.perf_counter() - t0) * 1000

    yield pred, converter(sample), latency


def benchmark_model[T](
  dataset: Dataset[T],
  model: InferenceModel,
  name: str,
  val_cfg: ValidationConfig,
  converter: ToImageResult[T],
) -> tuple[float, EvaluationMetrics]:
  """Run benchmark for a single model."""
  logger.info(f"Benchmarking {name}")

  results = list(
    tqdm(
      run_inference_iterator(model, dataset, converter), total=len(dataset), desc=name, leave=False
    )
  )
  preds, gts, latencies = zip(*results)

  metrics = Validator(list(gts), list(preds), config=val_cfg).compute_metrics()
  # Skip first sample for latency to avoid initialization overhead
  avg_latency = float(np.mean(latencies[1:])) if len(latencies) > 1 else float(latencies[0])

  return avg_latency, metrics


def run_benchmark(train_config: TrainConfig, ann_path: Path | None = None) -> None:
  """Run inference latency and accuracy benchmark."""
  save_dir = train_config.paths.path_to_save

  if ann_path:
    dataset = train_config.dataset.create_dataset(ann_path, mode="val")
  else:
    loader = train_config.dataset.create_loader(batch_size=1, num_workers=1)
    _, val_loader, _ = loader.build_dataloaders(distributed=False)
    dataset = val_loader.dataset

  converter = (
    segmentation_sample_to_image_result
    if train_config.task == Task.SEGMENT
    else detection_sample_to_image_result
  )

  from d_fine.infer import Torch_model, TRT_model, OV_model, ONNX_model

  models: dict[str, InferenceModel] = {}
  if Torch_model:
    models["Torch"] = Torch_model.from_train_config(train_config)

  trt_path = save_dir / "model.engine"
  if trt_path.exists() and TRT_model:
    models["TensorRT"] = TRT_model.from_train_config(train_config)

  ov_path = save_dir / "model.xml"
  if ov_path.exists() and OV_model:
    models["OpenVINO"] = OV_model.from_train_config(train_config)

  onnx_path = save_dir / "model.onnx"
  if onnx_path.exists() and ONNX_model:
    models["ONNX"] = ONNX_model.from_train_config(train_config)

  ov_int8_path = save_dir / "model_int8.xml"
  if ov_int8_path.exists() and OV_model:
    eval_cfg = train_config.get_evaluation_config(dataset.num_classes)
    models["OpenVINO INT8"] = OV_model(eval_cfg, ov_int8_path)

  val_cfg = ValidationConfig(
    conf_threshold=train_config.conf_thresh,
    iou_threshold=train_config.iou_thresh,
    label_to_name=dataset.label_to_name,
  )

  results = {
    name: benchmark_model(dataset, m, name, val_cfg, converter) for name, m in models.items()
  }

  # Format and display results
  table_data = {
    name: metrics.to_dict() | {"latency_ms": lat} for name, (lat, metrics) in results.items()
  }
  df = pd.DataFrame.from_dict(table_data, orient="index")
  summary_cols = [c for c in df.columns if not isinstance(df[c].iloc[0], dict)]
  summary_df = df[summary_cols]

  print("\n" + tabulate(summary_df.round(4), headers="keys", tablefmt="pretty"))

  save_dir.mkdir(parents=True, exist_ok=True)
  save_path = save_dir / "inference_benchmark.csv"
  df.to_csv(save_path)
  logger.info(f"Results saved to {save_path}")
