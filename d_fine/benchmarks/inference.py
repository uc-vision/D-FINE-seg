from __future__ import annotations

import time
from pathlib import Path

import torch
from loguru import logger
from tqdm import tqdm

from d_fine.config import Mode, TrainConfig
from d_fine.dataset import get_loader_class
from d_fine.infer.torch_model import Torch_model
from d_fine.validation.metrics import ValidationConfig
from d_fine.validation.utils import coco_to_image_results
from d_fine.validation.validator import Validator


def run_benchmark(config: TrainConfig, ann_path: Path | None = None) -> None:
  """Run inference latency and accuracy benchmark."""
  # 1. Setup data loader
  if ann_path:
    # Use provided annotation file instead of the default val_ann
    config = config.model_copy(
      update={"dataset": config.dataset.model_copy(update={"val_ann": str(ann_path)})}
    )

  loader_class = get_loader_class(config.dataset)
  loader = loader_class(config.dataset, batch_size=1, num_workers=4)
  _, val_loader, _ = loader.build_dataloaders()
  dataset = val_loader.dataset

  # 2. Setup model
  logger.info(f"Loading model for experiment: {config.exp_name}")
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

  # Load ground truth from COCO file
  from lib_detection.annotation.coco import CocoDataset as CocoFile

  logger.info(f"Loading ground truth from {dataset.annotation_file}")
  coco_gt = CocoFile(dataset.annotation_file)
  gt_results = coco_to_image_results(coco_gt)

  # 3. Warmup
  logger.info("Warming up model...")
  dummy_img = torch.randn(1, 3, config.dataset.img_size[1], config.dataset.img_size[0])
  device = torch_model.device
  dummy_img = dummy_img.to(device)
  for _ in range(10):
    _ = torch_model.model(dummy_img)
  if device.type == "cuda":
    torch.cuda.synchronize()

  # 4. Run inference
  preds = []
  times = []

  logger.info(f"Running inference on {len(dataset)} images...")
  for idx in tqdm(range(len(dataset))):
    sample = dataset.get_data(idx)
    img_rgb = sample.image

    start = time.perf_counter()
    res = torch_model(img_rgb)
    if device.type == "cuda":
      torch.cuda.synchronize()
    end = time.perf_counter()

    preds.append(res)
    times.append(end - start)

  # 5. Compute metrics
  avg_latency = sum(times) / len(times) * 1000  # ms
  fps = 1.0 / (sum(times) / len(times))

  val_config = ValidationConfig(
    conf_threshold=config.conf_thresh,
    iou_threshold=config.iou_thresh,
    label_to_name=loader.label_to_name,
  )

  validator = Validator(gt=gt_results, preds=preds, config=val_config)
  metrics = validator.compute_metrics()

  # 6. Report results
  print("\n" + "=" * 60)
  print(f"BENCHMARK RESULTS: {config.exp_name}")
  print("=" * 60)
  print(f"{'Metric':<25} | {'Value':<20}")
  print("-" * 60)
  print(f"{'Latency':<25} | {avg_latency:.2f} ms")
  print(f"{'FPS':<25} | {fps:.2f}")
  print("-" * 60)

  results_dict = metrics.to_dict()
  for k, v in results_dict.items():
    if isinstance(v, (float, int)):
      print(f"{k:<25} | {v:.4f}")
    elif isinstance(v, dict):
      # Skip nested class metrics for brevity in summary
      continue
  print("=" * 60 + "\n")
