from __future__ import annotations

import time
from pathlib import Path

import cv2
import click
import numpy as np
import pandas as pd
import torch
from loguru import logger
from tabulate import tabulate
from tqdm import tqdm

from collections.abc import Iterator
from d_fine.config import TrainConfig
from d_fine.infer.base import InferenceModel
from d_fine.dataset.base import Dataset
from d_fine.validation import Validator, ValidationConfig, EvaluationMetrics
from d_fine.validation.utils import raw_sample_to_image_result
from d_fine.core.types import ImageResult

torch.multiprocessing.set_sharing_strategy("file_system")


def run_inference(
    model: InferenceModel,
    dataset: Dataset,
) -> Iterator[tuple[np.ndarray, ImageResult, ImageResult, str, float]]:
    """Iterator yielding (image, prediction, ground_truth, path, latency_ms) for each sample."""
    for i in range(len(dataset)):
        sample = dataset.get_data(i)
        
        t0 = time.perf_counter()
        pred_res = model(sample.image)
        latency_ms = (time.perf_counter() - t0) * 1000
        
        gt_res = raw_sample_to_image_result(sample)
        yield sample.image, pred_res, gt_res, sample.path, latency_ms


def test_model(
    dataset: Dataset,
    model: InferenceModel,
    name: str,
    val_cfg: ValidationConfig,
) -> tuple[float, EvaluationMetrics]:
    """Run model inference on test set and return latency and metrics."""
    logger.info(f"Testing {name} model")
    
    results = run_inference(model, dataset)
    _, all_preds, all_gt, _, latency = zip(*tqdm(results, total=len(dataset), desc=f"Benchmarking {name}"))

    metrics = Validator(all_gt, all_preds, config=val_cfg).compute_metrics()
    return float(np.mean(latency[1:])), metrics


@click.command()
@click.option("--project-name", required=True, help="Project name")
@click.option("--base-path", required=True, type=click.Path(exists=True, path_type=Path), help="Base project path")
@click.option("--exp-name", type=str, help="Experiment name (defaults to latest)")
def main(project_name: str, base_path: Path, exp_name: str | None) -> None:
    """Benchmark model performance.
    
    Loads config from saved training experiment.
    """
    try:
        train_config = TrainConfig.load_from_experiment(base_path, exp_name)
    except FileNotFoundError as e:
        raise click.BadParameter(str(e))
    
    save_dir = train_config.paths.path_to_save
    loader = train_config.dataset.create_loader(batch_size=1, num_workers=1)
    val_loader, test_loader, _ = loader.build_dataloaders(distributed=False)
    
    label_to_name = loader.label_to_name

    from d_fine.infer import Torch_model, TRT_model, OV_model, ONNX_model

    models: dict[str, InferenceModel] = {}
    if Torch_model: models["Torch"] = Torch_model.from_train_config(train_config)
    if TRT_model: models["TensorRT"] = TRT_model.from_train_config(train_config)
    if OV_model: models["OpenVINO"] = OV_model.from_train_config(train_config)
    if ONNX_model: models["ONNX"] = ONNX_model.from_train_config(train_config)
    
    ov_int8_path = save_dir / "model_int8.xml"
    if ov_int8_path.exists() and OV_model:
        model_config = train_config.get_model_config(loader.num_classes)
        models["OpenVINO INT8"] = OV_model(
            model_config=model_config,
            model_path=str(ov_int8_path),
            max_batch_size=1,
        )

    val_cfg = ValidationConfig(
        conf_threshold=train_config.conf_thresh,
        iou_threshold=train_config.iou_thresh,
        label_to_name=label_to_name,
    )

    all_results: dict[str, tuple[float, EvaluationMetrics]] = {}
    for model_name, model in models.items():
        all_results[model_name] = test_model(
            val_loader.dataset,
            model,
            model_name,
            val_cfg,
        )

    # Convert to dataframe for table display
    table_data = {
        name: metrics.to_dict() | {"latency (ms)": lat} 
        for name, (lat, metrics) in all_results.items()
    }
    metrics_df = pd.DataFrame.from_dict(table_data, orient="index")
    # Filter out nested per-class dicts for the summary table
    summary_df = metrics_df[[c for c in metrics_df.columns if not isinstance(metrics_df[c].iloc[0], dict)]]
    tabulated_data = tabulate(summary_df.round(4), headers="keys", tablefmt="pretty", showindex=True)
    print("\n" + tabulated_data)



if __name__ == "__main__":
    main()
