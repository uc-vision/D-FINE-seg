import time
from pathlib import Path
from shutil import rmtree
from typing import Any

import cv2
import click
import numpy as np
import pandas as pd
import torch
from loguru import logger
from tabulate import tabulate
from torch.utils.data import DataLoader
from tqdm import tqdm

from d_fine.config import TrainConfig
from d_fine.utils import process_boxes, process_masks
from d_fine.validation import Validator
from d_fine.validation.visualization import visualize as visualize_pred
from d_fine.infer.onnx_model import ONNX_model
from d_fine.infer.ov_model import OV_model
from d_fine.infer.torch_model import Torch_model
from d_fine.infer.trt_model import TRT_model

torch.multiprocessing.set_sharing_strategy("file_system")


def test_model(
    test_loader: DataLoader,
    data_path: Path,
    output_path: Path,
    model,
    name: str,
    conf_thresh: float,
    iou_thresh: float,
    to_visualize: bool,
    processed_size: tuple[int, int],
    keep_ratio: bool,
    device: str,
    label_to_name: dict[int, str],
):
    logger.info(f"Testing {name} model")
    latency = []
    all_gt = []
    all_preds = []

    output_path = output_path / name
    output_path.mkdir(exist_ok=True, parents=True)

    for _, targets, img_paths in tqdm(test_loader, total=len(test_loader)):
        for img_path, target in zip(img_paths, targets):
            img = cv2.imread(str(data_path / "images" / img_path))

            gt_boxes = process_boxes(
                target["boxes"][None],
                processed_size,
                target["orig_size"][None],
                keep_ratio,
                device,
            )[0].cpu()

            gt_dict = {"boxes": gt_boxes, "labels": target["labels"].int()}
            if "masks" in target:
                gt_dict["masks"] = process_masks(
                    target["masks"][None], processed_size, target["orig_size"][None], keep_ratio
                )[0].cpu()
            all_gt.append(gt_dict)

            t0 = time.perf_counter()
            pred = model(img)[0]
            latency.append((time.perf_counter() - t0) * 1000)

            pred_dict = {
                "boxes": torch.from_numpy(pred["boxes"]),
                "labels": torch.from_numpy(pred["labels"]),
                "scores": torch.from_numpy(pred["scores"]),
            }
            if "mask_probs" in pred:
                pred_masks = (pred["mask_probs"] >= conf_thresh).astype(np.uint8)
                pred_dict["masks"] = torch.from_numpy(pred_masks)
            all_preds.append(pred_dict)

            if to_visualize:
                visualize_pred(
                    img=img,
                    boxes=pred["boxes"],
                    labels=pred["labels"],
                    scores=pred["scores"],
                    output_path=output_path,
                    img_path=str(img_path),
                    label_to_name=label_to_name,
                    masks=pred_masks if "mask_probs" in pred else None,
                )

    validator = Validator(
        all_gt,
        all_preds,
        conf_thresh=conf_thresh,
        iou_thresh=iou_thresh,
        label_to_name=label_to_name,
    )
    metrics = validator.compute_metrics(extended=False)

    # as inference done with a conf threshold, mAPs don't make much sense
    metrics.pop("mAP_50")
    metrics.pop("mAP_50_95")
    metrics["latency"] = round(np.mean(latency[1:]), 1)
    return metrics


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
    temp_loader = train_config.dataset.create_loader(batch_size=1, num_workers=1)
    val_loader, test_loader, _ = temp_loader.build_dataloaders(distributed=False)
    num_classes = val_loader.dataset.num_classes
    label_to_name = val_loader.dataset.label_to_name
    model_config = train_config.get_model_config(num_classes)

    models: dict[str, Any] = {
        "Torch": Torch_model(
            train_config=train_config,
            model_config=model_config,
        ),
        "TensorRT": TRT_model(
            model_path=str(save_dir / "model.engine"),
            model_config=model_config.model_copy(update={"rect": False}),
        ),
        "OpenVINO": OV_model(
            model_path=str(save_dir / "model.xml"),
            model_config=model_config,
            max_batch_size=1,
        ),
        "ONNX": ONNX_model(
            model_path=str(save_dir / "model.onnx"),
            model_config=model_config.model_copy(update={"rect": False, "half": False}),
        ),
    }
    
    ov_int8_path = save_dir / "model_int8.xml"
    if ov_int8_path.exists():
        models["OpenVINO INT8"] = OV_model(
            model_path=str(ov_int8_path),
            model_config=model_config,
            max_batch_size=1,
        )

    output_path = train_config.paths.path_to_save / "bench_images"
    if output_path.exists():
        rmtree(output_path)

    all_metrics: dict[str, dict[str, float]] = {}
    for model_name, model in models.items():
        all_metrics[model_name] = test_model(
            val_loader,
            train_config.dataset.path_to_data,
            output_path,
            model,
            model_name,
            0.5,
            0.5,
            to_visualize=True,
            processed_size=train_config.img_size,
            keep_ratio=train_config.keep_ratio,
            device=str(train_config.device),
            label_to_name=label_to_name,
        )

    metrics_df = pd.DataFrame.from_dict(all_metrics, orient="index")
    tabulated_data = tabulate(metrics_df.round(4), headers="keys", tablefmt="pretty", showindex=True)
    print("\n" + tabulated_data)


if __name__ == "__main__":
    main()
