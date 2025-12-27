from pathlib import Path

import click
import nncf
import numpy as np
import openvino as ov
import torch
from loguru import logger
from torch.utils.data import DataLoader

from d_fine.config import TrainConfig
from d_fine.dl.train import Trainer
from d_fine.validation import Validator


@click.command()
@click.option("--project-name", required=True, help="Project name")
@click.option("--base-path", required=True, type=click.Path(exists=True, path_type=Path), help="Base project path")
@click.option("--exp-name", type=str, help="Experiment name (defaults to latest)")
def main(project_name: str, base_path: Path, exp_name: str | None) -> None:
    """Run INT8 quantization with accuracy control on OpenVINO IR model.
    
    Loads config from saved training experiment.
    Expects FP32 IR at <train_config.paths.path_to_save>/model.xml.
    """
    try:
        train_config = TrainConfig.load_from_experiment(base_path, exp_name)
    except FileNotFoundError as e:
        raise click.BadParameter(str(e))
    
    save_dir = train_config.paths.path_to_save
    fp32_xml_path = save_dir / "model.xml"
    if not fp32_xml_path.exists():
        raise FileNotFoundError(f"FP32 OpenVINO model not found: {fp32_xml_path}")

    logger.info(f"Using FP32 OpenVINO model: {fp32_xml_path}")

    temp_loader = train_config.dataset.create_loader(batch_size=1, num_workers=train_config.num_workers)
    _, val_loader, _ = temp_loader.build_dataloaders(distributed=False)
    logger.info(f"Val images: {len(val_loader.dataset)}")

    label_to_name = val_loader.dataset.label_to_name
    num_labels = val_loader.dataset.num_classes

    # OpenVINO model
    core = ov.Core()
    model = core.read_model(fp32_xml_path)

    # NNCF datasets
    # DataLoader returns: (images, targets, img_paths)
    def transform_fn(data_item: tuple[torch.Tensor, dict[str, torch.Tensor], str]) -> np.ndarray:
        images, _, _ = data_item
        # NNCF expects numpy inputs for OpenVINO model
        return images.numpy().astype(np.float32)

    calibration_dataset = nncf.Dataset(val_loader, transform_fn)
    validation_dataset = nncf.Dataset(val_loader, transform_fn)

    def validate(compiled_model: ov.CompiledModel, validation_loader: DataLoader) -> float:
        """
        compiled_model: openvino.CompiledModel
        validation_loader: torch.utils.data.DataLoader
        returns: mAP_50 (float)
        """
        output_logits = compiled_model.output("logits")
        output_boxes = compiled_model.output("boxes")

        all_preds: list[dict[str, torch.Tensor]] = []
        all_gt: list[dict[str, torch.Tensor]] = []

        for inputs, targets, _ in validation_loader:
            # inputs: [B, C, H, W], torch on CPU
            inputs_np = inputs.numpy().astype(np.float32)
            ov_res = compiled_model(inputs_np)

            logits_np = ov_res[output_logits]
            boxes_np = ov_res[output_boxes]

            logits = torch.from_numpy(logits_np)
            boxes = torch.from_numpy(boxes_np)

            outputs = {"pred_logits": logits, "pred_boxes": boxes}

            orig_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).float()

            preds = Trainer.preds_postprocess(
                inputs,
                outputs,
                orig_sizes,
                num_labels=num_labels,
                keep_ratio=train_config.keep_ratio,
                conf_thresh=train_config.conf_thresh,
            )
            gt = Trainer.gt_postprocess(
                inputs,
                targets,
                orig_sizes,
                keep_ratio=train_config.keep_ratio,
            )

            all_preds.extend(preds)
            all_gt.extend(gt)

        validator = Validator(
            all_gt,
            all_preds,
            label_to_name=label_to_name,
            conf_thresh=train_config.conf_thresh,
            iou_thresh=train_config.iou_thresh,
        )
        metrics = validator.compute_metrics(extended=False)
        f1_score = metrics["f1"]
        logger.info(f"Validation F1-score: {f1_score:.4f}")
        return f1_score

    # Run quantization with accuracy control
    max_drop = train_config.export.ov_int8_max_drop
    subset_size = train_config.export.ov_int8_subset_size

    logger.info(
        f"Starting INT8 quantization with accuracy control: "
        f"max_drop={max_drop}, subset_size={subset_size}"
    )

    quantized_model = nncf.quantize_with_accuracy_control(
        model,
        calibration_dataset=calibration_dataset,
        validation_dataset=validation_dataset,
        validation_fn=validate,
        max_drop=max_drop,
        drop_type=nncf.DropType.ABSOLUTE,
        preset=nncf.QuantizationPreset.MIXED,  # better accuracy than PERFORMANCE
        subset_size=subset_size,
    )

    # Save INT8 model
    int8_xml_path = save_dir / "model_int8.xml"
    int8_xml_path.parent.mkdir(parents=True, exist_ok=True)

    # Save without additional FP16 compression
    ov.save_model(quantized_model, str(int8_xml_path), compress_to_fp16=False)
    logger.info(f"INT8 model with accuracy control saved to: {int8_xml_path}")


if __name__ == "__main__":
    main()
