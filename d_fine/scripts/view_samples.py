from __future__ import annotations

from pathlib import Path

import click
import torch
from loguru import logger

from d_fine.config import TrainConfig
from d_fine.dataset import get_loader_class
from d_fine.validation.visualization import image_result_to_panoptic
from lib_detection.annotation.viewer import InstanceViewer, Key


def run_view_samples(config: TrainConfig) -> None:
  """Visualize training samples with augmentations, image by image from batches."""
  logger.info(f"Visualizing training samples for {config.exp_name}")

  loader_class = get_loader_class(config.dataset)
  loader = loader_class(config.dataset, batch_size=config.batch_size, num_workers=0)
  train_loader, _, _ = loader.build_dataloaders()
  
  label_to_name = loader.label_to_name
  # Use a temporary class config for the viewer until we have one from the panoptic image
  from lib_gaussian import ClassConfig, ClassInfo
  temp_class_config = ClassConfig(
    classes={name: ClassInfo(name=name, class_only=False) for name in label_to_name.values()}
  )
  viewer = InstanceViewer(temp_class_config, window_name="Training Samples")

  logger.info("Interacting with viewer: [Space/Right] Next sample, [Esc] Exit")

  from d_fine.validation import dataloader_target_to_image_result

  for images, targets, paths in train_loader:
    batch_size = images.shape[0]
    for i in range(batch_size):
      t = targets[i]
      res = dataloader_target_to_image_result(t)

      panoptic = image_result_to_panoptic(
        res, label_to_name, image=images[i], denormalize=True, config=config.dataset.img_config
      )
      
      viewer.set_image(panoptic, f"Batch Sample: {t['paths'][0].name if t['paths'] else 'None'}")

      while True:
        key = viewer.wait_for_key()
        if key == Key.ESC:
          return
        elif key in {Key.SPACE, Key.RIGHT_ARROW}:
          break


@click.command()
@click.argument("base-path", type=click.Path(exists=True, path_type=Path))
@click.option(
  "--search-path",
  multiple=True,
  type=click.Path(exists=True, path_type=Path),
  help="Additional Hydra search paths",
)
@click.option("--project-name", default="d-fine", help="Project name for wandb")
@click.option("--exp-name", default="default", help="Experiment name")
@click.argument("overrides", nargs=-1)
def main(
  base_path: Path, search_path: tuple[Path, ...], project_name: str, exp_name: str, overrides: tuple[str, ...]
) -> None:
  """Visualize training samples.

  BASE_PATH: Base dataset path.
  OVERRIDES: Additional Hydra config overrides (e.g., model_name=s task=segment)
  """
  override_list = list(overrides)
  if search_path:
    # Format search paths for Hydra
    paths = [f"file://{p.absolute()}" for p in search_path]
    override_list.append(f"hydra.searchpath=[{','.join(paths)}]")

  config = TrainConfig.from_hydra(
    base_path=base_path,
    overrides=override_list,
    project_name=project_name,
    exp_name=exp_name,
  )
  run_view_samples(config)


if __name__ == "__main__":
  main()
