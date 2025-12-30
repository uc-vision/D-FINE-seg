from pathlib import Path
from shutil import rmtree

import click
from loguru import logger
from tqdm import tqdm
import torch

from d_fine.config import TrainConfig
from d_fine.infer.torch_model import Torch_model
from d_fine.core import image_utils
from lib_detection.annotation.viewer import InstanceViewer, Key
from lib_gaussian import ClassConfig, ClassInfo
from ucvision_utility.concurrent.threaded import map_iter_threaded


@click.command()
@click.argument("training_path", type=click.Path(exists=True, path_type=Path))
@click.argument("image_dir", type=click.Path(exists=True, path_type=Path))
@click.option("--conf-thresh", type=float, help="Confidence threshold (default: from config)")
@click.option("--view", is_flag=True, help="View results interactively")
@click.option("--output-path", type=click.Path(path_type=Path), help="Output path for results")
def main(
  training_path: Path,
  image_dir: Path,
  conf_thresh: float | None,
  view: bool,
  output_path: Path | None,
) -> None:
  """Run inference with D-FINE-seg.

  TRAINING_PATH: Path to training output directory (contains config.json and model.pt)
  IMAGE_DIR: Directory containing images to process
  """
  try:
    # Load config.json from experiment directory
    config_path = training_path / "config.json"
    if not config_path.exists():
      # Try loading from base_path and exp_name if training_path is just base_path
      train_config = TrainConfig.load_from_experiment(training_path)
    else:
      train_config = TrainConfig.load(config_path)
  except Exception as e:
    raise click.BadParameter(f"Failed to load config: {e}")

  # Load label_to_name from classes.json
  classes_path = training_path / "classes.json"
  if not classes_path.exists():
    raise FileNotFoundError(f"Classes configuration not found at {classes_path}")

  from d_fine.config import ClassConfig

  class_config_obj = ClassConfig.load(classes_path)
  label_to_name = class_config_obj.label_to_name
  num_classes = len(label_to_name)

  model_path = training_path / "model.pt"
  if not model_path.exists():
    raise FileNotFoundError(f"Model file not found at {model_path}")

  torch_model = Torch_model.from_train_config(
    train_config,
    num_classes=num_classes,
    model_path=model_path,
    device=torch.device(train_config.device),
  )

  if view:
    class_config = ClassConfig(
      classes={name: ClassInfo(name=name, class_only=False) for name in label_to_name.values()}
    )
    viewer = InstanceViewer(class_config, window_name="D-FINE Inference")

  if output_path:
    if output_path.exists():
      rmtree(output_path)
    output_path.mkdir(parents=True)

  img_paths = [
    f for f in image_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
  ]

  def process_image(img_path: Path):
    img_rgb = image_utils.load_image(img_path)
    res = torch_model(img_rgb)
    return img_rgb, res, img_path

  for img_rgb, res, img_path in tqdm(
    map_iter_threaded(process_image, img_paths), total=len(img_paths), desc="Inference"
  ):
    if output_path:
      from d_fine.validation.visualization import visualize

      vis_img = visualize(img_rgb, res, label_to_name)
      image_utils.save_image(output_path / img_path.name, vis_img)

    if view:
      from lib_detection.annotation.instance_mask import PanopticImage

      panoptic = PanopticImage(rgb=img_rgb, instances=res.masks, class_config=class_config)
      viewer.set_image(panoptic, img_path.name)

      while True:
        key = viewer.wait_for_key()
        if key == Key.ESC:
          return
        elif key in {Key.SPACE, Key.RIGHT_ARROW}:
          break


if __name__ == "__main__":
  main()
