from __future__ import annotations

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from typing import TYPE_CHECKING

from lib_detection.annotation.instance_mask import PanopticImage
from lib_detection.annotation.drawing import (
  ColorMode,
  Font,
  blend_rgba,
  instances_to_images,
  colorize_instances,
  draw_instance_boxes,
)
from lib_gaussian import ClassConfig, ClassInfo

from .utils import processed_sample_to_instances, dataloader_target_to_instances

if TYPE_CHECKING:
  from ..core.types import ImageResult
  from ..dataset.dataset import ProcessedSample, Dataset
  from ..config import ImageConfig


def render_panoptic(
  panoptic: PanopticImage,
  alpha: float = 0.5,
  show_masks: bool = True,
  mode: ColorMode = ColorMode.Instance,
) -> NDArray[np.uint8]:
  """Render a PanopticImage to a high-quality RGB ndarray using lib_detection primitives."""
  h, w = panoptic.rgb.shape[:2]
  img = panoptic.rgb.copy()

  if show_masks and panoptic.instances:
    # 1. Generate overlay
    instance_image, _, _ = instances_to_images(panoptic.instances, np.array([h, w]))

    if mode == ColorMode.Instance:
      overlay = colorize_instances(instance_image.cpu().numpy())
    else:
      overlay = np.zeros((h, w, 4), dtype=np.uint8)

    # 2. Blend
    img = blend_rgba(img, overlay, alpha)

    # 3. Draw boxes and labels
    class_names = panoptic.class_config.to_label_names()
    font = Font(font_size=12)

    from lib_detection.annotation.drawing import class_colors

    colors = class_colors(len(class_names))
    img = draw_instance_boxes(img, panoptic.instances, class_names, colors, font)

  return img


def processed_sample_to_panoptic(
  sample: ProcessedSample,
  label_to_name: dict[int, str],
  image: torch.Tensor | np.ndarray | None = None,
  denormalize: bool = True,
  config: ImageConfig | None = None,
) -> PanopticImage:
  """Convert a ProcessedSample to a PanopticImage for visualization."""
  if image is None:
    image = sample.image

  if isinstance(image, torch.Tensor):
    if denormalize:
      if config is None:
        from d_fine.config import ImageConfig

        config = ImageConfig(img_size=(image.shape[2], image.shape[1]))
      img_denorm = config.denormalize(image)
      img_np = (img_denorm.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
    else:
      img_np = (image.cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
  else:
    img_np = image.astype(np.uint8)

  instances = processed_sample_to_instances(sample)

  class_config = ClassConfig(
    classes={name: ClassInfo(name=name, class_only=False) for name in label_to_name.values()}
  )
  return PanopticImage(rgb=img_np, instances=instances, class_config=class_config)


def image_result_to_panoptic(
  res: ImageResult,
  label_to_name: dict[int, str],
  image: torch.Tensor | np.ndarray,
  denormalize: bool = True,
  config: ImageConfig | None = None,
) -> PanopticImage:
  """Convert an ImageResult to a PanopticImage for visualization."""
  if isinstance(image, torch.Tensor):
    if denormalize:
      if config is None:
        from d_fine.config import ImageConfig

        config = ImageConfig(img_size=(image.shape[2], image.shape[1]))
      img_denorm = config.denormalize(image)
      img_np = (img_denorm.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
    else:
      img_np = (image.cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
  else:
    img_np = image.astype(np.uint8)

  instances = res.masks

  class_config = ClassConfig(
    classes={name: ClassInfo(name=name, class_only=False) for name in label_to_name.values()}
  )
  return PanopticImage(rgb=img_np, instances=instances, class_config=class_config)


def visualize_processed_sample(
  sample: ProcessedSample,
  label_to_name: dict[int, str],
  image: torch.Tensor | np.ndarray | None = None,
  denormalize: bool = True,
  config: ImageConfig | None = None,
) -> NDArray[np.uint8]:
  """Visualize a ProcessedSample (output from dataloader)."""
  panoptic = processed_sample_to_panoptic(
    sample, label_to_name, image=image, denormalize=denormalize, config=config
  )
  return render_panoptic(panoptic)


def visualize_image_result(
  res: ImageResult,
  label_to_name: dict[int, str],
  image: torch.Tensor | np.ndarray,
  denormalize: bool = True,
  config: ImageConfig | None = None,
) -> NDArray[np.uint8]:
  """Visualize an ImageResult (output from model)."""
  panoptic = image_result_to_panoptic(
    res, label_to_name, image=image, denormalize=denormalize, config=config
  )
  return render_panoptic(panoptic)


def visualize_batch(
  images: torch.Tensor,
  targets: list[dict],
  label_to_name: dict[int, str],
  dataset: Dataset | None = None,
) -> list[NDArray[np.uint8]]:
  """Visualize a batch of images and targets."""
  from .utils import dataloader_target_to_image_result

  vis_images = []
  for i in range(len(targets)):
    res = dataloader_target_to_image_result(targets[i])

    # Use config from dataset if available
    config = dataset.config.img_config if dataset is not None else None

    img_vis = visualize_image_result(
      res, label_to_name, image=images[i], denormalize=bool(dataset), config=config
    )
    vis_images.append(img_vis)
  return vis_images
