from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

from d_fine.config import ImageConfig, Mode


def init_augs(img_config: ImageConfig, mode: Mode) -> A.Compose:
  """Initialize augmentations based on mode and configuration."""
  if mode == Mode.TRAIN:
    cfg = img_config.augs
    g = cfg.geometric
    h = cfg.hsv
    
    transforms = [
      A.ShiftScaleRotate(
        shift_limit=g.shift_limit,
        scale_limit=g.scale_limit,
        rotate_limit=g.rotate_limit,
        p=0.5
      ),
      A.HorizontalFlip(p=cfg.left_right_flip),
      A.VerticalFlip(p=cfg.up_down_flip),
      A.RandomBrightnessContrast(
        brightness_limit=cfg.brightness.limit,
        p=cfg.brightness.prob
      ),
      A.HueSaturationValue(
        hue_shift_limit=h.hue_shift_limit,
        sat_shift_limit=h.sat_shift_limit,
        val_shift_limit=h.val_shift_limit,
        p=h.prob
      ),
      A.GaussNoise(std_range=cfg.noise.std_range, p=cfg.noise.prob),
      A.Blur(blur_limit=cfg.blur.limit, p=cfg.blur.prob),
    ]
  else:
    transforms = []

  # Always include resizing and normalization
  tw, th = img_config.target_size
  if not img_config.keep_aspect:
    transforms.append(A.Resize(height=th, width=tw))
  else:
    transforms.append(A.LongestMaxSize(max_size=max(tw, th)))
    transforms.append(A.PadIfNeeded(
      min_height=th,
      min_width=tw,
      border_mode=0,
      value=(114, 114, 114)
    ))

  transforms.extend([
    A.Normalize(mean=img_config.norm_mean, std=img_config.norm_std),
    ToTensorV2(),
  ])

  return A.Compose(
    transforms,
    bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels", "indices"]),
  )
