from __future__ import annotations

import math
import random
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

from d_fine.config import Mode
from d_fine.config import ImageConfig
from d_fine.core.box_utils import box_candidates


class LetterboxRect:
  """Resize and pad image to target size while maintaining aspect ratio."""

  def __init__(self, height: int, width: int, color=(114, 114, 114)):
    self.height = height
    self.width = width
    self.color = color

  def __call__(
    self,
    image: np.ndarray,
    bboxes: np.ndarray = None,
    labels: np.ndarray = None,
    mask: np.ndarray = None,
  ) -> dict:
    h, w = image.shape[:2]
    r = min(self.height / h, self.width / w)
    nh, nw = int(h * r), int(w * r)
    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (self.height - nh) // 2
    bottom = self.height - nh - top
    left = (self.width - nw) // 2
    right = self.width - nw - left
    image = cv2.copyMakeBorder(
      image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.color
    )
    res = {"image": image}
    if bboxes is not None:
      bboxes = bboxes.copy()
      bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * r + left
      bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * r + top
      res["bboxes"] = bboxes
    if labels is not None:
      res["labels"] = labels
    if mask is not None:
      mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
      mask = cv2.copyMakeBorder(mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
      res["mask"] = mask
    return res


def random_affine(
  img, targets=(), masks=None, degrees=10, translate=0.1, scale=0.1, shear=10, border=(0, 0)
):
  """Apply random affine transformation to image and targets."""
  height = img.shape[0] + border[0] * 2
  width = img.shape[1] + border[1] * 2
  C = np.eye(3)
  C[0, 2] = -img.shape[1] / 2
  C[1, 2] = -img.shape[0] / 2
  R = np.eye(3)
  a = random.uniform(-degrees, degrees)
  s = random.uniform(1 - scale, 1 + scale)
  R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)
  S = np.eye(3)
  S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
  S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
  T = np.eye(3)
  T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width
  T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height
  M = T @ S @ R @ C
  if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():
    if masks is not None:
      img = cv2.warpAffine(
        img, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114)
      )
      masks = cv2.warpAffine(
        masks, M[:2], dsize=(width, height), flags=cv2.INTER_NEAREST, borderValue=0
      )
    else:
      img = cv2.warpAffine(
        img, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114)
      )
  n = len(targets)
  if n:
    new = np.zeros((n, 4))
    xy = np.ones((n * 4, 3))
    xy[:, :2] = targets[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)
    xy = xy @ M.T
    xy = xy[:, :2].reshape(n, 8)
    x = xy[:, [0, 2, 4, 6]]
    y = xy[:, [1, 3, 5, 7]]
    new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
    new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
    new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)
    i = box_candidates(box1=targets[:, :4].T * s, box2=new.T, area_thr=0.1)
    targets = targets[i]
    targets[:, :4] = new[i]
  return img, targets, masks


def train_augs(img_config: ImageConfig) -> A.Compose:
  target_h, target_w = img_config.target_h, img_config.target_w
  crop_size = img_config.crop_size
  geo_cfg = img_config.augs.geometric

  geometric = [
    A.HorizontalFlip(p=img_config.augs.left_right_flip),
    A.VerticalFlip(p=img_config.augs.up_down_flip),
    A.RandomRotate90(p=img_config.augs.rotate_90_prob),
    A.ShiftScaleRotate(
      shift_limit=geo_cfg.shift_limit,
      scale_limit=geo_cfg.scale_limit,
      rotate_limit=geo_cfg.rotate_limit,
      interpolation=cv2.INTER_LINEAR,
      mask_interpolation=cv2.INTER_NEAREST,
      border_mode=cv2.BORDER_CONSTANT,
      value=(114, 114, 114),
      p=1.0,
    ),
  ]

  crop = [A.RandomCrop(height=crop_size[0], width=crop_size[1], p=1.0)] if crop_size else []

  resize = [
    LetterboxRect(height=target_h, width=target_w)
    if img_config.keep_aspect
    else A.Resize(height=target_h, width=target_w, interpolation=cv2.INTER_AREA)
  ]

  pixel = [
    A.CoarseDropout(
      num_holes_range=(1, 2),
      hole_height_range=(0.05, 0.15),
      hole_width_range=(0.05, 0.15),
      p=img_config.augs.coarse_dropout_prob,
    ),
    A.RandomBrightnessContrast(
      brightness_limit=img_config.augs.brightness.limit, p=img_config.augs.brightness.prob
    ),
    A.RandomGamma(gamma_limit=img_config.augs.gamma.limit, p=img_config.augs.gamma.prob),
    A.Blur(blur_limit=(3, max(3, img_config.augs.blur.limit)), p=img_config.augs.blur.prob),
    A.GaussNoise(std_range=img_config.augs.noise.std_range, p=img_config.augs.noise.prob),
    A.ToGray(p=img_config.augs.to_gray_prob),
    A.HueSaturationValue(
      hue_shift_limit=img_config.augs.hsv.hue_shift_limit,
      sat_shift_limit=img_config.augs.hsv.sat_shift_limit,
      val_shift_limit=img_config.augs.hsv.val_shift_limit,
      p=img_config.augs.hsv.prob,
    ),
  ]

  return A.Compose(
    geometric
    + crop
    + resize
    + pixel
    + [A.Normalize(mean=img_config.norm[0], std=img_config.norm[1]), ToTensorV2()],
    bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
  )


def eval_augs(img_config: ImageConfig) -> A.Compose:
  target_h, target_w = img_config.target_h, img_config.target_w
  crop_size = img_config.crop_size

  crop = [A.CenterCrop(height=crop_size[0], width=crop_size[1], p=1.0)] if crop_size else []

  resize = [
    LetterboxRect(height=target_h, width=target_w, color=(114, 114, 114))
    if img_config.keep_aspect
    else A.Resize(target_h, target_w, interpolation=cv2.INTER_AREA)
  ]

  return A.Compose(
    crop + resize + [A.Normalize(mean=img_config.norm[0], std=img_config.norm[1]), ToTensorV2()],
    bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
  )


def init_augs(img_config: ImageConfig, mode: Mode) -> A.Compose:
  return train_augs(img_config) if mode == Mode.TRAIN else eval_augs(img_config)
