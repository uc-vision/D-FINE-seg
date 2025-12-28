from __future__ import annotations

import random
import numpy as np
import torch
import cv2


def box_candidates(
  box1: np.ndarray,
  box2: np.ndarray,
  wh_thr: float = 2,
  ar_thr: float = 100,
  area_thr: float = 0.1,
  eps: float = 1e-16,
) -> np.ndarray:
  """Check if boxes are valid candidates for augmentation."""
  w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
  w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
  ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))
  return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)


def refine_boxes_from_affine(
  targets: np.ndarray, M: np.ndarray
) -> tuple[np.ndarray, torch.Tensor, np.ndarray]:
  """Refine boxes using affine transformation matrix."""
  xy = np.ones((len(targets) * 4, 3), dtype=np.float32)
  xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(-1, 2)
  xy = xy @ M.T
  xy = xy[:, :2].reshape(-1, 8)
  new_boxes_np = np.stack(
    [
      xy[:, [0, 2, 4, 6]].min(1),
      xy[:, [1, 3, 5, 7]].min(1),
      xy[:, [0, 2, 4, 6]].max(1),
      xy[:, [1, 3, 5, 7]].max(1),
    ],
    axis=1,
  )

  keep = box_candidates(box1=targets[:, 1:5].T, box2=new_boxes_np.T, area_thr=0.1)

  return (
    targets[keep],
    torch.as_tensor(new_boxes_np[keep], dtype=torch.float32),
    np.where(keep)[0],
  )


def abs_xyxy_to_norm_xywh(box: np.ndarray, w: int, h: int) -> np.ndarray:
  """Convert absolute XYXY to normalized XYWH."""
  if len(box) == 0:
    return box
  new_box = box.copy()
  new_box[:, 0] = (box[:, 0] + box[:, 2]) / (2 * w)
  new_box[:, 1] = (box[:, 1] + box[:, 3]) / (2 * h)
  new_box[:, 2] = (box[:, 2] - box[:, 0]) / w
  new_box[:, 3] = (box[:, 3] - box[:, 1]) / h
  return new_box


def norm_xywh_to_abs_xyxy(box: np.ndarray, w: int, h: int) -> np.ndarray:
  """Convert normalized XYWH to absolute XYXY."""
  if len(box) == 0:
    return box
  new_box = box.copy()
  new_box[:, 0] = (box[:, 0] - box[:, 2] / 2) * w
  new_box[:, 1] = (box[:, 1] - box[:, 3] / 2) * h
  new_box[:, 2] = (box[:, 0] + box[:, 2] / 2) * w
  new_box[:, 3] = (box[:, 1] + box[:, 3] / 2) * h
  return new_box


def get_transform_matrix(
  img_shape: tuple[int, int],
  target_shape: tuple[int, int],
  degrees: float,
  scale: float,
  shear: float,
  translate: float,
) -> tuple[np.ndarray, float]:
  """Compute transformation matrix for mosaic augmentation."""
  import math

  # Rotation and Scale
  C = np.eye(3)
  C[0, 2] = -img_shape[1] / 2  # x center
  C[1, 2] = -img_shape[0] / 2  # y center

  R = np.eye(3)
  angle = random.uniform(-degrees, degrees)
  if isinstance(scale, tuple):
    s = random.uniform(scale[0], scale[1])
  else:
    s = random.uniform(1 - scale, 1 + scale)
  R[:2] = cv2.getRotationMatrix2D(angle=(angle), center=(0, 0), scale=s)

  # Shear
  S = np.eye(3)
  S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
  S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

  # Translation
  T = np.eye(3)
  T[0, 2] = (
    random.uniform(0.5 - translate, 0.5 + translate) * target_shape[1]
  )  # x translation (pixels)
  T[1, 2] = (
    random.uniform(0.5 - translate, 0.5 + translate) * target_shape[0]
  )  # y translation (pixels)

  # Combined matrix
  M = T @ S @ R @ C
  return M, s
