from __future__ import annotations

import cv2
import numpy as np
import torch


def refine_boxes_from_affine(
  targets: np.ndarray, M: np.ndarray, area_thr: float = 0.01
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Apply affine transform to boxes and filter invalid ones.

  Targets: [N, 5] (label, x1, y1, x2, y2)
  M: [3, 3] affine matrix
  """
  if targets.size == 0:
    return targets, np.zeros(0, dtype=bool), np.zeros(0, dtype=int)

  n = len(targets)
  boxes = targets[:, 1:5]
  
  # Corners [N, 4, 2]
  corners = np.zeros((n, 4, 2))
  corners[:, 0] = boxes[:, [0, 1]]  # x1, y1
  corners[:, 1] = boxes[:, [2, 1]]  # x2, y1
  corners[:, 2] = boxes[:, [2, 3]]  # x2, y2
  corners[:, 3] = boxes[:, [0, 3]]  # x1, y2

  # Transform corners
  corners_flat = corners.reshape(-1, 2)
  corners_hom = np.hstack([corners_flat, np.ones((len(corners_flat), 1))])
  t_corners = corners_hom @ M[:2].T
  t_corners = t_corners.reshape(n, 4, 2)

  # New bounding boxes [N, 4]
  new_boxes = np.zeros((n, 4))
  new_boxes[:, 0] = t_corners[:, :, 0].min(1)
  new_boxes[:, 1] = t_corners[:, :, 1].min(1)
  new_boxes[:, 2] = t_corners[:, :, 0].max(1)
  new_boxes[:, 3] = t_corners[:, :, 1].max(1)

  # Filter candidates
  keep = box_candidates(box1=boxes.T, box2=new_boxes.T, area_thr=area_thr)
  
  res_targets = np.hstack([targets[keep, :1], new_boxes[keep]])
  return res_targets, keep, np.where(keep)[0]


def box_candidates(
  box1: np.ndarray,
  box2: np.ndarray,
  wh_thr: float = 2,
  ar_thr: float = 100,
  area_thr: float = 0.1,
  eps: float = 1e-16,
) -> np.ndarray:
  """Filter box candidates based on size and aspect ratio changes.

  Args:
      box1: [4, N] (x1, y1, x2, y2) before transform
      box2: [4, N] (x1, y1, x2, y2) after transform
  """
  w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
  w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
  ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
  return (
    (w2 > wh_thr)
    & (h2 > wh_thr)
    & (w2 * h2 / (w1 * h1 + eps) > area_thr)
    & (ar < ar_thr)
  )


def get_transform_matrix(
  img_size: tuple[int, int],
  target_size: tuple[int, int],
  degrees: float = 0.0,
  scale: tuple[float, float] = (1.0, 1.0),
  shear: float = 0.0,
  translate: float = 0.0,
) -> tuple[np.ndarray, float]:
  """Compute affine transform matrix for mosaic/aug.
  
  img_size and target_size are (width, height).
  """
  w, h = img_size
  tw, th = target_size

  # Center
  C = np.eye(3)
  C[0, 2] = -w / 2
  C[1, 2] = -h / 2

  # Rotation and Scale
  R = np.eye(3)
  a = np.random.uniform(-degrees, degrees)
  s = np.random.uniform(scale[0], scale[1])
  R[:2] = cv2.getRotationMatrix2D(center=(0, 0), angle=a, scale=s)

  # Shear
  S = np.eye(3)
  S[0, 1] = np.tan(np.random.uniform(-shear, shear) * np.pi / 180)
  S[1, 0] = np.tan(np.random.uniform(-shear, shear) * np.pi / 180)

  # Translation
  T = np.eye(3)
  T[0, 2] = np.random.uniform(0.5 - translate, 0.5 + translate) * tw
  T[1, 2] = np.random.uniform(0.5 - translate, 0.5 + translate) * th

  # Combined matrix
  M = T @ S @ R @ C
  return M, s


def abs_xyxy_to_norm_xywh(boxes: torch.Tensor, w: int, h: int) -> torch.Tensor:
  """Convert absolute xyxy to normalized xywh."""
  if boxes.numel() == 0:
    return boxes
  res = boxes.clone()
  res[:, [0, 2]] /= w
  res[:, [1, 3]] /= h
  res[:, 2] -= res[:, 0]
  res[:, 3] -= res[:, 1]
  res[:, 0] += res[:, 2] / 2
  res[:, 1] += res[:, 3] / 2
  return res


def norm_xywh_to_abs_xyxy(boxes: torch.Tensor, w: int, h: int) -> torch.Tensor:
  """Convert normalized xywh to absolute xyxy."""
  if boxes.numel() == 0:
    return boxes
  res = boxes.clone()
  res[:, 0] -= res[:, 2] / 2
  res[:, 1] -= res[:, 3] / 2
  res[:, 2] += res[:, 0]
  res[:, 3] += res[:, 1]
  res[:, [0, 2]] *= w
  res[:, [1, 3]] *= h
  return res
