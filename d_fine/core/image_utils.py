from __future__ import annotations

from pathlib import Path
from collections.abc import Iterator

import cv2
import numpy as np


def load_image(path: Path) -> np.ndarray:
  """Load image from path and return as RGB numpy array."""
  img = cv2.imread(str(path))
  if img is None:
    raise FileNotFoundError(f"Failed to load image: {path}")
  return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_image(path: Path, img: np.ndarray) -> None:
  """Save RGB image as BGR to path."""
  cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def load_video(path: Path) -> Iterator[np.ndarray]:
  """Iterate over video frames as RGB numpy arrays."""
  cap = cv2.VideoCapture(str(path))
  try:
    while True:
      ret, frame = cap.read()
      if not ret:
        break
      yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  finally:
    cap.release()
