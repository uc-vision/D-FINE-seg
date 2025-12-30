from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from lib_detection.annotation import InstanceMask

from d_fine.core.types import ImageResult
from d_fine.dataset.detection.sample import DetectionSample
from d_fine.dataset.segmentation.sample import SegmentationSample
from d_fine.config import ImageConfig, Task, Mode
from d_fine.dataset.coco.coco_dataset import CocoDatasetConfig


def get_dataset_config(base_path: Path, task: Task, img_config: ImageConfig) -> CocoDatasetConfig:
  return CocoDatasetConfig(
    base_path=base_path,
    img_config=img_config,
    task=task,
    mode=Mode.TRAIN,
  )


def find_test_datasets() -> list[Path]:
  """Find all COCO JSON files in test_data."""
  test_data_dir = Path(__file__).parent / "test_data"
  return sorted(list(test_data_dir.glob("*.json")))


def generate_random_image(w: int, h: int) -> np.ndarray:
  """Generate a random RGB image."""
  return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)


def generate_random_boxes(n: int, w: int, h: int) -> np.ndarray:
  """Generate n random bounding boxes [label, x1, y1, x2, y2]."""
  if n == 0:
    return np.zeros((0, 5), dtype=np.float32)

  labels = np.random.randint(0, 80, (n, 1)).astype(np.float32)

  # Generate random corners
  x = np.random.randint(0, w - 10, (n, 2))
  y = np.random.randint(0, h - 10, (n, 2))

  x1 = x.min(axis=1, keepdims=True)
  x2 = x.max(axis=1, keepdims=True) + 5
  y1 = y.min(axis=1, keepdims=True)
  y2 = y.max(axis=1, keepdims=True) + 5

  return np.hstack([labels, x1, y1, x2, y2]).astype(np.float32)


def generate_random_id_map(w: int, h: int, boxes: np.ndarray) -> np.ndarray:
  """Generate a random ID map based on boxes."""
  id_map = np.zeros((h, w), dtype=np.uint16)
  for i, box in enumerate(boxes):
    x1, y1, x2, y2 = box[1:5].astype(int)
    # Clip to image boundaries
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 > x1 and y2 > y1:
      id_map[y1:y2, x1:x2] = i + 1
  return id_map


def create_random_detection_sample(w: int = 640, h: int = 480, n: int = 5) -> DetectionSample:
  """Create a randomized DetectionSample."""
  image = generate_random_image(w, h)
  targets = generate_random_boxes(n, w, h)
  return DetectionSample(
    image=image,
    targets=targets,
    orig_size=torch.tensor([w, h]),
    paths=(Path(f"random_{random.randint(0, 1000)}.jpg"),),
  )


def create_random_segmentation_sample(w: int = 640, h: int = 480, n: int = 5) -> SegmentationSample:
  """Create a randomized SegmentationSample."""
  image = generate_random_image(w, h)
  targets = generate_random_boxes(n, w, h)
  id_map = generate_random_id_map(w, h, targets)
  return SegmentationSample(
    image=image,
    labels=targets[:, 0].astype(np.int64),
    id_map=id_map,
    orig_size=torch.tensor([w, h]),
    paths=(Path(f"random_{random.randint(0, 1000)}.jpg"),),
  )


def make_random_instance_mask(label=0, score=1.0, shape=(10, 10), offset=(0, 0)):
  mask = torch.rand(shape) > 0.5
  if not mask.any():
    mask[0, 0] = True
  return InstanceMask(mask=mask, label=label, offset=offset, score=score)


@st.composite
def instance_mask_strategy(draw):
  w = draw(st.integers(min_value=1, max_value=20))
  h = draw(st.integers(min_value=1, max_value=20))
  # Ensure at least one pixel is True to avoid zero area
  mask_np = draw(arrays(np.bool_, (h, w)))
  if not np.any(mask_np):
    mask_np[0, 0] = True
  mask = torch.from_numpy(mask_np)
  label = draw(st.integers(min_value=0, max_value=10))
  x = draw(st.integers(min_value=0, max_value=100))
  y = draw(st.integers(min_value=0, max_value=100))
  score = draw(st.floats(min_value=0, max_value=1))
  return InstanceMask(mask=mask, label=label, offset=(x, y), score=score)


def jitter_mask(m: InstanceMask, jitter: int = 3) -> InstanceMask:
  """Shift an instance mask by a random amount."""
  dx, dy = np.random.randint(-jitter, jitter + 1, size=2).tolist()
  return InstanceMask(
    mask=m.mask, label=m.label, offset=(m.offset[0] + dx, m.offset[1] + dy), score=0.99
  )


def jitter_results(results: list[ImageResult], jitter: int = 3) -> list[ImageResult]:
  """Shift every instance in a list of ImageResults by a random amount."""
  from lib_detection.annotation import stack_boxes

  new_results = []
  for res in results:
    jittered_masks = [jitter_mask(m, jitter) for m in res.masks]
    new_results.append(
      ImageResult(
        labels=res.labels,
        boxes=stack_boxes(jittered_masks) if jittered_masks else torch.zeros((0, 4)),
        img_size=res.img_size,
        scores=torch.ones(len(res.labels)) * 0.99,
        masks=jittered_masks,
      )
    )
  return new_results
