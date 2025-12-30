import json
import random
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from d_fine.config import ImageConfig, Mode, MosaicAugsConfig, MultiscaleConfig, Task
from d_fine.dataset.coco.coco_dataset import CocoDatasetConfig
from d_fine.dataset.dataset import CompactMasks, ProcessedSample
from d_fine.dataset.detection import DetectionDataset, DetectionSample
from d_fine.dataset.segmentation import SegmentationDataset, SegmentationSample
from d_fine.tests.utils import (
  create_random_detection_sample,
  create_random_segmentation_sample,
  find_test_datasets,
  get_dataset_config,
)


@pytest.fixture
def test_data_dir():
  return Path(__file__).parent / "test_data"


@pytest.fixture
def image_config():
  return ImageConfig(
    img_size=(640, 640), keep_aspect=True, mosaic_augs=MosaicAugsConfig(mosaic_prob=0.5)
  )


@pytest.mark.parametrize(
  "task, dataset_class, sample_class",
  [
    (Task.DETECT, DetectionDataset, DetectionSample),
    (Task.SEGMENT, SegmentationDataset, SegmentationSample),
  ],
)
@pytest.mark.parametrize("data_path", find_test_datasets())
@patch("d_fine.dataset.coco.coco_dataset.load_image")
def test_dataset_execution_paths(
  mock_load_image, task, dataset_class, sample_class, data_path, image_config
):
  # Mock image to match what's expected in the dataset
  mock_load_image.return_value = np.zeros((1536, 1536, 3), dtype=np.uint8)

  config = get_dataset_config(data_path.parent.parent, task, image_config)
  dataset = dataset_class(config, data_path)

  # 1. Test without mosaic
  with patch("random.random", return_value=1.0):
    sample = dataset[0]
    assert sample.image.shape == (3, 640, 640)
    assert sample.labels.ndim == 1
    assert sample.boxes.ndim == 2
    assert sample.boxes.shape[1] == 4

  # 2. Test with mosaic
  with patch("random.random", return_value=0.0):
    sample = dataset[0]
    assert sample.image.shape == (3, 640, 640)
    assert len(sample.paths) == 4


def test_compact_masks_box_derivation():
  # Create a simple 10x10 ID map with one object
  id_map = torch.zeros((10, 10), dtype=torch.uint16)
  id_map[2:5, 3:7] = 1  # Rows 2,3,4; Cols 3,4,5,6. Inclusive Box: [3, 2, 6, 4]

  indices = torch.tensor([0], dtype=torch.int64)  # Index 0 corresponds to ID 1
  masks = CompactMasks(id_map=id_map, indices=indices)

  boxes = masks.get_boxes()
  assert boxes.shape == (1, 4)
  assert torch.allclose(boxes[0], torch.tensor([3.0, 2.0, 6.0, 4.0]))


def test_segmentation_sample_box_derivation():
  image = np.zeros((10, 10, 3), dtype=np.uint8)
  id_map = np.zeros((10, 10), dtype=np.uint16)
  id_map[1:3, 1:4] = 1  # y: 1,2; x: 1,2,3 -> [1, 1, 3, 2]
  id_map[5:8, 6:9] = 2  # y: 5,6,7; x: 6,7,8 -> [6, 5, 8, 7]

  sample = SegmentationSample(
    image=image,
    labels=np.array([10, 20], dtype=np.int64),
    orig_size=torch.tensor([10, 10]),
    paths=(Path("test.jpg"),),
    id_map=id_map,
  )

  boxes = sample.get_boxes()
  assert boxes.shape == (2, 4)
  assert np.allclose(boxes[0], [1.0, 1.0, 3.0, 2.0])
  assert np.allclose(boxes[1], [6.0, 5.0, 8.0, 7.0])


def test_segmentation_sample_apply_transform():
  sample = create_random_segmentation_sample(w=100, h=100, n=2)

  def mock_transform(image, bboxes, class_labels, mask, indices, **kwargs):
    return {
      "image": torch.zeros((3, 50, 50)),
      "bboxes": bboxes,
      "class_labels": class_labels,
      "mask": np.zeros((50, 50), dtype=np.uint16),
      "indices": indices,
    }

  processed = sample.apply_transform(mock_transform)
  assert isinstance(processed, ProcessedSample)
  assert processed.image.shape == (3, 50, 50)
  assert len(processed.labels) == len(processed.masks.indices)
  assert processed.masks.id_map.shape == (50, 50)


def test_segmentation_sample_warp_affine():
  sample = create_random_segmentation_sample(w=100, h=100, n=1)

  # Shift by 10 pixels
  M = np.array([[1, 0, 10], [0, 1, 10], [0, 0, 1]], dtype=np.float32)
  warped = sample.warp_affine(M, (100, 100))

  assert warped.image.shape == (100, 100, 3)
  assert warped.id_map.shape == (100, 100)
  if sample.id_map.any():
    assert warped.id_map.any()


def test_detection_sample_apply_transform():
  sample = create_random_detection_sample(w=100, h=100, n=2)

  def mock_transform(image, bboxes, class_labels, **kwargs):
    return {
      "image": torch.zeros((3, 50, 50)),
      "bboxes": bboxes,
      "class_labels": class_labels,
    }

  processed = sample.apply_transform(mock_transform)
  assert isinstance(processed, ProcessedSample)
  assert processed.image.shape == (3, 50, 50)
  assert len(processed.labels) == len(processed.boxes)
  assert processed.boxes.shape[1] == 4


def test_detection_sample_warp_affine():
  sample = create_random_detection_sample(w=100, h=100, n=1)
  orig_box = sample.targets[0, 1:].copy()

  M = np.array([[1, 0, 10], [0, 1, 10], [0, 0, 1]], dtype=np.float32)
  warped = sample.warp_affine(M, (200, 200))

  assert warped.image.shape == (200, 200, 3)
  assert warped.targets.shape == (1, 5)
  expected_box = orig_box + [10, 10, 10, 10]
  assert np.allclose(warped.targets[0, 1:], expected_box)


def test_detection_dataset_len(tmp_path):
  ann_file = tmp_path / "anns.json"
  coco_data = {
    "info": {"year": 2025, "version": "1.0", "description": "test", "date_created": "2025-01-01"},
    "licenses": [{"id": 0, "name": "test", "url": "test"}],
    "categories": [{"id": 1, "name": "cat", "supercategory": "none", "isthing": True}],
    "images": [{"id": 1, "file_name": "1.jpg", "height": 100, "width": 100, "license": 0}],
    "annotations": [],
  }
  with open(ann_file, "w") as f:
    json.dump(coco_data, f)

  img_cfg = ImageConfig(img_size=(640, 640), mosaic_augs=MosaicAugsConfig(mosaic_prob=0.0))
  cfg = CocoDatasetConfig(base_path=tmp_path, task=Task.DETECT, mode=Mode.TRAIN, img_config=img_cfg)

  (tmp_path / "images").mkdir()
  (tmp_path / "images" / "1.jpg").touch()

  dataset = DetectionDataset(cfg, ann_file)
  assert len(dataset) == 1
