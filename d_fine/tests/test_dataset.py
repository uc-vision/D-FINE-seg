import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
from d_fine.config import Mode, Task, ImageConfig, MultiscaleConfig
from d_fine.dataset.detection.dataset import DetectionDataset
from d_fine.dataset.segmentation.dataset import SegmentationDataset
from d_fine.dataset.coco.coco_dataset import CocoDatasetConfig
from d_fine.dataset.dataset import ProcessedSample

@pytest.fixture
def test_data_path():
    return Path(__file__).parent / "test_data" / "cherry_subset.json"

@pytest.fixture
def mock_config():
    img_cfg = ImageConfig(img_size=(640, 640))
    return CocoDatasetConfig(
        base_path=Path("/tmp/fake_data"),
        task=Task.DETECT,
        mode=Mode.TRAIN,
        img_config=img_cfg,
        multiscale=MultiscaleConfig(prob=0.0)
    )

@patch("d_fine.dataset.coco.coco_dataset.load_image")
def test_detection_dataset(mock_load, test_data_path, mock_config):
    # Mock image (H, W, 3)
    mock_load.return_value = np.zeros((1536, 1536, 3), dtype=np.uint8)
    
    ds = DetectionDataset(mock_config, test_data_path)
    assert len(ds) > 0
    
    sample = ds.get_data(0)
    assert sample.image.shape == (1536, 1536, 3)
    assert sample.targets.shape[1] == 5  # [cls, x1, y1, x2, y2]
    assert len(sample.paths) == 1

    processed = ds[0]
    assert isinstance(processed, ProcessedSample)
    assert processed.image.shape == (3, 640, 640)
    assert processed.labels.dtype == torch.int64
    assert processed.boxes.shape[1] == 4

@patch("d_fine.dataset.coco.coco_dataset.load_image")
def test_segmentation_dataset(mock_load, test_data_path, mock_config):
    # Mock image (H, W, 3)
    mock_load.return_value = np.zeros((1536, 1536, 3), dtype=np.uint8)
    
    # Update config for segmentation
    seg_config = mock_config.model_copy(update={"task": Task.SEGMENT})
    
    ds = SegmentationDataset(seg_config, test_data_path)
    assert len(ds) > 0
    
    sample = ds.get_data(0)
    assert sample.image.shape == (1536, 1536, 3)
    assert sample.labels.ndim == 1
    assert sample.id_map.shape == (1536, 1536)
    
    processed = ds[0]
    assert isinstance(processed, ProcessedSample)
    assert processed.image.shape == (3, 640, 640)
    assert processed.masks.id_map.shape == (640, 640)

@patch("d_fine.dataset.coco.coco_dataset.load_image")
def test_detection_mosaic(mock_load, test_data_path, mock_config):
    mock_load.return_value = np.zeros((1536, 1536, 3), dtype=np.uint8)
    
    # Enable mosaic
    mosaic_config = mock_config.model_copy(
        update={"img_config": mock_config.img_config.model_copy(
            update={"mosaic_augs": mock_config.img_config.mosaic_augs.model_copy(update={"mosaic_prob": 1.0})}
        )}
    )
    
    ds = DetectionDataset(mosaic_config, test_data_path)
    processed = ds[0]
    assert isinstance(processed, ProcessedSample)
    assert processed.image.shape == (3, 640, 640)
    # Mosaic paths should have 4 entries (original + 3 random)
    assert len(processed.paths) == 4

@patch("d_fine.dataset.coco.coco_dataset.load_image")
def test_segmentation_mosaic(mock_load, test_data_path, mock_config):
    mock_load.return_value = np.zeros((1536, 1536, 3), dtype=np.uint8)
    
    # Update config for segmentation and enable mosaic
    seg_mosaic_config = mock_config.model_copy(
        update={
            "task": Task.SEGMENT,
            "img_config": mock_config.img_config.model_copy(
                update={"mosaic_augs": mock_config.img_config.mosaic_augs.model_copy(update={"mosaic_prob": 1.0})}
            )
        }
    )
    
    ds = SegmentationDataset(seg_mosaic_config, test_data_path)
    processed = ds[0]
    assert isinstance(processed, ProcessedSample)
    assert processed.image.shape == (3, 640, 640)
    assert len(processed.paths) == 4
    assert processed.masks.id_map.shape == (640, 640)
