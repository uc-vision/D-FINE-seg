import torch
import numpy as np
from pathlib import Path
from d_fine.dataset.detection.sample import DetectionSample
from d_fine.dataset.dataset import CompactMasks, ProcessedSample
from d_fine.core.box_utils import refine_boxes_from_affine

def test_detection_sample_apply_transform():
    # Setup dummy data
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    targets = np.array([
        [0, 10, 10, 20, 20],  # class 0, x1, y1, x2, y2
        [1, 50, 50, 70, 70]   # class 1, x1, y1, x2, y2
    ], dtype=np.float32)
    paths = (Path("test.jpg"),)
    orig_size = torch.tensor([100, 100])
    
    sample = DetectionSample(
        image=image,
        targets=targets,
        paths=paths,
        orig_size=orig_size
    )
    
    # Mock transform (Albumentations-like)
    def mock_transform(image, bboxes, class_labels):
        return {
            "image": torch.zeros((3, 50, 50)),
            "bboxes": bboxes, # keep same relative boxes for simplicity in mock
            "class_labels": class_labels
        }
    
    processed = sample.apply_transform(mock_transform)
    
    assert isinstance(processed, ProcessedSample)
    assert processed.image.shape == (3, 50, 50)
    assert len(processed.labels) == 2
    assert torch.equal(processed.labels, torch.tensor([0, 1], dtype=torch.int64))
    assert processed.boxes.shape == (2, 4)
    assert isinstance(processed.masks, CompactMasks)
    assert processed.masks.id_map.shape == (50, 50)

def test_detection_sample_warp_affine():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    targets = np.array([
        [0, 10, 10, 20, 20]
    ], dtype=np.float32)
    sample = DetectionSample(
        image=image,
        targets=targets,
        paths=(Path("test.jpg"),),
        orig_size=torch.tensor([100, 100])
    )
    
    # Shift by 10 pixels in both directions
    M = np.array([
        [1, 0, 10],
        [0, 1, 10],
        [0, 0, 1]
    ], dtype=np.float32)
    
    warped = sample.warp_affine(M, (100, 100))
    
    assert warped.image.shape == (100, 100, 3)
    # The box [10, 10, 20, 20] should become [20, 20, 30, 30]
    assert warped.targets.shape == (1, 5)
    assert warped.targets[0, 0] == 0
    assert np.allclose(warped.targets[0, 1:], [20, 20, 30, 30])


def test_detection_dataset_len(tmp_path):
    from d_fine.dataset.coco.coco_dataset import CocoDatasetConfig
    from d_fine.dataset.detection.dataset import DetectionDataset
    from d_fine.config import Task, Mode, ImageConfig, AugsConfig, MosaicAugsConfig
    import json

    # Create a dummy COCO file
    ann_file = tmp_path / "anns.json"
    coco_data = {
        "images": [{"id": 1, "file_name": "1.jpg", "height": 100, "width": 100}],
        "annotations": [],
        "categories": [{"id": 1, "name": "cat"}]
    }
    with open(ann_file, "w") as f:
        json.dump(coco_data, f)

    img_cfg = ImageConfig(
        img_size=(640, 640),
        augs=AugsConfig(),
        mosaic_augs=MosaicAugsConfig(mosaic_prob=0.0) # Disable mosaic for simple test
    )
    cfg = CocoDatasetConfig(
        base_path=tmp_path,
        task=Task.DETECT,
        mode=Mode.TRAIN,
        img_config=img_cfg
    )

    # We need to mock CocoFile.load to avoid actual file loading if possible, 
    # but since we wrote a real file, we can just use it.
    # However, load_image will still fail unless we create the directory.
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "1.jpg").touch()

    dataset = DetectionDataset(cfg, ann_file)
    assert len(dataset) == 1
