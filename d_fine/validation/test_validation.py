import pytest
import torch
import numpy as np
from pathlib import Path
from hypothesis import given, strategies as st, settings
from hypothesis.extra.numpy import arrays
from d_fine.validation.matcher import greedy_match, box_iou_fn, mask_iou_fn, Match
from d_fine.validation.validator import Validator
from d_fine.validation.utils import coco_to_image_results
from d_fine.core.types import ImageResult
from image_detection.annotation import InstanceMask, stack_boxes
from image_detection.annotation.coco import load_coco_json

def make_random_instance_mask(label=0, score=1.0, shape=(10, 10), offset=(0, 0)):
    mask = torch.rand(shape) > 0.5
    if not mask.any():
        mask[0, 0] = True
    return InstanceMask(mask=mask, label=label, offset=offset, score=score)

def jitter_mask(m: InstanceMask, jitter: int = 3) -> InstanceMask:
    """Shift an instance mask by a random amount."""

    dx, dy = np.random.randint(-jitter, jitter + 1, size=2).tolist()
    return InstanceMask(
        mask=m.mask, 
        label=m.label, 
        offset=(m.offset[0] + dx, m.offset[1] + dy), 
        score=0.99
    )

def jitter_results(results: list[ImageResult], jitter: int = 3) -> list[ImageResult]:
    """Shift every instance in a list of ImageResults by a random amount."""
    return [
        ImageResult(
            labels=res.labels,
            boxes=stack_boxes(jittered_masks := [jitter_mask(m, jitter) for m in res.masks]),
            img_size=res.img_size,
            scores=torch.ones(len(res.labels)) * 0.99,
            masks=jittered_masks
        ) for res in results
    ]

def find_test_datasets() -> list[Path]:
    """Locate all test datasets in the test_data folder."""
    data_dir = Path(__file__).parent / "test_data"
    return list(data_dir.glob("*.json"))

@pytest.mark.parametrize("data_path", find_test_datasets())
def test_validator_real_gt_against_itself(data_path: Path):
    coco = load_coco_json(data_path)
    gt_results = coco_to_image_results(coco)
    
    label_to_name = {cat.id: cat.name for cat in coco.categories}
    validator = Validator(gt_results, gt_results, label_to_name=label_to_name)
    metrics = validator.compute_metrics()
    
    # GT vs GT should be perfect
    expected_tps = sum(len(r.labels) for r in gt_results)
    
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    assert metrics.iou == 1.0
    assert metrics.tps == expected_tps
    assert metrics.fps == 0
    assert metrics.fns == 0
    
    # Bbox AP
    assert metrics.map_50 == 1.0
    assert metrics.map_75 == 1.0
    assert metrics.map_50_95 == 1.0
    
    # Mask AP
    assert metrics.map_50_mask == 1.0
    assert metrics.map_75_mask == 1.0
    assert metrics.map_50_95_mask == 1.0

@pytest.mark.parametrize("data_path", find_test_datasets())
def test_validator_real_gt_jittered(data_path: Path):
    coco = load_coco_json(data_path)
    gt_results = coco_to_image_results(coco)
    pred_results = jitter_results(gt_results, jitter=3)
    
    label_to_name = {cat.id: cat.name for cat in coco.categories}
    validator = Validator(gt_results, pred_results, label_to_name=label_to_name)
    metrics = validator.compute_metrics()
    
    # Results should be high but not perfect
    assert metrics.iou < 1.0
    assert metrics.tps == sum(len(r.labels) for r in gt_results)
    assert metrics.fps == 0
    assert metrics.fns == 0

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

@given(st.lists(instance_mask_strategy(), min_size=1, max_size=5))
def test_mask_iou_fn_self(masks):
    img_res = ImageResult(
        labels=torch.tensor([m.label for m in masks], dtype=torch.int64),
        boxes=torch.stack([torch.tensor([m.offset[0], m.offset[1], m.offset[0] + m.mask.shape[1], m.offset[1] + m.mask.shape[0]]) for m in masks]).float(),
        img_size=(200, 200),
        scores=torch.ones(len(masks), dtype=torch.float32),
        masks=masks
    )
    ious = mask_iou_fn(img_res, img_res)
    assert ious.shape == (len(masks), len(masks))
    for i in range(len(masks)):
        assert ious[i, i] == pytest.approx(1.0)

def test_greedy_match_simple():
    ious = torch.tensor([
        [0.9, 0.1],
        [0.2, 0.8]
    ])
    res = greedy_match(ious, 0.5)
    assert len(res.matches) == 2
    assert res.matches[0].pred_idx == 0 and res.matches[0].gt_idx == 0
    assert res.matches[1].pred_idx == 1 and res.matches[1].gt_idx == 1

def test_greedy_match_conflict():
    ious = torch.tensor([
        [0.9, 0.85],
        [0.1, 0.2]
    ])
    # Greedy should pick (0,0) first, then (1,1) is only 0.2 < 0.5
    res = greedy_match(ious, 0.5)
    assert len(res.matches) == 1
    assert res.matches[0].pred_idx == 0 and res.matches[0].gt_idx == 0
    assert res.unmatched_preds == [1]
    assert res.unmatched_gt == [1]

def test_validator_perfect_match():
    m = make_random_instance_mask(label=1, offset=(10, 10))
    gt = [ImageResult(
        labels=torch.tensor([1], dtype=torch.int64),
        boxes=torch.tensor([[10, 10, 20, 20]], dtype=torch.float32),
        img_size=(100, 100),
        scores=torch.tensor([1.0], dtype=torch.float32),
        masks=[m]
    )]
    pred = [ImageResult(
        labels=torch.tensor([1], dtype=torch.int64),
        boxes=torch.tensor([[10, 10, 20, 20]], dtype=torch.float32),
        img_size=(100, 100),
        scores=torch.tensor([1.0], dtype=torch.float32),
        masks=[m]
    )]
    
    validator = Validator(gt, pred, label_to_name={1: "test"})
    metrics = validator.compute_metrics()
    
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    assert metrics.iou == 1.0

def test_validator_no_match():
    m1 = make_random_instance_mask(label=1, offset=(10, 10))
    m2 = make_random_instance_mask(label=1, offset=(50, 50))
    gt = [ImageResult(
        labels=torch.tensor([1], dtype=torch.int64),
        boxes=torch.tensor([[10, 10, 20, 20]], dtype=torch.float32),
        img_size=(100, 100),
        scores=torch.tensor([1.0], dtype=torch.float32),
        masks=[m1]
    )]
    pred = [ImageResult(
        labels=torch.tensor([1], dtype=torch.int64),
        boxes=torch.tensor([[50, 50, 60, 60]], dtype=torch.float32),
        img_size=(100, 100),
        scores=torch.tensor([1.0], dtype=torch.float32),
        masks=[m2]
    )]
    
    validator = Validator(gt, pred, label_to_name={1: "test"}, iou_thresh=0.5)
    metrics = validator.compute_metrics()
    
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0
    assert metrics.tps == 0
    assert metrics.fps == 1
    assert metrics.fns == 1

@given(st.integers(min_value=1, max_value=5))
def test_validator_random_count(count):
    # Just ensure it doesn't crash with random number of instances
    masks_gt = [make_random_instance_mask(label=1) for _ in range(count)]
    masks_pred = [make_random_instance_mask(label=1) for _ in range(count)]
    
    gt = [ImageResult(
        labels=torch.ones(count, dtype=torch.int64),
        boxes=torch.randn(count, 4),
        img_size=(100, 100),
        scores=torch.ones(count, dtype=torch.float32),
        masks=masks_gt
    )]
    pred = [ImageResult(
        labels=torch.ones(count, dtype=torch.int64),
        boxes=torch.randn(count, 4),
        img_size=(100, 100),
        scores=torch.rand(count),
        masks=masks_pred
    )]
    
    validator = Validator(gt, pred, label_to_name={1: "test"})
    metrics = validator.compute_metrics()
    assert metrics is not None

