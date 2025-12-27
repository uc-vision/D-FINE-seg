from __future__ import annotations

import math
import random
import time
from pathlib import Path
from collections.abc import Callable, Iterator
import datetime

import cv2
import numpy as np
import torch
from torch import Tensor

from image_detection.annotation.coco import InstanceMask
from .core.dist_utils import is_main_process, get_rank, get_world_size, reduce_dict
from .core.logging import SmoothedValue, MetricLogger

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

def box_candidates(box1, box2, wh_thr=2, ar_thr=100, area_thr=0.1, eps=1e-16):
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))
    return (
        (w2 > wh_thr)
        & (h2 > wh_thr)
        & (w2 * h2 / (w1 * h1 + eps) > area_thr)
        & (ar < ar_thr)
    )

def abs_xyxy_to_norm_xywh(box, w, h):
    if len(box) == 0:
        return box
    new_box = box.copy()
    new_box[:, 0] = (box[:, 0] + box[:, 2]) / (2 * w)
    new_box[:, 1] = (box[:, 1] + box[:, 3]) / (2 * h)
    new_box[:, 2] = (box[:, 2] - box[:, 0]) / w
    new_box[:, 3] = (box[:, 3] - box[:, 1]) / h
    return new_box

def norm_xywh_to_abs_xyxy(box, w, h):
    if len(box) == 0:
        return box
    new_box = box.copy()
    new_box[:, 0] = (box[:, 0] - box[:, 2] / 2) * w
    new_box[:, 1] = (box[:, 1] - box[:, 3] / 2) * h
    new_box[:, 2] = (box[:, 0] + box[:, 2] / 2) * w
    new_box[:, 3] = (box[:, 1] + box[:, 3] / 2) * h
    return new_box

def get_transform_matrix(img_shape, target_shape, variant, jitter=0.3, jitter_type="uniform"):
    if variant == "resize":
        return np.array([[target_shape[1] / img_shape[1], 0, 0], [0, target_shape[0] / img_shape[0], 0], [0, 0, 1]])
    elif variant == "scale_jitter":
        scale = random.uniform(1 - jitter, 1 + jitter) if jitter_type == "uniform" else np.random.normal(1, jitter)
        return np.array([[scale * target_shape[1] / img_shape[1], 0, 0], [0, scale * target_shape[0] / img_shape[0], 0], [0, 0, 1]])
    return np.eye(3)

def random_affine(img, targets=(), masks=None, degrees=10, translate=0.1, scale=0.1, shear=10, border=(0, 0)):
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
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))
            masks = cv2.warpAffine(masks, M[:2], dsize=(width, height), flags=cv2.INTER_NEAREST, borderValue=0)
        else:
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))
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

def get_mosaic_coordinate(mosaic_index, xc, yc, w, h, target_h, target_w):
    if mosaic_index == 0:
        x1, y1, x2, y2 = max(xc - w, 0), max(yc - h, 0), xc, yc
        small_coord = w - (x2 - x1), h - (y2 - y1), w, h
    elif mosaic_index == 1:
        x1, y1, x2, y2 = xc, max(yc - h, 0), min(xc + w, target_w * 2), yc
        small_coord = 0, h - (y2 - y1), min(w, x2 - x1), h
    elif mosaic_index == 2:
        x1, y1, x2, y2 = max(xc - w, 0), yc, xc, min(target_h * 2, yc + h)
        small_coord = w - (x2 - x1), 0, w, min(y2 - y1, h)
    elif mosaic_index == 3:
        x1, y1, x2, y2 = xc, yc, min(xc + w, target_w * 2), min(target_h * 2, yc + h)
        small_coord = 0, 0, min(w, x2 - x1), min(y2 - y1, h)
    return (x1, y1, x2, y2), small_coord

class LetterboxRect:
    def __init__(self, target_h: int, target_w: int):
        self.target_h = target_h
        self.target_w = target_w

    def __call__(self, image: np.ndarray, bboxes: np.ndarray = None, labels: np.ndarray = None, mask: np.ndarray = None) -> dict:
        h, w = image.shape[:2]
        r = min(self.target_h / h, self.target_w / w)
        nh, nw = int(h * r), int(w * r)
        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        top = (self.target_h - nh) // 2
        bottom = self.target_h - nh - top
        left = (self.target_w - nw) // 2
        right = self.target_w - nw - left
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
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

def draw_mask(image, mask, color=(255, 0, 0), alpha=0.5):
    """Draw a mask on an image. Default color is Red (RGB)."""
    mask = mask.astype(bool)
    image[mask] = image[mask] * (1 - alpha) + np.array(color) * alpha
    return image.astype(np.uint8)

def vis_one_box(img, box, label, score: float | None = None, label_to_name: dict[int, str] | None = None, color=None, mode="gt"):
    """Visualize a single bounding box. Default colors for RGB: GT=Green, Pred=Red."""
    if color is None:
        color = (0, 255, 0) if mode == "gt" else (255, 0, 0)
    
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    name = label_to_name[label] if label_to_name else str(label)
    txt = f"{name} {score:.2f}" if score is not None else name
    cv2.putText(img, txt, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def setup_multi_processes():
    import os
    if os.name != "nt":
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_config(config_path):
    import yaml
    from omegaconf import OmegaConf
    return OmegaConf.load(config_path)

def get_vram_usage():
    if torch.cuda.is_available():
        t = torch.cuda.get_device_properties(0).total_memory
        a = torch.cuda.memory_allocated(0)
        return round((a / t) * 100, 2)
    return 0.0

def calculate_remaining_time(one_epoch_time, epoch_start_time, epoch, num_epochs, cur_iter, num_batches):
    current_epoch_elapsed = time.time() - epoch_start_time
    if cur_iter > 0:
        estimated_total_epoch = (current_epoch_elapsed / cur_iter) * num_batches
    else:
        estimated_total_epoch = one_epoch_time if one_epoch_time is not None else 0
    remaining_epochs = num_epochs - epoch
    remaining_current = estimated_total_epoch - current_epoch_elapsed
    if one_epoch_time is not None:
        remaining_future = remaining_epochs * one_epoch_time
    else:
        remaining_future = remaining_epochs * estimated_total_epoch
    total_remaining = max(0, remaining_current + remaining_future)
    return str(datetime.timedelta(seconds=int(total_remaining)))

def visualize(img_paths, targets, results, label_to_name, **kwargs):
    vis_images = []
    for i in range(len(img_paths)):
        img_path, img = img_paths[i], None
        if isinstance(img_path, (str, Path)):
            p = Path(img_path)
            if not p.is_absolute() and kwargs.get("dataset_path"):
                p = Path(kwargs["dataset_path"]) / p
            img = load_image(p)
        elif torch.is_tensor(img_path):
            img = img_path.permute(1, 2, 0).cpu().numpy()
            img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
            img = img.astype(np.uint8).copy()
        if img is None: continue
        
        for box, label in zip(targets[i]["boxes"], targets[i]["labels"]):
            vis_one_box(img, box.numpy(), label.item(), None, label_to_name, mode="gt")
        res = results[i]
        if "boxes" in res:
            for box, label, score in zip(res["boxes"], res["labels"], res["scores"]):
                vis_one_box(img, box.numpy(), label.item(), score.item(), label_to_name, mode="pred")
        vis_images.append(img)
    return vis_images

def save_metrics(all_metrics, val_metrics, loss, epoch, path_to_save, use_wandb=False):
    if use_wandb:
        from .validation import wandb_logger
        wandb_logger(None, val_metrics, epoch=epoch, mode="val")
        import wandb
        wandb.log({"train/loss": loss, "epoch": epoch})
