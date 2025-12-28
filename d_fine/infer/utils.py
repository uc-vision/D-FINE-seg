import cv2
import numpy as np
import torch
import torch.nn.functional as F

from d_fine.core.types import ImageResult
from d_fine.config import EvaluationConfig
from lib_detection.annotation.coco import InstanceMask


def letterbox(
  im: np.ndarray,
  new_shape: tuple[int, int] = (640, 640),
  color: tuple[int, int, int] = (114, 114, 114),
  auto: bool = True,
  scale_fill: bool = False,
  scaleup: bool = True,
  stride: int = 32,
) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
  """Resize and pad image while meeting stride-multiple constraints."""
  shape = im.shape[:2]  # current shape [height, width]
  if isinstance(new_shape, int):
    new_shape = (new_shape, new_shape)

  # Scale ratio (new / old)
  r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
  if not scaleup:  # only scale down, do not scale up (for better val mAP)
    r = min(r, 1.0)

  # Compute padding
  ratio = r, r  # width, height ratios
  new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
  dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
  if auto:  # minimum rectangle
    dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
  elif scale_fill:  # stretch
    dw, dh = 0.0, 0.0
    new_unpad = (new_shape[1], new_shape[0])
    ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

  dw /= 2  # divide padding into 2 sides
  dh /= 2

  if shape[::-1] != new_unpad:  # resize
    im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_AREA)
  top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
  left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
  im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
  return im, ratio, (dw, dh)


def compute_nearest_size(shape: tuple[int, int], target_size: int, stride: int = 32) -> list[int]:
  """Get nearest size that is divisible by stride."""
  scale = target_size / max(shape)
  new_shape = [int(round(dim * scale)) for dim in shape]
  return [max(stride, int(np.ceil(dim / stride) * stride)) for dim in new_shape]


def preprocess(
  img: np.ndarray,
  target_size: tuple[int, int],
  keep_aspect: bool,
  rect: bool,
  stride: int = 32,
  dtype: np.dtype = np.float32,
) -> tuple[torch.Tensor, tuple[int, int]]:
  """Uniform preprocessing for all model types."""
  orig_size = (img.shape[0], img.shape[1])

  if not keep_aspect:
    img = cv2.resize(img, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
  elif rect:
    h_t, w_t = compute_nearest_size(img.shape[:2], max(target_size))
    img = letterbox(img, (h_t, w_t), stride=stride, auto=False)[0]
  else:
    img = letterbox(img, target_size, stride=stride, auto=False)[0]

  img = img.transpose(2, 0, 1)  # HWC to CHW
  img = np.ascontiguousarray(img, dtype=dtype)
  img = torch.from_numpy(img) / 255.0
  return img, orig_size


def scale_boxes(
  boxes: np.ndarray, orig_size: tuple[float, float], proc_size: tuple[float, float]
) -> np.ndarray:
  """Scale boxes from proc_size to orig_size without ratio keeping."""
  h0, w0 = orig_size
  hp, wp = proc_size
  new_boxes = boxes.copy()
  new_boxes[:, [0, 2]] *= w0 / wp
  new_boxes[:, [1, 3]] *= h0 / hp
  return new_boxes


def scale_boxes_ratio_kept(
  boxes: np.ndarray, proc_size: tuple[float, float], orig_size: tuple[float, float]
) -> np.ndarray:
  """Scale boxes from proc_size to orig_size while keeping aspect ratio (reversing padding)."""
  hp, wp = proc_size
  h0, w0 = orig_size
  gain = min(hp / h0, wp / w0)
  padw = (wp - w0 * gain) / 2
  padh = (hp - h0 * gain) / 2

  new_boxes = boxes.copy()
  new_boxes[:, [0, 2]] = (boxes[:, [0, 2]] - padw) / gain
  new_boxes[:, [1, 3]] = (boxes[:, [1, 3]] - padh) / gain
  return new_boxes


def process_boxes(
  boxes: torch.Tensor,
  processed_size: tuple[int, int],
  orig_sizes: torch.Tensor,
  keep_aspect: bool,
  device: torch.device,
) -> torch.Tensor:
  """
  Convert normalized boxes to absolute coordinates based on image sizes.
  boxes: [B, Q, 4] in norm xywh
  processed_size: (H_proc, W_proc)
  orig_sizes: [B, 2] (H_orig, W_orig)
  """
  B = boxes.shape[0]
  hp, wp = processed_size
  new_boxes = torch.zeros_like(boxes)

  # norm xywh -> abs xyxy in processed size
  new_boxes[:, :, 0] = (boxes[:, :, 0] - boxes[:, :, 2] / 2) * wp
  new_boxes[:, :, 1] = (boxes[:, :, 1] - boxes[:, :, 3] / 2) * hp
  new_boxes[:, :, 2] = (boxes[:, :, 0] + boxes[:, :, 2] / 2) * wp
  new_boxes[:, :, 3] = (boxes[:, :, 1] + boxes[:, :, 3] / 2) * hp

  new_boxes_np = new_boxes.cpu().numpy()
  orig_sizes_np = orig_sizes.cpu().numpy()
  proc_size_np = (float(hp), float(wp))

  for i in range(B):
    if keep_aspect:
      new_boxes_np[i] = scale_boxes_ratio_kept(new_boxes_np[i], proc_size_np, orig_sizes_np[i])
    else:
      new_boxes_np[i] = scale_boxes(new_boxes_np[i], orig_sizes_np[i], proc_size_np)

  return torch.from_numpy(new_boxes_np).to(device)


def process_masks(
  masks: torch.Tensor,
  bboxes: torch.Tensor,
  labels: torch.Tensor,
  scores: torch.Tensor,
  processed_size: tuple[int, int],
  orig_sizes: torch.Tensor,
  keep_aspect: bool = True,
  conf_thresh: float = 0.0,
) -> list[list[InstanceMask]]:
  """
  Efficiently process masks by only interpolating within bounding boxes.
  Returns a list of lists of InstanceMask objects.
  """
  B, K, Hm, Wm = masks.shape
  Hp, Wp = processed_size

  results = []
  for i in range(B):
    h0, w0 = int(orig_sizes[i, 0]), int(orig_sizes[i, 1])
    img_instances = []

    gain = min(Hp / h0, Wp / w0) if keep_aspect else 1.0
    padw = (Wp - w0 * gain) / 2 if keep_aspect else 0.0
    padh = (Hp - h0 * gain) / 2 if keep_aspect else 0.0

    for j in range(K):
      box = bboxes[i, j]
      label = int(labels[i, j].item())
      score = scores[i, j].item()

      x1, y1, x2, y2 = box.tolist()

      # Map box to Hm, Wm for cropping
      mx1 = int(max(0, min(Wp, x1)) * (Wm / Wp))
      my1 = int(max(0, min(Hp, y1)) * (Hm / Hp))
      mx2 = int(max(mx1 + 1, min(Wp, x2)) * (Wm / Wp))
      my2 = int(max(my1 + 1, min(Hp, y2)) * (Hm / Hp))

      # Map box to original size
      if keep_aspect:
        ox1, oy1 = (x1 - padw) / gain, (y1 - padh) / gain
        ox2, oy2 = (x2 - padw) / gain, (y2 - padh) / gain
      else:
        ox1, oy1 = x1 * (w0 / Wp), y1 * (h0 / Hp)
        ox2, oy2 = x2 * (w0 / Wp), y2 * (h0 / Hp)

      ox1, oy1 = int(max(0, min(w0, ox1))), int(max(0, min(h0, oy1)))
      ox2, oy2 = int(max(ox1 + 1, min(w0, ox2))), int(max(oy1 + 1, min(h0, oy2)))

      m_crop = masks[i, j, my1:my2, mx1:mx2]
      if m_crop.numel() == 0:
        continue

      m_final = F.interpolate(
        m_crop[None, None], size=(oy2 - oy1, ox2 - ox1), mode="bilinear", align_corners=False
      )[0, 0]
      m_bool = m_final >= conf_thresh

      img_instances.append(
        InstanceMask(mask=m_bool.cpu(), label=label, offset=(ox1, oy1), score=score)
      )

    results.append(img_instances)
  return results


def postprocess_ground_truth(
  inputs: torch.Tensor,
  targets: list[dict[str, torch.Tensor]],
  orig_sizes: torch.Tensor,
  keep_aspect: bool,
) -> list[ImageResult]:
  """Postprocess ground truth targets."""
  results = []
  processed_size = tuple(inputs.shape[2:])

  for idx, target in enumerate(targets):
    lab = target["labels"]
    box = process_boxes(
      target["boxes"][None], processed_size, orig_sizes[idx][None], keep_aspect, inputs.device
    )

    img_size = tuple(orig_sizes[idx].int().tolist())
    masks = []

    if "masks" in target and target["masks"] is not None and target["masks"].numel() > 0:
      gt_m = target["masks"].float()
      Hp, Wp = processed_size
      pb = target["boxes"].clone()
      pb[:, 0] = (target["boxes"][:, 0] - target["boxes"][:, 2] / 2) * Wp
      pb[:, 1] = (target["boxes"][:, 1] - target["boxes"][:, 3] / 2) * Hp
      pb[:, 2] = (target["boxes"][:, 0] + target["boxes"][:, 2] / 2) * Wp
      pb[:, 3] = (target["boxes"][:, 1] + target["boxes"][:, 3] / 2) * Hp

      instances = process_masks(
        gt_m.unsqueeze(0),
        pb.unsqueeze(0),
        lab.unsqueeze(0),
        torch.ones_like(lab.unsqueeze(0), dtype=torch.float32),
        processed_size=processed_size,
        orig_sizes=orig_sizes[idx].unsqueeze(0),
        keep_aspect=keep_aspect,
        conf_thresh=0.5,
      )
      masks = instances[0]

    results.append(
      ImageResult(
        labels=lab.detach().cpu(),
        boxes=box.squeeze(0).detach().cpu(),
        img_size=img_size,
        scores=torch.ones_like(lab, dtype=torch.float32),
        masks=masks,
      )
    )
  return results


def postprocess_predictions(
  outputs: dict[str, torch.Tensor],
  orig_sizes: torch.Tensor,
  config: EvaluationConfig,
  processed_size: tuple[int, int],
) -> list[ImageResult]:
  """Postprocess model predictions."""
  logits, boxes = outputs["pred_logits"], outputs["pred_boxes"]
  has_masks = ("pred_masks" in outputs) and (outputs["pred_masks"] is not None)
  pred_masks = outputs["pred_masks"] if has_masks else None
  B, Q = logits.shape[:2]

  hp, wp = processed_size
  pb = torch.zeros_like(boxes)
  pb[:, :, 0] = (boxes[:, :, 0] - boxes[:, :, 2] / 2) * wp
  pb[:, :, 1] = (boxes[:, :, 1] - boxes[:, :, 3] / 2) * hp
  pb[:, :, 2] = (boxes[:, :, 0] + boxes[:, :, 2] / 2) * wp
  pb[:, :, 3] = (boxes[:, :, 1] + boxes[:, :, 3] / 2) * hp

  abs_boxes = process_boxes(boxes, processed_size, orig_sizes, config.keep_aspect, logits.device)

  if config.use_focal_loss:
    scores_all = torch.sigmoid(logits)
    flat = scores_all.flatten(1)
    K = min(config.num_top_queries, flat.shape[1])
    topk_scores, topk_idx = torch.topk(flat, K, dim=-1)
    topk_labels = topk_idx % config.num_classes
    topk_qidx = topk_idx // config.num_classes
  else:
    probs = torch.softmax(logits, dim=-1)[:, :, :-1]
    topk_scores, topk_labels = probs.max(dim=-1)
    K = min(config.num_top_queries, Q)
    topk_scores, order = torch.topk(topk_scores, K, dim=-1)
    topk_labels = topk_labels.gather(1, order)
    topk_qidx = order

  results = []
  for b in range(B):
    sb = topk_scores[b]
    lb = topk_labels[b]
    qb = topk_qidx[b]

    keep = sb >= config.conf_thresh
    sb, lb, qb = sb[keep], lb[keep], qb[keep]
    bb = abs_boxes[b].gather(0, qb.unsqueeze(-1).repeat(1, 4))

    img_size = tuple(orig_sizes[b].int().tolist())
    masks = []

    if has_masks and qb.numel() > 0:
      mb = pred_masks[b, qb].unsqueeze(0)
      qb_pb = pb[b, qb].unsqueeze(0)
      qb_lb = lb.unsqueeze(0)
      qb_sb = sb.unsqueeze(0)

      instances = process_masks(
        mb,
        qb_pb,
        qb_lb,
        qb_sb,
        processed_size=processed_size,
        orig_sizes=orig_sizes[b].unsqueeze(0),
        keep_aspect=config.keep_aspect,
        conf_thresh=0.0,
      )
      masks = instances[0]

    results.append(
      ImageResult(
        labels=lb.detach().cpu(),
        boxes=bb.detach().cpu(),
        scores=sb.detach().cpu(),
        img_size=img_size,
        masks=masks,
      )
    )
  return results


def cleanup_masks(masks: np.ndarray) -> np.ndarray:
  """Simple NumPy-only mask cleanup to avoid redundant implementations."""
  if masks.ndim == 2:
    return (masks > 0.5).astype(np.uint8)
  return (masks > 0.5).astype(np.uint8)


def to_instance_masks(raw_res_dict: dict, conf_thresh: float | None = None) -> list[InstanceMask]:
  """Convert a detection result dictionary to a list of InstanceMask objects.

  Args:
      raw_res_dict: Dictionary with keys "boxes", "labels", "scores", and "masks"
      conf_thresh: Optional confidence threshold to filter results

  Returns:
      List of InstanceMask objects
  """
  labels = torch.from_numpy(raw_res_dict["labels"])
  boxes = torch.from_numpy(raw_res_dict["boxes"])
  scores = torch.from_numpy(raw_res_dict["scores"])
  masks = raw_res_dict.get("masks", [])
  img_size = raw_res_dict.get("img_size", (0, 0))

  res = ImageResult(labels=labels, boxes=boxes, img_size=img_size, scores=scores, masks=masks)

  if conf_thresh is not None:
    res = res.filter(conf_thresh)

  return res.masks
