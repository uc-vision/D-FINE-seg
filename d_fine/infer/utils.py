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
  """Resize and pad image while meeting stride-multiple constraints.
  new_shape: (width, height)
  """
  shape = im.shape[:2]  # current shape [height, width]
  tw, th = new_shape

  # Scale ratio (new / old)
  r = min(th / shape[0], tw / shape[1])
  if not scaleup:  # only scale down, do not scale up (for better val mAP)
    r = min(r, 1.0)

  # Compute padding
  ratio = r, r  # width, height ratios
  new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
  dw, dh = tw - new_unpad[0], th - new_unpad[1]  # wh padding
  if auto:  # minimum rectangle
    dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
  elif scale_fill:  # stretch
    dw, dh = 0.0, 0.0
    new_unpad = (tw, th)
    ratio = tw / shape[1], th / shape[0]  # width, height ratios

  dw /= 2  # divide padding into 2 sides
  dh /= 2

  if shape[::-1] != new_unpad:  # resize
    im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_AREA)
  top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
  left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
  im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
  return im, ratio, (dw, dh)


def compute_nearest_size(
  shape: tuple[int, int], target_size: int, stride: int = 32
) -> tuple[int, int]:
  """Get nearest size that is divisible by stride.
  shape: (height, width)
  returns: (width, height)
  """
  scale = target_size / max(shape)
  h, w = shape
  th, tw = int(round(h * scale)), int(round(w * scale))
  w_final = max(stride, int(np.ceil(tw / stride) * stride))
  h_final = max(stride, int(np.ceil(th / stride) * stride))
  return w_final, h_final


def preprocess(
  img: np.ndarray,
  target_size: tuple[int, int],
  keep_aspect: bool,
  rect: bool,
  stride: int = 32,
  dtype: np.dtype = np.float32,
) -> tuple[torch.Tensor, tuple[int, int]]:
  """Uniform preprocessing for all model types.
  target_size: (width, height)
  returns: (tensor, orig_size=(width, height))
  """
  orig_size = (img.shape[1], img.shape[0])

  if not keep_aspect:
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
  elif rect:
    w_t, h_t = compute_nearest_size(img.shape[:2], max(target_size))
    img = letterbox(img, (w_t, h_t), stride=stride, auto=False)[0]
  else:
    img = letterbox(img, target_size, stride=stride, auto=False)[0]

  img = img.transpose(2, 0, 1)  # HWC to CHW
  img = np.ascontiguousarray(img, dtype=dtype)
  img = torch.from_numpy(img) / 255.0
  return img, orig_size


def scale_boxes(
  boxes: np.ndarray, orig_size: tuple[float, float], proc_size: tuple[float, float]
) -> np.ndarray:
  """Scale boxes from proc_size to orig_size without ratio keeping.
  orig_size: (width, height)
  proc_size: (width, height)
  """
  w0, h0 = orig_size
  wp, hp = proc_size
  new_boxes = boxes.copy()
  new_boxes[:, [0, 2]] *= w0 / wp
  new_boxes[:, [1, 3]] *= h0 / hp
  return new_boxes


def scale_boxes_ratio_kept(
  boxes: np.ndarray, proc_size: tuple[float, float], orig_size: tuple[float, float]
) -> np.ndarray:
  """Scale boxes from proc_size to orig_size while keeping aspect ratio (reversing padding).
  proc_size: (width, height)
  orig_size: (width, height)
  """
  wp, hp = proc_size
  w0, h0 = orig_size
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
  processed_size: (width, height)
  orig_sizes: [B, 2] (width, height)
  """
  B = boxes.shape[0]
  wp, hp = processed_size
  new_boxes = torch.zeros_like(boxes)

  # norm xywh -> abs xyxy in processed size
  new_boxes[:, :, 0] = (boxes[:, :, 0] - boxes[:, :, 2] / 2) * wp
  new_boxes[:, :, 1] = (boxes[:, :, 1] - boxes[:, :, 3] / 2) * hp
  new_boxes[:, :, 2] = (boxes[:, :, 0] + boxes[:, :, 2] / 2) * wp
  new_boxes[:, :, 3] = (boxes[:, :, 1] + boxes[:, :, 3] / 2) * hp

  new_boxes_np = new_boxes.cpu().numpy()
  orig_sizes_np = orig_sizes.cpu().numpy()
  proc_size_np = (float(wp), float(hp))

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
  processed_size: (width, height)
  orig_sizes: [B, 2] (width, height)
  """
  B, K, Hm, Wm = masks.shape
  Wp, Hp = processed_size

  results = []
  for i in range(B):
    w0, h0 = int(orig_sizes[i, 0]), int(orig_sizes[i, 1])
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


def postprocess_predictions(
  outputs: dict[str, torch.Tensor],
  orig_sizes: torch.Tensor,
  config: EvaluationConfig,
  processed_size: tuple[int, int],
) -> list[ImageResult]:
  """Postprocess model predictions.
  processed_size: (width, height)
  orig_sizes: [B, 2] (width, height)
  """
  logits, boxes = outputs["pred_logits"], outputs["pred_boxes"]
  has_masks = ("pred_masks" in outputs) and (outputs["pred_masks"] is not None)
  pred_masks = outputs["pred_masks"] if has_masks else None
  B, Q = logits.shape[:2]

  wp, hp = processed_size
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

    img_size = tuple(orig_sizes[b].int().tolist()) # (width, height)
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
