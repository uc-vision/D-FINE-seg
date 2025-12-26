import cv2
import numpy as np
import torch

from d_fine.config import ModelConfig
from d_fine.utils import cleanup_masks, norm_xywh_to_abs_xyxy, process_boxes as default_process_boxes, process_masks, scale_boxes, scale_boxes_ratio_kept


def process_boxes(boxes, processed_sizes, orig_sizes, keep_ratio, device):
    boxes = boxes.cpu().numpy()
    final_boxes = np.zeros_like(boxes)
    for idx, box in enumerate(boxes):
        final_boxes[idx] = norm_xywh_to_abs_xyxy(
            box, processed_sizes[idx][0], processed_sizes[idx][1]
        )

    for i in range(len(orig_sizes)):
        if keep_ratio:
            final_boxes[i] = scale_boxes_ratio_kept(
                final_boxes[i], processed_sizes[i], orig_sizes[i]
            )
        else:
            final_boxes[i] = scale_boxes(final_boxes[i], orig_sizes[i], processed_sizes[i])
    return torch.tensor(final_boxes).to(device)


def process_masks(
    pred_masks,  # Tensor [B, Q, Hm, Wm] or [Q, Hm, Wm]
    processed_size,  # (H, W) of network input (after your A.Compose)
    orig_sizes,  # Tensor [B, 2] (H, W)
    keep_ratio: bool,
) -> list[torch.Tensor]:
    """
    Returns list of length B with masks resized to original image sizes:
    Each item: Float Tensor [Q, H_orig, W_orig] in [0,1] (no thresholding here).
    - Handles letterbox padding removal if keep_ratio=True.
    - Works for both batched and single-image inputs.
    """
    single = pred_masks.dim() == 3  # [Q,Hm,Wm]
    if single:
        pred_masks = pred_masks.unsqueeze(0)  # -> [1,Q,Hm,Wm]

    B, Q, Hm, Wm = pred_masks.shape
    device = pred_masks.device
    dtype = pred_masks.dtype

    # 1) Upsample masks to processed (input) size
    proc_h, proc_w = int(processed_size[0]), int(processed_size[1])
    masks_proc = torch.nn.functional.interpolate(
        pred_masks, size=(proc_h, proc_w), mode="bilinear", align_corners=False
    )  # [B,Q,Hp,Wp] with Hp=proc_h, Wp=proc_w

    out = []
    for b in range(B):
        H0, W0 = int(orig_sizes[b, 0].item()), int(orig_sizes[b, 1].item())
        m = masks_proc[b]  # [Q, Hp, Wp]
        if keep_ratio:
            # Compute same gain/pad as in scale_boxes_ratio_kept
            gain = min(proc_h / H0, proc_w / W0)
            padw = round((proc_w - W0 * gain) / 2 - 0.1)
            padh = round((proc_h - H0 * gain) / 2 - 0.1)

            # Remove padding before final resize
            y1 = max(padh, 0)
            y2 = proc_h - max(padh, 0)
            x1 = max(padw, 0)
            x2 = proc_w - max(padw, 0)
            m = m[:, y1:y2, x1:x2]  # [Q, cropped_h, cropped_w]

        # 2) Resize to original size
        m = torch.nn.functional.interpolate(
            m.unsqueeze(0), size=(H0, W0), mode="bilinear", align_corners=False
        ).squeeze(0)  # [Q, H0, W0]
        out.append(m.clamp_(0, 1).to(device=device, dtype=dtype))

    if single:
        return [out[0]]
    return out


def letterbox(
    im,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=True,
    scale_fill=False,
    scaleup=True,
    stride=32,
):
    # Resize and pad image while meeting stride-multiple constraints
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
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return im, ratio, (dw, dh)


def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


def postprocess_ground_truth(
    inputs: torch.Tensor,
    targets: list[dict[str, torch.Tensor]],
    orig_sizes: torch.Tensor,
    keep_ratio: bool,
) -> list[dict[str, torch.Tensor]]:
    """
    Postprocess ground truth targets, mapping boxes and masks from processed size to original size.
    
    Args:
        inputs: Input tensor [B, C, H, W]
        targets: List of target dicts, one per image
        orig_sizes: Original image sizes tensor [B, 2] (H, W)
        keep_ratio: Whether aspect ratio was kept during preprocessing
    
    Returns:
        List of dicts, one per image, with keys: labels, boxes, masks
    """
    from d_fine.utils import process_boxes, process_masks
    
    results = []
    for idx, target in enumerate(targets):
        lab = target["labels"]
        box = process_boxes(
            target["boxes"][None],
            inputs[idx].shape[1:],
            orig_sizes[idx][None],
            keep_ratio,
            inputs.device,
        )
        result = dict(labels=lab.detach().cpu(), boxes=box.squeeze(0).detach().cpu())

        if (
            "masks" in targets[idx]
            and targets[idx]["masks"] is not None
            and targets[idx]["masks"].numel() > 0
        ):
            gt_m = targets[idx]["masks"].to(
                dtype=inputs.dtype, device=inputs.device
            )
            gt_m = gt_m.unsqueeze(0)
            masks_list = process_masks(
                gt_m,
                processed_size=inputs[idx].shape[1:],
                orig_sizes=orig_sizes[idx].unsqueeze(0),
                keep_ratio=keep_ratio,
            )
            result["masks"] = (masks_list[0].clamp(0, 1) >= 0.5).to(torch.uint8).detach().cpu()
        else:
            result["masks"] = torch.zeros(
                (0, int(orig_sizes[idx, 0].item()), int(orig_sizes[idx, 1].item())),
                dtype=torch.uint8,
            )

        results.append(result)
    return results


def mask2poly(masks: np.ndarray, img_shape: tuple[int, int]) -> list[np.ndarray]:
    """
    Convert binary masks to normalized polygon coordinates for YOLO segmentation format.

    Args:
        masks: Binary masks array of shape [N, H, W] where N is number of instances
        img_shape: Tuple of (height, width) of the original image

    Returns:
        List of normalized polygon coordinates, each as array of shape [num_points, 2]
        with values in range [0, 1]. Returns empty array for invalid masks.
    """
    h, w = img_shape[:2]
    polys = []

    for mask in masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Get the largest contour
            contour = max(contours, key=cv2.contourArea)
            contour = contour.reshape(-1, 2)
            if len(contour) >= 3:  # Need at least 3 points for a valid polygon
                # Normalize coordinates
                norm_contour = contour.astype(np.float32)
                norm_contour[:, 0] /= w
                norm_contour[:, 1] /= h
                polys.append(norm_contour)
            else:
                polys.append(np.array([]))
        else:
            polys.append(np.array([]))

    return polys


def postprocess_predictions(
    outputs: dict[str, torch.Tensor],
    orig_sizes: torch.Tensor,
    config: ModelConfig,
    processed_size: tuple[int, int],
    include_all_for_map: bool = False,
) -> list[dict[str, torch.Tensor]]:
    """
    Consolidated postprocessing function for model predictions.
    
    Args:
        outputs: Dict with "pred_logits", "pred_boxes", optionally "pred_masks"
        orig_sizes: Original image sizes as tensor [B, 2]
        config: ModelConfig with postprocessing parameters
        processed_size: Processed image size as (H, W) tuple
        include_all_for_map: If True, include all_boxes/all_scores/all_labels for mAP
    
    Returns:
        List of dicts, one per image, with keys: labels, boxes, scores, optionally masks/mask_probs (all torch tensors)
    """
    logits, boxes = outputs["pred_logits"], outputs["pred_boxes"]
    has_masks = ("pred_masks" in outputs) and (outputs["pred_masks"] is not None)
    pred_masks = outputs["pred_masks"] if has_masks else None
    B, Q = logits.shape[:2]
    
    processed_size_tensor = torch.tensor(processed_size, device=logits.device)
    boxes = default_process_boxes(boxes, processed_size_tensor, orig_sizes, config.keep_ratio, logits.device)
    
    if config.use_focal_loss:
        scores_all = torch.sigmoid(logits)
        flat = scores_all.flatten(1)
        K = min(config.num_top_queries, flat.shape[1])
        topk_scores, topk_idx = torch.topk(flat, K, dim=-1)
        topk_labels = topk_idx - (topk_idx // config.n_outputs) * config.n_outputs
        topk_qidx = topk_idx // config.n_outputs
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
        
        sb = sb[keep]
        lb = lb[keep]
        qb = qb[keep]
        
        bb = boxes[b].gather(0, qb.unsqueeze(-1).repeat(1, 4))
        
        out = {
            "labels": lb.detach().cpu(),
            "boxes": bb.detach().cpu(),
            "scores": sb.detach().cpu(),
        }
        
        if include_all_for_map:
            all_bb = boxes[b].gather(0, topk_qidx[b].unsqueeze(-1).repeat(1, 4))
            out["all_boxes"] = all_bb.detach().cpu()
            out["all_scores"] = topk_scores[b].detach().cpu()
            out["all_labels"] = topk_labels[b].detach().cpu()
        
        if has_masks and qb.numel() > 0:
            mb = pred_masks[b, qb]
            mb = mb.to(dtype=torch.float16)
            
            masks_list = process_masks(
                mb.unsqueeze(0),
                processed_size=processed_size_tensor,
                orig_sizes=orig_sizes[b].unsqueeze(0),
                keep_ratio=config.keep_ratio,
            )
            
            mask_probs = masks_list[0].to(dtype=torch.float32)
            
            if include_all_for_map:
                out["mask_probs"] = mask_probs.detach().cpu()
                out["masks"] = (mask_probs.clamp(0, 1) >= config.conf_thresh).to(torch.uint8).detach().cpu()
                out["masks"] = cleanup_masks(out["masks"], out["boxes"])
                del out["mask_probs"]
            else:
                out["mask_probs"] = mask_probs.detach().cpu()
                out["mask_probs"] = cleanup_masks(out["mask_probs"], out["boxes"])
        
        results.append(out)
    
    return results


def predictions_to_numpy(predictions: list[dict[str, torch.Tensor]]) -> list[dict[str, np.ndarray]]:
    """Convert predictions from torch tensors to numpy arrays.
    
    Args:
        predictions: List of prediction dicts with torch tensors
    
    Returns:
        List of prediction dicts with numpy arrays
    """
    return [
        {k: v.numpy() if isinstance(v, torch.Tensor) else v for k, v in pred.items()}
        for pred in predictions
    ]

