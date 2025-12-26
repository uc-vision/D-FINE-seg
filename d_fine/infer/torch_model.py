
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from numpy.typing import NDArray
from torchvision.ops import nms

from d_fine.core.dfine import build_model
from d_fine.config import ModelConfig, TrainConfig
from d_fine.utils import cleanup_masks
from d_fine.infer.utils import letterbox, mask2poly, process_boxes, process_masks


class Torch_model:
    def __init__(
        self,
        train_config: TrainConfig | None = None,
        model_config: ModelConfig | None = None,
        model_name: str | None = None,
        model_path: str | None = None,
        use_nms: bool = False,
        device: str | None = None,
    ):
        if train_config is not None:
            model_name = train_config.model_name
            model_path = str(train_config.paths.path_to_save / "model.pt")
            if model_config is None:
                model_config = train_config.get_model_config(train_config.num_classes)
        
        if model_name is None or model_path is None:
            raise ValueError("Either train_config must be provided, or model_name and model_path must be provided")
        
        if model_config is None:
            raise ValueError("model_config must be provided")
        
        self.input_size = (model_config.input_height, model_config.input_width)
        self.n_outputs = model_config.n_outputs
        self.model_name = model_name
        self.model_path = model_path
        self.rect = model_config.rect
        self.half = model_config.half
        self.keep_ratio = model_config.keep_ratio
        self.use_nms = use_nms
        self.enable_mask_head = model_config.enable_mask_head
        self.channels = 3
        self.debug_mode = False

        self.conf_threshs = [model_config.conf_thresh] * self.n_outputs

        self.device = device if device else model_config.device

        self.np_dtype = model_config.np_dtype

        self._load_model()
        self._test_pred()

    def _load_model(self):
        self.model = build_model(
            self.model_name,
            self.n_outputs,
            self.enable_mask_head,
            self.device,
            img_size=None,
        )
        self.model.load_state_dict(
            torch.load(self.model_path, weights_only=True, map_location=torch.device("cpu")),
            strict=False,
        )
        self.model.eval()
        self.model.to(self.device)

        logger.info(f"Torch model, Device: {self.device}")

    def _test_pred(self) -> None:
        random_image = np.random.randint(0, 255, size=(1100, 1000, self.channels), dtype=np.uint8)
        processed_inputs, processed_sizes, original_sizes = self._prepare_inputs(random_image)
        preds = self._predict(processed_inputs)
        self._postprocess(preds, processed_sizes, original_sizes)


    def _preds_postprocess(
        self,
        outputs,
        processed_sizes,
        original_sizes,
        num_top_queries=300,
        use_focal_loss=True,
    ) -> list[dict[str, torch.Tensor]]:
        """
        returns List with BS length. Each element is a dict {"labels", "boxes", "scores"}
        """
        logits, boxes = outputs["pred_logits"], outputs["pred_boxes"]
        has_masks = ("pred_masks" in outputs) and (outputs["pred_masks"] is not None)
        pred_masks = outputs["pred_masks"] if has_masks else None  # [B,Q,Hm,Wm]
        B, Q = logits.shape[:2]

        boxes = process_boxes(
            boxes, processed_sizes, original_sizes, self.keep_ratio, self.device
        )  # B x TopQ x 4

        # scores/labels and preliminary topK over all Q*C
        if use_focal_loss:
            scores_all = torch.sigmoid(logits)  # [B,Q,C]
            flat = scores_all.flatten(1)  # [B, Q*C]
            # pre-topk to avoid scanning all queries later
            K = min(num_top_queries, flat.shape[1])
            topk_scores, topk_idx = torch.topk(flat, K, dim=-1)  # [B,K]
            topk_labels = topk_idx - (topk_idx // self.n_outputs) * self.n_outputs  # [B,K]
            topk_qidx = topk_idx // self.n_outputs  # [B,K]
        else:
            probs = torch.softmax(logits, dim=-1)[:, :, :-1]  # [B,Q,C-1]
            topk_scores, topk_labels = probs.max(dim=-1)  # [B,Q]
            # keep at most K queries per image by score
            K = min(num_top_queries, Q)
            topk_scores, order = torch.topk(topk_scores, K, dim=-1)  # [B,K]
            topk_labels = topk_labels.gather(1, order)  # [B,K]
            topk_qidx = order

        results = []
        for b in range(B):
            sb = topk_scores[b]
            lb = topk_labels[b]
            qb = topk_qidx[b]
            # Apply per-class confidence thresholds
            conf_threshs_tensor = torch.tensor(self.conf_threshs, device=sb.device)
            keep = sb >= conf_threshs_tensor[lb]

            sb = sb[keep]
            lb = lb[keep]
            qb = qb[keep]
            # gather boxes once
            bb = boxes[b].gather(0, qb.unsqueeze(-1).repeat(1, 4))

            out = {
                "labels": lb.detach().cpu().numpy(),
                "boxes": bb.detach().cpu().numpy(),
                "scores": sb.detach().cpu().numpy(),
            }

            if has_masks and qb.numel() > 0:
                # gather only kept masks, then cast to half to save mem during resizing
                mb = pred_masks[b, qb]  # [K', Hm, Wm] logits or probs
                mb = mb.to(dtype=torch.float16)  # reduce VRAM and RAM during resize
                # resize to original size (list of length 1)
                masks_list = process_masks(
                    mb.unsqueeze(0),  # [1,K',Hm,Wm]
                    processed_size=np.array(processed_sizes[b]),  # (Hin, Win)
                    orig_sizes=np.array(original_sizes[b])[None],  # [1,2]
                    keep_ratio=self.keep_ratio,
                )
                out["mask_probs"] = (
                    masks_list[0].to(dtype=torch.float32).detach().cpu().numpy()
                )  # [B, H, W]
                # clean up masks outside of the corresponding bbox
                out["mask_probs"] = cleanup_masks(out["mask_probs"], out["boxes"])

            results.append(out)

        return results

    def _compute_nearest_size(self, shape, target_size, stride=32) -> tuple[int, int]:
        """
        Get nearest size that is divisible by 32
        """
        scale = target_size / max(shape)
        new_shape = [int(round(dim * scale)) for dim in shape]

        # Make sure new dimensions are divisible by the stride
        new_shape = [max(stride, int(np.ceil(dim / stride) * stride)) for dim in new_shape]
        return new_shape

    def _preprocess(self, img: NDArray, stride: int = 32) -> torch.tensor:
        if not self.keep_ratio:  # simple resize
            img = cv2.resize(
                img, (self.input_size[1], self.input_size[0]), interpolation=cv2.INTER_AREA
            )
        elif self.rect:  # keep ratio and cut paddings
            target_height, target_width = self._compute_nearest_size(
                img.shape[:2], max(*self.input_size)
            )
            img = letterbox(img, (target_height, target_width), stride=stride, auto=False)[0]
        else:  # keep ratio adding paddings
            img = letterbox(
                img, (self.input_size[0], self.input_size[1]), stride=stride, auto=False
            )[0]

        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, then HWC to CHW
        img = np.ascontiguousarray(img, dtype=self.np_dtype)
        img /= 255.0

        # save debug image
        if self.debug_mode:
            debug_img = img.reshape([1, *img.shape])
            debug_img = debug_img[0].transpose(1, 2, 0)  # CHW to HWC
            debug_img = (debug_img * 255.0).astype(np.uint8)  # Convert to uint8
            debug_img = debug_img[:, :, ::-1]  # RGB to BGR for saving
            cv2.imwrite("torch_infer.jpg", debug_img)
        return img

    def _prepare_inputs(self, inputs):
        original_sizes = []
        processed_sizes = []

        if isinstance(inputs, np.ndarray) and inputs.ndim == 3:  # single image
            processed_inputs = self._preprocess(inputs)[None]
            original_sizes.append((inputs.shape[0], inputs.shape[1]))
            processed_sizes.append((processed_inputs[0].shape[1], processed_inputs[0].shape[2]))

        elif isinstance(inputs, np.ndarray) and inputs.ndim == 4:  # batch of images
            processed_inputs = np.zeros(
                (inputs.shape[0], self.channels, self.input_size[0], self.input_size[1]),
                dtype=self.np_dtype,
            )
            for idx, image in enumerate(inputs):
                processed_inputs[idx] = self._preprocess(image)
                original_sizes.append((image.shape[0], image.shape[1]))
                processed_sizes.append(
                    (processed_inputs[idx].shape[1], processed_inputs[idx].shape[2])
                )

        tensor = torch.from_numpy(processed_inputs)  # no copying
        if self.device == "cuda":
            tensor = tensor.pin_memory().to(self.device, non_blocking=True)
        else:
            tensor = tensor.to(self.device)
        return tensor, processed_sizes, original_sizes

    @torch.no_grad()
    def _predict(self, inputs) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
        return self.model(inputs)

    def _postprocess(
        self,
        preds: torch.tensor,
        processed_sizes: list[tuple[int, int]],
        original_sizes: list[tuple[int, int]],
    ):
        output = self._preds_postprocess(preds, processed_sizes, original_sizes)
        if self.use_nms:
            for idx, res in enumerate(output):
                boxes, scores, classes, masks = non_max_suppression(
                    res["boxes"],
                    res["scores"],
                    res["labels"],
                    masks=res.get("mask_probs", None),
                    iou_threshold=0.5,
                )
                output[idx]["boxes"] = boxes
                output[idx]["scores"] = scores
                output[idx]["labels"] = classes
                if "mask_probs" in res:
                    output[idx]["mask_probs"] = masks

        return output

    @torch.no_grad()
    def __call__(self, inputs: NDArray[np.uint8]) -> list[dict[str, np.ndarray]]:
        """
        Input image as ndarray (BGR, HWC) or BHWC
        Output:
            List of batch size length. Each element is a dict {"labels", "boxes", "scores"}
            labels: np.ndarray of shape (N,), dtype np.int64
            boxes: np.ndarray of shape (N, 4), dtype np.float32, abs values
            scores: np.ndarray of shape (N,), dtype np.float32
            masks: np.ndarray of shape (N, H, W), dtype np.uint8. N = number of objects
        """
        processed_inputs, processed_sizes, original_sizes = self._prepare_inputs(inputs)
        preds = self._predict(processed_inputs)
        return self._postprocess(preds, processed_sizes, original_sizes)



def filter_preds(preds, conf_threshs: list[float]):
    conf_threshs = torch.tensor(conf_threshs, device=preds[0]["scores"].device)
    for pred in preds:
        mask = pred["scores"] >= conf_threshs[pred["labels"]]
        pred["scores"] = pred["scores"][mask]
        pred["boxes"] = pred["boxes"][mask]
        pred["labels"] = pred["labels"][mask]
    return preds


def non_max_suppression(boxes, scores, classes, masks=None, iou_threshold=0.5):
    """
    Applies Non-Maximum Suppression (NMS) to filter bounding boxes.

    Parameters:
    - boxes (torch.Tensor): Tensor of shape (N, 4) containing bounding boxes in [x1, y1, x2, y2] format.
    - scores (torch.Tensor): Tensor of shape (N,) containing confidence scores for each box.
    - classes (torch.Tensor): Tensor of shape (N,) containing class indices for each box.
    - masks (torch.Tensor, optional): Tensor of shape (N, H, W) containing masks for each box.
    - iou_threshold (float): Intersection Over Union (IOU) threshold for NMS.

    Returns:
    - filtered_boxes (torch.Tensor): Tensor containing filtered bounding boxes after NMS.
    - filtered_scores (torch.Tensor): Tensor containing confidence scores of the filtered boxes.
    - filtered_classes (torch.Tensor): Tensor containing class indices of the filtered boxes.
    - filtered_masks (torch.Tensor or None): Tensor containing masks of the filtered boxes, or None if masks was None.
    """
    boxes = torch.from_numpy(boxes) if not isinstance(boxes, torch.Tensor) else boxes
    scores = torch.from_numpy(scores) if not isinstance(scores, torch.Tensor) else scores
    classes = torch.from_numpy(classes) if not isinstance(classes, torch.Tensor) else classes
    if masks is not None:
        masks = torch.from_numpy(masks) if not isinstance(masks, torch.Tensor) else masks
    # Prepare lists to collect the filtered boxes, scores, and classes
    filtered_boxes = []
    filtered_scores = []
    filtered_classes = []
    filtered_masks = []

    # Get unique classes present in the detections
    unique_classes = classes.unique()

    # Step 2: Perform NMS for each class separately
    for unique_class in unique_classes:
        # Get indices of boxes belonging to the current class
        cls_mask = classes == unique_class
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]

        # Apply NMS for the current class
        nms_indices = nms(cls_boxes, cls_scores, iou_threshold)

        # Collect the filtered boxes, scores, and classes
        filtered_boxes.append(cls_boxes[nms_indices])
        filtered_scores.append(cls_scores[nms_indices])
        filtered_classes.append(classes[cls_mask][nms_indices])

        if masks is not None:
            cls_masks = masks[cls_mask]
            filtered_masks.append(cls_masks[nms_indices])

    # Step 3: Concatenate the results
    if filtered_boxes:
        filtered_boxes = torch.cat(filtered_boxes)
        filtered_scores = torch.cat(filtered_scores)
        filtered_classes = torch.cat(filtered_classes)
        if masks is not None:
            filtered_masks = torch.cat(filtered_masks)
        else:
            filtered_masks = None
    else:
        # If no boxes remain after NMS, return empty tensors
        filtered_boxes = torch.empty((0, 4))
        filtered_scores = torch.empty((0,))
        filtered_classes = torch.empty((0,), dtype=classes.dtype)

        filtered_masks = None
        if masks is not None:
            filtered_masks = torch.empty(
                (0, *masks.shape[1:]), dtype=masks.dtype, device=masks.device
            )

    return filtered_boxes, filtered_scores, filtered_classes, filtered_masks
