
import cv2
import numpy as np
import tensorrt as trt
import torch
from numpy.typing import NDArray

from d_fine.config import ModelConfig
from d_fine.utils import cleanup_masks
from d_fine.infer.utils import letterbox, mask2poly, process_boxes, process_masks


class TRT_model:
    def __init__(
        self,
        model_path: str,
        model_config: ModelConfig,
    ) -> None:
        self.input_size = (model_config.input_height, model_config.input_width)
        self.n_outputs = model_config.n_outputs
        self.model_path = model_path
        self.rect = model_config.rect
        self.half = model_config.half
        self.keep_ratio = model_config.keep_ratio
        self.channels = 3

        self.conf_threshs = [model_config.conf_thresh] * self.n_outputs

        self.device = model_config.device
        self.np_dtype = model_config.np_dtype

        self._load_model()
        self._test_pred()

    def _load_model(self):
        self.TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(self.model_path, "rb") as f, trt.Runtime(self.TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

    @staticmethod
    def _torch_dtype_from_trt(trt_dtype):
        if trt_dtype == trt.float32:
            return torch.float32
        elif trt_dtype == trt.float16:
            return torch.float16
        elif trt_dtype == trt.int32:
            return torch.int32
        elif trt_dtype == trt.int8:
            return torch.int8
        else:
            raise TypeError(f"Unsupported TensorRT data type: {trt_dtype}")

    def _test_pred(self) -> None:
        random_image = np.random.randint(0, 255, size=(1100, 1000, self.channels), dtype=np.uint8)
        processed_inputs, processed_sizes, original_sizes = self._prepare_inputs(random_image)
        preds = self._predict(processed_inputs)
        self._postprocess(preds, processed_sizes, original_sizes)


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

    def _predict(self, img: torch.Tensor) -> list[torch.Tensor]:
        # 1) make contiguous and grab the full (B, C, H, W) shape
        img = img.contiguous()
        batch_shape = tuple(img.shape)

        # 2) prepare our buffer-pointer list
        n_io = self.engine.num_io_tensors
        bindings: list[int] = [None] * n_io
        outputs: list[torch.Tensor] = []

        # 3) for each I/O slot, either bind the input or allocate an output
        for i in range(n_io):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            dims = tuple(self.engine.get_tensor_shape(name))
            dt = self.engine.get_tensor_dtype(name)
            t_dt = self._torch_dtype_from_trt(dt)

            if mode == trt.TensorIOMode.INPUT:
                # set our actual batch‐shape on the context
                ok = self.context.set_input_shape(name, batch_shape)
                assert ok, f"Failed to set input shape for {name} -> {batch_shape}"
                # point that binding at our tensor’s data ptr
                bindings[i] = img.data_ptr()
            else:
                # allocate a matching output tensor (B, *dims[1:])
                out_shape = (batch_shape[0],) + dims[1:]
                out = torch.empty(out_shape, dtype=t_dt, device=self.device)
                outputs.append(out)
                bindings[i] = out.data_ptr()

        # 4) run inference
        self.context.execute_v2(bindings)

        # 5) return all output tensors
        return outputs

    def _postprocess(
        self,
        outputs: torch.Tensor,
        processed_sizes: list[tuple[int, int]],
        original_sizes: list[tuple[int, int]],
        num_top_queries=300,
        use_focal_loss=True,
    ) -> list[dict[str, NDArray]]:
        """
        returns List with BS length. Each element is a dict {"labels", "boxes", "scores"}
        """
        logits, boxes = outputs[0], outputs[1]
        has_masks = len(outputs) > 2 and outputs[2] is not None
        pred_masks = outputs[2] if has_masks else None  # [B,Q,Hm,Wm]
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



