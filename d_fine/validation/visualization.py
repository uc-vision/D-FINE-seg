from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from loguru import logger
from numpy.typing import NDArray
from tqdm import tqdm

from ..utils import abs_xyxy_to_norm_xywh, draw_mask, vis_one_box
from ..infer.torch_model import Torch_model


def visualize(
    img: NDArray[np.uint8],
    boxes: NDArray[np.float32],
    labels: NDArray[np.int64],
    scores: NDArray[np.float32],
    output_path: Path,
    img_path: str,
    label_to_name: dict[int, str],
    masks: NDArray[np.uint8] | None = None,
) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    for box, label, score in zip(boxes, labels, scores):
        vis_one_box(img, box, label, mode="pred", label_to_name=label_to_name, score=score)
    if masks is not None:
        for i in range(masks.shape[0]):
            img = draw_mask(img, masks[i])
    if len(boxes):
        cv2.imwrite((str(f"{output_path / Path(img_path).stem}.jpg")), img)


def save_yolo_annotations(
    res: dict[str, np.ndarray | list[np.ndarray] | None],
    output_path: Path,
    img_path: str,
    img_shape: tuple[int, ...],
) -> None:
    output_path.mkdir(parents=True, exist_ok=True)

    if len(res["boxes"]) == 0:
        return

    has_polys = "polys" in res and res["polys"] is not None and len(res["polys"]) > 0

    with open(output_path / f"{Path(img_path).stem}.txt", "a") as f:
        for idx, (class_id, box) in enumerate(zip(res["labels"], res["boxes"])):
            if has_polys:
                # YOLO segmentation format: class_id x1 y1 x2 y2 x3 y3 ...
                poly = res["polys"][idx]
                if len(poly) >= 3:  # Need at least 3 points for a valid polygon
                    norm_coords = []
                    for point in poly:
                        norm_coords.append(f"{point[0]:.6f}")
                        norm_coords.append(f"{point[1]:.6f}")
                    f.write(f"{int(class_id)} {' '.join(norm_coords)}\n")
            else:
                # YOLO detection format: class_id x_center y_center width height
                norm_box = abs_xyxy_to_norm_xywh(box[None], img_shape[0], img_shape[1])[0]
                f.write(
                    f"{int(class_id)} {norm_box[0]:.6f} {norm_box[1]:.6f} {norm_box[2]:.6f} {norm_box[3]:.6f}\n"
                )


def crops(
    or_img: NDArray[np.uint8],
    res: dict[str, np.ndarray | list[np.ndarray] | None],
    paddings: dict[str, float],
    output_path: Path,
    output_stem: str,
) -> None:
    if isinstance(paddings["w"], float):
        paddings["w"] = int(or_img.shape[1] * paddings["w"])
    if isinstance(paddings["h"], float):
        paddings["h"] = int(or_img.shape[0] * paddings["h"])

    for crop_id, box in enumerate(res["boxes"]):
        x1, y1, x2, y2 = map(int, box.tolist())
        crop = or_img[
            max(y1 - paddings["h"], 0) : min(y2 + paddings["h"], or_img.shape[0]),
            max(x1 - paddings["w"], 0) : min(x2 + paddings["w"], or_img.shape[1]),
        ]

        (output_path / "crops").mkdir(parents=True, exist_ok=True)
        cv2.imwrite((str(output_path / "crops" / f"{output_stem}_{crop_id}.jpg")), crop)


def figure_input_type(folder_path: Path) -> Literal["image", "video"]:
    video_types = ["mp4", "avi", "mov", "mkv"]
    img_types = ["jpg", "png", "jpeg"]

    for f in folder_path.iterdir():
        if f.suffix[1:] in video_types:
            data_type = "video"
            break
        elif f.suffix[1:] in img_types:
            data_type = "image"
            break
    logger.info(
        f"Inferencing on data type: {data_type}, path: {folder_path}",
    )
    return data_type


def process_frame(
    img: NDArray[np.uint8],
    or_img: NDArray[np.uint8],
    torch_model: Torch_model,
    batch: int,
    conf_thresh: float,
    output_path: Path,
    label_to_name: dict[int, str],
    img_path: str,
    to_crop: bool,
    paddings: dict[str, float],
) -> set[int]:
    raw_res = torch_model(img)

    if "mask_probs" in raw_res[0]:
        raw_res[0]["masks"] = (raw_res[batch]["mask_probs"] >= conf_thresh).astype(np.uint8)
    res = {
        "boxes": raw_res[batch]["boxes"],
        "labels": raw_res[batch]["labels"],
        "scores": raw_res[batch]["scores"],
    }
    if "masks" in raw_res[0]:
        res["masks"] = raw_res[batch]["masks"]
        res["polys"] = torch_model.mask2poly(res["masks"], img.shape)

    visualize(
        img=img,
        boxes=res["boxes"],
        labels=res["labels"],
        scores=res["scores"],
        output_path=output_path / "images",
        img_path=img_path,
        label_to_name=label_to_name,
        masks=res.get("masks", None),
    )

    labels = set(res["labels"])

    save_yolo_annotations(
        res=res, output_path=output_path / "labels", img_path=img_path, img_shape=img.shape
    )

    if to_crop:
        crops(or_img, res, paddings, output_path, Path(img_path).stem)

    return labels


def run_images(
    torch_model: Torch_model,
    folder_path: Path,
    output_path: Path,
    label_to_name: dict[int, str],
    to_crop: bool,
    paddings: dict[str, float],
    conf_thresh: float,
) -> None:
    batch = 0
    imag_paths = [img.name for img in folder_path.iterdir() if not str(img).startswith(".")]
    labels = set()
    for img_path in tqdm(imag_paths):
        img = cv2.imread(str(folder_path / img_path))
        or_img = img.copy()
        labels.update(process_frame(
            img, or_img, torch_model, batch, conf_thresh, output_path, label_to_name, img_path, to_crop, paddings
        ))

    with open(output_path / "labels.txt", "w") as f:
        for class_id in labels:
            f.write(f"{label_to_name[int(class_id)]}\n")


def run_videos(
    torch_model: Torch_model,
    folder_path: Path,
    output_path: Path,
    label_to_name: dict[int, str],
    to_crop: bool,
    paddings: dict[str, float],
    conf_thresh: float,
) -> None:
    batch = 0
    vid_paths = [vid.name for vid in folder_path.iterdir() if not str(vid.name).startswith(".")]
    labels = set()
    for vid_path in tqdm(vid_paths):
        vid = cv2.VideoCapture(str(folder_path / vid_path))
        success, img = vid.read()
        idx = 0
        while success:
            idx += 1
            frame_name = f"{Path(vid_path).stem}_frame_{idx}"
            labels.update(process_frame(
                img, img, torch_model, batch, conf_thresh, output_path, label_to_name, frame_name, to_crop, paddings
            ))
            success, img = vid.read()

    with open(output_path / "labels.txt", "w") as f:
        for class_id in labels:
            f.write(f"{label_to_name[int(class_id)]}\n")

