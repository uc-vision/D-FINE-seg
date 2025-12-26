from __future__ import annotations

import cv2

import albumentations as A
from albumentations.pytorch import ToTensorV2

from d_fine.config import Mode
from d_fine.config import ImageConfig, DatasetConfig
from d_fine.utils import LetterboxRect


def init_augs(img_config: ImageConfig, mode: Mode, dataset_cfg: DatasetConfig) -> A.Compose:
    if img_config.keep_ratio:
        scaleup = mode == Mode.TRAIN
        resize = [
            LetterboxRect(
                height=img_config.target_h,
                width=img_config.target_w,
                color=(114, 114, 114),
                scaleup=scaleup,
                always_apply=True,
            )
        ]
    else:
        resize = [A.Resize(img_config.target_h, img_config.target_w, interpolation=cv2.INTER_AREA)]

    norm = [
        A.Normalize(mean=img_config.norm[0], std=img_config.norm[1]),
        ToTensorV2(),
    ]

    if mode == Mode.TRAIN:
        augs = [
            A.CoarseDropout(
                num_holes_range=(1, 2),
                hole_height_range=(0.05, 0.15),
                hole_width_range=(0.05, 0.15),
                p=dataset_cfg.augs.coarse_dropout_prob,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=dataset_cfg.augs.brightness.limit,
                p=dataset_cfg.augs.brightness.prob,
            ),
            A.RandomGamma(
                gamma_limit=dataset_cfg.augs.gamma.limit,
                p=dataset_cfg.augs.gamma.prob,
            ),
            A.Blur(
                blur_limit=dataset_cfg.augs.blur.limit,
                p=dataset_cfg.augs.blur.prob,
            ),
            A.GaussNoise(
                var_limit=tuple(s ** 2 for s in dataset_cfg.augs.noise.std_range),
                p=dataset_cfg.augs.noise.prob,
            ),
            A.ToGray(p=dataset_cfg.augs.to_gray_prob),
            A.HueSaturationValue(
                hue_shift_limit=dataset_cfg.augs.hsv.hue_shift_limit,
                sat_shift_limit=dataset_cfg.augs.hsv.sat_shift_limit,
                val_shift_limit=dataset_cfg.augs.hsv.val_shift_limit,
                p=dataset_cfg.augs.hsv.prob,
            ),
            A.Affine(
                rotate=[90, 90],
                p=dataset_cfg.augs.rotate_90_prob,
                fit_output=True,
            ),
            A.HorizontalFlip(p=dataset_cfg.augs.left_right_flip),
            A.VerticalFlip(p=dataset_cfg.augs.up_down_flip),
            A.Rotate(
                limit=dataset_cfg.augs.rotation.degree,
                p=dataset_cfg.augs.rotation.prob,
                interpolation=cv2.INTER_AREA,
                border_mode=cv2.BORDER_CONSTANT,
                fill=(114, 114, 114),
            ),
        ]

        return A.Compose(
            augs + resize + norm,
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
        )
    elif mode in [Mode.VAL, Mode.TEST, Mode.BENCH]:
        return A.Compose(
            resize + norm,
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

