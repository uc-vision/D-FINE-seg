from __future__ import annotations

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

from d_fine.config import Mode
from d_fine.config import ImageConfig, DatasetConfig
from d_fine.utils import LetterboxRect


def init_augs(img_config: ImageConfig, mode: Mode, dataset_cfg: DatasetConfig) -> A.Compose:
    target_h, target_w = img_config.target_h, img_config.target_w
    use_crop = img_config.use_crop

    if mode == Mode.TRAIN:
        geometric_augs = [
            A.HorizontalFlip(p=dataset_cfg.augs.left_right_flip),
            A.VerticalFlip(p=dataset_cfg.augs.up_down_flip),
            A.RandomRotate90(p=dataset_cfg.augs.rotate_90_prob),
            
            A.ShiftScaleRotate(
                shift_limit=dataset_cfg.augs.shift_limit,
                scale_limit=dataset_cfg.augs.scale_limit,
                rotate_limit=dataset_cfg.augs.rotation.degree,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT,
                value=(114, 114, 114),
                p=1.0
            ),
        ]

        if use_crop:
            geometric_augs.append(A.RandomCrop(height=target_h, width=target_w, p=1.0))
        elif img_config.keep_ratio:
            geometric_augs.append(LetterboxRect(height=target_h, width=target_w))
        else:
            geometric_augs.append(A.Resize(height=target_h, width=target_w, interpolation=cv2.INTER_AREA))

        pixel_augs = [
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
                blur_limit=(3, max(3, dataset_cfg.augs.blur.limit)),
                p=dataset_cfg.augs.blur.prob,
            ),
            A.GaussNoise(
                std_range=dataset_cfg.augs.noise.std_range,
                p=dataset_cfg.augs.noise.prob,
            ),
            A.ToGray(p=dataset_cfg.augs.to_gray_prob),
            A.HueSaturationValue(
                hue_shift_limit=dataset_cfg.augs.hsv.hue_shift_limit,
                sat_shift_limit=dataset_cfg.augs.hsv.sat_shift_limit,
                val_shift_limit=dataset_cfg.augs.hsv.val_shift_limit,
                p=dataset_cfg.augs.hsv.prob,
            ),
        ]

        return A.Compose(
            geometric_augs + pixel_augs + [A.Normalize(mean=img_config.norm[0], std=img_config.norm[1]), ToTensorV2()],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
        )

    # VAL / TEST / BENCH pipeline
    if use_crop:
        resize_aug = [A.CenterCrop(height=target_h, width=target_w, p=1.0)]
    elif img_config.keep_ratio:
        resize_aug = [LetterboxRect(height=target_h, width=target_w, color=(114, 114, 114))]
    else:
        resize_aug = [A.Resize(target_h, target_w, interpolation=cv2.INTER_AREA)]

        return A.Compose(
        resize_aug + [A.Normalize(mean=img_config.norm[0], std=img_config.norm[1]), ToTensorV2()],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
        )
