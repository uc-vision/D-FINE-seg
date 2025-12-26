from pathlib import Path

import click
import hydra
import numpy as np
from loguru import logger
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split


def split(
    data_path: Path,
    train_split: float,
    val_split: float,
    images_path: Path,
    ignore_negatives: bool,
    seed: int,
    shuffle: bool = True,
) -> None:
    test_split = 1 - train_split - val_split
    if test_split <= 0.001:
        test_split = 0

    img_paths = [x.name for x in images_path.iterdir() if not str(x.name).startswith(".")]

    if not shuffle:
        img_paths.sort()

    if ignore_negatives:
        for img_path in img_paths:
            if not (images_path.parent / "labels" / f"{Path(img_path).stem}.txt").exists():
                img_paths.remove(img_path)

    indices = np.arange(len(img_paths))
    train_idxs, temp_idxs = train_test_split(
        indices, test_size=(1 - train_split), random_state=seed, shuffle=shuffle
    )

    if test_split:
        test_idxs, val_idxs = train_test_split(
            temp_idxs,
            test_size=(val_split / (val_split + test_split)),
            random_state=seed,
            shuffle=shuffle,
        )
    else:
        val_idxs = temp_idxs
        test_idxs = []

    splits = {"train": train_idxs, "val": val_idxs}
    if test_split:
        splits["test"] = test_idxs

    for split_name, split in splits.items():
        with open(data_path / f"{split_name}.csv", "w") as f:
            for num, idx in enumerate(split):
                f.write(str(img_paths[idx]) + "\n")
            logger.info(f"{split_name}: {num + 1}")


@click.command()
@click.option("--data-path", required=True, type=click.Path(exists=True, path_type=Path), help="Data path")
@click.option("--train-split", type=float, default=0.85, help="Train split ratio")
@click.option("--val-split", type=float, default=0.15, help="Val split ratio")
@click.option("--ignore-negatives", is_flag=True, help="Ignore images without labels")
@click.option("--seed", type=int, default=42, help="Random seed")
@click.option("--shuffle/--no-shuffle", default=True, help="Shuffle data")
def main(
    data_path: Path,
    train_split: float,
    val_split: float,
    ignore_negatives: bool,
    seed: int,
    shuffle: bool,
) -> None:
    """Create train/val/test splits."""
    split(
        data_path,
        train_split,
        val_split,
        data_path / "images",
        ignore_negatives,
        seed,
        shuffle=shuffle,
    )


@hydra.main(version_base=None, config_path="../../", config_name="config")
def main_hydra(cfg: DictConfig) -> None:
    """Hydra entrypoint for backward compatibility."""
    data_path = Path(cfg.train.data_path)

    split(
        data_path,
        cfg.split.train_split,
        cfg.split.val_split,
        data_path / "images",
        cfg.split.ignore_negatives,
        cfg.train.seed,
        shuffle=cfg.split.shuffle,
    )


if __name__ == "__main__":
    main()
