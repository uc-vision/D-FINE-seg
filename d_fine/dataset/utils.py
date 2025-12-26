from pathlib import Path

import pandas as pd


def get_loader_class(root_path: Path, format: str = "auto"):
    """Get loader class for specified dataset format.
    
    Args:
        root_path: Path to dataset root directory
        format: Dataset format - "coco", "yolo", or "auto" (default: "auto")
                If "auto", detects format by checking for COCO annotations file
    
    Returns:
        Loader class (CocoLoader or YoloLoader)
    """
    from d_fine.dataset.coco.coco_dataset import load_coco_dataset_if_available
    from d_fine.dataset.coco.coco_loader import CocoLoader
    from d_fine.dataset.yolo.yolo_loader import YoloLoader
    
    if format == "coco":
        return CocoLoader
    elif format == "yolo":
        return YoloLoader
    elif format == "auto":
        coco_dataset, _ = load_coco_dataset_if_available(root_path)
        return CocoLoader if coco_dataset is not None else YoloLoader
    else:
        raise ValueError(f"Unknown format: {format}. Must be 'coco', 'yolo', or 'auto'")


def get_splits(root_path: Path) -> dict:
    splits = {"train": None, "val": None, "test": None}
    for split_name in splits.keys():
        if (root_path / f"{split_name}.csv").exists():
            splits[split_name] = pd.read_csv(
                root_path / f"{split_name}.csv", header=None
            )
        else:
            splits[split_name] = []
    assert len(splits["train"]) and len(splits["val"]), (
        "Train and Val splits must be present"
    )
    return splits

