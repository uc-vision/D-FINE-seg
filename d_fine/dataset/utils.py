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

