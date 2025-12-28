from __future__ import annotations

from pathlib import Path
from d_fine.dataset.coco.coco_dataset import CocoLoader


def get_loader_class(root_path: Path, format: str = "auto") -> type[CocoLoader]:
  """Get loader class for specified dataset format.

  Args:
      root_path: Path to dataset root directory
      format: Dataset format - "coco", or "auto" (default: "auto")
              If "auto", detects format by checking for COCO annotations file

  Returns:
      Loader class (CocoLoader)
  """
  if format == "coco":
    return CocoLoader
  elif format == "auto":
    # Check if annotations directory exists
    if (root_path / "annotations").exists():
      return CocoLoader
    raise ValueError(f"No COCO dataset found at {root_path}")
  else:
    raise ValueError(f"Unknown or unsupported format: {format}. Must be 'coco' or 'auto'")
