from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class ConfusionMatrix:
  n_classes: int
  matrix: np.ndarray = field(init=False)
  class_to_idx: dict[int, int] = field(default_factory=dict)

  def __post_init__(self):
    self.matrix = np.zeros((self.n_classes + 1, self.n_classes + 1), dtype=int)

  def update(self, gt_label: int | None, pred_label: int | None):
    g_idx = (
      self.class_to_idx.get(gt_label, self.n_classes) if gt_label is not None else self.n_classes
    )
    p_idx = (
      self.class_to_idx.get(pred_label, self.n_classes)
      if pred_label is not None
      else self.n_classes
    )
    self.matrix[g_idx, p_idx] += 1

  def plot(self, save_path: Path, label_to_name: dict[int, str]):
    class_labels = [
      label_to_name.get(cls_id, str(cls_id)) for cls_id in sorted(self.class_to_idx.keys())
    ] + ["background"]
    plt.figure(figsize=(10, 8))
    plt.imshow(self.matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_labels))
    plt.xticks(tick_marks, class_labels, rotation=45)
    plt.yticks(tick_marks, class_labels)

    thresh = self.matrix.max() / 2.0
    for i, j in np.ndindex(self.matrix.shape):
      plt.text(
        j,
        i,
        format(self.matrix[i, j], "d"),
        ha="center",
        va="center",
        color="white" if self.matrix[i, j] > thresh else "black",
      )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
