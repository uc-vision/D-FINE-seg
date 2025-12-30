from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class ConfusionMatrix:
  n_classes: int
  matrix: np.ndarray
  class_to_idx: dict[int, int] = field(default_factory=dict)

  @classmethod
  def empty(cls, n_classes: int, class_to_idx: dict[int, int]) -> ConfusionMatrix:
    return cls(
      n_classes=n_classes,
      matrix=np.zeros((n_classes + 1, n_classes + 1), dtype=int),
      class_to_idx=class_to_idx,
    )

  def update(self, gt_label: int | None, pred_label: int | None) -> ConfusionMatrix:
    g_idx = (
      self.class_to_idx.get(gt_label, self.n_classes) if gt_label is not None else self.n_classes
    )
    p_idx = (
      self.class_to_idx.get(pred_label, self.n_classes)
      if pred_label is not None
      else self.n_classes
    )
    new_matrix = self.matrix.copy()
    new_matrix[g_idx, p_idx] += 1
    return ConfusionMatrix(
      n_classes=self.n_classes,
      matrix=new_matrix,
      class_to_idx=self.class_to_idx,
    )

  def __add__(self, other: ConfusionMatrix) -> ConfusionMatrix:
    return ConfusionMatrix(
      n_classes=self.n_classes,
      matrix=self.matrix + other.matrix,
      class_to_idx=self.class_to_idx,
    )

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
