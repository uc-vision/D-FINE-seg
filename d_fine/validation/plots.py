from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from loguru import logger
from collections.abc import Callable
from d_fine.core.types import ImageResult
from .metrics import EvaluationMetrics
from .confusion_matrix import ConfusionMatrix


class ValidationPlotter:
  def __init__(self, thresholds: np.ndarray, label_to_name: dict[int, str]):
    self.thresholds = thresholds
    self.label_to_name = label_to_name

  def save_all_plots(
    self,
    path_to_save: Path,
    preds: list[ImageResult],
    compute_metrics_fn: Callable[[list[ImageResult]], EvaluationMetrics],
    conf_matrix: ConfusionMatrix | None = None,
  ):
    path_to_save.mkdir(parents=True, exist_ok=True)

    if conf_matrix:
      conf_matrix.plot(path_to_save / "confusion_matrix.png", self.label_to_name)

    precisions, recalls, f1_scores = [], [], []

    for threshold in self.thresholds:
      filtered_preds = [p.filter(threshold) for p in preds]
      metrics = compute_metrics_fn(filtered_preds)
      precisions.append(metrics.core.precision)
      recalls.append(metrics.core.recall)
      f1_scores.append(metrics.core.f1)

    self._plot_curves(
      self.thresholds,
      precisions,
      recalls,
      path_to_save / "precision_recall_vs_threshold.png",
      "Precision",
      "Recall",
    )
    self._plot_single_curve(
      self.thresholds, f1_scores, path_to_save / "f1_score_vs_threshold.png", "F1 Score"
    )

    best_idx = len(f1_scores) - np.argmax(f1_scores[::-1]) - 1
    logger.info(
      f"Best Threshold for object detection: {round(self.thresholds[best_idx], 2)} "
      f"with F1 Score: {round(f1_scores[best_idx], 3)}"
    )

  def _plot_curves(self, x, y1, y2, path, label1, label2):
    plt.figure()
    plt.plot(x, y1, label=label1, marker="o")
    plt.plot(x, y2, label=label2, marker="o")
    plt.xlabel("Threshold")
    plt.ylabel("Value")
    plt.title(f"{label1} and {label2} vs Threshold")
    plt.legend()
    plt.grid(True)
    plt.savefig(path)
    plt.close()

  def _plot_single_curve(self, x, y, path, label):
    plt.figure()
    plt.plot(x, y, label=label, marker="o")
    plt.xlabel("Threshold")
    plt.ylabel(label)
    plt.title(f"{label} vs Threshold")
    plt.grid(True)
    plt.savefig(path)
    plt.close()
