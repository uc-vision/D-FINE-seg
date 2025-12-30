from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ValidationConfig:
  conf_threshold: float = 0.5
  iou_threshold: float = 0.5
  label_to_name: dict[int, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PerClassMetrics:
  tps: int = 0
  fps: int = 0
  fns: int = 0
  ious: tuple[float, ...] = field(default_factory=tuple)

  @classmethod
  def tp(cls, iou: float) -> PerClassMetrics:
    return cls(tps=1, ious=(float(iou),))

  @classmethod
  def fp(cls, iou: float = 0.0) -> PerClassMetrics:
    return cls(fps=1, ious=(float(iou),))

  @classmethod
  def fn(cls, iou: float = 0.0) -> PerClassMetrics:
    return cls(fns=1, ious=(float(iou),))

  def __add__(self, other: PerClassMetrics) -> PerClassMetrics:
    return PerClassMetrics(
      tps=self.tps + other.tps,
      fps=self.fps + other.fps,
      fns=self.fns + other.fns,
      ious=self.ious + tuple(other.ious),
    )

  @property
  def precision(self) -> float:
    total = self.tps + self.fps
    return self.tps / total if total > 0 else 0.0

  @property
  def recall(self) -> float:
    total = self.tps + self.fns
    return self.tps / total if total > 0 else 0.0

  @property
  def f1(self) -> float:
    p, r = self.precision, self.recall
    return 2 * p * r / (p + r) if p + r > 0 else 0.0

  @property
  def avg_iou(self) -> float:
    return float(np.mean(self.ious)) if self.ious else 0.0


@dataclass(frozen=True)
class APMetrics:
  map_50: float = 0.0
  map_75: float = 0.0
  map_50_95: float = 0.0

  def to_dict(self, prefix: str = "") -> dict[str, float]:
    return {
      f"{prefix}map_50": self.map_50,
      f"{prefix}map_75": self.map_75,
      f"{prefix}map_50_95": self.map_50_95,
    }


@dataclass(frozen=True)
class CoreMetrics:
  f1: float = 0.0
  precision: float = 0.0
  recall: float = 0.0
  iou: float = 0.0

  def to_dict(self) -> dict[str, float]:
    return {
      "f1": self.f1,
      "precision": self.precision,
      "recall": self.recall,
      "iou": self.iou,
    }


@dataclass(frozen=True)
class Counts:
  tps: int = 0
  fps: int = 0
  fns: int = 0

  def to_dict(self) -> dict[str, int]:
    return {
      "tps": self.tps,
      "fps": self.fps,
      "fns": self.fns,
    }


@dataclass(frozen=True)
class EvaluationMetrics:
  core: CoreMetrics = field(default_factory=CoreMetrics)
  counts: Counts = field(default_factory=Counts)
  bbox: APMetrics = field(default_factory=APMetrics)
  mask: APMetrics = field(default_factory=APMetrics)
  per_class: dict[int, PerClassMetrics] = field(default_factory=dict)

  @property
  def f1(self) -> float:
    return self.core.f1

  @property
  def precision(self) -> float:
    return self.core.precision

  @property
  def recall(self) -> float:
    return self.core.recall

  @property
  def iou(self) -> float:
    return self.core.iou

  @property
  def map_50(self) -> float:
    return self.bbox.map_50

  @property
  def map_75(self) -> float:
    return self.bbox.map_75

  @property
  def map_50_95(self) -> float:
    return self.bbox.map_50_95

  @property
  def map_50_mask(self) -> float:
    return self.mask.map_50

  @property
  def map_75_mask(self) -> float:
    return self.mask.map_75

  @property
  def map_50_95_mask(self) -> float:
    return self.mask.map_50_95

  @property
  def tps(self) -> int:
    return self.counts.tps

  @property
  def fps(self) -> int:
    return self.counts.fps

  @property
  def fns(self) -> int:
    return self.counts.fns

  def to_dict(self) -> dict[str, float | int | dict[str, float]]:
    res: dict[str, float | int | dict[str, float]] = {
      "f1": self.f1,
      "precision": self.precision,
      "recall": self.recall,
      "iou": self.iou,
      "tps": self.tps,
      "fps": self.fps,
      "fns": self.fns,
    }
    res.update(self.bbox.to_dict())
    res.update(self.mask.to_dict(prefix="mask_"))

    for cls_id, m in self.per_class.items():
      res[f"class_{cls_id}"] = {
        "f1": m.f1,
        "precision": m.precision,
        "recall": m.recall,
        "iou": m.avg_iou,
      }
    return res

  @classmethod
  def from_dict(cls, d: dict[str, float]) -> EvaluationMetrics:
    return cls(
      core=CoreMetrics(
        f1=d.get("f1", 0.0),
        precision=d.get("precision", 0.0),
        recall=d.get("recall", 0.0),
        iou=d.get("iou", 0.0),
      ),
      counts=Counts(
        tps=int(d.get("tps", 0)),
        fps=int(d.get("fps", 0)),
        fns=int(d.get("fns", 0)),
      ),
      bbox=APMetrics(
        map_50=d.get("map_50", 0.0),
        map_75=d.get("map_75", 0.0),
        map_50_95=d.get("map_50_95", 0.0),
      ),
      mask=APMetrics(
        map_50=d.get("mask_map_50", 0.0),
        map_75=d.get("mask_map_75", 0.0),
        map_50_95=d.get("mask_map_50_95", 0.0),
      ),
    )

  def __repr__(self) -> str:
    parts = [
      f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
      for k, v in self.to_dict().items()
      if not isinstance(v, dict)
    ]
    return f"EvaluationMetrics({', '.join(parts)})"
