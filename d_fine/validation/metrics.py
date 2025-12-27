from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

@dataclass
class PerClassMetrics:
    tps: int = 0
    fps: int = 0
    fns: int = 0
    ious: list[float] = field(default_factory=list)

    def add_tp(self, iou: float):
        self.tps += 1
        self.ious.append(iou)

    def add_fp(self, iou: float = 0.0):
        self.fps += 1
        self.ious.append(iou)

    def add_fn(self, iou: float = 0.0):
        self.fns += 1
        self.ious.append(iou)

    def __add__(self, other: PerClassMetrics) -> PerClassMetrics:
        return PerClassMetrics(
            tps=self.tps + other.tps,
            fps=self.fps + other.fps,
            fns=self.fns + other.fns,
            ious=self.ious + other.ious
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


@dataclass
class EvaluationMetrics:
    f1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    iou: float = 0.0
    map_50: float = 0.0
    map_75: float = 0.0
    map_50_95: float = 0.0
    map_50_mask: float = 0.0
    map_75_mask: float = 0.0
    map_50_95_mask: float = 0.0
    tps: int = 0
    fps: int = 0
    fns: int = 0
    per_class: dict[int, PerClassMetrics] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, float | int | dict[str, float]]:
        res: dict[str, float | int | dict[str, float]] = {k: v for k, v in self.__dict__.items() if not k.startswith("_") and k != "per_class"}
        for cls_id, m in self.per_class.items():
            res[f"class_{cls_id}"] = {
                "f1": m.f1,
                "precision": m.precision,
                "recall": m.recall,
                "iou": m.avg_iou
            }
        return res
    
    @classmethod
    def from_dict(cls, d: dict[str, float]) -> EvaluationMetrics:
        from dataclasses import fields
        valid_keys = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid_keys})

    def __repr__(self) -> str:
        parts = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                 for k, v in self.to_dict().items() if not isinstance(v, dict)]
        return f"EvaluationMetrics({', '.join(parts)})"

