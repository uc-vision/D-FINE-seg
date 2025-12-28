import time
import datetime
import torch
import torch.distributed as dist
from collections import deque
from typing import Dict, Any, Optional, Iterable
from loguru import logger
from tqdm import tqdm


class SmoothedValue:
  """Track a series of values and provide access to smoothed values over a window or the global average."""

  def __init__(self, window_size: int = 20, fmt: Optional[str] = None):
    if fmt is None:
      fmt = "{median:.4f} ({global_avg:.4f})"
    self.deque = deque(maxlen=window_size)
    self.total = 0.0
    self.count = 0
    self.fmt = fmt

  def update(self, value: float, n: int = 1):
    self.deque.append(value)
    self.count += n
    self.total += value * n

  def synchronize_between_processes(self):
    if not dist.is_available() or not dist.is_initialized():
      return
    t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
    dist.barrier()
    dist.all_reduce(t)
    self.count, self.total = int(t[0].item()), t[1].item()

  @property
  def median(self) -> float:
    return float(torch.tensor(list(self.deque)).median().item()) if self.deque else 0.0

  @property
  def avg(self) -> float:
    return (
      float(torch.tensor(list(self.deque), dtype=torch.float32).mean().item())
      if self.deque
      else 0.0
    )

  @property
  def global_avg(self) -> float:
    return self.total / self.count if self.count > 0 else 0.0

  @property
  def max(self) -> float:
    return float(max(self.deque)) if self.deque else 0.0

  @property
  def value(self) -> float:
    return self.deque[-1] if self.deque else 0.0

  def __str__(self) -> str:
    return self.fmt.format(
      median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value
    )


class MetricLogger:
  def __init__(self, delimiter: str = "\t"):
    self.meters: Dict[str, SmoothedValue] = {}
    self.delimiter = delimiter

  def update(self, **kwargs):
    for k, v in kwargs.items():
      if isinstance(v, torch.Tensor):
        v = v.item()
      assert isinstance(v, (float, int))
      if k not in self.meters:
        self.meters[k] = SmoothedValue()
      self.meters[k].update(v)

  def __getattr__(self, attr: str) -> SmoothedValue:
    if attr in self.meters:
      return self.meters[attr]
    if attr in self.__dict__:
      return self.__dict__[attr]
    raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

  def __str__(self) -> str:
    loss_str = []
    for name, meter in self.meters.items():
      loss_str.append(f"{name}: {str(meter)}")
    return self.delimiter.join(loss_str)

  def synchronize_between_processes(self):
    for meter in self.meters.values():
      meter.synchronize_between_processes()

  def add_meter(self, name: str, meter: SmoothedValue):
    self.meters[name] = meter

  def log_every(self, iterable: Iterable[Any], print_freq: int, header: str = ""):
    """Log metrics every print_freq iterations using tqdm for progress tracking."""
    pbar = tqdm(iterable, desc=header, dynamic_ncols=True)

    for i, obj in enumerate(pbar):
      yield obj
      if i % print_freq == 0:
        pbar.set_postfix({k: f"{v.median:.4f}" for k, v in self.meters.items()})
