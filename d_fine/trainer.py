import math
import time
from copy import deepcopy
from pathlib import Path
from shutil import rmtree

import numpy as np
import torch
from loguru import logger
from collections.abc import Callable
from torch.amp import GradScaler, autocast
from torch.nn import SyncBatchNorm, Module
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

from d_fine.core import dfine
from d_fine.core import dist_utils
from d_fine.core.types import ImageResult
from d_fine.core import utils as core_utils
from d_fine.config import DecisionMetric, Task, TrainConfig, LoggerType
from d_fine.dl.logging import Logger, WandbLogger, NullLogger
from d_fine.infer.utils import postprocess_ground_truth, postprocess_predictions
from d_fine.dataset.dataset import ProcessedSample
from d_fine.validation import Validator, ValidationConfig, EvaluationMetrics
from d_fine.validation import visualization as vis_utils
from d_fine.config import ClassConfig


class ModelEMA:
  model: Module
  ema_scheduler: Callable[[int], float]

  def __init__(self, student: Module, ema_momentum: float) -> None:
    # unwrap DDP if needed
    if isinstance(student, DDP):
      student = student.module
    self.model = deepcopy(student).eval()
    for param in self.model.parameters():
      param.requires_grad_(False)

    def ema_scheduler(x: int) -> float:
      return ema_momentum * (1 - math.exp(-x / 2000))

    self.ema_scheduler = ema_scheduler

  def update(self, iters: int, student: Module) -> None:
    # unwrap DDP if needed
    if isinstance(student, DDP):
      student = student.module

    student = student.state_dict()
    with torch.no_grad():
      momentum = self.ema_scheduler(iters)
      for name, param in self.model.state_dict().items():
        if param.dtype.is_floating_point:
          param *= momentum
          param += (1.0 - momentum) * student[name].detach()


class Trainer:
  cfg: TrainConfig
  project_name: str
  exp_name: str
  distributed: bool
  rank: int
  world_size: int
  is_main: bool
  local_rank: int
  device: torch.device
  decision_metrics: list[DecisionMetric]
  results_logger: Logger
  train_loader: DataLoader[ProcessedSample]
  val_loader: DataLoader[ProcessedSample]
  test_loader: DataLoader[ProcessedSample] | None
  train_sampler: Sampler | None
  num_labels: int
  model: Module
  ema_model: ModelEMA | None
  loss_fn: Module
  optimizer: Optimizer
  scheduler: OneCycleLR
  scaler: GradScaler
  early_stopping_steps: int

  def __init__(self, cfg: TrainConfig) -> None:
    self.cfg = cfg
    self.project_name = cfg.project_name
    self.exp_name = cfg.exp_name

    self.distributed = self.cfg.ddp.enabled
    self.rank = dist_utils.get_rank()
    self.world_size = dist_utils.get_world_size()
    self.is_main = self.rank == 0
    if self.distributed:
      self.local_rank = dist_utils.get_local_rank()
      self.device = torch.device("cuda", self.local_rank)
    else:
      self.local_rank = 0
      self.device = torch.device(self.cfg.device.value)

    enable_mask_head = self.cfg.dataset.task == Task.SEGMENT
    self.decision_metrics = list(self.cfg.decision_metrics)
    if enable_mask_head and DecisionMetric.F1 not in self.decision_metrics:
      self.decision_metrics.append(DecisionMetric.F1)

    if self.is_main:
      if self.cfg.logger == LoggerType.WANDB:
        self.results_logger = WandbLogger(
          project=self.project_name, name=self.exp_name, config=self.cfg.model_dump()
        )
      else:
        self.results_logger = NullLogger()
    else:
      self.results_logger = NullLogger()

    logger.info(f"Experiment: {self.exp_name}, Task: {self.cfg.dataset.task.value}")
    seed = self.cfg.seed + self.rank if self.distributed else self.cfg.seed
    core_utils.set_seeds(seed, self.cfg.cudnn_fixed)

    base_loader = self.cfg.dataset.create_loader(
      batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers
    )
    self.train_loader, self.val_loader, self.test_loader = base_loader.build_dataloaders(
      distributed=self.distributed
    )

    self.train_sampler = self.train_loader.sampler

    self.num_labels = self.train_loader.dataset.num_classes

    if self.is_main:
      # Save config for reproducibility and inference
      self.cfg.paths.path_to_save.mkdir(parents=True, exist_ok=True)
      self.cfg.save(self.cfg, self.cfg.paths.path_to_save / "config.json")

      # Also save a simpler class_config.json for compatibility
      class_config = ClassConfig(
        label_to_name=self.train_loader.dataset.label_to_name,
        conf_thresh=self.cfg.conf_thresh,
        iou_thresh=self.cfg.iou_thresh,
      )
      class_config.save(self.cfg.paths.path_to_save / "class_config.json")

    self.model = dfine.build_model(
      self.cfg.model_name,
      self.num_labels,
      enable_mask_head,
      str(self.device),
      img_size=self.cfg.dataset.img_config.img_size,
      pretrained_model_path=self.cfg.pretrained_model_path,
    )
    self.results_logger.watch(self.model)
    if self.distributed:
      if torch.cuda.is_available():
        if self.cfg.batch_size < 4:  # SyncBatch is useful for small batches
          self.model = SyncBatchNorm.convert_sync_batchnorm(self.model).to(self.device)
        self.model = DDP(
          self.model,
          device_ids=[self.local_rank],
          output_device=self.local_rank,
          find_unused_parameters=False,
        )
      else:
        # CPU DDP fallback (unlikely, but safe)
        self.model = DDP(self.model)

    self.ema_model = None
    if self.cfg.use_ema:
      self.ema_model = ModelEMA(self.model, self.cfg.ema_momentum)
      if self.is_main:
        logger.info("EMA model will be evaluated and saved")

    self.loss_fn = dfine.build_loss(
      self.cfg.model_name,
      self.num_labels,
      label_smoothing=self.cfg.label_smoothing,
      enable_mask_head=enable_mask_head,
    )

    self.optimizer = dfine.build_optimizer(
      self.model,
      lr=self.cfg.base_lr,
      backbone_lr=self.cfg.backbone_lr,
      betas=self.cfg.betas,
      weight_decay=self.cfg.weight_decay,
      base_lr=self.cfg.base_lr,
    )

    max_lr = self.cfg.base_lr * 2
    if self.cfg.model_name in ["l", "x"]:  # per group max lr for big models
      max_lr = [
        self.cfg.backbone_lr * 2,
        self.cfg.backbone_lr * 2,
        self.cfg.base_lr * 2,
        self.cfg.base_lr * 2,
      ]
    self.scheduler = OneCycleLR(
      self.optimizer,
      max_lr=max_lr,
      epochs=self.cfg.epochs,
      steps_per_epoch=len(self.train_loader) // max(self.cfg.b_accum_steps, 1),
      pct_start=self.cfg.cycler_pct_start,
      cycle_momentum=False,
    )

    if self.cfg.amp_enabled:
      self.scaler = GradScaler()

  def preds_postprocess(
    self, inputs: torch.Tensor, outputs: dict[str, torch.Tensor], orig_sizes: torch.Tensor
  ) -> list[ImageResult]:
    """
    returns list with BS length.
    """
    evaluation_config = self.cfg.get_evaluation_config(self.num_labels)
    return postprocess_predictions(
      outputs=outputs,
      orig_sizes=orig_sizes,
      config=evaluation_config,
      processed_size=tuple(inputs.shape[2:]),
    )

  def gt_postprocess(
    self, inputs: torch.Tensor, targets: list[dict[str, torch.Tensor]], orig_sizes: torch.Tensor
  ) -> list[ImageResult]:
    return postprocess_ground_truth(
      inputs=inputs,
      targets=targets,
      orig_sizes=orig_sizes,
      keep_aspect=self.cfg.dataset.img_config.keep_aspect,
    )

  @torch.no_grad()
  def get_preds_and_gt(
    self, val_loader: DataLoader
  ) -> tuple[list[ImageResult], list[ImageResult], list[np.ndarray] | None]:
    """
    Outputs gt and preds.
    Returns (all_gt, all_preds, eval_images)
    """
    all_gt, all_preds = [], []
    eval_images = None
    model = self.ema_model.model if self.ema_model else self.model

    model.eval()
    with torch.inference_mode():
      pbar = (
        tqdm(val_loader, desc="Validation inference", leave=False) if self.is_main else val_loader
      )
      for idx, (inputs, targets, img_paths) in enumerate(pbar):
        inputs = inputs.to(self.device)

        with autocast(str(self.device), enabled=self.cfg.amp_enabled, cache_enabled=True):
          raw_res = model(inputs)

        targets = [
          {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
          for t in targets
        ]
        orig_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).float().to(self.device)

        gt = self.gt_postprocess(inputs, targets, orig_sizes)
        preds = self.preds_postprocess(inputs, raw_res, orig_sizes)

        if self.cfg.visualize_eval is not None and idx == 0:
          eval_images = vis_utils.visualize_batch(
            inputs[: self.cfg.visualize_eval],
            gt[: self.cfg.visualize_eval],
            preds[: self.cfg.visualize_eval],
            label_to_name=self.train_loader.dataset.label_to_name,
            dataset=self.train_loader.dataset,
          )

        for gt_instance, pred_instance in zip(gt, preds):
          all_preds.append(pred_instance)
          all_gt.append(gt_instance)

    return all_gt, all_preds, eval_images

  def evaluate(
    self,
    val_loader: DataLoader,
    val_cfg: ValidationConfig,
    path_to_save: Path | None,
    extended: bool,
    mode: str | None = None,
  ) -> EvaluationMetrics | None:
    # All ranks perform inference on their portion of the data
    local_gt, local_preds, eval_images = self.get_preds_and_gt(val_loader=val_loader)

    # Gather predictions from all ranks to rank 0
    if self.distributed:
      all_preds, all_gt = dist_utils.gather_predictions(local_preds, local_gt)
      dist_utils.synchronize()  # Ensure all ranks are done before continuing
    else:
      all_gt, all_preds = local_gt, local_preds

    # Log evaluation images
    if eval_images:
      self.results_logger.log_images(mode if mode else "eval", "images", eval_images)

    # Only rank 0 computes metrics
    metrics = None
    if self.is_main and all_preds is not None and all_gt is not None:
      validator = Validator(
        all_gt, all_preds, config=val_cfg, mask_batch_size=self.cfg.mask_batch_size
      )
      metrics = validator.compute_metrics(extended=extended)
      if path_to_save and mode:  # val and test
        validator.save_plots(path_to_save / "plots" / mode)

    # Synchronize before returning so all ranks wait for metrics computation
    if self.distributed:
      dist_utils.synchronize()
    return metrics

  def save_model(self, metrics: EvaluationMetrics, best_metric: float) -> float:
    model_to_save = self.model
    if self.ema_model:
      model_to_save = self.ema_model.model

    if isinstance(model_to_save, DDP):
      model_to_save = model_to_save.module

    self.cfg.paths.path_to_save.mkdir(parents=True, exist_ok=True)
    torch.save(model_to_save.state_dict(), self.cfg.paths.path_to_save / "last.pt")

    # mean from chosen metrics
    metrics_dict = metrics.to_dict()
    decision_metric = np.mean([metrics_dict[metric.value] for metric in self.decision_metrics])

    if decision_metric > best_metric:
      best_metric = decision_metric
      logger.info("Saving new best model🔥")
      torch.save(model_to_save.state_dict(), self.cfg.paths.path_to_save / "model.pt")
      self.early_stopping_steps = 0
    else:
      self.early_stopping_steps += 1
    return best_metric

  def train(self) -> None:
    best_metric = 0
    cur_iter = 0
    ema_iter = 0
    self.early_stopping_steps = 0
    one_epoch_time = None

    def optimizer_step():
      """
      Clip grads, optimizer step, zero grad, EMA model update
      """
      nonlocal ema_iter
      if self.cfg.amp_enabled:
        if self.cfg.clip_max_norm:
          self.scaler.unscale_(self.optimizer)
          torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_max_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()

      else:
        if self.cfg.clip_max_norm:
          torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_max_norm)
        self.optimizer.step()

      self.optimizer.zero_grad()

      if self.ema_model:
        ema_iter += 1
        self.ema_model.update(ema_iter, self.model)

    for epoch in range(1, self.cfg.epochs + 1):
      if self.is_main:
        self.results_logger.step(epoch)

      if self.distributed and self.train_sampler is not None:
        self.train_sampler.set_epoch(epoch)

      epoch_start_time = time.time()
      self.model.train()
      self.loss_fn.train()
      losses = []

      data_iter = self.train_loader
      if self.is_main:
        data_iter = tqdm(self.train_loader, unit="batch")

      for batch_idx, (inputs, targets, _) in enumerate(data_iter):
        if self.is_main:
          data_iter.set_description(f"Epoch {epoch}/{self.cfg.epochs}")

        if inputs is None:
          continue
        cur_iter += 1

        inputs = inputs.to(self.device)

        if batch_idx == 0 and self.cfg.visualize_loader is not None:
          from d_fine.dataset.loader_utils import log_debug_images_from_batch

          log_debug_images_from_batch(
            inputs,
            targets,
            self.train_loader.dataset,
            self.results_logger,
            num_images=self.cfg.visualize_loader,
          )
        targets = [
          {
            k: (v.to(self.device) if (v is not None and hasattr(v, "to")) else v)
            for k, v in t.items()
          }
          for t in targets
        ]

        lr = self.optimizer.param_groups[-1]["lr"]

        b_accum_steps = max(self.cfg.b_accum_steps, 1)
        if self.cfg.amp_enabled:
          with autocast(str(self.device), cache_enabled=True):
            output = self.model(inputs, targets=targets)
          with autocast(str(self.device), enabled=False):
            loss_dict = self.loss_fn(output, targets)
          loss = sum(loss_dict.values()) / b_accum_steps
          self.scaler.scale(loss).backward()

        else:
          output = self.model(inputs, targets=targets)
          loss_dict = self.loss_fn(output, targets)
          loss = sum(loss_dict.values()) / b_accum_steps
          loss.backward()

        if batch_idx == 0 and self.cfg.visualize_training is not None:
          orig_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).float().to(self.device)
          gt = self.gt_postprocess(inputs, targets, orig_sizes)
          preds = self.preds_postprocess(inputs, output, orig_sizes)
          train_vis_images = vis_utils.visualize_batch(
            inputs[: self.cfg.visualize_training],
            gt[: self.cfg.visualize_training],
            preds[: self.cfg.visualize_training],
            label_to_name=self.train_loader.dataset.label_to_name,
            dataset=self.train_loader.dataset,
          )
          self.results_logger.log_images("train", "predictions", train_vis_images)

        if (batch_idx + 1) % b_accum_steps == 0:
          optimizer_step()
          self.scheduler.step()

        losses.append(loss.item())

        if self.is_main:
          data_iter.set_postfix(
            loss=np.mean(losses) * b_accum_steps,
            eta=core_utils.calculate_remaining_time(
              one_epoch_time,
              epoch_start_time,
              epoch,
              self.cfg.epochs,
              cur_iter,
              len(self.train_loader),
            ),
            vram=f"{core_utils.get_vram_usage()}%",
          )

      # Final update for any leftover gradients from an incomplete accumulation step
      b_accum_steps = max(self.cfg.b_accum_steps, 1)
      if (batch_idx + 1) % b_accum_steps != 0:
        optimizer_step()

      self.results_logger.log_value("train", "lr", lr)

      # All ranks run evaluation (inference is distributed, metrics computed on rank 0)
      val_cfg = ValidationConfig(
        conf_threshold=self.cfg.conf_thresh,
        iou_threshold=self.cfg.iou_thresh,
        label_to_name=self.train_loader.dataset.label_to_name,
      )
      metrics = self.evaluate(
        val_loader=self.val_loader, val_cfg=val_cfg, extended=False, path_to_save=None
      )

      # Only rank 0 saves and logs
      if self.is_main:
        best_metric = self.save_model(metrics, best_metric)
        self.results_logger.log_values("val", metrics.to_dict())
        self.results_logger.log_value("val", "loss", np.mean(losses) * b_accum_steps)

      one_epoch_time = time.time() - epoch_start_time

      local_stop = False
      if (
        self.is_main
        and self.cfg.early_stopping
        and self.early_stopping_steps >= self.cfg.early_stopping
      ):
        local_stop = True

      if self.distributed:
        stop_flag = bool(int(dist_utils.broadcast_scalar(int(local_stop), src=0)))
      else:
        stop_flag = local_stop

      if stop_flag:
        if self.is_main:
          logger.info("Early stopping")
        break


def run_training(train_config: TrainConfig) -> None:
  """Run training with given TrainConfig.

  Args:
      train_config: Training configuration (includes project_name and exp_name)
  """
  ddp_enabled = train_config.ddp.enabled
  if ddp_enabled:
    dist_utils.init_distributed_mode()

  trainer = Trainer(train_config)

  try:
    t_start = time.time()
    trainer.train()
  except KeyboardInterrupt:
    if dist_utils.is_main_process():
      logger.warning("Interrupted by user")
  except Exception as e:
    if dist_utils.is_main_process():
      logger.error(e)
  finally:
    if dist_utils.is_main_process():
      logger.info("Evaluating best model...")
      # exp_name = core_utils.get_latest_experiment_name(train_config.exp_name, train_config.paths.path_to_save)

      _, val_loader_temp, _ = train_config.dataset.create_loader(
        batch_size=1, num_workers=0
      ).build_dataloaders(distributed=False)
      num_labels = val_loader_temp.dataset.num_classes

      model = dfine.build_model(
        train_config.model_name,
        num_labels,
        train_config.dataset.task == Task.SEGMENT,
        train_config.device.value,
        img_size=train_config.dataset.img_config.img_size,
      )
      checkpoint_path = train_config.paths.path_to_save / "model.pt"
      if checkpoint_path.exists():
        try:
          model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        except Exception as e:
          logger.warning(f"Could not load checkpoint from {checkpoint_path}: {e}")
          logger.info("Skipping checkpoint evaluation")
          return
      if trainer.ema_model:
        trainer.ema_model.model = model
      else:
        trainer.model = model

      # rebuild val and test loaders without DDP for evaluation
      if ddp_enabled:
        base_loader = train_config.dataset.create_loader(
          batch_size=train_config.batch_size, num_workers=train_config.num_workers
        )
        _, val_loader_eval, test_loader_eval = base_loader.build_dataloaders(distributed=False)
        trainer.val_loader = val_loader_eval
        trainer.test_loader = test_loader_eval
        trainer.distributed = False  # turn off DDP inside evaluate

      val_cfg = ValidationConfig(
        conf_threshold=trainer.cfg.conf_thresh,
        iou_threshold=trainer.cfg.iou_thresh,
        label_to_name=val_loader_temp.dataset.label_to_name,
      )
      val_metrics = trainer.evaluate(
        val_loader=trainer.val_loader,
        val_cfg=val_cfg,
        path_to_save=trainer.cfg.paths.path_to_save,
        extended=True,
        mode="val",
      )
      trainer.results_logger.step(train_config.epochs + 1)
      trainer.results_logger.log_values("val", val_metrics.to_dict())

      test_metrics = None
      if trainer.test_loader:
        test_metrics = trainer.evaluate(
          val_loader=trainer.test_loader,
          val_cfg=val_cfg,
          path_to_save=trainer.cfg.paths.path_to_save,
          extended=True,
          mode="test",
        )
        if test_metrics:
          trainer.results_logger.step(train_config.epochs + 1)
          trainer.results_logger.log_values("test", test_metrics.to_dict())

      logger.info(f"Full training time: {(time.time() - t_start) / 60 / 60:.2f} hours")

    if ddp_enabled:
      dist_utils.cleanup_distributed()
