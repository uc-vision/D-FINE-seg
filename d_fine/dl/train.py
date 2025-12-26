import math
import time
from copy import deepcopy
from pathlib import Path
from shutil import rmtree

import numpy as np
import torch
import wandb
import click
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf
from loguru import logger
from torch.amp import GradScaler, autocast
from torch.nn import SyncBatchNorm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from d_fine.core import dfine
from d_fine.core import dist_utils
from d_fine.config import DecisionMetric, Task, TrainConfig
from d_fine import utils as dl_utils
from d_fine.infer.utils import postprocess_ground_truth, postprocess_predictions
from d_fine.validator import Validator


class ModelEMA:
    def __init__(self, student, ema_momentum):
        # unwrap DDP if needed
        if isinstance(student, DDP):
            student = student.module
        self.model = deepcopy(student).eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.ema_scheduler = lambda x: ema_momentum * (1 - math.exp(-x / 2000))

    def update(self, iters, student):
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

        if self.is_main:
            self.init_dirs()

        enable_mask_head = self.cfg.dataset.task == Task.SEGMENT
        decision_metrics = list(self.cfg.decision_metrics)
        if enable_mask_head and DecisionMetric.F1 not in decision_metrics:
            decision_metrics.append(DecisionMetric.F1)

        if self.cfg.use_wandb and self.is_main:
            wandb.init(
                project=self.project_name,
                name=self.exp_name,
                config=self.cfg.model_dump(),
            )

        log_file = self.cfg.paths.path_to_save / "train_log.txt"
        if (not self.distributed) or self.is_main:
            log_file.unlink(missing_ok=True)
            logger.add(log_file, format="{message}", level="INFO", rotation="10 MB")

        logger.info(f"Experiment: {self.exp_name}, Task: {self.cfg.dataset.task.value}")
        seed = self.cfg.seed + self.rank if self.distributed else self.cfg.seed
        dl_utils.set_seeds(seed, self.cfg.cudnn_fixed)

        base_loader = self.cfg.dataset.create_loader(
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
        )
        self.train_loader, self.val_loader, self.test_loader = base_loader.build_dataloaders(
            distributed=self.distributed
        )
        self.train_sampler = getattr(base_loader, "train_sampler", None)
        
        # Save class config to separate file for inference
        if self.is_main:
            import json
            class_config = {
                "label_to_name": self.train_loader.dataset.label_to_name,
                "conf_thresh": self.cfg.conf_thresh,
                "iou_thresh": self.cfg.iou_thresh,
            }
            class_config_path = self.cfg.paths.path_to_save / "class_config.json"
            with open(class_config_path, "w") as f:
                json.dump(class_config, f, indent=2)
        if self.cfg.ignore_background_epochs:
            self.train_loader.dataset.ignore_background = True

        self.num_labels = self.train_loader.dataset.num_classes

        self.model = dfine.build_model(
            self.cfg.model_name,
            self.num_labels,
            enable_mask_head,
            str(self.device),
            img_size=self.cfg.dataset.img_config.img_size,
            pretrained_model_path=self.cfg.pretrained_model_path,
        )
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

        if self.cfg.use_wandb and self.is_main:
            wandb.watch(self.model)

    def init_dirs(self):
        for path in [self.cfg.dataset.debug_img_path, self.cfg.paths.infer_path]:
            if path is not None:
                if path.exists():
                    rmtree(path)
                path.mkdir(exist_ok=True, parents=True)

        self.cfg.paths.path_to_save.mkdir(exist_ok=True, parents=True)
        TrainConfig.save(self.cfg, self.cfg.paths.path_to_save / "config.json")

    def preds_postprocess(
        self,
        inputs: torch.Tensor,
        outputs: dict[str, torch.Tensor],
        orig_sizes: torch.Tensor,
    ) -> list[dict[str, torch.Tensor]]:
        """
        returns List with BS length. Each element is a dict {"labels", "boxes", "scores"}
        """
        model_config = self.cfg.get_model_config(self.num_labels)
        return postprocess_predictions(
            outputs=outputs,
            orig_sizes=orig_sizes,
            config=model_config,
            processed_size=tuple(inputs.shape[2:]),
            include_all_for_map=True,
        )

    def gt_postprocess(self, inputs, targets, orig_sizes):
        return postprocess_ground_truth(
            inputs=inputs,
            targets=targets,
            orig_sizes=orig_sizes,
            keep_ratio=self.cfg.dataset.keep_ratio,
        )

    @torch.no_grad()
    def get_preds_and_gt(
        self, val_loader: DataLoader
    ) -> tuple[list[dict[str, torch.Tensor]], list[dict[str, torch.Tensor]], list[np.ndarray] | None]:
        """
        Outputs gt and preds. Each is a List of dicts. 1 dict = 1 image.
        Returns (all_gt, all_preds, eval_images)
        """
        all_gt, all_preds = [], []
        eval_images = None
        model = self.ema_model.model if self.ema_model else self.model

        model.eval()
        with torch.inference_mode():
            for idx, (inputs, targets, img_paths) in enumerate(val_loader):
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

                if self.cfg.to_visualize_eval and idx <= 5:
                    eval_images = dl_utils.visualize(
                        img_paths,
                        gt,
                        preds,
                        dataset_path=self.cfg.dataset.data_path / "images",
                        path_to_save=self.cfg.paths.infer_path,
                        label_to_name=self.train_loader.dataset.label_to_name,
                    )

                for gt_instance, pred_instance in zip(gt, preds):
                    all_preds.append(dl_utils.encode_sample_masks_to_rle(pred_instance))
                    all_gt.append(dl_utils.encode_sample_masks_to_rle(gt_instance))

        return all_gt, all_preds, eval_images

    def evaluate(
        self,
        val_loader: DataLoader,
        conf_thresh: float,
        iou_thresh: float,
        path_to_save: Path,
        extended: bool,
        mode: str = None,
    ) -> dict[str, float]:
        # All ranks perform inference on their portion of the data
        local_gt, local_preds, eval_images = self.get_preds_and_gt(val_loader=val_loader)

        # Gather predictions from all ranks to rank 0
        if self.distributed:
            all_preds, all_gt = dist_utils.gather_predictions(local_preds, local_gt)
            dist_utils.synchronize()  # Ensure all ranks are done before continuing
        else:
            all_gt, all_preds = local_gt, local_preds

        # Log evaluation images to wandb
        if self.cfg.use_wandb and self.is_main and eval_images:
            wandb_images = [
                wandb.Image(img, caption=f"eval_{i}")
                for i, img in enumerate(eval_images)
            ]
            mode_prefix = f"{mode}/" if mode else ""
            wandb.log({f"{mode_prefix}eval_images": wandb_images})

        # Only rank 0 computes metrics
        metrics = None
        if self.is_main and all_preds is not None and all_gt is not None:
            validator = Validator(
                all_gt,
                all_preds,
                conf_thresh=conf_thresh,
                iou_thresh=iou_thresh,
                label_to_name=self.train_loader.dataset.label_to_name,
                mask_batch_size=self.cfg.mask_batch_size,
            )
            metrics = validator.compute_metrics(extended=extended)
            if path_to_save:  # val and test
                validator.save_plots(path_to_save / "plots" / mode)

        # Synchronize before returning so all ranks wait for metrics computation
        if self.distributed:
            dist_utils.synchronize()
        return metrics

    def save_model(self, metrics, best_metric):
        model_to_save = self.model
        if self.ema_model:
            model_to_save = self.ema_model.model

        if isinstance(model_to_save, DDP):
            model_to_save = model_to_save.module

        self.cfg.paths.path_to_save.mkdir(parents=True, exist_ok=True)
        torch.save(model_to_save.state_dict(), self.cfg.paths.path_to_save / "last.pt")

        # mean from chosen metrics
        decision_metric = np.mean([metrics[metric.value] for metric in self.cfg.decision_metrics])

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

        def optimizer_step(step_scheduler: bool):
            """
            Clip grads, optimizer step, scheduler step, zero grad, EMA model update
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

            if step_scheduler:
                self.scheduler.step()
            self.optimizer.zero_grad()

            if self.ema_model:
                ema_iter += 1
                self.ema_model.update(ema_iter, self.model)

        for epoch in range(1, self.cfg.epochs + 1):
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
                
                if self.cfg.use_wandb and self.is_main and batch_idx == 0:
                    from d_fine.dataset.loader_utils import log_debug_images_from_batch
                    log_debug_images_from_batch(
                        inputs, targets, self.train_loader.dataset, wandb, num_images=5
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

                if (batch_idx + 1) % b_accum_steps == 0:
                    optimizer_step(step_scheduler=True)

                losses.append(loss.item())

                if self.is_main:
                    data_iter.set_postfix(
                        loss=np.mean(losses) * b_accum_steps,
                        eta=dl_utils.calculate_remaining_time(
                            one_epoch_time,
                            epoch_start_time,
                            epoch,
                            self.cfg.epochs,
                            cur_iter,
                            len(self.train_loader),
                        ),
                        vram=f"{dl_utils.get_vram_usage()}%",
                    )

            # Final update for any leftover gradients from an incomplete accumulation step
            b_accum_steps = max(self.cfg.b_accum_steps, 1)
            if (batch_idx + 1) % b_accum_steps != 0:
                optimizer_step(step_scheduler=False)

            if self.cfg.use_wandb and self.is_main:
                wandb.log({"lr": lr, "epoch": epoch})

            # All ranks run evaluation (inference is distributed, metrics computed on rank 0)
            metrics = self.evaluate(
                val_loader=self.val_loader,
                conf_thresh=self.cfg.conf_thresh,
                iou_thresh=self.cfg.iou_thresh,
                extended=False,
                path_to_save=None,
            )

            # Only rank 0 saves and logs
            if self.is_main:
                best_metric = self.save_model(metrics, best_metric)
                dl_utils.save_metrics(
                    {},
                    metrics,
                    np.mean(losses) * b_accum_steps,
                    epoch,
                    path_to_save=None,
                    use_wandb=self.cfg.use_wandb,
                )

            if (
                epoch >= self.cfg.epochs - self.cfg.dataset.mosaic_augs.no_mosaic_epochs
                and self.train_loader.dataset.has_mosaic()
            ):
                self.train_loader.dataset.close_mosaic()

            if epoch == self.cfg.ignore_background_epochs:
                self.train_loader.dataset.ignore_background = False
                logger.info("Including background images")

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
            exp_name = dl_utils.get_latest_experiment_name(train_config.exp_name, train_config.paths.path_to_save)

            val_loader_temp = train_config.dataset.create_loader(
                batch_size=1,
                num_workers=0,
            ).build_dataloaders(distributed=False)[1]
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
                    model.load_state_dict(
                        torch.load(checkpoint_path, weights_only=True)
                    )
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
                    batch_size=train_config.batch_size,
                    num_workers=train_config.num_workers,
                )
                _, val_loader_eval, test_loader_eval = base_loader.build_dataloaders(
                    distributed=False
                )
                trainer.val_loader = val_loader_eval
                trainer.test_loader = test_loader_eval
                trainer.distributed = False  # turn off DDP inside evaluate

            val_metrics = trainer.evaluate(
                val_loader=trainer.val_loader,
                conf_thresh=trainer.cfg.conf_thresh,
                iou_thresh=trainer.cfg.iou_thresh,
                path_to_save=trainer.cfg.paths.path_to_save,
                extended=True,
                mode="val",
            )
            if train_config.use_wandb:
                dl_utils.wandb_logger(None, val_metrics, epoch=train_config.epochs + 1, mode="val")

            test_metrics = {}
            if trainer.test_loader:
                test_metrics = trainer.evaluate(
                    val_loader=trainer.test_loader,
                    conf_thresh=trainer.cfg.conf_thresh,
                    iou_thresh=trainer.cfg.iou_thresh,
                    path_to_save=trainer.cfg.paths.path_to_save,
                    extended=True,
                    mode="test",
                )
                if train_config.use_wandb:
                    dl_utils.wandb_logger(None, test_metrics, epoch=-1, mode="test")

            dl_utils.log_metrics_locally(
                all_metrics={"val": val_metrics, "test": test_metrics},
                path_to_save=train_config.paths.path_to_save,
                epoch=0,
                extended=True,
            )
            logger.info(f"Full training time: {(time.time() - t_start) / 60 / 60:.2f} hours")

        if ddp_enabled:
            dist_utils.cleanup_distributed()


@click.command()
@click.option("--project-name", required=True, help="Project name for wandb")
@click.option("--coco", type=click.Path(exists=True, path_type=Path), help="Path to COCO dataset (sets base path)")
@click.option("--yolo", type=click.Path(exists=True, path_type=Path), help="Path to YOLO dataset (sets base path)")
@click.argument("overrides", nargs=-1)
def main(project_name: str, coco: Path | None, yolo: Path | None, overrides: tuple[str, ...]) -> None:
    """Train D-FINE model.
    
    OVERRIDES: Additional Hydra config overrides (e.g., model_name=s task=segment)
    """
    # Validate that exactly one dataset path is provided
    if not coco and not yolo:
        raise click.BadParameter("Either --coco or --yolo must be provided")
    if coco and yolo:
        raise click.BadParameter("Cannot specify both --coco and --yolo")
    
    # Determine base path and dataset type from dataset path
    base_path = coco if coco else yolo
    dataset_type = "coco" if coco else "yolo"
    
    # Build overrides list - set train.base_path once, config files reference it
    override_list = [
        f"project_name={project_name}",
        f"train.base_path={base_path}",
        f"dataset={dataset_type}",
    ]
    
    override_list.extend(overrides)
    
    # Register custom resolvers
    OmegaConf.register_new_resolver("lookup", lambda d, key: d[key], replace=True)
    
    # Initialize Hydra and compose config with overrides
    import importlib.util
    from pathlib import Path
    _dfine_spec = importlib.util.find_spec("d_fine")
    config_path = Path(_dfine_spec.origin).parent / "config"
    with initialize(config_path=str(config_path), version_base=None):
        cfg = compose(config_name="config", overrides=override_list)
    
    # Instantiate TrainConfig recursively
    train_config = instantiate(
        cfg.train,
        _convert_="all",
        project_name=project_name,
        exp_name=cfg.exp,
    )
    run_training(train_config)


if __name__ == "__main__":
    main()
