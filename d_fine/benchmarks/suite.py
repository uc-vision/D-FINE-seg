import click
from pathlib import Path
from d_fine.config import TrainConfig
from d_fine.benchmarks import inference, batching
from d_fine.scripts import view_samples


def resolve_config(
  target: str, base_path: Path | None = None, overrides: list[str] | None = None
) -> TrainConfig:
  """Resolve target to a TrainConfig object."""
  target_path = Path(target)

  # If it's a directory containing config.json, it's an experiment
  if target_path.is_dir() and (target_path / "config.json").exists():
    return TrainConfig.load(target_path / "config.json")

  # Otherwise treat as generic overrides. If target is a file that doesn't exist, assume it's a preset name.
  final_overrides = list(overrides) if overrides else []
  if not target_path.exists():
    final_overrides.append(f"preset={target}")
  
  return TrainConfig.from_hydra(base_path=base_path, overrides=final_overrides)


@click.group()
def cli():
  """D-FINE Benchmark Suite"""
  pass


@cli.command()
@click.argument("target")
@click.option(
  "--base-path",
  type=click.Path(exists=True, path_type=Path),
  help="Base dataset path (overrides auto-inference)",
)
@click.option(
  "--ann-path",
  type=click.Path(exists=True, path_type=Path),
  help="Optional override for COCO annotations",
)
@click.option(
  "-o", "--override", multiple=True, help="Hydra overrides (e.g. -o train.batch_size=16)"
)
def run_inference(target, base_path, ann_path, override):
  """Run inference latency and accuracy benchmark.

  TARGET can be an experiment directory or a preset name/path.
  """
  config = resolve_config(target, base_path, list(override))
  inference.run_benchmark(config, ann_path=ann_path)


@cli.command()
@click.argument("target")
@click.option(
  "--base-path",
  type=click.Path(exists=True, path_type=Path),
  help="Base dataset path (overrides auto-inference)",
)
@click.option("-o", "--override", multiple=True, help="Hydra overrides")
def run_batching(target, base_path, override):
  """Run batching throughput benchmark.

  TARGET can be an experiment directory or a preset name/path.
  """
  config = resolve_config(target, base_path, list(override))
  batching.run_benchmark(config)


@cli.command()
@click.argument("target")
@click.option("--base-path", type=click.Path(exists=True, path_type=Path), help="Base dataset path")
@click.option("-o", "--override", multiple=True, help="Hydra overrides")
def view_samples(target, base_path, override):
  """Visualize training samples with augmentations.

  TARGET can be an experiment directory or a preset name/path.
  """
  config = resolve_config(target, base_path, list(override))
  view_samples.run_view_samples(config)


@cli.command()
@click.argument("target")
@click.option(
  "--base-path",
  type=click.Path(exists=True, path_type=Path),
  help="Base dataset path (overrides auto-inference)",
)
@click.option(
  "--ann-path",
  type=click.Path(exists=True, path_type=Path),
  help="Optional override for COCO annotations",
)
@click.option("-o", "--override", multiple=True, help="Hydra overrides")
@click.pass_context
def run_all(ctx, target, base_path, ann_path, override):
  """Run all benchmarks"""
  config = resolve_config(target, base_path, list(override))
  inference.run_benchmark(config, ann_path=ann_path)
  batching.run_benchmark(config)


if __name__ == "__main__":
  cli()
