from pathlib import Path
import click
from pathlib import Path
from d_fine.config import TrainConfig
from d_fine.trainer import run_training


@click.command()
@click.argument("base-path", type=click.Path(exists=True, path_type=Path))
@click.option(
  "--preset",
  type=click.Path(exists=True, dir_okay=False, path_type=Path),
  help="Path to preset YAML",
)
@click.option("--project-name", default="d-fine", help="Project name for wandb")
@click.option("--exp-name", default="default", help="Experiment name")
@click.argument("overrides", nargs=-1)
def main(
  base_path: Path, preset: Path | None, project_name: str, exp_name: str, overrides: tuple[str, ...]
) -> None:
  """Train D-FINE model.

  BASE_PATH: Base dataset path.
  OVERRIDES: Additional Hydra config overrides (e.g., model_name=s task=segment)
  """
  # Load config using the unified from_hydra method
  train_config = TrainConfig.from_hydra(
    base_path=base_path,
    preset=preset,
    overrides=list(overrides),
    project_name=project_name,
    exp_name=exp_name,
  )

  run_training(train_config)


if __name__ == "__main__":
  main()
