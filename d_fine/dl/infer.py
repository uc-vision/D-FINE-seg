from pathlib import Path
from shutil import rmtree

import click

from d_fine.config import TrainConfig
from d_fine.infer.torch_model import Torch_model
from d_fine.validation.visualization import figure_input_type, run_images, run_videos


@click.command()
@click.option("--project-name", required=True, help="Project name")
@click.option("--base-path", required=True, type=click.Path(exists=True, path_type=Path), help="Base project path")
@click.option("--exp-name", type=str, help="Experiment name (defaults to latest)")
def main(project_name: str, base_path: Path, exp_name: str | None) -> None:
    """Run inference on images or videos.
    
    Loads config from saved training experiment.
    """
    # Load saved config using helper method
    try:
        train_config = TrainConfig.load_from_experiment(base_path, exp_name)
    except FileNotFoundError as e:
        raise click.BadParameter(str(e))
    
    temp_loader = train_config.dataset.create_loader(
        batch_size=1, num_workers=0
    )
    _, val_loader_temp, _ = temp_loader.build_dataloaders(distributed=False)

    model_config = train_config.get_model_config(val_loader_temp.dataset.num_classes)
    torch_model = Torch_model(
        train_config=train_config,
        model_config=model_config,
    )

    data_type = figure_input_type(train_config.dataset.path_to_test_data)

    if train_config.paths.infer_path.exists():
        rmtree(train_config.paths.infer_path)

    run_fn = run_images if data_type == "image" else run_videos
    run_fn(
        torch_model,
        train_config.dataset.path_to_test_data,
        train_config.paths.infer_path,
        label_to_name=val_loader_temp.dataset.label_to_name,
        to_crop=train_config.infer.to_crop,
        paddings=train_config.infer.paddings,
        conf_thresh=train_config.conf_thresh,
    )


if __name__ == "__main__":
    main()
