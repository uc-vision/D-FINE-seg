from pathlib import Path
from shutil import rmtree

import cv2
import click
from loguru import logger
from tqdm import tqdm
from d_fine.config import TrainConfig
from d_fine.infer.torch_model import Torch_model
from d_fine.validation.visualization import visualize
from d_fine import utils as dl_utils


@click.command()
@click.option("--project-name", required=True, help="Project name")
@click.option("--base-path", required=True, type=click.Path(exists=True, path_type=Path), help="Base project path")
@click.option("--exp-name", type=str, help="Experiment name (defaults to latest)")
def main(project_name: str, base_path: Path, exp_name: str | None) -> None:
    """Run inference on images or videos.
    
    Loads config from saved training experiment.
    """
    try:
        train_config = TrainConfig.load_from_experiment(base_path, exp_name)
    except FileNotFoundError as e:
        raise click.BadParameter(str(e))
    
    loader = train_config.dataset.create_loader(
        batch_size=1, num_workers=0
    )
    _, val_loader_temp, _ = loader.build_dataloaders(distributed=False)

    torch_model = Torch_model.from_train_config(train_config)

    data_path = train_config.dataset.path_to_test_data
    if not data_path.exists():
        logger.error(f"Test data path does not exist: {data_path}")
        return

    output_path = train_config.paths.infer_path
    if output_path.exists():
        rmtree(output_path)
    output_path.mkdir(parents=True)

    img_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    vid_extensions = {".mp4", ".avi", ".mov", ".mkv"}

    label_to_name = loader.label_to_name

    for f in tqdm(list(data_path.iterdir()), desc="Inference"):
        if f.name.startswith("."): continue
        
        if f.suffix.lower() in img_extensions:
            img = dl_utils.load_image(f)
            res = torch_model(img)
            vis_img = visualize(img, res, label_to_name)
            dl_utils.save_image(output_path / f.name, vis_img)
            
        elif f.suffix.lower() in vid_extensions:
            for frame_idx, frame in enumerate(dl_utils.load_video(f)):
                res = torch_model(frame)
                vis_img = visualize(frame, res, label_to_name)
                out_name = f"{f.stem}_frame_{frame_idx:06d}.jpg"
                dl_utils.save_image(output_path / out_name, vis_img)


if __name__ == "__main__":
    main()
