import time
from pathlib import Path

import cv2
import click
import numpy as np
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm

from d_fine.config import TrainConfig
from d_fine.infer.torch_model import Torch_model


@click.command()
@click.option("--project-name", required=True, help="Project name")
@click.option("--base-path", required=True, type=click.Path(exists=True, path_type=Path), help="Base project path")
@click.option("--exp-name", type=str, help="Experiment name (defaults to latest)")
def main(project_name: str, base_path: Path, exp_name: str | None) -> None:
    """Test batching functionality."""
    try:
        train_config = TrainConfig.load_from_experiment(base_path, exp_name)
    except FileNotFoundError as e:
        raise click.BadParameter(str(e))
    
    loader = train_config.dataset.create_loader(
        batch_size=1, num_workers=0
    )
    _, val_loader_temp, _ = loader.build_dataloaders(distributed=False)
    
    torch_model = Torch_model.from_train_config(train_config)

    from d_fine import utils as dl_utils
    img_folder = train_config.dataset.data_path / "images"
    img = dl_utils.load_image(next(img_folder.iterdir()))

    res = {"bs": [], "throughput": [], "latency_per_image": []}
    images = 512
    bss = [1, 2, 4, 8, 16, 32]

    for bs in bss:
        if bs > 1:
            imgs = np.repeat(img[None, :, :, :], bs, axis=0)
        else:
            imgs = img

        t0 = time.perf_counter()
        for _ in tqdm(range(images // bs), desc=f"Batch size {bs}"):
            if bs > 1:
                _ = torch_model.predict_batch(imgs)
            else:
                _ = torch_model(imgs)
        t1 = time.perf_counter()

        latency_per_image = (t1 - t0) * 1000 / images
        throughput = images / (t1 - t0)

        res["bs"].append(bs)
        res["latency_per_image"].append(latency_per_image)
        res["throughput"].append(throughput)

    df = pd.DataFrame(res)
    df.to_csv(train_config.paths.path_to_save / "batched_infer.csv", index=False)

    tabulated_data = tabulate(df.round(1), headers="keys", tablefmt="pretty", showindex=False)
    print("\n" + tabulated_data)


if __name__ == "__main__":
    main()
