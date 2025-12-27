# If image is absent in the path - remove from csv file. Also remove duplicated rows
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../../", config_name="config")
def main(cfg: DictConfig) -> None:
    data_path = Path(cfg.train.data_path) / "images"
    def exists(x):
        return (data_path / x).exists()
    
    for split in ["train", "val"]:
        df = pd.read_csv(Path(cfg.train.data_path) / f"{split}.csv", header=None)
        df = df[df[0].apply(exists)]
        df = df.drop_duplicates(keep="first")
        df.to_csv(Path(cfg.train.data_path) / f"{split}.csv", header=None, index=None)


if __name__ == "__main__":
    main()
