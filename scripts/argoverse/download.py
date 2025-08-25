"""Download the Argoverse V2 dataset."""

import os

from tqdm import tqdm

from split import TEST, TRAIN, VAL


def download_argoverse_v2_sensor_dataset(
    data_root: str = "data/av2",
    dataset: str = "sensor",
    split: str = "train",
) -> None:
    """Download the Argoverse V2 Sensor dataset."""
    # Create the data directory.
    dataset_dir = os.path.join(data_root, dataset)
    os.makedirs(dataset_dir, exist_ok=True)

    url = f"s3://argoverse/datasets/av2/{dataset}"

    if split == "train":
        log_list = TRAIN
    elif split == "val":
        log_list = VAL
    elif split == "test":
        log_list = TEST
    else:
        raise ValueError(f"Invalid split: {split}")

    # Download the Argoverse V2 Sensor dataset.
    for log in tqdm(log_list):
        log_dir = os.path.join(dataset_dir, split, log)
        os.makedirs(log_dir, exist_ok=True)

        os.system(
            f"s5cmd --log error --no-sign-request cp '{url}/{split}/{log}/*' {log_dir}/"
        )


if __name__ == "__main__":
    """Download the Argoverse V2 Sensor dataset."""
    for split in [
        # "train",
        "val",
        # "test",
    ]:
        download_argoverse_v2_sensor_dataset(split=split)
