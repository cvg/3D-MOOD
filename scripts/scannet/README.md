# ScanNet v2

We follow the procedure in [mmdetection3d](https://github.com/open-mmlab/mmdetection3d).

- Download [ScanNet](https://github.com/ScanNet/ScanNet).
Link or move the `scans` under `data` folder.

- Extract RGB image with poses by running `python scripts/scannet/extract_posed_images.py --max-images-per-scene -1`.
Note that we extract all posed images for each ScanNet scenes, which require 2 TB disk space.

## ScanNet Settings

- Extract point clouds and annotations by running `python scripts/scannet/batch_load_scannet_data.py`.

- Generate ScanNet validation annotation and data folder.
```bash
python scripts/scannet/create_data.py
```

- If you want to use HDF5 as data backend. You can run and get `val.hdf5`:
```bash
cd data/scannet
python -m vis4d.data.io.to_hdf5 -p val
```

The directory structure after pre-processing should be as below

```bash
data
├── scannet
│   ├── meta_data
│   ├── scans
│   ├── scannet_instance_data
│   ├── val
│   ├── val.hdf5
│   ├── annotations
│   │   ├── ScanNet_val.json
│   ├── posed_images
│   │   ├── scenexxxx_xx
│   │   │   ├── xxxxxx.txt
│   │   │   ├── xxxxxx.jpg
│   │   │   ├── intrinsic.txt
```

## ScanNet200 Settings

- If you want to test `ScanNet200` settings. Please run this below:
```bash
python scripts/scannet/batch_load_scannet_data.py --output_folder data/scannet/scannet200_instance_data --scannet200
python scripts/scannet/create_data.py --scannet200
```

```bash
data
├── scannet
│   ├── meta_data
│   ├── scans
│   ├── scannet_instance_data
│   ├── scannet200_instance_data
│   ├── val
│   ├── val.hdf5
│   ├── annotations
│   │   ├── ScanNet_val.json
│   │   ├── ScanNet200_val.json
│   ├── posed_images
│   │   ├── scenexxxx_xx
│   │   │   ├── xxxxxx.txt
│   │   │   ├── xxxxxx.jpg
│   │   │   ├── intrinsic.txt
```
