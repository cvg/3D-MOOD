# Data Preparation

## Omni3D

We provide the instruction to prepare Omni3D data and annotations.
We also provide the preprocessed HDF5 depth GT [here](https://huggingface.co/datasets/RoyYang0714/3D-MOOD).

```bash
cd data

wget https://dl.fbaipublicfiles.com/omni3d_data/Omni3D_json.zip
unzip Omni3D_json.zip

mkdir omni3d
mkdir omni3d/annotations
cp datasets/Omni3D/*.json omni3d/annotations/
```

The data folder will look like this:
```bash
data
└── omni3d
    └── annotations
        ├── XXX.json
```

### KITTI Object

Download the left color images from [KITTI's official website](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d).

Unzip or softlink the data to `data/KITTI_object` as:

```bash
data
└── KITTI_object
    └── training
        ├── image_2
```

The following is how to generate the depth GT for KITTI_Object.
Note some of the images can't find the corresponding depth.

1. Download the official dataset from [here](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction), including the raw data (about 200G) and fine-grained ground-truth depth maps. 

2. Put everything under `data/kitti_depth` folder. And make sure you already download the and prepare the Omni3D datasets.

3. Generate depth by running:
```bash
python scripts/kitti/gen_depth.py
```

The final data folder structure will be like:

```bash
data
├── omni3d
├── KITTI_object
├── kitti_depth
├── KITTI_object_depth
```

### NuScenes

1. Download the `trainval` and `CAN bus` from the [official nuScenes website](https://www.nuscenes.org/nuscenes#download).
Unzip or softlink the data into the root `data/nuscenes`.

2. Generate depth by running:
```bash
python scripts/gen_nusc_depth.py
```

The final data folder structure will be like:

```bash
data
├── omni3d
├── nuscenes
│   ├── samples
│   ├── trainval
│   ├── maps
│   ├── can_bus
├── nuscenes_depth
```

### Objectron

1. Get the data provided by Omni3D.

```bash
wget https://dl.fbaipublicfiles.com/omni3d_data/objectron_images.zip
unzip objectron_images.zip
```

Unzip or softlink the data to `data/objectron` as:

```bash
data
└── objectron
    └── train
        ├── XXX.jpg
```

2. We need different python env to extract depth for Objectron.

```bash
conda create -n objectron python=3.8 -y

conda activate objectron

pip install -r scripts/objectron/requirements.txt
```

3. Generate depth.

```bash
python scripts/objectron/generate_depth.py
conda deactivate
```

The scripts will download the original objectron data under `data/objectron_video` and dump the depth GT at `data/objectron_depth`.

The final data folder structure will be like:

```bash
data
├── omni3d
├── objectron
├── objectron_video
├── objectron_depth
```

### SUN RGB-D

Download the "SUNRGBD V1" images at SUN RGB-D's official website.
```bash
wget https://rgbd.cs.princeton.edu/data/SUNRGBD.zip
```

Unzip or softlink the data to `data/SUNRGBD`.

Since it already provide the depth GT so we don't need to extract on our own.

The final data folder structure will be like:

```bash
data
├── omni3d
├── SUNRGBD
│   ├── kv1
│   ├── kv2
│   ├── realsense
```

### ARKitScenes

1. Download the data provided by Omni3D.

```bash
wget https://dl.fbaipublicfiles.com/omni3d_data/ARKitScenes_images.zip
unzip ARKitScenes_images.zip
```

Unzip or softlink the data to `data/ARKitScenes` as:

```bash
data
└── ARKitScenes
    └── train
```

2. Generate depth.

```bash
python scripts/arkitscenes/gen_depth.py
```

This script will download `3dod` data under `data/ARKitScenes` and dump GT depth under `data/ARKitScenes_depth`.

The final data folder structure will be like:

```bash
data
├── omni3d
├── ARKitScenes
├── ARKitScenes_depth
```

### Hypersim

1. Clone the hypersim repo under `data` folder.

```bash
cd data

mkdir hypersim

git clone https://github.com/apple/ml-hypersim.git

cd ml-hypersim
```

2. Follow the [instruction](https://github.com/apple/ml-hypersim/tree/main/contrib/99991) and download the data.

```bash
mkdir data

# Image
python contrib/99991/download.py -c .tonemap.jpg -d ../hypersim --silent

# Depth
python contrib/99991/download.py -c .depth_meters.hdf5 -d ../hypersim --silent

cd ../..
```

3. Generate GT Depth.

```bash
python scripts/gen_hypersim_depth.py
```

The final data folder structure will be like:

```bash
data
├── omni3d
├── hypersim
├── hypersim_depth
```

## Argoverse 2

1. Download Argoverse 2 data.

```bash
python scripts/argoverse/download.py
```

It will download the sensor dataset under `data/av2`.

2. Convert dataset.

```bash
python scripts/argoverse/convert_av2.py
```

The final data folder structure will be like:

```bash
data
├── av2
├── argoverse
├── argoverse
│   ├── val
│   ├── annotations
```

- If you want to use HDF5 as data backend. You can run and get `val.hdf5`:
```bash
cd data/argoverse
python -m vis4d.data.io.to_hdf5 -p val
```

## ScanNet

Please refer [here](../scripts/scannet/README.md) for preprocessing ScanNet.
