"""Utility functions for image processing operations."""

from __future__ import annotations

import numpy as np
from matplotlib.pyplot import get_cmap
from PIL import Image
from vis4d.common.typing import (
    NDArrayBool,
    NDArrayF32,
    NDArrayUI8,
    NDArrayUI16,
)


def save_depth_map(
    depth_map: NDArrayF32, filename: str, depth_scale: float = 256.0
) -> None:
    """Dump depth map.

    Args:
        depth_map (NDArrayF32): Depth map to dump.
        filename (str): Path to dump depth map.
        depth_scale (float): Depth scale.
    """
    numpy_image = (depth_map * depth_scale).astype(np.uint16)
    numpy_image = colorize(numpy_image)
    Image.fromarray(numpy_image).save(filename)


def colorize(
    value: NDArrayUI16,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "magma_r",
) -> Image.Image:
    if value.ndim > 2:
        return value
    invalid_mask = value < 1e-3
    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    value = (value - vmin) / (vmax - vmin)  # vmin..vmax

    # set color
    cmapper = get_cmap(cmap)
    value = cmapper(value, bytes=True)  # (nxmx4)
    value[invalid_mask] = 0
    img = value[..., :3]
    return img


def get_pointcloud_from_rgbd(
    image: NDArrayUI8,
    depth: NDArrayF32,
    intrinsic_matrix: NDArrayF32,
    mask: NDArrayBool,
    remove_height: float | None = None,
) -> NDArrayF32:
    """Get pointcloud from RGBD image.

    Args:
        image (np.array): RGB image. Shape: (H, W, 3)
        depth (np.array): Depth image. Shape: (H, W)
        mask (np.ndarray): Mask of valid depth values. Shape: (H, W)
        intrinsic_matrix (np.array): Intrinsic matrix of camera. Shape: (3, 3)
        extrinsic_matrix (np.array, optional): Extrinsic matrix of camera.
            Shape: (4, 4). Defaults to None.
        voxelize (bool, optional): Whether to voxelize the pointcloud.

    Returns:
        NDArrayF32: Pointcloud. Shape: (N, 6)
    """
    # Mask the depth array
    masked_depth = np.ma.masked_where(mask == False, depth)

    # Create idx array
    idxs = np.indices(masked_depth.shape)
    u_idxs = idxs[1]
    v_idxs = idxs[0]

    # Get only non-masked depth and idxs
    z = masked_depth[~masked_depth.mask]
    compressed_u_idxs = u_idxs[~masked_depth.mask]
    compressed_v_idxs = v_idxs[~masked_depth.mask]
    image = np.stack(
        [image[..., i][~masked_depth.mask] for i in range(image.shape[-1])],
        axis=-1,
    )

    # Calculate local position of each point
    # Apply vectorized math to depth using compressed arrays
    cx = intrinsic_matrix[0, 2]
    fx = intrinsic_matrix[0, 0]
    x = (compressed_u_idxs - cx) * z / fx
    cy = intrinsic_matrix[1, 2]
    fy = intrinsic_matrix[1, 1]

    # Flip y as we want +y pointing up not down
    y = (compressed_v_idxs - cy) * z / fy

    # Remove height
    if remove_height is not None:
        mask = y >= remove_height
        x = x[mask]
        y = y[mask]
        z = z[mask]
        image = image[mask]
    else:
        x = x.reshape(-1)
        y = y.reshape(-1)
        z = z.reshape(-1)
        image = image.reshape(-1, 3)

    x_y_z_local = np.stack((x, y, z), axis=-1)

    return np.concatenate([x_y_z_local, image], axis=-1)


def save_file_ply(xyz: NDArrayF32, rgb: NDArrayF32, pc_file: str) -> None:
    """Save point cloud to ply file."""
    if rgb.max() < 1.001:
        rgb = rgb * 255.0
    rgb = rgb.astype(np.uint8)

    with open(pc_file, "w") as f:
        # headers
        f.writelines(
            [
                "ply\n" "format ascii 1.0\n",
                "element vertex {}\n".format(xyz.shape[0]),
                "property float x\n",
                "property float y\n",
                "property float z\n",
                "property uchar red\n",
                "property uchar green\n",
                "property uchar blue\n",
                "end_header\n",
            ]
        )

        for i in range(xyz.shape[0]):
            str_v = "{:10.6f} {:10.6f} {:10.6f} {:d} {:d} {:d}\n".format(
                xyz[i][0],
                xyz[i, 1],
                xyz[i, 2],
                rgb[i, 0],
                rgb[i, 1],
                rgb[i, 2],
            )
            f.write(str_v)
