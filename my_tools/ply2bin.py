import glob
import os
import numpy as np
import pdb
from tools.dataset_converters.ply import read_ply


def voxel_sample(points, grid_size):
    idx = np.arange(points.shape[0])
    # voxelization with idx
    boundary_min = np.min(points, axis=0)
    boundary_max = np.max(points, axis=0)
    sample_voxel_size = [grid_size, grid_size, grid_size]
    voxel_nums = ((boundary_max - boundary_min) / sample_voxel_size + 1).astype(
        np.uint64
    )
    print(voxel_nums)
    print(voxel_nums[0] * voxel_nums[1] * voxel_nums[2])
    choices = np.zeros((voxel_nums[0] * voxel_nums[1] * voxel_nums[2]))
    voxel_nums[2] = voxel_nums[0] * voxel_nums[1]
    voxel_nums[1] = voxel_nums[0]
    voxel_nums[0] = 1
    voxel_indices = ((points - boundary_min) / sample_voxel_size).astype(np.uint64)
    # get idx
    voxel_indices = np.sum(voxel_indices * voxel_nums, axis=1)
    # down sample
    unique_indices = np.unique(voxel_indices)
    choices[voxel_indices] = idx
    choices = choices[unique_indices].astype(np.uint64)
    return choices


if __name__ == "__main__":
    root = (
        "/home/bisifu/bsf/code/mmdetection3d/data/mmdetection3d_data/SensatUrban/test"
    )
    paths = glob.glob(os.path.join(root, "*.ply"))
    for path in paths:
        print(path)
        data = read_ply(path)
        points = np.vstack(
            (data["x"], data["y"], data["z"], data["red"], data["green"], data["blue"])
        ).T
        points[:, 3:] /= 255
        idx = voxel_sample(points[:, :3], 0.15)
        # pdb.set_trace()
        save_path = path.replace(".ply", ".bin")
        with open(save_path, "wb") as f:
            f.write(points[idx, :].tobytes())
