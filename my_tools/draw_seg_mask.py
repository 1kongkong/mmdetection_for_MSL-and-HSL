import numpy as np
import glob
from mmdet3d.visualization import Det3DLocalVisualizer
from tools.dataset_converters.ply import read_ply
import pdb
import cv2
import time
import argparse

semseg_cmap_titan = (
    np.array(
        [
            [159, 159, 165],
            [175, 240, 0],
            [254, 0, 0],
            [0, 151, 0],
            [240, 139, 71],
            [10, 77, 252],
            [190, 0, 0],
            [181, 121, 11],
        ]
    )
    # / 255
)
semseg_cmap_sensat = (
    np.array(
        [
            [85, 107, 47],
            [0, 255, 0],
            [163, 148, 128],
            [41, 49, 101],
            [0, 0, 0],
            [0, 0, 255],
            [255, 0, 255],
            [255, 255, 10],
            [89, 47, 95],
            [213, 58, 116],
            [182, 67, 47],
            [0, 255, 255],
            [0, 191, 255],
        ]
    )
    # / 255
)

semseg_cmap_dict = {"titan": semseg_cmap_titan, "sensaturban": semseg_cmap_sensat}


def draw_seg_photo_acc(bev_map, points):
    for i in range(bev_map.shape[0]):
        t0 = time.time()
        points_sub = points[np.where(points[:, 1] == i)[0], :]
        t1 = time.time()
        for j in range(bev_map.shape[1]):
            t2 = time.time()
            points_grid = points_sub[np.where(points_sub[:, 0] == j)[0], :]
            t3 = time.time()
            if points_grid.shape[0] == 0:
                continue
            t4 = time.time()
            bev_map[i][j] = points_grid[np.argmax(points_grid[:, 2]), 3:6]
            t5 = time.time()
            # print("y:" + str(t1 - t0))
            # print("x:" + str(t3 - t2))
            # print("max:" + str(t5 - t4))
            # pdb.set_trace()
    return bev_map


def draw_seg_photo(points, grid_size, save_path):
    points[:, :3] = points[:, :3] - np.min(points[:, :3], axis=0)
    size = np.max(points[:, :2], axis=0) // grid_size + 1
    size = size.astype(np.int32)
    points[:, :2] = points[:, :2] // grid_size
    points[:, 1] = np.max(points[:, 1]) - points[:, 1]
    bev_map = np.ones([size[1], size[0], 3], dtype=np.float32) * 255
    t1 = time.time()
    bev_map = draw_seg_photo_acc(bev_map, points)
    t2 = time.time()
    print(t2 - t1)
    bev_map = cv2.cvtColor(bev_map, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, bev_map)


def voxel_sample(points, grid_size):
    idx = np.arange(points.shape[0])
    # voxelization with idx
    boundary_min = np.min(points, axis=0)
    boundary_max = np.max(points, axis=0)
    sample_voxel_size = [grid_size, grid_size, grid_size]
    voxel_nums = ((boundary_max - boundary_min) / sample_voxel_size + 1).astype(
        np.uint64
    )
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", dest="path", type=str, help="the path of model workdir"
    )
    args = parser.parse_args()

    paths = glob.glob(args.path + "/*.ply")
    for key in semseg_cmap_dict.keys():
        if key in args.path:
            semseg_cmap = semseg_cmap_dict[key]
            break
    for path in paths:
        print(path)
        data = read_ply(path)
        points = np.vstack([data["x"], data["y"], data["z"]]).T
        label = data["pred"]
        color = semseg_cmap[label]
        points_with_mask = np.concatenate((points, color), axis=1)
        t1 = time.time()
        choices = voxel_sample(points, 0.5)
        t2 = time.time()
        print(t2 - t1)
        save_path = path.replace(".ply", ".png")
        draw_seg_photo(points_with_mask[choices, :], 0.5, save_path)
        # pdb.set_trace()
