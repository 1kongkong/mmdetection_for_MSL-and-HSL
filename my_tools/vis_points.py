import open3d as o3d
import numpy as np
import glob
from tools.dataset_converters.ply import read_ply
import pdb
import matplotlib.pyplot as plt
from time import sleep
import cv2

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
    / 255
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
    / 255
)

semseg_cmap_dict = {"titan": semseg_cmap_titan, "sensaturban": semseg_cmap_sensat}

semseg_right_cmap = (
    np.array(
        [
            [255, 0, 0],
            [0, 255, 0],
        ]
    )
    / 255
)


def vis(point_cloud):
    # 创建 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    # 创建高度图参数
    # height_map = pcd.compute_point_cloud_distance()

    # 创建可视化窗口
    o3d.visualization.draw_geometries([pcd])


def point_visual(xyz, label, save_path):
    for key in semseg_cmap_dict.keys():
        if key in save_path:
            semseg_cmap = semseg_cmap_dict[key]
            break
    label = np.array(label[:, 0], dtype="int64")
    color = semseg_cmap[label, :]

    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(xyz)
    pcd_gt.colors = o3d.utility.Vector3dVector(color)
    # o3d.visualization.draw_geometries([pcd_gt])

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    render_option = vis.get_render_option()
    render_option.point_size = 1.5
    render_option.background_color = np.asarray([255, 255, 255])
    vis.add_geometry(pcd_gt)
    vis.update_geometry(pcd_gt)
    vis.poll_events()
    vis.update_renderer()
    pdb.set_trace()
    # vis.capture_screen_image(save_path)
    # vis.destroy_window()
    # color = vis.capture_screen_float_buffer(True)
    # color = np.asarray(color) * 255
    # color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
    # cv2.imwrite(save_path, color)


if __name__ == "__main__":
    root = "/home/bisifu/bsf/code/mmdetection3d/work_dirs/dual_kpfcnn_1xb6-cosine-100e_sensaturban-seg/20231203_103749"
    paths = glob.glob(root + "/*.ply")
    for path in paths:
        data = read_ply(path)
        points = np.vstack([data["x"], data["y"], data["z"], data["pred"]]).T
        points[:, :3] = points[:, :3] - np.min(points[:, :3], axis=0)
        save_path = path.replace(".ply", ".jpg")
        point_visual(points[:, :3], points[:, -1:], save_path)
        # pdb.set_trace()
