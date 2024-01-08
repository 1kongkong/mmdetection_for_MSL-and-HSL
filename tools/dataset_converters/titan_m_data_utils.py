import os
from concurrent import futures as futures
from os import path as osp

import mmengine
import numpy as np
from tools.dataset_converters.ply import read_ply


class TitanMData(object):
    """Titan data.

    Generate Titan infos for outdoor_converter.

    Args:
        root_path (str): Root path of the raw data.
        split (str, optional): Set split type of the data. Default: 'area_1'.
    """

    def __init__(self, root_path, split="area_1"):
        self.root_dir = root_path
        self.split = split
        self.data_dir = osp.join(root_path, "origin_data")

    def get_infos(self, has_label=True):
        """Get data infos.

        This method gets information from the raw data.

        Args:
            has_label (bool, optional): Whether the data has label.
                Default: True.

        Returns:
            infos (list[dict]): Information of the raw data.
        """
        path = osp.join(self.data_dir, self.split + ".ply")
        print(f"Start process {path}")
        info = dict()
        pc_info = {"num_features": 6, "lidar_idx": f"{self.split}"}
        info["point_cloud"] = pc_info
        pts_filename = osp.join(self.root_dir, "titan_data", f"{self.split}_point.npy")
        pts_semantic_mask_path = osp.join(
            self.root_dir, "titan_data", f"{self.split}_sem_label.npy"
        )
        mmengine.mkdir_or_exist(osp.join(self.root_dir, "titan_data"))

        cloud = read_ply(path)
        point = (
            np.vstack(
                (
                    cloud["x"],
                    cloud["y"],
                    cloud["z"],
                    cloud["c1"],
                    cloud["c2"],
                    cloud["c3"],
                    cloud["channel_num"],
                )
            )
            .astype(np.float32)
            .T
        )
        label = cloud["class"].astype(np.int64)
        np.save(pts_filename, point)
        np.save(pts_semantic_mask_path, label)
        print(f"Finish process {path}")

        path_pointdir = osp.join(self.root_dir, "points")
        path_labeldir = osp.join(self.root_dir, "semantic_mask")
        mmengine.mkdir_or_exist(path_pointdir)
        mmengine.mkdir_or_exist(path_labeldir)
        path_point_bin = osp.join(path_pointdir, f"{self.split}.bin")
        path_label_bin = osp.join(path_labeldir, f"{self.split}.bin")
        with open(path_point_bin, "wb") as f:
            f.write(point.tobytes())
        with open(path_label_bin, "wb") as f:
            f.write(label.tobytes())
        info["pts_path"] = osp.join("points", f"{self.split}.bin")
        info["pts_semantic_mask_path"] = osp.join("semantic_mask", f"{self.split}.bin")

        return [info]


class TitanMSegData(object):
    """Titan dataset used to generate infos for semantic segmentation task.

    Args:
        data_root (str): Root path of the raw data.
        ann_file (str): The generated scannet infos.
        split (str, optional): Set split type of the data. Default: 'train'.
        num_points (int, optional): Number of points in each data input.
            Default: 8192.
        label_weight_func (function, optional): Function to compute the
            label weight. Default: None.
    """

    def __init__(
        self, data_root, ann_file, num_points, split="Area_1", label_weight_func=None
    ):
        self.data_root = data_root
        self.data_infos = mmengine.load(ann_file)
        self.split = split
        self.num_points = num_points

        self.all_ids = np.arange(7)  # all possible ids
        self.cat_ids = np.array([0, 1, 2, 3, 4, 5, 6])  # used for seg task
        self.ignore_index = len(self.cat_ids)

        self.cat_id2class = (
            np.ones((self.all_ids.shape[0],), dtype=np.int64) * self.ignore_index
        )

        for i, cat_id in enumerate(self.cat_ids):
            self.cat_id2class[cat_id] = i

        self.label_weight_func = (
            (lambda x: 1.0 / np.log(1.2 + x))
            if label_weight_func is None
            else label_weight_func
        )

    def get_seg_infos(self):
        scene_idxs, label_weight = self.get_scene_idxs_and_label_weight()
        save_folder = osp.join(self.data_root, "seg_info")
        mmengine.mkdir_or_exist(save_folder)
        np.save(
            osp.join(save_folder, f"{self.split}_resampled_scene_idxs.npy"), scene_idxs
        )
        np.save(osp.join(save_folder, f"{self.split}_label_weight.npy"), label_weight)
        print(f"{self.split} resampled area index and label weight saved")

    def _convert_to_label(self, mask):
        """Convert class_id in loaded segmentation mask to label."""
        if isinstance(mask, str):
            if mask.endswith("npy"):
                mask = np.load(mask)
            else:
                mask = np.fromfile(mask, dtype=np.int64)
        label = self.cat_id2class[mask]
        return label

    def get_scene_idxs_and_label_weight(self):
        """Compute scene_idxs for data sampling and label weight for loss
        calculation.

        We sample more times for scenes with more points. Label_weight is
        inversely proportional to number of class points.
        """
        num_classes = len(self.cat_ids)  # used for seg task
        num_point_all = []
        label_weight = np.zeros((num_classes + 1,))  # ignore_index
        # for data_info in self.data_infos:
        for data_info in self.data_infos:
            label = self._convert_to_label(
                osp.join(self.data_root, data_info["pts_semantic_mask_path"])
            )
            num_point_all.append(label.shape[0])
            class_count, _ = np.histogram(label, range(num_classes + 2))
            label_weight += class_count

        sample_prob = np.array(num_point_all) / float(np.sum(num_point_all))
        num_iter = int(np.sum(num_point_all) / float(self.num_points) / 4)  # 适当降低采样次数
        scene_idxs = []
        for idx in range(len(self.data_infos)):
            scene_idxs.extend([idx] * int(round(sample_prob[idx] * num_iter)))
        scene_idxs = np.array(scene_idxs).astype(np.int32)

        # calculate label weight, adopted from PointNet++
        label_weight = label_weight[:-1].astype(np.float32)
        label_weight = label_weight / label_weight.sum()
        label_weight = self.label_weight_func(label_weight).astype(np.float32)

        return scene_idxs, label_weight
