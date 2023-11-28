import os

import mmengine
import numpy as np
import glob

from tools.dataset_converters.titan_data_utils import TitanData, TitanSegData
from tools.dataset_converters.hsl_data_utils import HSLData, HSLSegData
from tools.dataset_converters.sensaturban_data_utils import SensatUrbanData, SensatUrbanSegData


def create_outdoor_info_file(
    data_path,
    pkl_prefix="titan",
    save_path=None,
    use_v1=False,
):
    """Create outdoor/airborne lidar information file.

    Get information of the raw data and save it to the pkl file.

    Args:
        data_path (str): Path of the data.
        pkl_prefix (str, optional): Prefix of the pkl to be saved.
            Default: 'titan'.
        save_path (str, optional): Path of the pkl to be saved. Default: None.
        use_v1 (bool, optional): Whether to use v1. Default: False.
        workers (int, optional): Number of threads to be used. Default: 4.
    """
    assert os.path.exists(data_path)
    assert pkl_prefix in ["titan", "sensaturban", "hsl"], f"unsupported outdoor dataset {pkl_prefix}"
    save_path = data_path if save_path is None else save_path
    assert os.path.exists(save_path)

    # generate infos for segmentation task
    if pkl_prefix == "titan":
        splits = [f"area_{i}" for i in range(1, 33)]
        for split in splits:
            dataset = TitanData(root_path=data_path, split=split)
            info = dataset.get_infos()
            filename = os.path.join(save_path, f"{pkl_prefix}_infos_{split}.pkl")
            mmengine.dump(info, filename, "pkl")
            print(f"{pkl_prefix} info {split} file is saved to {filename}")
            seg_dataset = TitanSegData(
                data_root=data_path,
                ann_file=filename,
                num_points=8192,
                split=split,
                label_weight_func=lambda x: 1.0 / np.log(1.2 + x),
            )
            seg_dataset.get_seg_infos()
    elif pkl_prefix == "HSL" or pkl_prefix == "hsl":
        splits = [f"area_{i}" for i in range(1, 13)]
        for split in splits:
            dataset = HSLData(root_path=data_path, split=split)
            info = dataset.get_infos()
            filename = os.path.join(save_path, f"{pkl_prefix}_infos_{split}.pkl")
            mmengine.dump(info, filename, "pkl")
            print(f"{pkl_prefix} info {split} file is saved to {filename}")
            seg_dataset = HSLSegData(
                data_root=data_path,
                ann_file=filename,
                num_points=50000,
                split=split,
                label_weight_func=lambda x: 1.0 / np.log(1.2 + x),
            )
            seg_dataset.get_seg_infos()
    elif pkl_prefix == "sensaturban":
        all_paths = glob.glob(os.path.join(data_path, "origin_data", "*.ply"))
        splits = [path.split("/")[-1].split('.')[0] for path in all_paths]
        for split in splits:
            dataset = SensatUrbanData(root_path=data_path, split=split)
            info = dataset.get_infos()
            filename = os.path.join(save_path, f"{pkl_prefix}_infos_{split}.pkl")
            mmengine.dump(info, filename, "pkl")
            print(f"{pkl_prefix} info {split} file is saved to {filename}")
            seg_dataset = SensatUrbanSegData(
                data_root=data_path,
                ann_file=filename,
                num_points=8192,
                split=split,
                label_weight_func=lambda x: 1.0 / np.log(1.2 + x),
            )
            seg_dataset.get_seg_infos()
    else:
        raise NotImplementedError(f"Don't support {pkl_prefix} dataset.")
