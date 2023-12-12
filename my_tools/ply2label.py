from tools.dataset_converters.ply import read_ply
import numpy as np
import glob
import os
from scipy.spatial import KDTree
import pdb


def upsample_label(test_points, predict_points, filename):
    save_path = filename.replace(".ply", ".label")
    tree = KDTree(predict_points[:, :3])
    _, idx = tree.query(test_points[:, :3])
    test_labels = predict_points[idx, -1].astype(np.int32)
    test_labels = test_labels.reshape(-1, 1)
    np.savetxt(save_path, np.c_[test_labels], fmt="%d", delimiter=" ")
    # with open(save_path, "wb") as f:
    #     f.write(test_labels)
    # pdb.set_trace()


if __name__ == "__main__":
    predict_ply_root = (
        "work_dirs/dual_kpfcnn_1xb6-cosine-100e_sensaturban-seg/20231203_103749"
    )
    test_ply_root = "/home/bisifu/bsf/code/mmdetection3d/data/SensatUrban/test"
    test_ply_paths = glob.glob(os.path.join(test_ply_root, "*.ply"))
    for test_ply_path in test_ply_paths:
        filename = test_ply_path.split("/")[-1]
        predict_ply_path = os.path.join(predict_ply_root, filename)
        if not os.path.exists(predict_ply_path):
            print(f"{predict_ply_path} is not exists !")
            exit()
        test_data = read_ply(test_ply_path)
        test_points = np.vstack((test_data["x"], test_data["y"], test_data["z"])).T
        predict_data = read_ply(predict_ply_path)
        predict_points = np.vstack(
            (
                predict_data["x"],
                predict_data["y"],
                predict_data["z"],
                predict_data["pred"],
            )
        ).T
        upsample_label(test_points, predict_points, predict_ply_path)
