from tools.dataset_converters.ply import read_ply
import numpy as np
import glob
import os

if __name__ == "__main__":
    root = "/home/bisifu/bsf/code/mmdetection3d/data/Titan_M/origin_data"
    paths = glob.glob(os.path.join(root, "*.ply"))
    for path in paths:
        print(path)
        data = read_ply(path)
        points = np.vstack([data["c1"], data["c2"], data["c3"]])
        print(np.max(points, axis=1))
        print(np.min(points, axis=1))
