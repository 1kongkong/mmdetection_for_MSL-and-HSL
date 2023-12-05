from mmcv.ops.group_points import grouping_operation
import numpy as np
import torch
import pdb

if __name__ == "__main__":
    point_num = 100000
    point = np.random.random((point_num, 3))
    idx = np.arange(point_num // 10).reshape(-1, 1)
    idx = np.hstack([idx for _ in range(20)])
    batch_size = 6
    points = np.vstack([point for _ in range(batch_size)])
    idxs = np.vstack([idx for _ in range(batch_size)])
    batch_points = [point_num for _ in range(batch_size)]
    batch_idxs = [point_num // 10 for _ in range(batch_size)]

    # to tensor of gpu
    points = torch.tensor(points, dtype=torch.float32, device="cuda:0")
    idxs = torch.tensor(idxs, dtype=torch.int32, device="cuda:0")
    batch_points = torch.tensor(batch_points, dtype=torch.int32, device="cuda:0")
    batch_idxs = torch.tensor(batch_idxs, dtype=torch.int32, device="cuda:0")
    # points = points.unsqueeze(0)
    # idxs = idxs.unsqueeze(0)
    neighbors = grouping_operation(points, idxs, batch_points, batch_idxs)
    # neighbors = grouping_operation(points, idxs)
    pdb.set_trace()
