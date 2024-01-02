import torch
from torch_points_kernels import ball_query as cuda_ball_query
from typing import Optional

from my_tools.gather import gather


def ball_query(
    radiu: float,
    sample_num: int,
    xyz: torch.Tensor,
    center_xyz: torch.Tensor,
    xyz_batch_cnt: Optional[torch.Tensor] = None,
    center_xyz_batch_cnt: Optional[torch.Tensor] = None,
):
    device = xyz.device
    # import time

    # t1 = time.time()
    if xyz_batch_cnt is not None and center_xyz_batch_cnt is not None:
        assert len(xyz_batch_cnt) == len(center_xyz_batch_cnt)
        batch_x = torch.zeros((xyz.shape[1]), dtype=torch.long, device=device)
        batch_y = torch.zeros((center_xyz.shape[1]), dtype=torch.long, device=device)
        cml_cnt1 = 0
        cml_cnt2 = 0
        for i in range(len(xyz_batch_cnt)):
            batch_x[cml_cnt1 : cml_cnt1 + xyz_batch_cnt[i].item()] = i
            batch_y[cml_cnt2 : cml_cnt2 + center_xyz_batch_cnt[i].item()] = i
            cml_cnt1 += xyz_batch_cnt[i].item()
            cml_cnt2 += center_xyz_batch_cnt[i].item()
        index, _ = cuda_ball_query(
            radiu,
            sample_num,
            xyz.squeeze(0).contiguous(),
            center_xyz.squeeze(0).contiguous(),
            "partial_dense",
            batch_x,
            batch_y,
        )

        index[index < 0] = xyz.shape[1]
        index = index.unsqueeze(0)
    else:
        index, _ = cuda_ball_query(radiu, sample_num, xyz, center_xyz)
        index[index < 0] = xyz.shape[1]
        # B, N, _ = xyz.shape
        # _, M, _ = center_xyz.shape
        # xyz = xyz.view(-1, 3).contiguous()
        # center_xyz = center_xyz.view(-1, 3).contiguous()
        # batch_x = torch.zeros((xyz.shape[0]), dtype=torch.long, device=device)
        # batch_y = torch.zeros((center_xyz.shape[0]), dtype=torch.long, device=device)
        # for i in range(B):
        #     batch_x[i * N : (i + 1) * N] = i
        #     batch_y[i * M : (i + 1) * M] = i
        # index, _ = cuda_ball_query(
        #     radiu, sample_num, xyz, center_xyz, "partial_dense", batch_x, batch_y
        # )
        # index = index.view(B, M, -1)
        # for i in range(B):
        #     index[i, ...] -= i * N
        # index[index < 0] = N
    # t2 = time.time()
    # print(f"ball_query:{t2 - t1}")
    return index.contiguous()


if __name__ == "__main__":
    import pdb
    import time

    # batch_knn
    b, m, n = 8, 8000, 8000
    k = 10

    center_xyz = torch.randn((b, n, 3), device="cuda:0")
    # center_xyz = torch.randn((b, m, 3))
    xyz = torch.randn((b, m, 3), device="cuda:0")
    # xyz = torch.randn((b, n, 3))
    t1 = time.time()
    index = ball_query(2, k, xyz, center_xyz)
    t2 = time.time()
    xyz = torch.cat([xyz, torch.zeros_like(xyz[:, :1, :]) - 1e6], 1)
    t3 = time.time()
    neighbor_xyz = gather(xyz, index)
    t4 = time.time()
    print(t2 - t1)
    print(t4 - t3)

    # stack_knn
    xyz = torch.randn((m * b, 3), device="cuda:0").unsqueeze(0)
    center_xyz = torch.randn((n * b, 3), device="cuda:0").unsqueeze(0)
    length = torch.tensor([m] * b, device="cuda:0")
    length_center = torch.tensor([n] * b, device="cuda:0")
    t1 = time.time()
    index = ball_query(2, k, xyz, center_xyz, length, length_center)
    t2 = time.time()
    xyz = torch.cat([xyz, torch.zeros_like(xyz[:, :1, :] - 1e6)], 1)
    t3 = time.time()
    neighbor_xyz = gather(xyz, index)
    t4 = time.time()

    print(t2 - t1)
    print(t4 - t3)
    # pdb.set_trace()
