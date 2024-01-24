import torch
import points_query
from my_tools.gather import gather


def knn(k, xyz, center_xyz, length=None, length_center=None):
    """
    Args:
        k (int): number of nearest neighbors.
        xyz (torch.Tensor): (B, N, 3)
        center_xyz (torch.Tensor, optional): (B, npoint, 3) if transposed
            is False, else (B, 3, npoint). centers of the knn query.
        length (List[Tensor,]): [N1,N2,...,Nn]
        length_center (List[Tensor,]): [M1,M2,...,Mm]
    Return:
        index: B, M, k or 1,M1+M2+...+Mm,k
        dist: B, M, k or 1,M1+M2+...+Mm,k
    """
    device = center_xyz.device

    if length is not None and length_center is not None:
        assert len(length) == len(length_center)
        cum_l = 0
        cum_cl = 0
        b, m, _ = center_xyz.shape
        index = torch.zeros((b, m, k), dtype=torch.long, device=device)
        dist = torch.zeros((b, m, k), dtype=torch.float32, device=device)
        for i in range(len(length)):
            points_query.knn_query(
                k,
                xyz[:, cum_l : cum_l + length[i].item(), :],
                center_xyz[:, cum_cl : cum_cl + length_center[i].item(), :],
                index[:, cum_cl : cum_cl + length_center[i].item(), :],
                dist[:, cum_cl : cum_cl + length_center[i].item(), :],
            )
            index[:, cum_cl : cum_cl + length_center[i].item(), :] += cum_l
            cum_l = cum_l + length[i].item()
            cum_cl = cum_cl + length_center[i].item()
    else:
        b, m, _ = center_xyz.shape
        index = torch.zeros((b, m, k), dtype=torch.long, device=device)
        dist = torch.zeros((b, m, k), dtype=torch.float32, device=device)
        points_query.knn_query(k, xyz, center_xyz, index, dist)

    return index


def mask_knn(
    k_neighbor,
    xyz,
    center_xyz,
    mask_xyz,
    mask_center_xyz,
    length=None,
    length_center=None,
):
    """
    Args:
        k (int): number of nearest neighbors.
        xyz (torch.Tensor): (B, N, 3)
        center_xyz (torch.Tensor, optional): (B, npoint, 3) if transposed
            is False, else (B, 3, npoint). centers of the knn query.
        mask_xyz (List[List[Tensor]]) : [[n1,n2,...],],
        mask_center_xyz (List[List[Tensor]]) : [[m1,m2,...],],
        length (List[Tensor,]): [N1,N2,...,Nn]
        length_center (List[Tensor,]): [M1,M2,...,Mm]
    Return:
        index: B, M, k or 1, M1+M2+...+Mm, k
        dist: B, M, k or 1, M1+M2+...+Mm, k
    """
    device = center_xyz.device
    if length is not None and length_center is not None:
        assert (
            len(length) == len(length_center) == len(mask_xyz) == len(mask_center_xyz)
        )
        cum_mask_xyz = [[0] for _ in range(len(mask_xyz))]
        cum_mask_center_xyz = [[0] for _ in range(len(mask_xyz))]
        for i in range(len(mask_xyz)):
            for j in range(len(mask_xyz[i])):
                cum_mask_xyz[i].append(cum_mask_xyz[i][-1] + mask_xyz[i][j])
                cum_mask_center_xyz[i].append(
                    cum_mask_center_xyz[i][-1] + mask_center_xyz[i][j]
                )

        cum_l = 0
        cum_cl = 0
        b, m, _ = center_xyz.shape

        # print(f"center_xyz:{m}")
        # print(f"mask_xyz:{mask_xyz}")
        # print(f"length:{length}")
        # print(f"mask_xyz")

        index = torch.zeros((b, m, 3 * k_neighbor + 1), dtype=torch.long, device=device)
        dist = torch.zeros(
            (b, m, 3 * k_neighbor + 1), dtype=torch.float32, device=device
        )
        for i in range(len(length)):
            for j in range(len(cum_mask_xyz[i]) - 1):
                ct_start = cum_mask_xyz[i][j]
                ct_end = cum_mask_xyz[i][j + 1]
                count = 0
                for k in range(len(cum_mask_xyz[i]) - 1):
                    k_tem = k_neighbor
                    xyz_start = cum_mask_center_xyz[i][k]
                    xyz_end = cum_mask_center_xyz[i][k + 1]
                    if j == k:
                        k_tem += 1
                    points_query.knn_query(
                        k_tem,
                        xyz[:, cum_l + xyz_start : cum_l + xyz_end, :],
                        center_xyz[:, cum_cl + ct_start : cum_cl + ct_end, :],
                        index[
                            :,
                            cum_cl + ct_start : cum_cl + ct_end,
                            count : count + k_tem,
                        ],
                        dist[
                            :,
                            cum_cl + ct_start : cum_cl + ct_end,
                            count : count + k_tem,
                        ],
                    )

                    # print(f"{cum_cl + ct_start}:{cum_cl + ct_end}")
                    # print(f"{count}:{count + k_neighbor}")
                    # print(f"{cum_l + xyz_start}")
                    # print(index.shape)
                    index[
                        :,
                        cum_cl + ct_start : cum_cl + ct_end,
                        count : count + k_tem,
                    ] += (
                        cum_l + xyz_start
                    )
                    if k_tem > k_neighbor:
                        index[
                            :,
                            cum_cl + ct_start : cum_cl + ct_end,
                            count,
                        ] = index[
                            :,
                            cum_cl + ct_start : cum_cl + ct_end,
                            count + k_tem - 1,
                        ]
                    count += k_neighbor
            cum_l = cum_l + length[i].item()
            cum_cl = cum_cl + length_center[i].item()
    else:
        raise NotImplementedError
    return index[:, :, :-1]


if __name__ == "__main__":
    import pdb
    import time

    # batch_knn
    b, m, n = 8, 8000, 8000
    k = 10
    center_xyz = torch.randn((b, n, 3), device="cuda:2")
    # center_xyz = torch.randn((b, m, 3))
    xyz = torch.randn((b, m, 3), device="cuda:2")
    # xyz = torch.randn((b, n, 3))
    t1 = time.time()
    index = knn(k, xyz, center_xyz)
    t2 = time.time()
    # xyz = torch.cat([xyz, torch.zeros_like(xyz[:, :1, :]) - 1e6], 1)
    t3 = time.time()
    neighbor_xyz = gather(xyz, index)
    t4 = time.time()
    print(t2 - t1)
    print(t4 - t3)
    # stack_knn
    xyz = torch.randn((m * b, 3), device="cuda:2").unsqueeze(0)
    center_xyz = torch.randn((n * b, 3), device="cuda:2").unsqueeze(0)
    length = torch.tensor([m] * b, device="cuda:2")
    length_center = torch.tensor([n] * b, device="cuda:2")
    t1 = time.time()
    index = knn(k, xyz, center_xyz, length=length, length_center=length_center)
    t2 = time.time()
    t3 = time.time()
    neighbor_xyz = gather(xyz, index)
    t4 = time.time()

    print(t2 - t1)
    print(t4 - t3)
    # pdb.set_trace()
