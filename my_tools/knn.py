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
