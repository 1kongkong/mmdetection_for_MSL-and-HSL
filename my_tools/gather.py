import torch
import time


def gather(
    features: torch.Tensor, indices: torch.Tensor, transpose=False
) -> torch.Tensor:
    """
    Args:
            features (Tensor): Tensor of features to group, input shape is
                (B, N, C) or stacked inputs (N1 + N2 ..., C).
            indices (Tensor):  The indices of features to group with, input
                shape is (B, npoint, nsample) or stacked inputs
                (M1 + M2 ..., nsample).
            transpose (bool): features.shape == (B,C,N) ?
    Returns:
            Tensor: Grouped features, the shape is (B, C, npoint, nsample)
            or (M1 + M2 ..., C, nsample).
    """
    # t1 = time.time()
    if len(features.shape) == 2:
        N, C = features.shape
        M, K = indices.shape
        extended_idx = indices.unsqueeze(-1).expand(M, K, C)
        extended_feats = features.unsqueeze(-2).expand(N, K, C)
        neighbors_feats = torch.gather(extended_feats, 0, extended_idx)  # M K C
        neighbors_feats = neighbors_feats.permute(0, 2, 1).contiguous()

    elif len(features.shape) == 3:
        if transpose:
            features = features.permute(0, 2, 1).contiguous()
        B, N, C = features.shape
        B, M, K = indices.shape
        extended_idx = indices.unsqueeze(-1).expand(B, M, K, C)
        extended_feats = features.unsqueeze(-2).expand(B, N, K, C)
        neighbors_feats = torch.gather(extended_feats, 1, extended_idx)  # B M K C
        neighbors_feats = neighbors_feats.permute(0, 3, 1, 2).contiguous()
    else:
        raise NotImplementedError

    del features, extended_feats, extended_idx
    ## 测试不稳定耗时
    # torch.cuda.empty_cache()
    # t2 = time.time()
    # print(f'gather:{t2-t1}')
    return neighbors_feats  #  M C K or B C M K
