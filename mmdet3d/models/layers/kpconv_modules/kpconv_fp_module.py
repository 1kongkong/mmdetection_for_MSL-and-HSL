# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch
from mmcv.cnn import ConvModule
from mmcv.ops import three_interpolate, three_nn
from mmengine.model import BaseModule
from torch import Tensor
from torch import nn as nn

from mmdet3d.utils import ConfigType, OptMultiConfig

# from mmcv.ops.knn import knn
from mmcv.ops.group_points import grouping_operation
from my_tools.knn import knn
from my_tools.gather import gather


class KPConvFPModule(BaseModule):
    """Point feature propagation module used in PointNets.

    Propagate the features from one set to another.

    Args:
        mlp_channels (list[int]): List of mlp channels.
        norm_cfg (:obj:`ConfigDict` or dict): Config dict for normalization
            layer. Defaults to dict(type='BN2d').
        init_cfg (:obj:`ConfigDict` or dict or List[:obj:`Contigdict` or dict],
            optional): Initialization config dict. Defaults to None.
    """

    def __init__(
        self,
        mlp_channels: List[int],
        norm_cfg: ConfigType = dict(type="BN1d"),
        act_cfg: ConfigType = dict(type="LeakyReLU"),
        init_cfg: OptMultiConfig = None,
    ) -> None:
        super(KPConvFPModule, self).__init__(init_cfg=init_cfg)
        self.mlps = nn.Sequential()
        for i in range(len(mlp_channels) - 1):
            self.mlps.add_module(
                f"layer{i}",
                ConvModule(
                    mlp_channels[i],
                    mlp_channels[i + 1],
                    kernel_size=1,
                    stride=1,
                    conv_cfg=dict(type="Conv1d"),
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                ),
            )

    def _knn_query(self, target_points, source_points):
        """
        Args:
            target_points (List[Tensor,]): ((B,N,3),...)
            source_points (List[Tensor,]): ((B,M,3),...)
        Returns:
            idx (List[Tensor,]): ((B,N,k))
        """
        idx = knn(1, source_points, target_points, False).transpose(1, 2).contiguous()
        return idx

    def _len2pre_sum(self, patch_len):
        patch_len_pre_sum = [0]
        for l in patch_len:
            patch_len_pre_sum.append(patch_len_pre_sum[-1] + l)
        return patch_len_pre_sum

    def _batch_knn_query(self, target_points, source_points, target_len, source_len):
        """ """
        idx = knn(1, target_points, source_points, target_len, source_len)
        # idx = torch.zeros((1, target_points.shape[1], 1), dtype=torch.int32, device=target_points.device)
        # target_len_pre_sum = self._len2pre_sum(target_len)
        # source_len_pre_sum = self._len2pre_sum(source_len)
        # for i in range(len(target_len_pre_sum) - 1):
        #     t_start, t_end = target_len_pre_sum[i], target_len_pre_sum[i + 1]
        #     s_start, s_end = source_len_pre_sum[i], source_len_pre_sum[i + 1]
        #     idx[:, t_start:t_end, :] = self._knn_query(
        #         target_points[:, t_start:t_end, :], source_points[:, s_start:s_end, :]
        #     )
        return idx

    def forward(
        self,
        target: Tensor,
        source: Tensor,
        target_feats: Tensor,
        source_feats: Tensor,
        target_len: List[Tensor] = None,
        source_len: List[Tensor] = None,
    ) -> Tensor:
        """Forward.

        Args:
            target (Tensor): (B, n, 3) Tensor of the xyz positions of
                the target features.
            source (Tensor): (B, m, 3) Tensor of the xyz positions of
                the source features.
            target_feats (Tensor): (B, C1, n) Tensor of the features to be
                propagated to.
            source_feats (Tensor): (B, C2, m) Tensor of features
                to be propagated.

        Return:
            Tensor: (B, M, N) M = mlp[-1], Tensor of the target features.
        """
        idx = self._batch_knn_query(source, target, source_len, target_len)
        upsample_feats = gather(source_feats, idx, True)
        # if target_len is None and source_len is None:
        #     idx = self._batch_knn_query(target, source)
        #     upsample_feats = grouping_operation(source_feats, idx)
        # else:
        #     idx = self._batch_knn_query(target, source, target_len, source_len)
        #     upsample_feats = grouping_operation(
        #         source_feats.squeeze(0).permute(1, 0), idx.squeeze(0), source_len, target_len
        #     )
        #     upsample_feats = upsample_feats.permute(1, 0, 2).unsqueeze(0)
        upsample_feats = upsample_feats.squeeze(-1)
        upsample_feats = torch.cat([upsample_feats, target_feats], dim=1)

        return self.mlps(upsample_feats)

        # if source is not None:
        #     dist, idx = three_nn(target, source)
        #     dist_reciprocal = 1.0 / (dist + 1e-8)
        #     norm = torch.sum(dist_reciprocal, dim=2, keepdim=True)
        #     weight = dist_reciprocal / norm

        #     if source_feats.dtype == torch.float16:
        #         source_feats = source_feats.to(torch.float32)

        #     interpolated_feats = three_interpolate(source_feats, idx, weight)
        #     interpolated_feats.to(torch.float16)
        # else:
        #     interpolated_feats = source_feats.expand(*source_feats.size()[0:2], target.size(1))
        # if target_feats is not None:
        #     new_features = torch.cat([interpolated_feats, target_feats], dim=1)  # (B, C2 + C1, n)
        # else:
        #     new_features = interpolated_feats
        # # new_features = new_features.unsqueeze(-1)
        # new_features = self.mlps(new_features)

        # return new_features.squeeze(-1)
