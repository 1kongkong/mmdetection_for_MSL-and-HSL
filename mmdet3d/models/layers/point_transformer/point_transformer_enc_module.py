# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from torch import Tensor
from torch.nn import functional as F

from mmdet3d.utils import ConfigType
from mmcv.ops.group_points import grouping_operation
from mmcv.ops.furthest_point_sample import furthest_point_sample
from mmcv.ops.knn import knn
from .point_transformer import PointTranformerBlock


class PointTransfomerEncModule(nn.Module):
    """Base module for point set abstraction module used in PointNets.

    Args:
        num_point (int): Number of points.
        radii (List[float]): List of radius in each ball query.
        num_sample (List[int]): Number of samples in each ball query.
        mlp_channels (List[List[int]]): Specify of the pointnet before
            the global pooling for each scale.
        pool_mod (str): Type of pooling method. Defaults to 'max'.
    """

    def __init__(
        self,
        num_point: int,
        num_sample: List[int],
        channels: List[List[int]],
        is_head: bool = False,
        pool_mod: str = "max",
    ) -> None:
        super(PointTransfomerEncModule, self).__init__()

        assert pool_mod in ["max", "avg"]

        if isinstance(channels, tuple):
            channels = list(map(list, channels))
        self.channels = channels

        self.pool_mod = pool_mod
        self.TransitionDown = nn.ModuleList()
        self.point_transformers = nn.ModuleList()

        for i in range(len(channels) - 1):
            if i == 0:
                self.TransitionDown.append(
                    TransitionDown(
                        channels[i],
                        channels[i + 1],
                        num_point,
                        num_sample,
                        pool_mod,
                        is_head,
                    )
                )
            else:
                self.point_transformers.append(
                    PointTranformerBlock(
                        channels[i],
                        channels[i + 1],
                        share_planes=8,
                        num_sample=num_sample,
                    )
                )

    def forward(
        self,
        points_xyz: Tensor,
        features: Optional[Tensor] = None,
    ) -> Tuple[Tensor]:
        """Forward.

        Args:
            points_xyz (Tensor): (B, N, 3) xyz coordinates of the features.
            features (Tensor, optional): (B, C, N) Features of each point.
                Defaults to None.
            indices (Tensor, optional): (B, num_point) Index of the features.
                Defaults to None.
            target_xyz (Tensor, optional): (B, M, 3) New coords of the outputs.
                Defaults to None.

        Returns:
            Tuple[Tensor]:

                - new_xyz: (B, M, 3) Where M is the number of points.
                  New features xyz.
                - new_features: (B, M, sum_k(mlps[k][-1])) Where M is the
                  number of points. New feature descriptors.
                - indices: (B, M) Where M is the number of points.
                  Index of the features.
        """

        # sample points, (B, num_point, 3), (B, num_point)
        for layer in self.TransitionDown:
            new_xyz, features = layer(points_xyz, features)
        for layer in self.point_transformers:
            features = layer(new_xyz, features)

        return new_xyz, features


class TransitionDown(nn.Module):
    def __init__(
        self,
        input_dims,
        output_dims,
        num_point,
        num_sample=16,
        pool_mod="max",
        is_head=False,
    ):
        super(TransitionDown, self).__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.num_point = num_point
        self.num_sample = num_sample
        self.pool_mod = pool_mod
        self.is_head = is_head

        if self.is_head:
            self.layers = nn.Sequential(
                ConvModule(
                    input_dims,
                    output_dims,
                    1,
                    1,
                    bias=False,
                    conv_cfg=dict(type="Conv1d"),
                    norm_cfg=dict(type="BN1d"),
                    act_cfg=dict(type="ReLU"),
                )
            )

        else:
            self.layers = nn.Sequential(
                ConvModule(
                    3 + input_dims,
                    output_dims,
                    1,
                    1,
                    bias=False,
                    conv_cfg=dict(type="Conv2d"),
                    norm_cfg=dict(type="BN2d"),
                    act_cfg=dict(type="ReLU"),
                )
            )

    def _knn_query(self, points, target_points):
        """
        Args:
            points (Tensor): (B,N,3)
            target_points (Tensor): (B,M,3)
        Returns:
            idx (Tensor): (B,M,k)
        """
        idx = (
            knn(self.num_sample, points, target_points, False)
            .transpose(1, 2)
            .contiguous()
        )
        return idx

    def _pool_features(self, features: Tensor) -> Tensor:
        """Perform feature aggregation using pooling operation.

        Args:
            features (Tensor): (B, C, N, K) Features of locally grouped
                points before pooling.

        Returns:
            Tensor: (B, C, N) Pooled features aggregating local information.
        """
        if self.pool_mod == "max":
            # (B, C, N, 1)
            new_features = F.max_pool2d(features, kernel_size=[1, features.size(3)])
        elif self.pool_mod == "avg":
            # (B, C, N, 1)
            new_features = F.avg_pool2d(features, kernel_size=[1, features.size(3)])
        else:
            raise NotImplementedError

        return new_features.squeeze(-1).contiguous()

    def forward(self, xyz, feature):
        """
        Args:
            xyz (Tensor): [B,N,3]
            feature (Tensor): [B,C_in,N]
        Return:
            x (Tensor): [B,C_out,N]
            indices (Tensor): [B,N,k]
        """
        if self.is_head:
            new_xyz = xyz
        else:
            fps_idx = furthest_point_sample(xyz, self.num_point)
            new_xyz = grouping_operation(xyz.permute(0, 2, 1), fps_idx.unsqueeze(-1))
            new_xyz = new_xyz.squeeze(-1).permute(0, 2, 1).contiguous()

        indices = self._knn_query(xyz, new_xyz)

        if self.is_head:
            feature = self.layers(feature)
        else:
            feature = torch.cat([xyz.permute(0, 2, 1), feature], dim=1)
            feature = grouping_operation(feature, indices)
            feature = self.layers(feature)
            feature = self._pool_features(feature)

        return new_xyz, feature
