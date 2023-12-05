import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from torch import Tensor
from torch.nn import functional as F

from mmdet3d.utils import ConfigType
from mmcv.ops.group_points import grouping_operation
from mmcv.ops.knn import knn
from mmcv.ops import three_interpolate, three_nn
from .point_transformer import PointTranformerBlock


class PointTransfomerDecModule(nn.Module):
    def __init__(self, channels, num_sample, is_head=False):
        super(PointTransfomerDecModule, self).__init__()
        self.channels = channels
        if isinstance(channels, tuple):
            channels = list(map(list, channels))
        self.channels = channels
        self.TransitionUp = nn.ModuleList()
        self.point_transformers = nn.ModuleList()
        for i in range(len(channels) - 1):
            if i == 0:
                self.TransitionUp.append(
                    TransitionUp(channels[i], channels[i + 1], is_head)
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

    def forward(self, xyz, feature, target_xyz=None, target_feature=None):
        """
        Args:
            xyz (tensor): [B,N,3]
            feature (tensor): [B,C1,N]
            target_xyz (tensor): [B,M,3]
            target_feature (tensor): [B,C2,M]
        Return:
            feature (tensor): [B,C3,M]
        """
        for layer in self.TransitionUp:
            feature = layer(xyz, feature, target_xyz, target_feature)
        for layer in self.point_transformers:
            if target_xyz is not None:
                feature = layer(target_xyz, feature)
            else:
                feature = layer(xyz, feature)
        return feature


class TransitionUp(nn.Module):
    def __init__(self, input_dims, output_dims, is_head=False):
        """
        Args:
            input_dims (int)
            output_dims (int)
            is_head (bool)
        """
        super(TransitionUp, self).__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.is_head = is_head
        if is_head:
            self.linear1 = ConvModule(
                input_dims,
                input_dims,
                1,
                1,
                bias=False,
                conv_cfg=dict(type="Conv1d"),
                act_cfg=dict(type="ReLU"),
            )
            self.linear2 = ConvModule(
                2 * input_dims,
                output_dims,
                1,
                1,
                bias=False,
                conv_cfg=dict(type="Conv1d"),
                norm_cfg=dict(type="BN1d"),
                act_cfg=dict(type="ReLU"),
            )
        else:
            self.linear1 = ConvModule(
                input_dims,
                output_dims,
                1,
                1,
                bias=False,
                conv_cfg=dict(type="Conv1d"),
                norm_cfg=dict(type="BN1d"),
                act_cfg=dict(type="ReLU"),
            )
            self.linear2 = ConvModule(
                output_dims,
                output_dims,
                1,
                1,
                bias=False,
                conv_cfg=dict(type="Conv1d"),
                norm_cfg=dict(type="BN1d"),
                act_cfg=dict(type="ReLU"),
            )

    def forward(self, xyz, feature, target_xyz=None, target_feature=None):
        """
        xyz (tensor): [B,N,3]
            feature (tensor): [B,C1,N]
            target_xyz (tensor): [B,M,3]
            target_feature (tensor): [B,C2,M]
        Return:
            feature (tensor): [B,C3,M]
        """

        if self.is_head:
            N = feature.shape[2]
            feature_mean = torch.mean(feature, dim=2, keepdim=True)
            feature_mean = self.linear1(feature_mean)
            feature = torch.cat([feature, feature_mean.repeat(1, 1, N)], dim=1)
            feature = self.linear2(feature)
        else:
            dist, idx = three_nn(target_xyz, xyz)
            dist_reciprocal = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_reciprocal, dim=2, keepdim=True)
            weight = dist_reciprocal / norm
            ## 适配AMP
            feature = self.linear1(feature)
            if feature.dtype == torch.float16:
                feature = feature.to(torch.float32)
                interpolated_feats = three_interpolate(feature, idx, weight)
                interpolated_feats = interpolated_feats.to(torch.float16)
            else:
                interpolated_feats = three_interpolate(feature, idx, weight)

            feature = self.linear2(target_feature) + interpolated_feats
        return feature
