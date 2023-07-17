from typing import List, Tuple, Optional
from mmengine.model import BaseModule
import torch
from torch import Tensor
from torch import nn as nn
import copy

from mmdet3d.registry import MODELS
from mmcv.ops.knn import knn
# from mmcv.ops.ball_query import ball_query
from mmdet3d.utils import ConfigType
from ..layers.kpconv_modules.kpconv import KPConvBlock, KPResNetBlock


@MODELS.register_module()
class KPFCNNBackbone(BaseModule):
    '''
    '''

    def __init__(
            self,
            num_point: int,
            in_channels: int,
            kernel_size: int,
            k_neighbor: int,
            sample_nums: List[int],
            kpconv_channels: List[List[int]],
            weight_norm: bool = False,
            norm_cfg: ConfigType = dict(type='BN1d'),
            act_cfg: ConfigType = dict(type='LeakyReLU', negative_slope=0.1),
    ):
        super(KPFCNNBackbone, self).__init__()

        assert len(sample_nums) + 1 == len(kpconv_channels)

        if isinstance(kpconv_channels, tuple):
            kpconv_channels = list(map(list, kpconv_channels))

        if isinstance(num_point, int):
            self.num_point = [num_point]
        elif isinstance(num_point, list) or isinstance(num_point, tuple):
            self.num_point = num_point
        elif num_point is None:
            self.num_point = None
        else:
            raise NotImplementedError('Error type of num_point!')

        self.kpconvs = nn.ModuleList()
        self.sample_nums = sample_nums
        self.k_neighbor = k_neighbor
        self.kpconv_channels = kpconv_channels

        channel_list = copy.deepcopy(kpconv_channels)
        channel_list.insert(0, [in_channels])
        for i in range(1, len(channel_list)):
            for j in range(len(channel_list[i])):
                if j == 0:
                    self.kpconvs.append(
                        KPConvBlock(
                            kernel_size,
                            channel_list[i - 1][-1],
                            channel_list[i][j],
                            k_neighbor,
                            weight_norm,
                            norm_cfg=norm_cfg,
                            act_cfg=act_cfg))
                else:
                    self.kpconvs.append(
                        KPResNetBlock(
                            kernel_size,
                            channel_list[i][j - 1],
                            channel_list[i][j],
                            k_neighbor,
                            weight_norm,
                            strided=True if j == len(channel_list[i]) - 1 else False,
                            norm_cfg=norm_cfg,
                            act_cfg=act_cfg))

    def _random_sample(self, points):
        permutation = torch.randperm(points.size(1))
        points_random = points[:, permutation, :]
        points_downsample = [points]
        for sample_num in self.sample_nums:
            points_downsample.append(
                points_random[:, :sample_num, :].contiguous())
        return points_downsample

    def _query(self, points):
        """
        Args:
            points (List[Tensor,]): ((B,N,3),...)
        Returns:
            idx_self (List[Tensor,]): ((B,npoint,k))
            idx_downsample (List[Tensor,]): ((B,npoint,k))
        """
        idx_self = []
        idx_downsample = []
        for i in range(len(points)):
            idx_self.append(
                knn(self.k_neighbor, points[i], points[i], False).transpose(1, 2).contiguous())
            if i != 0:
                idx_downsample.append(
                    knn(self.k_neighbor, points[i - 1], points[i], False).transpose(1, 2).contiguous())
        return idx_self, idx_downsample

    def _split_point_feats(self, points: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """Split coordinates and features of input points.

        Args:
            points (torch.Tensor): Point coordinates with features,
                with shape (B, N, 3 + input_feature_dim).

        Returns:
            torch.Tensor: Coordinates of input points.
            torch.Tensor: Features of input points.
        """
        xyz = points[..., 0:3].contiguous()
        if points.size(-1) > 3:
            features = points[..., 3:].transpose(1, 2).contiguous()
        else:
            features = None

        return xyz, features

    def forward(self, inputs: Tuple[Tensor]) -> Tuple[Tensor]:
        """
        Args:
            inputs (Tuple[Tensor]):
                - point (torch.Tensor): (B, N, 3)
                - feature (torch.Tensor): (B, N, C)
        Returns:
            Tuple[Tensor]:
                - featuremap ()
                - point
        """
        points, features = self._split_point_feats(inputs)
        points_downsample = self._random_sample(points)  # [0,1,2,3,4]
        idx_self, idx_downsample = self._query(
            points_downsample)  # [0,1,2,3,4],[0,1,2,3]
        layer_num = 0
        # features = torch.cat([points.transpose(1,2).contiguous(), features],dim = 1)
        feature_set = []
        for i in range(len(self.kpconv_channels)):
            for j in range(len(self.kpconv_channels[i])):
                if j != len(self.kpconv_channels[i]) - 1 or i == len(self.kpconv_channels) - 1:  # 最后一层不进行下采样，其它层的最后一次卷积进行下采样
                    _, features, _ = self.kpconvs[layer_num](points_downsample[i], features, points_downsample[i],
                                                             idx_self[i])
                else:
                    feature_set.append(features)
                    _, features, _ = self.kpconvs[layer_num](points_downsample[i], features,
                                                             points_downsample[i + 1], idx_downsample[i])
                layer_num += 1
        feature_set.append(features)
        out = dict(points=points_downsample, features=feature_set)
        return out
