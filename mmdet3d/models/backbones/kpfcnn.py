from typing import List, Tuple, Optional
from mmengine.model import BaseModule
import torch
from torch import Tensor
from torch import nn as nn
import copy

from mmdet3d.registry import MODELS
from mmcv.ops.knn import knn
from mmcv.ops.ball_query import ball_query

# from mmcv.ops.ball_query import ball_query
from mmdet3d.utils import ConfigType
from ..layers.kpconv_modules.kpconv import KPConvBlock, KPResNetBlock


@MODELS.register_module()
class KPFCNNBackbone(BaseModule):
    """ """

    def __init__(
        self,
        num_point: int,
        in_channels: int,
        kernel_size: int,
        k_neighbor: int,
        sample_nums: List[int],
        kpconv_channels: List[List[int]],
        voxel_size: List[float] = None,
        radius: List[float] = None,
        weight_norm: bool = False,
        norm_cfg: ConfigType = dict(type="BN1d"),
        act_cfg: ConfigType = dict(type="LeakyReLU", negative_slope=0.1),
    ):
        super(KPFCNNBackbone, self).__init__()

        assert len(sample_nums) + 1 == len(kpconv_channels)
        if isinstance(radius, list) and isinstance(voxel_size, list):
            assert len(radius) == len(voxel_size) + 1 == len(kpconv_channels)

        if isinstance(kpconv_channels, tuple):
            kpconv_channels = list(map(list, kpconv_channels))

        if isinstance(num_point, int):
            self.num_point = [num_point]
        elif isinstance(num_point, list) or isinstance(num_point, tuple):
            self.num_point = num_point
        elif num_point is None:
            self.num_point = None
        else:
            raise NotImplementedError("Error type of num_point!")

        self.kpconvs = nn.ModuleList()
        self.sample_nums = sample_nums
        self.k_neighbor = k_neighbor
        self.kpconv_channels = kpconv_channels
        self.radius = radius
        self.voxel_size = voxel_size

        self.channel_list = copy.deepcopy(kpconv_channels)
        self.channel_list.insert(0, [in_channels])
        for i in range(1, len(self.channel_list)):
            for j in range(len(self.channel_list[i])):
                if j == 0:
                    self.kpconvs.append(
                        KPConvBlock(
                            kernel_size,
                            self.channel_list[i - 1][-1],
                            self.channel_list[i][j],
                            radius[i - 1] if radius else radius,
                            k_neighbor,
                            weight_norm,
                            norm_cfg=norm_cfg,
                            act_cfg=act_cfg,
                        )
                    )
                else:
                    self.kpconvs.append(
                        KPResNetBlock(
                            kernel_size,
                            self.channel_list[i][j - 1],
                            self.channel_list[i][j],
                            radius[i - 1] if radius else radius,
                            k_neighbor,
                            weight_norm,
                            strided=True
                            if j == len(self.channel_list[i]) - 1
                            else False,
                            norm_cfg=norm_cfg,
                            act_cfg=act_cfg,
                        )
                    )

    def _random_sample(self, points):
        permutation = torch.randperm(points.size(1))
        points_random = points[:, permutation, :]
        points_downsample = [points]
        for sample_num in self.sample_nums:
            points_downsample.append(points_random[:, :sample_num, :].contiguous())
        return points_downsample

    def _voxel_sample(self, points, voxel_size):
        # points = points.squeeze()
        boundary_min = torch.min(points, dim=0)[0]
        boundary_max = torch.max(points, dim=0)[0]
        voxel_nums = ((boundary_max - boundary_min) / voxel_size + 1).to(torch.int32)
        sampled_point_cloud = torch.zeros(
            (voxel_nums[0] * voxel_nums[1] * voxel_nums[2], 3)
        ).to("cuda")
        voxel_nums[2] = voxel_nums[0] * voxel_nums[1]
        voxel_nums[1] = voxel_nums[0]
        voxel_nums[0] = 1
        voxel_indices = ((points - boundary_min) / voxel_size).to(torch.int32)
        voxel_indices = torch.sum(voxel_indices * voxel_nums, dim=1)
        unique_indices = torch.unique(voxel_indices)
        sampled_point_cloud[voxel_indices] = points
        sampled_point_cloud = sampled_point_cloud[unique_indices]

        return sampled_point_cloud.contiguous()

    def _len2pre_sum(self, patch_len):
        patch_len_pre_sum = [0]
        for l in patch_len:
            patch_len_pre_sum.append(patch_len_pre_sum[-1] + l)
        return patch_len_pre_sum

    def _voxel_sample_batch(self, points, patch_len):
        device = points.device
        patch_len_pre_sum = self._len2pre_sum(patch_len)
        downsample_points_list = [points]
        downsample_len_list = [
            torch.tensor(patch_len, dtype=torch.int32, device=device)
        ]

        for voxel_size in self.voxel_size:
            downsample_len = []
            downsample_points = []
            for i in range(len(patch_len)):
                idx_left = patch_len_pre_sum[i]
                idx_right = patch_len_pre_sum[i + 1]
                downsample_points.append(
                    self._voxel_sample(points[idx_left:idx_right, :], voxel_size)
                )
                downsample_len.append(downsample_points[-1].shape[0])
            downsample_points_list.append(torch.cat(downsample_points, dim=0))
            downsample_len_list.append(
                torch.tensor(downsample_len, dtype=torch.int32, device=device)
            )
        return downsample_points_list, downsample_len_list

    def _knn_query(self, points):
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
                knn(self.k_neighbor, points[i], points[i], False)
                .transpose(1, 2)
                .contiguous()
            )
            if i != 0:
                idx_downsample.append(
                    knn(self.k_neighbor, points[i - 1], points[i], False)
                    .transpose(1, 2)
                    .contiguous()
                )
        return idx_self, idx_downsample

    def _ball_query(self, points, length, radius):
        """
        Args:
            points (List[Tensor,]): ((B,N,3),...)
            length (List[Tensor,]): ((l1,l2,...),(l1,l2,...))
            radius (List[int]): (r1,r2,...)
        Returns:
            idx_self (List[Tensor,]): ((B,npoint,k))
            idx_downsample (List[Tensor,]): ((B,npoint,k))
        """
        idx_self = []
        idx_downsample = []

        for i in range(len(points)):
            idx = ball_query(
                0,
                radius[i],
                self.k_neighbor,
                points[i],
                points[i],
                length[i],
                length[i],
            )

            idx_self.append(idx)
            if i != 0:
                idx = ball_query(
                    0,
                    radius[i],
                    self.k_neighbor,
                    points[i - 1],
                    points[i],
                    length[i - 1],
                    length[i],
                )
                idx_downsample.append(idx)

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

    def forward(self, inputs) -> Tuple[Tensor]:
        """
        Args:
            inputs (Tuple[Tensor]) or (List[Tensor]):
                - point (torch.Tensor): (B, N, 3)
                - feature (torch.Tensor): (B, N, C)
        Returns:
            Tuple[Tensor]:
                - featuremap ()
                - point
        """
        ## process patch

        isvoxel = False
        if isinstance(inputs, list):  # 如果inputs是list说明未进行拼接，是不等长的patch
            length = [x.shape[0] for x in inputs]
            inputs = torch.cat(inputs).unsqueeze(0)
            isvoxel = True

        points, features = self._split_point_feats(inputs)

        # down sample and query (rand or voxel / knn or ball)
        if isvoxel:
            points = points.squeeze(0)
            points_downsample, length_downsample = self._voxel_sample_batch(
                points, length
            )
            idx_self, idx_downsample = self._ball_query(
                points_downsample, length_downsample, self.radius
            )
            points_downsample = [x.unsqueeze(0) for x in points_downsample]
            idx_self = [x.unsqueeze(0) for x in idx_self]
            idx_downsample = [x.unsqueeze(0) for x in idx_downsample]
        else:
            points_downsample = self._random_sample(points)  # [0,1,2,3,4]
            idx_self, idx_downsample = self._knn_query(
                points_downsample
            )  # [0,1,2,3,4],[0,1,2,3]
            length_downsample = [None for _ in range(len(self.kpconv_channels))]

        layer_num = 0
        # features = torch.cat([points.transpose(1,2).contiguous(), features],dim = 1)
        feature_set = []
        for i in range(len(self.kpconv_channels)):
            for j in range(len(self.kpconv_channels[i])):
                if (
                    j != len(self.kpconv_channels[i]) - 1
                    or i == len(self.kpconv_channels) - 1
                ):  # 最后一层不进行下采样，其它层的最后一次卷积进行下采样
                    _, features, _ = self.kpconvs[layer_num](
                        points_downsample[i],
                        features,
                        points_downsample[i],
                        idx_self[i],
                        length_downsample[i],
                        length_downsample[i],
                    )
                else:
                    feature_set.append(features)
                    _, features, _ = self.kpconvs[layer_num](
                        points_downsample[i],
                        features,
                        points_downsample[i + 1],
                        idx_downsample[i],
                        length_downsample[i],
                        length_downsample[i + 1],
                    )
                layer_num += 1
        feature_set.append(features)
        out = dict(
            points=points_downsample, features=feature_set, length=length_downsample
        )
        return out
