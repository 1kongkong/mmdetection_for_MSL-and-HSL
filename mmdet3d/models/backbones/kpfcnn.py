from typing import List, Tuple, Optional
from mmengine.model import BaseModule
import torch
from torch import Tensor
from torch import nn as nn
import copy

from mmdet3d.registry import MODELS

# from mmcv.ops.knn import knn
from my_tools.knn import knn
from my_tools.ball_query import ball_query

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
        kpconv_channels: List[List[int]],
        spa_used: bool = False,
        sample_method: str = "rand",  # {'rand','grid','grid+rand'}
        query_method: str = "knn",  # {'knn','ball'}
        voxel_size: List[float] = None,
        radius: List[float] = None,
        weight_norm: bool = False,
        norm_cfg: ConfigType = dict(type="BN1d"),
        act_cfg: ConfigType = dict(type="ReLU"),
    ):
        super(KPFCNNBackbone, self).__init__()

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
        self.k_neighbor = k_neighbor
        self.kpconv_channels = kpconv_channels
        self.radius = radius
        self.voxel_size = voxel_size
        self.sample_method = sample_method
        self.query_method = query_method
        self.spa_used = spa_used
        # 可选的下采样方法(首次下采样在dataloader pipline, 模型里均为后续下采样)
        if "rand" in sample_method:
            self.sample = self._random_sample_batch
        else:
            self.sample = self._voxel_sample_batch
        # 可选的近邻搜索方法
        if query_method == "knn":
            self.query = self._knn_query
        elif query_method == "ball":
            self.query = self._ball_query
        else:
            raise NotImplementedError

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
                            strided=(
                                True if j == len(self.channel_list[i]) - 1 else False
                            ),
                            norm_cfg=norm_cfg,
                            act_cfg=act_cfg,
                        )
                    )

    def _random_sample_batch(self, points):
        """
        Args:
            points (Tensor) : (B,N,3)
        Return:
            points_downsample (List(Tensor,)) : [(B,N,3),..]
            downsample_len_list (List(Tensor,)) : [(n1,n2,...),...]
        """
        if "grid" in self.sample_method:
            device = points.device
            patch_len_pre_sum = self._len2pre_sum(self.patch_len)
            points_downsample = [points]

            downsample_len_list = [
                torch.tensor(self.patch_len, dtype=torch.int32, device=device)
            ]
            for i in range(len(self.kpconv_channels) - 1):
                downsample_len_list.append(downsample_len_list[-1] // 4)
                points_downsample.append([])

            for i in range(len(self.patch_len)):
                idx_left = patch_len_pre_sum[i]
                idx_right = patch_len_pre_sum[i + 1]
                sub_points = points[:, idx_left:idx_right, :]
                permutation = torch.randperm(sub_points.size(1))
                sub_points = sub_points[:, permutation, :]
                for j in range(1, len(downsample_len_list)):
                    sample_num = downsample_len_list[j][i].item()
                    points_downsample[j].append(sub_points[:, :sample_num, :])

            for i in range(1, len(points_downsample)):
                points_downsample[i] = torch.cat(points_downsample[i], dim=1)

        else:
            permutation = torch.randperm(points.size(1))
            points_random = points[:, permutation, :]
            points_downsample = [points]
            sample_num = points.shape[1]
            for i in range(len(self.kpconv_channels) - 1):
                sample_num //= 4
                points_downsample.append(points_random[:, :sample_num, :].contiguous())
            downsample_len_list = [None for _ in range(len(self.kpconv_channels))]

        return points_downsample, downsample_len_list

    def _voxel_sample_batch(self, points):
        """ "
        Args:
            points (Tensor) : (B,N,3)
        Return:
            points_downsample (List(Tensor,)) : [(B,N,3),..]
            downsample_len_list (List(Tensor,)) : [(n1,n2,...),...]
        """
        device = points.device
        patch_len_pre_sum = self._len2pre_sum(self.patch_len)
        downsample_points_list = [points]
        downsample_len_list = [
            torch.tensor(self.patch_len, dtype=torch.int32, device=device)
        ]

        for voxel_size in self.voxel_size:
            downsample_len = []
            downsample_points = []
            for i in range(len(self.patch_len)):
                idx_left = patch_len_pre_sum[i]
                idx_right = patch_len_pre_sum[i + 1]
                downsample_points.append(
                    self._voxel_sample(
                        points[:, idx_left:idx_right, :].squeeze(0), voxel_size
                    )
                )
                downsample_len.append(downsample_points[-1].shape[1])

            downsample_points_list.append(torch.cat(downsample_points, dim=1))
            downsample_len_list.append(
                torch.tensor(downsample_len, dtype=torch.int32, device=device)
            )

        return downsample_points_list, downsample_len_list

    def _voxel_sample(self, points, voxel_size):
        """ "
        Args:
            points (Tensor) : (N,3)
            voxel_size (Float):
        Return:
            sampled_point_cloud (List(Tensor,)) : [(1,N,3),..]
        """
        device = points.device
        boundary_min = torch.min(points, dim=0)[0]
        boundary_max = torch.max(points, dim=0)[0]
        voxel_nums = ((boundary_max - boundary_min) / voxel_size + 1).to(torch.int64)
        sampled_point_cloud = torch.zeros(
            (voxel_nums[0] * voxel_nums[1] * voxel_nums[2], 3), device=device
        )
        voxel_nums[2] = voxel_nums[0] * voxel_nums[1]
        voxel_nums[1] = voxel_nums[0]
        voxel_nums[0] = 1
        voxel_indices = ((points - boundary_min) / voxel_size).to(torch.int64)
        voxel_indices = torch.sum(voxel_indices * voxel_nums, dim=1)
        unique_indices = torch.unique(voxel_indices)
        sampled_point_cloud[voxel_indices] = points
        sampled_point_cloud = sampled_point_cloud[unique_indices, :]

        return sampled_point_cloud.unsqueeze(0).contiguous()

    def _len2pre_sum(self, patch_len):
        patch_len_pre_sum = [0]
        for l in patch_len:
            patch_len_pre_sum.append(patch_len_pre_sum[-1] + l)
        return patch_len_pre_sum

    def _knn_query(self, points, length=None):
        """
        Args:
            points (List[Tensor,]): ((B,N,3),...)
            length (List[Tensor,]): ((l1,l2,...),(l1,l2,...))
        Returns:
            idx_self (List[Tensor,]): ((B,npoint,k))
            idx_downsample (List[Tensor,]): ((B,npoint,k))
        """
        idx_self = []
        idx_downsample = []
        for i in range(len(points)):
            idx = knn(
                self.k_neighbor,
                points[i],
                points[i],
                length[i] if "grid" in self.sample_method else None,
                length[i] if "grid" in self.sample_method else None,
            )
            if "grid" in self.sample_method:
                idx.unsqueeze(0).contiguous()
            idx_self.append(idx)
            if i != 0:
                idx = knn(
                    self.k_neighbor,
                    points[i - 1],
                    points[i],
                    length[i - 1] if "grid" in self.sample_method else None,
                    length[i] if "grid" in self.sample_method else None,
                )
                if "grid" in self.sample_method:
                    idx.unsqueeze(0).contiguous()
                idx_downsample.append(idx)
        return idx_self, idx_downsample

    def _ball_query(self, points, length=None):
        """
        Args:
            points (List[Tensor,]): ((B,N,3),...)
            radius (List[int]): (r1,r2,...)
            length (List[Tensor,]): ((l1,l2,...),(l1,l2,...))
        Returns:
            idx_self (List[Tensor,]): ((B,npoint,k))
            idx_downsample (List[Tensor,]): ((B,npoint,k))
        """
        idx_self = []
        idx_downsample = []
        for i in range(len(points)):
            idx = ball_query(
                self.radius[i],
                self.k_neighbor,
                points[i],
                points[i],
                length[i] if "grid" in self.sample_method else None,
                length[i] if "grid" in self.sample_method else None,
            )
            # if "grid" in self.sample_method:
            #     idx.unsqueeze(0).contiguous()
            idx_self.append(idx)
            if i != 0:
                idx = ball_query(
                    self.radius[i],
                    self.k_neighbor,
                    points[i - 1],
                    points[i],
                    length[i - 1] if "grid" in self.sample_method else None,
                    length[i] if "grid" in self.sample_method else None,
                )
                # if "grid" in self.sample_method:
                #     idx.unsqueeze(0).contiguous()
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
            if self.spa_used:
                one_feature = torch.ones_like(points[..., 0:1])
                features = torch.cat([points[..., 3:9], one_feature], dim=-1)
            else:
                one_feature = torch.ones_like(points[..., 0:1])
                features = torch.cat([points[..., [3, 4, 5, 8]], one_feature], dim=-1)
            features = features.permute(0, 2, 1).contiguous()
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

        if "grid" in self.sample_method:
            self.patch_len = [x.shape[0] for x in inputs]
            inputs = torch.cat(inputs).unsqueeze(0)  # 1,N,C

        points, features = self._split_point_feats(
            inputs
        )  # B N 3 and B N C / N 3 and N C

        # down sample and query (rand or voxel / knn or ball)
        points_downsample, length_downsample = self.sample(points)

        # print(length_downsample)

        idx_self, idx_downsample = self.query(points_downsample, length_downsample)

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
                    )
                else:
                    feature_set.append(features)
                    _, features, _ = self.kpconvs[layer_num](
                        points_downsample[i],
                        features,
                        points_downsample[i + 1],
                        idx_downsample[i],
                    )
                layer_num += 1
        feature_set.append(features)
        out = dict(
            points=points_downsample, features=feature_set, length=length_downsample
        )
        return out
