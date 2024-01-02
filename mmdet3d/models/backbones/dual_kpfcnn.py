from typing import List, Tuple, Optional
from mmengine.model import BaseModule
import torch
from torch import Tensor
from torch import nn as nn
import copy

from mmdet3d.registry import MODELS

from mmdet3d.utils import ConfigType
from ..layers.kpconv_modules.kpconv import KPConvBlock, KPResNetBlock
from mmdet3d.models.backbones.kpfcnn import KPFCNNBackbone
from mmdet3d.models.layers.fusion_layers import (
    fusion_block,
    cross_att_fusion_block,
    cross_att_fusion_block_2,
)


@MODELS.register_module()
class Dual_KPFCNNBackbone(KPFCNNBackbone):
    def __init__(
        self,
        num_point: int,
        in_channels: int,
        kernel_size: int,
        k_neighbor: int,
        kpconv_channels: List[List[int]],
        sample_method: str = "rand",  # {'rand','grid','grid+rand'}
        query_method: str = "knn",  # {'knn','ball'}
        voxel_size: List[float] = None,
        radius: List[float] = None,
        weight_norm: bool = False,
        norm_cfg: ConfigType = dict(type="BN1d"),
        act_cfg: ConfigType = dict(type="LeakyReLU", negative_slope=0.1),
    ):
        super(Dual_KPFCNNBackbone, self).__init__(
            num_point=num_point,
            in_channels=in_channels // 2,
            kernel_size=kernel_size,
            k_neighbor=k_neighbor,
            kpconv_channels=kpconv_channels,
            sample_method=sample_method,
            query_method=query_method,
            voxel_size=voxel_size,
            radius=radius,
            weight_norm=weight_norm,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

        self.kpconvs_0 = self.kpconvs
        self.kpconvs_1 = nn.ModuleList()
        self.fusions = nn.ModuleList()

        for i in range(1, len(self.channel_list)):
            # self.fusions.append(fusion_block(self.channel_list[i][-1], norm_cfg=norm_cfg, act_cfg=act_cfg))
            # self.fusions.append(cross_att_fusion_block(self.channel_list[i][-1]))
            self.fusions.append(cross_att_fusion_block_2(self.channel_list[i][-1]))
            for j in range(len(self.channel_list[i])):
                if j == 0:
                    self.kpconvs_1.append(
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
                    self.kpconvs_1.append(
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

        points, features = self._split_point_feats(inputs)

        # down sample and query (rand or voxel / knn or ball)
        points_downsample, length_downsample = self.sample(points)
        # print(length_downsample)

        idx_self, idx_downsample = self.query(points_downsample, length_downsample)

        layer_num = 0
        feature_spa = features[:, 3:, :]
        feature_spe = features[:, :3, :]
        feature_set = []
        for i in range(len(self.kpconv_channels)):
            for j in range(len(self.kpconv_channels[i])):
                if (
                    j != len(self.kpconv_channels[i]) - 1
                    or i == len(self.kpconv_channels) - 1
                ):  # 最后一层不进行下采样，其它层的最后一次卷积进行下采样
                    _, feature_spa, _ = self.kpconvs_0[layer_num](
                        points_downsample[i],
                        feature_spa,
                        points_downsample[i],
                        idx_self[i],
                    )
                    _, feature_spe, _ = self.kpconvs_1[layer_num](
                        points_downsample[i],
                        feature_spe,
                        points_downsample[i],
                        idx_self[i],
                    )
                else:
                    # feature_set.append(self.fusions[i](feature_spa, feature_spe, idx_self[i]))
                    feature_set.append(
                        self.fusions[i](
                            points_downsample[i], feature_spa, feature_spe, idx_self[i]
                        )
                    )
                    _, feature_spa, _ = self.kpconvs_0[layer_num](
                        points_downsample[i],
                        feature_spa,
                        points_downsample[i + 1],
                        idx_downsample[i],
                    )
                    _, feature_spe, _ = self.kpconvs_1[layer_num](
                        points_downsample[i],
                        feature_spe,
                        points_downsample[i + 1],
                        idx_downsample[i],
                    )
                layer_num += 1
        # feature_set.append(self.fusions[-1](feature_spa, feature_spe, idx_self[-1]))
        feature_set.append(
            self.fusions[-1](
                points_downsample[-1], feature_spa, feature_spe, idx_self[-1]
            )
        )
        out = dict(
            points=points_downsample, features=feature_set, length=length_downsample
        )
        return out
