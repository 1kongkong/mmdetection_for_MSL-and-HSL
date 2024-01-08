from typing import List, Tuple, Optional
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType
import torch
from torch import Tensor
from torch import nn as nn
from ..layers.kpconv_modules.kpconv import KPConvBlock
from ..layers.point_transformer.point_transformer import PointTranformerBlock
from my_tools.knn import knn
from my_tools.gather import gather


@MODELS.register_module()
class CrossInterpolatePreP(BaseModule):
    def __init__(
        self,
        in_channels: List[int],
        kernel_size: int,
        k_neighbor: int,
        weight_norm: bool = False,
        norm_cfg: ConfigType = dict(type="BN1d"),
        act_cfg: ConfigType = dict(type="LeakyReLU", negative_slope=0.1),
    ):
        super(CrossInterpolatePreP, self).__init__()

        self.ExtractFeat1 = KPConvBlock(
            kernel_size=kernel_size,
            input_dim=3,
            output_dim=16,
            weight_norm=weight_norm,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.ExtractFeat2 = KPConvBlock(
            kernel_size=kernel_size,
            input_dim=16,
            output_dim=8,
            weight_norm=weight_norm,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.CrossAttention1 = CrossInterpolateBlock(
            input_dims=8,
            output_dims=8,
            share_planes=2,
            num_sample=k_neighbor,
        )
        self.CrossAttention2 = PointTranformerBlock(
            input_dims=6,
            output_dims=6,
            share_planes=1,
            num_sample=k_neighbor,
        )


class CrossInterpolateBlock(PointTranformerBlock):
    def __init__(self, input_dims, output_dims, share_planes, num_sample):
        super(CrossInterpolateBlock, self).__init__(
            input_dims=input_dims,
            output_dims=output_dims,
            share_planes=share_planes,
            num_sample=num_sample,
        )
        del self.linear_v
        pass

    def _knn_query(self, points, channel_num):
        """
        Args:
            points (Tensor): (B,N,3)
            channel_num (Tensor) : (B,N,1)
        Returns:
            idx (Tensor): (B,N,k)
        """
        idx = knn(self.num_sample, points, points, False).transpose(1, 2).contiguous()
        return idx

    def forward(self, xyz, features, intensity, channel_num):
        """
        args:
            xyz: [B,N,3]
            features: [B,C1,N]
            intensity: [B,3,N]
            channel_num: [B,N,1]
        return:
            x: [B,C2,N]
        """
        x_q, x_k, x_v = (
            self.linear_q(features),
            self.linear_k(features),
            intensity,
        )  # B,C,N

        indices = self._knn_query(xyz, channel_num)

        p_r = gather(xyz, indices)
        x_k = gather(x_k, indices, True)
        x_v = gather(x_v, indices, True)

        p_r = self.linear_p(p_r)
        tem_p = p_r

        w = x_k - x_q.unsqueeze(-1) + tem_p  # B,C,N,K
        w = self.linear_w(w)

        w = self.softmax(w)  # B,C,N,K
        b, c, n, nsample = x_v.shape
        w = w.permute(0, 2, 3, 1)  # B,N,K,C
        x_v = x_v.permute(0, 2, 3, 1)
        p_r = p_r.permute(0, 2, 3, 1)
        s = self.share_planes
        x = (
            ((x_v + p_r).view(b, n, nsample, s, c // s) * w.unsqueeze(3))
            .sum(2)
            .view(b, n, c)
        )

        return x
