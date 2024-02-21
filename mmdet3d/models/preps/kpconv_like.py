from typing import List, Tuple, Optional
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType
import numpy as np
import torch
from torch import Tensor
from torch import nn as nn
import torch.nn.functional as f
from mmcv.cnn import ConvModule
from ..layers.kpconv_modules.kpconv import KPConvBlock
from ..layers.point_transformer.point_transformer import PointTranformerBlock
from .cross_attention import NNInterpolatePreP
from my_tools.knn import knn, mask_knn
from my_tools.gather import gather
from my_tools.vis_points import vis


@MODELS.register_module()
class KPConvPreP(NNInterpolatePreP):
    def __init__(
        self,
        kernel_size: int,
        k_neighbor: int,
        weight_norm: bool = False,
        norm_cfg: ConfigType = dict(type="BN1d"),
        act_cfg: ConfigType = dict(type="LeakyReLU", negative_slope=0.1),
        init_cfg=None,
    ):
        super(KPConvPreP, self).__init__(init_cfg=init_cfg)

        self.k_neighbor = k_neighbor
        self.ExtractFeat1 = KPConvBlock(
            kernel_size=kernel_size,
            input_dim=3,
            output_dim=18,
            weight_norm=weight_norm,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.ExtractFeat2 = KPConvBlock(
            kernel_size=kernel_size,
            input_dim=18,
            output_dim=3,
            weight_norm=weight_norm,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

    def forward(self, inputs):

        points = self.parse_inputs(inputs)

        ## get spe gt & channel info
        self.spe_gt = points[:, :, 3:6]
        self.spe_pred = []
        self.channel_info = points[:, :, 6].squeeze()

        mask = self._generate_mask(inputs)

        xyz, features = self._split_point_feats(points)  # B N 3 and B N C / N 3 and N C

        mask_idx = mask_knn(
            6,
            points,
            points,
            mask,
            mask,
            self.patch_len,
            self.patch_len,
        ).contiguous()
        intensity = features[:, :3, :].clone()

        _, features_spe, _ = self.ExtractFeat1(xyz, intensity, xyz, mask_idx)

        self_idx = knn(self.k_neighbor, xyz, xyz, self.patch_len, self.patch_len)

        _, features_spe, _ = self.ExtractFeat2(xyz, features_spe, xyz, self_idx)

        features[:, :3, :] = features_spe
        features = features.permute(0, 2, 1)

        self.spe_pred.append(features[..., :3])

        # 保留原intensity
        spe_mask = torch.ones_like(features[..., :3])
        idx_x = torch.arange(0, spe_mask.shape[1])
        idx_y = self.channel_info.to(torch.int64).cpu()
        spe_mask[:, idx_x, idx_y] = 0
        new_features = features.clone()
        new_features[..., :3] = new_features[..., :3] * spe_mask + intensity.permute(0, 2, 1)

        points = torch.cat([xyz, new_features], dim=-1)
        inputs = self.restored_inputs(points)

        return inputs

    def loss(self):
        self.spe_gt = self.spe_gt.squeeze()
        losses = dict()
        loss_mask = torch.zeros_like(self.spe_gt)
        idx_x = torch.arange(0, loss_mask.shape[0])
        idx_y = self.channel_info.to(torch.int64).cpu()
        # 通道 mask
        loss_mask[idx_x, idx_y] = 1
        # 异常光谱值 mask
        loss_mask[self.spe_gt >= 0.9] = 0
        num_point = torch.sum(loss_mask)

        for i, spe_pred in enumerate(self.spe_pred):
            loss_all_channel = torch.abs(self.spe_gt - spe_pred.squeeze())
            loss_spe = torch.sum(loss_all_channel * loss_mask) / num_point
            losses[f"loss_spe_{i}"] = loss_spe
        return losses
