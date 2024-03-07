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
from my_tools.knn import knn, mask_knn
from my_tools.gather import gather
from my_tools.vis_points import vis
from .cross_attention import NNInterpolatePreP


@MODELS.register_module()
class spa(NNInterpolatePreP):
    def __init__(
        self,
        k,
        init_cfg=None,
    ):
        super(spa, self).__init__(init_cfg=init_cfg)
        self.k = k
        self.SpatialAttention1 = SpatialAttention()
        self.SpatialAttention2 = SpatialAttention()
        self.SpatialAttention3 = SpatialAttention()

    def forward(self, inputs):
        points = self.parse_inputs(inputs)

        ## get spe gt & channel info
        self.spe_gt = points[:, :, 3:6]
        self.channel_info = points[:, :, 6].squeeze()

        mask = self._generate_mask(inputs)

        xyz, features = self._split_point_feats(points)  # B N 3 and B N C / N 3 and N C

        mask_idx = mask_knn(
            self.k,
            points,
            points,
            mask,
            mask,
            self.patch_len,
            self.patch_len,
        ).contiguous()
        self.spe_pred = []
        intensity = features[:, :3, :].clone()

        # print("---SpatialAttention1---")
        channel_1 = self.SpatialAttention1(
            xyz, intensity[:, 0:1, :], mask_idx[..., : self.k]
        )
        # print("---SpatialAttention2---")
        channel_2 = self.SpatialAttention2(
            xyz, intensity[:, 1:2, :], mask_idx[..., self.k : 2 * self.k]
        )
        # print("---SpatialAttention3---")
        channel_3 = self.SpatialAttention3(
            xyz, intensity[:, 2:, :], mask_idx[..., 2 * self.k :]
        )
        features_spe = torch.cat([channel_1, channel_2, channel_3], dim=1)
        self.spe_pred.append(features_spe.permute(0, 2, 1))

        features[:, :3, :] = features_spe

        features = features.permute(0, 2, 1)

        # 保留原intensity
        spe_mask = torch.ones_like(features[..., :3])
        idx_x = torch.arange(0, spe_mask.shape[1])
        idx_y = self.channel_info.to(torch.int64).cpu()
        spe_mask[:, idx_x, idx_y] = 0
        new_features = features.clone()
        new_features[..., :3] = new_features[..., :3] * spe_mask + intensity.permute(
            0, 2, 1
        )

        points = torch.cat([xyz, new_features], dim=-1)
        inputs = self.restored_inputs(points)

        return inputs

    def loss(self):
        # entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-8), dim=-1)
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


@MODELS.register_module()
class spa_spe1(NNInterpolatePreP):
    def __init__(
        self,
        k,
        init_cfg=None,
    ):
        super(spa_spe1, self).__init__(init_cfg=init_cfg)
        self.k = k
        self.SpatialAttention1 = SpatialAttention()
        self.SpatialAttention2 = SpatialAttention()
        self.SpatialAttention3 = SpatialAttention()

        self.CrossAttention1 = SpectralAttention(3, 3)

    def forward(self, inputs):
        points = self.parse_inputs(inputs)

        ## get spe gt & channel info
        self.spe_gt = points[:, :, 3:6]
        self.channel_info = points[:, :, 6].squeeze()

        mask = self._generate_mask(inputs)

        xyz, features = self._split_point_feats(points)  # B N 3 and B N C / N 3 and N C

        mask_idx = mask_knn(
            self.k,
            points,
            points,
            mask,
            mask,
            self.patch_len,
            self.patch_len,
        ).contiguous()
        self.spe_pred = []
        intensity = features[:, :3, :].clone()

        # print("---SpatialAttention1---")
        channel_1 = self.SpatialAttention1(
            xyz, intensity[:, 0:1, :], mask_idx[..., : self.k]
        )

        # print("---SpatialAttention2---")
        channel_2 = self.SpatialAttention2(
            xyz, intensity[:, 1:2, :], mask_idx[..., self.k : 2 * self.k]
        )

        # print("---SpatialAttention3---")
        channel_3 = self.SpatialAttention3(
            xyz, intensity[:, 2:, :], mask_idx[..., 2 * self.k :]
        )
        features_spe = torch.cat([channel_1, channel_2, channel_3], dim=1)
        self.spe_pred.append(features_spe.permute(0, 2, 1))

        # print("---SpectralAttention1---")
        self_idx = knn(self.k, xyz, xyz, self.patch_len, self.patch_len)
        features_spe = self.CrossAttention1(
            xyz, features_spe, features_spe, self_idx[..., 1:]
        )
        self.spe_pred.append(features_spe.permute(0, 2, 1))

        features[:, :3, :] = features_spe

        features = features.permute(0, 2, 1)

        ## get spe pred
        # import pdb
        # pdb.set_trace()

        # 保留原intensity
        spe_mask = torch.ones_like(features[..., :3])
        idx_x = torch.arange(0, spe_mask.shape[1])
        idx_y = self.channel_info.to(torch.int64).cpu()
        spe_mask[:, idx_x, idx_y] = 0
        new_features = features.clone()
        new_features[..., :3] = new_features[..., :3] * spe_mask + intensity.permute(
            0, 2, 1
        )

        points = torch.cat([xyz, new_features], dim=-1)
        inputs = self.restored_inputs(points)

        return inputs

    def loss(self):
        # entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-8), dim=-1)
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


@MODELS.register_module()
class spa_spe2(NNInterpolatePreP):
    def __init__(
        self,
        k,
        init_cfg=None,
    ):
        super(spa_spe2, self).__init__(init_cfg=init_cfg)
        self.k = k
        self.SpatialAttention1 = SpatialAttention()
        self.SpatialAttention2 = SpatialAttention()
        self.SpatialAttention3 = SpatialAttention()

        self.CrossAttention1 = SpectralAttention(3, 3)
        self.CrossAttention2 = SpectralAttention(3, 3)

    def forward(self, inputs):
        points = self.parse_inputs(inputs)

        ## get spe gt & channel info
        self.spe_gt = points[:, :, 3:6]
        self.channel_info = points[:, :, 6].squeeze()

        mask = self._generate_mask(inputs)

        xyz, features = self._split_point_feats(points)  # B N 3 and B N C / N 3 and N C

        mask_idx = mask_knn(
            self.k,
            points,
            points,
            mask,
            mask,
            self.patch_len,
            self.patch_len,
        ).contiguous()
        self.spe_pred = []
        intensity = features[:, :3, :].clone()

        # print("---SpatialAttention1---")
        channel_1 = self.SpatialAttention1(
            xyz, intensity[:, 0:1, :], mask_idx[..., : self.k]
        )

        # print("---SpatialAttention2---")
        channel_2 = self.SpatialAttention2(
            xyz, intensity[:, 1:2, :], mask_idx[..., self.k : 2 * self.k]
        )

        # print("---SpatialAttention3---")
        channel_3 = self.SpatialAttention3(
            xyz, intensity[:, 2:, :], mask_idx[..., 2 * self.k :]
        )
        features_spe = torch.cat([channel_1, channel_2, channel_3], dim=1)
        self.spe_pred.append(features_spe.permute(0, 2, 1))

        # print("---SpectralAttention1---")
        self_idx = knn(self.k, xyz, xyz, self.patch_len, self.patch_len)
        features_spe = self.CrossAttention1(
            xyz, features_spe, features_spe, self_idx[..., 1:]
        )
        self.spe_pred.append(features_spe.permute(0, 2, 1))

        # print("---SpectralAttention2---")
        features_spe = self.CrossAttention2(
            xyz, features_spe, features_spe, self_idx[..., 1:]
        )
        self.spe_pred.append(features_spe.permute(0, 2, 1))

        features[:, :3, :] = features_spe

        features = features.permute(0, 2, 1)

        ## get spe pred
        # import pdb
        # pdb.set_trace()

        # 保留原intensity
        spe_mask = torch.ones_like(features[..., :3])
        idx_x = torch.arange(0, spe_mask.shape[1])
        idx_y = self.channel_info.to(torch.int64).cpu()
        spe_mask[:, idx_x, idx_y] = 0
        new_features = features.clone()
        new_features[..., :3] = new_features[..., :3] * spe_mask + intensity.permute(
            0, 2, 1
        )

        points = torch.cat([xyz, new_features], dim=-1)
        inputs = self.restored_inputs(points)

        return inputs

    def loss(self):
        # entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-8), dim=-1)
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


@MODELS.register_module()
class spa_spe2_noloss(spa_spe2):
    def __init__(
        self,
        k,
        init_cfg=None,
    ):
        super(spa_spe2_noloss, self).__init__(k=k, init_cfg=init_cfg)

    def loss(self):
        return None

@MODELS.register_module()
class spa_spe2_nogaussian(spa_spe2):
    def __init__(
        self,
        k,
        init_cfg=None,
    ):
        super(spa_spe2_nogaussian, self).__init__(k=k, init_cfg=init_cfg)
        self.k = k
        self.SpatialAttention1 = SpatialAttention(False)
        self.SpatialAttention2 = SpatialAttention(False)
        self.SpatialAttention3 = SpatialAttention(False)

        self.CrossAttention1 = SpectralAttention(3, 3, False)
        self.CrossAttention2 = SpectralAttention(3, 3, False)

@MODELS.register_module()
class spa_spe3(NNInterpolatePreP):
    def __init__(
        self,
        k,
        init_cfg=None,
    ):
        super(spa_spe3, self).__init__(init_cfg=init_cfg)
        self.k = k
        self.SpatialAttention1 = SpatialAttention()
        self.SpatialAttention2 = SpatialAttention()
        self.SpatialAttention3 = SpatialAttention()

        self.CrossAttention1 = SpectralAttention(3, 3)
        self.CrossAttention2 = SpectralAttention(3, 3)
        self.CrossAttention3 = SpectralAttention(3, 3)

    def forward(self, inputs):
        points = self.parse_inputs(inputs)

        ## get spe gt & channel info
        self.spe_gt = points[:, :, 3:6]
        self.channel_info = points[:, :, 6].squeeze()

        mask = self._generate_mask(inputs)

        xyz, features = self._split_point_feats(points)  # B N 3 and B N C / N 3 and N C

        mask_idx = mask_knn(
            self.k,
            points,
            points,
            mask,
            mask,
            self.patch_len,
            self.patch_len,
        ).contiguous()
        self.spe_pred = []
        intensity = features[:, :3, :].clone()

        # print("---SpatialAttention1---")
        channel_1 = self.SpatialAttention1(
            xyz, intensity[:, 0:1, :], mask_idx[..., : self.k]
        )

        # print("---SpatialAttention2---")
        channel_2 = self.SpatialAttention2(
            xyz, intensity[:, 1:2, :], mask_idx[..., self.k : 2 * self.k]
        )

        # print("---SpatialAttention3---")
        channel_3 = self.SpatialAttention3(
            xyz, intensity[:, 2:, :], mask_idx[..., 2 * self.k :]
        )
        features_spe = torch.cat([channel_1, channel_2, channel_3], dim=1)
        self.spe_pred.append(features_spe.permute(0, 2, 1))

        # print("---SpectralAttention1---")
        self_idx = knn(self.k, xyz, xyz, self.patch_len, self.patch_len)
        features_spe = self.CrossAttention1(
            xyz, features_spe, features_spe, self_idx[..., 1:]
        )
        self.spe_pred.append(features_spe.permute(0, 2, 1))

        # print("---SpectralAttention2---")
        features_spe = self.CrossAttention2(
            xyz, features_spe, features_spe, self_idx[..., 1:]
        )
        self.spe_pred.append(features_spe.permute(0, 2, 1))

        # print("---SpectralAttention3---")
        features_spe = self.CrossAttention3(
            xyz, features_spe, features_spe, self_idx[..., 1:]
        )
        self.spe_pred.append(features_spe.permute(0, 2, 1))

        features[:, :3, :] = features_spe

        features = features.permute(0, 2, 1)

        ## get spe pred
        # import pdb
        # pdb.set_trace()

        # 保留原intensity
        spe_mask = torch.ones_like(features[..., :3])
        idx_x = torch.arange(0, spe_mask.shape[1])
        idx_y = self.channel_info.to(torch.int64).cpu()
        spe_mask[:, idx_x, idx_y] = 0
        new_features = features.clone()
        new_features[..., :3] = new_features[..., :3] * spe_mask + intensity.permute(
            0, 2, 1
        )

        points = torch.cat([xyz, new_features], dim=-1)
        inputs = self.restored_inputs(points)

        return inputs

    def loss(self):
        # entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-8), dim=-1)
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


@MODELS.register_module()
class spa_spe4(NNInterpolatePreP):
    def __init__(
        self,
        k,
        init_cfg=None,
    ):
        super(spa_spe4, self).__init__(init_cfg=init_cfg)
        self.k = k
        self.SpatialAttention1 = SpatialAttention()
        self.SpatialAttention2 = SpatialAttention()
        self.SpatialAttention3 = SpatialAttention()

        self.CrossAttention1 = SpectralAttention(3, 3)
        self.CrossAttention2 = SpectralAttention(3, 3)
        self.CrossAttention3 = SpectralAttention(3, 3)
        self.CrossAttention4 = SpectralAttention(3, 3)

    def forward(self, inputs):
        points = self.parse_inputs(inputs)

        ## get spe gt & channel info
        self.spe_gt = points[:, :, 3:6]
        self.channel_info = points[:, :, 6].squeeze()

        mask = self._generate_mask(inputs)

        xyz, features = self._split_point_feats(points)  # B N 3 and B N C / N 3 and N C

        mask_idx = mask_knn(
            self.k,
            points,
            points,
            mask,
            mask,
            self.patch_len,
            self.patch_len,
        ).contiguous()
        self.spe_pred = []
        intensity = features[:, :3, :].clone()

        # print("---SpatialAttention1---")
        channel_1 = self.SpatialAttention1(
            xyz, intensity[:, 0:1, :], mask_idx[..., : self.k]
        )

        # print("---SpatialAttention2---")
        channel_2 = self.SpatialAttention2(
            xyz, intensity[:, 1:2, :], mask_idx[..., self.k : 2 * self.k]
        )

        # print("---SpatialAttention3---")
        channel_3 = self.SpatialAttention3(
            xyz, intensity[:, 2:, :], mask_idx[..., 2 * self.k :]
        )
        features_spe = torch.cat([channel_1, channel_2, channel_3], dim=1)
        self.spe_pred.append(features_spe.permute(0, 2, 1))

        # print("---SpectralAttention1---")
        self_idx = knn(self.k, xyz, xyz, self.patch_len, self.patch_len)
        features_spe = self.CrossAttention1(
            xyz, features_spe, features_spe, self_idx[..., 1:]
        )
        self.spe_pred.append(features_spe.permute(0, 2, 1))

        # print("---SpectralAttention2---")
        features_spe = self.CrossAttention2(
            xyz, features_spe, features_spe, self_idx[..., 1:]
        )
        self.spe_pred.append(features_spe.permute(0, 2, 1))

        # print("---SpectralAttention3---")
        features_spe = self.CrossAttention3(
            xyz, features_spe, features_spe, self_idx[..., 1:]
        )
        self.spe_pred.append(features_spe.permute(0, 2, 1))

        # print("---SpectralAttention4---")
        features_spe = self.CrossAttention4(
            xyz, features_spe, features_spe, self_idx[..., 1:]
        )
        self.spe_pred.append(features_spe.permute(0, 2, 1))

        features[:, :3, :] = features_spe

        features = features.permute(0, 2, 1)

        ## get spe pred
        # import pdb
        # pdb.set_trace()

        # 保留原intensity
        spe_mask = torch.ones_like(features[..., :3])
        idx_x = torch.arange(0, spe_mask.shape[1])
        idx_y = self.channel_info.to(torch.int64).cpu()
        spe_mask[:, idx_x, idx_y] = 0
        new_features = features.clone()
        new_features[..., :3] = new_features[..., :3] * spe_mask + intensity.permute(
            0, 2, 1
        )

        points = torch.cat([xyz, new_features], dim=-1)
        inputs = self.restored_inputs(points)

        return inputs

    def loss(self):
        # entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-8), dim=-1)
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


class SpatialAttention(nn.Module):
    def __init__(self, gaussian=True):
        super(SpatialAttention, self).__init__()

        self.linear_p = nn.Sequential(
            ConvModule(
                3,
                3,
                1,
                1,
                bias=True,
                conv_cfg=dict(type="Conv2d"),
                norm_cfg=dict(type="BN2d"),
                act_cfg=dict(type="ReLU"),
            ),
            ConvModule(
                3,
                1,
                1,
                1,
                bias=True,
                conv_cfg=dict(type="Conv2d"),
                act_cfg=None,
            ),
        )
        self.softmax = nn.Softmax(dim=-1)
        self.gaussian = gaussian

    def forward(self, xyz, intensity, indices):
        """
        args:
            xyz: [B,N,3]
            features: [B,C1,N]
            intensity: [B,3,N]
            channel_num: [B,N,1]
        return:
            x: [B,C2,N]
        """

        xyz_neighbor = gather(xyz, indices)
        i_neighbor = gather(intensity, indices, True)

        pos = xyz_neighbor - xyz.permute(0, 2, 1).unsqueeze(-1)  # B 3 N K
        if self.gaussian:
            pos_gaussian = torch.exp(-2 * (pos) ** 2)
        else:
            pos_gaussian = pos
        # print(f"pos:{pos[0,:,0,:]}")
        # print(f"pos_dist:{torch.sum(pos[0,:,0,:]**2,dim=0,keepdim=True)}")
        w = self.linear_p(pos_gaussian)  # B,C,N,K
        # print(f"w:{w[0,:,0,:]}")
        w = self.softmax(w)
        # print(f"w softmax:{w[0,:,0,:]}")  # B C N K
        # print("********************")

        # print(f"conv1:{self.linear_p[0].conv.weight}")
        # print(f"bias1:{self.linear_p[0].conv.bias}")
        # print(f"conv2:{self.linear_p[1].conv.weight}")
        # print(f"bias2:{self.linear_p[1].conv.bias}")
        # print(f"bn0.a:{self.linear_p[0].norm.weight}")
        # print(f"bn0.b:{self.linear_p[0].norm.bias}")

        x = torch.sum(i_neighbor * w, dim=-1)

        return x


class SpectralAttention(nn.Module):
    def __init__(self, input_dims, output_dims, gaussian=True):
        super(SpectralAttention, self).__init__()
        self.linear_q = ConvModule(
            input_dims,
            output_dims,
            1,
            1,
            bias=True,
            conv_cfg=dict(type="Conv1d"),
            act_cfg=None,
        )
        self.linear_k = ConvModule(
            input_dims,
            output_dims,
            1,
            1,
            bias=True,
            conv_cfg=dict(type="Conv1d"),
            act_cfg=None,
        )
        self.linear_v = ConvModule(
            input_dims,
            output_dims,
            1,
            1,
            bias=True,
            conv_cfg=dict(type="Conv1d"),
            act_cfg=None,
        )
        self.linear_w = nn.Sequential(
            nn.BatchNorm2d(output_dims + 3),
            nn.ReLU(inplace=True),
            ConvModule(
                output_dims + 3,
                output_dims + 3,
                1,
                1,
                bias=True,
                conv_cfg=dict(type="Conv2d"),
                norm_cfg=dict(type="BN2d"),
                act_cfg=dict(type="ReLU"),
            ),
            ConvModule(
                output_dims + 3,
                output_dims,
                1,
                1,
                bias=True,
                conv_cfg=dict(type="Conv2d"),
                act_cfg=None,
            ),
        )
        self.softmax = nn.Softmax(dim=-1)
        self.gaussian = gaussian

    def forward(self, xyz, features, intensity, indices):
        """
        args:
            xyz: [B,N,3]
            features: [B,C1,N]
            intensity: [B,3,N]
            channel_num: [B,N,1]
        return:
            x: [B,C2,N]
        """
        q = self.linear_q(features)
        k = self.linear_k(features)
        # v = self.linear_v(intensity)
        v = intensity

        xyz_neighbor = gather(xyz, indices)
        i_neighbor = gather(v, indices, True)
        k_neighbor = gather(k, indices, True)

        feat = k_neighbor - q.unsqueeze(-1)
        if self.gaussian:
            feat_gaussian = torch.exp(-2 * (feat) ** 2)
        else:
            feat_gaussian = feat
        # print(f"feat:{feat[0,:,0,:]}")
        # print(f"feat_dist:{torch.sum(feat[0,:,0,:]**2,dim=0,keepdim=True)}")

        pos = xyz_neighbor - xyz.permute(0, 2, 1).unsqueeze(-1)  # B 3 N K
        if self.gaussian:
            pos_gaussian = torch.exp(-2 * (pos) ** 2)
        else:
            pos_gaussian = pos
        # print(f"pos:{pos[0,:,0,:]}")
        # print(f"pos_dist:{torch.sum(pos[0,:,0,:]**2,dim=0,keepdim=True)}")

        w = self.linear_w(torch.cat([feat_gaussian, pos_gaussian], dim=1))  # B,C,N,K
        # print(f"w:{w[0,:,0,:]}")
        w = self.softmax(w)
        # print(f"w softmax:{w[0,:,0,:]}")  # B C N K

        x = torch.sum(i_neighbor * w, dim=-1)

        return x
