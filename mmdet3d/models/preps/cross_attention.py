from typing import List, Tuple, Optional
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType
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

f.sot


@MODELS.register_module()
class BasePreP(BaseModule):
    def __init__(
        self,
    ):
        super(BasePreP, self).__init__()

    def forward(self, inputs):
        select_idx = [0, 1, 2, 7, 8, 9, 10, 11, 12]
        inputs = [x[:, select_idx] for x in inputs]

        return inputs

    def loss(self):
        return None


@MODELS.register_module()
class NNInterpolatePreP(BasePreP):
    def __init__(
        self,
    ):
        super(NNInterpolatePreP, self).__init__()

    def _generate_mask(self, inputs):
        mask = []
        for i in range(len(inputs)):
            mask_patch = []
            channel_info = inputs[i][:, 6]
            for j in torch.unique(channel_info):
                mask_patch.append(torch.sum(channel_info == j).item())
            mask.append(mask_patch)
        self.mask = mask
        return mask

    def _split_point_feats(self, points):
        xyz = points[:, :, :3]
        features = points[:, :, [3, 4, 5, 7, 8, 9]]
        features = features.permute(0, 2, 1)

        return xyz, features

    def forward(self, inputs):
        self.patch_len = [x.shape[0] for x in inputs]
        points = torch.cat(inputs).unsqueeze(0)  # 1,N,C
        self.patch_len = torch.tensor(self.patch_len, device=points.device)
        mask = self._generate_mask(inputs)

        xyz, features = self._split_point_feats(points)  # B N 3 and B N C / N 3 and N C

        mask_idx = mask_knn(
            1,
            points,
            points,
            mask,
            mask,
            self.patch_len,
            self.patch_len,
        ).contiguous()

        intensity = features[:, :3, :].clone()
        intensity_neighbor = gather(intensity, mask_idx, True)  # B,C,N,K
        features[:, :3, :] = torch.sum(intensity_neighbor, dim=-1)

        features = features.permute(0, 2, 1)

        points = torch.cat([xyz, features], dim=-1)

        inputs = []
        cml = 0
        for length in self.patch_len:
            inputs.append(points[:, cml : cml + length, :].squeeze(0))
            cml += length

        return inputs


@MODELS.register_module()
class IDWInterpolatePreP(NNInterpolatePreP):
    def __init__(
        self,
        k,
    ):
        super(IDWInterpolatePreP, self).__init__()
        self.k = k

    def forward(self, inputs):
        self.patch_len = [x.shape[0] for x in inputs]
        points = torch.cat(inputs).unsqueeze(0)  # 1,N,C
        self.patch_len = torch.tensor(self.patch_len, device=points.device)
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
        b, c, n = features.shape
        features_spa = features[:, 3:, :]
        intensity = features[:, :3, :].clone()

        intensity_neighbor = gather(intensity, mask_idx, True)  # B,C,N,K
        spa_neighbor = gather(features_spa, mask_idx, True)
        spa_neighbor_diff = spa_neighbor - features_spa.unsqueeze(-1)
        dist = torch.sqrt(torch.sum(spa_neighbor_diff**2, dim=1)).view(b, n, 3, -1)
        dist_rep = 1.0 / (dist + 1e-6)
        weight = (dist_rep / torch.sum(dist_rep, dim=-1, keepdim=True) + 1e-6).view(
            b, 1, n, -1
        )
        feature_spe = torch.sum(intensity_neighbor * weight, dim=-1)
        features[:, :3, :] = feature_spe

        features = features.permute(0, 2, 1)
        points = torch.cat([xyz, features], dim=-1)
        inputs = []
        cml = 0
        for length in self.patch_len:
            inputs.append(points[:, cml : cml + length, :].squeeze(0))
            cml += length

        return inputs


@MODELS.register_module()
class CrossInterpolatePreP(NNInterpolatePreP):
    def __init__(
        self,
        kernel_size: int,
        k_neighbor: int,
        weight_norm: bool = False,
        norm_cfg: ConfigType = dict(type="BN1d"),
        act_cfg: ConfigType = dict(type="LeakyReLU", negative_slope=0.1),
    ):
        super(CrossInterpolatePreP, self).__init__()

        self.k_neighbor = k_neighbor
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
            output_dims=3,
            share_planes=1,
            num_sample=5,
        )
        self.CrossAttention2 = CrossInterpolateBlock(
            input_dims=6,
            output_dims=3,
            share_planes=1,
            num_sample=5,
        )

    def forward(self, inputs):
        self.patch_len = [x.shape[0] for x in inputs]
        points = torch.cat(inputs).unsqueeze(0)  # 1,N,C

        ## get spe gt & channel info
        self.spe_gt = points[:, :, 3:6]
        self.channel_info = points[:, :, 6].squeeze()

        self.patch_len = torch.tensor(self.patch_len, device=points.device)
        mask = self._generate_mask(inputs)

        xyz, features = self._split_point_feats(points)  # B N 3 and B N C / N 3 and N C

        self_idx = knn(self.k_neighbor, xyz, xyz, self.patch_len, self.patch_len)
        _, features_spa, _ = self.ExtractFeat1(xyz, features[:, 3:, :], xyz, self_idx)
        _, features_spa, _ = self.ExtractFeat2(xyz, features_spa, xyz, self_idx)
        mask_idx = (
            mask_knn(
                5,
                points,
                points,
                mask,
                mask,
                self.patch_len,
                self.patch_len,
            )
            .transpose(1, 2)
            .contiguous()
        )
        intensity = features[:, :3, :].clone()
        feature_spe = self.CrossAttention1(xyz, features_spa, intensity, mask_idx)
        features[:, :3, :] = feature_spe
        feature_spe = self.CrossAttention2(xyz, features, intensity, mask_idx)
        features[:, :3, :] = feature_spe

        features = features.permute(0, 2, 1)

        ## get spe pred
        self.spe_pred = features[:, :, :3]

        points = torch.cat([xyz, features], dim=-1)
        inputs = []
        cml = 0
        for length in self.patch_len:
            inputs.append(points[:, cml : cml + length, :].squeeze(0))
            cml += length

        return inputs

    def loss(self):
        self.spe_gt = self.spe_gt.squeeze()
        self.spe_pred = self.spe_pred.squeeze()
        losses = dict()
        loss_all_channel = torch.abs(self.spe_gt - self.spe_pred)
        loss_mask = torch.zeros_like(loss_all_channel)

        idx_x = torch.arange(0, loss_mask.shape[0])
        idx_y = self.channel_info.to(torch.int64).cpu()
        loss_mask[idx_x, idx_y] = 1

        loss_spe = torch.mean(torch.sum(loss_all_channel * loss_mask, dim=1))
        losses["loss_spe"] = loss_spe * 15
        return losses


@MODELS.register_module()
class CrossInterpolatePreP2(NNInterpolatePreP):
    def __init__(
        self,
        k,
    ):
        super(CrossInterpolatePreP2, self).__init__()
        self.k = k
        self.CrossAttention1 = CrossInterpolateBlock(
            input_dims=3,
            output_dims=3,
            share_planes=1,
            num_sample=k,
        )
        self.CrossAttention2 = CrossInterpolateBlock(
            input_dims=6,
            output_dims=3,
            share_planes=1,
            num_sample=k,
        )

    def forward(self, inputs):
        self.patch_len = [x.shape[0] for x in inputs]
        points = torch.cat(inputs).unsqueeze(0)  # 1,N,C

        ## get spe gt & channel info
        self.spe_gt = points[:, :, 3:6]
        self.channel_info = points[:, :, 6].squeeze()

        self.patch_len = torch.tensor(self.patch_len, device=points.device)
        mask = self._generate_mask(inputs)

        xyz, features = self._split_point_feats(points)  # B N 3 and B N C / N 3 and N C

        mask_idx = (
            mask_knn(
                self.k,
                points,
                points,
                mask,
                mask,
                self.patch_len,
                self.patch_len,
            )
            .transpose(1, 2)
            .contiguous()
        )

        features_spa = features[:, 3:, :]
        intensity = features[:, :3, :].clone()
        # print("----------------")
        # print(intensity[0, :5, 0])
        feature_spe = self.CrossAttention1(xyz, features_spa, intensity, mask_idx)
        # print(feature_spe[0, :5, 0])
        features[:, :3, :] = feature_spe
        feature_spe = self.CrossAttention2(xyz, features, intensity, mask_idx)
        # print(feature_spe[0, :5, 0])
        features[:, :3, :] = feature_spe

        features = features.permute(0, 2, 1)

        ## get spe pred
        self.spe_pred = features[:, :, :3]
        # import pdb
        # pdb.set_trace()
        points = torch.cat([xyz, features], dim=-1)
        inputs = []
        cml = 0
        for length in self.patch_len:
            inputs.append(points[:, cml : cml + length, :].squeeze(0))
            cml += length
            # vis(inputs[-1][:, :3].cpu().numpy(), inputs[-1][:, 3:6].cpu().numpy())
        torch.cuda.empty_cache()
        return inputs

    def loss(self):
        self.spe_gt = self.spe_gt.squeeze()
        self.spe_pred = self.spe_pred.squeeze()
        losses = dict()
        loss_all_channel = torch.abs(self.spe_gt - self.spe_pred)
        loss_mask = torch.zeros_like(loss_all_channel)

        idx_x = torch.arange(0, loss_mask.shape[0])
        idx_y = self.channel_info.to(torch.int64).cpu()
        loss_mask[idx_x, idx_y] = 1

        loss_spe = torch.mean(torch.sum(loss_all_channel * loss_mask, dim=1))
        losses["loss_spe"] = loss_spe * 15
        return losses


@MODELS.register_module()
class CrossInterpolatePreP4(NNInterpolatePreP):
    def __init__(
        self,
        k,
    ):
        super(CrossInterpolatePreP4, self).__init__()
        self.k = k
        self.CrossAttention1 = CrossInterpolateBlock2(
            input_dims=3,
            output_dims=3,
            v_dims=3,
            share_planes=1,
            num_sample=k,
        )
        self.CrossAttention2 = CrossInterpolateBlock2(
            input_dims=6,
            output_dims=3,
            v_dims=3,
            share_planes=1,
            num_sample=k,
        )

    def forward(self, inputs):
        self.patch_len = [x.shape[0] for x in inputs]
        points = torch.cat(inputs).unsqueeze(0)  # 1,N,C

        ## get spe gt & channel info
        self.spe_gt = points[:, :, 3:6]
        self.channel_info = points[:, :, 6].squeeze()

        self.patch_len = torch.tensor(self.patch_len, device=points.device)
        mask = self._generate_mask(inputs)

        xyz, features = self._split_point_feats(points)  # B N 3 and B N C / N 3 and N C

        mask_idx = (
            mask_knn(
                self.k,
                points,
                points,
                mask,
                mask,
                self.patch_len,
                self.patch_len,
            )
            .transpose(1, 2)
            .contiguous()
        )
        self.spe_pred = []
        features_spa = features[:, 3:, :]
        intensity = features[:, :3, :].clone()
        # print("----------------")
        # print(intensity[0, :5, 0])
        feature_spe = self.CrossAttention1(xyz, features_spa, intensity, mask_idx)
        # print(feature_spe[0, :5, 0])
        self.spe_pred.append(feature_spe.permute(0, 2, 1))
        features[:, :3, :] = feature_spe
        feature_spe = self.CrossAttention2(xyz, features, intensity, mask_idx)
        # print(feature_spe[0, :5, 0])
        self.spe_pred.append(feature_spe.permute(0, 2, 1))
        features[:, :3, :] = (features[:, :3, :] + feature_spe) / 2

        features = features.permute(0, 2, 1)

        # import pdb
        # pdb.set_trace()
        points = torch.cat([xyz, features], dim=-1)
        inputs = []
        cml = 0
        for length in self.patch_len:
            inputs.append(points[:, cml : cml + length, :].squeeze(0))
            cml += length
            # vis(inputs[-1][:, :3].cpu().numpy(), inputs[-1][:, 3:6].cpu().numpy())
        torch.cuda.empty_cache()
        return inputs

    def loss(self):
        self.spe_gt = self.spe_gt.squeeze()
        losses = dict()
        for i, spe_pred in enumerate(self.spe_pred):
            loss_all_channel = torch.abs(self.spe_gt - spe_pred.squeeze())
            loss_mask = torch.zeros_like(loss_all_channel)
            idx_x = torch.arange(0, loss_mask.shape[0])
            idx_y = self.channel_info.to(torch.int64).cpu()
            loss_mask[idx_x, idx_y] = 1
            loss_spe = torch.mean(torch.sum(loss_all_channel * loss_mask, dim=1))
            losses[f"loss_spe_{i}"] = loss_spe * 15
        return losses


@MODELS.register_module()
class CrossInterpolatePreP5(NNInterpolatePreP):
    def __init__(
        self,
        k,
    ):
        super(CrossInterpolatePreP5, self).__init__()
        self.k = k
        self.CrossAttention1 = CrossInterpolateBlock3(
            input_dims=3,
            output_dims=1,
            v_dims=1,
            share_planes=1,
            num_sample=k,
        )
        self.CrossAttention2 = CrossInterpolateBlock3(
            input_dims=3,
            output_dims=1,
            v_dims=1,
            share_planes=1,
            num_sample=k,
        )
        self.CrossAttention3 = CrossInterpolateBlock3(
            input_dims=3,
            output_dims=1,
            v_dims=1,
            share_planes=1,
            num_sample=k,
        )
        self.CrossAttention = CrossInterpolateBlock3(
            input_dims=3,
            output_dims=3,
            v_dims=1,
            share_planes=1,
            num_sample=k,
        )

    def forward(self, inputs):
        self.patch_len = [x.shape[0] for x in inputs]
        points = torch.cat(inputs).unsqueeze(0)  # 1,N,C

        ## get spe gt & channel info
        self.spe_gt = points[:, :, 3:6]
        self.channel_info = points[:, :, 6].squeeze()

        self.patch_len = torch.tensor(self.patch_len, device=points.device)
        mask = self._generate_mask(inputs)

        xyz, features = self._split_point_feats(points)  # B N 3 and B N C / N 3 and N C

        mask_idx = (
            mask_knn(
                self.k,
                points,
                points,
                mask,
                mask,
                self.patch_len,
                self.patch_len,
            )
            .transpose(1, 2)
            .contiguous()
        )
        self.spe_pred = []
        features_spa = features[:, 3:, :]
        intensity = features[:, :3, :].clone()
        features[:, 0, :] = self.CrossAttention1(
            xyz, features_spa, intensity[:, 0:1, :], mask_idx[..., : self.k]
        )
        features[:, 1, :] = self.CrossAttention2(
            xyz, features_spa, intensity[:, 1:2, :], mask_idx[..., self.k : 2 * self.k]
        )
        features[:, 2, :] = self.CrossAttention3(
            xyz, features_spa, intensity[:, 2:, :], mask_idx[..., 2 * self.k :]
        )
        self.spe_pred.append(features[:, :3, :].permute(0, 2, 1))
        self_idx = knn(self.k, xyz, xyz, self.patch_len, self.patch_len)
        features[:, :3, :] = self.CrossAttention(
            xyz, features_spa, features[:, :3, :], self_idx
        )

        features = features.permute(0, 2, 1)

        # import pdb
        # pdb.set_trace()
        points = torch.cat([xyz, features], dim=-1)
        inputs = []
        cml = 0
        for length in self.patch_len:
            inputs.append(points[:, cml : cml + length, :].squeeze(0))
            cml += length
            # vis(inputs[-1][:, :3].cpu().numpy(), inputs[-1][:, 3:6].cpu().numpy())
        torch.cuda.empty_cache()
        return inputs

    def loss(self):
        self.spe_gt = self.spe_gt.squeeze()
        losses = dict()
        for i, spe_pred in enumerate(self.spe_pred):
            loss_all_channel = torch.abs(self.spe_gt - spe_pred.squeeze())
            loss_mask = torch.zeros_like(loss_all_channel)
            idx_x = torch.arange(0, loss_mask.shape[0])
            idx_y = self.channel_info.to(torch.int64).cpu()
            # 通道 mask
            loss_mask[idx_x, idx_y] = 1
            # 异常光谱值 mask
            loss_mask[self.spe_gt >= 0.9] = 0
            loss_spe = torch.mean(torch.sum(loss_all_channel * loss_mask, dim=1))
            losses[f"loss_spe_{i}"] = loss_spe
        return losses


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
        indices = indices.permute(0, 2, 1).contiguous()
        x_q, x_k, x_v = (
            self.linear_q(features),
            self.linear_k(features),
            intensity,
        )  # B,C,N

        p_r = gather(xyz, indices)
        x_k = gather(x_k, indices, True)
        x_v = gather(x_v, indices, True)

        p_r = self.linear_p(p_r - xyz.permute(0, 2, 1).unsqueeze(-1))
        tem_p = p_r

        w = x_k - x_q.unsqueeze(-1) + tem_p  # B,C,N,K
        w = self.linear_w(w)
        b, c, n, nsample = x_v.shape

        w = w.view(b, c, n, 3, -1)
        w = self.softmax(w)

        w = w.view(b, c, n, -1)

        w = w.permute(0, 2, 3, 1)  # B,N,K,C
        x_v = x_v.permute(0, 2, 3, 1)
        p_r = p_r.permute(0, 2, 3, 1)
        s = self.share_planes
        x = ((x_v).view(b, n, nsample, s, c // s) * w.unsqueeze(3)).sum(2).view(b, n, c)

        return x.permute(0, 2, 1).contiguous()


class CrossInterpolateBlock2(PointTranformerBlock):
    def __init__(self, input_dims, output_dims, v_dims, share_planes, num_sample):
        super(CrossInterpolateBlock2, self).__init__(
            input_dims=input_dims,
            output_dims=output_dims,
            share_planes=share_planes,
            num_sample=num_sample,
        )
        if input_dims != v_dims:
            del self.linear_v
            self.linear_v = ConvModule(
                v_dims, output_dims, 1, 1, bias=True, conv_cfg=dict(type="Conv1d")
            )
        pass

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
        indices = indices.permute(0, 2, 1).contiguous()
        x_q, x_k, x_v = (
            self.linear_q(features),
            self.linear_k(features),
            self.linear_v(intensity),
        )  # B,C,N

        p_r = gather(xyz, indices)
        x_k = gather(x_k, indices, True)
        x_v = gather(x_v, indices, True)

        p_r = self.linear_p(p_r - xyz.permute(0, 2, 1).unsqueeze(-1))
        tem_p = p_r

        w = x_k - x_q.unsqueeze(-1) + tem_p  # B,C,N,K
        w = self.linear_w(w)
        b, c, n, nsample = x_v.shape

        w = w.view(b, c, n, 3, -1)
        w = self.softmax(w)

        w = w.view(b, c, n, -1)

        w = w.permute(0, 2, 3, 1)  # B,N,K,C
        x_v = x_v.permute(0, 2, 3, 1)
        p_r = p_r.permute(0, 2, 3, 1)
        s = self.share_planes
        x = ((x_v).view(b, n, nsample, s, c // s) * w.unsqueeze(3)).sum(2).view(b, n, c)

        return x.permute(0, 2, 1).contiguous()


class CrossInterpolateBlock3(PointTranformerBlock):
    def __init__(self, input_dims, output_dims, v_dims, share_planes, num_sample):
        super(CrossInterpolateBlock3, self).__init__(
            input_dims=input_dims,
            output_dims=output_dims,
            share_planes=share_planes,
            num_sample=num_sample,
        )
        if input_dims != v_dims:
            del self.linear_v
            self.linear_v = ConvModule(
                v_dims, output_dims, 1, 1, bias=False, conv_cfg=dict(type="Conv1d")
            )
        del self.linear_p
        self.linear_p = nn.Sequential(
            ConvModule(
                4,
                4,
                1,
                1,
                bias=True,
                conv_cfg=dict(type="Conv2d"),
                norm_cfg=dict(type="BN2d"),
                act_cfg=dict(type="ReLU"),
            ),
            ConvModule(3, output_dims, 1, 1, bias=True, conv_cfg=dict(type="Conv2d")),
        )
        pass

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
        indices = indices.permute(0, 2, 1).contiguous()
        x_q, x_k, x_v = (
            self.linear_q(features),
            self.linear_k(features),
            self.linear_v(intensity),
        )  # B,C,N

        p_r = gather(xyz, indices)
        x_k = gather(x_k, indices, True)
        x_v = gather(x_v, indices, True)

        pos_diff = p_r - xyz.permute(0, 2, 1).unsqueeze(-1)  # B 3 N K
        pos_dist = torch.sqrt(torch.sum(pos_diff**2, dim=1, keepdim=True) + 1e-6)
        pos_features = torch.cat([pos_diff, pos_dist], dim=1)
        gate = self.linear_p(pos_features)
        gate = torch.bernoulli(f.sigmoid(gate)) * (-(10**10))

        w = x_k - x_q.unsqueeze(-1)  # B,C,N,K
        w = self.linear_w(w)
        w = w + gate

        w = self.softmax(w)

        w = w.permute(0, 2, 3, 1)  # B,N,K,C
        x_v = x_v.permute(0, 2, 3, 1)
        x = torch.sum(x_v * w, dim=2)

        return x.permute(0, 2, 1).contiguous()
