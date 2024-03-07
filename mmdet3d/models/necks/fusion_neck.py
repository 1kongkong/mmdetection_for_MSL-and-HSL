from mmengine.model import BaseModule
import torch
import numpy as np
from torch import nn as nn
from typing import Dict, Optional
from mmcv.cnn import ConvModule

from mmdet3d.registry import MODELS
from my_tools.gather import gather


@MODELS.register_module()
class VectorCrossAttentionNeck(BaseModule):
    def __init__(self, input_dims, share_planes):
        """
        Args:
            input_dims(List):
            share_planes(List):
        """
        super(VectorCrossAttentionNeck, self).__init__()
        assert len(input_dims) == len(share_planes)
        self.input_dims = input_dims
        self.spa_MHA = nn.ModuleList()
        self.spe_MHA = nn.ModuleList()

        for i in range(len(input_dims)):
            self.spa_MHA.append(PointTransformerLayer(input_dims[i], share_planes[i]))
            self.spe_MHA.append(PointTransformerLayer(input_dims[i], share_planes[i]))

    def _extract_feature(self, inputs):
        coord = inputs["points"]
        spa_feat = inputs.pop("spa_feature")
        spe_feat = inputs.pop("spe_feature")
        neighbor_indices = inputs["self_index"]

        return coord, spa_feat, spe_feat, neighbor_indices

    def forward(self, inputs):
        """
        coord: [B,N,3]
        spa: [B,C,N]
        spe: [B,C,N]
        """
        coord, spa, spe, neighbor_indices = self._extract_feature(inputs)
        fusion_feature = []

        for i in range(len(self.input_dims)):
            spa_feature = self.spa_MHA[i](coord[i], spa[i], spe[i], neighbor_indices[i])
            spe_feature = self.spe_MHA[i](coord[i], spe[i], spa[i], neighbor_indices[i])
            spa_feature = spa[i] + spa_feature
            spe_feature = spe[i] + spe_feature
            fusion_feature.append(torch.cat([spa_feature, spe_feature], dim=1))
        inputs["features"] = fusion_feature

        return inputs


@MODELS.register_module()
class ConcatNeck(BaseModule):
    def __init__(self, input_dims):
        """
        Args:
            input_dims(List):
            share_planes(List):
        """
        super(ConcatNeck, self).__init__()
        self.input_dims = input_dims

    def _extract_feature(self, inputs):
        coord = inputs["points"]
        spa_feat = inputs.pop("spa_feature")
        spe_feat = inputs.pop("spe_feature")
        neighbor_indices = inputs["self_index"]

        return coord, spa_feat, spe_feat, neighbor_indices

    def forward(self, inputs):
        """
        coord: [B,N,3]
        spa: [B,C,N]
        spe: [B,C,N]
        """
        coord, spa, spe, neighbor_indices = self._extract_feature(inputs)
        fusion_feature = []

        for i in range(len(self.input_dims)):
            fusion_feature.append(torch.cat([spa[i], spe[i]], dim=1))
        inputs["features"] = fusion_feature

        return inputs


@MODELS.register_module()
class VectorSelfAttentionNeck(BaseModule):
    def __init__(self, input_dims, share_planes):
        """
        Args:
            input_dims(List):
            share_planes(List):
        """
        super(VectorSelfAttentionNeck, self).__init__()
        assert len(input_dims) == len(share_planes)
        self.input_dims = input_dims
        self.spa_MHA = nn.ModuleList()
        self.spe_MHA = nn.ModuleList()

        for i in range(len(input_dims)):
            self.spa_MHA.append(PointTransformerLayer(input_dims[i], share_planes[i]))
            self.spe_MHA.append(PointTransformerLayer(input_dims[i], share_planes[i]))

    def _extract_feature(self, inputs):
        coord = inputs["points"]
        spa_feat = inputs.pop("spa_feature")
        spe_feat = inputs.pop("spe_feature")
        neighbor_indices = inputs["self_index"]

        return coord, spa_feat, spe_feat, neighbor_indices

    def forward(self, inputs):
        """
        coord: [B,N,3]
        spa: [B,C,N]
        spe: [B,C,N]
        """
        coord, spa, spe, neighbor_indices = self._extract_feature(inputs)
        fusion_feature = []

        for i in range(len(self.input_dims)):
            spa_feature = self.spa_MHA[i](coord[i], spa[i], spa[i], neighbor_indices[i])
            spe_feature = self.spe_MHA[i](coord[i], spe[i], spe[i], neighbor_indices[i])
            spa_feature = spa[i] + spa_feature
            spe_feature = spe[i] + spe_feature
            fusion_feature.append(torch.cat([spa_feature, spe_feature], dim=1))
        inputs["features"] = fusion_feature

        return inputs


class PointTransformerLayer(nn.Module):
    def __init__(self, in_planes, share_planes=4):
        super().__init__()
        out_planes = in_planes
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.linear_q = nn.Conv1d(in_planes, mid_planes, 1)
        self.linear_k = nn.Conv1d(in_planes, mid_planes, 1)
        self.linear_v = nn.Conv1d(in_planes, out_planes, 1)
        self.linear_p = nn.Sequential(
            nn.Conv2d(3, 3, 1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, out_planes, 1),
        )
        self.linear_w = nn.Sequential(
            nn.BatchNorm2d(mid_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_planes, mid_planes // share_planes, 1),
            nn.BatchNorm2d(mid_planes // share_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_planes // share_planes, out_planes // share_planes, 1),
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, coord, q, k, neighbor_indices) -> torch.Tensor:
        """
        coord: [B,N,3]
        q: [B,C,N]
        k: [B,C,N]
        neighbor_indices: [B,N,k]
        """
        x_q, x_k, x_v = self.linear_q(q), self.linear_k(k), self.linear_v(q)  # B,C,N
        p_r = gather(
            coord.permute(0, 2, 1).contiguous(), neighbor_indices, True
        )  # B,3,N,K
        x_k = gather(x_k, neighbor_indices, True)  # B,C,N,K
        x_v = gather(x_v, neighbor_indices, True)  # B,C,N,K

        p_r = self.linear_p(p_r)  # B,C,N,K

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
        return x.permute(0, 2, 1).contiguous()


@MODELS.register_module()
class CrossAttentionNeck(BaseModule):
    def __init__(self, input_dims):
        """
        Args:
            input_dim(List):
        """
        super(CrossAttentionNeck, self).__init__()
        self.input_dims = input_dims
        self.spa_MHA = nn.ModuleList()
        self.spe_MHA = nn.ModuleList()

        for i in range(len(input_dims)):
            self.spa_MHA.append(CrossAttention(input_dims[i]))
            self.spe_MHA.append(CrossAttention(input_dims[i]))

    def _extract_feature(self, inputs):
        coord = inputs["points"]
        spa_feat = inputs.pop("spa_feature")
        spe_feat = inputs.pop("spe_feature")
        neighbor_indices = inputs["self_index"]

        return coord, spa_feat, spe_feat, neighbor_indices

    def forward(self, inputs):
        """
        coord: [B,N,3]
        spa: [B,C,N]
        spe: [B,C,N]
        """
        coord, spa, spe, neighbor_indices = self._extract_feature(inputs)
        fusion_feature = []

        for i in range(len(self.input_dims)):
            spa_feature = self.spa_MHA[i](coord[i], spa[i], spe[i], neighbor_indices[i])
            spe_feature = self.spe_MHA[i](coord[i], spe[i], spa[i], neighbor_indices[i])
            spa_feature = spa[i] + spa_feature
            spe_feature = spe[i] + spe_feature
            fusion_feature.append(torch.cat([spa_feature, spe_feature], dim=1))
        inputs["features"] = fusion_feature

        return inputs


class CrossAttention(nn.Module):
    def __init__(self, input_dim):
        super(CrossAttention, self).__init__()

        self.W_Q = nn.Conv1d(input_dim, input_dim, 1, 1, bias=False)
        self.W_K = nn.Conv1d(input_dim, input_dim, 1, 1, bias=False)
        self.W_V = nn.Conv1d(input_dim, input_dim, 1, 1)
        self.pos = nn.Sequential(
            nn.Conv2d(3, 3, 1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, 1, 1),
        )
        self.scores = nn.Sequential(
            nn.Conv2d(2, 2, 1),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(2, 1, 1),
        )

    def forward(self, coord, feature_spa, feature_spe, neighbor_indices):
        """
        feature_q: [batch_size, num_points, channel]
        feature_k: [batch_size, num_points, channel]
        """
        coord = coord.permute(0, 2, 1).contiguous()
        Q = self.W_Q(feature_spa)  # B C N
        K = self.W_K(feature_spe)
        V = self.W_V(feature_spe)
        neighbor_coord = gather(coord, neighbor_indices, True)
        coord = coord.unsqueeze(-1)
        coord_diff = neighbor_coord - coord  # B 3 N K
        coord_gaussian = torch.exp(-2 * (coord_diff) ** 2)
        scores_coord = self.pos(coord_gaussian)

        K = gather(K, neighbor_indices, True)  # B C N K
        V = gather(V, neighbor_indices, True)  # B C N K

        Q = Q.permute(0, 2, 1).contiguous()  # B N C
        K = K.permute(0, 2, 3, 1).contiguous()  # B N K C

        scores_feat = torch.matmul(Q.unsqueeze(2), K.transpose(-1, -2)) / np.sqrt(
            Q.shape[-1]
        )  # B N 1 C x B N C K -> B N 1 K
        scores = torch.cat([scores_coord, scores_feat.permute(0, 2, 1, 3)], dim=1)
        scores = self.scores(scores)  # B 1 N K

        attn = nn.Softmax(dim=-1)(scores)
        fusion_feat = torch.sum(attn * V, dim=-1)
        return fusion_feat


@MODELS.register_module()
class SKNetNeck(BaseModule):
    def __init__(self, input_dims):
        super(SKNetNeck, self).__init__()

        self.input_dims = input_dims
        self.SKNet = nn.ModuleList()

        for i in range(len(input_dims)):
            self.SKNet.append(sknet_block(input_dims[i]))

    def _extract_feature(self, inputs):
        spa_feat = inputs.pop("spa_feature")
        spe_feat = inputs.pop("spe_feature")

        return spa_feat, spe_feat

    def forward(self, inputs):
        """
        coord: [B,N,3]
        spa: [B,C,N]
        spe: [B,C,N]
        """
        spa, spe = self._extract_feature(inputs)
        fusion_feature = []

        for i in range(len(self.input_dims)):
            feature = self.SKNet[i](spa[i], spe[i])
            fusion_feature.append(feature)
        inputs["features"] = fusion_feature

        return inputs


class sknet_block(nn.Module):
    def __init__(
        self,
        input_dim: int,
        r: int = 16,
        L: int = 32,
        norm_cfg: Optional[Dict] = dict(type="BN1d"),
        act_cfg: Optional[Dict] = dict(type="LeakyReLU", negative_slope=0.1),
    ):
        """Constructor
        Args:
            input_dim: input channel dimensionality.
            r: the radio for compute d, the length of z.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(sknet_block, self).__init__()
        d = max(input_dim // r, L)
        self.M = 2
        self.input_dim = input_dim

        self.downfc = ConvModule(
            input_dim,
            d,
            1,
            stride=1,
            bias=False,
            conv_cfg=dict(type="Conv1d"),
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.upfc = ConvModule(
            d,
            4 * input_dim,
            1,
            stride=1,
            bias=True,
            conv_cfg=dict(type="Conv1d"),
            norm_cfg=None,
            act_cfg=None,
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, spa, spe):
        B, C, N = spa.shape
        fea_U = spa + spe
        fea_s = torch.mean(fea_U, dim=-1, keepdim=True)
        fea_z = self.downfc(fea_s)
        fea_att = self.upfc(fea_z)
        fea_att = fea_att.reshape(B, 4, C, -1)
        attention_vectors_0 = self.softmax(fea_att[:, :2, :, :])
        attention_vectors_1 = self.softmax(fea_att[:, 2:, :, :])
        spa_att_0 = attention_vectors_0[:, :1, :, :].squeeze(1)
        spe_att_0 = attention_vectors_0[:, 1:, :, :].squeeze(1)
        spa_att_1 = attention_vectors_1[:, :1, :, :].squeeze(1)
        spe_att_1 = attention_vectors_1[:, 1:, :, :].squeeze(1)
        fea_v_0 = spa * spa_att_0 + spe * spe_att_0
        fea_v_1 = spa * spa_att_1 + spe * spe_att_1
        fea_v = torch.cat([fea_v_0, fea_v_1], dim=1)
        return fea_v
