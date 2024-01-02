import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional
import numpy as np

from mmcv.cnn import ConvModule
from mmcv.cnn.bricks.activation import build_activation_layer
from mmcv.cnn.bricks.norm import build_norm_layer

# from mmcv.ops.group_points import grouping_operation
from my_tools.gather import gather


class cross_att_fusion_block(nn.Module):
    def __init__(
        self,
        input_dim: int,
    ):
        """ """
        super(cross_att_fusion_block, self).__init__()
        self.input_dim = input_dim
        self.spa_MHA = CrossAttention(self.input_dim)
        self.spe_MHA = CrossAttention(self.input_dim)

    def forward(self, spa, spe, neighbor_indices):
        """
        spa: [B,C,N]
        spe: [B,C,N]
        """
        dims = spa.shape[1]
        feature = torch.cat([spa, spe], dim=1)
        # neighbors_feats = grouping_operation(feature, neighbor_indices)
        neighbors_feats = gather(feature, neighbor_indices, True)
        feature_q = feature.permute(0, 2, 1).contiguous()
        neighbors_feats = neighbors_feats.permute(0, 2, 3, 1).contiguous()  # B, N, K, C
        spa_feature = self.spa_MHA(feature_q[..., :dims], neighbors_feats[..., dims:])
        spe_feature = self.spe_MHA(feature_q[..., dims:], neighbors_feats[..., :dims])

        fusion_feature = torch.cat([spa_feature, spe_feature], dim=1)

        return feature + fusion_feature


class CrossAttention(nn.Module):
    def __init__(self, input_dim):
        super(CrossAttention, self).__init__()

        self.d_k = input_dim
        self.W_Q = nn.Linear(input_dim, input_dim, bias=False)
        self.W_K = nn.Linear(input_dim, input_dim, bias=False)
        self.W_V = nn.Linear(input_dim, input_dim)
        # self.fc = nn.Linear(input_dim, input_dim)

    def forward(self, feature_spa, feature_spe):
        """
        feature_q: [batch_size, num_points, channel]
        feature_k: [batch_size, num_points, k_neighbors, channel]
        """
        B, N, k, C = feature_spe.shape

        Q = self.W_Q(feature_spa)
        K = self.W_K(feature_spe)
        V = self.W_V(feature_spe)

        context, attn = ScaledDotProductAttention()(
            Q, K, V
        )  # context: [batch_size, n_heads, len_q, d_v]
        # attn: [batch_size, n_heads, len_q, len_k]
        # context: [batch_size, num_points, n_heads, 1, d_v]
        # context = context.squeeze(-2).reshape(B, N, self.n_heads * self.d_k)
        # output = self.fc(context)  # [batch_size, num_points, k_neighbors, input_dim]
        return context.permute(0, 2, 1)


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V):  #
        """
        Q: [batch_size, num_points, d_k]
        K: [batch_size, num_points, k_neighbors, d_k]
        V: [batch_size, num_points, k_neighbors, d_v]
        """
        # scores : [batch_size, num_points, len_k, k_neighbors, k_neighbors]
        scores = torch.matmul(Q.unsqueeze(2), K.transpose(-1, -2)) / np.sqrt(
            Q.shape[-1]
        )
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V).squeeze(
            2
        )  # [batch_size, num_points, n_heads, 1, d_v]
        return context, attn


class cross_att_fusion_block_2(nn.Module):
    def __init__(
        self,
        input_dim: int,
    ):
        """ """
        super(cross_att_fusion_block_2, self).__init__()
        self.input_dim = input_dim
        self.n_heads = 2
        self.spa_MHA = PointTransformerLayer(self.input_dim)
        self.spe_MHA = PointTransformerLayer(self.input_dim)

    def forward(self, coord, spa, spe, neighbor_indices):
        """
        coord: [B,N,3]
        spa: [B,C,N]
        spe: [B,C,N]
        """
        spa_feature = self.spa_MHA(coord, spa, spe, neighbor_indices)
        spe_feature = self.spe_MHA(coord, spe, spa, neighbor_indices)
        spa_feature = spa + spa_feature
        spe_feature = spe + spe_feature

        fusion_feature = torch.cat([spa_feature, spe_feature], dim=1)

        return fusion_feature


class PointTransformerLayer(nn.Module):
    def __init__(self, in_planes, share_planes=16):
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
        # p_r = grouping_operation(coord.permute(0, 2, 1).contiguous(), neighbor_indices)  # B,3,N,K
        # x_k = grouping_operation(x_k, neighbor_indices)  # B,C,N,K
        # x_v = grouping_operation(x_v, neighbor_indices)  # B,C,N,K
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


class fusion_block(nn.Module):
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
        super(fusion_block, self).__init__()
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
