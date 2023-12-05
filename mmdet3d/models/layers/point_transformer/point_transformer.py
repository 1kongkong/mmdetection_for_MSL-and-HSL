import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.ops.knn import knn
from mmcv.ops.group_points import grouping_operation


class PointTranformerBlock(nn.Module):
    def __init__(self, input_dims, output_dims, share_planes, num_sample):
        super(PointTranformerBlock, self).__init__()
        self.mid_dims = mid_dims = output_dims // 1
        self.output_dims = output_dims
        self.share_planes = share_planes
        self.num_sample = num_sample
        self.linear_q = ConvModule(
            input_dims, mid_dims, 1, 1, bias=True, conv_cfg=dict(type="Conv1d")
        )
        self.linear_k = ConvModule(
            input_dims, mid_dims, 1, 1, bias=True, conv_cfg=dict(type="Conv1d")
        )
        self.linear_v = ConvModule(
            input_dims, mid_dims, 1, 1, bias=True, conv_cfg=dict(type="Conv1d")
        )
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
            ConvModule(3, output_dims, 1, 1, bias=True, conv_cfg=dict(type="Conv2d")),
        )
        self.linear_w = nn.Sequential(
            nn.BatchNorm2d(mid_dims),
            nn.ReLU(inplace=True),
            ConvModule(
                mid_dims,
                mid_dims // share_planes,
                1,
                1,
                bias=True,
                conv_cfg=dict(type="Conv2d"),
                norm_cfg=dict(type="BN2d"),
                act_cfg=dict(type="ReLU"),
            ),
            ConvModule(
                mid_dims // share_planes,
                output_dims // share_planes,
                1,
                1,
                bias=True,
                conv_cfg=dict(type="Conv2d"),
            ),
        )
        self.softmax = nn.Softmax(dim=-1)

    def _knn_query(self, points):
        """
        Args:
            points (Tensor): (B,N,3)
        Returns:
            idx (Tensor): (B,N,k)
        """
        idx = knn(self.num_sample, points, points, False).transpose(1, 2).contiguous()
        return idx

    def forward(self, xyz, features):
        """
        args:
            xyz: [B,N,3]
            features: [B,C1,N]
        return:
            x: [B,C2,N]
        """
        x_q, x_k, x_v = (
            self.linear_q(features),
            self.linear_k(features),
            self.linear_v(features),
        )  # B,C,N
        indices = self._knn_query(xyz)

        p_r = grouping_operation(xyz.permute(0, 2, 1).contiguous(), indices)  # B,3,N,K
        x_k = grouping_operation(x_k, indices)  # B,C,N,K
        x_v = grouping_operation(x_v, indices)  # B,C,N,K

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
