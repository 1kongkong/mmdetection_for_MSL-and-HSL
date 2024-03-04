import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
from torch import Tensor

# from mmcv.ops.group_points import grouping_operation
from .kernel_points import load_kernels
from mmcv.cnn import ConvModule
from mmcv.cnn.bricks.activation import build_activation_layer
from mmcv.cnn.bricks.norm import build_norm_layer
from my_tools.gather import gather


class KPConv(nn.Module):
    def __init__(
        self,
        kernel_size: int,
        input_dim: int,
        output_dim: int,
        radius: int = None,
        neighbor_num: int = 20,
        weight_norm: bool = False,
        dimension: int = 3,
        bias: bool = False,
        inf=1e6,
        eps=1e-9,
    ):
        """Initialize parameters for KPConv.

        Modified from [KPConv-PyTorch](https://github.com/HuguesTHOMAS/KPConv-PyTorch).

        Deformable KPConv is not supported.

        Args:
             kernel_size: Number of kernel points.
             input_dim: dimension of input features.
             output_dim: dimension of output features.
             radius: radius used for kernel point init.
             sigma: influence radius of each kernel point.
             dimension: dimension of the point space.
             bias: use bias or not (default: False)
             inf: value of infinity to generate the padding point
             eps: epsilon for gaussian influence
        """
        super(KPConv, self).__init__()

        # Save parameters
        self.kernel_size = kernel_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.radius = radius
        self.K = neighbor_num
        self.weight_norm = weight_norm
        self.dimension = dimension
        self.has_bias = bias

        self.inf = inf
        self.eps = eps

        # Initialize weights
        self.weights = nn.Parameter(
            torch.zeros(self.kernel_size, input_dim, output_dim)
        )
        if self.has_bias:
            self.bias = nn.Parameter(torch.zeros(self.output_dim))

        # Reset parameters
        self._reset_parameters()

        # Initialize kernel points
        self.register_buffer("kernel_points", self._initialize_kernel_points())

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.has_bias:
            nn.init.zeros_(self.bias)

    def _initialize_kernel_points(self):
        """
        Initialize the kernel point positions in a sphere
        :return: the tensor of kernel points
        """
        # Create one kernel disposition (as numpy array). Choose the KP distance to center thanks to the KP extent
        if self.radius:
            kernel_points = load_kernels(
                self.radius, self.kernel_size, dimension=self.dimension, fixed="center"
            )
        else:
            kernel_points = load_kernels(
                1, self.kernel_size, dimension=self.dimension, fixed="center"
            )
        # if kernel_points is None:
        return torch.from_numpy(kernel_points).float()

    def forward(
        self, points_xyz, features, center_xyz, neighbor_indices
    ) -> Tuple[Tensor]:
        """
        input:
            points_xyz -> [B,N,3]
            features -> [B,D,N]
            center_xyz -> [B,M,3]
            neighbor_num -> [B,M,K]
        output:
            output_feats->B,out_dim,N
        """

        # Get neighbor points
        # input: (B, C, N) , (B, M, K)
        # return: (B, C, M, K)
        features_trans = features.transpose(1, 2).contiguous()
        points = torch.cat([points_xyz, features_trans], dim=-1)
        shadow_point = torch.zeros_like(points[:, :1, :])
        shadow_point[:, :, :3] -= 1e3
        points = torch.cat([points, shadow_point], dim=1)
        neighbors = gather(points, neighbor_indices)  # B 3+N M

        neighbors_xyz = (
            neighbors[:, :3, ...].permute(0, 2, 3, 1).contiguous()
        )  # B, M, K, 3
        neighbors_feats = (
            neighbors[:, 3:, ...].permute(0, 2, 3, 1).contiguous()
        )  # B, M, K, C

        neighbors_xyz = neighbors_xyz - center_xyz.unsqueeze(2)
        dist = torch.sqrt(torch.sum(neighbors_xyz**2, dim=3))

        # Get all difference matrices [n_points, n_neighbors, n_kernel_points, dim]
        m_dist = torch.max(dist, dim=2, keepdim=True)[0]  # B N 1
        kernel_points = self.kernel_points  # k 3
        if not self.radius:
            # 1 1 k 3 * B N 1 1 -> B N k 3
            kernel_points = kernel_points[None, None, :, :] * m_dist[:, :, :, None]
            # B,N,K,k,3
            differences = neighbors_xyz.unsqueeze(3) - kernel_points.unsqueeze(2)
        else:
            differences = neighbors_xyz.unsqueeze(3) - kernel_points  # B,N,K,k,3

        # Get the square distances [batchsize, n_points, n_neighbors, n_kernel_points]
        sq_distances = torch.sum(differences**2, dim=4)  # B,N,K,k

        # Get Kernel point influences [batchsize, n_points, n_kernel_points, n_neighbors]
        # Influence decrease linearly with the distance, and get to zero when d = sigma.
        if not self.radius:
            neighbor_weights = torch.clamp(
                1 - torch.sqrt(sq_distances) / (m_dist.unsqueeze(-1) / 2.1 + 1e-5),
                min=0.0,
            )
        else:
            neighbor_weights = torch.clamp(
                1 - torch.sqrt(sq_distances) / (self.radius / 2.1 + 1e-5), min=0.0
            )

        # weight normalization
        if self.weight_norm:
            neighbor_weights = neighbor_weights / (
                torch.sum(neighbor_weights, dim=-2, keepdim=True) + 1e-5
            )  # B,N,K,k
            mask_ = torch.sum(neighbor_weights, dim=2)  # B,N,k
            mask = torch.sum(torch.gt(mask_, 1e-5), dim=-1, keepdim=False)  # B,N

        neighbor_weights = torch.transpose(neighbor_weights, 2, 3)  # B,N,k,K

        # Apply distance weights [n_points, n_kernel_points, input_dim]
        weighted_feats = torch.matmul(neighbor_weights, neighbors_feats)  # B N k D

        # Apply network weights [n_kernel_points, n_points, output_dim]
        weighted_feats = weighted_feats.permute(0, 2, 1, 3).contiguous()  # B,k,N,D

        # kernel_outputs = self.conv(weighted_feats)  # B,out_dim,N,k
        kernel_outputs = torch.matmul(weighted_feats, self.weights)  # B k N out_dim

        # Convolution sum [n_points, output_dim]
        output_feats = torch.sum(kernel_outputs, dim=1, keepdim=False)  # B,N,out_dim

        # normalization term.
        if self.weight_norm:
            output_feats = output_feats / (mask.unsqueeze(-1) + 1e-5)
            return output_feats.permute(0, 2, 1).contiguous()  # B,out_dim,N

        # neighbor_feats_sum = torch.sum(neighbors_feats, dim=-1)  # B,N,K
        # neighbor_num = torch.sum(torch.gt(neighbor_feats_sum, 0.0), dim=-1)  # B N
        # neighbor_num = torch.max(neighbor_num, torch.ones_like(neighbor_num))
        # output_feats = output_feats / neighbor_num.unsqueeze(-1)  # B N 1

        return output_feats.permute(0, 2, 1).contiguous()  # B,out_dim,N

    def __repr__(self):
        repr_str = self.__class__.__name__ + "("
        repr_str += "kernel_size: {}, input_dim: {}, output_dim: {}, bias: {}".format(
            self.kernel_size, self.input_dim, self.output_dim, self.has_bias
        )
        repr_str += ")"
        return repr_str


class KPConvBlock(nn.Module):
    def __init__(
        self,
        kernel_size,
        input_dim,
        output_dim,
        radius=None,
        neighbor_num=20,
        weight_norm=False,
        norm_cfg: Optional[Dict] = dict(type="BN1d"),
        act_cfg: Optional[Dict] = dict(type="ReLU"),
    ):
        r"""Initialize a KPConv block with ReLU and BatchNorm.

        Args:
            kernel_size: number of kernel points
            input_dim: dimension input features
            output_dim: dimension input features

        """
        super(KPConvBlock, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.radius = radius

        self.KPConv = KPConv(
            kernel_size,
            input_dim,
            output_dim,
            radius,
            neighbor_num,
            weight_norm,
            bias=False,
        )
        self.norm_name, self.norm_layer = build_norm_layer(norm_cfg, output_dim)
        self.leaky_relu = build_activation_layer(act_cfg)

    def forward(self, points_xyz, features, center_xyz, neighbor_indices):
        x = self.KPConv(points_xyz, features, center_xyz, neighbor_indices)
        x = self.norm_layer(x)
        x = self.leaky_relu(x)

        return points_xyz, x, center_xyz


class KPResNetBlock(nn.Module):
    def __init__(
        self,
        kernel_size,
        input_dim,
        output_dim,
        radius=None,
        neighbor_num=20,
        weight_norm=False,
        strided=False,
        norm_cfg: Optional[Dict] = dict(type="BN1d"),
        act_cfg: Optional[Dict] = dict(type="ReLU"),
    ):
        r"""Initialize a ResNet bottleneck block.

        Args:
            kernel_size: number of kernel points
            input_dim: dimension input features
            output_dim: dimension input features
            radius: convolution radius
            sigma: influence radius of each kernel point
            group_norm: group number for GroupNorm
            strided: strided or not
        """
        super(KPResNetBlock, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.strided = strided
        self.radius = radius

        hidden_dim = output_dim // 4

        if input_dim != hidden_dim:
            self.unary1 = ConvModule(
                input_dim,
                hidden_dim,
                kernel_size=1,
                stride=1,
                bias=False,
                conv_cfg=dict(type="Conv1d"),
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            )
        else:
            self.unary1 = nn.Identity()

        self.KPConv = KPConv(
            kernel_size,
            hidden_dim,
            hidden_dim,
            radius,
            neighbor_num,
            weight_norm,
            bias=False,
        )
        self.norm_name, self.norm_layer = build_norm_layer(norm_cfg, hidden_dim)

        self.unary2 = ConvModule(
            hidden_dim,
            output_dim,
            kernel_size=1,
            stride=1,
            bias=False,
            conv_cfg=dict(type="Conv1d"),
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

        if input_dim != output_dim:
            self.unary_shortcut = ConvModule(
                input_dim,
                output_dim,
                kernel_size=1,
                stride=1,
                bias=False,
                conv_cfg=dict(type="Conv1d"),
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            )
        else:
            self.unary_shortcut = nn.Identity()

        self.leaky_relu = build_activation_layer(act_cfg)

    def _maxpool(self, features, neighbor_indices):
        features = torch.cat(
            [features, torch.zeros_like(features[:, :, :1]) - 1e4], dim=2
        )
        neighbor_features = gather(features, neighbor_indices, True)
        return neighbor_features.max(dim=-1)[0]

    def forward(
        self,
        points_xyz,
        features,
        center_xyz,
        neighbor_indices,
    ):
        x = self.unary1(features)
        x = self.KPConv(points_xyz, x, center_xyz, neighbor_indices)
        x = self.norm_layer(x)
        x = self.leaky_relu(x)
        x = self.unary2(x)

        if self.strided:
            shortcut = self._maxpool(features, neighbor_indices)
        else:
            shortcut = features
        shortcut = self.unary_shortcut(shortcut)

        x = x + shortcut
        x = self.leaky_relu(x)

        return points_xyz, x, center_xyz


# if __name__ == '__main__':
# model = KPConv(15, 4, 6, 1000, 0.4)
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# cloud1 = 10*torch.randn(1, 10, 3).to(device)
# cloud2 = 10*torch.randn(1, 20, 3).to(device)
# feat = 1000*torch.randn(1, 20, 4).to(device)
# out = model(feat, cloud1, cloud2)
