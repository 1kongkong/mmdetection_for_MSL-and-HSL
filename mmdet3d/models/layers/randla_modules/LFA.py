import torch
import torch.nn as nn
from .share_mlp import SharedMLP
from mmcv.ops.knn import knn
from mmcv.ops.group_points import grouping_operation


class LocalFeatureAggregation(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        num_neighbors,
    ):
        super(LocalFeatureAggregation, self).__init__()

        self.num_neighbors = num_neighbors

        self.mlp1 = SharedMLP(
            in_channel, out_channel // 2, activation_fn=nn.LeakyReLU(0.2)
        )
        self.mlp2 = SharedMLP(out_channel, 2 * out_channel)
        self.shortcut = SharedMLP(in_channel, 2 * out_channel, bn=True)

        self.lse1 = LocalSpatialEncoding(out_channel // 2, num_neighbors)
        self.lse2 = LocalSpatialEncoding(out_channel // 2, num_neighbors)

        self.pool1 = AttentivePooling2(out_channel, out_channel // 2)
        self.pool2 = AttentivePooling2(out_channel, out_channel)

        self.lrelu = nn.LeakyReLU()

    def forward(self, coords, features):
        r"""
        Forward pass

        Parameters
        ----------
        coords: torch.Tensor, shape (B, N, 3)
            coordinates of the point cloud
        features: torch.Tensor, shape (B, d_in, N, 1)
            features of the point cloud

        Returns
        -------
        torch.Tensor, shape (B, 2*d_out, N, 1)
        """
        idx = (
            knn(self.num_neighbors, coords, coords, False).transpose(1, 2).contiguous()
        )

        x = self.mlp1(features)

        x = self.lse1(coords, x, idx)
        x = self.pool1(x)

        x = self.lse2(coords, x, idx)
        x = self.pool2(x)

        return self.lrelu(self.mlp2(x) + self.shortcut(features))


class LocalSpatialEncoding(nn.Module):
    def __init__(self, d, num_neighbors):
        super(LocalSpatialEncoding, self).__init__()

        self.num_neighbors = num_neighbors
        self.mlp = SharedMLP(10, d, bn=True, activation_fn=nn.ReLU())

    def forward(self, coords, features, idx):
        r"""
        Forward pass

        Parameters
        ----------
        coords: torch.Tensor, shape (B, N, 3)
            coordinates of the point cloud
        features: torch.Tensor, shape (B, d, N, 1)
            features of the point cloud
        neighbors: tuple

        Returns
        -------
        torch.Tensor, shape (B, 2*d, N, K)
        """
        # finding neighboring points
        B, N, K = idx.size()
        coords = coords.permute(0, 2, 1).contiguous()
        neighbors = grouping_operation(coords, idx)
        # idx(B, N, K), coords(B, N, 3)
        # neighbors[b, i, n, k] = coords[b, idx[b, n, k], i] = extended_coords[b, i, extended_idx[b, i, n, k], k]
        extended_coords = coords.unsqueeze(-1).expand(B, 3, N, K)
        # if USE_CUDA:
        #     neighbors = neighbors.cuda()
        # relative point position encoding
        dist = torch.sqrt(
            torch.sum((extended_coords - neighbors) ** 2, dim=1, keepdim=True)
        )
        concat = torch.cat(
            (extended_coords, neighbors, extended_coords - neighbors, dist),
            dim=1,
        )
        return torch.cat((self.mlp(concat), features.expand(B, -1, N, K)), dim=-3)


class AttentivePooling2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentivePooling2, self).__init__()

        self.channel_fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // 8, in_channels, 1, bias=False),
        )
        self.channel_sigmoid = nn.Sigmoid()

        self.spatial_fc = nn.Sequential(
            nn.Conv2d(2, 1, 1, bias=False),
        )
        self.spatial_sigmoid = nn.Sigmoid()

        self.mlp = SharedMLP(
            in_channels, out_channels, bn=True, activation_fn=nn.ReLU()
        )

    def forward(self, x):
        r"""
        Forward pass

        Parameters
        ----------
        x: torch.Tensor, shape (B, d_in, N, K)

        Returns
        -------
        torch.Tensor, shape (B, d_out, N, 1)
        """
        # computing attention scores

        avg_out = self.channel_fc(torch.mean(x, dim=3, keepdim=True))
        max_out = self.channel_fc(torch.max(x, dim=3, keepdim=True)[0])
        channel_scores = self.channel_sigmoid(avg_out + max_out)
        x = torch.mul(x, channel_scores)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        spatial_scores = self.spatial_sigmoid(
            self.spatial_fc(torch.cat((avg_out, max_out), dim=1))
        )
        x = torch.mul(x, spatial_scores)

        # scores = self.score_fn(x.permute(0,2,3,1)).permute(0,3,1,2)

        # sum over the neighbors
        features = torch.sum(x, dim=-1, keepdim=True)  # shape (B, d_in, N, 1)

        return self.mlp(features)


class AttentivePooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentivePooling, self).__init__()

        self.score_fn = nn.Sequential(
            nn.Linear(in_channels, in_channels, bias=False), nn.Softmax(dim=-2)
        )
        self.mlp = SharedMLP(
            in_channels, out_channels, bn=True, activation_fn=nn.ReLU()
        )

    def forward(self, x):
        r"""
        Forward pass

        Parameters
        ----------
        x: torch.Tensor, shape (B, d_in, N, K)

        Returns
        -------
        torch.Tensor, shape (B, d_out, N, 1)
        """
        # computing attention scores
        scores = self.score_fn(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # sum over the neighbors
        features = torch.sum(scores * x, dim=-1, keepdim=True)  # shape (B, d_in, N, 1)

        return self.mlp(features)
