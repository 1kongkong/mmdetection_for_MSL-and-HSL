import numpy as np
from typing import Dict, List, Sequence, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
from torch import Tensor
from mmdet3d.registry import MODELS
from mmcv.cnn import ConvModule
from mmcv.ops.knn import knn
from mmcv.ops.group_points import grouping_operation
from mmdet3d.models.layers.randla_modules import SharedMLP, LocalFeatureAggregation


@MODELS.register_module()
class RandLANetBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_points: Sequence[int] = (8192, 2048, 512, 128, 32),
        num_samples: Sequence[int] = (16, 16, 16, 16, 16),
        enc_channels: Sequence[Sequence[int]] = (
            (8, 16),
            (32, 64),
            (128, 128),
            (256, 256),
        ),
    ):
        super(RandLANetBackbone, self).__init__()
        self.in_channels = in_channels
        self.num_points = num_points
        self.num_samples = num_samples
        self.num_enc = len(enc_channels)
        self.Enc_modules = nn.ModuleList()

        self.start_fc = ConvModule(
            in_channels,
            enc_channels[0][0],
            1,
            1,
            conv_cfg=dict(type="Conv1d"),
            norm_cfg=dict(type="BN1d"),
        )
        for enc_index in range(self.num_enc):
            cur_enc_channels = list(enc_channels[enc_index])
            self.Enc_modules.append(
                LocalFeatureAggregation(
                    in_channel=cur_enc_channels[0],
                    out_channel=cur_enc_channels[1],
                    num_neighbors=num_samples[enc_index],
                )
            )

        self.mlp = SharedMLP(512, 512, activation_fn=nn.ReLU())

    def _random_sample(self, points):
        permutation = torch.randperm(points.size(1))
        points_random = points[:, permutation, :]
        points_downsample = [points]
        for num_point in self.num_points[1:]:
            points_downsample.append(points_random[:, :num_point, :].contiguous())
        return points_downsample

    def _split_point_feats(self, points: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """Split coordinates and features of input points.

        Args:
            points (torch.Tensor): Point coordinates with features,
                with shape (B, N, 3 + input_feature_dim).

        Returns:
            torch.Tensor: Coordinates of input points.
            torch.Tensor: Features of input points.
        """
        xyz = points[..., 0:3].contiguous()
        if points.size(-1) > 3:
            features = points[..., 3:].transpose(1, 2).contiguous()
        else:
            features = None

        return xyz, features

    def forward(self, inputs):
        r"""
        Forward pass

        Parameters
        ----------
        inputs: torch.Tensor, shape (B, N, d_in)
            input points

        Returns
        -------
        torch.Tensor, shape (B, num_classes, N)
            segmentation scores for each point
        """
        N = inputs.size(1)

        coords, features = self._split_point_feats(inputs)

        features = self.start_fc(features)  # shape (B, d, N, 1)
        features = features.unsqueeze(-1)

        # <<<<<<<<<< ENCODER
        feature_stack = []
        points_downsample = self._random_sample(coords)

        for i, lfa in enumerate(self.Enc_modules):
            features = lfa(points_downsample[i], features)
            feature_stack.append(features.clone())
            features = features[:, :, : self.num_points[i + 1]]

        # # >>>>>>>>>> ENCODER

        features = self.mlp(features)
        feature_stack.append(features)

        res = {
            "points": points_downsample,
            "features": feature_stack,
        }

        return res
