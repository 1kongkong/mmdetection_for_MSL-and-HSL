# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Sequence

from torch import Tensor, nn

from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptMultiConfig
from mmdet3d.models.layers.point_transformer import PointTransfomerEncModule
from .base_pointnet import BasePointNet


@MODELS.register_module()
class PointTransformerBackbone(BasePointNet):
    """PointTransformer.

    Args:
        in_channels (int): Input channels of point cloud.
        num_points (tuple[int]): The number of points which each SA
            module samples.
        num_samples (tuple[int]): The number of samples for ball
            query in each SA module.
        enc_channels (tuple[tuple[int]]): Out channels of each mlp in SA module.

    """

    def __init__(
        self,
        in_channels: int,
        num_points: Sequence[int] = (8192, 2048, 512, 128, 32),
        num_samples: Sequence[int] = (8, 16, 16, 16, 16),
        enc_channels: Sequence[Sequence[int]] = (
            (32, 32),
            (64, 64, 64),
            (128, 128, 128, 128),
            (256, 256, 256, 256, 256, 256),
            (512, 512, 512),
        ),
        init_cfg: OptMultiConfig = None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.num_enc = len(enc_channels)
        self.Enc_modules = nn.ModuleList()
        enc_in_channel = in_channels  # number of channels without xyz

        for enc_index in range(self.num_enc):
            cur_enc_channels = list(enc_channels[enc_index])
            cur_enc_channels = [enc_in_channel] + cur_enc_channels
            enc_in_channel = cur_enc_channels[-1]

            self.Enc_modules.append(
                PointTransfomerEncModule(
                    num_point=num_points[enc_index],
                    num_sample=num_samples[enc_index],
                    channels=cur_enc_channels,
                    is_head=enc_index == 0,
                )
            )

    def forward(self, points: Tensor) -> Dict[str, List[Tensor]]:
        """Forward pass.

        Args:
            points (torch.Tensor): point coordinates with features,
                with shape (B, N, 3 + input_feature_dim).

        Returns:
            dict[str, list[torch.Tensor]]: Outputs after SA and FP modules.

                - fp_xyz (list[torch.Tensor]): The coordinates of
                    each fp features.
                - fp_features (list[torch.Tensor]): The features
                    from each Feature Propagate Layers.
                - fp_indices (list[torch.Tensor]): Indices of the
                    input points.
        """
        xyz, features = self._split_point_feats(points)

        enc_xyz = [xyz]
        enc_features = [features]

        for i in range(self.num_enc):
            cur_xyz, cur_features = self.Enc_modules[i](enc_xyz[i], enc_features[i])
            enc_xyz.append(cur_xyz)
            enc_features.append(cur_features)

        ret = dict(
            enc_xyz=enc_xyz,
            enc_features=enc_features,
        )
        return ret
