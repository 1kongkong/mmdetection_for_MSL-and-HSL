# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Sequence

from torch import Tensor, nn
from my_tools.knn import knn

from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptMultiConfig
from mmdet3d.models.layers.point_transformer import PointTransfomerEncModule
from .point_transformer import PointTransformerBackbone


@MODELS.register_module()
class Dual_PointTransformerBackbone(PointTransformerBackbone):
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
        in_channels_spa: int,
        in_channels_spe: int,
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
        super(Dual_PointTransformerBackbone, self).__init__(
            in_channels=in_channels_spa,
            num_points=num_points,
            num_samples=num_samples,
            enc_channels=enc_channels,
            init_cfg=init_cfg,
        )

        self.num_samples = num_samples
        self.Enc_modules_spa = self.Enc_modules
        self.Enc_modules_spe = nn.ModuleList()

        self.num_enc = len(enc_channels)
        enc_in_channel = in_channels_spe  # number of channels without xyz

        for enc_index in range(self.num_enc):
            cur_enc_channels = list(enc_channels[enc_index])
            cur_enc_channels = [enc_in_channel] + cur_enc_channels
            enc_in_channel = cur_enc_channels[-1]

            self.Enc_modules_spe.append(
                PointTransfomerEncModule(
                    num_point=num_points[enc_index],
                    num_sample=num_samples[enc_index],
                    channels=cur_enc_channels,
                    is_head=enc_index == 0,
                )
            )

    def _knn_query(self, points):
        """
        Args:
            points (Tensor): (B,N,3)
            target_points (Tensor): (B,M,3)
        Returns:
            idx (Tensor): (B,M,k)
        """
        self_idx = []

        for i, xyz in enumerate(points):
            idx = knn(self.num_samples[i], xyz, xyz)
            self_idx.append(idx)

        return self_idx

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
        enc_xyz2 = [xyz]
        enc_features_spa = [features[:, 3:, :]]
        enc_features_spe = [features[:, :3, :]]

        for i in range(self.num_enc):
            cur_xyz1, cur_features_spa = self.Enc_modules_spa[i](
                enc_xyz[i], enc_features_spa[i]
            )
            cur_xyz2, cur_features_spe = self.Enc_modules_spe[i](
                enc_xyz[i], enc_features_spe[i]
            )

            enc_xyz.append(cur_xyz1)
            enc_xyz2.append(cur_xyz2)
            enc_features_spa.append(cur_features_spa)
            enc_features_spe.append(cur_features_spe)
        import pdb

        pdb.set_trace()
        self_idx = self._knn_query(enc_xyz[1:])

        ret = dict(
            points=enc_xyz[1:],
            spa_feature=enc_features_spa[1:],
            spe_feature=enc_features_spe[1:],
            self_index=self_idx,
        )
        return ret
