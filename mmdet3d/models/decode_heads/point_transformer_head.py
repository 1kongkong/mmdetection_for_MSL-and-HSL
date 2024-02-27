# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Sequence, Tuple

from mmcv.cnn.bricks import ConvModule
from torch import Tensor
from torch import nn as nn

from mmdet3d.registry import MODELS
from .decode_head import Base3DDecodeHead
from mmdet3d.models.layers.point_transformer import PointTransfomerDecModule


@MODELS.register_module()
class PointTransformerHead(Base3DDecodeHead):
    r"""PointTransformer decoder head.

    Args:
        dec_channels (Sequence[Sequence[int]]): Tuple of mlp channels in dec
            modules. Defaults to ((768, 256, 256), (384, 256, 256),
            (320, 256, 128), (128, 128, 128, 128)).
        dec_norm_cfg (dict or :obj:`ConfigDict`): Config of norm layers used
            in dec modules. Defaults to dict(type='BN2d').
    """

    def __init__(
        self,
        num_samples: Sequence[int] = (16, 16, 16, 16, 8),
        dec_channels: Sequence[Sequence[int]] = (
            (512, 512),
            (256, 256),
            (128, 128),
            (64, 64),
            (32, 32),
        ),
        **kwargs
    ) -> None:
        super(PointTransformerHead, self).__init__(**kwargs)
        assert len(num_samples) == len(dec_channels)
        self.num_dec = len(dec_channels)
        self.dec_modules = nn.ModuleList()

        self.Dec_modules = nn.ModuleList()
        dec_in_channel = dec_channels[0][0]
        for dec_index in range(self.num_dec):
            cur_dec_channels = list(dec_channels[dec_index])
            cur_dec_channels = [dec_in_channel] + cur_dec_channels
            self.Dec_modules.append(
                PointTransfomerDecModule(
                    channels=cur_dec_channels,
                    num_sample=num_samples[dec_index],
                    is_head=dec_index == 0,
                )
            )
            dec_in_channel = cur_dec_channels[-1]

        self.pre_seg_conv = ConvModule(
            dec_channels[-1][-1],
            self.channels,
            kernel_size=1,
            bias=True,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )

    def _extract_input(self, feat_dict: dict) -> Tuple[List[Tensor], List[Tensor]]:
        """Extract inputs from features dictionary.

        Args:
            feat_dict (dict): Feature dict from backbone.

        Returns:
            Tuple[List[Tensor], List[Tensor]]: Coordinates and features of
            multiple levels of points.
        """
        if feat_dict.get("enc_xyz", None) is not None:
            enc_xyz = feat_dict["enc_xyz"]
            enc_features = feat_dict["enc_features"]
        else:
            enc_xyz = feat_dict["points"]
            enc_features = feat_dict["features"]

        assert len(enc_xyz) == len(enc_features)
        return enc_xyz, enc_features

    def forward(self, feat_dict: dict) -> Tensor:
        """Forward pass.

        Args:
            feat_dict (dict): Feature dict from backbone.

        Returns:
            Tensor: Segmentation map of shape [B, num_classes, N].
        """
        enc_xyz, enc_features = self._extract_input(feat_dict)

        num_enc = len(enc_xyz) - 1
        dec_features = []
        for i in range(self.num_dec):
            if i == 0:
                dec_feature = self.Dec_modules[i](
                    enc_xyz[num_enc - i],
                    enc_features[num_enc - i],
                )
            else:
                dec_feature = self.Dec_modules[i](
                    enc_xyz[num_enc - i + 1],
                    dec_features[-1],
                    enc_xyz[num_enc - i],
                    enc_features[num_enc - i],
                )
            dec_features.append(dec_feature)

        output = self.pre_seg_conv(dec_feature)
        output = self.cls_seg(output)

        return output
