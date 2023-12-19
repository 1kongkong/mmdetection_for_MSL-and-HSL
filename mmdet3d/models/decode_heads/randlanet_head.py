from typing import List, Sequence, Tuple

from torch import Tensor
from torch import nn as nn
import torch

from mmdet3d.registry import MODELS
from .decode_head import Base3DDecodeHead
from mmdet3d.models.layers.randla_modules import SharedMLP
from mmcv.ops.knn import knn
from mmcv.ops.group_points import grouping_operation


@MODELS.register_module()
class RandLANetHead(Base3DDecodeHead):
    """KPFCNN decoder head.

    Decoder head used in `PointNet++ <https://arxiv.org/abs/1706.02413>`_.
    Refer to the `official code <https://github.com/charlesq34/pointnet2>`_.

    Args:
        fp_channels (Sequence[Sequence[int]]): Tuple of mlp channels in FP
            modules. Defaults to ((768, 256, 256), (384, 256, 256),
            (320, 256, 128), (128, 128, 128, 128)).
        fp_norm_cfg (dict or :obj:`ConfigDict`): Config of norm layers used
            in FP modules. Defaults to dict(type='BN2d').
    """

    def __init__(
        self,
        dec_channels: Sequence[Sequence[int]] = (
            (1024, 256),
            (512, 128),
            (256, 32),
            (64, 8),
        ),
        **kwargs
    ) -> None:
        super(RandLANetHead, self).__init__(**kwargs)

        self.FP_modules = nn.ModuleList()
        # decoding layers
        decoder_kwargs = dict(transpose=True, bn=True, activation_fn=nn.ReLU())
        self.Dec_modules = nn.ModuleList()

        for dec_channel in dec_channels:
            self.Dec_modules.append(
                SharedMLP(dec_channel[0], dec_channel[1], **decoder_kwargs)
            )

        # https://github.com/charlesq34/pointnet2/blob/master/models/pointnet2_sem_seg.py#L40
        self.pre_seg_conv = nn.Sequential(
            SharedMLP(dec_channels[-1][-1], 64, bn=True, activation_fn=nn.ReLU()),
            SharedMLP(64, 32, bn=True, activation_fn=nn.ReLU()),
        )

    def _extract_input(self, input):
        return input["points"], input["features"]

    def forward(self, input_dict) -> Tensor:
        """Forward pass.

        Args:
            feat_dict (dict): Feature dict from backbone.

        Returns:
            Tensor: Segmentation map of shape [B, num_classes, N].
        """
        coords, features = self._extract_input(input_dict)
        fp_feature = features.pop()
        fp_coord = coords.pop()
        # <<<<<<<<<< DECODER
        for mlp in self.Dec_modules:
            idx = knn(1, fp_coord, coords[-1], False).transpose(1, 2).contiguous()
            fp_coord = coords.pop()

            upsample_feats = grouping_operation(fp_feature.squeeze(-1), idx)
            # upsample_feats = upsample_feats.squeeze(-1)

            fp_feature = torch.cat((upsample_feats, features.pop()), dim=1)
            fp_feature = mlp(fp_feature)

        output = self.pre_seg_conv(fp_feature)
        output = self.cls_seg(output.squeeze(-1))

        return output
