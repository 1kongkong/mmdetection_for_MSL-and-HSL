from typing import List, Sequence, Tuple

from mmcv.cnn.bricks import ConvModule
from torch import Tensor
from torch import nn as nn
import numpy as np
import torch

from mmdet3d.models.layers import KPConvFPModule
from mmdet3d.registry import MODELS
from mmdet3d.utils.typing_utils import ConfigType
from .decode_head import Base3DDecodeHead
from mmdet3d.structures.det3d_data_sample import SampleList


@MODELS.register_module()
class KPFCNNHead(Base3DDecodeHead):
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
        fp_channels: Sequence[Sequence[int]] = (
            (1536, 512, 512),
            (768, 256, 256),
            (384, 128, 128),
            (192, 128, 128),
        ),
        stack: bool = True,
        **kwargs
    ) -> None:
        super(KPFCNNHead, self).__init__(**kwargs)

        self.num_fp = len(fp_channels)
        self.FP_modules = nn.ModuleList()
        self.stack = stack
        for cur_fp_mlps in fp_channels:
            self.FP_modules.append(
                KPConvFPModule(
                    mlp_channels=cur_fp_mlps,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                )
            )

        # https://github.com/charlesq34/pointnet2/blob/master/models/pointnet2_sem_seg.py#L40
        self.pre_seg_conv = ConvModule(
            fp_channels[-1][-1],
            self.channels,
            kernel_size=1,
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )

    def _extract_input(self, input_dict: dict) -> Tensor:
        """Extract inputs from features dictionary.

        Args:
            feat_dict (dict): Feature dict from backbone.

        Returns:
            torch.Tensor: Points for decoder.
        """
        xyz, features = input_dict["points"], input_dict["features"]

        return xyz, features

    def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tensor:
        gt_semantic_segs = [
            data_sample.gt_pts_seg.pts_semantic_mask
            for data_sample in batch_data_samples
        ]
        if self.stack:
            gt = torch.stack(gt_semantic_segs, dim=0)
        else:
            gt = torch.cat(gt_semantic_segs).unsqueeze(0)
        return gt

    def forward(self, input_dict) -> Tensor:
        """Forward pass.

        Args:
            feat_dict (dict): Feature dict from backbone.

        Returns:
            Tensor: Segmentation map of shape [B, num_classes, N].
        """
        sa_xyz, sa_features = self._extract_input(input_dict)
        length = input_dict["length"]
        fp_feature = sa_features[-1]

        for i in range(self.num_fp):
            fp_feature = self.FP_modules[i](
                sa_xyz[-i - 2],
                sa_xyz[-i - 1],
                sa_features[-i - 2],
                fp_feature,
                length[-i - 2],
                length[-i - 1],
            )
        output = self.pre_seg_conv(fp_feature)
        output = self.cls_seg(output)

        return output
