from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor
from torch import nn as nn

from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from ...structures.det3d_data_sample import OptSampleList, SampleList
from ..utils import add_prefix
from .base import Base3DSegmentor
from scipy.spatial import KDTree
from .encoder_decoder import EncoderDecoder3D


@MODELS.register_module()
class PreP_EncoderDecoder3D(EncoderDecoder3D):
    def __init__(
        self,
        prep: ConfigType,
        backbone: ConfigType,
        decode_head: ConfigType,
        neck: OptConfigType = None,
        auxiliary_head: OptMultiConfig = None,
        loss_regularization: OptMultiConfig = None,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        data_preprocessor: OptConfigType = None,
        init_cfg: OptMultiConfig = None,
    ) -> None:
        super(PreP_EncoderDecoder3D, self).__init__(
            backbone,
            decode_head,
            neck,
            auxiliary_head,
            loss_regularization,
            train_cfg,
            test_cfg,
            data_preprocessor,
            init_cfg,
        )
        self.prep = MODELS.build(prep)

    def extract_feat(self, batch_inputs: Tensor) -> dict:
        """Extract features from points."""
        batch_inputs = self.perp(batch_inputs)
        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)
        return x
