from mmengine.model import BaseModule
from torch import nn as nn

from mmdet3d.models.layers.pointnet_modules import PointFPModule
from mmdet3d.registry import MODELS


@MODELS.register_module()
class KPFCNNNeck(BaseModule):
    ...