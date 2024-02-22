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
        batch_inputs = self.prep(batch_inputs)
        if self.train_cfg.get("stack", True):
            batch_inputs = torch.stack(batch_inputs)
        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

    def loss(
        self, batch_inputs_dict: dict, batch_data_samples: SampleList
    ) -> Dict[str, Tensor]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs_dict (dict): Input sample dict which
                includes 'points' and 'imgs' keys.

                - points (List[Tensor]): Point cloud of each sample.
                - imgs (Tensor, optional): Image tensor has shape (B, C, H, W).
            batch_data_samples (List[:obj:`Det3DDataSample`]): The det3d data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.

        Returns:
            Dict[str, Tensor]: A dictionary of loss components.
        """

        # extract features using backbone
        # if self.train_cfg.get("stack", True):
        #     batch_inputs_dict["points"] = torch.stack(batch_inputs_dict["points"])

        # from my_tools.vis_points import vis
        # import pdb

        # tem = points.clone().detach().cpu().numpy()
        # for i in range(tem.shape[0]):
        #     vis(tem[i, :, :3])
        # pdb.set_trace()

        x = self.extract_feat(batch_inputs_dict["points"])

        losses = dict()

        loss_decode = self._decode_head_forward_train(x, batch_data_samples)
        losses.update(loss_decode)

        loss_spe = self.prep.loss()
        if loss_spe is not None:
            spe_loss_factor = self.train_cfg.get("spe_loss_factor", None)
            if spe_loss_factor is not None:
                for key in loss_spe.keys():
                    loss_spe[key] = loss_spe[key] * spe_loss_factor
            losses.update(loss_spe)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, batch_data_samples)
            losses.update(loss_aux)

        if self.with_regularization_loss:
            loss_regularize = self._loss_regularization_forward_train()
            losses.update(loss_regularize)

        return losses

    # def gridsample_slide_inference(self, point: Tensor, input_meta: dict) -> Tensor:
    #     """Inference by gridsample and sliding-window with overlap.
    #     Args:
    #         point (Tensor): Input points of shape [N, 3+C].
    #         input_meta (dict): Meta information of input sample.

    #     Returns:
    #         Tensor: The output segmentation map of shape [num_classes, N].
    #     """
    #     if "titan" in input_meta["lidar_path"].lower():
    #         grid_size = 0.4
    #     elif "sensaturban" in input_meta["lidar_path"].lower():
    #         grid_size = 0.2
    #     else:
    #         raise NotImplementedError

    #     if self.test_cfg.get("voxel_size", None):
    #         grid_size = self.test_cfg.voxel_size
    #     print(f"grid_size:{grid_size}")
    #     choice = self.voxel_sample(point[:, :3], grid_size)
    #     # choice, _ = torch.sort(choice)
    #     point_sub = point[choice, :].contiguous()

    #     # 创建映射索引
    #     tree = KDTree(point_sub[:, :3].cpu().numpy())
    #     _, idx = tree.query(point[:, :3].cpu().numpy(), k=3, workers=4)
    #     idx = torch.tensor(idx, device=point_sub.device, dtype=torch.long)

    #     block_size = self.test_cfg.block_size
    #     sample_rate = self.test_cfg.sample_rate
    #     use_normalized_coord = self.test_cfg.use_normalized_coord
    #     batch_size = self.test_cfg.batch_size

    #     # patch_points is of shape [K*N, 3+C], patch_idxs is of shape [K*N]
    #     patch_points, patch_idxs = self._sliding_whole_patch_generation(
    #         point_sub, block_size, sample_rate, use_normalized_coord
    #     )
    #     for i in range(len(patch_points)):
    #         # 临时更改
    #         channel_idx = torch.argsort(patch_points[i][:, 6])
    #         # channel_idx = torch.argsort(patch_points[i][:, 9])
    #         patch_points[i] = patch_points[i][channel_idx]
    #         patch_idxs[i] = patch_idxs[i][channel_idx]

    #     seg_logits = []  # save patch predictions

    #     for batch_idx in range(0, len(patch_points), batch_size):
    #         batch_points = patch_points[batch_idx : batch_idx + batch_size]
    #         batch_seg_logit = self.encode_decode(
    #             batch_points, [input_meta] * batch_size
    #         )
    #         batch_seg_logit = batch_seg_logit.transpose(1, 2).contiguous()
    #         seg_logits.append(batch_seg_logit.view(-1, self.num_classes))

    #     # aggregate per-point logits by indexing sum and dividing count
    #     seg_logits = torch.cat(seg_logits, dim=0)  # [K*N, num_classes]
    #     patch_idxs = torch.cat(patch_idxs)
    #     expand_patch_idxs = patch_idxs.unsqueeze(1).repeat(1, self.num_classes)
    #     preds_sub = point_sub.new_zeros(
    #         (point_sub.shape[0], self.num_classes)
    #     ).scatter_add_(dim=0, index=expand_patch_idxs, src=seg_logits)
    #     count_mat = torch.bincount(patch_idxs)
    #     preds_sub = preds_sub / count_mat[:, None]

    #     # 清理显存
    #     del point, point_sub, patch_points, patch_idxs, expand_patch_idxs
    #     torch.cuda.empty_cache()

    #     preds_sub = preds_sub.half()
    #     preds = preds_sub[idx, :]
    #     if len(preds.shape) == 3:
    #         preds = torch.mean(preds, dim=-2)

    #     return preds.transpose(0, 1).contiguous()  # to [num_classes, K*N]

    def slide_inference(self, point: Tensor, input_meta: dict, rescale: bool) -> Tensor:
        """Inference by sliding-window with overlap.

        Args:
            point (Tensor): Input points of shape [N, 3+C].
            input_meta (dict): Meta information of input sample.
            rescale (bool): Whether transform to original number of points.
                Will be used for voxelization based segmentors.

        Returns:
            Tensor: The output segmentation map of shape [num_classes, N].
        """
        num_points = self.test_cfg.num_points
        block_size = self.test_cfg.block_size
        sample_rate = self.test_cfg.sample_rate
        use_normalized_coord = self.test_cfg.use_normalized_coord
        batch_size = self.test_cfg.batch_size * num_points

        # patch_points is of shape [K*N, 3+C], patch_idxs is of shape [K*N]
        patch_points, patch_idxs = self._sliding_patch_generation(
            point, num_points, block_size, sample_rate, use_normalized_coord
        )
        feats_dim = patch_points.shape[1]
        seg_logits = []  # save patch predictions

        for batch_idx in range(0, patch_points.shape[0], batch_size):
            batch_points = patch_points[batch_idx : batch_idx + batch_size]
            batch_points = batch_points.view(-1, num_points, feats_dim)
            # batch_seg_logit is of shape [B, num_classes, N]
            batch_points = [batch_points[i, ...] for i in range(batch_points.shape[0])]
            batch_seg_logit = self.encode_decode(
                batch_points, [input_meta] * batch_size
            )
            batch_seg_logit = batch_seg_logit.transpose(1, 2).contiguous()
            seg_logits.append(batch_seg_logit.view(-1, self.num_classes))

        # aggregate per-point logits by indexing sum and dividing count
        seg_logits = torch.cat(seg_logits, dim=0)  # [K*N, num_classes]
        expand_patch_idxs = patch_idxs.unsqueeze(1).repeat(1, self.num_classes)
        preds = point.new_zeros((point.shape[0], self.num_classes)).scatter_add_(
            dim=0, index=expand_patch_idxs, src=seg_logits
        )
        count_mat = torch.bincount(patch_idxs)
        preds = preds / count_mat[:, None]

        # TODO: if rescale and voxelization segmentor

        return preds.transpose(0, 1)  # to [num_classes, K*N]

    def _forward(
        self, batch_inputs_dict: dict, batch_data_samples: OptSampleList = None
    ) -> Tensor:
        """Network forward process.

        Args:
            batch_inputs_dict (dict): Input sample dict which includes 'points'
                and 'imgs' keys.

                - points (List[Tensor]): Point cloud of each sample.
                - imgs (Tensor, optional): Image tensor has shape (B, C, H, W).
            batch_data_samples (List[:obj:`Det3DDataSample`]): The det3d data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.

        Returns:
            Tensor: Forward output of model without any post-processes.
        """
        # if self.train_cfg.get("stack", True):
        #     batch_inputs_dict["points"] = torch.stack(batch_inputs_dict["points"])

        x = self.extract_feat(batch_inputs_dict["points"])
        return self.decode_head.forward(x)
