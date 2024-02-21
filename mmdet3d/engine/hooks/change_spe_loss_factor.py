from mmengine.hooks import Hook
from typing import Dict, List, Tuple
from mmdet3d.registry import HOOKS


@HOOKS.register_module()
class Change_spe_loss_factor(Hook):
    def __init__(
        self,
        update_epoch: List[int] = [],
        factor: List[float] = [],
    ):
        assert len(update_epoch) == len(factor)
        self.epoch_factor_map = dict()
        for i, epoch in enumerate(update_epoch):
            self.epoch_factor_map[epoch - 1] = factor[i]

    def before_train_epoch(self, runner):
        if runner.epoch in self.epoch_factor_map.keys():
            runner.logger.info("spe loss factor update!")
            runner.model.train_cfg.spe_loss_factor = self.epoch_factor_map[runner.epoch]


@HOOKS.register_module()
class freeze(Hook):
    def __init__(
        self,
        freeze_epoch,
        freeze_key,
        freeze_bn: bool = False,
    ):
        self.freeze_key = freeze_key
        self.freeze_epoch = freeze_epoch
        pass

    def before_train_epoch(self, runner):
        if runner.epoch == self.freeze_epoch:
            runner.logger.info(f"frozen")
            for name, param in runner.model.named_parameters():
                if self.freeze_key in name:
                    param.requires_grad = False
                    runner.logger.info(f"{name} weights frozen")
            runner.model.prep.eval()
            runner.logger.info("prep eval")
