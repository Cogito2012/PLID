import torch
import math
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


class CustomCosineAnnealingWarmupRestarts(CosineAnnealingWarmRestarts):

    def __init__(self, warmup_epochs, lr_decay=1.0, lr_decay_epochs=None, *args, **kwargs):

        super(CustomCosineAnnealingWarmupRestarts, self).__init__(*args, **kwargs)

        # Init optimizer with low learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.eta_min

        self.warmup_epochs = warmup_epochs
        self.lr_decay = lr_decay
        self.lr_decay_epochs = lr_decay_epochs

        self.warmup_lrs = []
        for base_lr in self.base_lrs:
            # Get target LR after warmup is complete
            target_lr = self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * warmup_epochs / self.T_i)) / 2

            # Linearly interpolate between minimum lr and target_lr
            linear_step = (target_lr - self.eta_min) / self.warmup_epochs
            warmup_lrs = [self.eta_min + linear_step * (n + 1) for n in range(warmup_epochs)]
            self.warmup_lrs.append(warmup_lrs)

    def step(self, epoch=None):

        # Called on super class init
        if epoch is None:
            super(CustomCosineAnnealingWarmupRestarts, self).step(epoch=epoch)

        else:
            if epoch < self.warmup_epochs:
                for i, param_group in enumerate(self.optimizer.param_groups):
                    param_group['lr'] = self.warmup_lrs[i][epoch]

                # Fulfill misc super() funcs
                self.last_epoch = math.floor(epoch)
                self.T_cur = epoch
                self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

            else:
                # decay the base_lr which will be used in the step() call for cosine restart
                if self.lr_decay_epochs and epoch + 1 in self.lr_decay_epochs:
                    self.base_lrs = [lr * self.lr_decay for lr in self.base_lrs]
                super(CustomCosineAnnealingWarmupRestarts, self).step(epoch=epoch)