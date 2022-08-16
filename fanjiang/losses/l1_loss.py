import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from fanjiang.builder import CRITERIONS


def plot_hist(counts, smoothed_counts, save_f="results/counts.jpg"):
    _, ax = plt.subplots(1, 2, figsize=(10, 6))
    ax[0].hist(counts)
    ax[1].hist(smoothed_counts)
    plt.savefig(save_f, bbox_inches='tight', pad_inches=0.0, dpi=200)    
    

@CRITERIONS.register()
class BalancedL1Loss(torch.nn.Module):
    def __init__(
        self, 
        gamma=0.5,
        momentum=0.9,
        repeat_thr=1.0,
        loss_weight=1.0,
        bins=np.arange(0.2, 1.0, 0.05),
    ):
        super().__init__()
        self.bins = bins
        self.gamma = gamma
        self.momentum = momentum
        self.repeat_thr = repeat_thr
        self.loss_weight = loss_weight
        self.num_bin = len(self.bins)
        self.register_buffer("counts", torch.ones(self.num_bin) * 1e4)


    @torch.no_grad()
    def compute_weight(self, targets):
        weights = torch.ones_like(targets)
        masks = []
        for i in range(self.num_bin):
            if i == self.num_bin - 1:
                mask = (targets >= self.bins[i])
            else:
                mask = (targets >= self.bins[i]) & (targets < self.bins[i+1])
            masks.append(mask)
            self.counts[i] = self.momentum * self.counts[i] + (1 - self.momentum) * mask.sum()
        
        counts = self.counts

        for i in range(self.num_bin):
            freq = counts[i] / counts.sum()
            wi = (self.repeat_thr / freq) ** self.gamma
            weights[masks[i]] = wi

        return weights


    def forward(self, outputs, targets):
        weights = self.compute_weight(targets)
        loss = F.l1_loss(outputs, targets, reduction='none')
        loss = torch.sum(loss * weights) / weights.sum()
        # loss = torch.sum(loss * weights) / self.counts.sum()
        return loss * self.loss_weight


