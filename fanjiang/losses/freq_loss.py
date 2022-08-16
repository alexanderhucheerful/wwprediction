import torch
import torch.nn.functional as F
from fanjiang.builder import CRITERIONS


@CRITERIONS.register()
class FreqLoss(torch.nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, outputs, targets):  
        outputs = torch.fft.fftn(outputs, dim=(-2, -1)).real
        targets = torch.fft.fftn(targets, dim=(-2, -1)).real 
        loss = F.l1_loss(outputs, targets) * self.loss_weight
        return loss
