import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fanjiang.builder import CRITERIONS
from fanjiang.models.flow import RAFT
from fanjiang.layers import Resample2d


@CRITERIONS.register()
class WrapLoss(nn.Module):
    def __init__(self, alpha=50, loss_weight=1.0, amp=False):
        super(WrapLoss, self).__init__()
        self.alpha = alpha 
        self.loss_weight = loss_weight 
        self.resample2d = Resample2d()
        self.model = RAFT(amp=amp).eval()

        for param in self.model.parameters():
            param.requires_grad = False


    @torch.no_grad()
    def visualize(_img1, _img2, _img1_wraped, _flo):
        import os
        import cv2
        from fanjiang.utils.flow_viz import flow_to_image
        from fanjiang.utils.visualizer import color_radar

        save_dir = "results/flow"
        os.makedirs(save_dir, exist_ok=True)

        for i in range(len(_img1)):
            img1 = _img1[i, 0].cpu().numpy()
            img2 = _img2[i, 0].cpu().numpy()
            img1_wraped = _img1_wraped[i, 0].cpu().numpy()
            flo = _flo[i].permute(1,2,0).cpu().numpy()

            img1 = color_radar(img1)
            img2 = color_radar(img2)
            img1_wraped = color_radar(img1_wraped)
            flo = flow_to_image(flo)

            img = np.concatenate([img1, img2, img1_wraped, flo], axis=1)
            save_f = os.path.join(save_dir, f"{i:02d}.png")
            cv2.imwrite(save_f, img)
                

    def forward(self, outputs, targets):
        _, ch, img_h, img_w = targets.shape 
        
        id1 = np.random.choice(np.arange(0, ch-2), 5)
        id2 = id1 + np.random.randint(1, 3)

        o1 = outputs[:, id1].view(-1, 1, img_h, img_w).detach()
        o2 = outputs[:, id2].view(-1, 1, img_h, img_w)

        t1 = targets[:, id1].view(-1, 1, img_h, img_w)
        t2 = targets[:, id2].view(-1, 1, img_h, img_w)

        _t1 = t1.repeat(1, 3, 1, 1) * 255
        _t2 = t2.repeat(1, 3, 1, 1) * 255
        _, flo = self.model(_t2, _t1, iters=12, test_mode=True)

        ### warp I1 and O1
        warp_t1 = self.resample2d(t1, flo)
        warp_o1 = self.resample2d(o1, flo)

        ### compute non-occlusion mask: exp(-alpha * || F_i2 - Warp(F_i1) ||^2 )
        mask = torch.exp(-self.alpha * torch.sum(t2 - warp_t1, dim=1, keepdim=True).pow(2))

        self.visualize(t1, t2, warp_t1 * mask, flo)

        loss = F.l1_loss(o2 * mask, warp_o1 * mask) * self.loss_weight
        return loss
