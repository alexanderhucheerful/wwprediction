import torch.nn as nn
import torch.nn.functional as F
from fanjiang.builder import CRITERIONS

@CRITERIONS.register()
class StyleLoss(nn.Module):
    def __init__(self, loss_weights=[1e2, 1.0, 1.0, 1.0]):
        super(StyleLoss, self).__init__()
        self.loss_weights = loss_weights 

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (b * h * w * ch)
        return G

    def forward(self, out_feats, tgt_feats, masks=None):
        loss = 0.0
        for i in range(len(tgt_feats)):
            out_feat = out_feats[i]
            tgt_feat = tgt_feats[i]
            if masks is not None:
                out_feat *= masks[i]
                tgt_feat *= masks[i]
            out_gram = self.compute_gram(out_feat) 
            tgt_gram = self.compute_gram(tgt_feat)
            loss += F.l1_loss(out_gram, tgt_gram) * self.loss_weights[i]
        return loss
