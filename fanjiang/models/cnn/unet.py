from statistics import mode

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange
from fanjiang.builder import MODELS
from fanjiang.gan import RandomProj
from fanjiang.layers import SEBlock, Upsample, trunc_normal_
from fanjiang.losses import StyleLoss, BalancedL1Loss, WrapLoss
from fanjiang.losses.style_loss import StyleLoss
from fanjiang.transformer import SwinTransformerBlock


class SwinRNN(nn.Module):

    def __init__(
            self,
            in_channels,
            channels,
            image_size,
            window_size,
            depth=4,
            num_heads=8,            
            mlp_ratio=4.,
        ):

        super().__init__()
        self.depth = depth
        self.image_size = image_size

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = SwinTransformerBlock(
                dim=channels,
                input_resolution=image_size,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
            )
            self.blocks.append(block)

        self.linear_proj = nn.Linear(in_channels + channels, channels, bias=False)

        self.conv_proj = nn.Conv2d(channels * depth, channels, kernel_size=3, padding=1)

        self.final_layer = nn.Conv2d(
            channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    
    def forward(self, input, feat):
        img_h, img_w = input.shape[-2:]

        xt =  rearrange(input, 'n c h w -> n (h w) c')
        feat = self.linear_proj(torch.cat([xt, feat], dim=-1))      

        feats = []
        for _, blk in enumerate(self.blocks):
            feat = blk(feat)
            feats.append(feat)
        ht = feats[-1]

        feat = torch.cat(feats, dim=-1)
        feat = rearrange(feat,  'n (h w) c -> n c h w', h=img_h, w=img_w)
        feat = F.gelu(self.conv_proj(feat))

        output = self.final_layer(feat) + input
        return output, ht


class EmbedSequential(nn.Sequential):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, SEBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x

class Encoder(nn.Module):
    def __init__(self, in_channels, enc_channels, norm="BN"):
        super().__init__()
        self.blocks = nn.ModuleList()   
        for i in range(len(enc_channels)):
            layer = []
            if i == 0:
                layer.append(SEBlock(in_channels, enc_channels[i], norm=norm))
            else:  
                layer.append(SEBlock(enc_channels[i-1], enc_channels[i], norm=norm))

            self.blocks.append(EmbedSequential(*layer))

    
    def forward(self, x, emb=None):
        outs = []
        for _, blk in enumerate(self.blocks):
            x = blk(x, emb=emb)
            outs.append(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)

        return outs


class Decoder(nn.Module):
    def __init__(
            self, 
            enc_channels, 
            dec_channels,
            out_channels,            
            gfs_channels=0, 
            norm="BN"
        ):
        super().__init__()
        
        self.up_blocks = nn.ModuleList()   
        self.attn_blocks = nn.ModuleList()   
        self.pred_blocks = nn.ModuleList()   

        num_layers = len(enc_channels)
        
        for i in range(num_layers):
            if i == num_layers - 1:
                cat_channels = enc_channels[i] + gfs_channels
            else:
                cat_channels = enc_channels[i] + dec_channels[i]

            layer = [SEBlock(cat_channels, dec_channels[i], norm=norm)]

            self.attn_blocks.append(EmbedSequential(*layer))

            self.pred_blocks.append(
                nn.Conv2d(dec_channels[i], out_channels, kernel_size=1)
            )

            if i > 0:
                self.up_blocks.append(
                    Upsample(dec_channels[i], dec_channels[i-1])
                )


    def forward(self, feats, emb=None):
        outs = []
        x_up = None 
        for i in range(len(feats) - 1, -1, -1):
            x = feats[i] 

            if x_up is not None:
                x = torch.cat([x, x_up], dim=1)

            x = self.attn_blocks[i](x, emb=emb)

            out = self.pred_blocks[i](x)          
            outs.append(out)

            if i > 0:
                x_up = self.up_blocks[i-1](x)

        return outs



@MODELS.register()
class UNet(nn.Module):
    def __init__(
            self,
            levels,
            num_classes,            
            input_frames, 
            future_frames,
            enc_channels,      
            dec_channels,  
            radar_channels,
            gfs_channels=0,
            level_mixing=False,
            border=0,
            norm="BN",
        ):
        super().__init__()
        self.levels = levels
        self.level_mixing = level_mixing
        self.num_classes = num_classes
        self.input_frames = input_frames
        self.future_frames = future_frames
        self.with_gfs = gfs_channels > 0
        self.border = border

        in_channels = input_frames * radar_channels
        out_channels = future_frames * num_classes
        self.encoder = Encoder(in_channels, enc_channels, norm=norm)
        self.decoder = Decoder(enc_channels, dec_channels, gfs_channels, out_channels, norm=norm)
        self.final_layer = nn.Conv2d(out_channels * len(enc_channels), out_channels, 1)

        self.loss_l1 = WeightedL1Loss(loss_weight=10.0, gamma=0.7, bins=np.arange(0.3, 1.0, 0.05))
        self.proj_t = Proj(in_dim=future_frames)
        self.proj_c = Proj(in_dim=num_classes)
        self.loss_style = StyleLoss()
        self.loss_wrap = WrapLoss()

        self.lateral = nn.Sequential(
            nn.LayerNorm(enc_channels[-1]), 
            nn.Linear(enc_channels[-1], 256, bias=False)
        )

        self.swinrnn = SwinRNN(
            in_channels=num_classes,
            channels=256, 
            image_size=(16, 16), 
            window_size=8,
        )


    def process_radar(self, data):
        if self.level_mixing:
            levels = np.arange(self.levels[1], self.levels[-1], 0.5)
        else:
            levels = self.levels[1:-1]

        lvl = np.random.choice(levels)
        idx = levels.index(lvl)

        radar = data["radar"] / 255
        imgs = [radar[:, :, [0, -4, -3, -2, -1]]] 

        if lvl in self.levels:
            idx = self.levels.index(lvl)
            img = radar[:, :, [idx]]
        else:
            prev_idx = idx // 3
            next_idx = prev_idx + 1
            curr_idx = idx % 3

            assert curr_idx in [1, 2], curr_idx
            assert lvl > self.levels[prev_idx], (lvl, self.levels[prev_idx])
            assert lvl < self.levels[next_idx], (lvl, self.levels[next_idx])

            img = radar[:, :, [prev_idx, next_idx]] 
            img = F.interpolate(
                img, size=(4, *img.shape[-2:]), 
                mode="trilinear", align_corners=False
            )
            img = img[:, :, [curr_idx]]

        imgs.insert(1, img)
        imgs = torch.cat(imgs, dim=2) 

        inputs = imgs[:, :self.input_frames].flatten(1, 2)
        targets = imgs[:, self.input_frames:, :-3].flatten(1, 2)

        lvl = torch.tensor(lvl / self.levels[-1], device=inputs.device)
        emb = self.level_embed.weight[idx]
        emb = self.encoding(lvl)
        return inputs, targets, emb
        
    
    def process_gfs(self, data):
        gfs = data["gfs"] / 20
        gfs = rearrange(gfs, 'n t c h w -> n (t c) h w')
        return gfs


    def process_for_eval(self, outputs):
        radar, prec, wind = outputs.unbind(dim=2)

        radar = [radar[:, t-1] for t in self.eval_tid]
        prec = [prec[:, t-10:t].sum(1) for t in self.eval_tid]
        wind = [wind[:, t-10:t].max(1)[0] for t in self.eval_tid]

        radar = torch.stack(radar, dim=1)
        prec = torch.stack(prec, dim=1)
        wind = torch.stack(wind, dim=1)
        
        outputs = torch.stack([radar, prec, wind], dim=2)
        return outputs[:, :, self.eval_cid]


    def valid_crop(self, inputs, size=256):
        H, W = inputs.shape[-2:]
        x1 = y1 = self.border
        x2 = W - self.border
        y2 = H - self.border
        if self.training and y2 - y1 > size:
            x1 = np.random.randint(x1, x2 - size + 1)
            y1 = np.random.randint(y1, y2 - size + 1)
            x2 = x1 + size
            y2 = y1 + size 
        return x1, y1, x2, y2


    def losses(self, outputs, targets, cid=1, tid=1, stride=1, loss_weight=1):
        if targets.shape[-2:] != outputs.shape[-2:]:
            targets = F.interpolate(
                targets, size=outputs.shape[-2:], mode='nearest'
            )

        loss = {}
        loss[f"loss_l1_x{stride}"] = self.loss_l1(outputs, targets) * loss_weight
        outputs = rearrange(outputs, 'n (t c) h w -> n t c h w', t=self.future_frames)
        targets = rearrange(targets, 'n (t c) h w -> n t c h w', t=self.future_frames)

        out_feats = self.proj_t(outputs[:, :, cid])
        tgt_feats = self.proj_t(targets[:, :, cid])
        masks = self.proj_t.forward_mask(targets[:, :, cid])        
        loss[f"loss_sty_x{stride}"] = self.loss_style(out_feats, tgt_feats, masks) * loss_weight

        out_feats = self.proj_c(outputs[:, tid])
        tgt_feats = self.proj_c(targets[:, tid])
        masks = self.proj_c.forward_mask(targets[:, tid])        
        loss[f"loss_sty_x{stride}"] += self.loss_style(out_feats, tgt_feats, masks) * loss_weight
        return loss
    
    def forward_rnn(self, feat, inputs):
        feat = rearrange(feat, 'n c h w -> n (h w) c')
        inputs = rearrange(inputs, 'n (t c) h w -> n t c h w', c=self.num_classes)        
        feat = self.lateral(feat)
        outputs = []
        for t in range(self.future_frames):
            output, feat = self.swinrnn(inputs[:, t], feat)
            outputs.append(output)
        outputs = torch.cat(outputs, dim=1)
        return outputs

    
    def forward(self, data, info={}):
        inputs, targets = self.process_radar(data)

        img_h, img_w = inputs.shape[-2:]

        feats = self.encoder(inputs)

        if self.with_gfs:
            gfs = self.process_gfs(data)
            gfs = F.interpolate(
                gfs, size=feats[-1].shape[-2:], mode="bilinear", align_corners=True
            )
            feats[-1] =  torch.cat([feats[-1], gfs], dim=1)

        outs = self.decoder(feats)

        outputs_rnn = self.forward_rnn(feats[-1], outs[0].detach())

        if self.training:
            outs_x2 = outs[-2].sigmoid()

        for i, out in enumerate(outs[:-1]):
            outs[i] = F.interpolate(
                out, size=(img_h, img_w), mode='bilinear', align_corners=True
            )   

        outputs_rnn = F.interpolate(
            outputs_rnn, 
            size=(img_h, img_w),
            mode="bilinear", align_corners=True)
        
        outs.append(outputs_rnn)

        # outputs = self.final_layer(torch.cat(outs, dim=1)).sigmoid() + outputs_rnn
        
        x1, y1, x2, y2 = self.valid_crop(inputs, size=256)
        cut_outputs = outputs[:, :, y1:y2, x1:x2]
        cut_targets = targets[:, :, y1:y2, x1:x2]
        cut_outs_x2 = outs_x2[:, :, y1//2:y2//2, x1//2:x2//2]

        if self.training:
            tid = np.random.randint(self.future_frames)
            cid = np.random.randint(self.num_classes)  
            info["loss"] = {}
            info["loss"].update(self.losses(cut_outputs, cut_targets, cid, stride=1, loss_weight=1.0))
            info["loss"].update(self.losses(cut_outs_x2, cut_targets, cid, stride=1, loss_weight=0.5))
            return info


        info["mean"] = info["mean"][self.eval_cid]
        info["std"] = info["std"][self.eval_cid]
        info["output"] = self.process_for_eval(cut_outputs)
        info["target"] = self.process_for_eval(cut_targets)           
        return info



    def inference(self, data):
        feats = self.encoder(data[0])

        if self.with_gfs:
            gfs = data[1]
            gfs = F.interpolate(
                gfs, size=feats[-1].shape[-2:], mode="bilinear", align_corners=True
            )
            feats[-1] =  torch.cat([feats[-1], gfs], dim=1)

        feat = torch.cat(self.decoder(feats), dim=1)
        outputs = self.output(feat).sigmoid()

        if self.border > 0:
            return outputs[:, :, self.border:-self.border, self.border:-self.border]

        return outputs
