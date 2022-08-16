import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Normal, kl_divergence

from einops import repeat, rearrange
from fanjiang.builder import MODELS
from fanjiang.layers import (Conv2d, TimeEncoding, c2_msra_fill, trunc_normal_)

from .transformer import SwinEncoder, SwinBlock

from tqdm import tqdm

__all__ = ["SwinRNN", "SwinDecoder", "MultiSwinDecoder", "SwinLayer"]


def plot_diffusion(samples, save_dir="log_images"):
    import os
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    eval_names = ["z500", "t850", "t2m", "u10", "v10", "tp"]
    os.makedirs(save_dir, exist_ok=True)

    samples = [sample[0].cpu().numpy() for sample in samples]

    for i, name in enumerate(eval_names):
        ims = []
        fig = plt.figure()
        plt.axis("off")

        for sample in samples:
            im = plt.imshow(sample[i], animated=True)
            ims.append([im])

        animate = animation.ArtistAnimation(fig, ims, interval=2000, blit=True, repeat_delay=1000)
        save_f = os.path.join(save_dir, f"{name}_ddpm.gif")
        animate.save(save_f, dpi=300)


class SwinLayer(nn.Module):

    def __init__(
            self,
            in_channels,
            channels,
            input_size,
            depth,
            num_heads,
            window_size,
            mlp_ratio=4.,
            drop_path=0,
        ):

        super().__init__()
        self.input_size = input_size
        self.depth = depth
        self.blocks = nn.ModuleList()

        for i in range(depth):
            blk = SwinBlock(
                dim=channels,
                input_size=input_size,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path,
            )
            self.blocks.append(blk)

        self.linear_proj = nn.Linear(in_channels + channels, channels, bias=False)

        self.conv_proj = Conv2d(
            channels * depth, channels, kernel_size=3, padding=1,
            #activation=F.gelu,
        )

        self.apply(self._init_weights)

        self.in_channels = in_channels
        self.channels = channels


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, x, state, emb=None, zt=None):
        state = torch.cat([x, state], dim=1)
        state =  self.linear_proj(rearrange(state, 'n c h w -> n (h w) c'))

        if emb is not None:
            state = state + emb.unsqueeze(0).unsqueeze(0)

        if zt is not None:
            zt = rearrange(zt, 'n c h w -> n (h w) c')
            state = state + zt

        states  = []
        for i, blk in enumerate(self.blocks):
            state = blk(state)
            states.append(state)
        #return rearrange(states[-1],  'n (h w) c -> n c h w', h=x.size(2))

        state = torch.cat(states, dim=-1)
        state = rearrange(state,  'n (h w) c -> n c h w', h=x.size(2))
        state = self.conv_proj(state)
        return state

    def flops(self):
        flops = 0
        for i, blk in enumerate(self.blocks):
            flops += blk.flops()
        #self.linear_proj
        flops += self.input_size[0] * self.input_size[1] * (self.in_channels + self.channels) * self.channels
        #self.conv_proj
        flops += self.input_size[0] * self.input_size[1] * 9 * self.channels * self.depth * self.channels
        return flops


class SwinDecoder(nn.Module):
    def __init__(
            self,
            in_channels,
            channels,
            image_size,
            window_size=8,
            patch_size=4,
            depth=4,
            num_heads=8,
            drop_path=0,
        ):
        super().__init__()

        self.down_layer = nn.Conv2d(
            in_channels, channels, kernel_size=patch_size, stride=patch_size
        )
        c2_msra_fill(self.down_layer)

        image_size = [sz // patch_size for sz in image_size]

        self.attn_layer = SwinLayer(
            channels, channels,
            image_size, depth, num_heads, window_size,
            drop_path=drop_path,
        )
        # self.pred_layer = nn.Conv2d(channels, in_channels, 3, padding=1)

        self.up_layer = nn.ConvTranspose2d(
            # in_channels, in_channels, kernel_size=patch_size, stride=patch_size
            channels, in_channels, kernel_size=patch_size, stride=patch_size
        )
        c2_msra_fill(self.up_layer)

        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.channels = channels

    def forward(self, input, state, emb=None, zt=None):
        xt = self.down_layer(input)
        state = self.attn_layer(xt, state, emb=emb, zt=zt)
        output = self.up_layer(state) + input
        return output, state

    def flops(self):
        flops = 0
        #self.down_layer
        flops += self.image_size[0] * self.image_size[1] * self.patch_size * self.patch_size * self.in_channels * self.channels
        flops += self.attn_layer.flops()
        #self.up_layer
        flops += (self.image_size[0] * self.image_size[1] * self.patch_size ** 2) * self.patch_size * self.patch_size * self.channels * self.in_channels
        return flops

class MultiSwinDecoder(nn.Module):
    def __init__(
            self,
            in_channels,
            channels,
            image_size,
            window_size=8,
            patch_size=4,
            depth=4,
            num_heads=8,
            num_layers=4,
        ):
        super().__init__()
        self.attn_layers = nn.ModuleList()
        self.pred_layers = nn.ModuleList()

        self.down_layer = nn.Conv2d(
            in_channels, channels, kernel_size=patch_size, stride=patch_size
        )
        c2_msra_fill(self.down_layer)

        image_size = [sz // patch_size for sz in image_size]

        for i in range(num_layers):
            input_resolution = [sz // 2 **i for sz in image_size]

            self.attn_layers.append(
                SwinLayer(
                    channels,
                    channels,
                    input_resolution, depth, num_heads, window_size
                )
            )

            self.pred_layers.append(
                Conv2d(
                    channels,
                    in_channels, kernel_size=3, stride=1, padding=1,
                )
            )

        self.up_layer = nn.ConvTranspose2d(
            in_channels*num_layers, in_channels, kernel_size=patch_size, stride=patch_size
        )

        c2_msra_fill(self.up_layer)

        self.channels = channels
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_layers = num_layers

    def forward(self, input, states, emb=None, zt=None):
        xt = self.down_layer(input)
        img_h, img_w = xt.shape[-2:]

        outputs = []
        for i in range(len(states)):
            h = img_h // 2 ** i
            w = img_w // 2 ** i

            xt_i = F.interpolate(
                xt,
                size=(h, w),
                mode='bilinear',
                align_corners=False
            )
            states[i] = self.attn_layers[i](xt_i, states[i], emb=emb)

            feat = F.interpolate(
                states[i],
                size=(img_h, img_w),
                mode='bilinear',
                align_corners=False
            )
            output = self.pred_layers[i](feat)
            outputs.append(output)

        output = torch.cat(outputs, dim=1)
        output = self.up_layer(output) + input
        return output, states

    def flops(self):
        flops = 0
        #self.down_layer
        flops += self.image_size[0] * self.image_size[1] * self.patch_size * self.patch_size * self.in_channels * self.channels
        for attn in self.attn_layers:
            flops += attn.flops()

        for i in range(len(self.attn_layers)):
            h = self.image_size[0] // 2 ** i
            w = self.image_size[1] // 2 ** i
            #interpolate
            flops += h * w * 11
            #interpolate 2
            flops += self.image_size[0] * self.image_size[1] * 11
            #self.pred_layers
            flops += 9 * self.channels * self.in_channels * h * w

        #self.up_layer
        flops += (self.image_size[0] * self.image_size[1] * self.patch_size ** 2) * self.patch_size * self.patch_size * self.in_channels * self.num_layers * self.in_channels
        return flops



@MODELS.register()
class SwinRNN(nn.Module):

    def __init__(
        self,
        in_channels,
        channels,
        input_frames,
        future_frames,
        embed_drop=0,
        eval_index=[0],
        eval_fid=False,
        members=1,
        freeze_at=[],
        overrides=[],
        superres=None,
        perturbation=None,
        *,
        encoder,
        decoder,
        buffers,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.input_frames = input_frames
        self.future_frames = future_frames
        self.members = members
        self.eval_index = eval_index
        self.eval_fid = eval_fid
        self.use_const = "const" in buffers
        self.freeze_at = freeze_at

        self.encoder = encoder
        self.decoder = decoder
        self.superres = superres
        self.perturbation = perturbation

        for name in buffers:
            self.register_buffer(name, torch.tensor(buffers[name]), persistent=False)

        self.time_embed = TimeEncoding(channels, drop=embed_drop)


    def freeze_stages(self):

        if "encoder" in self.freeze_at:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False

        if "time_embed" in self.freeze_at:
            self.time_embed.eval()
            for param in self.time_embed.parameters():
                param.requires_grad = False

        if "decoder" in self.freeze_at:
            self.decoder.eval()
            for param in self.decoder.parameters():
                param.requires_grad = False


    def normalize(self, x):
        x[:, :, :-1] = (x[:, :, :-1] - self.mean[:-1]) / self.std[:-1]
        x[:, :, -1] = torch.log(1 + x[:, :, -1].clamp(min=0))
        return x

    def inv_normalize(self, x):
        x = x.unsqueeze(1)
        mean = self.mean[self.eval_index]
        std = self.std[self.eval_index]
        x[:, :, :-1] = x[:, :, :-1] * std[:-1] + mean[:-1]
        x[:, :, -1] = x[:, :, -1].exp() - 1
        return x

    def process_data(self, data):
        imgs = data["imgs"].cuda(non_blocking=True)
        imgs = self.normalize(imgs)
        inputs = imgs[:, :self.input_frames]
        targets = imgs[:, self.input_frames:, :self.in_channels]

        if self.superres is None:
            return inputs, targets, inputs, targets[:, -1]

        imgs_hr = data["imgs_hr"].cuda(non_blocking=True)
        inputs_hr = imgs_hr[:, 0]
        targets_hr = imgs_hr[:, -1]
        return inputs, targets, inputs_hr, targets_hr


    def losses_hr(self, output, target, prefix="loss_hr"):
        if output.shape[1] == len(self.eval_index):
            loss_l1 = F.l1_loss(output[:, :-1], target[:, :-1])
        else:
            loss_l1 = F.l1_loss(
                output[:, self.eval_index[:-1]],
                target[:, self.eval_index[:-1]]
            )

        loss_l1_tp  = F.l1_loss(output[:, -1], target[:, -1]) * 0.5

        loss = {
            f"{prefix}_l2": F.mse_loss(output[:, :-1], target[:, :-1]),
            f"{prefix}_l1": loss_l1,
            f"{prefix}_l1_tp": loss_l1_tp,
        }
        return loss


    def losses(self, outputs, targets, prefix="loss"):
        loss_l1 = F.l1_loss(
            outputs[:, :, self.eval_index[:-1]],
            targets[:, :, self.eval_index[:-1]],
        )

        loss_l1_tp  = F.l1_loss(outputs[:, :, -1], targets[:, :, -1]) * 0.5

        loss = {
            f"{prefix}_l2": F.mse_loss(outputs[:, :, :-1], targets[:, :, :-1]),
            f"{prefix}_l1": loss_l1,
            f"{prefix}_l1_tp": loss_l1_tp,
        }
        return loss


    def forward(self, data):

        if not self.training:
            return self.inference_tta(data)

        inputs, targets, inputs_hr, targets_hr = self.process_data(data)
        idx = data["idx"]
        tid = data["tid"]
        shift = data["shift"]
        bid = torch.arange(len(tid))

        future_frames = tid.max() + 1 if self.perturbation is None else self.future_frames

        targets = targets[:, :future_frames]

        sampling_ratio = data.get("sampling_ratio", 0)

        teacher_forcing = sampling_ratio > 0

        bs, ts, _, img_h, img_w = inputs.shape

        if teacher_forcing:
            masks = torch.bernoulli(
                torch.ones(
                    bs, future_frames, 1, 1, 1, device = inputs.device
                ) * sampling_ratio
            )

        output = inputs[:, -1, :self.in_channels]


        if self.use_const:
            const = repeat(self.const, 'c h w -> b t c h w', b=bs, t=ts)
            inputs = torch.cat([inputs, const], dim=2)

        state = self.encoder(inputs)

        outputs = []

        steps = torch.arange(future_frames, device=inputs.device)
        tembs = self.time_embed(steps)

        if self.perturbation is not None:
            loss_kl = 0
            state_p, state_q = self.perturbation.init_state(state)

        for t in range(future_frames):

            if self.perturbation is not None:

                dist_p, state_p = self.perturbation.prior(output, state_p, emb=tembs[t])
                dist_q, state_q = self.perturbation.posterior(targets[:, t], state_q, emb=tembs[t])
                loss_kl += kl_divergence(dist_q, dist_p).sum(dim=(1, 2, 3)).mean(0) * 1e-4
                zt = dist_q.rsample()

                zt = F.interpolate(
                    zt,
                    size=output.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

                output = output + zt

            output, state = self.decoder(output, state, emb=tembs[t])
            outputs.append(output)

            if teacher_forcing:
                mask = masks[:, t]
                output = targets[:, t] * mask + output * (1 - mask)

        outputs = torch.stack(outputs, dim=1)

        outputs_eval = outputs[bid, tid]
        temb = tembs[t].view(1, -1).repeat(bs, 1)

        loss_dict = self.losses(outputs.float(), targets.float())

        if self.superres is not None:

            for i, s in enumerate(shift):
                outputs_eval[i] = torch.roll(outputs_eval[i], -s.item(), dims=-1)

            outputs_hr = self.superres(
                targets_hr,
                lowres_x=outputs_eval,
                lowres_t=temb,
                hist_x=inputs_hr,
            )
            loss_dict.update(self.losses_hr(outputs_hr, targets_hr))

        if self.perturbation is not None:
            loss_dict.update(dict(loss_kl=loss_kl))
            # loss_dict.update(dict(loss_kl=loss_kl/future_frames))

        return loss_dict


    def inference_tta(self, data):
        inputs, targets, inputs_hr, targets_hr = self.process_data(data)

        idx = data["idx"]
        outputs = []

        for _ in range(self.members):
            output_eval, temb = self.inference(inputs)

            if self.superres is not None:
            #if False:
                noise = torch.randn_like(targets_hr)
                output = self.superres(
                    noise,
                    lowres_x=output_eval,
                    lowres_t=temb,
                    hist_x=inputs_hr,
                )
            else:
                output = F.interpolate(
                    output_eval,
                    size=targets_hr.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            outputs.append(output)


        outputs_hr = torch.stack(outputs).mean(dim=0)

        if outputs_hr.shape[1] != len(self.eval_index):
            outputs_hr = outputs_hr[:, self.eval_index]
            targets_hr = targets_hr[:, self.eval_index]


        #from fanjiang.utils.plot_era5 import plot_var

        #output_baseline = F.interpolate(
        #    output_eval[:, self.eval_index],
        #    size=targets_hr.shape[-2:],
        #    mode="bilinear",
        #    align_corners=False,
        #)

        #imgs = torch.cat([targets_hr[0], output_baseline[0], outputs_hr[0]], dim=1)
        #plot_var(
        #    imgs, 
        #    exp="SwinIR", 
        #    init_time="2018", 
        #    lead_time=self.future_frames * 6,
        #)

        results = dict(
            idx=idx,
            output=self.inv_normalize(outputs_hr.clone()),
            target=self.inv_normalize(targets_hr.clone()),
        )

        if self.eval_fid:
            import kornia.augmentation as K
            aug = K.CenterCrop(192)
            results["fid"] = dict(
                output=aug(outputs_hr) / 10,
                target=aug(targets_hr) / 10,
            )

        return results



    def inference(self, inputs):
        bs, ts = inputs.shape[:2]

        output = inputs[:, -1, :self.in_channels]

        if self.use_const:
            const = repeat(self.const, 'c h w -> b t c h w', b=bs, t=ts)
            inputs = torch.cat([inputs, const], dim=2)

        state = self.encoder(inputs)
        steps = torch.arange(self.future_frames, device=inputs.device)
        tembs = self.time_embed(steps)

        if self.perturbation is not None:
            state_p, _ = self.perturbation.init_state(state)

        for t in range(self.future_frames):

            if self.perturbation is not None:
                dist_p, state_p = self.perturbation.prior(output, state_p, emb=tembs[t])
                zt = dist_p.sample()

                zt = F.interpolate(
                    zt,
                    size=output.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

                output = output + zt

            output, state = self.decoder(output, state, emb=tembs[t])

        temb = tembs[t].view(1, -1).repeat(bs, 1)
        return output, temb



    def flops(self):
        flops = 0
        flops += self.encoder.flops()
        flops += self.decoder.flops() * self.future_frames
        return flops
