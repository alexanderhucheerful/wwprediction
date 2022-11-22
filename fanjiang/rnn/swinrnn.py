import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fanjiang.builder import MODELS
from fanjiang.layers import Conv2d, get_norm, to_2tuple, trunc_normal_
from fanjiang.transformer import SwinEncoder, SwinTransformerBlock, StyleSwinTransformerBlock
from fanjiang.layers import constant_fill
from fanjiang.layers import StepEncoding
from torch.distributions import Normal, MultivariateNormal, kl_divergence
import torch.autograd as autograd


class SwinLayer(nn.Module):

    def __init__(
            self,
            in_channels,
            channels,
            input_resolution,
            depth,
            num_heads,
            window_size,
            mlp_ratio=4.,
            use_noise=False,
            use_style=True,
        ):

        super().__init__()
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_noise = use_noise
        self.use_style = use_style

        self.attn_layers = nn.ModuleList()

        for i in range(depth):
            attn = SwinTransformerBlock(
                dim=channels,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
            )
            self.attn_layers.append(attn)

        self.proj = nn.Linear(in_channels + channels, channels, bias=False)
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, xt, feat, zt=None, emb=None):
        xt =  rearrange(xt, 'n c h w -> n (h w) c')
        feat = self.proj(torch.cat([xt, feat], dim=-1))

        if emb is not None:
            feat = feat + emb.unsqueeze(0).unsqueeze(0)

        if zt is not None:
            zt = rearrange(zt, 'n c h w -> n (h w) c')
            feat = feat + zt

        feats = []
        for i, attn in enumerate(self.attn_layers):
            feat = attn(feat)
            feats.append(feat)

        return feats



class GaussianLayer(nn.Module):

    def __init__(
            self,
            in_channels,
            channels,
            embed_dim,
            noise_dim,
            image_size,
            patch_size,
            window_size,
            depth=2,
            num_heads=8,
            mlp_ratio=4.,
            init_mean=None,
            init_var=None,
            scale=4,
        ):

        super().__init__()
        self.depth = depth
        self.scale = scale
        self.patch_size = patch_size

        self.shortcut = nn.Conv2d(in_channels, channels, 1)

        self.res_embed = ResEmbed(in_channels, embed_dim, patch_size)
        self.linear_proj = nn.Linear(embed_dim + channels, channels, bias=False)

        image_size = [sz // patch_size for sz in image_size]
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

        self.conv_proj = Conv2d(
            channels * depth, channels, kernel_size=3, padding=1,
            activation=F.gelu,
        )

        self.mean = nn.Conv2d(channels, noise_dim, 3, padding=1)
        # self.logvar = nn.Conv2d(channels, noise_dim, 3, padding=1)
        self.cov = nn.Conv2d(channels, noise_dim, 3, padding=1)

        if init_mean is not None:
            constant_fill(self.mean, init_mean)

        if init_var is not None:
            # constant_fill(self.logvar, init_var)
            constant_fill(self.cov, init_var)

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def cal_L(self, cov):
        cov = cov @ cov.transpose(-2, -1) # (256, 128, 128)
        L = torch.tril(cov)
        L = L + torch.eye(L.shape[-1], device=L.device).unsqueeze(0)
        return L

    def forward(self, xt, feat, emb=None):
        img_h, img_w = xt.shape[-2:]

        shortcut = self.shortcut(xt)
        if self.patch_size > 1:
            shortcut = F.interpolate(shortcut, size=self.image_size, mode='bilinear', align_corners=False)

        xt = self.res_embed(xt)

        xt =  rearrange(xt, 'n c h w -> n (h w) c')
        feat = self.linear_proj(torch.cat([xt, feat], dim=-1))

        if emb is not None:
            feat = feat + emb.unsqueeze(0).unsqueeze(0)

        feats = []
        for _, blk in enumerate(self.blocks):
            feat = blk(feat)
            feats.append(feat)

        state = feats[-1]
        feat = torch.cat(feats, dim=-1)
        feat = rearrange(feat,  'n (h w) c -> n c h w', h=self.image_size[0])
        feat = self.conv_proj(feat) + shortcut

        # mean = self.mean(feat)
        # logvar = self.logvar(feat)
        # sigma = logvar.mul(0.5).exp()
        # dist = Normal(mean, sigma)

        scale = self.scale
        mu = self.mean(feat)
        cov = self.cov(feat)
        mu = F.interpolate(mu, size=(img_h//scale, img_w//scale), mode='bilinear', align_corners=False)
        cov = F.interpolate(cov, size=(img_h//scale, img_w//scale), mode='bilinear', align_corners=False)

        mu = rearrange(mu, 'n c h w -> (n c) (h w)') # (256, 128)
        cov = rearrange(cov, 'n c h w -> (n c) (h w) 1') # (256, 128, 1)
        L = self.cal_L(cov) # (256, 128, 128)
        dist = MultivariateNormal(mu, scale_tril=L)

        return dist, state #, mu, L


class ResEmbed(nn.Module):
    def __init__(self,
            in_channels,
            embed_dim,
            patch_size=4,
            groups=1,
        ):
        super().__init__()
        patch_size = to_2tuple(patch_size)

        mid_channels = 2 * in_channels * (patch_size[0] * patch_size[1])

        self.down = Conv2d(
            in_channels, mid_channels,
            kernel_size=(patch_size[0], patch_size[1]),
            stride=(patch_size[0], patch_size[1]),
            padding=(0, 0),
            norm=get_norm("GN", mid_channels, groups=1)
        )

        self.conv1 = nn.Conv2d(mid_channels, mid_channels * 2, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(mid_channels * 2, mid_channels, kernel_size=(3, 3), padding=(1, 1))

        self.proj = nn.Conv2d(
            mid_channels, embed_dim,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
        )


    def forward(self, x):
        x = self.down(x)

        residual = x
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        x += residual

        x = self.proj(x)
        return x


class SwinHead(nn.Module):
    def __init__(
            self,
            in_channels,
            embed_dim,
            input_frames,
            image_size,
            window_size=8,
            patch_size=4,
            depth=6,
            num_layers=4,
            num_heads=[4, 8, 16, 32],
            dims=[16, 32, 64, 128],
        ):
        super().__init__()
        self.down_layers = nn.ModuleList()
        self.attn_layers = nn.ModuleList()
        self.proj_layers = nn.ModuleList()
        self.pred_layers = nn.ModuleList()

        self.res_embed = ResEmbed(in_channels, embed_dim, patch_size)

        channels = in_channels * patch_size ** 2
        image_size = [sz // patch_size for sz in image_size]

        for i in range(num_layers):
            input_resolution = (
                image_size[0] // 2**i, image_size[1] //  2**i
            )

            self.attn_layers.append(
                SwinLayer(
                    embed_dim,
                    dims[i],
                    input_resolution, depth, num_heads[i], window_size
                )
            )

            self.proj_layers.append(
                Conv2d(
                    dims[i] * depth, dims[i], kernel_size=3, padding=1,
                    activation=F.gelu,
                )
            )

            self.pred_layers.append(
                Conv2d(
                    dims[i],
                    # embed_dim + dims[i],
                    # dims[i] if i == num_layers - 1 else channels + dims[i],
                    in_channels, kernel_size=3, stride=1, padding=1,
                )
            )



        if patch_size > 1:
            self.final_layer = nn.ConvTranspose2d(
                in_channels*num_layers, 
                in_channels, 
                kernel_size=patch_size, 
                stride=patch_size
            )
        else:
            self.final_layer = nn.Conv2d(
                in_channels * num_layers,
                in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            )


    def forward(self, inputs, states, zt=None, emb=None):
        zt_i = zt
        xt = self.res_embed(inputs)
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

            if zt is not None:
                zt_i = F.interpolate(
                    zt,
                    size=(h, w),
                    mode='bilinear',
                    align_corners=False
                )

            all_states = self.attn_layers[i](xt_i, states[i], zt=zt_i, emb=emb)

            states[i] = all_states[-1]
            feat = torch.cat(all_states, dim=-1)
            feat = rearrange(feat,  'n (h w) c -> n c h w', h=h, w=w)
            feat = self.proj_layers[i](feat)

            feat = F.interpolate(
                feat,
                size=(img_h, img_w),
                mode='bilinear',
                align_corners=False
            )
            output = self.pred_layers[i](feat)
            outputs.append(output)

        output = torch.cat(outputs, dim=1)
        output = self.final_layer(output) + inputs
        return output, states



@MODELS.register()
class SwinRNN(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        embed_dim,
        noise_dim,
        input_frames,
        future_frames,
        image_size,
        patch_size=8,
        window_size=8,
        dims=[256, 256, 256, 256],
        eval_index=[7, 23, 65, 66, 67],
        with_vrnn=False,
        cov_scale=4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_frames = input_frames
        self.future_frames = future_frames
        self.eval_index = eval_index
        self.with_vrnn = with_vrnn
        self.scale = cov_scale

        self.encoder = SwinEncoder(
            in_channels=in_channels,
            embed_dim=embed_dim,
            input_frames=input_frames,
            image_size=image_size,
            patch_size=patch_size,
            window_size=window_size,
            dims=dims
        )


        self.decoder = SwinHead(
            in_channels=out_channels,
            embed_dim=embed_dim,
            input_frames=input_frames,
            image_size=image_size,
            window_size=window_size,
            patch_size=patch_size,
            dims=dims,
        )

        if self.with_vrnn:
            self.dist_p = GaussianLayer(
                in_channels=out_channels,
                channels=dims[0],
                embed_dim=embed_dim,
                noise_dim=noise_dim,
                image_size=image_size,
                patch_size=patch_size,
                window_size=window_size,
                init_mean=0,
                init_var=0,
                scale=self.scale,
            )

            self.dist_q = GaussianLayer(
                in_channels=out_channels,
                channels=dims[0],
                embed_dim=embed_dim,
                noise_dim=noise_dim,
                image_size=image_size,
                patch_size=patch_size,
                window_size=window_size,
                scale=self.scale
            )

        self.register_buffer("sampling_ratio", torch.zeros([]))
        self.register_buffer("sampling_scale", torch.zeros([]))
        self.step_emb = StepEncoding(dims[0])


    def process_data(self, data):
        radars = data["radars"]
        inputs = radars[:, :self.input_frames]
        targets = radars[:, self.input_frames:]
        return inputs, targets

    def process_weather_bench(self, data):
        fields = data["fields"]
        inputs = fields[:, :self.input_frames]
        targets = fields[:, self.input_frames:, :self.out_channels]
        return inputs, targets

    def inference_tta(self, data, info={}, num=1):
        inputs, targets = self.process_weather_bench(data)
        
        outputs = []
        for i in range(num):
            output = self.inference(inputs, info) # n,t,c,h,w
            outputs.append(output)

        outputs = torch.stack(outputs, dim=0) # s,n,t,c,h,w
        outputs = torch.mean(outputs, dim=0) # n,t,c,h,w

        info["output"] = outputs
        info["target"] = targets
        return info


    def forward(self, data, info={}):
        # return self.inference_tta(data, info, num=10)
        
        inputs, targets = self.process_weather_bench(data)

        if self.training:
            self.sampling_ratio = inputs.new_tensor(info.get("sampling_ratio", 0))
            # self.sampling_scale = inputs.new_tensor(info.get("sampling_scale", 0))

        teacher_forcing = self.training and self.sampling_ratio > 0

        # if self.training:
        #     future_frames = np.random.randint(1, self.future_frames + 1)
        # else:
        future_frames = self.future_frames

        bs, hist_l, _, img_h, img_w = inputs.shape

        if teacher_forcing:
            masks = torch.bernoulli(
                torch.ones(
                    bs, future_frames, 1, 1, 1, device = inputs.device
                ) * self.sampling_ratio
            )

        const = inputs.new_tensor(info["const"])
        const = const[None, None].repeat(bs, hist_l, 1, 1, 1)

        feats = self.encoder(torch.cat([inputs, const], dim=2))
        output = inputs[:, -1, :self.out_channels]

        outputs = []
        steps = torch.arange(future_frames, device=inputs.device)
        embedings = self.step_emb(steps.view(-1, 1))

        if self.with_vrnn:
            loss_kl = 0
            state_p = feats[0].clone()
            state_q = feats[0].clone()


        for t in range(future_frames):

            if self.with_vrnn:
                dist_p, state_p = self.dist_p(output, state_p, emb=embedings[t])
                dist_q, state_q = self.dist_q(targets[:, t], state_q, emb=embedings[t])
                loss_kl += kl_divergence(dist_q, dist_p).view(bs, -1).sum(dim=-1).mean() * 1e-4
                zt = dist_q.rsample() if self.training else dist_p.rsample()
                # output = output + zt # * self.sampling_scale
                zt = zt.view(bs, -1, img_h//self.scale, img_w//self.scale)
                output = output + F.interpolate(zt, size=(img_h, img_w), mode='bilinear', align_corners=False)

            
            output, feats = self.decoder(output, feats, emb=embedings[t])
            outputs.append(output)

            if teacher_forcing:
                mask = masks[:, t]
                output = targets[:, t] * mask + output * (1 - mask)

        outputs = torch.stack(outputs, dim=1)
        targets = targets[:, :future_frames]


        if self.training:

            info["loss"] = {
                "loss_l2": F.mse_loss(outputs[:, :, :-1], targets[:, :, :-1]),
                "loss_l1": F.l1_loss(outputs[:, :, self.eval_index], targets[:, :, self.eval_index]),
                "loss_l1_tp": F.l1_loss(outputs[:, :, -1], targets[:, :, -1]) * 0.5,
            }

            if self.with_vrnn:
                info["loss"].update(dict(loss_kl=loss_kl))


        info["output"] = outputs
        info["target"] = targets
        return info


    def inference(self, inputs, info):
        bs, lt, _, img_h, img_w = inputs.shape

        const = inputs.new_tensor(info["const"])
        const = const[None, None].repeat(bs, lt, 1, 1, 1)

        feats = self.encoder(torch.cat([inputs, const], dim=2))

        output = inputs[:, -1, :self.out_channels]

        outputs = []
        steps = torch.arange(self.future_frames, device=inputs.device)
        embedings = self.step_emb(steps.view(-1, 1))

        state_p = feats[0].clone()
        for t in range(self.future_frames):

            if self.with_vrnn:
                dist_p, state_p = self.dist_p(output, state_p, emb=embedings[t])
                zt = dist_p.rsample()
                # output = output + zt # * self.sampling_scale
                zt = zt.view(bs, -1, img_h//self.scale, img_w//self.scale)
                output = output + F.interpolate(zt, size=(img_h, img_w), mode='bilinear', align_corners=False)

            output, feats = self.decoder(output, feats, emb=embedings[t], zt=zt)
            outputs.append(output)
        
        outputs = torch.stack(outputs, dim=1)
        return outputs
