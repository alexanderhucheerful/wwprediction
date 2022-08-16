import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def sample_schedule(steps, rho, sigma_min, sigma_max):
    N = steps
    inv_rho = 1 / rho
    steps = torch.arange(steps, dtype = torch.float32)
    sigmas = (sigma_max ** inv_rho + steps / (N - 1) * (sigma_min ** inv_rho - sigma_max ** inv_rho)) ** rho
    sigmas = F.pad(sigmas, (0, 1), value = 0.) # last step is sigma value of 0.
    return sigmas


class DSM(nn.Module):
    def __init__(
        self,
        steps=32,
        sigma_min=0.002,
        sigma_max=80,
        sigma_data=0.5,
        rho=7,
        p_mean=-1.2,
        p_std=1.2,
        s_churn=90,
        s_tmin=0.05,
        s_tmax=50,
        s_noise=1.003,
        *,
        unet,
    ):
        super().__init__()
        self.steps = steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.p_mean = p_mean
        self.p_std = p_std
        self.s_churn = s_churn
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise

        sigmas = sample_schedule(steps, rho, sigma_min, sigma_max)

        gammas = torch.where(
            (sigmas >= s_tmin) & (sigmas <= s_tmax),
            min(s_churn / steps, math.sqrt(2) - 1), 0.
        )

        self.register_buffer("sigmas", sigmas)
        self.register_buffer("gammas", gammas)

        self.unet = unet


    @torch.no_grad()
    def p_sample_loop(self, shape, lowres_x, lowres_t=None):
        device = lowres_x.device

        init_sigma = self.sigmas[0]
        x = init_sigma * torch.randn(shape, device=device)

        for i in range(self.steps):
            sigma = self.sigmas[i:i+1]
            sigma_next = self.sigmas[i+1:i+2]
            gamma = self.gammas[i:i+1]

            eps = self.s_noise * torch.randn(shape, device=device)

            sigma_hat = sigma + gamma * sigma
            x_hat = x + (sigma_hat ** 2 - sigma ** 2).sqrt() * eps

            x_out = self.forward_unet(x_hat, sigma_hat, lowres_x=lowres_x)
            d1 = (x_hat - x_out) / sigma_hat
            x_next = x_hat + (sigma_next - sigma_hat) * d1

            if sigma_next != 0:
                x_out = self.forward_unet(x_hat, sigma_hat, lowres_x=lowres_x)
                d2 = (x_next - x_out) / sigma_next
                x_next = x_hat + 0.5 * (sigma_next - sigma_hat) * (d1 + d2)

            x = x_next

        return x


    def c_skip(self, sigma):
        return (self.sigma_data ** 2) / (sigma ** 2 + self.sigma_data ** 2)

    def c_out(self, sigma):
        return sigma * self.sigma_data * (self.sigma_data ** 2 + sigma ** 2) ** -0.5

    def c_in(self, sigma):
        return 1 * (sigma ** 2 + self.sigma_data ** 2) ** -0.5

    def c_noise(self, sigma):
        return torch.log(sigma.clamp(min=1e-20)) * 0.25

    def loss_weight(self, sigma):
        return (sigma ** 2 + self.sigma_data ** 2) * (sigma * self.sigma_data) ** -2

    def noise_distribution(self, x):
        return (self.p_mean + self.p_std * torch.randn((x.size(0),), device = x.device)).exp()


    def forward_unet(self, xt, sigma, lowres_x=None):
        sigma_ = sigma.reshape(-1, 1, 1, 1)

        output = self.unet(
            self.c_in(sigma_) * xt,
            self.c_noise(sigma),
            lowres_x=lowres_x
        )

        output = self.c_skip(sigma_) * xt + self.c_out(sigma_) * output
        return output


    def forward(self, x, lowres_x=None):
        sigma = self.noise_distribution(x)
        noise = torch.randn_like(x) * sigma.reshape(-1, 1, 1, 1)

        xt = x + noise
        output = self.forward_unet(xt, sigma, lowres_x=lowres_x)

        loss = F.mse_loss(output, x, reduction = 'none').mean(dim=(1, 2, 3))
        loss = torch.mean(self.loss_weight(sigma) * loss)
        return loss



