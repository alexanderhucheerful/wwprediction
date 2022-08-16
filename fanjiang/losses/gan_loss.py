import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from fanjiang.builder import CRITERIONS


@CRITERIONS.register()
class GANLoss(nn.Module):
    """Define GAN loss.

    Args:
        gan_type (str): Support 'vanilla', 'lsgan', 'wgan', 'hinge',
            'wgan-logistic-ns'.
        real_label_val (float): The value for real label. Default: 1.0.
        fake_label_val (float): The value for fake label. Default: 0.0.
        loss_weight (float): Loss weight. Default: 1.0.
            Note that loss_weight is only for generators; and it is always 1.0
            for discriminators.
    """

    def __init__(self,
                 gan_type,
                 real_label_val=1.0,
                 fake_label_val=0.0,
                 loss_weight=1.0):
        super().__init__()
        self.gan_type = gan_type
        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan':
            self.loss = self._wgan_loss
        elif self.gan_type == 'wgan-logistic-ns':
            self.loss = self._wgan_logistic_ns_loss
        elif self.gan_type == 'hinge':
            self.loss = nn.ReLU()
        else:
            raise NotImplementedError(
                f'GAN type {self.gan_type} is not implemented.')

    def _wgan_loss(self, input, target):
        """wgan loss.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return -input.mean() if target else input.mean()

    def _wgan_logistic_ns_loss(self, input, target):
        """WGAN loss in logistically non-saturating mode.

        This loss is widely used in StyleGANv2.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """

        return F.softplus(-input).mean() if target else F.softplus(
            input).mean()

    def get_target_label(self, input, target_is_real):
        """Get target label.

        Args:
            input (Tensor): Input tensor.
            target_is_real (bool): Whether the target is real or fake.

        Returns:
            (bool | Tensor): Target tensor. Return bool for wgan, otherwise, \
                return Tensor.
        """

        if self.gan_type in ['wgan', 'wgan-logistic-ns']:
            return target_is_real
        target_val = (
            self.real_label_val if target_is_real else self.fake_label_val)
        return input.new_ones(input.size()) * target_val

    def forward(self, input, target_is_real, is_disc=False):
        """
        Args:
            input (Tensor): The input for the loss module, i.e., the network
                prediction.
            target_is_real (bool): Whether the targe is real or fake.
            is_disc (bool): Whether the loss for discriminators or not.
                Default: False.

        Returns:
            Tensor: GAN loss value.
        """
        target_label = self.get_target_label(input, target_is_real)
        if self.gan_type == 'hinge':
            if is_disc:  # for discriminators in hinge-gan
                input = -input if target_is_real else input
                loss = self.loss(1 + input).mean()
            else:  # for generators in hinge-gan
                loss = -input.mean()
        else:  # other gan types
            loss = self.loss(input, target_label)

        # loss_weight is always 1.0 for discriminators
        return loss if is_disc else loss * self.loss_weight


@CRITERIONS.register()
class PathRegularizer(nn.Module):
    def __init__(
            self,
            decay=0.01,
            interval=None,
            loss_weight=1.,
        ):
        super().__init__()    
        self.decay = decay
        self.interval = interval
        self.loss_weight = loss_weight
        self.register_buffer('pl_mean', torch.tensor(0.).cuda())

    def forward(self, fake_img, latents):

        noise = torch.randn_like(fake_img) / np.sqrt(
            fake_img.shape[2] * fake_img.shape[3])

        grad = autograd.grad(
            outputs=[(fake_img * noise).sum()],
            inputs=[latents],
            create_graph=True,
            only_inputs=True
        )[0]

        pl_lengths = grad.square().sum(2).mean(1).sqrt()
        pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.decay)
        pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.decay)
        self.pl_mean.copy_(pl_mean.detach())

        pl_penalty = (pl_lengths - pl_mean).square()
        loss = pl_penalty * self.loss_weight
        return loss.mean()



@CRITERIONS.register()
class R1Regularizer(nn.Module):
    def __init__(
            self,
            gamma=10,
            interval=1,
            loss_weight=1.,
        ):
        super().__init__()    
        self.interval = interval
        self.loss_weight = gamma / 2 * interval * loss_weight

    def forward(self, real_img, real_logits):
        grad = autograd.grad(
            outputs=[real_logits.sum()],
            inputs=[real_img],
            create_graph=True,
            only_inputs=True
        )[0]

        # r1_penalty = grad.square().sum([1,2,3])
        r1_penalty = (grad.flatten(1).norm(2, dim=1) ** 2)        
        loss = r1_penalty * self.loss_weight
        return loss.mean()
