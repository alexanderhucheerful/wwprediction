from einops.einops import rearrange
import torch
import torch.nn as nn
from .helpers import to_2tuple

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.real1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        self.imag1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        self.real2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        self.imag2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))


    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        weights1 = torch.complex(self.real1, self.imag1).clone()
        weights2 = torch.complex(self.real2, self.imag2).clone()

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class NeuralOperator(nn.Module):
    def __init__(
            self, 
            in_channels, 
            out_channels, 
            image_size=128, 
            patch_size=16, 
        ):
        super(NeuralOperator, self).__init__()
        image_size = to_2tuple(image_size)
        patch_size = to_2tuple(patch_size)

        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])

        self.in_channels = in_channels
        self.out_channels = out_channels

        scale = (1 / (in_channels * out_channels))
        self.real = nn.Parameter(scale * torch.rand(in_channels, out_channels, patch_size[0], patch_size[1]))
        self.imag = nn.Parameter(scale * torch.rand(in_channels, out_channels, patch_size[0], patch_size[1]))

        num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Linear(num_patches * in_channels, 2 * in_channels, bias=False)
        self.invproj = nn.Linear(2 * out_channels, num_patches * out_channels, bias=False)


    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        x = rearrange(
            x, 'n c (h h2) (w w2) -> n (h2 w2) (h w c)', 
            h2=self.patch_size[0], w2=self.patch_size[1]
        )

        coef = self.proj(x) 
        coef = rearrange(coef, 'n (h w) c -> n c h w')

        x_ft = torch.complex(coef[:, :self.in_channels], coef[:, self.in_channels:])
        weight = torch.complex(self.real, self.imag)
        out_ft = self.compl_mul2d(x_ft, weight)
        out_ft = torch.cat([out_ft.real, out_ft.imag], dim=1) 
        out_ft = rearrange(out_ft, 'n c h w -> n (h w) c')
        x = self.invproj(out_ft)

        x = rearrange(
            x, 'n (h2 w2) (h w c) -> n c (h h2) (w w2)', 
            h2=self.patch_size[0], w2=self.patch_size[1], 
            h=self.grid_size[0], w=self.grid_size[1]
        )
        return x


class MultiHeadOperator(nn.Module):
    def __init__(
            self, 
            image_size, 
            in_channels, 
            out_channels, 
        ):
        super(MultiHeadOperator, self).__init__()

        self.layers = nn.ModuleList()
        for patch_size in [4, 8, 16, 32]:
            self.layers.append(
                NeuralOperator(image_size, patch_size, in_channels, out_channels)
            )

    def forward(self, x):
        feats = []
        for layer in self.layers:
            feats.append(layer(x))
        feats = torch.cat(feats, dim=1)
        return feats


class FourierConv2d(nn.Module):

    def __init__(
            self, 
            in_channels, 
            out_channels, 
            groups=1, 
            fft_norm='ortho'
        ):
        super(FourierConv2d, self).__init__()

        self.conv_layer = nn.Conv2d(
            in_channels=in_channels * 2,
            out_channels=out_channels * 2,
            kernel_size=1, 
            stride=1, 
            padding=0, 
            groups=groups, 
            bias=False
        )
        self.fft_norm = fft_norm

    def forward(self, x):
        batch = x.shape[0]

        ffted = torch.fft.rfftn(x, dim=(-2, -1), norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)

        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(0, 1, 3, 4, 2).contiguous() 
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])
        output = torch.fft.irfftn(ffted, s=x.shape[-2:], dim=(-2, -1), norm=self.fft_norm)
        return output
