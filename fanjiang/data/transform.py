import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torchvision import transforms as T
from fanjiang.layers import to_2tuple


class HRollTransform:
    def __init__(self, window, dims=[-1]):
        self.window = window
        self.dims = dims
        self.params = {}

    def __call__(self, img):
        if np.random.rand() < 0.5:
            self.params["shift"] = 0
            return img

        shift = np.random.randint(-self.window, self.window + 1)
        self.params["shift"] = shift
        return torch.roll(img, shift, self.dims)


class Rot90Transform:
    def __init__(self):
        pass
    def __call__(self, img):
        if np.random.rand() < 0.5:
            return img
        k = np.random.randint(-3, 4)
        img = torch.rot90(img, k, dims=(2, 3))
        return img


class CopyPastTransform:
    def __init__(self, thresh=127.5):
        self.thresh = thresh

    def __call__(self, img):
        if np.random.rand() < 0.5:
            return img
        diff = (img[1:] - img[:-1]) >= self.thresh
        img = F.avg_pool2d(img, kernel_size=5, stride=1, padding=2)
        mask = (img <= 10)[:-1] & diff
        return mask


class ResizeCropTransform:
    def __init__(self, crop_size=512, max_size=800):
        self.crop_size = to_2tuple(crop_size)
        self.img_size = np.arange(crop_size, max_size, 32)

    def __call__(self, img):
        if np.random.rand() < 0.5:
            return img

        img_h = img_w = np.random.choice(self.img_size)

        if (img_h, img_w) == self.crop_size:
            return img

        img = F.interpolate(
            img, size=(img_h, img_w), mode="bilinear", align_corners=False
        )

        ref_h , ref_w = img.shape[-2:]
        x1 = np.random.randint(0, ref_w - self.crop_size[1])
        y1 = np.random.randint(0, ref_h - self.crop_size[0])
        x2 = x1 + self.crop_size[1]
        y2 = y1 + self.crop_size[0]

        return img[:, :, y1:y2, x1:x2]


class NoiseTransform:
    def __init__(self):
        pass
    def __call__(self, img):
        if np.random.rand() < 0.5:
            return img
        noise = torch.randn_like(img)
        return img + noise



class BlurTransform:
    def __init__(self, kernel_size=5, sigma=(0.1, 2.0)):
        self.blur = T.GaussianBlur(kernel_size, sigma)

    def __call__(self, img):
        if np.random.rand() < 0.5:
            return img
        return self.blur(img)


class PosterizeTransform:
    def __init__(self, bits=8, p=0.5):
        self.posterize = T.RandomPosterize(bits, p)

    def __call__(self, img):
        if np.random.rand() < 0.5:
            return img
        img = self.posterize(img.byte())
        return img.float()


class ColorJitter:
    def __init__(self, minval=0, maxval=255):
        self.minval = minval
        self.maxval = maxval

    def __call__(self, img):
        if np.random.rand() < 0.5:
            return img
        delta = np.random.uniform(0, 0.1) * 255
        img = torch.clamp(img + delta, self.minval, self.maxval)
        return img


class ErasingTransform:
    def __init__(self, patch_size=4, mask_ratio=0.5, prob=0.5):
        self.prob = prob
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

    def __call__(self, img):
        if np.random.rand() < self.prob:
            return img

        bs, _, _, img_h, img_w = img.shape
        grid_h = img_h // self.patch_size
        grid_w = img_w // self.patch_size

        img = rearrange(
            img,
            'n t c (h h2) (w w2) -> (n t) (h w) (h2 w2 c)',
            h2=self.patch_size, w2=self.patch_size,
        )

        noise = torch.rand(bs, img.shape[1], device=img.device)
        idx = noise.argsort(dim = -1)

        num_masked = int(self.mask_ratio * img.shape[1])
        idx = idx[:, :num_masked]

        idx_n = torch.arange(bs, device=img.device).unsqueeze(1)
        img[idx_n, idx] = 0

        img = rearrange(
            img, '(n t) (h w) (h2 w2 c) -> n t c (h h2) (w w2)',
            n=bs,
            h=grid_h, w=grid_w,
            h2=self.patch_size, w2=self.patch_size,
        )
        return img


class AugmentationList:
    def __init__(self, augs, training=True):
        self.training = training
        self.augs = augs
        self.params = {}

    def __call__(self, img):

        if not self.training or len(self.augs) == 0:
            return img

        for aug in self.augs:
            img = aug(img)
            self.params.update(aug.params)

        return img
