import itertools
import numpy as np
import torch
import torch.distributed as dist
from einops.einops import rearrange
from fanjiang.core import DatasetEvaluator
from fanjiang.utils import comm
from scipy import linalg

from fanjiang.builder import METRICS

from .fid_inception import InceptionV3

@METRICS.register()
class FID(DatasetEvaluator):

    def __init__(self, in_channels):

        model = InceptionV3(
            in_chans=in_channels,
            normalize_input=False,
            resize_input=False,
        ).cuda()

        self.model = model.eval()

    def reset(self):
        self.real_feats = []
        self.fake_feats = []
        self.real_mean = None
        self.real_cov = None

    def process(self, outputs):
        output = outputs["fid"]["output"]
        target = outputs["fid"]["target"]

        self.feed_op(output, "fakes")
        self.feed_op(target, "reals")


    def evaluate(self):
        """Summarize the results.

        Returns:
            dict | list: Summarized results.
        """

        real_feats = comm.gather(self.real_feats, dst=0)
        real_feats = list(itertools.chain(*real_feats))

        fake_feats = comm.gather(self.fake_feats, dst=0)
        fake_feats = list(itertools.chain(*fake_feats))

        # calculate reference inception stat
        if not comm.is_main_process():
            return

        real_feats = torch.cat(real_feats, dim=0)
        feats_np = real_feats.numpy()
        self.real_mean = np.mean(feats_np, 0)
        self.real_cov = np.cov(feats_np, rowvar=False)

        # calculate fake inception stat
        fake_feats = torch.cat(fake_feats, dim=0)
        fake_feats_np = fake_feats.numpy()
        fake_mean = np.mean(fake_feats_np, 0)
        fake_cov = np.cov(fake_feats_np, rowvar=False)

        # calculate distance between real and fake statistics
        fid, _, _ = self._calc_fid(fake_mean, fake_cov, self.real_mean,
                                        self.real_cov)

        result = {"FID": fid}
        return result


    @torch.no_grad()
    def feed_op(self, batch, mode):
        feat = self.model(batch)[0].view(batch.shape[0], -1)

        # # gather all of images if using distributed training
        # if dist.is_initialized():
        #     ws = dist.get_world_size()
        #     placeholder = [torch.zeros_like(feat) for _ in range(ws)]
        #     dist.all_gather(placeholder, feat)
        #     feat = torch.cat(placeholder, dim=0)

        # # in distributed training, we only collect features at rank-0.
        # if (dist.is_initialized()
        #         and dist.get_rank() == 0) or not dist.is_initialized():

        if mode == 'reals':
            self.real_feats.append(feat.cpu())
        elif mode == 'fakes':
            self.fake_feats.append(feat.cpu())
        else:
            raise ValueError(
                f"The expected mode should be set to 'reals' or 'fakes,\
                but got '{mode}'")


    @staticmethod
    def _calc_fid(sample_mean, sample_cov, real_mean, real_cov, eps=1e-6):
        """Refer to the implementation from:

        https://github.com/rosinality/stylegan2-pytorch/blob/master/fid.py#L34
        """
        cov_sqrt, _ = linalg.sqrtm(sample_cov @ real_cov, disp=False)

        if not np.isfinite(cov_sqrt).all():
            print('product of cov matrices is singular')
            offset = np.eye(sample_cov.shape[0]) * eps
            cov_sqrt = linalg.sqrtm(
                (sample_cov + offset) @ (real_cov + offset))

        if np.iscomplexobj(cov_sqrt):
            if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
                m = np.max(np.abs(cov_sqrt.imag))

                raise ValueError(f'Imaginary component {m}')

            cov_sqrt = cov_sqrt.real

        mean_diff = sample_mean - real_mean
        mean_norm = mean_diff @ mean_diff

        trace = np.trace(sample_cov) + np.trace(
            real_cov) - 2 * np.trace(cov_sqrt)

        fid = mean_norm + trace

        return fid, mean_norm, trace


