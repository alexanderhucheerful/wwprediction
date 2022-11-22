import itertools
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from fanjiang.builder import METRICS
from fanjiang.core import DatasetEvaluator
from fanjiang.utils import comm
from fanjiang.utils.file_io import PathManager
from fanjiang.utils.logger import log_first_n
from tabulate import tabulate
from termcolor import colored


@METRICS.register()
class MSE(DatasetEvaluator):
    def __init__(self, interval, eval_index, eval_names, eval_frames):
        super().__init__()
        self.interval = interval
        self.eval_index = eval_index
        self.eval_names = eval_names
        self.eval_frames = eval_frames

    def reset(self):
        self.idx = []

        self.mse = {}
        self.ssim = {}
        self.predictions = {}

        for t in self.eval_frames:
            for name in self.eval_names:
                key = "{}_{}".format(name, t)
                self.mse[key] = []
                self.ssim[key] = []
                self.predictions[key] = []


    def process(self, outputs):
        idx = outputs["idx"]
        self.idx.append(idx)
        # import pdb
        # pdb.set_trace()
        output = outputs["output"].cpu() # n,t,c,h,w
        target = outputs["target"].cpu()

        mean = outputs["mean"] # (68,)
        std = outputs["std"] # (68,)

        for t in self.eval_frames: # 20

            for j, name in zip(self.eval_index, self.eval_names):

                if name == "tp":
                    output_j = output[:, t, j].exp() - 1
                    target_j = target[:, t, j].exp() - 1
                else:
                    output_j = output[:, t, j] * std[j] + mean[j] # n,h,w
                    target_j = target[:, t, j] * std[j] + mean[j]

                mse = F.mse_loss(target_j, output_j).sqrt()
                data_range = target_j.max() - target_j.min()
                ssim = _ssim_compute(target_j[:, None], output_j[:, None], data_range)

                key = "{}_{}".format(name, t)
                self.mse[key].append(mse.item())
                self.ssim[key].append(ssim.item())
                self.predictions[key].append(output_j)


    def evaluate(self, save_dir=None):
        comm.synchronize()
        # import pdb
        # pdb.set_trace()
        for t in self.eval_frames:
            for name in self.eval_names:
                key = "{}_{}".format(name, t)

                mse = comm.gather(self.mse[key], dst=0)
                ssim = comm.gather(self.ssim[key], dst=0)

                self.mse[key] = list(itertools.chain(*mse))
                self.ssim[key] = list(itertools.chain(*ssim))

                predictions = comm.gather(self.predictions[key], dst=0)
                self.predictions[key] = list(itertools.chain(*predictions))

        idx = comm.gather(self.idx, dst=0)
        idx = list(itertools.chain(*idx))

        if not comm.is_main_process():
            return

        csv_results = []
        for t in self.eval_frames:
            lead_time = (t + 1) * self.interval
            result = [lead_time]
            for name in self.eval_names:
                key = "{}_{}".format(name, t)
                mse = np.mean(self.mse[key])
                ssim = np.mean(self.ssim[key])
                result.extend([mse, ssim])
            csv_results.append(result)

        columns = ["Hours"]
        for name in self.eval_names:
            columns.extend([f"MSE_{name}", f"SSIM_{name}"])

        csv_results = tabulate(
            csv_results,
            headers=columns,
            tablefmt="grid",
            numalign="left",
            stralign="center",
        )

        if save_dir:
            save_f = os.path.join(save_dir, "results.txt")
            with PathManager.open(save_f, "w") as f:
                f.write(csv_results)

            PathManager.mkdirs(save_dir)
            self.predictions["idx"] = idx
            save_f = os.path.join(save_dir, "predictions.pth")

            with PathManager.open(save_f, "wb") as f:
                torch.save(self.predictions, f)

        log_first_n(
            logging.INFO,
            "Mse results:\n" + colored(csv_results, "cyan"),
            key="message",
        )

        return {}



def _gaussian(kernel_size, sigma, dtype: torch.dtype, device: torch.device):
    """Computes 1D gaussian kernel.
    Args:
        kernel_size: size of the gaussian kernel
        sigma: Standard deviation of the gaussian kernel
        dtype: data type of the output tensor
        device: device of the output tensor
    Example:
        >>> _gaussian(3, 1, torch.float, 'cpu')
        tensor([[0.2741, 0.4519, 0.2741]])
    """
    dist = torch.arange(start=(1 - kernel_size) / 2, end=(1 + kernel_size) / 2, step=1, dtype=dtype, device=device)
    gauss = torch.exp(-torch.pow(dist / sigma, 2) / 2)
    return (gauss / gauss.sum()).unsqueeze(dim=0)  # (1, kernel_size)


def _gaussian_kernel(
    channel, kernel_size, sigma, dtype: torch.dtype, device: torch.device
):
    """Computes 2D gaussian kernel.
    Args:
        channel: number of channels in the image
        kernel_size: size of the gaussian kernel as a tuple (h, w)
        sigma: Standard deviation of the gaussian kernel
        dtype: data type of the output tensor
        device: device of the output tensor
    Example:
        >>> _gaussian_kernel(1, (5,5), (1,1), torch.float, "cpu")
        tensor([[[[0.0030, 0.0133, 0.0219, 0.0133, 0.0030],
                  [0.0133, 0.0596, 0.0983, 0.0596, 0.0133],
                  [0.0219, 0.0983, 0.1621, 0.0983, 0.0219],
                  [0.0133, 0.0596, 0.0983, 0.0596, 0.0133],
                  [0.0030, 0.0133, 0.0219, 0.0133, 0.0030]]]])
    """

    gaussian_kernel_x = _gaussian(kernel_size[0], sigma[0], dtype, device)
    gaussian_kernel_y = _gaussian(kernel_size[1], sigma[1], dtype, device)
    kernel = torch.matmul(gaussian_kernel_x.t(), gaussian_kernel_y)  # (kernel_size, 1) * (1, kernel_size)
    return kernel.expand(channel, 1, kernel_size[0], kernel_size[1])


def _ssim_compute(
        preds,
        target,
        data_range,
        kernel_size = (11, 11),
        sigma = (1.5, 1.5),
        k1 = 0.01,
        k2 = 0.03,
    ):

    c1 = pow(k1 * data_range, 2)
    c2 = pow(k2 * data_range, 2)
    device = preds.device

    channel = preds.size(1)
    dtype = preds.dtype
    kernel = _gaussian_kernel(channel, kernel_size, sigma, dtype, device)
    pad_h = (kernel_size[0] - 1) // 2
    pad_w = (kernel_size[1] - 1) // 2

    preds = F.pad(preds, (pad_h, pad_h, pad_w, pad_w), mode="reflect")
    target = F.pad(target, (pad_h, pad_h, pad_w, pad_w), mode="reflect")

    input_list = torch.cat((preds, target, preds * preds, target * target, preds * target))  # (5 * B, C, H, W)
    outputs = F.conv2d(input_list, kernel, groups=channel)
    output_list = outputs.split(preds.shape[0])

    mu_pred_sq = output_list[0].pow(2)
    mu_target_sq = output_list[1].pow(2)
    mu_pred_target = output_list[0] * output_list[1]

    sigma_pred_sq = output_list[2] - mu_pred_sq
    sigma_target_sq = output_list[3] - mu_target_sq
    sigma_pred_target = output_list[4] - mu_pred_target

    upper = 2 * sigma_pred_target + c2
    lower = sigma_pred_sq + sigma_target_sq + c2

    ssim_idx = ((2 * mu_pred_target + c1) * upper) / ((mu_pred_sq + mu_target_sq + c1) * lower)
    ssim_idx = ssim_idx[..., pad_h:-pad_h, pad_w:-pad_w]

    return ssim_idx.mean()


