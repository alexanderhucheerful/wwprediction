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
    def __init__(self, interval, eval_index, eval_names, eval_frames, n_samples=20):
        super().__init__()
        self.interval = interval
        self.eval_index = eval_index
        self.eval_names = eval_names
        self.eval_frames = eval_frames
        self.n_samples = n_samples

    def reset(self):
        self.idx = []
        self.mse = {}
        self.predictions = {}

        for n in range(self.n_samples):
            for t in self.eval_frames:
                for name in self.eval_names:
                    key = "{}_{}_{}".format(n, name, t)
                    self.mse[key] = []
                    self.predictions[key] = []


    def process(self, outputs):
        idx = outputs["idx"]
        self.idx.append(idx)

        output = outputs["output"].cpu()
        target = outputs["target"].cpu()

        mean = outputs["mean"]
        std = outputs["std"]

        for n in range(self.n_samples):
            for t in self.eval_frames:
                for j, name in zip(self.eval_index, self.eval_names):
                    if name == "tp":
                        output_j = output[n, :, t, j].exp() - 1
                        target_j = target[:, t, j].exp() - 1
                    else:
                        output_j = output[n, :, t, j] * std[j] + mean[j]
                        target_j = target[:, t, j] * std[j] + mean[j]

                    mse = F.mse_loss(target_j, output_j).sqrt()
                    key = "{}_{}_{}".format(n, name, t)
                    self.mse[key].append(mse.item())
                    self.predictions[key].append(output_j)


    def evaluate(self, save_dir=None):
        comm.synchronize()

        for n in range(self.n_samples):
            for t in self.eval_frames:
                for name in self.eval_names:
                    key = "{}_{}_{}".format(n, name, t)
                    mse = comm.gather(self.mse[key], dst=0)
                    self.mse[key] = list(itertools.chain(*mse))
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
                key = "{}_{}_{}".format(0, name, t)
                mse = np.mean(self.mse[key])
                result.append(mse)
            csv_results.append(result)


        columns = ["Hours"]
        for name in self.eval_names:
            columns.append(f"MSE_{name}")

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

            for n in range(self.n_samples):
                predictions = {}
                for t in self.eval_frames:
                    for name in self.eval_names:
                        old_key = "{}_{}_{}".format(n, name, t)
                        new_key = "{}_{}".format(name, t)
                        predictions[new_key] = self.predictions[old_key]
                predictions["idx"] = idx
                save_f = os.path.join(save_dir, f"predictions_{n:02d}.pth")

                with PathManager.open(save_f, "wb") as f:
                    torch.save(predictions, f)


        log_first_n(
            logging.INFO,
            "MSE results:\n" + colored(csv_results, "cyan"),
            key="message",
        )
        return {}

