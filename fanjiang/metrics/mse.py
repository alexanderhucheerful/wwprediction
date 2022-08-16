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


@METRICS.register()
class MSE(DatasetEvaluator):
    def __init__(
            self,
            eval_names,
            lead_times,
            output_dir="",
        ):
        super().__init__()
        self.eval_names = eval_names
        self.lead_times = lead_times
        self.output_dir = output_dir

    def reset(self):
        self.idx = []

        self.mse = {}
        self.predictions = {}

        for t in self.lead_times:
            for name in self.eval_names:
                key = "{}_{}".format(name, t)
                self.mse[key] = []
                self.predictions[key] = []


    def process(self, outputs):
        idx = outputs["idx"].cpu()
        self.idx.append(idx)

        output = outputs["output"].cpu()
        target = outputs["target"].cpu()

        for i, t in enumerate(self.lead_times):
            for j, name in enumerate(self.eval_names):
                output_j = output[:, i, j]
                target_j = target[:, i, j]
                mse = F.mse_loss(target_j, output_j).sqrt()
                key = "{}_{}".format(name, t)
                self.mse[key].append(mse.item())
                self.predictions[key].append(output_j)


    def evaluate(self):
        comm.synchronize()

        for t in self.lead_times:
            for name in self.eval_names:
                key = "{}_{}".format(name, t)

                mse = comm.gather(self.mse[key], dst=0)
                self.mse[key] = list(itertools.chain(*mse))

                predictions = comm.gather(self.predictions[key], dst=0)
                self.predictions[key] = list(itertools.chain(*predictions))

        idx = comm.gather(self.idx, dst=0)
        idx = list(itertools.chain(*idx))

        if not comm.is_main_process():
            return

        csv_results = []
        for t in self.lead_times:
            result = [t]
            for name in self.eval_names:
                key = "{}_{}".format(name, t)
                mse = np.mean(self.mse[key])
                result.append(mse)
            csv_results.append(result)

        assert len(csv_results) == 1

        results = {}
        for i, key in enumerate(["Hours"] + list(self.eval_names)):
            results[key] = csv_results[0][i]

        if os.path.exists(self.output_dir):

            self.predictions["idx"] = idx
            save_f = os.path.join(self.output_dir, "predictions.pth")

            with PathManager.open(save_f, "wb") as f:
                torch.save(self.predictions, f)

        return results

