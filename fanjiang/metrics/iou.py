import itertools
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fanjiang.builder import METRICS
from fanjiang.core import DatasetEvaluator
from fanjiang.utils import comm
from fanjiang.utils.file_io import PathManager
from fanjiang.utils.logger import log_first_n
from tabulate import tabulate
from termcolor import colored


@METRICS.register()
class IOU(DatasetEvaluator):
    """
        +-------------+-------------------------------+
        |             |           Observed            |
        |             +---------------+---------------+
        |             |      Y        |      N        |
        +---------+---+---------------+---------------+
        |         | Y |      A        |      B        |
        |Predicted+---+---------------+---------------+
        |         | N |      C        |      D        |
        +---------+---+---------------+---------------+
        A: hits (tp)
        B: false alarms (fp)
        C: misses (fn)
        D: correct rejections (tn)
        E: correct random forecasts 
    """
    def __init__(
            self, 
            eval_bins=[0],
            eval_times=[0],
            bin_weights=[1],
        ):
        super().__init__()
        self.eps = 1e-6
        self.eval_bins = eval_bins
        self.eval_times = eval_times
        self.bin_weights = bin_weights

        assert len(eval_bins) == len(bin_weights)


    def __repr__(self):
        return "TS, BIAS"

    def reset(self):
        self.tp = {}
        self.fp = {}
        self.fn = {}
        self.tn = {}

        for time in self.eval_times:
            for thresh in self.eval_bins:
                name = "{}_{}".format(time, thresh)
                self.tp[name] = []
                self.fp[name] = []
                self.tn[name] = []
                self.fn[name] = []


    def process(self, outputs):
        output = outputs["output"]
        target = outputs["target"]

        std = target.new_tensor(outputs["std"]).view(1, -1, 1, 1)
        mean = target.new_tensor(outputs["mean"]).view(1, -1, 1, 1)

        for tid, time in enumerate(self.eval_times):
            output_t = output[:, tid] * std + mean
            target_t = target[:, tid] * std + mean                    
            
            for bid, thresh in enumerate(self.eval_bins):
                if bid == len(self.eval_bins):
                    pred = output_t >= thresh 
                    gt = target_t >= thresh
                else:
                    thresh_high = self.eval_bins[bid+1]
                    pred = (output_t >= thresh) & (output_t < thresh_high)
                    gt = (target_t >= thresh) & (target_t < thresh_high)                    

                tp = ((pred == 1) & (gt == 1)).sum()
                fp = ((pred == 1) & (gt == 0)).sum()
                tn = ((pred == 0) & (gt == 0)).sum()
                fn = ((pred == 0) & (gt == 1)).sum()

                name = "{}_{}".format(time, thresh)

                self.tp[name].append(tp.cpu().item())
                self.fp[name].append(fp.cpu().item())
                self.tn[name].append(tn.cpu().item())
                self.fn[name].append(fn.cpu().item())


    def evaluate(self, save_dir=""):
        for time in self.eval_times:
            for thresh in self.eval_bins:
                name = "{}_{}".format(time, thresh)
                tp = comm.gather(self.tp[name], dst=0)
                fp = comm.gather(self.fp[name], dst=0)
                tn = comm.gather(self.tn[name], dst=0)
                fn = comm.gather(self.fn[name], dst=0)
                self.tp[name] = list(itertools.chain(*tp))
                self.fp[name] = list(itertools.chain(*fp))
                self.tn[name] = list(itertools.chain(*tn))
                self.fn[name] = list(itertools.chain(*fn))

        if not comm.is_main_process():
            return


        score = 0
        results = []

        for time in self.eval_times:
            result = [time]          
            for bid, thresh in enumerate(self.eval_bins):
                name = "{}_{}".format(time, thresh)

                tp = np.sum(self.tp[name]) # hists
                tn = np.sum(self.tn[name]) # correct rejections
                fp = np.sum(self.fp[name]) # false alarms
                fn = np.sum(self.fn[name]) # misses

                ts = tp / (tp + fn + fp + self.eps)
                bias = (tp + fp) / (tp + fn + self.eps)

                N = tp + fp + fn + tn 
                E = ((tp + fp) * (tp + fn) + (fp + tn) * (fn + tn)) / N
                hss = (tp + tn - E) / (N - E + self.eps)

                factor = (np.exp(-np.abs(1-bias))**0.2)
                score += ts *  factor * self.bin_weights[bid]
                # score += hss *  factor * self.bin_weights[bid]

                result.append(ts)
                result.append(hss)
                result.append(bias)
            
            results.append(result)

        score /= len(self.eval_times)

        columns = ["Minutes"]
        for thresh in self.eval_bins:
            columns.append("TS{}".format(thresh))
            columns.append("HSS{}".format(thresh))
            columns.append("BIAS{}".format(thresh))

        if save_dir:
            df = pd.DataFrame(results, columns=columns)
            def plot_metric(y, save_name):
                ax = df.plot(x=columns[0], y=y, kind="bar", width=0.8, figsize=(10, 6))            
                for p in ax.patches:
                    txt = "{:.3f}".format(p.get_height())
                    ax.annotate(txt, (p.get_x() * 1.005, p.get_height() * 1.005), rotation=45, size=5)
                save_f = os.path.join(save_dir, save_name)
                plt.savefig(save_f, bbox_inches='tight', pad_inches=0.0)     

            # plot_metric(columns[1::3], "ts.pdf")           
            # plot_metric(columns[3::3], "bias.pdf")  

        results = tabulate(
            results,
            headers=columns,
            tablefmt="grid",
            numalign="left",
            stralign="center",
        )

        if save_dir:
            save_f = os.path.join(save_dir, "results.txt")
            with PathManager.open(save_f, "w") as f:
                f.write(results)       
        
        log_first_n(
            logging.INFO,
            "Iou metrics:\n" + colored(results, "cyan"),
            key="message",
        )      

        return dict(ts=score)     
        