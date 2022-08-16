import itertools

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import torch
from einops.einops import rearrange
from fanjiang.builder import METRICS
from fanjiang.core import DatasetEvaluator
from fanjiang.utils import comm
from fanjiang.utils.events import get_event_storage
from fanjiang.utils.visualizer import VisImage, plot_to_image


@METRICS.register()
class PSD(DatasetEvaluator):
    def __init__(
            self, 
            image_size=128, 
            eval_time=-1, 
            beta=8.0,
            freq_thresh=0.1,
        ):
        super().__init__()
        self.eval_time = eval_time
        self.image_size = image_size
        self.freq_thresh = freq_thresh * image_size
        self.bins = np.arange(0.5, image_size // 2 + 1, 1.)
        self.centers = 0.5 * (self.bins[1:] + self.bins[:-1])

        xx = torch.fft.fftfreq(image_size) * image_size
        yy = torch.fft.fftfreq(image_size) * image_size
        grids = torch.meshgrid(xx, yy)
        self.knorm = torch.sqrt(grids[0]**2 + grids[1]**2)

        window = torch.kaiser_window(image_size, periodic=False, beta=beta)
        window *= window.square().sum().rsqrt()
        self.window = window.ger(window).unsqueeze(0).unsqueeze(1)

        if torch.cuda.is_available():
            self.window = self.window.cuda()

        
    def reset(self):
        self.real_spectrums = []
        self.fake_spectrums = []        
    
    def process(self, outputs):
        output = outputs["output"]
        target = outputs["target"]

        real_img = output[:, [self.eval_time]] 
        fake_img = target[:, [self.eval_time]]

        real_spectrum =  self.compute_spectrum(real_img).cpu()
        fake_spectrum =  self.compute_spectrum(fake_img).cpu()

        self.real_spectrums.append(real_spectrum)
        self.fake_spectrums.append(fake_spectrum)


    def evaluate(self):
        real_spectrums = comm.gather(self.real_spectrums, dst=0)
        fake_spectrums = comm.gather(self.fake_spectrums, dst=0)
        real_spectrums = list(itertools.chain(*real_spectrums))
        fake_spectrums = list(itertools.chain(*fake_spectrums))

        if not comm.is_main_process():
            return

        real_spectrum = torch.stack(real_spectrums).mean(0)
        fake_spectrum = torch.stack(fake_spectrums).mean(0)

        del real_spectrums
        del fake_spectrums

        storage = get_event_storage()
        psd_score, psd_img = self.draw_psd(real_spectrum, fake_spectrum)
        storage.put_image("PSD", psd_img)

        real_heat = self.draw_heatmap(real_spectrum)
        fake_heat = self.draw_heatmap(fake_spectrum)
        heat_img = np.concatenate((real_heat, fake_heat), axis=1)
        storage.put_image("Heatmap", heat_img.transpose(2, 0, 1))

        result = {"PSD": psd_score}
        return result


    def draw_heatmap(self, hmap):
        hmap = torch.log10(hmap) * 10
        hmap = torch.fft.fftshift(hmap)
        hmap = torch.cat([hmap, hmap[:1, :]], dim=0)
        hmap = torch.cat([hmap, hmap[:, :1]], dim=1)     
        hmap = hmap.cpu().numpy()

        vis_img = VisImage(self.image_size, self.image_size)
        freqs = np.linspace(-0.5, 0.5, num=hmap.shape[0], endpoint=True) * self.image_size
        ticks = np.linspace(freqs[0], freqs[-1], num=5, endpoint=True)
        levels = np.linspace(-40, 20, num=13, endpoint=True)

        vis_img.ax.set_xlim(ticks[0], ticks[-1])
        vis_img.ax.set_ylim(ticks[0], ticks[-1])
        vis_img.ax.set_xticks(ticks)
        vis_img.ax.set_yticks(ticks)

        vis_img.ax.contourf(freqs, freqs, hmap, levels=levels, extend='both', cmap='Blues')
        vis_img.fig.gca().set_aspect('equal')
        vis_img.ax.contour(freqs, freqs, hmap, levels=levels, extend='both', linestyles='solid', linewidths=1, colors='midnightblue', alpha=0.2)
        return vis_img.get_image()

    def draw_psd(self, real_spectrum, fake_spectrum):
        knorm = self.knorm.flatten().numpy()
        real_spectrum = real_spectrum.flatten().numpy()
        fake_spectrum = fake_spectrum.flatten().numpy()
        real_spectrum, _, _ = stats.binned_statistic(knorm, real_spectrum, statistic = "mean", bins = self.bins)
        fake_spectrum, _, _ = stats.binned_statistic(knorm, fake_spectrum, statistic = "mean", bins = self.bins)

        bins = self.bins[1:] > self.freq_thresh
        score = np.abs(1 - np.sum(fake_spectrum[bins]) / np.sum(real_spectrum[bins]))

        fig = plt.figure()
        plt.loglog(self.centers, real_spectrum, linewidth=2, label = "real")
        plt.loglog(self.centers, fake_spectrum, linewidth=2, label = "fake")
        plt.xlabel("$Frequency$", fontsize=18)
        plt.ylabel("$PSD$", fontsize=18)
        plt.legend()
        plt.tight_layout()
        # plt.close()
        return score, plot_to_image(fig)

    
    def compute_spectrum(self, image):
        spectrum = torch.fft.fftn(image * self.window, dim=(2,3)).abs().mean(dim=[0, 1])
        return spectrum 

