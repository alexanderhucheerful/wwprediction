import json
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from fanjiang.builder import DATASETS
from torch.utils.data import Dataset
from einops import rearrange
import xarray as xr
import logging
import pandas as pd

logger = logging.getLogger(__name__)

@DATASETS.register()
class WeatherBench(Dataset):
    def __init__(
            self,
            training,
            data_dir,
            step=6,
            interval=6,
            input_times=36,
            future_times=120,
            years=['1979', '2015'],
            names=['geopotential_500', 'temperature_850'],
        ):
        
        ds = self.load_data(data_dir, names, years)
        # ds = sorted(ds)

        data = []
        for name in ['z', 't']:
            data.append(ds[name])
        data = xr.concat(data, 'level').transpose('time', 'lat', 'lon', 'level')

        self.data = data
        self.training = training
        self.data_dir = data_dir
        self.step = step
        self.interval = interval
        self.input_times = input_times
        self.total_times = input_times + future_times

        # self.cal_mean_std(data, data_dir + "/zt", True)
        self.cal_mean_std(data, data_dir + "/ztuvr_t2muv10tp", True)
        # self.cal_mean_std(data, data_dir + "/ztvopv", True)
        # self.cal_mean_std(data, data_dir + "/ztt2m", True)
        # self.cal_mean_std(data, data_dir + "/ztqr", True)

        self.const = np.load(os.path.join(data_dir, "constants", "const.npy"))

        self.times = data.isel(time=slice(0, -self.total_times, interval)).time.values
        self.init_times = self.data.isel(time=slice(self.input_times-1, None)).time.values
        self.num = len(self.times)

        logger.info(f"num: {self.num}, time: {self.times[0]} ~ {self.times[-1]}")


    def cal_mean_std(self, data, data_dir, from_file=False):
        mean_f = os.path.join(data_dir, 'mean.npy')
        if from_file and os.path.exists(mean_f):
            self.mean = np.load(mean_f)
        else:
            self.mean = data.mean(('time', 'lat', 'lon')).compute().values
            np.save(mean_f, self.mean)

        std_f = os.path.join(data_dir, 'std.npy')
        if from_file and os.path.exists(std_f):
            self.std = np.load(std_f)
        else:
            self.std = data.std('time').mean(('lat', 'lon')).compute().values
            np.save(std_f, self.std)


    def load_data(self, data_dir, names, years):
        data = []
        for name in names:
            field = xr.open_mfdataset(f'{data_dir}/{name}/*.nc', combine='by_coords')
            data.append(field)
        data = xr.merge(data, compat='override')
        data = data.sel(time=slice(*years))
        return data

    def __len__(self):
        return self.num

    def _rand_another(self):
        return np.random.randint(self.num)

    def __getitem__(self, idx):
        while True:
            data = self.prepare_data(idx)
            if self.training and data is None:
                idx = self._rand_another()
                continue
            return data

    def prepare_data(self, idx):
        if self.training:
            idx = idx * self.interval + np.random.randint(0, self.interval)
        else:
            idx = idx * self.interval

        inds = np.arange(idx, idx + self.total_times)
        inds = inds[self.step-1::self.step]
        times = self.data.isel(time=inds).time.values

        if not self.training:
            input_frames = self.input_times // self.step
            assert times[input_frames-1] == self.init_times[idx], (input_frames, times[input_frames-1], self.init_times[idx])

        imgs = []
        for time in times:
            time = pd.to_datetime(str(time)).strftime("%Y%m%d%H")
            img_name = "{}.npy".format(time)
            # img_f = os.path.join(self.data_dir, "zt", img_name)
            img_f = os.path.join(self.data_dir, "ztuvr_t2muv10tp", img_name)
            # img_f = os.path.join(self.data_dir, "ztvopv", img_name)
            # img_f = os.path.join(self.data_dir, "ztt2m", img_name)
            # img_f = os.path.join(self.data_dir, "ztqr", img_name)

            if not os.path.exists(img_f):
                logger.info(f"{img_f} does not exist")
                return None

            img = np.load(img_f).astype(np.float32)
            imgs.append(img)

        fields = np.stack(imgs)
        fields[:, :, :, :-1] = (fields[:, :, :, :-1] - self.mean) / self.std
        fields = torch.tensor(fields)
        fields = rearrange(fields, 't h w c -> t c h w')
        tp = torch.log(1 + fields[:, -1].clamp(min=0) / 1e-3)
        fields[:, -1] = tp

        # assert fields.max() < 30
        # print(fields.min(), fields.max())

        # if self.training and np.random.rand() > 0.5:
        #     shift_size = 4
        #     shift = np.random.randint(-shift_size, shift_size+1)
        #     fields = fields.roll(shift, dims=[-1])

        inputs = [fields]

        seq_info = {
            "input_names": ['fields'],
            "idx": idx,
            "mean": self.mean,
            "std": self.std,
            "const": self.const,
        }
        inputs.append(seq_info)
        return inputs

