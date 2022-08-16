import logging
import os
import random
import numpy as np
import torch
import zarr
from fanjiang.builder import DATASETS
from torch.utils.data import Dataset
from torchvision import transforms as T

from .transform import AugmentationList

@DATASETS.register()
class ERA5(Dataset):
    def __init__(
            self,
            data_lr,
            data_hr="",
            src_step=6,
            dst_step=6,
            interval=1,
            training=True,
            superres=False,
            input_frames=6,
            future_frames=20,
            eval_index=[],
            augmentations=[],
            datetimes=['19790101', '20151231'],
        ):

        self.data = zarr.open(data_lr, 'r')

        if superres:
            self.data_hr = zarr.open(data_hr,'r')

        self.training = training
        self.superres = superres
        self.src_step = src_step
        self.dst_step = dst_step
        self.step = dst_step // src_step

        self.interval = interval

        self.input_frames = input_frames * self.step
        self.future_frames = future_frames * self.step
        self.total_frames = self.input_frames + self.future_frames

        self.augmentations = AugmentationList(augmentations, training)

        times = self.data.attrs['times']
        self.times = np.array([t for t in times if (t[:8] >= datetimes[0] and t[:8] <= datetimes[1])])

        self.init_times = self.times[slice(self.input_frames - self.step, -self.future_frames)]
        self.num_seq = len(self.init_times[::interval])

        logger = logging.getLogger(__name__)
        stage = "train" if self.training else "test"
        logger.info(f"num_seq: {self.num_seq}, {stage}_time: {self.times[0]} ~ {self.times[-1]} -> {self.init_times[0]} ~ {self.init_times[-1]}")


    def __len__(self):
        return self.num_seq

    def _rand_another(self):
        return np.random.randint(self.num_seq)

    def __getitem__(self, idx):
        while True:
            data = self.prepare_data(idx)
            if self.training and data is None:
                idx = self._rand_another()
                #self.logger.info(f"{idx} not exist")
                continue

            return data

    def prepare_data(self, idx):
        ind = idx * self.interval

        if self.training and self.src_step == 1:
            ind += np.random.randint(0, self.interval)

        inds = np.arange(ind, ind + self.total_frames, self.step)
        times = self.times[inds] # 6 + 20

        init_time = self.init_times[ind]
        #print(init_time)

        if not self.training:
            assert times[self.input_frames-1] == init_time 

        imgs = [self.data[f'/{t}'] for t in times]
        imgs = np.array(imgs, dtype=np.float32)
        imgs = torch.as_tensor(imgs).flip(dims=(2,))

        imgs = self.augmentations(imgs)
        shift = self.augmentations.params.get("shift", 0)

        if self.training:
            tid = random.randint(0, self.future_frames-1)
        else:
            tid = self.future_frames - 1

        idx = torch.as_tensor(idx)
        tid = torch.as_tensor(tid)
        shift = torch.as_tensor(shift)

        data = dict(
            idx=idx,
            tid=tid,
            shift=shift,
            imgs=imgs,
        )

        if self.superres:
            imgs_hr = [self.data_hr[f'/{times[tid + self.input_frames]}']]
            imgs_hr = np.array(imgs_hr, dtype=np.float32)
            imgs_hr = torch.as_tensor(imgs_hr)
            data["imgs_hr"] = imgs_hr

        return data
