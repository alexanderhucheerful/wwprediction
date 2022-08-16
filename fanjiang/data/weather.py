import json
import logging
import os

#import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import xarray as xr
import zarr
from einops import rearrange
from fanjiang.builder import DATASETS
from fanjiang.layers import to_2tuple
from fanjiang.utils import latlon_to_xy
from fanjiang.utils.visualizer import visual_seq
from torch.utils.data import Dataset


@DATASETS.register()
class Weather(Dataset):
    def __init__(
            self,
            ann_f,
            crop_size,
            training=True,
            total_frames=48,
        ):
        self.crop_size = to_2tuple(crop_size)
        self.training = training
        self.total_frames = total_frames

        data = self.load_annotations(ann_f)

        self.seqs = data['seqs']
        self.info = data['info']
        self.with_radar = "radar" in self.info
        self.with_gfs = "gfs" in self.info
        self.with_station = "stations" in self.info

        self.aug = create_transform()


    def __len__(self):
        return len(self.seqs)

    def center_crop(self, img):
        ref_h , ref_w = img.shape[-2:]
        x1 = (ref_w - self.crop_size[1]) // 2
        y1 = (ref_h - self.crop_size[0]) // 2
        x2 = x1 + self.crop_size[1]
        y2 = y1 + self.crop_size[0]
        return x1, y1, x2, y2


    def station_crop(self, imgs, region_id):
        img = imgs.mean(dim=(0, 1))

        if img.shape[-2:] == self.crop_size:
            return (0, 0) + self.crop_size

        def create_mask(img, region):
            coords = []
            masks = np.zeros(img.shape[-2:], dtype=np.uint8)
            for (lon, lat) in self.info["stations"].values():
                if (lat >= region[1] and lat <= region[0]
                    and lon >= region[2] and lon <= region[3]):
                    coord = latlon_to_xy(img, (lat, lon), region)
                    coords.append(coord)
                    cv2.circle(masks, coord, 100, (255), -1)
            coords = np.array(coords)
            masks = torch.tensor(masks, dtype=torch.bool)
            return masks, coords

        region = self.info["regions"][region_id]
        masks, coords = create_mask(img, region)

        if len(coords) == 0:
            return self.center_crop(img)

        xx = coords[:, 0]
        yy = coords[:, 1]
        vals = img[yy, xx].numpy()

        if vals.sum() == 0:
            return self.center_crop(img)

        if not self.training:
            idx = vals.argmax()
        else:
            prob = vals / vals.sum()
            idx = np.random.choice(len(prob), 1, p=prob).item()

        coord = coords[idx]
        x1 = max(0, coord[0] - self.crop_size[1] // 2)
        y1 = max(0, coord[1] - self.crop_size[0] // 2)

        ref_h , ref_w = img.shape[-2:]
        if x1 + self.crop_size[1] >= ref_w:
            x1 = ref_w - self.crop_size[1] - 1

        if y1 + self.crop_size[0] >= ref_h:
            y1 = ref_h - self.crop_size[0] - 1

        x2 = x1 + self.crop_size[1]
        y2 = y1 + self.crop_size[0]
        return x1, y1, x2, y2


    def region_crop(self, gfs, region_id):
        src_region = self.info["regions"]["A"]
        dst_region = self.info["regions"][region_id]
        x1, y1 = latlon_to_xy(gfs, (dst_region[0], dst_region[2]), src_region)
        x2, y2 = latlon_to_xy(gfs, (dst_region[1], dst_region[3]), src_region)
        return gfs[:, :, y1:y2, x1:x2]

    def load_annotations(self, ann_f):
        with open(ann_f) as f:
            data = json.load(f)
        return data

    def load_gfs(self, img_name):
        img_dir = self.info["gfs"]["img_dir"]
        img_f = os.path.join(img_dir, img_name)
        img = np.load(img_f) # c p h w
        return img

    def load_radar(self, img_name):
        img_dir = self.info["radar"]["img_dir"]
        img_h, img_w = self.info["radar"]["img_size"]
        img_f = os.path.join(img_dir, img_name)
        img = cv2.imread(img_f, -1)
        img = img.reshape(img_h, img_w, -1)
        return img


    def __getitem__(self, idx):
        seq = self.seqs[idx]
        names = seq["names"]
        region_id = seq["region_id"]

        inputs = []
        radar_imgs = []
        gfs_imgs = []
        input_names = []

        if self.with_radar:

            for img_name in names[:self.total_frames]:
                radar_img = self.load_radar(img_name)
                radar_imgs.append(radar_img)

            radar_imgs = torch.tensor(np.array(radar_imgs, dtype=np.float32))
            radar_imgs = rearrange(radar_imgs, 't h w c -> t c h w')

            if self.with_station:
                x1, y1, x2, y2 = self.station_crop(radar_imgs, region_id)
                radar_imgs = radar_imgs[:, :, y1:y2, x1:x2]

             # vis first channel of radar by uncomment these lines
            prefix = os.path.basename(names[0])[:-4]
            visual_seq(radar_imgs[:, 0] / 255, "results/input/radar/{}".format(prefix))

            inputs.append(radar_imgs)
            input_names.append("radar")


        if self.with_gfs:
            gfs_names = seq["gfs_names"]
            for gfs_name in gfs_names:
                gfs_img = self.load_gfs(gfs_name)
                gfs_imgs.append(gfs_img)
            gfs_imgs = torch.tensor(np.array(gfs_imgs, dtype=np.float32))
            gfs_imgs = rearrange(gfs_imgs, 't c p h w -> t (c p) h w')

            if self.with_radar:
                gfs_imgs = self.region_crop(gfs_imgs, region_id)
                resolution = self.info["gfs"]["resolution"]
                img_h = int(self.crop_size[0] / resolution + 0.5)
                img_w = int(self.crop_size[1] / resolution + 0.5)
                y1 = y1 // resolution
                x1 = x1 // resolution
                x2 = x1 + img_w
                y2 = y1 + img_h
                gfs_imgs = gfs_imgs[:, :, y1:y2, x1:x2]

                if gfs_imgs.shape[-2:] != (img_h, img_w):
                    gfs_imgs = F.interpolate(
                        gfs_imgs, size=(img_h, img_w), mode="bilinear", align_corners=False
                    )

                # vis 10 channels of gfs
                prefix = os.path.basename(gfs_names[0])[:-4]
                visual_seq(gfs_imgs[:, :10], "results/input/gfs/{}".format(prefix))


            inputs.append(gfs_imgs)
            input_names.append("gfs")

        seq_info = {
            "idx": idx,
            "input_names": input_names,
        }
        inputs.append(seq_info)
        return inputs


@DATASETS.register()
class WeatherBench(Dataset):
    def __init__(
            self,
            training,
            data_dir,
            norm_dir="",
            step=6,
            interval=1,
            input_frames=6,
            future_frames=20,
            years=['1979', '2016'],
            names=['geopotential_500', 'temperature_850'],
        ):

        ds = self.load_data(data_dir, names, years)
        data = [ds[name] for name in ds]
        data = xr.concat(data, 'level').transpose('time', 'lat', 'lon', 'level')

        self.data = data
        self.step = step
        self.training = training
        self.interval = interval
        self.data_dir = data_dir
        self.input_times = input_times
        self.total_times = input_times + future_times
        self.cal_mean_std(data, norm_dir)

        self.times = data.isel(time=slice(0, -(self.total_times-1), interval)).time.values
        self.init_times = self.data.isel(time=slice(self.input_times-1, None)).time.values

        self.num_seq = len(self.times)
        logger.info(f"num_seq: {self.num_seq}, init_times: {self.init_times[0]} ~ {self.init_times[-1]}")


    def cal_mean_std(self, data, norm_dir):
        mean_f = os.path.join(norm_dir, 'mean.npy')
        if os.path.exists(mean_f):
            self.mean = np.load(mean_f)
        else:
            self.mean = data.mean(('time', 'lat', 'lon')).compute().values
            np.save(mean_f, self.mean)

        std_f = os.path.join(norm_dir, 'std.npy')
        if os.path.exists(std_f):
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
        return self.num_seq

    def __getitem__(self, idx):
        if self.training:
            idx = idx * self.interval + np.random.randint(0, self.interval)
        else:
            idx = idx * self.interval

        inds = np.arange(idx, idx + self.total_times)
        inds = inds[self.step-1::self.step]
        times = self.data.isel(time=inds).time.values

        if not self.training:
            input_frames = self.input_times // self.step
            assert times[input_frames-1] == self.init_times[idx], (times[input_frames], self.init_times[idx])

        imgs = []
        for time in times:
            time = pd.to_datetime(str(time)).strftime("%Y%m%d%H")
            img_name = "{}.npy".format(time)
            img_f = os.path.join(self.data_dir, "ztuvr", img_name)
            img = np.load(img_f).astype(np.float32)
            imgs.append(img)

        imgs = np.stack(imgs)
        imgs = (imgs - self.mean) / self.std
        imgs = torch.tensor(imgs)
        imgs = rearrange(imgs, 't h w c -> t c h w')

        inputs = [imgs]
        seq_info = {
            "input_names": ['gfs'],
            "idx": idx,
            "std": self.std,
            "mean": self.mean,
        }
        inputs.append(seq_info)
        return inputs



@DATASETS.register()
class WeatherBenchZarr(Dataset):
    def __init__(
            self,
            training,
            data_dir,
            mean_f="",
            norm_f="",
            const_f="",
            src_step=6,
            dst_step=6,
            interval=6,
            window_size=8,
            input_frames=6,
            future_frames=20,
            years=['1979', '2015'],
        ):

        data_hr_dir = os.path.join(os.path.dirname(data_dir), "0.25deg")

        self.data = zarr.open(data_dir, 'r')
        self.data_hr = zarr.open(data_hr_dir, 'r')

        self.training = training
        self.data_dir = data_dir
        self.src_step = src_step
        self.dst_step = dst_step
        self.step = dst_step // src_step

        self.window_size = window_size
        self.interval = interval

        self.input_frames = input_frames * self.step
        self.future_frames = future_frames * self.step
        self.total_frames = self.input_frames + self.future_frames

        self.mean = np.load(os.path.join(os.path.dirname(data_dir), mean_f))
        self.norm = np.load(os.path.join(os.path.dirname(data_dir), norm_f))
        # self.const = np.load(os.path.join(os.path.dirname(data_dir), const_f))

        times = self.data.attrs['times']
        times_hr = self.data_hr.attrs['times']
        self.times = np.array([t for t in times if (t[:4] >= years[0] and t[:4] <= years[1])])
        self.times_hr = np.array([t for t in times_hr if (t[:4] >= years[0] and t[:4] <= years[1])])

        self.valid_times = self.times[slice(self.input_frames - self.step, -self.future_frames)]
        self.num_seq = len(self.valid_times[::interval])

        logger = logging.getLogger(__name__)
        stage = "train" if self.training else "test"
        logger.info(f"num_seq: {self.num_seq}, {stage}_time: {self.times[0]} ~ {self.times[-1]} -> {self.valid_times[0]} ~ {self.valid_times[-1]}")


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

    def normalize(self, x):
        mean = self.mean[:-1].reshape(1, -1, 1, 1)
        norm = self.norm[:-1].reshape(1, -1, 1, 1)
        x[:, :-1] = (x[:, :-1] - mean) / norm
        x = torch.tensor(x)
        tp = torch.log(1 + x[:, -1].clamp(min=0))
        x[:, -1] = tp
        return x

    def prepare_data(self, idx):
        idx = idx * self.interval

        if self.training and self.src_step == 1:
            idx += np.random.randint(0, self.interval)

        inds = np.arange(idx, idx + self.total_frames, self.step)
        times = self.times[inds] # 6 + 20

        if not self.training:
            input_frames = self.input_frames
            assert times[input_frames-1] == self.valid_times[idx], (input_frames, times[input_frames-1], self.valid_times[idx])
            # print(f"loading: [{idx:03d}/{self.num_seq}], init_time: {self.valid_times[idx]}, lead_time: {times[-1]}")

        imgs = [self.data[f'/{t}'] for t in times]
        imgs = np.array(imgs, dtype=np.float32)
        imgs = self.normalize(imgs).flip(dims=(2,))

        if self.training:
            tid = np.random.randint(self.future_frames)
        else:
            tid = self.future_frames - 1

        # print(self.training,  times[tid + self.input_frames], times[-1])

        imgs_hr = [self.data_hr[f'/{times[tid + self.input_frames]}']]
        imgs_hr = np.array(imgs_hr, dtype=np.float32)
        imgs_hr = self.normalize(imgs_hr)

        # if not self.training:
        #     import  matplotlib.pyplot as plt
        #     fig, ax = plt.subplots(6, 2)
        #     ax[0, 0].imshow(imgs[-1, 7])
        #     ax[0, 1].imshow(imgs_hr[-1, 7])
        #     ax[1, 0].imshow(imgs[-1, 23])
        #     ax[1, 1].imshow(imgs_hr[-1, 23])
        #     ax[2, 0].imshow(imgs[-1, 65])
        #     ax[2, 1].imshow(imgs_hr[-1, 65])
        #     ax[3, 0].imshow(imgs[-1, 66])
        #     ax[3, 1].imshow(imgs_hr[-1, 66])
        #     ax[4, 0].imshow(imgs[-1, 67])
        #     ax[4, 1].imshow(imgs_hr[-1, 67])
        #     plt.axis("off")
        #     plt.savefig("gt_pair.png", bbox_inches='tight', pad_inches=0.0, dpi=200)
        #     exit(0)

        assert torch.any(torch.isnan(imgs_hr)) == False, (imgs_hr.min(), imgs_hr.max())


        if self.training and np.random.rand() > 0.5:
            shift_size = 8
            shift = np.random.randint(-shift_size, shift_size+1)
            imgs = imgs.roll(shift, dims=[-1])
            imgs_hr = imgs_hr.roll(shift, dims=[-1])

        inputs = [imgs, imgs_hr]

        seq_info = {
            "input_names": ['imgs', 'imgs_hr'],
            "idx": idx,
            "tid": tid,
            "mean": self.mean,
            "norm": self.norm,
            # "const": self.const,
        }
        inputs.append(seq_info)
        return inputs

