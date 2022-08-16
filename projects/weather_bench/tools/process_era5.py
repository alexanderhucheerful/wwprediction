import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import functools
import multiprocessing as mp
import os
from datetime import timedelta
import numpy as np
import xesmf as xe
import pandas as pd
import xarray as xr
import zarr
import json
from fanjiang.utils.logger import setup_logger
from numcodecs import Blosc
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--save_dir', type=str, required=True)
parser.add_argument('--deg', type=float, default=0)
parser.add_argument('--years', type=str, nargs="+", default=["19790101", "20181231"])
parser.add_argument('--roll_tp', action="store_true")

parser.add_argument('--src_step', type=int, default=6)
parser.add_argument('--dst_step', type=int, default=6)
parser.add_argument('--interval', type=int, default=8)
parser.add_argument('--lead_time', type=int, default=120)
parser.add_argument('--input_frames', type=int, default=6)
parser.add_argument('--future_frames', type=int, default=20)
parser.add_argument('--eid', type=int, nargs="+", default=[7, 23, 65, 66, 67, 68])

args = parser.parse_args()


logger = setup_logger(name=__name__)


pname_to_prefix = {
    "geopotential": "era5_geopotential",
    "temperature": "era5_temperature",
    "u_component_of_wind": "era5_u_component_of_wind",
    "v_component_of_wind": "era5_v_component_of_wind",
    "relative_humidity": "era5_relative_humidity",
    "2m_temperature": "era5_2mt",
    "10m_u_component_of_wind": "era5_2mt",
    "10m_v_component_of_wind": "era5_2mt",
    "total_precipitation_hourly_rolling": "era5_2mt"
}

pname_to_vname = {
    "geopotential": "z",
    "temperature": "t",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "relative_humidity": "r",
    "2m_temperature": "t2m",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "total_precipitation_hourly_rolling": "tp"
}


def get_info(date):
    for pname, prefix in pname_to_prefix.items():
        src_f = os.path.join(args.data_dir, pname, date[:4], f'{prefix}_{date}.nc')

        if not os.path.exists(src_f):
            logger.info(f"{src_f} not exist!!!")

        data_j = xr.open_mfdataset(src_f, combine='by_coords')

        if args.deg > 0:
            data_j = regrid(data_j, args.deg)

        print(data_j)
        longitude = data_j.longitude.values
        latitude = data_j.latitude.values
        level = data_j.level.values
        return level, latitude, longitude


def regrid(
        ds_in,
        ddeg_out,
        method='bilinear',
        reuse_weights=True
):
    """
    Regrid horizontally.
    :param ds_in: Input xarray dataset
    :param ddeg_out: Output resolution
    :param method: Regridding method
    :param reuse_weights: Reuse weights for regridding
    :return: ds_out: Regridded dataset
    """
    # Rename to ESMF compatible coordinates
    if 'latitude' in ds_in.coords:
        ds_in = ds_in.rename({'latitude': 'lat', 'longitude': 'lon'})

    # Create output grid
    grid_out = xr.Dataset(
        {
            'lat': (['lat'], np.arange(-90+ddeg_out/2, 90, ddeg_out)),
            'lon': (['lon'], np.arange(0, 360, ddeg_out)),
        }
    )

    # Create regridder
    regridder = xe.Regridder(
        ds_in, grid_out, method, periodic=True,
    )

    ds_out = regridder(ds_in)

    # Set attributes since they get lost during regridding

    ds_out.attrs.update(ds_in.attrs)

    if 'lat' in ds_out.coords:
        ds_out = ds_out.rename({'lat': 'latitude', 'lon': 'longitude'})

    return ds_out


def zarr16(time):
    try:
        x = data[f'/{time}'][:]
        x[:-1] = (x[:-1] - mean[:-1]) / std[:-1]
        x[-1] = np.log(1 + x[-1].clip(min=0))
        x = x.astype(np.float16)
        tensor = data_out.zeros(time, shape=x.shape, chunks=False, dtype='f2', compressor=Blosc(cname='zstd', clevel=5, shuffle=Blosc.SHUFFLE))
        tensor[:] = x

    except Exception as e:
        print('downsample error ', time, ' ', e)



def zarr_one_day(date):
    logger.info(f"{date}")

    data = []
    for pname, prefix in pname_to_prefix.items():
        src_f = os.path.join(args.data_dir, pname, date[:4], f'{prefix}_{date}.nc')

        # if pname == "total_precipitation":
        #     src_f = os.path.join("data/total_precipitation_sum", date[:4], f'{prefix}_{date}.nc')

        if not os.path.exists(src_f):
            logger.info(f"{src_f} not exist!!!")
            return []

        try:
            data_j = xr.open_mfdataset(src_f, combine='by_coords')
        except:
            logger.info(f"open {src_f} failed!!!")
            return []

        if "level" not in data_j.coords:
            level = xr.DataArray([1], coords={'level': [1]}, dims=['level'])
            data_j = data_j.expand_dims({'level': level}, axis=1)

        vname = pname_to_vname[pname]
        data.append(data_j[vname])

    data = xr.concat(data, "level")

    if args.deg > 0:
        data = regrid(data, args.deg)

    times = []
    for x in data:
        time = pd.to_datetime(str(x.time.values)).strftime("%Y%m%d%H")
        times.append(time)
        tensor = root.zeros(time, shape=x.shape, chunks=False, dtype='f4', compressor=Blosc(cname='zstd', clevel=5, shuffle=Blosc.SHUFFLE))
        tensor[:] = x.values
        # logger.info(f"{time} {x.shape}")

    return times


def roll_tp(inputs):
    prev_date, date = inputs

    pname = "total_precipitation_hourly"
    prefix = "era5_2mt"

    tp_f = os.path.join(args.data_dir, pname, date[:4], f'{prefix}_{date}.nc')
    prev_tp_f = os.path.join(args.data_dir, pname, prev_date[:4], f'{prefix}_{prev_date}.nc')

    if not os.path.exists(tp_f):
        logger.info(f"not exist {tp_f}")
        return []

    if not os.path.exists(prev_tp_f):
        logger.info(f"not exist {prev_tp_f}")
        prev_tp = None
    else:
        prev_tp = xr.open_mfdataset(prev_tp_f, combine='by_coords').tp

    tp = xr.open_mfdataset(tp_f, combine='by_coords').tp

    time = tp.time

    if prev_tp is not None:
        tp = xr.concat([prev_tp, tp], dim="time")

    tp = tp.rolling(time=6).sum() * 1000
    tp = tp.sel(time=time[::6]).fillna(0)

    save_dir = os.path.join(args.save_dir, date[:4])
    os.makedirs(save_dir, exist_ok=True)
    dst_f = os.path.join(save_dir, f'{prefix}_{date}.nc')
    tp.to_netcdf(dst_f)

    times = []
    for x in tp:
        time = pd.to_datetime(str(x.time.values)).strftime("%Y%m%d%H")
        times.append(time)

    return times


def prepare_gt():
    data_dir = args.data_dir
    years = args.years
    eid = args.eid
    interval = args.interval
    lead_time = args.lead_time
    step = args.dst_step // args.src_step
    input_frames = args.input_frames * step
    future_frames = args.future_frames * step

    data = zarr.open(data_dir, 'r')
    times = data.attrs['times']
    times = np.array([t for t in times if (t >= years[0] and t <= years[1])])

    init_times = times[slice(input_frames - step, -future_frames, interval)]
    init_times = [pd.to_datetime(t, format="%Y%m%d%H").to_numpy() for t in init_times]
    logger.info(f"num: {len(init_times)}, init_times: {init_times[0]} ~ {init_times[-1]}")

    # mean = np.load("data/mean.npy").reshape(-1, 1, 1)
    # norm = np.load("data/norm.npy").reshape(-1, 1, 1)
    # def normalize(x):
    #     x[-1] = np.log(1 + x[-1].clip(min=0))
    #     x[:-1] = (x[:-1] - mean[:-1]) / norm[:-1]
    #     return x

    imgs = []

    for i, t in enumerate(init_times):
        lt = t + np.timedelta64(lead_time, 'h')
        lt = pd.to_datetime(str(lt)).strftime("%Y%m%d%H")

        if lt > years[1]:
            logger.info(f"forecast time {lt} exceed validation range!!!")
            continue

        logger.info(f"loading: [{i:03d}/{len(init_times)}], init_time: {t}, lead_time: {lt}")
        img = np.array(data[f'/{lt}'], dtype=np.float32)[eid]
        # img = normalize(np.array(data[f'/{lt}'], dtype=np.float32))
        imgs.append(img)

    init_times = init_times[:len(imgs)]
    imgs = np.stack(imgs, axis=1)

    # min_val = imgs.min(axis=(1, 2, 3))
    # max_val = imgs.max(axis=(1, 2, 3))
    # np.save("min.npy", min_val)
    # np.save("max.npy", max_val):q
    # from IPython import embed; embed()

    latitudes = data.attrs["latitudes"]
    longitudes = data.attrs["longitudes"]

    gt = xr.Dataset(
        data_vars=dict(
           z500=(["time", "lat", "lon"], imgs[0]),
           t850=(["time", "lat", "lon"], imgs[1]),
           t2m=(["time", "lat", "lon"], imgs[2]),
           u10=(["time", "lat", "lon"], imgs[3]),
           v10=(["time", "lat", "lon"], imgs[4]),
           tp=(["time", "lat", "lon"], imgs[5]),
        ),
        coords=dict(
            time=init_times,
            lat=latitudes,
            lon=longitudes,
        ),
    )

    d = args.lead_time // 24
    save_f = os.path.join(args.save_dir, f"gt_{d:02d}d.nc")
    gt.to_netcdf(save_f)



root = zarr.open(args.save_dir, mode='a')

dates = []
# for date in pd.date_range('19790101', '19791231', freq="1d"):
for date in pd.date_range('19790104', '20181231', freq="1d"):

    prev_date = date + timedelta(days=-1)
    prev_date = prev_date.strftime("%Y%m%d")
    date = date.strftime("%Y%m%d")

    if args.roll_tp:
        dates.append((prev_date, date))
    else:
        dates.append(date)

logger.info(f"num_days: {len(dates)}")

prepare_gt()
exit(0)

levels, latitudes, longitudes = get_info(dates[0][1]) if args.roll_tp else get_info(dates[0])
logger.info(f"levels: {levels}")
logger.info(f"latitudes: {len(latitudes)}, {latitudes[0]} ~ {latitudes[-1]} ")
logger.info(f"longitudes: {len(longitudes)}, {longitudes[0]} ~ {longitudes[-1]} ")

# roll_tp(dates[10])
# zarr_one_day(dates[10])


pool = mp.Pool(16)
worker = functools.partial(
    roll_tp if args.roll_tp else zarr_one_day,
)
times = pool.map(worker, dates)
times = sorted(np.concatenate(times).tolist())
logger.info(f"times: {times[0]} ~ {times[-1]} {len(times)}")

root.attrs["times"] = times
root.attrs["variables"] = list(pname_to_vname.values())
root.attrs["levels"] = levels.tolist()
root.attrs["latitudes"] = latitudes.tolist()
root.attrs["longitudes"] = longitudes.tolist()
