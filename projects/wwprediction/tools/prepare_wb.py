import argparse
import functools
import multiprocessing as mp
import os
import numpy as np
import xarray as xr
import pandas as pd
from fanjiang.utils.logger import setup_logger
logger = setup_logger(name=__name__)


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default="")
parser.add_argument('--save_dir', type=str, default="")
parser.add_argument('--names', type=str, nargs="+", default=["u_component_of_wind", "v_component_of_wind"])
parser.add_argument('--short_names', type=str, nargs="+", default=["u", "v"])
parser.add_argument('--years', type=str, nargs="+", default=["1979", "2018"])
args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)

def load_data(data_dir, names, short_names, years):
    data = []
    for name in names:
        field = xr.open_mfdataset(f'{data_dir}/{name}/*.nc', combine='by_coords')
        data.append(field)
    data = xr.merge(data, compat='override')
    data = data.sel(time=slice(*years))
    data = [data[name] for name in short_names]
    data = xr.concat(data, 'level').transpose('time', 'lat', 'lon', 'level')
    return data

data = load_data(args.data_dir, args.names, args.short_names, args.years)
data = data.rolling(time=6).sum()

# from IPython import embed; embed()
# exit(0)


# mean_f = os.path.join(args.save_dir, "mean.npy")
# if not os.path.exists(mean_f):
#     mean = data.mean(('time', 'lat', 'lon')).compute()
#     np.save(mean_f, mean.values)
#     logger.info(f"mean: {mean}")

# std_f = os.path.join(args.save_dir, "std.npy")
# if not os.path.exists(std_f):
#     std = data.std('time').mean(('lat', 'lon')).compute()
#     np.save(std_f, std.values)
#     logger.info(f"std: {std}")


def save_single(idx):
    if idx < 6:
        return
    d = data.isel(time=idx)
    time = pd.to_datetime(str(d.time.values)).strftime("%Y%m%d%H")
    save_name = f"{time}.npy"
    save_f = os.path.join(args.save_dir, save_name)
    np.save(save_f, d.values)

ids = np.arange(0, data.shape[0])
logger.info(f"num: {len(ids)}")

# save_single(ids[0])

pool = mp.Pool(16)
worker = functools.partial(
    save_single,
)
pool.map(worker, ids)



