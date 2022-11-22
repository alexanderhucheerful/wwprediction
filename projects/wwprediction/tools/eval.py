import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from fanjiang.utils import visual_seq
from fanjiang.utils.logger import setup_logger

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default="")
parser.add_argument('--pred_names', type=str, nargs='+')
parser.add_argument('--interval', type=int, default=1)
parser.add_argument('--total_frames', type=int, default=156)
parser.add_argument('--eval_times', type=int, nargs='+', default=np.arange(0, 20))
parser.add_argument('--eval_names', type=str, nargs='+', default=["z", "t"])
args = parser.parse_args()

logger = setup_logger(name=__name__)


def compute_weighted_rmse(da_fc, da_true, mean_dims=xr.ALL_DIMS):
    """
    Compute the RMSE with latitude weighting from two xr.DataArrays.

    Args:
        da_fc (xr.DataArray): Forecast. Time coordinate must be validation time.
        da_true (xr.DataArray): Truth.
        mean_dims: dimensions over which to average score
    Returns:
        rmse: Latitude weighted root mean squared error
    """
    error = da_fc - da_true
    weights_lat = np.cos(np.deg2rad(error.lat))
    weights_lat /= weights_lat.mean()
    rmse = np.sqrt(((error)**2 * weights_lat).mean(mean_dims))
    return rmse


def compute_weighted_acc(da_fc, da_true, mean_dims=xr.ALL_DIMS):
    """
    Compute the ACC with latitude weighting from two xr.DataArrays.
    WARNING: Does not work if datasets contain NaNs

    Args:
        da_fc (xr.DataArray): Forecast. Time coordinate must be validation time.
        da_true (xr.DataArray): Truth.
        mean_dims: dimensions over which to average score
    Returns:
        acc: Latitude weighted acc
    """

    clim = da_true.mean('time')
    try:
        t = np.intersect1d(da_fc.time, da_true.time)
        fa = da_fc.sel(time=t) - clim
    except AttributeError:
        t = da_true.time.values
        fa = da_fc - clim
    a = da_true.sel(time=t) - clim

    weights_lat = np.cos(np.deg2rad(da_fc.lat))
    weights_lat /= weights_lat.mean()
    w = weights_lat

    fa_prime = fa - fa.mean()
    a_prime = a - a.mean()

    acc = (
            np.sum(w * fa_prime * a_prime) /
            np.sqrt(
                np.sum(w * fa_prime ** 2) * np.sum(w * a_prime ** 2)
            )
    )
    return acc

colors = {
    'Persistence': '0.2',
    'Climatology': '0.5',
    'Weekly clim.': '0.7',
    'IFS': '#984ea3',
    'IFS T42': '#4daf4a',
    'IFS T63': '#377eb8',
    'LR (iterative)': '#ff7f00',
    'LR (direct)': '#ff7f00',
    'CNN (iterative)': '#e41a1c',
    'CNN (direct)': '#e41a1c',
}


def plot_metric(score, name, exp, save_f, ylabel=None, title=None, ylim=None):
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    if name in score:
        score[name].plot(c=colors[exp], label=exp, lw=3, ax=ax)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(ylim)
    ax.set_xlim(0, 122)
    ax.set_xticks([0, 24, 48, 72, 96, 120])
    ax.set_xticklabels([0, 1, 2, 3, 4, 5])
    ax.set_xlabel('Forecast time [days]')
    plt.tight_layout()
    fig.savefig(save_f, bbox_inches='tight')


def load_data(path, var, years=slice('2017', '2018')):
    ds = xr.open_mfdataset(f'{path}/*.nc', combine='by_coords')[var]
    if var in ['z', 't']:
        try:
            ds = ds.sel(level=500 if var == 'z' else 850).drop('level')
        except ValueError:
            ds = ds.drop('level')
    return ds.sel(time=years)

def load_predicition(pred_f, valid):
    valid = valid.isel(time=slice(args.total_frames-1, None))
    valid_time=valid.time,
    lat=valid.lat,
    lon=valid.lon

    logger.info(
        f"num: {len(valid_time)}, time: {valid_time.values[0]} ~ {valid_time.values[-1]}"
    )
    predictions = torch.load(pred_f)
    idx = np.concatenate(predictions["idx"]) // args.interval
    logger.info(f"min: {idx.min()} max: {idx.max()} l1: {len(idx)} l2: {len(valid_time)}")

    data = []
    for name in args.eval_names:
        data_t = []
        for time in args.eval_times:
            key = "{}_{}".format(time, name)
            pred = np.concatenate(predictions[key])[idx]
            lead_time = [(time + 1) * args.interval] * len(idx)
            pred = xr.DataArray(
                pred,
                dims=['time', 'lead_time', 'lat', 'lon'],
                coords={'time': valid_time, 'lead_time':lead_time,'lat': lat, 'lon': lon},
                name=name
            )
            data_t.append(pred)
        data_t = xr.concat(data_t, "lead_time")
        data.append(data_t)

    return xr.merge(data)


def eval_direct(pred, valid):
    rmse = compute_weighted_rmse(pred, valid).load()
    logger.info(f"rmse_z500: {rmse['z'].values}")
    logger.info(f"rmse_t850: {rmse['t'].values}")
    acc = compute_weighted_acc(pred, valid).load()
    logger.info(f"acc_z500: {acc['z'].values}")
    logger.info(f"acc_t850: {acc['t'].values}")

def eval_iterative(pred, valid, func, interval):
    results = []
    for t in pred.lead_time:
        t = int(t)
        pred_t = pred.sel(lead_time=t)
        pred_t['time'] = pred_t.time + np.timedelta64(t, 'h')
        valid_t = valid.isel(time=slice(t, None, interval))
        result = func(pred_t, valid_t)
        results.append(result)
    results = xr.concat(results, "lead_time").load()
    return results

def eval_nc(pred_f, valid, func):
    pred = xr.open_dataset(pred_f)
    return eval_iterative(pred, valid, func, interval=12)

def eval_pth(pred_f, valid, func):
    pred = load_predicition(pred_f, valid)
    return eval_iterative(pred, valid, func, interval=args.interval)


def eval(valid):
    z500_valid = load_data(f'{args.data_dir}/geopotential_500', 'z')
    t850_valid = load_data(f'{args.data_dir}/temperature_850', 't')
    valid = xr.merge([z500_valid, t850_valid], compat='override')

    def eval_single(func):
        results = {}
        for pred_f in args.pred_names:
            if pred_f.endswith(".nc"):
                results["IFS"] = eval_nc(pred_f, valid, func)
            elif pred_f.endswith(".pth"):
                results["SwinRNN"] = eval_pth(pred_f, valid, func)
        return results

    results_mse = eval_single(compute_weighted_rmse)
    results_acc = eval_single(compute_weighted_acc)

    # plot_metric(mse, "z", ylabel=r"RMSE [m$^2$ s$^{-2}$]", title="Z500", exp="IFS", save_f="z500_rmse.pdf")
    # plot_metric(acc, "z", ylabel=r"ACC [m$^2$ s$^{-2}$]", title="Z500", exp="IFS", save_f="z500_acc.pdf")
    # plot_metric(mse, "t", ylabel=r"RMSE [K]", title="T850", exp="IFS", save_f="t850_rmse.pdf")
    # plot_metric(acc, "t", ylabel=r"ACC [K]", title="T850", exp="IFS", save_f="t850_acc.pdf")


eval()



