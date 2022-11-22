import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from fanjiang.utils import visual_seq
from fanjiang.utils.logger import setup_logger
import cartopy.crs as ccrs
from torchvision.utils import make_grid
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
# import fanjiang.utils.colormaps as cmap
from cartopy.util import add_cyclic_point
import numpy.ma as ma

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default="")
parser.add_argument('--save_dir', type=str, default="")
parser.add_argument('--pred_dir', type=str, default="")
parser.add_argument('--step', type=int, default=6)
parser.add_argument('--interval', type=int, default=1)
parser.add_argument('--lead_time', type=int, default=120)
parser.add_argument('--input_frames', type=int, default=6)
parser.add_argument('--future_frames', type=int, default=20)
parser.add_argument('--metrics', type=str, nargs='+', default=[])
parser.add_argument('--ensembles', type=str, nargs='+', default=[])
parser.add_argument('--eval_names', type=str, nargs='+', default=["z", "t"])

args = parser.parse_args()
logger = setup_logger(name=__name__)

os.makedirs(args.save_dir, exist_ok=True)

def compute_weighted_rmse(da_fc, da_true, mean_dims=xr.ALL_DIMS, eval_point=True, return_gt=False, write_gt=False):
    """
    Compute the RMSE with latitude weighting from two xr.DataArrays.

    Args:
        da_fc (xr.DataArray): Forecast. Time coordinate must be validation time.
        da_true (xr.DataArray): Truth.
        mean_dims: dimensions over which to average score
    Returns:
        rmse: Latitude weighted root mean squared error
    """

    if eval_point:
        point = [0.0, 53.4375]
        # point = [61.875, 14.0625]
        fc_point = da_fc.isel(time=-1).sel(lon=point[0], lat=point[1])

        if return_gt:
            da_true = da_true.isel(time=-1) - da_fc.isel(time=-1) * 0
            true_point = da_true.sel(lon=point[0], lat=point[1])

            if write_gt:
                title_t = r"T850 [K]"
                title_z = r"Z500 [m$^2$ s$^{-2}$]"

                save_f = os.path.join(args.save_dir, "sample_z500_gt.png")
                imcol(da_true.z, save_f, cmap="cividis", vmin=47000, vmax=58000, title=title_z, point=true_point.z)

                save_f = os.path.join(args.save_dir, "sample_t850_gt.png")
                imcol(da_true.t, save_f, cmap="RdYlBu_r", vmin=240, vmax=310, title=title_t, point=true_point.t)

            return true_point

        return fc_point


    error = da_fc - da_true
    weights_lat = np.cos(np.deg2rad(error.lat))
    weights_lat /= weights_lat.mean()
    rmse = np.sqrt(((error)**2 * weights_lat).mean(mean_dims))
    return rmse


def compute_weighted_ssr(da_fc, da_true, mean_dims=xr.ALL_DIMS):
    """
    Compute the RMSE with latitude weighting from two xr.DataArrays.

    Args:
        da_fc (xr.DataArray): Forecast. Time coordinate must be validation time.
        da_true (xr.DataArray): Truth.
        mean_dims: dimensions over which to average score
    Returns:
        rmse: Latitude weighted root mean squared error
    """

    rmse = compute_weighted_rmse(da_fc.mean("n_samples"), da_true, eval_point=False)
    fc_var = da_fc.var("n_samples")
    weights_lat = np.cos(np.deg2rad(fc_var.lat))
    weights_lat /= weights_lat.mean()
    spread = np.sqrt((fc_var * weights_lat).mean(mean_dims))
    return spread / rmse


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
    'T42': 'lightgray',
    'T63': 'darkgray',
    'IFS': 'gray',

    'Weekly clim.': 'black',

    # 'T42': 'lightskyblue',
    # 'T63': 'deepskyblue',
    # 'IFS': 'dodgerblue',

    # 'Naive CNN': 'lightgreen',
    # 'Cubed UNet': 'limegreen',
    # 'ResNet (pretrained)': 'gold',
    # 'FourCastNet': 'orange',

    'Naive CNN': 'lightgreen',
    'Cubed UNet': 'limegreen',
    'ResNet (pretrained)': 'lightseagreen',
    'FourCastNet': 'seagreen',

    'SwinRNN': 'indianred',
    'SwinVRNN': 'brown',
    'SwinVRNN*': 'maroon',

    'SwinRNN Dropout': 'blue',
    'SwinVRNN Feat': 'deepskyblue',
    'SwinVRNN Input': 'lightskyblue',
    'SwinVRNN Input Cov': 'blue',

    'ERA5': 'red',
}

def plot_metric(results, name, save_name, ylabel=None, title=None, ylim=None, lw=1):
    lead_time = args.lead_time
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))

    for exp, score in results.items():
        if name in score:
            if exp in ["Naive CNN", "Cubed UNet", "ResNet (pretrained)", "FourCastNet"]:
                ax.scatter(score['lead_time'], score[name], c=colors[exp], label=exp, lw=1, zorder=10)
            elif "samples" in exp:
                score[name].plot(c="steelblue", lw=0.1, ax=ax)
            elif exp in ["Weekly clim."]:
                ax.axhline(score[name][0], ls='--', c=colors[exp], label=exp, lw=lw)
            else:
                score[name].plot(c=colors[exp], label=exp, lw=lw, ax=ax)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(ylim)
    ax.set_xlim(0, lead_time + 2)
    ax.set_xticks(np.arange(0, lead_time+24, 24))
    ax.set_xticklabels(np.arange(0, lead_time//24+1))

    ax.set_xlabel('Forecast time [days]')
    # ax.legend(fontsize=8)
    plt.tight_layout()

    save_f = os.path.join(args.save_dir, save_name)
    fig.savefig(save_f, bbox_inches='tight',  pad_inches=0.0, transparent='true', dpi=300)
    plt.close()



def imcol(data, save_f, title='', point=None, **kwargs):
    if not 'vmin' in kwargs.keys():
        mx = np.abs(data.max().values)
        kwargs['vmin'] = -mx; kwargs['vmax'] = mx

    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

    I = data.plot(ax=ax, transform=ccrs.PlateCarree(), add_colorbar=False, add_labels=False,
                  rasterized=True, **kwargs)

    # cb = fig.colorbar(I, ax=ax, orientation='horizontal', pad=0.01, shrink=0.90)

    if point:
        ax.scatter(point.lon, point.lat, c=colors["ERA5"], edgecolors='k', lw=1, zorder=10)

    ax.set_title(title)
    ax.coastlines(alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_f, bbox_inches='tight', pad_inches=0.0, transparent='true', dpi=300)
    plt.close()


def plot_grid(img_dir, save_f, nrow=5, padding=4):
    imgs = []
    imgs_h = []

    paths = find_all(img_dir, postfix=".png")

    for i, img_f in enumerate(paths):
        img = cv2.imread(img_f, -1)

        if i == 0:
            height = (img.shape[0] // padding) * padding
            width = (img.shape[1] // padding) * padding

        img = cv2.resize(img, (width, height))

        row = i // nrow
        col = i % nrow

        if col < nrow - 1:
            img = cv2.copyMakeBorder(img, 0, 0, 0, padding, 0)
            imgs_h.append(img)
        else:
            img = cv2.hconcat(imgs_h)
            img = cv2.copyMakeBorder(img, 0, padding, 0, 0, 0)
            imgs.append(img)
            del imgs_h[:]

    img = cv2.vconcat(imgs)
    save_f = save_f.replace("*", "_plus")
    cv2.imwrite(save_f, img)


def load_data(path, var, years=slice('2017', '2018')):
    ds = xr.open_mfdataset(f'{path}/*.nc', combine='by_coords')[var]
    if var in ['z', 't']:
        try:
            ds = ds.sel(level=500 if var == 'z' else 850).drop('level')
        except ValueError:
            ds = ds.drop('level')
    return ds.sel(time=years)

def load_predicition(pred_f, valid):

    predictions = torch.load(pred_f)
    idx = np.concatenate(predictions["idx"])
    valid = valid.isel(time=idx)

    lat=valid.lat
    lon=valid.lon
    valid_time=valid.time
    idx = idx // args.interval

    pred_name = os.path.basename(pred_f)
    logger.info(f" {pred_name} min: {idx.min()} max: {idx.max()} l1: {len(idx)} l2: {len(valid_time)}")

    data = []
    for name in args.eval_names:
        data_t = []
        for t in np.arange(args.future_frames):
            key = "{}_{}".format(name, t)
            pred = np.concatenate(predictions[key])[idx]
            lead_time = [(t + 1) * args.step]
            pred = xr.DataArray(
                pred[:, None],
                dims=['time', 'lead_time', 'lat', 'lon'],
                coords={'time': valid_time, 'lead_time':lead_time, 'lat': lat, 'lon': lon},
                name=name
            )
            data_t.append(pred)
        data_t = xr.concat(data_t, "lead_time")
        data.append(data_t)

    return xr.merge(data)


def eval_iterative(pred, valid, func, return_gt=False):
    lead_time = args.lead_time

    results = []
    for t in pred.lead_time:
        t = int(t)

        if t > lead_time:
            break

        pred_t = pred.sel(lead_time=t)
        pred_t['time'] = pred_t.time + np.timedelta64(t, 'h')
        pred_t = pred_t.where(pred_t['time.year'] < 2019, drop=True)
        valid_t = valid.sel(time=pred_t.time.values)

        num_data = min(len(pred_t.time), len(valid_t.time))
        idx = np.arange(num_data)
        pred_t = pred_t.isel(time=idx)
        valid_t = valid_t.isel(time=idx)

        assert pred_t.time.values[0] == valid_t.time.values[0]
        assert pred_t.time.values[-1] == valid_t.time.values[-1]

        if return_gt:
            result = func(pred_t, valid_t, return_gt=True, write_gt=t==6)
        else:
            result = func(pred_t, valid_t)

        results.append(result)

    results = xr.concat(results, "lead_time").load()
    return results

def eval_nc(pred_f, valid, func):
    pred = xr.open_dataset(pred_f)
    if "tp" in pred:
        pred['tp'] *= 1000
    return eval_iterative(pred, valid, func)

def eval_pth(pred_f, valid, func):
    input_times = args.input_frames * args.step
    valid = valid.isel(time=slice(input_times - 1, None))
    pred = load_predicition(pred_f, valid)
    return eval_iterative(pred, valid, func)


def plot_example(pred, valid, init_time, method):
    # input_times = args.input_frames * args.step
    # valid = valid.isel(time=slice(input_times - 1, None))
    # pred = load_predicition(pred_f, valid)

    t_range = [(-7.5, 7.5), (-10, 10), (-20, 20)]
    tp_range = [(-7.5, 7.5), (-10, 10), (-20, 20)]
    z_range = [(-1000, 1000), (-2000, 2000), (-4000, 4000)]

    z_name = r'Z500 [m$^2$ s$^{-2}$]'
    t_name = 'T850 [K]'
    t2m_name = 'T2M [K]'
    tp_name = 'TP [mm]'
    u10_name = 'U10 [m s$^{-1}$]'
    v10_name = 'V10 [m s$^{-1}$]'

    for row, lead_time in enumerate([3, 5, 14]):
        lead_time = int(lead_time * 24)
        pred_t = pred.sel(lead_time=lead_time)
        pred_t['time'] = pred_t.time + np.timedelta64(lead_time, 'h')
        forecast_time = init_time + np.timedelta64(lead_time, 'h')
        pred_t = pred_t.sel(time=forecast_time)
        valid_t = valid.sel(time=forecast_time)
        t = np.datetime_as_string(init_time, unit="h")

        # z500
        save_f = os.path.join(args.save_dir, "{}_{}_{}_{}.png".format("ERA5", "Z500", t, lead_time))
        imcol(valid_t.z, save_f, cmap="cividis", vmin=47000, vmax=58000, title=f"ERA5 {z_name} t={lead_time}h")
        save_f = os.path.join(args.save_dir, "{}_{}_{}_{}.png".format(method, "Z500", t, lead_time))
        imcol(pred_t.z, save_f, cmap="cividis", vmin=47000, vmax=58000, title=f"{method} {z_name} t={lead_time}h")
        vmin, vmax = z_range[row]
        save_f = os.path.join(args.save_dir, "Error_{}_{}_{}_{}.png".format(method, "Z500", t, lead_time))
        imcol(pred_t.z - valid_t.z, save_f, cmap="BrBG", vmin=vmin, vmax=vmax, title=f"Error {method} {z_name} t={lead_time}h")

        # t850
        save_f = os.path.join(args.save_dir, "{}_{}_{}_{}.png".format("ERA5", "T850", t, lead_time))
        imcol(valid_t.t, save_f, cmap="RdYlBu_r", vmin=240, vmax=310, title=f"ERA5 {t_name} t={lead_time}h")
        save_f = os.path.join(args.save_dir, "{}_{}_{}_{}.png".format(method, "T850", t, lead_time))
        imcol(pred_t.t, save_f, cmap="RdYlBu_r", vmin=240, vmax=310, title=f"{method} {t_name} t={lead_time}h")
        vmin, vmax = t_range[row]
        save_f = os.path.join(args.save_dir, "Error_{}_{}_{}_{}.png".format(method, "T850", t, lead_time))
        imcol(pred_t.t - valid_t.t, save_f, cmap="BrBG", vmin=vmin, vmax=vmax,  title=f"Error {method} {t_name} t={lead_time}h")

        # t2m
        save_f = os.path.join(args.save_dir, "{}_{}_{}_{}.png".format("ERA5", "T2M", t, lead_time))
        imcol(valid_t.t2m, save_f, cmap="RdYlBu_r", vmin=240, vmax=310, title=f"ERA5 {t2m_name} t={lead_time}h")
        save_f = os.path.join(args.save_dir, "{}_{}_{}_{}.png".format(method, "T2M", t, lead_time))
        imcol(pred_t.t2m, save_f, cmap="RdYlBu_r", vmin=240, vmax=310, title=f"{method} {t2m_name} t={lead_time}h")
        vmin, vmax = t_range[row]
        save_f = os.path.join(args.save_dir, "Error_{}_{}_{}_{}.png".format(method, "T2M", t, lead_time))
        imcol(pred_t.t2m - valid_t.t2m, save_f, cmap="BrBG", vmin=vmin, vmax=vmax, title=f"Error {method} {t2m_name} t={lead_time}h")

        # tp
        save_f = os.path.join(args.save_dir, "{}_{}_{}_{}.png".format("ERA5", "TP", t, lead_time))
        imcol(valid_t.tp.clip(min=0), save_f, cmap="jet", vmin=-10, vmax=30, title=f"ERA5 {tp_name} t={lead_time}h")
        save_f = os.path.join(args.save_dir, "{}_{}_{}_{}.png".format(method, "TP", t, lead_time))
        imcol(pred_t.tp.clip(min=0), save_f, cmap="jet", vmin=-10, vmax=30, title=f"{method} {tp_name} t={lead_time}h")
        save_f = os.path.join(args.save_dir, "Error_{}_{}_{}_{}.png".format(method, "TP", t, lead_time))
        vmin, vmax = tp_range[row]
        imcol(pred_t.tp - valid_t.tp, save_f, cmap="BrBG", vmin=vmin, vmax=vmax, title=f"Error {method} {tp_name} t={lead_time}h")

        # u10
        save_f = os.path.join(args.save_dir, "{}_{}_{}_{}.png".format("ERA5", "U10", t, lead_time))
        imcol(valid_t.u10, save_f, cmap="bwr", vmin=-30, vmax=30, title=f"ERA5 {u10_name} t={lead_time}h")
        save_f = os.path.join(args.save_dir, "{}_{}_{}_{}.png".format(method, "U10", t, lead_time))
        imcol(pred_t.u10, save_f, cmap="bwr", vmin=-30, vmax=30, title=f"{method} {u10_name} t={lead_time}h")
        save_f = os.path.join(args.save_dir, "Error_{}_{}_{}_{}.png".format(method, "U10", t, lead_time))
        imcol(pred_t.u10 - valid_t.u10, save_f, cmap="BrBG", vmin=-20, vmax=20, title=f"Error {method} {u10_name} t={lead_time}h")

        # v10
        save_f = os.path.join(args.save_dir, "{}_{}_{}_{}.png".format("ERA5", "V10", t, lead_time))
        imcol(valid_t.v10, save_f, cmap="bwr", vmin=-30, vmax=30, title=f"ERA5 {v10_name} t={lead_time}h")
        save_f = os.path.join(args.save_dir, "{}_{}_{}_{}.png".format(method, "V10", t, lead_time))
        imcol(pred_t.v10, save_f, cmap="bwr", vmin=-30, vmax=30, title=f"{method} {v10_name} t={lead_time}h")
        save_f = os.path.join(args.save_dir, "Error_{}_{}_{}_{}.png".format(method, "V10", t, lead_time))
        imcol(pred_t.v10 - valid_t.v10, save_f, cmap="BrBG", vmin=-20, vmax=20, title=f"Error {method} {v10_name} t={lead_time}h")



def eval():
    z500_valid = load_data(f'{args.data_dir}/geopotential_500', 'z')
    t850_valid = load_data(f'{args.data_dir}/temperature_850', 't')
    t2m_valid = load_data(f'{args.data_dir}/2m_temperature', 't2m')
    u10_valid = load_data(f'{args.data_dir}/10m_u_component_of_wind', 'u10')
    v10_valid = load_data(f'{args.data_dir}/10m_v_component_of_wind', 'v10')
    tp_valid = load_data(f'{args.data_dir}/total_precipitation', 'tp').rolling(time=6).sum() * 1000
    tp_valid.name = 'tp'
    # valid = xr.merge([z500_valid, t850_valid, t2m_valid, u10_valid, v10_valid], compat='override')
    valid = xr.merge([z500_valid, t850_valid, t2m_valid, u10_valid, v10_valid, tp_valid], compat='override')
    tp_valid = load_data(f'{args.data_dir}/total_precipitation', 'tp').rolling(time=6).sum() * 1000

    def eval_single(func, with_base=False, with_ifs=False, with_dl=False, with_gt=False):
        results = {}

        ifs_results = {
            "IFS": "tigge_5.625deg.nc",
            "T63": "t63_5.625deg.nc",
            "T42": "t42_5.625deg.nc",
        }

        base_results = {
            "Weekly clim.": {
                "z": ("lead_time", [816, 816]),
                "t": ("lead_time", [3.50, 3.50]),
                "t2m": ("lead_time", [3.19, 3.19]),
                "tp": ("lead_time", [2.32, 2.32])},
        }

        dl_results = {
            "Naive CNN": {
                "z": ("lead_time", [626, 757]),
                "t": ("lead_time", [2.87, 3.37])},

            "Cubed UNet": {
                "z": ("lead_time", [373, 611]),
                "t": ("lead_time", [1.98, 2.87])},

            "ResNet (pretrained)": {
                "z": ("lead_time", [268, 499]),
                "t": ("lead_time", [1.65, 2.41]),
                "t2m": ("lead_time", [1.48, 1.92]),
                "tp": ("lead_time", [2.23, 2.33])},

            "FourCastNet": {
                "z": ("lead_time", [240, 480]),
                "t": ("lead_time", [1.5, 2.5]),
                "t2m": ("lead_time", [1.5, 2.0]),
                "tp": ("lead_time", [2.2, 2.5])},
        }

        if with_base:
            for method, result in base_results.items():
                results[method] = xr.Dataset(
                    data_vars=result, coords={"lead_time": [72, 120]}
                )

        if with_ifs:
            for method, name in ifs_results.items():
                pred_f = os.path.join(args.pred_dir, name)
                assert os.path.exists(pred_f), pred_f
                results[method] = eval_nc(pred_f, valid, func)

        if with_dl:
            for method, result in dl_results.items():
                results[method] = xr.Dataset(
                    data_vars=result, coords={"lead_time": [72, 120]}
                )

        input_times = args.input_frames * args.step
        gt = valid.isel(time=slice(input_times - 1, None))

        for method in args.ensembles:
            preds = []
            pred_dir = os.path.join(args.pred_dir, method)

            for n, name in enumerate(sorted(os.listdir(pred_dir))):
                if name.endswith(".pth"):
                    pred_f = os.path.join(pred_dir, name)
                    assert os.path.exists(pred_f), pred_f
                    pred = load_predicition(pred_f, gt)
                    preds.append(pred)

                    # logger.info(f"{n} {name}")

                    if "ssr" not in args.metrics:
                        results[f"{method}_{n:02d}"] = eval_iterative(pred, gt, func)

            if "ssr" in args.metrics:
                pred = xr.concat(preds, "n_samples")
                results[f"{method}"] = eval_iterative(pred, gt, func)


        if with_gt:
            results[f"ERA5"] = eval_iterative(pred, gt, func, return_gt=True)

        our_results = {
            # "SwinRNN": "swinrnn_depth6.pth",
            # "SwinVRNN": "swinvrnn_depth5.pth",
            # "SwinVRNN*": "swinvrnn_plus.pth",
        }

        for method, name in our_results.items():
            pred_f = os.path.join(args.pred_dir, name)
            assert os.path.exists(pred_f), pred_f
            init_time = np.datetime64("2018-04-03T11:00")
            # init_time = np.datetime64("2018-10-04T11:00")
            # init_time = np.datetime64("2018-09-10T11:00")
            pred = load_predicition(pred_f, gt)

            if method in ["SwinRNN", "SwinVRNN*"]:
                plot_example(pred, gt, init_time, method=method)

            results[method] = eval_iterative(pred, gt, func)


        return results

    if "mse" in args.metrics:
        if args.ensembles:
            results_mse = eval_single(
                compute_weighted_rmse,
                with_base=False,
                with_ifs=False,
                with_dl=False,
                with_gt=True,
            )

            for exp in results_mse:
                result = results_mse[exp]
                logger.info(f"{exp}")
                for name in result:
                    score = result[name].values[-1]
                    lead_time = result[name].lead_time.values[-1]
                    logger.info(f'lead_time: {lead_time}h, MSE_{name}: {score:.3f}')

            title_t = r"T850 [K]"
            title_z = r"Z500 [m$^2$ s$^{-2}$]"
            plot_metric(results_mse, "t", "sample_t850.png", ylim=(270, 288), title=title_t)
            plot_metric(results_mse, "z", "sample_z500.png", ylim=(51000, 58000), title=title_z)
        else:
            results_mse = eval_single(
                compute_weighted_rmse,
                with_base=False,
                with_ifs=False,
                with_dl=False,
            )

            for exp in results_mse:
                result = results_mse[exp]
                logger.info(f"{exp}")
                for name in result:
                    # print(exp, name, result[name])
                    score = result[name].values[-1]
                    lead_time = result[name].lead_time.values[-1]
                    logger.info(f'lead_time: {lead_time}h, MSE_{name}: {score:.3f}')

            plot_metric(results_mse, "t", "mse_t850.png", ylabel=r"RMSE [K]", title="T850")
            plot_metric(results_mse, "z", "mse_z500.png", ylabel=r"RMSE [m$^2$ s$^{-2}$]", title="Z500")
            plot_metric(results_mse, "t2m", "mse_t2m.png", ylabel=r"RMSE [K]", title="T2M")
            plot_metric(results_mse, "tp", "mse_tp.png", ylabel=r"RMSE [mm]", title="TP")
            plot_metric(results_mse, "u10", "mse_u10.png", ylabel=r"RMSE [m s$^{-1}$]", title="U10")
            plot_metric(results_mse, "v10", "mse_v10.png", ylabel=r"RMSE [m s$^{-1}$]", title="V10")


    if "ssr" in args.metrics:
        results_spread = eval_single(
            compute_weighted_ssr,
            with_base=False,
            with_ifs=False,
            with_dl=False,
            with_gt=False,
        )

        ch = -1
        for exp in results_spread:
            result = results_spread[exp]
            logger.info(f"{exp}")
            for name in result:
                score = result[name].values[ch]
                lead_time = result[name].lead_time.values[ch]
                logger.info(f'lead_time: {lead_time}h, SSR_{name}: {score:.3f}')



    if "acc" in args.metrics:
        results_acc = eval_single(compute_weighted_acc, with_base=False, with_ifs=True)

        for exp in results_acc:
            result = results_acc[exp]
            logger.info(f"{exp}")
            for name in result:
                # print(exp, name, result[name])
                score = result[name].values[-1]
                lead_time = result[name].lead_time.values[-1]
                logger.info(f'lead_time: {lead_time}h, ACC_{name}: {score:.3f}')


        plot_metric(results_acc, "t", "acc_t850.png", ylim=(0, 1), ylabel=r"ACC [K]", title="T850")
        plot_metric(results_acc, "z", "acc_z500.png", ylim=(0, 1), ylabel=r"ACC [m$^2$ s$^{-2}$]", title="Z500")
        plot_metric(results_acc, "t2m", "acc_t2m.png", ylim=(0, 1), ylabel=r"ACC [K]", title="T2M")
        plot_metric(results_acc, "tp", "acc_tp.png", ylim=(0, 1), ylabel=r"ACC [mm]", title="TP")
        plot_metric(results_acc, "u10", "acc_u10.png", ylim=(0, 1), ylabel=r"ACC [m s$^{-1}$]", title="U10")
        plot_metric(results_acc, "v10", "acc_v10.png", ylim=(0, 1), ylabel=r"ACC [m s$^{-1}$]", title="V10")


eval()



