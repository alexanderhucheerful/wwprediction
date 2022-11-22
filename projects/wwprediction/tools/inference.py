import argparse
import datetime as dt
import os

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from einops import rearrange
from fanjiang.cnn import UNet
from fanjiang.core.checkpoint import Checkpointer
from fanjiang.dataset import GfsReader, RadarReader, SwanReader
from fanjiang.utils import add_time, latlon_to_xy, strptime, visual_seq
from fanjiang.utils.logger import setup_logger
from fanjiang.utils.visualizer import color_radar
from netCDF4 import Dataset
from tqdm import tqdm
from fanjiang.utils.plot_radar import plot_single
from fanjiang.builder import build_metrics, build_model
from fanjiang.config import get_cfg
from fanjiang.core.evaluator import DatasetEvaluators
from tabulate import tabulate

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=str, default="")
parser.add_argument('--radar_dir', type=str, default="")
parser.add_argument('--gfs_dir', type=str, default="")
parser.add_argument('--save_dir', type=str, default="")

parser.add_argument('--init_time', type=str, default="")
parser.add_argument('--size', type=int, default=896)
parser.add_argument('--stride', type=int, default=256)
parser.add_argument('--border', type=int, default=0)
parser.add_argument('--region', type=float, nargs="+", default=[])

parser.add_argument('--input_frames', type=int, default=12)
parser.add_argument('--future_frames', type=int, default=36)
parser.add_argument('--field_channels', type=int, default=0)
parser.add_argument('--num_layers', type=int, default=5)
parser.add_argument('--norm', type=str, default="SyncBN")

args = parser.parse_args()
logger = setup_logger(name=__name__)

output_dir = os.path.join(args.save_dir, "output")
os.makedirs(output_dir, exist_ok=True)

target_dir = os.path.join(args.save_dir, "target")
os.makedirs(target_dir, exist_ok=True)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    return cfg


def save_netcdf(img, region, save_f):
    h, w = img.shape[:2]
    nlat, slat, wlon, elon = region

    rows = np.arange(nlat, slat, -0.01)
    cols = np.arange(wlon, elon, 0.01)

    assert h == len(rows), (h, len(rows))
    assert w == len(cols), (w, len(cols))

    rootgrp = Dataset(save_f, 'w', format='NETCDF4')
    rootgrp.createDimension('lat', size=h)
    rootgrp.createDimension('lon', size=w)

    latitudes = rootgrp.createVariable('lat','f4',('lat',), zlib=True)
    longitudes = rootgrp.createVariable('lon','f4',('lon',), zlib=True)
    latitudes.units = 'degrees north'
    longitudes.units = 'degrees east'
    latitudes[:] = rows[:]
    longitudes[:] = cols[:]

    data = rootgrp.createVariable("radar", 'f4', ('lat', 'lon'), zlib=True)
    data.units = "dbz"
    assert data.shape == img.shape, (data.shape, img.shape)
    data[:] = img[:]

    rootgrp.close()


def get_time(name):
    name = os.path.basename(name)[:-4]
    name = name.replace("_", "")
    time = name[-14:-2]
    return time

def expand_region(img, src_region, dst_region):
    img_h, img_w = img.shape[-2:]
    if args.size == img_w and args.size == img_h:
        x1 = y1 = 0
        x2 = img_w
        y2 = img_h
        region = dst_region
    else:
        test_size = args.size + args.stride
        nlat, slat, wlon, elon = dst_region
        lat = (nlat + slat) / 2.0
        lon = (wlon + elon) / 2.0
        deg = test_size / 200.
        nlat = lat + deg
        slat = lat - deg
        wlon = lon - deg
        elon = lon + deg
        x1, y1 = latlon_to_xy(img, (nlat, wlon), src_region)
        x2 = x1 + test_size
        y2 = y1 + test_size
        region = (nlat, slat, wlon, elon)

    bbox = (x1, y1, x2, y2)
    return bbox, region


def find_radar_names(radar_dir, pattern="0_{}.png"):
# def find_radar_names(radar_dir, pattern="Z_OTHE_RADAMCR_{}00.bin.BZ2"):
    names = []
    if args.init_time != "":
        t1 = add_time(args.init_time, hours=-1, minutes=6)
        # t2 = strptime(args.init_time)
        t2 = add_time(args.init_time, hours=3)
        times = pd.date_range(t1, t2, freq='6min')
        for i, time in enumerate(times):
            time = time.strftime("%Y%m%d%H%M")
            name = pattern.format(time)
            radar_f = os.path.join(radar_dir, name)
            assert os.path.exists(radar_f), (i, time, radar_f)
            names.append(radar_f)
    else:
        for f in os.listdir(radar_dir):
            names.append(os.path.join(radar_dir, f))
        names = sorted(names)
    return names


@torch.no_grad()
def slide_inference(model, radars, fields=None):
    border = args.border
    h_crop = w_crop = args.size
    h_stride = w_stride = args.stride
    batch, _, h_img, w_img = radars.size()
    channels = args.future_frames

    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
    outputs = radars.new_zeros((batch, channels, h_img, w_img))
    counts = radars.new_zeros((batch, 1, h_img, w_img))

    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)

            crop_radars = radars[:, :, y1:y2, x1:x2]

            if fields is None:
                with torch.no_grad():
                    crop_output = model.inference([crop_radars])
            else:
                field_size = int(args.size / 25 + 0.5)
                xx1 = x1 // 25
                yy1 = y1 // 25
                xx2 = xx1 + field_size
                yy2 = yy1 + field_size
                crop_fields = fields[:, :, yy1:yy2, xx1:xx2]
                assert crop_fields.shape[-2:] == (50, 50), (crop_fields.shape)

                with torch.no_grad():
                    crop_output = model.inference([crop_radars, crop_fields])

            output = crop_output.new_zeros(batch, channels, h_crop, w_crop)
            if border > 0:
                output[:, :, border:-border, border:-border] = crop_output
            else:
                output = crop_output

            outputs += F.pad(
                output,
                (int(x1), int(outputs.shape[3] - x2), int(y1), int(outputs.shape[2] - y2))
            )
            counts[:, :, y1+border:y2-border, x1+border:x2-border] += 1

    counts[counts == 0] = 1.
    output = outputs / counts
    return output



def inference_seq(reader, model, radar_names):
    input_frames = args.input_frames

    data = np.array([reader(name) for name in radar_names], dtype=np.float32)
    data = data / reader.std

    # visual_seq(radars * reader.std_inv, save_dir=os.path.join(args.save_dir, "inputs"))

    data = torch.tensor(data)
    target = data[input_frames:]
    radars = data[:input_frames]



    # init_time = add_time(init_time, hours=8)

    # bbox, region = expand_region(radars[0], args.region, reader.region)
    # x1, y1, x2, y2 = bbox
    # logger.info(f"input xywh: {x1} {y1} {x2 - x1} {y2 - y1}")
    # radars = radars[:, y1:y2, x1:x2]

    if torch.cuda.is_available():
        radars = radars.cuda()

    radars = rearrange(radars, 't h w -> 1 t h w')
    logger.info(f"radars: {radars.shape}, {radars.min()}, {radars.max()}")

    if args.field_channels > 0:
        gfs_dir = args.gfs_dir
        gfs_reader = GfsReader()

        gfs_names = []
        for f in os.listdir(gfs_dir):
            gfs_names.append(os.path.join(gfs_dir, f))
        gfs_names = sorted(gfs_names)

        # print(gfs_names)
        fields = np.array([gfs_reader(name) for name in gfs_names])
        # visual_seq(fields[:, 0], save_dir="debug/input/field")

        fields = torch.tensor(fields)
        if torch.cuda.is_available():
            fields = fields.cuda()

        fields = rearrange(fields, 't c p h w -> 1 (t c p) h w')
        fields = fields / 10
        logger.info(f"fields: {fields.shape}, {fields.min()}, {fields.max()}")

        outputs = slide_inference(model, radars, fields) # 1 x t x h x w
    else:
        outputs = slide_inference(model, radars) # 1 x t x h x w

    output = outputs[0].cpu()
    output = output * reader.std_inv
    target = target * reader.std_inv

    logger.info(f"output: {output.shape}, {output.min()}, {output.max()}")
    logger.info(f"target: {target.shape}, {target.min()}, {target.max()}")

    init_time = args.init_time

    for t, output_t in enumerate(output, 1):
        lead_time = t * 6
        title = f"{init_time} + {lead_time:03d}min"
        save_name = f"{init_time}_{t:02d}.png"
        save_f = os.path.join(output_dir, save_name)
        msg = "output: {:02d} {:.3f} {:.3f} {}".format(t, output_t.min(), output_t.max(), save_name)
        logger.info(msg)
        plot_single(output_t, args.region, title, save_f)

    for t, target_t in enumerate(target, 1):
        lead_time = t * 6
        title = f"{init_time} + {lead_time:03d}min"
        save_name = f"{init_time}_{t:02d}.png"
        save_f = os.path.join(target_dir, save_name)
        msg = "target: {:02d} {:.3f} {:.3f} {}".format(t, target_t.min(), target_t.max(), save_name)
        logger.info(msg)
        plot_single(target_t, args.region, title, save_f)


    results = {
        "output": output[None],
        "target": target[None],
    }
    return results



cfg = setup(args)
model = build_model(cfg.MODEL.GENERATOR)
# logger.info("Model:\n{}".format(model))

metrics = build_metrics(cfg.METRICS)
# logger.info("Metrics:\n{}".format(metrics))

evaluator = DatasetEvaluators(metrics)

if torch.cuda.is_available():
    model = model.cuda()

model.eval()
check_pointer = Checkpointer(model)
check_pointer.load(cfg.MODEL.WEIGHTS)


def main():
    radar_names = find_radar_names(args.radar_dir)
    logger.info(f"num radars: {len(radar_names)}")

    reader = RadarReader(radar_type="png")
    # reader = SwanReader()
    outputs = inference_seq(reader, model, radar_names)

    evaluator.reset()
    evaluator.process(outputs)
    results = evaluator.evaluate(save_dir=args.save_dir)
    results = dict(sorted(results.items()))

    results = tabulate(
        [results],
        headers="keys",
        tablefmt="grid",
        numalign="left",
        stralign="center",
    )
    print(results)


    # seqs = []
    # leads = [radar_names[0]]
    # for radar_name in radar_names[1:]:
    #     time = get_time(radar_name)
    #     init_time = get_time(leads[-1])

    #     if len(leads) == input_frames:
    #         seqs.append(leads)
    #         del leads[0]

    #     if add_time(init_time, minutes=10) == time:
    #         leads.append(radar_name)
    #     else:
    #         del leads[:]
    #         leads.append(radar_name)

    # logger.info(f"num seqs: {len(seqs)}")

    # for seq in tqdm(seqs):
    #     inference_seq(model, seq)


if __name__ == "__main__":
    main()

