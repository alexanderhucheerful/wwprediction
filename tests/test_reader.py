import argparse
import functools
import multiprocessing as mp
import os
from shutil import copyfile
import numpy as np 

from fanjiang.base import find_all
from fanjiang.dataset import field_reader, find_latest, read_gfs

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default="")
parser.add_argument('--save_dir', type=str, default="")
parser.add_argument('--postfix', type=str, default="")
args = parser.parse_args()


var_names = {
    'u': 'U component of wind',
    'v': 'V component of wind',
    'q': 'Relative humidity',
}
train_region = (54.2, 15, 95, 128)

src_f = "data/gfs.0p25.2021102500.f000.grib2"
save_dir = "data/debug"
os.makedirs(save_dir, exist_ok=True)

read_gfs(src_f, var_names['u'], train_region, save_dir)

# pool = mp.Pool(mp.cpu_count() // 2)
# worker = functools.partial(
#     read_gfs, 
#     name=var_names['u'], 
#     region=train_region, 
#     save_dir=save_dir
# )
# pool.map(worker, [src_f])
# gfs_names = open("data/gfs_name.txt")



