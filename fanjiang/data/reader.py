import bz2
import struct
from datetime import datetime

#import cv2
import numpy as np
import pandas as pd
import xarray as xr
from fanjiang.utils import add_time


def deg2uv(direction, speed):
    """
    将风向和风速转化为u，v， 风向是风的来向

    Converts direction and speed into u,v wind

    :param direction: wind direction (mathmatical angle, degree)
    :param speed: wind magnitude (m/s)
    :returns: u and v wind components (m/s)
    """
    u = speed * np.sin(np.pi * (direction + 180.) / 180.)
    v = speed * np.cos(np.pi * (direction + 180.) / 180.)
    return u, v


class GfsReader:
    def __init__(
            self, 
            names=("U component of wind", "V component of wind", "Relative humidity"), 
            region=(54.2, 12.2, 73, 135), 
            levels=(200, 250, 300, 400, 500, 600, 700, 850, 925, 1000),
        ):
        self.names = names 
        self.region = region 
        self.levels = levels

    def __call__(self, img_f):
        ds = pygrib.open(img_f)
        var_names = [d.name for d in ds]
        var_names = np.unique(var_names)
        top, bottom, left, right = self.region

        try:
            data = ds.select(name=self.names)
        except:
            return -1

        lat, lon = data[0].latlons()
        mask = (lat >= bottom) & (lat <= top) & (lon >= left) & (lon <= right)
        x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
        x1 = int(x)
        y1 = int(y)
        x2 = int(x + w)
        y2 = int(y + h)

        images = []
        for field in data:
            if field.typeOfLevel == "isobaricInhPa":
                if field.level in self.levels:
                    images.append(field.values[y1:y2, x1:x2]) # q u v 
                    # print(field.name, field.level)
        images = np.array(images, dtype=np.float32)

        q = images[::3]
        u = images[1::3]
        v = images[2::3]

        # print(q.shape, q.min(), q.max())
        # print(u.shape, u.min(), u.max())
        # print(v.shape, v.min(), v.max())

        images = np.stack([u, v, q], axis=0) # c x p x h x w
        images = np.flip(images, -2) # vertical flip !!!
        return images / 20

class RadarReader:
    def __init__(self, img_type="jpg", region=[]):
        self.img_type = img_type
        self.region = region

    def decode_img(self, src_f):
        img =  cv2.imread(src_f, -1)
        return img

    def decode_nc(self, src_f):
        data = xr.open_dataset(src_f)
        return data

    def decode_bin(self, src_f):
        with open(src_f, 'rb') as f:
            buf = f.read()
            f.close()

        slat = float(struct.unpack('i', buf[124:128])[0]) / 1000
        wlon = float(struct.unpack('i', buf[128:132])[0]) / 1000
        nlat = float(struct.unpack('i', buf[132:136])[0]) / 1000
        elon = float(struct.unpack('i', buf[136:140])[0]) / 1000

        timezone = struct.unpack('i', buf[96:100])[0]
        year = struct.unpack('h', buf[100:102])[0]
        month = struct.unpack('h', buf[102:104])[0]
        day = struct.unpack('h', buf[104:106])[0]
        hour = struct.unpack('h', buf[106:108])[0]
        minute = struct.unpack('h', buf[108:110])[0]
        time = "{:04d}{:02d}{:02d}{:02d}{:02d}".format(year, month, day, hour, minute)

        if timezone == 28800:
            time = add_time(time, hours=-8)

        self.time = time 
        cols = struct.unpack('i', buf[148:152])[0]
        rows = struct.unpack('i', buf[152:156])[0]
        compress_flag = struct.unpack('h', buf[166:168])[0]
        compress_data = buf[256:]

        self.shape = (rows, cols)
        self.region = (nlat, slat, wlon, elon)

        data = None
        if compress_flag == 1:
            data = bz2.decompress(compress_data)
        else:
            data = compress_data

        data = np.frombuffer(data, dtype=np.int16).reshape((rows, cols)).astype(np.float32)
        data = data * (data >= 0) 
        data = np.clip(data / 1000, 0, 1)
        return data


    def decode_swan(self, filename, compressed=True):
        if compressed:
            f = bz2.BZ2File(filename, 'rb')
        else:
            f = open(filename, 'rb')
        if f is None:
            return

        ZonName = struct.unpack('12s', f.read(12))
        DataName = struct.unpack('38s', f.read(38))
        vflag = struct.unpack('8s', f.read(8))
        version = struct.unpack('8s', f.read(8))
        year = struct.unpack('H', f.read(2))[0]
        month = struct.unpack('H', f.read(2))[0]
        day = struct.unpack('H', f.read(2))[0]
        hour = struct.unpack('H', f.read(2))[0]
        minute = struct.unpack('H', f.read(2))[0]
        interval = struct.unpack('H', f.read(2))[0]
        cols = struct.unpack('H', f.read(2))[0]
        rows = struct.unpack('H', f.read(2))[0]
        levels = struct.unpack('H', f.read(2))[0]
        num_radar = struct.unpack('i', f.read(4))[0]

        time = "{:04d}{:02d}{:02d}{:02d}{:02d}".format(
            year, month, day, hour, minute
        )
        self.time = time 

        wlon = struct.unpack('f', f.read(4))[0]
        nlat = struct.unpack('f', f.read(4))[0]
        clon = struct.unpack('f', f.read(4))[0]
        clat = struct.unpack('f', f.read(4))[0]
        xres = struct.unpack('f', f.read(4))[0]
        yres = struct.unpack('f', f.read(4))[0]
        
        slat = 2 * clat - nlat      
        elon = 2 * clon - wlon        
        self.region = (nlat, slat, wlon, elon)          

        self.heights = struct.unpack('40f', f.read(160))
        self.radar_name = struct.unpack('320s', f.read(320))
        self.longitude = struct.unpack('20f', f.read(80))
        self.latitude = struct.unpack('20f', f.read(80))
        self.altitude = struct.unpack('20f', f.read(80))
        
        self.mflag = struct.unpack('20s', f.read(20))
        self.data_type = struct.unpack('h', f.read(2))[0]
        self.dim = struct.unpack('h', f.read(2))[0]
        self.reserved = struct.unpack('168s', f.read(168))
        
        num_data = cols * rows * levels
        data = struct.unpack('%dB' % num_data, f.read(num_data))
        data = np.array(data, dtype=np.float32).reshape(rows, cols)
        data = np.clip((data - 66) / 2, 0, 70)
        return data / 70


    def __call__(self, src_f):
        if self.img_type == "nc":
            return self.decode_nc(src_f)
        elif self.img_type == "bin":
            return self.decode_bin(src_f)
        elif self.img_type == "swan":
            return self.decode_swan(src_f)
        return self.decode_img(src_f)
            

class WindReader:
    def __init__(self, region):
        self.region = region 
        self.names = [
            "station", "datetime", 
            "lat", "lon", "alt", 
            "direction", "speed", 
            "direction_avg", "speed_avg"
        ]

    def __call__(self, src_f):

        winds = pd.read_csv(
            src_f, 
            header=0,
            names=self.names,
            index_col="station",
            parse_dates=['datetime'], 
            date_parser=lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'),
            na_values=[999999, 999998, 999017])

        # winds.dropna(inplace=True)

        nan_mask = winds.direction_avg.isnull() | winds.speed_avg.isnull()        

        nlat, slat, wlon, elon = self.region
        region_mask = (winds["lat"] >= slat) & (winds["lat"] <= nlat) & (winds["lon"] >= wlon) & (winds["lon"] <= elon)

        valid = ~nan_mask & region_mask
        winds = winds[valid]
        return winds

