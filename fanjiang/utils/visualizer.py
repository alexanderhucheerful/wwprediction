import io
import json
import os

#import cv2
import matplotlib.figure as mplfigure
import matplotlib.pyplot as plt
import numpy as np
import torch
from fanjiang.utils import latlon_to_xy
from matplotlib.backends.backend_agg import FigureCanvasAgg
# from mpl_toolkits.basemap import Basemap
from PIL import Image
from torchvision.transforms import ToTensor

__all__ = ["VisImage"]


radar_cmap = {
    0: [0, 0, 246],
    5: [1, 160, 246],
    10: [0, 236, 236],
    15: [1, 255, 0],
    20: [0, 200, 0],
    25: [1, 144, 0],
    30: [255, 255, 0],
    35: [231, 192, 0],
    40: [255, 144, 0],
    45: [255, 0, 0],
    50: [214, 0, 0],
    55: [192, 0, 0],
    60: [255, 0, 240],
    65: [120, 0, 132],
    70: [173, 140, 240],
}

# de4 plot_to_image(fig):
#     fig.canvas.draw()
#     s, (width, height) = fig.canvas.print_to_buffer()
#     buffer = np.frombuffer(s, dtype="uint8")
#     img_rgba = buffer.reshape(height, width, 4)
#     rgb, alpha = np.split(img_rgba, [3], axis=2)
#     return rgb.astype("uint8")

def plot_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='jpg')
    # plt.close(fig)
    buf.seek(0)
    image = Image.open(buf)
    image = ToTensor()(image).unsqueeze(0)
    print(image.shape)
    return image


def color_radar(img, max_dbz=70):
    # img value is 0 ~ 1
    img = np.clip(img * max_dbz, 0, max_dbz)
    cmap = [(i, color[::-1]) for i, color in radar_cmap.items()]
    color_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for i in range(len(cmap)):
        mask = img >= cmap[i][0]
        if i + 1 < len(cmap):
            mask &= img < cmap[i + 1][0]
        color_img += np.expand_dims(mask, axis=2) * np.expand_dims(cmap[i][1], axis=0).astype(np.uint8)
    return color_img


def overlay_station(img, region, station_f, radius=100):
    stations = json.load(open(station_f))
    for name in stations:
        lon, lat = stations[name]
        x, y = latlon_to_xy(img, (lat, lon), region)
        cv2.circle(img, (x, y), radius, (255, 255, 255), -1)
    return img

def overlay_map(img, region, map_f, save_f):
    nlat, slat, wlon, elon = region
    lats = np.arange(slat, nlat, 0.01)
    lons = np.arange(wlon, elon, 0.01)
    xs, ys = np.meshgrid(lons, lats)
    # fig = plt.figure(figsize=(8, 6))
    map = Basemap(
        llcrnrlon=lons[0], llcrnrlat=lats[0], urcrnrlon=lons[-1], urcrnrlat=lats[-1]
    )
    plt.contourf(xs, ys, img[::-1, :])
    map.readshapefile(map_f, 'states', drawbounds=True)
    plt.savefig(save_f, bbox_inches='tight', pad_inches=0.0, dpi=300)
    return img


def draw_points(img, coords=[]):
    img = color_radar(img)
    for coord in coords:
        cv2.circle(img, tuple(coord), 4, (0, 0, 0), -1)
    return img


def visual_seq(seq, save_dir, coords=[]):
    os.makedirs(save_dir, exist_ok=True)
    num_levels = seq.shape[1] if seq.ndim == 4 else 1 

    if torch.is_tensor(seq):
        seq = seq.numpy()

    for t, img in enumerate(seq):
        save_name = "{:02d}.jpg".format(t)
        save_f = os.path.join(save_dir, save_name)        

        if num_levels == 1:
            img = draw_points(img, coords)
            cv2.imwrite(save_f, img)
        else:
            fig, ax = plt.subplots(num_levels)
            for lvl in range(num_levels):
                ax[lvl].axis("off")
                ax[lvl].imshow(img[lvl])
            plt.savefig(save_f, bbox_inches='tight', pad_inches=0.0, dpi=300)    
            plt.close()



class VisImage:
    def __init__(self, height, width, scale=1.0):
        self.scale = scale
        self.width = width
        self.height = height
        self._setup_figure()

    def _setup_figure(self):
        """
        Args:
            Same as in :meth:`__init__()`.

        Returns:
            fig (matplotlib.pyplot.figure): top level container for all the image plot elements.
            ax (matplotlib.pyplot.Axes): contains figure elements and sets the coordinate system.
        """
        fig = mplfigure.Figure(frameon=False)
        self.dpi = fig.get_dpi()
        # add a small 1e-2 to avoid precision lost due to matplotlib's truncation
        # (https://github.com/matplotlib/matplotlib/issues/15363)
        fig.set_size_inches(
            (self.width * self.scale + 1e-2) / self.dpi,
            (self.height * self.scale + 1e-2) / self.dpi,
        )
        self.canvas = FigureCanvasAgg(fig)
        # self.canvas = mpl.backends.backend_cairo.FigureCanvasCairo(fig)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.axis("off")

        # Need to imshow this first so that other patches can be drawn on top
        # ax.imshow(img, extent=(0, self.width, self.height, 0), interpolation="nearest")

        self.fig = fig
        self.ax = ax


    def save(self, filepath):
        """
        Args:
            filepath (str): a string that contains the absolute path, including the file name, where
                the visualized image will be saved.
        """
        self.fig.savefig(filepath)
    
    def get_image(self):
        """
        Returns:
            ndarray:
                the visualized image of shape (H, W, 3) (RGB) in uint8 type.
                The shape is scaled w.r.t the input image using the given `scale` argument.
        """
        canvas = self.canvas
        s, (width, height) = canvas.print_to_buffer()
        # buf = io.BytesIO()  # works for cairo backend
        # canvas.print_rgba(buf)
        # width, height = self.width, self.height
        # s = buf.getvalue()

        buffer = np.frombuffer(s, dtype="uint8")

        img_rgba = buffer.reshape(height, width, 4)
        rgb, alpha = np.split(img_rgba, [3], axis=2)
        return rgb.astype("uint8")

