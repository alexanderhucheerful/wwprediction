import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from matplotlib.colors import BoundaryNorm, ListedColormap

matplotlib.rcParams['font.sans-serif'] = "Times New Roman"
matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rcParams['font.weight'] = "bold"

prov_shp = shpreader.Reader('data/resource/shapefiles/Province.shp')
# city_shp = shpreader.Reader('data/resource/City_9/City_9.shp')
city_shp = shpreader.Reader('data/resource/gadm36/gadm36_CHN_1.shp')

level_cc = [0,0.1,0.3,0.5,0.6,0.7,0.8,0.85,0.9,0.92,0.94,0.96,0.97,0.98,0.99]
level_ref = [-32768, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
level_zdr =[-4,-3,-2,-1,0,0.2,0.5,0.8,1,1.5,2,2.5,3,3.5,4,5]
level_kdp = [-0.8,-0.4,-0.2,-0.1,0.1,0.15,0.22,0.33,0.5,0.75,1.1,1.7,2.4,3.1,7,20]

cmap_cc = ListedColormap(
    [
        "b","aqua","darkturquoise", "darkgreen","green","seagreen",
        "limegreen", "lime","yellow","khaki","gold","orange",
        "red","firebrick","darkred","magenta"
    ]
)
cmap_ref = ListedColormap(
    [
        '#FFFFFF',"dodgerblue",
        "#01F508", "#00A433","green","yellow","#FFDC01",
        "orange", "red","firebrick",
        "darkred","magenta","darkmagenta"
    ]
)
cmap_zdr = ListedColormap(
    [
        "black","dimgray","gray", "darkgray","mediumslateblue","green",
        "limegreen", "lime","yellow","khaki","gold","orange",
        "red","firebrick","darkred","magenta"
    ]
)

cmap_kdp = ListedColormap(
    [
        "cyan","aqua","darkturquoise", "darkgray","darkgray","green",
        "limegreen", "lime","yellow","khaki","gold","orange",
        "red","firebrick","darkred","magenta"
    ]
)

norm_ref = BoundaryNorm(level_ref, ncolors=cmap_ref.N, clip=True)
norm_zdr = BoundaryNorm(level_zdr, ncolors=cmap_zdr.N, clip=True)
norm_cc = BoundaryNorm(level_cc, ncolors=cmap_cc.N, clip=True)
norm_kdp = BoundaryNorm(level_kdp, ncolors=cmap_kdp.N, clip=True)

norm_dict = {"cc": norm_cc, "ref": norm_ref, "zdr": norm_zdr, "kdp": norm_kdp}
cmap_dict = {"cc": cmap_cc, "ref": cmap_ref, "zdr": cmap_zdr, "kdp": cmap_kdp}
level_dict = {"cc": level_cc, "ref": level_ref, "zdr": level_zdr, "kdp": level_kdp}


def get_time(radar_f, postfix="_qc.nc"):
    name = os.path.basename(radar_f)
    name = name.rstrip(postfix)
    time = name[-12:]
    return time


def plot_single(img, region, title, save_f, vname="ref"):
    img_h, img_w = img.shape[:2]
    lat_max, lat_min, lon_min, lon_max = region 
    lat1d = np.linspace(lat_min + 0.01, lat_max, img_h)
    lon1d = np.linspace(lon_min + 0.01, lon_max, img_w)
    grid_lat, grid_lon = np.meshgrid(lat1d, lon1d, indexing="ij")

    projection = ccrs.PlateCarree()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=projection)
    ax.add_feature(cfeature.LAKES.with_scale('50m'), zorder=4, alpha=0.3)
    ax.add_feature(cfeature.RIVERS.with_scale('50m'), zorder=5, alpha=0.5)

    ax.add_feature(cfeature.ShapelyFeature(prov_shp.geometries(), projection, \
                                                edgecolor='k', facecolor='none'),
                        linewidth=2.5, linestyle='-', zorder=4, alpha=0.8)

    ax.add_feature(cfeature.ShapelyFeature(city_shp.geometries(), projection, \
                                                edgecolor='k', facecolor='none'),
                        linewidth=1., linestyle='-', zorder=4, alpha=0.8)
            

    ax.set_extent([lon_min, lon_max, lat_min, lat_max], projection)
        
        
    ax.gridlines(crs=ccrs.PlateCarree(),
        draw_labels=False,
        linewidth=1.2,
        color='k',
        alpha=0.5,
        linestyle='--',
        zorder=8)

    gci = ax.pcolormesh(grid_lon, grid_lat, img, cmap=cmap_dict[vname], norm=norm_dict[vname], zorder=3, alpha=1)

    ax.set_xticks(np.arange(np.round(lon_min,0), np.round(lon_max,0), 1), crs=projection)
    ax.set_xticklabels(np.arange(np.round(lon_min,0), np.round(lon_max,0), 1), fontsize=22)
    ax.set_yticks(np.arange(np.round(lat_min+0.5,0), np.round(lat_max+0.5,0), 2), crs=projection)
    ax.set_yticklabels(np.arange(np.round(lat_min+0.5,0), np.round(lat_max+0.5,0), 2), fontsize=22)

    lon_formatter = LongitudeFormatter()
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    cb = plt.colorbar(gci, shrink = 0.9)

    cb.set_ticks(level_dict[vname])
    cb.ax.set_yticklabels(level_dict[vname], fontsize=22)
    cb.set_label(f"{vname}", fontsize=28)
    
    plt.title(title, fontsize=25)
    plt.savefig(save_f, bbox_inches='tight', pad_inches=0.0, dpi=200)    
    plt.close()
