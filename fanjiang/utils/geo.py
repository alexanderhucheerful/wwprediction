import torch 

def latlon_to_xy(img, latlon, region):
    if torch.is_tensor(img):
        h, w = img.shape[-2:]
    else:
        h, w = img.shape[:2]
    x = int((latlon[1] - region[2]) * w / (region[3] - region[2]))
    y = int((latlon[0] - region[0]) * h / (region[1] - region[0]))
    return (x, y)

def xy_to_latlon(img, xy, region):
    if torch.is_tensor(img):
        h, w = img.shape[-2:]
    else:
        h, w = img.shape[:2]    
    h, w = img.shape[-2:]
    lon = xy[0] *  (region[3] - region[2]) / w + region[2]
    lat = xy[1] *  (region[1] - region[0]) / h + region[0]
    return (lat, lon)

def center_crop(region, size=896):
    nlat, slat, wlon, elon = region
    lat = (nlat + slat) / 2.0
    lon = (wlon + elon) / 2.0
    deg = size / 200.
    nlat = lat + deg
    slat = lat - deg
    wlon = lon - deg 
    elon = lon + deg
    return (nlat, slat, wlon, elon)