import cv2
import torch
from fanjiang.losses import PSD

f1 = "data/radar_g.jpg"
f2 = "data/radar_f.jpg"

radar_g = cv2.imread(f1)
radar_f = cv2.imread(f2)

radar_g = torch.tensor(radar_g[None]).permute(0, 3, 1, 2).float()
radar_f = torch.tensor(radar_f[None]).permute(0, 3, 1, 2).float()

psd = PSD(size=radar_g.shape[-2:])

spectrum_g, spectrum_f =  psd.compute_spectrum(radar_g, radar_f)
spectrum_g = spectrum_g.cpu().flatten().numpy()
spectrum_f = spectrum_f.cpu().flatten().numpy()

psd.show(spectrum_g, spectrum_f)


