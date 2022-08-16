import os
import torch 
import matplotlib.pyplot as plt


def plot_var(imgs, exp, init_time, lead_time, save_dir="log_images"):
    os.makedirs(save_dir, exist_ok=True)

    save_names = ["z500", "t850", "t2m", "u10", "v10", "tp"]
    cmaps = ["cividis", "RdYlBu_r", "RdYlBu_r", "bwr", "bwr", "jet"]

    var_names = [
        r'Z500 [m$^2$ s$^{-2}$]',
        r'T850 [K]',
        r'T2M [K]',
        r'U10 [m s$^{-1}$]',  
        r'V10 [m s$^{-1}$]',                    
        r'TP [mm]',
    ]

    if torch.is_tensor(imgs):
    	imgs = imgs.cpu().numpy()

    for i, name in enumerate(save_names):
        var_name = var_names[i]
        cmap = cmaps[i]
        img = imgs[i]

        title = f"{exp} {var_name} t={lead_time}h"
        save_f = os.path.join(save_dir, f"{exp}_{name}_{init_time}_{lead_time:03d}.png")

        _ = plt.figure()
        plt.axis("off")
        #plt.imshow(img) 
        plt.imshow(img, cmap=cmap)
        plt.title(title)
        plt.savefig(save_f, bbox_inches='tight', pad_inches=0.0, transparent='true', dpi=600)
        plt.close()  
