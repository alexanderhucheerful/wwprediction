import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from fanjiang.structure import cow_mask

noise = torch.randn(3, 1, 32, 32)
masks = cow_mask(noise, prop_range=(0, 0.2), sigma_range=(0.5, 1.5))
masks = F.interpolate(masks.float(), size=(896, 896), mode="bilinear", align_corners=True) < 0.5
# masks = masks.squeeze().numpy()
fig, ax = plt.subplots(1, 2)
ax[0].imshow(masks[0, 0].numpy(), cmap="gray")
ax[1].imshow(masks[1, 0].numpy(), cmap="gray")
ax[1].imshow(masks[2, 0].numpy(), cmap="gray")
plt.show()
