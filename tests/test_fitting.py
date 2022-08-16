import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange
from torch.utils.data import DataLoader, Dataset
from fanjiang.losses import LpLoss, HsLoss
from fanjiang.rnn import FNO2d, FNO3d, ConvGRU
import torch.nn.functional as F


def random_erasing(inputs, patch_size=16, masking_ratio=0.5):
    outputs = []
    for t in range(inputs.shape[1]):
        frame = inputs[:, t] # n x t x c x h x w

        patches = rearrange(
            frame, 'n c (h h2) (w w2) -> n (h w) c h2 w2', 
            h2=patch_size, w2=patch_size
        )
        batch, num_patches = patches.shape[:2]

        num_masked = int(masking_ratio * num_patches)

        rand_indices = torch.rand(batch, num_patches, device = device).argsort(dim = -1)
        masked_indices = rand_indices[:, :num_masked]
        batch_range = torch.arange(batch, device = device)[:, None]
        patches[batch_range, masked_indices] = 0

        h = w = frame.shape[-1] // patch_size
        frame = rearrange(patches,
            'n (h w) c h2 w2 -> n c (h h2) (w w2)', 
            h=h, w=w
        )

        # img = frame[0, 2].cpu().numpy()
        # plt.imshow(img)
        # plt.show()

        outputs.append(frame)
    return torch.stack(outputs, dim=1)

def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True, allow_unused=True)[0]
    return grad


class ImageFitting(Dataset):
    def __init__(self, img_dir, crop_size=128):
        super().__init__()
        self.crop_size = crop_size
        self.prepare_data(img_dir)

    def prepare_data(self, img_dir):
        images = []
        names = sorted(os.listdir(img_dir))

        for f in names:
            img = np.load(os.path.join(img_dir, f))
            images.append(img)

        images = torch.tensor(np.array(images))
        ref_h, ref_w = images.shape[-2:]

        x1 = (ref_w - self.crop_size) // 2
        y1 = (ref_h - self.crop_size) // 2     
        x2 = x1 + self.crop_size
        y2 = y1 + self.crop_size

        # t x c x p x h x w 
        crop_images = images[:, :, :, y1:y2, x1:x2]
        self.image_size = crop_images.shape[-2:]
        self.images = crop_images

    def __len__(self):
        return 1

    def __getitem__(self, idx):    
        if idx > 0: raise IndexError
        return self.images


model_name="fno"
channels = 52
level = 10
std = 1

device = torch.device("cpu")  
if torch.cuda.is_available():
    device = torch.device("cuda")

input_frames = 10
train_frames = 8
test_frames = 14

img_dir = "data/gfs_20200707"
save_dir = "debug"
os.makedirs(save_dir, exist_ok=True)

total_steps = 2000 # Since the whole image is our dataset, this just means 500 gradient descent steps.
steps_til_summary = 10
dataset = ImageFitting(img_dir)
image_size = dataset.image_size 

dataloader = DataLoader(
    dataset, 
    batch_size=1, 
    pin_memory=True, 
    num_workers=0
)

images = next(iter(dataloader))
images = images.to(device)
# n x t x c x p x h x w 
# inputs = images[:, :input_frames, :2, :].flatten(2, 3)
# targets = images[:, input_frames:, :2, :].flatten(2, 3)

inputs = images[:, :input_frames, :2, level]
targets = images[:, input_frames:, :2, level]

inputs = inputs / std
targets = targets / std

if model_name == "gru":
    model = ConvGRU(
        in_channels=input_frames * channels, 
        out_channels=channels, 
        hidden_channels=(64, 64, 64, 64),
    )
elif model_name == "fno":
    model = FNO2d(
        in_channels=input_frames * channels,
        out_channels=channels,
        hidden_channels=64,
        grid_channels=2,
        modes1=16, 
        modes2=16, 
    )



model = model.to(device)
optim = torch.optim.Adam(lr=1e-4, weight_decay=5e-4, params=model.parameters())


criterion = LpLoss(reduction="sum")
# criterion = HsLoss(reduction="sum")
# criterion = torch.nn.L1Loss()
metric = torch.nn.MSELoss()


def advection_equation(outputs, grids, t=-1):
    u, v = outputs[:, t].unbind(dim=1)

    du = gradient(u, grids)
    dv = gradient(v, grids)

    du_t, du_y, du_x = du.unbind(dim=2)
    dv_t, dv_y, dv_x = dv.unbind(dim=2)

    adv_u = du_t + u * du_x[:, t] + v * du_y[:, t]
    adv_v = dv_t + u * dv_x[:, t] + v * dv_y[:, t]
    loss = torch.norm(adv_u) + torch. norm(adv_v)
    return loss
    


model.train()
losses = []
for step in range(total_steps):
    outputs = model(inputs, train_frames)
    loss = criterion(outputs, targets[:, :train_frames])
    
    loss_adv = 0
    # loss_adv = advection_equation(outputs, grids)

    with torch.no_grad():
        error = metric(outputs * std, targets[:, :train_frames] * std)
        losses.append(error.item())

    if not step % steps_til_summary:
        print("Step %d, Total loss %0.6f adv loss: %0.6f error: %0.3f" % (step, loss, loss_adv, error))


    loss += loss_adv
    optim.zero_grad()
    loss.backward()
    optim.step()
    # break

losses = np.array(losses)
save_name = "{}_loss.npy".format(model_name)
save_f = os.path.join(save_dir, save_name)
np.save(save_f, losses)


model.eval()
with torch.no_grad():
    outputs = model(inputs, test_frames)
    error = metric(
        outputs[:, train_frames:] * std, targets[:, train_frames:] * std
    )
    print("error: {:.3f}".format(error.item()))
    assert outputs.shape == targets.shape, (outputs.shape, targets.shape)

outputs = outputs.squeeze(0).cpu().numpy() # t x c x h x w
targets = targets.squeeze(0).cpu().numpy()  # t x c x h x w


channels = outputs.shape[1]
for t in range(test_frames):
    fig, axes = plt.subplots(2, channels)

    for c in range(channels):
        if channels == 1:
            axes[0].axis("off")
            axes[1].axis("off")
            axes[0].imshow(targets[t, c])
            axes[1].imshow(outputs[t, c])
        else:            
            axes[0, c].axis("off")
            axes[1, c].axis("off")
            axes[0, c].imshow(targets[t, c])
            axes[1, c].imshow(outputs[t, c])

    if t < train_frames:
        save_name = "train_{:02d}h.jpg".format(t+1)
    else:
        save_name = "test_{:02d}h.jpg".format(t+1 - train_frames)

    save_f = os.path.join(save_dir, save_name)
    plt.savefig(save_f, bbox_inches='tight', pad_inches=0.0, dpi=600)
    plt.close()
