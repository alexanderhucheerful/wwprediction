import os 
import numpy as np

from omegaconf import OmegaConf
from fanjiang.config import LazyConfig
from fanjiang.config import LazyCall as L

from fanjiang.metrics import MSE, FID
from fanjiang.models import SwinRNN, SwinDecoder
from fanjiang.models.diffusion import DDPM, UnetSR
from fanjiang.models.transformer import SwinEncoder, SwinIR
from fanjiang.models.vae import GaussianPerturbation

from fanjiang.data import transform as T
from fanjiang.data import ERA5, build_test_loader, build_train_loader
from fanjiang.core import CosineParamScheduler, WarmupParamScheduler

members = 1 
eval_fid = False 
use_hist = False
superres = False 
perturbation = True 

lr = 0.0002
max_iter = 30000
teach_iters = 1000
warmup_iters = 2000
micro_batch_size = 8 
ccumulation_steps = 1
batch_size = micro_batch_size * ccumulation_steps

in_channels, embed_dim, patch_size, window_size, input_frames, future_frames = 69, 128, 2, 8, 6, 20 

dim_hidden = 128
image_size = (128, 256)
eval_index = (7, 23, 65, 66, 67, 68)
lead_time = future_frames * 6
# lead_times = (np.arange(0, future_frames) + 1) * 6

input_size = [sz // patch_size for sz in image_size]


exp_name = "swinrnn{}_unet{}_lr{}_itr{}_vrnn_d2".format(embed_dim, dim_hidden, lr, max_iter)
#exp_name = "swinrnn256_unet128_lr0.0005_itr30000_vrnn_tf" 
#exp_name = "swinrnn256_unet128_lr0.0005_itr30000_shuffle" 

work_dir = "/mnt/workspace"
data_dir = os.path.join(work_dir, "datasets/era5")
save_dir = os.path.join(work_dir, "output")

data_lr = os.path.join(data_dir, "1.40625deg")
data_hr = os.path.join(data_dir, "0.25deg.fp16") if superres else ""

output_dir = os.path.join(save_dir, exp_name)
checkpoint = os.path.join(save_dir, exp_name, "model_final.pth")

buffers = {
    "mean": np.load(os.path.join(data_dir, "mean_hr.npy")).reshape(-1, 1, 1),
    "std": np.load(os.path.join(data_dir, "std_hr.npy")).reshape(-1, 1, 1),
    "const": np.load(os.path.join(data_dir, "const_flip.npy")).reshape(-1, *image_size),
}

optimizer = LazyConfig.load("common/optim.py").AdamW
optimizer.lr = lr
optimizer.betas = (0.9, 0.95)
optimizer.weight_decay = 0.05

train = LazyConfig.load("common/train.py").train
train.max_iter = max_iter
train.accumulation_steps = ccumulation_steps
train.eval_period = 4000
train.checkpointer.period = 1000
train.output_dir = output_dir
#train.init_checkpoint = checkpoint

train.amp = True 
train.clip_grad.enabled = False 
train.clip_grad.max_norm = 1.0

train.ema.decay = 0.999
train.ema.warmup_iters = warmup_iters

train.teacher_forcing.enabled = False 
train.teacher_forcing.k = teach_iters

lr_multiplier = L(WarmupParamScheduler)(
    scheduler = L(CosineParamScheduler)(
        start_value=1, end_value=0
    ),
    warmup_length=warmup_iters / train.max_iter,
    warmup_factor=0.001,
)


model = L(SwinRNN)(
    in_channels=in_channels,
    channels=embed_dim,
    input_frames=input_frames,
    future_frames=future_frames,
    members=members,
    eval_index=eval_index,
    eval_fid=eval_fid,

    encoder=L(SwinEncoder)(
        in_channels=in_channels + 2,
        channels=embed_dim,
        input_frames=input_frames,
        image_size=image_size,
        patch_size=patch_size,
        window_size=window_size,
        depths=(6,),
        num_heads=(8,),
        out_indices=(0,),
        use_checkpoint=True,
    ),

    decoder = L(SwinDecoder)(
        in_channels=in_channels,
        channels=embed_dim,
        image_size=image_size,
        window_size=window_size,
        patch_size=patch_size,
        num_heads=8,
        depth=6,
    ),

    superres = L(DDPM)(
        N=500,
        sample_name="10",
        unet=L(UnetSR)(
            # dim_in=len(eval_index),
            dim_in=in_channels,
            dim=dim_hidden,
            patch_size=3,
            dim_emb=embed_dim,
        ),
    ) if superres else None,

    #superres = L(SwinIR)(
    #    img_size=image_size,
    #    in_chans=in_channels,
    #    upscale=6,
    #    embed_dim=128,
    #    window_size=8,
    #    mlp_ratio=2,
    #    depths=[6, 6, 6, 6],
    #    num_heads=[8, 8, 8, 8],
    #    upsampler="pixelshuffle"
    #) if superres else None,

    perturbation=L(GaussianPerturbation)(
        in_channels=in_channels,
        channels=embed_dim,
        embed_dim=embed_dim,
        image_size=image_size,
        patch_size=patch_size,
        window_size=window_size,
        depth=2,
    ) if perturbation else None,


    buffers=buffers,
)

dataloader = OmegaConf.create()

dataloader.train = L(build_train_loader)(
    dataset=L(ERA5)(
        data_lr=data_lr,
        data_hr=data_hr,
        interval=1,
        training=True,
        superres=superres,
        input_frames=input_frames,
        future_frames=future_frames,
        datetimes=('19790101', '20161231'),
        augmentations=[L(T.HRollTransform)(window=8)],
    ),
    micro_batch_size=micro_batch_size,
    num_workers=4,
    pin_memory=True,
)

dataloader.test = L(build_test_loader)(
    dataset=L(ERA5)(
        data_lr=data_lr,
        data_hr=data_hr,
        interval=32,
        training=False,
        superres=superres,
        input_frames=input_frames,
        future_frames=future_frames,
        datetimes=('20170101', '20181231'),
        #datetimes=('20181006', '20181012'),
    ),
    micro_batch_size=micro_batch_size,
    num_workers=4,
    pin_memory=True,
)


dataloader.evaluator = [
    L(MSE)(
        eval_names=("z500", "t850", "t2m", "u10", "v10", "tp"),
        lead_times=(lead_time,),
        output_dir=output_dir 
    ),
    #L(FID)(in_channels=len(eval_index)), 
]


