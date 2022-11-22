import argparse
import os
import numpy as np
import torch
from fanjiang.utils.logger import setup_logger

parser = argparse.ArgumentParser()
parser.add_argument('--save_f', type=str, default="")
parser.add_argument('--eval_names', type=str, nargs="+", default=["z", "t"])
parser.add_argument('--future_frames', type=int, default=20)

args = parser.parse_args()
logger = setup_logger(name=__name__)

save_dir=os.path.dirname(args.save_f)
os.makedirs(save_dir, exist_ok=True)


pred_list = [
    # "output/swin_decoder_fpn_head_4x/inference/predictions.pth",
    # "output/swinrnn_attn_hw_tf4000_depth5_4x/predictions.pth",

    # "output/swinrnn_ppn_depth5_4x/fpn_depth4_tta.pth",
    # "output/swinrnn_ppn_depth5_4x/ppn_depth4_tta.pth",
    # "output/swinrnn_ppn_depth5_4x/ppn_depth5_tta.pth",
    # "output/swinrnn_ppn_depth5_4x/ppn_depth5_tf_tta.pth",

    # "output/swinrnn_t2muv10tp_tf500_bs32_iter_16k_teach_8k/swinrnn_depth5.pth",
    # "output/swinrnn_t2muv10tp_tf500_bs32_iter_16k_teach_8k/swinvrnn_feat.pth",
    # "output/swinrnn_ppn_depth5_4x/pred05.pth",
    # "output/swinrnn_ppn_depth5_4x/pred06.pth",
    # "output/swinrnn_ppn_depth5_4x/pred07.pth",
    # "output/swinrnn_ppn_depth5_4x/pred08.pth",
    # "output/swinrnn_ppn_depth5_4x/pred09.pth",
    # "output/swinrnn_ppn_depth5_4x/pred10.pth",
    # "output/exp02/predictions_sample100_lead14.pth",
    # "output/exp08/predictions_sample100_lead14.pth",
    # "output/exp12/predictions_sample100_lead14.pth",
    "output/spread/sample_mean/predictions_swinvrnn.pth",
    "output/spread/sample_mean/predictions_swinvrnn_depth5.pth",
    "output/spread/sample_mean/predictions_swinvrnn_depth4.pth",
]

logger.info(f"ensemble: {len(pred_list)} model")

result = torch.load(pred_list[0])
avg_factor = len(pred_list)

for pred_f in pred_list[1:]:
    pred = torch.load(pred_f)
    logger.info(f"pred_f: {pred_f}")

    for name in args.eval_names:
        for t in np.arange(args.future_frames):
            key = "{}_{}".format(name, t)
            for i, v in enumerate(pred[key]):
                try:
                    result[key][i] += v
                except:
                    print(key, v.shape, result[key][i].shape)

for name in args.eval_names:
    logger.info(f"eval_name: {name}")

    for t in np.arange(args.future_frames):
        key = "{}_{}".format(name, t)
        for i, z in enumerate(result[key]):
            result[key][i] /= avg_factor


logger.info(f"save to: {args.save_f}")
with open(args.save_f, 'wb') as f:
    torch.save(result, f)

