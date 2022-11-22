
# python projects/wwprediction/tools/inference.py \
#     --config_file projects/wwprediction/configs/unet_cref_chongqing.yaml \
#     --radar_dir data/weather_chongqing/baiduyunpan/重庆市气象台训练数据/MCR/20210513 \
#     --save_dir results/202105132000 \
#     --init_time 202105132000 \
#     --size 896 \
#     --stride 256 \
#     --border 192 \
#     --input_frames 10 \
#     --future_frames 30 \

# python projects/wwprediction/tools/inference.py \
#     --config_file projects/wwprediction/configs/unet_cref_chongqing.yaml \
#     --radar_dir data/weather_chongqing/baiduyunpan/重庆市气象台训练数据/MCR/20210626 \
#     --save_dir results/202106262000 \
#     --init_time 202106262000 \
#     --size 896 \
#     --stride 256 \
#     --border 192 \
#     --input_frames 10 \
#     --future_frames 30 \

# python projects/wwprediction/tools/inference.py \
#     --config_file projects/wwprediction/configs/unet_cref_chongqing.yaml \
#     --radar_dir data/weather_chongqing/baiduyunpan/重庆市气象台训练数据/MCR/20210706 \
#     --save_dir results/202107062000 \
#     --init_time 202107062000 \
#     --size 896 \
#     --stride 256 \
#     --border 192 \
#     --input_frames 10 \
#     --future_frames 30 \

# python projects/wwprediction/tools/inference.py \
#     --config_file projects/wwprediction/configs/unet_cref_chongqing.yaml \
#     --radar_dir data/weather_chongqing/baiduyunpan/重庆市气象台训练数据/MCR/20210807 \
#     --save_dir results/202108070800 \
#     --init_time 202108070800 \
#     --size 896 \
#     --stride 256 \
#     --border 192 \
#     --input_frames 10 \
#     --future_frames 30 \

# python projects/wwprediction/tools/inference.py \
#     --config_file projects/wwprediction/configs/unet_cref_chongqing.yaml \
#     --radar_dir data/weather_chongqing/baiduyunpan/重庆市气象台训练数据/MCR/20210822 \
#     --save_dir results/202108222000 \
#     --init_time 202108222000 \
#     --size 896 \
#     --stride 256 \
#     --border 192 \
#     --input_frames 10 \
#     --future_frames 30 \

# python projects/wwprediction/tools/inference.py \
#     --config_file projects/wwprediction/configs/unet_cref_chongqing.yaml \
#     --radar_dir data/weather_chongqing/baiduyunpan/重庆市气象台训练数据/MCR/20210911 \
#     --save_dir results/202109112000 \
#     --init_time 202109112000 \
#     --size 896 \
#     --stride 256 \
#     --border 192 \
#     --input_frames 10 \
#     --future_frames 30 \


python projects/wwprediction/tools/inference.py \
    --config_file output/predrnn_radar_gan_ft/config.yaml \
    --radar_dir data/jiangsu/radar \
    --save_dir results/202104301200 \
    --init_time 202104301200 \
    --size 512 \
    --stride 256 \
    --border 0 \
    --input_frames 10 \
    --future_frames 30 \
    --region 35.2 30.4 116.33 121.93 \

python projects/wwprediction/tools/inference.py \
    --config_file output/predrnn_radar_gan_ft/config.yaml \
    --radar_dir data/jiangsu/radar \
    --save_dir results/202105150600 \
    --init_time 202105150600 \
    --size 512 \
    --stride 256 \
    --border 0 \
    --input_frames 10 \
    --future_frames 30 \
    --region 35.2 30.4 116.33 121.93 \

    # --weights output/unet_cref_jiangsu_repeat05/model_0007999.pth \
    # --weights output/unet_cref_jiangsu_2d/model_0047999.pth \
    # --save_dir results/202105150600 \
    # --init_time 202105150600 \
