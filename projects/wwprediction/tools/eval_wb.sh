python projects/wwprediction/tools/eval_wb.py \
    --data_dir data/weather_bench \
    --save_dir results \
    --pred_dir output/swinrnn_t2muv10tp_tf500_bs32_1x \
    --ifs data/weather_bench/baseline/tigge_5.625deg.nc \
    --step 6 \
    --interval 48 \
    --input_frames 6 \
    --future_frames 20 \
    --eval_names z t t2m u10 v10 tp \

