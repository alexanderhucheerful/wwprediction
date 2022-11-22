### train
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python tools/train_net.py \
    --config-file projects/wwprediction/configs/swinrnn_weather_bench.yaml \
    --num-gpus 8 \
    --dist-url "tcp://127.0.0.1:23456" \
    # --eval-only \

# evaluate
# output_dir="output/exp"
# python -m pdb projects/wwprediction/tools/eval_wb.py \
#     --data_dir data/weather_bench \
#     --save_dir ${output_dir}/results \
#     --pred_dir ${output_dir} \
#     --step 6 \
#     --interval 96 \
#     --lead_time 120 \
#     --input_frames 6 \
#     --future_frames 20 \
#     --eval_names z t t2m tp u10 v10 \
#     --metrics "mse" \
    # --ensembles n_samples_swinvrnn \
