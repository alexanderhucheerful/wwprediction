if [ $1 = "train" ]
then
    python tools/lazy_train_net.py \
        --config-file projects/weather_bench/configs/swinrnn.py \
        --num-gpus 8 --use-ema;
elif [ $1 = "test" ]
then
    for frame in {1..20}
    do
    python tools/lazy_train_net.py \
        --config-file projects/weather_bench/configs/swinrnn.py \
        --eval-only --use-ema --num-gpus 8 \
        model.future_frames=$frame \
        dataloader.test.dataset.future_frames=$frame;
    done
fi
