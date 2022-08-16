# Common training-related configs that are designed for "tools/lazyconfig_train_net.py"
# You can use your own instead, together with your own train_net.py

train = dict(
    output_dir="./output",
    init_checkpoint="",
    max_iter=90000,
    accumulation_steps=1,
    amp=False,  # options for Automatic Mixed Precision
    ddp=dict(  # options for DistributedDataParallel
        broadcast_buffers=False,
        find_unused_parameters=False,
        fp16_compression=False,
    ),
    ema=dict(decay=0.9999, warmup_iters=1000, device=None),
    checkpointer=dict(period=5000, max_to_keep=100),  # options for PeriodicCheckpointer
    clip_grad=dict(enabled=False, max_norm=1),
    teacher_forcing=dict(enabled=False, k=500),
    eval_period=5000,
    log_period=20,
    device="cuda"
)

