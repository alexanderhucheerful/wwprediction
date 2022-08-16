import logging
from fanjiang.config import LazyConfig, instantiate

from fanjiang.core import (
    EMA,
    LRMultiplier,
    Checkpointer,
    SimpleTrainer,
    create_ddp_model,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
    launch,
    inference_on_dataset,
    print_csv_format,
)


from fanjiang.utils import comm

logger = logging.getLogger("fanjiang")

def do_test(cfg, model):
    if "evaluator" in cfg.dataloader:
        ret = inference_on_dataset(
            model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
        )
        print_csv_format(ret)
        return ret

def do_train(args, cfg):
    model = instantiate(cfg.model)

    model.freeze_stages()
    flops = model.flops()
    params = sum(map(lambda x: x.numel(), model.parameters()))

    logger = logging.getLogger("fanjiang")
    logger.info("flops: {:.3f}G, params: {:.3f}M".format(flops/1e9, params/1e6))
    logger.info("Model:\n{}".format(model))

    model.to(cfg.train.device)


    cfg.optimizer.params.model = model
    # cfg.optimizer.params.overrides = model.overrides()

    optimizer = instantiate(cfg.optimizer)

    train_loader = instantiate(cfg.dataloader.train)
    lr_scheduler = LRMultiplier(optimizer, instantiate(cfg.lr_multiplier), cfg.train.max_iter)

    model_ema = EMA(model, **cfg.train.ema) if args.use_ema else None

    if args.deepspeed:
        import deepspeed
        model, optimizer, _, lr_scheduler = deepspeed.initialize(
            args=args,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )
    else:
        model = create_ddp_model(model, **cfg.train.ddp)

    cfg.train.deepspeed = args.deepspeed
    trainer = SimpleTrainer(model, train_loader, optimizer, cfg.train)

    checkpointer = Checkpointer(
        model,
        model_ema=model_ema,
        save_dir=cfg.train.output_dir,
        trainer=trainer,
    )

    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            hooks.LRScheduler(scheduler=lr_scheduler),

            hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer)
            if comm.is_main_process()
            else None,

            hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model)),

            hooks.PeriodicWriter(
                default_writers(cfg.train.output_dir, cfg.train.max_iter),
                period=cfg.train.log_period,
            )
            if comm.is_main_process()
            else None,
        ]
    )

    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)

    if args.resume and checkpointer.has_checkpoint():
        start_iter = trainer.iter + 1
    else:
        start_iter = 0
    trainer.train(start_iter, cfg.train.max_iter)


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)

    default_setup(cfg, args)

    if args.eval_only:
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        model = create_ddp_model(model)
        Checkpointer(model).load(cfg.train.init_checkpoint, use_ema=args.use_ema)
        do_test(cfg, model)
    else:
        do_train(args, cfg)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
