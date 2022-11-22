#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from fanjiang.config import get_cfg
from fanjiang.core import (AdversarialTrainer, default_argument_parser,
                           default_setup, launch)
from fanjiang.utils.events import EventStorage


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    # cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    trainer = AdversarialTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    if args.eval_only:
        with EventStorage() as storage:
            return trainer.test()

    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
