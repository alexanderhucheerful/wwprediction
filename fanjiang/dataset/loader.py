import torch
from fanjiang.builder import build_datasets
from fanjiang.utils.comm import get_world_size
from fanjiang.utils.env import seed_all_rng
from .sampler import InferenceSampler, TrainingSampler
import numpy as np


def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    transposed_batch = list(zip(*batch))
    data = [torch.stack(batch) for batch in transposed_batch[:-1]]
    common_info = transposed_batch[-1][0]

    inputs = {}
    for i, name in enumerate(common_info["input_names"]):
        inputs[name] = data[i]

    mean = common_info["mean"]
    std = common_info["std"]
    const = common_info["const"]

    info = { "mean": mean, "std": std, "const": const}

    if "idx" in common_info:
        info["idx"] = np.stack([batch['idx'] for batch in transposed_batch[-1]])

    return inputs, info


def worker_init_reset_seed(worker_id):
    initial_seed = torch.initial_seed() % 2 ** 31
    seed_all_rng(initial_seed + worker_id)


def create_train_loader(cfg):
    dataset = build_datasets(cfg.DATASETS.TRAIN)
    rst = dataset.__getitem__(0)
    sampler = TrainingSampler(len(dataset))

    total_batch_size = cfg.SOLVER.IMS_PER_BATCH
    world_size = get_world_size()
    assert (
        total_batch_size > 0 and total_batch_size % world_size == 0
    ), "Total batch size ({}) must be divisible by the number of gpus ({}).".format(
        total_batch_size, world_size
    )
    batch_size = total_batch_size // world_size

    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, batch_size, drop_last=True
    )  # drop_last so the batch always have the same size

    return torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
        worker_init_fn=worker_init_reset_seed,
        pin_memory=True,
    )


def create_test_loader(cfg):
    dataset = build_datasets(cfg.DATASETS.TEST)
    rst = dataset.__getitem__(0)
    sampler = InferenceSampler(len(dataset))

    total_batch_size = cfg.SOLVER.IMS_PER_BATCH
    world_size = get_world_size()
    batch_size = total_batch_size // world_size

    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, batch_size, drop_last=False
    )

    return torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
        pin_memory=True,
    )

