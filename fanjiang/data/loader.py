import numpy as np
import torch
import torch.utils.data as torchdata

from fanjiang.builder import build_datasets
from fanjiang.utils.comm import get_world_size
from fanjiang.utils.env import seed_all_rng

from .sampler import InferenceSampler, TrainingSampler

def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    data = {}

    for k in batch[0]:
        data[k] = torch.stack([x[k] for x in batch])

    return data


def worker_init_reset_seed(worker_id):
    initial_seed = torch.initial_seed() % 2 ** 31
    seed_all_rng(initial_seed + worker_id)



def build_train_loader(
    dataset,
    *,
    sampler=None,
    micro_batch_size,
    num_workers=0,
    pin_memory=False
):
    world_size = get_world_size()
    batch_size = micro_batch_size // world_size

    if sampler is None:
        sampler = TrainingSampler(len(dataset))

    batch_sampler = torchdata.sampler.BatchSampler(
        sampler, batch_size, drop_last=True
    )  # drop_last so the batch always have the same size

    return torchdata.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
        worker_init_fn=worker_init_reset_seed,
        pin_memory=pin_memory,
    )


def build_test_loader(
    dataset,
    *,
    sampler=None,
    micro_batch_size,
    num_workers=0,
    pin_memory=False,
):
    world_size = get_world_size()
    batch_size = micro_batch_size // world_size

    if sampler is None:
        sampler = InferenceSampler(len(dataset))

    batch_sampler = torchdata.sampler.BatchSampler(
        sampler, batch_size, drop_last=False
    )

    return torchdata.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
        pin_memory=pin_memory,
    )


