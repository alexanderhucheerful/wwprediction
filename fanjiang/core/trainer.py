import logging
import time
import weakref
from collections import OrderedDict
from copy import deepcopy
from typing import List, Mapping, Optional
from einops.einops import rearrange

import fanjiang.utils.comm as comm
import numpy as np
import torch
import torch.nn.functional as F
from fanjiang.builder import build_criterions, build_metrics, build_model
from fanjiang.dataset import create_test_loader, create_train_loader
from fanjiang.utils.events import EventStorage, get_event_storage
from fanjiang.utils.logger import _log_api_usage, setup_logger
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.cuda.amp import GradScaler, autocast

from . import hooks
from .checkpoint import Checkpointer
from .defaults import TensorboardXWriter, default_writers
from .evaluator import inference_on_dataset, print_csv_format
from .optimizer import build_lr_scheduler, build_optimizer
from torch.optim.swa_utils import AveragedModel

__all__ = ["TrainerBase", "SimpleTrainer", "AdversarialTrainer"]

HookBase = hooks.HookBase

def create_ddp_model(model, *, fp16_compression=False, **kwargs):
    """
    Create a DistributedDataParallel model if there are >1 processes.

    Args:
        model: a torch.nn.Module
        fp16_compression: add fp16 compression hooks to the ddp object.
            See more at https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook
        kwargs: other arguments of :module:`torch.nn.parallel.DistributedDataParallel`.
    """  # noqa
    if comm.get_world_size() == 1:
        return model
    if "device_ids" not in kwargs:
        kwargs["device_ids"] = [comm.get_local_rank()]
    ddp = DistributedDataParallel(model, **kwargs)
    if fp16_compression:
        from torch.distributed.algorithms.ddp_comm_hooks import \
            default as comm_hooks

        ddp.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)
    return ddp


def auto_scale_workers(cfg, num_workers: int):
    """
    When the config is defined for certain number of workers (according to
    ``cfg.SOLVER.REFERENCE_WORLD_SIZE``) that's different from the number of
    workers currently in use, returns a new cfg where the total batch size
    is scaled so that the per-GPU batch size stays the same as the
    original ``IMS_PER_BATCH // REFERENCE_WORLD_SIZE``.

    Other config options are also scaled accordingly:
    * training steps and warmup steps are scaled inverse proportionally.
    * learning rate are scaled proportionally, following :paper:`ImageNet in 1h`.

    For example, with the original config like the following:

    .. code-block:: yaml

        IMS_PER_BATCH: 16
        BASE_LR: 0.1
        REFERENCE_WORLD_SIZE: 8
        MAX_ITER: 5000
        STEPS: (4000,)
        CHECKPOINT_PERIOD: 1000

    When this config is used on 16 GPUs instead of the reference number 8,
    calling this method will return a new config with:

    .. code-block:: yaml

        IMS_PER_BATCH: 32
        BASE_LR: 0.2
        REFERENCE_WORLD_SIZE: 16
        MAX_ITER: 2500
        STEPS: (2000,)
        CHECKPOINT_PERIOD: 500

    Note that both the original config and this new config can be trained on 16 GPUs.
    It's up to user whether to enable this feature (by setting ``REFERENCE_WORLD_SIZE``).

    Returns:
        CfgNode: a new config. Same as original if ``cfg.SOLVER.REFERENCE_WORLD_SIZE==0``.
    """
    old_world_size = cfg.SOLVER.REFERENCE_WORLD_SIZE
    if old_world_size == 0 or old_world_size == num_workers:
        return cfg
    cfg = cfg.clone()
    frozen = cfg.is_frozen()
    cfg.defrost()

    assert (
        cfg.SOLVER.IMS_PER_BATCH % old_world_size == 0
    ), "Invalid REFERENCE_WORLD_SIZE in config!"
    scale = num_workers / old_world_size
    bs = cfg.SOLVER.IMS_PER_BATCH = int(round(cfg.SOLVER.IMS_PER_BATCH * scale))
    lr = cfg.SOLVER.BASE_LR = cfg.SOLVER.BASE_LR * scale
    max_iter = cfg.SOLVER.MAX_ITER = int(round(cfg.SOLVER.MAX_ITER / scale))
    warmup_iter = cfg.SOLVER.WARMUP_ITERS = int(round(cfg.SOLVER.WARMUP_ITERS / scale))
    cfg.SOLVER.STEPS = tuple(int(round(s / scale)) for s in cfg.SOLVER.STEPS)
    cfg.TEST.EVAL_PERIOD = int(round(cfg.TEST.EVAL_PERIOD / scale))
    cfg.SOLVER.CHECKPOINT_PERIOD = int(round(cfg.SOLVER.CHECKPOINT_PERIOD / scale))
    cfg.SOLVER.REFERENCE_WORLD_SIZE = num_workers  # maintain invariant
    logger = logging.getLogger(__name__)
    logger.info(
        f"Auto-scaling the config to batch_size={bs}, learning_rate={lr}, "
        f"max_iter={max_iter}, warmup={warmup_iter}."
    )

    if frozen:
        cfg.freeze()
    return cfg


class EMA:
    def __init__(self, model, decay=0.9999, warmup_iters=1000):
        self.model = deepcopy(model).eval()
        self.decay = lambda x: decay * (1 - np.exp(-x / warmup_iters))

        for p in self.model.parameters():
            p.requires_grad_(False)

    def update(self, model, iter):
        beta = self.decay(iter)

        with torch.no_grad():
            for p_ema, p in zip(self.model.parameters(), model.parameters()):
                p_ema.copy_(p.lerp(p_ema, beta))

            for b_ema, b in zip(self.model.buffers(), model.buffers()):
                b_ema.copy_(b)

class TrainerBase:
    """
    Base class for iterative trainer with hooks.

    The only assumption we made here is: the training runs in a loop.
    A subclass can implement what the loop is.
    We made no assumptions about the existence of dataloader, optimizer, model, etc.

    Attributes:
        iter(int): the current iteration.

        start_iter(int): The iteration to start with.
            By convention the minimum possible value is 0.

        max_iter(int): The iteration to end training.

        storage(EventStorage): An EventStorage that's opened during the course of training.
    """

    def __init__(self) -> None:
        self._hooks: List[HookBase] = []
        self.iter: int = 0
        self.start_iter: int = 0
        self.max_iter: int
        self.storage: EventStorage
        _log_api_usage("trainer." + self.__class__.__name__)

    def register_hooks(self, hooks: List[Optional[HookBase]]) -> None:
        """
        Register hooks to the trainer. The hooks are executed in the order
        they are registered.

        Args:
            hooks (list[Optional[HookBase]]): list of hooks
        """
        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and trainer cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
        self._hooks.extend(hooks)

    def train(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step()
                    self.after_step()
                # self.iter == max_iter can be used by `after_train` to
                # tell whether the training successfully finished or failed
                # due to exceptions.
                self.iter += 1
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    def before_train(self):
        for h in self._hooks:
            h.before_train()

    def after_train(self):
        self.storage.iter = self.iter
        for h in self._hooks:
            h.after_train()

    def before_step(self):
        # Maintain the invariant that storage.iter == trainer.iter
        # for the entire execution of each step
        self.storage.iter = self.iter

        for h in self._hooks:
            h.before_step()

    def after_step(self):
        for h in self._hooks:
            h.after_step()

    def run_step(self):
        raise NotImplementedError

    def state_dict(self):
        ret = {"iteration": self.iter}
        hooks_state = {}
        for h in self._hooks:
            sd = h.state_dict()
            if sd:
                name = type(h).__qualname__
                if name in hooks_state:
                    # TODO handle repetitive stateful hooks
                    continue
                hooks_state[name] = sd
        if hooks_state:
            ret["hooks"] = hooks_state
        return ret

    def load_state_dict(self, state_dict):
        logger = logging.getLogger(__name__)
        self.iter = state_dict["iteration"]
        for key, value in state_dict.get("hooks", {}).items():
            for h in self._hooks:
                try:
                    name = type(h).__qualname__
                except AttributeError:
                    continue
                if name == key:
                    h.load_state_dict(value)
                    break
            else:
                logger.warning(f"Cannot find the hook '{key}', its state_dict is ignored.")


class SimpleTrainer(TrainerBase):
    """
    A simple trainer for the most common type of task:
    single-cost single-optimizer single-data-source iterative optimization,
    optionally using data-parallelism.
    It assumes that every step, you:

    1. Compute the loss with a data from the data_loader.
    2. Compute the gradients with the above loss.
    3. Update the model with the optimizer.

    All other tasks during training (checkpointing, logging, evaluation, LR schedule)
    are maintained by hooks, which can be registered by :meth:`TrainerBase.register_hooks`.

    If you want to do anything fancier than this,
    either subclass TrainerBase and implement your own `run_step`,
    or write your own training loop.
    """

    def __init__(self, model, data_loader, optimizer):
        """
        Args:
            model: a torch Module. Takes a data from data_loader and returns a
                dict of losses.
            data_loader: an iterable. Contains data to be used to call model.
            optimizer: a torch optimizer.
        """
        super().__init__()

        """
        We set the model to training mode in the trainer.
        However it's valid to train a model that's in eval mode.
        If you want your model (or a submodule of it) to behave
        like evaluation during training, you can overwrite its train() method.
        """
        model.train()

        self.model = model
        self.data_loader = data_loader
        self._data_loader_iter = iter(data_loader)
        self.optimizer = optimizer

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        """
        If you want to do something with the losses, you can wrap the model.
        """
        loss_dict = self.model(data)
        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
            loss_dict = {"total_loss": loss_dict}
        else:
            losses = sum(loss_dict.values())

        """
        If you need to accumulate gradients or do something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()
        losses.backward()

        self._write_metrics(loss_dict, data_time)

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method. But it is
        suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
        """
        self.optimizer.step()

    def _write_metrics(
        self,
        loss_dict: Mapping[str, torch.Tensor],
        data_time: float,
        prefix: str = "",
    ) -> None:
        SimpleTrainer.write_metrics(loss_dict, data_time, prefix)

    @staticmethod
    def write_metrics(
        loss_dict: Mapping[str, torch.Tensor],
        data_time: float,
        prefix: str = "",
    ) -> None:
        """
        Args:
            loss_dict (dict): dict of scalar losses
            data_time (float): time taken by the dataloader iteration
            prefix (str): prefix for logging keys
        """
        metrics_dict = {k: v.detach().cpu().item() for k, v in loss_dict.items()}
        metrics_dict["data_time"] = data_time

        # Gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            storage = get_event_storage()

            # data_time among workers can have high variance. The actual latency
            # caused by data_time is the maximum among workers.
            data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
            storage.put_scalar("data_time", data_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
            }
            total_losses_reduced = sum(metrics_dict.values())
            if not np.isfinite(total_losses_reduced):
                raise FloatingPointError(
                    f"Loss became infinite or NaN at iteration={storage.iter}!\n"
                    f"loss_dict = {metrics_dict}"
                )

            storage.put_scalar("{}total_loss".format(prefix), total_losses_reduced)
            if len(metrics_dict) > 1:
                storage.put_scalars(**metrics_dict)

    def state_dict(self):
        ret = super().state_dict()
        ret["optimizer"] = self.optimizer.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.optimizer.load_state_dict(state_dict["optimizer"])

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
class AdversarialTrainer(TrainerBase):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        super().__init__()
        self.logger = logging.getLogger(__name__)
        if not self.logger.isEnabledFor(logging.INFO):
            setup_logger()

        self.with_ema = cfg.MODEL.WITH_EMA
        self.with_amp = cfg.MODEL.WITH_AMP

        self.device = cfg.MODEL.DEVICE
        self.vis_period = cfg.VIS_PERIOD
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.warmup_iter = cfg.SOLVER.WARMUP_ITERS
        self.anneal_iter = cfg.SOLVER.ANNEAL_ITERS

        self.cfg = cfg
        model = self.build_models(cfg.MODEL.GENERATOR).to(self.device)
        n_params = count_parameters(model)
        self.logger.info('Number of parameters in the model: ' + f'{n_params / 1e6:.2f} M.')

        optimizer = build_optimizer(cfg, model)
        scheduler = build_lr_scheduler(cfg, optimizer)
        self.grad_scaler = GradScaler(enabled=self.with_amp)
        # self.grad_scaler = NativeScaler(enabled=self.with_amp)

        if self.with_ema:
            self.model_ema = EMA(model)

        model = create_ddp_model(model, broadcast_buffers=False)

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        train_loader = create_train_loader(cfg)
        self.test_loader = create_test_loader(cfg)
        self.train_loader_iter = iter(train_loader)

        self.metrics = build_metrics(cfg.METRICS)
        self.criterions = build_criterions(cfg.CRITERIONS)

        self.checkpointer = Checkpointer(
            self.model,
            cfg.OUTPUT_DIR,
            save_to_disk=comm.is_main_process,
            trainer=weakref.proxy(self),
        )

        if self.with_ema:
            self.checkpointer_ema = Checkpointer(
                self.model_ema.model,
                cfg.OUTPUT_DIR,
                save_to_disk=comm.is_main_process,
                trainer=weakref.proxy(self),
            )


    @classmethod
    def build_models(cls, cfg):
        model = build_model(cfg)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        return model

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.

        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.

        Args:
            resume (bool): whether to do resume or not
        """
        self.checkpointer.resume_or_load(
            self.cfg.MODEL.WEIGHTS, resume=resume
        )

        if self.with_ema:
            self.checkpointer_ema.resume_or_load(
                self.cfg.MODEL.WEIGHTS, resume=resume
            )

        if resume and self.checkpointer.has_checkpoint():
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration
            self.start_iter = self.iter + 1


    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, writing events.
        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
        ]

        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(
                    self.checkpointer_ema if self.with_ema else self.checkpointer,
                    cfg.SOLVER.CHECKPOINT_PERIOD,
                    max_to_keep=cfg.SOLVER.CHECKPOINT_KEEP,
                    file_prefix="model"
                )
            )
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, self.test))

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(default_writers(cfg), period=cfg.LOG_PERIOD))
        return ret


    def train(self):
        """
        Run training.

        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        self.register_hooks(self.build_hooks())
        super().train(self.start_iter, self.max_iter)


    def test(self):
        results = OrderedDict()
        for _, dataset_name in enumerate(self.cfg.DATASETS.TEST):
            results_i = inference_on_dataset(
                self.model_ema.model if self.with_ema else self.model,
                self.test_loader,
                self.metrics,
                self.cfg.TEST.EVAL_NUM,
                self.cfg.TEST.SAVE_DIR,
            )
            results[dataset_name] = results_i

            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results


    # def adjust_teacher_forcing(self, k=7000):
    def adjust_teacher_forcing(self, k=500):
        iteration = self.iter + 1

        sampling_ratio = k / (k + np.exp(iteration / k))
        # sampling_scale = 1 - np.exp(-self.iter/k)

        if sampling_ratio < 1e-6:
            sampling_ratio = 0

        # if 1 - sampling_scale < 1e-6:
        #     sampling_scale = 1

        self.storage.put_scalars(
            sampling_ratio=sampling_ratio,
            # sampling_scale=sampling_scale,
            smoothing_hint=False
        )

        teach_info = {
            "sampling_ratio": sampling_ratio,
            # "sampling_scale": sampling_scale,
        }
        return teach_info


    def after_epoch(self):
        return self.iter % self.anneal_iter == 0

    def run_step(self):

        start = time.perf_counter()
        inputs, info = next(self.train_loader_iter)

        for name in inputs:
            inputs[name] = inputs[name].to(self.device, non_blocking=True)

        data_time = time.perf_counter() - start

        losses = {}
        self.model.requires_grad_(True)

        info.update(self.adjust_teacher_forcing())
        if (self.iter + 1) % self.cfg.LOG_PERIOD == 0:
            self.logger.info('sampling_ratio: {}'.format(info["sampling_ratio"]))
            # self.logger.info('sampling_scale: {}'.format(info["sampling_scale"]))

        with autocast(enabled=self.with_amp):
            outputs = self.model(inputs, info=info)
            losses.update(outputs["loss"])


        loss = sum(losses.values())
        self.optimizer.zero_grad()
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        # grad_norm = self.grad_scaler(loss, self.optimizer, parameters=self.model.parameters())
        self.model.requires_grad_(False)

        if self.with_ema:
            self.model_ema.update(self.model, self.iter)

        SimpleTrainer.write_metrics(losses, data_time)




