# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import math
import itertools
import logging
from enum import Enum
from typing import (Any, Callable, Dict, Iterable, List, Optional, Set, Type,
                    Union)

import torch
from fanjiang.config import CfgNode

from .scheduler import (CosineParamScheduler, LRMultiplier,
                        MultiStepParamScheduler, WarmupParamScheduler)

_GradientClipperInput = Union[torch.Tensor, Iterable[torch.Tensor]]
_GradientClipper = Callable[[_GradientClipperInput], None]


class GradientClipType(Enum):
    VALUE = "value"
    NORM = "norm"


def _create_gradient_clipper(cfg: CfgNode) -> _GradientClipper:
    """
    Creates gradient clipping closure to clip by value or by norm,
    according to the provided config.
    """
    cfg = copy.deepcopy(cfg)

    def clip_grad_norm(p: _GradientClipperInput):
        torch.nn.utils.clip_grad_norm_(p, cfg.CLIP_VALUE, cfg.NORM_TYPE)

    def clip_grad_value(p: _GradientClipperInput):
        torch.nn.utils.clip_grad_value_(p, cfg.CLIP_VALUE)

    _GRADIENT_CLIP_TYPE_TO_CLIPPER = {
        GradientClipType.VALUE: clip_grad_value,
        GradientClipType.NORM: clip_grad_norm,
    }
    return _GRADIENT_CLIP_TYPE_TO_CLIPPER[GradientClipType(cfg.CLIP_TYPE)]


def _generate_optimizer_class_with_gradient_clipping(
    optimizer: Type[torch.optim.Optimizer],
    *,
    per_param_clipper: Optional[_GradientClipper] = None,
    global_clipper: Optional[_GradientClipper] = None,
) -> Type[torch.optim.Optimizer]:
    """
    Dynamically creates a new type that inherits the type of a given instance
    and overrides the `step` method to add gradient clipping
    """
    assert (
        per_param_clipper is None or global_clipper is None
    ), "Not allowed to use both per-parameter clipping and global clipping"

    def optimizer_wgc_step(self, closure=None):
        if per_param_clipper is not None:
            for group in self.param_groups:
                for p in group["params"]:
                    per_param_clipper(p)
        else:
            # global clipper for future use with detr
            # (https://github.com/facebookresearch/detr/pull/287)
            all_params = itertools.chain(*[g["params"] for g in self.param_groups])
            global_clipper(all_params)
        super(type(self), self).step(closure)

    OptimizerWithGradientClip = type(
        optimizer.__name__ + "WithGradientClip",
        (optimizer,),
        {"step": optimizer_wgc_step},
    )
    return OptimizerWithGradientClip


def maybe_add_gradient_clipping(
    cfg: CfgNode, optimizer: Type[torch.optim.Optimizer]
) -> Type[torch.optim.Optimizer]:
    """
    If gradient clipping is enabled through config options, wraps the existing
    optimizer type to become a new dynamically created class OptimizerWithGradientClip
    that inherits the given optimizer and overrides the `step` method to
    include gradient clipping.

    Args:
        cfg: CfgNode, configuration options
        optimizer: type. A subclass of torch.optim.Optimizer

    Return:
        type: either the input `optimizer` (if gradient clipping is disabled), or
            a subclass of it with gradient clipping included in the `step` method.
    """
    if not cfg.SOLVER.CLIP_GRADIENTS.ENABLED:
        return optimizer
    if isinstance(optimizer, torch.optim.Optimizer):
        optimizer_type = type(optimizer)
    else:
        assert issubclass(optimizer, torch.optim.Optimizer), optimizer
        optimizer_type = optimizer

    grad_clipper = _create_gradient_clipper(cfg.SOLVER.CLIP_GRADIENTS)
    OptimizerWithGradientClip = _generate_optimizer_class_with_gradient_clipping(
        optimizer_type, per_param_clipper=grad_clipper
    )
    if isinstance(optimizer, torch.optim.Optimizer):
        optimizer.__class__ = OptimizerWithGradientClip  # a bit hacky, not recommended
        return optimizer
    else:
        return OptimizerWithGradientClip


def build_optimizer(cfg: CfgNode, model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Build an optimizer from config.
    """


    params = get_default_optimizer_params(
        model,
        base_lr=cfg.SOLVER.BASE_LR,
        weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
        bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
        weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
        # overrides=model.overrides,
    )

    optimizer_type = cfg.SOLVER.OPTIMIZER
    if optimizer_type == "SGD":
        return maybe_add_gradient_clipping(cfg, torch.optim.SGD)(
            params,
            lr=cfg.SOLVER.BASE_LR,
            momentum=cfg.SOLVER.MOMENTUM,
            nesterov=cfg.SOLVER.NESTEROV,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    elif optimizer_type == "ADAM":
        return maybe_add_gradient_clipping(cfg, torch.optim.Adam)(
            params,
            lr=cfg.SOLVER.BASE_LR,
            betas=cfg.SOLVER.BETAS,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )

    elif optimizer_type == "ADAMW":
        return maybe_add_gradient_clipping(cfg, torch.optim.AdamW)(
            params,
            lr=cfg.SOLVER.BASE_LR,
            betas=cfg.SOLVER.BETAS,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    else:
        raise NotImplementedError(f"no optimizer type {optimizer_type}")



def get_default_optimizer_params(
    model: torch.nn.Module,
    base_lr: Optional[float] = None,
    weight_decay: Optional[float] = None,
    weight_decay_norm: Optional[float] = None,
    bias_lr_factor: Optional[float] = 1.0,
    weight_decay_bias: Optional[float] = None,
    overrides: Optional[Dict[str, Dict[str, float]]] = None,
):
    """
    Get default param list for optimizer, with support for a few types of
    overrides. If no overrides needed, this is equivalent to `model.parameters()`.

    Args:
        base_lr: lr for every group by default. Can be omitted to use the one in optimizer.
        weight_decay: weight decay for every group by default. Can be omitted to use the one
            in optimizer.
        weight_decay_norm: override weight decay for params in normalization layers
        bias_lr_factor: multiplier of lr for bias parameters.
        weight_decay_bias: override weight decay for bias parameters
        overrides: if not `None`, provides values for optimizer hyperparameters
            (LR, weight decay) for module parameters with a given name; e.g.
            ``{"embedding": {"lr": 0.01, "weight_decay": 0.1}}`` will set the LR and
            weight decay values for all module parameters named `embedding`.

    For common detection models, ``weight_decay_norm`` is the only option
    needed to be set. ``bias_lr_factor,weight_decay_bias`` are legacy settings
    from Detectron1 that are not found useful.

    Example:
    ::
        torch.optim.SGD(get_default_optimizer_params(model, weight_decay_norm=0),
                       lr=0.01, weight_decay=1e-4, momentum=0.9)
    """
    if overrides is None:
        overrides = {}
    defaults = {}
    if base_lr is not None:
        defaults["lr"] = base_lr
    if weight_decay is not None:
        defaults["weight_decay"] = weight_decay
    bias_overrides = {}
    if bias_lr_factor is not None and bias_lr_factor != 1.0:
        # NOTE: unlike Detectron v1, we now by default make bias hyperparameters
        # exactly the same as regular weights.
        if base_lr is None:
            raise ValueError("bias_lr_factor requires base_lr")
        bias_overrides["lr"] = base_lr * bias_lr_factor
    if weight_decay_bias is not None:
        bias_overrides["weight_decay"] = weight_decay_bias
    if len(bias_overrides):
        if "bias" in overrides:
            raise ValueError("Conflicting overrides for 'bias'")
        overrides["bias"] = bias_overrides

    norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        # NaiveSyncBatchNorm inherits from BatchNorm2d
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
    )
    params: List[Dict[str, Any]] = []
    memo: Set[torch.nn.parameter.Parameter] = set()
    for module_name, module in model.named_modules():
        for param_name, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)

            hyperparams = copy.copy(defaults)
            if isinstance(module, norm_module_types) and weight_decay_norm is not None:
                hyperparams["weight_decay"] = weight_decay_norm

            # for prefix in overrides.get("prefix", {}):
            #     if prefix in module_name:
            #         hyperparams["lr"] *= 20

            hyperparams.update(overrides.get(param_name, {}))
            params.append({"params": [value], **hyperparams})
    return params



class CyclicScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        base_lr,
        anneal_decay=0.01,
        anneal_iters=10,
        anneal_strategy='cos',
        last_epoch=-1
    ):
        base_lrs = self._format_param(optimizer, base_lr)

        for base_lr, group in zip(base_lrs, optimizer.param_groups):
            group['start_lr'] = base_lr
            group['end_lr'] = base_lr * anneal_decay

        if anneal_strategy not in ['cos', 'linear']:
            raise ValueError("anneal_strategy must by one of 'cos' or 'linear', "
                             "instead got {}".format(anneal_strategy))
        elif anneal_strategy == 'cos':
            self.anneal_func = self._cosine_anneal
        elif anneal_strategy == 'linear':
            self.anneal_func = self._linear_anneal
        if not isinstance(anneal_iters, int) or anneal_iters < 0:
            raise ValueError("anneal_iters must be equal or greater than 0, got {}".format(
                             anneal_iters))
        self.anneal_iters = anneal_iters
        super(CyclicScheduler, self).__init__(optimizer, last_epoch)

    @staticmethod
    def _format_param(optimizer, lrs):
        if isinstance(lrs, (list, tuple)):
            if len(lrs) != len(optimizer.param_groups):
                raise ValueError("lrs must have the same length as "
                                 "optimizer.param_groups: lrs has {}, "
                                 "optimizer.param_groups has {}".format(
                                     len(lrs), len(optimizer.param_groups)))
            return lrs
        else:
            return [lrs] * len(optimizer.param_groups)

    @staticmethod
    def _linear_anneal(t):
        return t

    @staticmethod
    def _cosine_anneal(t):
        return (1 - math.cos(math.pi * t)) / 2

    def get_lr(self):
        progress = self._step_count % self.anneal_iters

        t = max(0, min(1, progress / max(1, self.anneal_iters)))
        alpha = self.anneal_func(t)

        return [group['end_lr'] * alpha + group["start_lr"] * (1 - alpha)
                for group in self.optimizer.param_groups]




def build_lr_scheduler(
    cfg: CfgNode, optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Build a LR scheduler from config.
    """
    name = cfg.SOLVER.LR_SCHEDULER_NAME

    # sched = CyclicScheduler(
    #     optimizer,
    #     cfg.SOLVER.BASE_LR,
    #     anneal_decay=cfg.SOLVER.ANNEAL_DECAY,
    #     anneal_iters=cfg.SOLVER.ANNEAL_ITERS,
    # )
    # return sched


    if name == "WarmupMultiStepLR":
        steps = [x for x in cfg.SOLVER.STEPS if x <= cfg.SOLVER.MAX_ITER]
        if len(steps) != len(cfg.SOLVER.STEPS):
            logger = logging.getLogger(__name__)
            logger.warning(
                "SOLVER.STEPS contains values larger than SOLVER.MAX_ITER. "
                "These values will be ignored."
            )
        sched = MultiStepParamScheduler(
            values=[cfg.SOLVER.GAMMA ** k for k in range(len(steps) + 1)],
            milestones=steps,
            num_updates=cfg.SOLVER.MAX_ITER,
        )
    elif name == "WarmupCosineLR":
        sched = CosineParamScheduler(1, 0)
    else:
        raise ValueError("Unknown LR scheduler: {}".format(name))

    sched = WarmupParamScheduler(
        sched,
        cfg.SOLVER.WARMUP_FACTOR,
        min(cfg.SOLVER.WARMUP_ITERS / cfg.SOLVER.MAX_ITER, 1.0),
        cfg.SOLVER.WARMUP_METHOD,
    )
    return LRMultiplier(optimizer, multiplier=sched, max_iter=cfg.SOLVER.MAX_ITER)
