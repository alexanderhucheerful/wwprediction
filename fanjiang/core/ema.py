from copy import deepcopy
import math
import torch
import torch.nn as nn

__all__ = ["ModelEMA", "EMA"]


class ModelEMA(nn.Module):
    """ Model Exponential Moving Average V2

    Keep a moving average of everything in the model state_dict (parameters and buffers).
    V2 of this module is simpler, it does not match params/buffers based on name but simply
    iterates in order. It works with torchscript (JIT of full model).

    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage

    A smoothed version of the weights is necessary for some training schemes to perform well.
    E.g. Google's hyper-params for training MNASNet, MobileNet-V3, EfficientNet, etc that use
    RMSprop with a short 2.4-3 epoch decay period and slow LR decay rate of .96-.99 requires EMA
    smoothing of weights to match results. Pay attention to the decay constant you are using
    relative to your update count per epoch.

    To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but
    disable validation of the EMA weights. Validation will have to be done manually in a separate
    process, or after the training stops converging.

    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """
    def __init__(self, model, decay=0.9999, warmup_iters=1000, device=None):
        super(ModelEMA, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = lambda x: decay * (1 - math.exp(-x / warmup_iters))

        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model, itr):
        decay = self.decay(itr)
        self._update(model, update_fn=lambda e, m: decay * e + (1. - decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


class EMA:
    def __init__(self, model, decay=0.999, warmup_iters=1000, device=None):
        self.module = deepcopy(model).eval()
        self.decay = lambda x: decay * (1 - math.exp(-x / warmup_iters))

        for p in self.module.parameters():
            p.requires_grad_(False)

        self.device = device

        if self.device is not None:
            self.module.to(device=device)


    def update(self, model, itr):
        beta = self.decay(itr)

        with torch.no_grad():
            for p_ema, p in zip(self.module.parameters(), model.parameters()):
                if self.device is not None:
                    p = p.to(self.device)
                p_ema.copy_(p.lerp(p_ema, beta))

            for b_ema, b in zip(self.module.buffers(), model.buffers()):
                if self.device is not None:
                    b = b.to(self.device)
                b_ema.copy_(b)

