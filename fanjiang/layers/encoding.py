import math
import numpy as np
import torch
from torch import nn

class StepEncoding(nn.Module):
    def __init__(self, dim, max_period=1e3):
        super().__init__()
        self.dim = dim
        self.num_freqs = dim // 2
        self.max_period = max_period
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)
        self.act = nn.SiLU()

    def forward(self, step):
        freqs = torch.arange(self.num_freqs, dtype=step.dtype, device=step.device) / self.num_freqs
        embedding = step * torch.exp(-np.log(self.max_period) * freqs[None])
        embedding = torch.cat([torch.sin(embedding), torch.cos(embedding)], dim=-1)
        embedding = self.fc2(self.act(self.fc1(embedding)))
        return embedding
