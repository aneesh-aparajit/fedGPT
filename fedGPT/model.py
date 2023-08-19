import math
from dataclasses import dataclass

import torch
import torch.nn as nn


# --------------------------------- Attention -------------------------------- #
class Attention(nn.Module):
    def __init__(self, n_embed: int, head_size: int) -> None:
        super().__init__()
        self.Q = nn.Linear(n_embed, head_size, bias=False)
        self.K = nn.Linear(n_embed, head_size, bias=False)
        self.V = nn.Linear(n_embed, head_size, bias=False)

        tril = torch.tril()

class nanoGPT(nn.Module):
    pass

