from collections import OrderedDict
from typing import Dict, Tuple

import torch
from flwr.common import NDArrays, Scalar
from hydra.utils import instantiate
from omegaconf import DictConfig

from fedGPT.engine import test


def get_on_fit_config(config: DictConfig):
    def on_fit_config(server_rounds: int):
        return {}

    return on_fit_config


def get_evaluate_fn(model_cfg, test_loader):
    def evaluate_fn(
        server_round: int, parameters: NDArrays, config
    ) -> Tuple[float, Dict[str, Scalar]]:
        model = instantiate(config=model_cfg)
        device = torch.device(
            "cuda:0" if torch.has_cuda else "mps" if torch.has_mps else "cpu"
        )

        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict)

        loss = test(model=model, dataloader=test_loader, device=device)

        return float(loss), {}

    return evaluate_fn

