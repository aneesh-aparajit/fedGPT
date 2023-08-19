from collections import OrderedDict
from typing import Dict, Tuple, List

import flwr as fl
import torch
from torch.cuda import amp
from flwr.common import NDArrays, Scalar
from hydra.utils import instantiate
from torch.utils.data import DataLoader

from fedGPT.model import nanoGPT
from fedGPT.engine import train, test
from fedGPT.utils import get_scheduler


class GptClient(fl.client.NumPyClient):
    def __init__(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        model_conf,
        optim_conf,
    ) -> None:
        super().__init__()

        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.model = instantiate(config=model_conf)
        self.model_conf = model_conf
        self.optim_conf = optim_conf
        self.device = torch.device(
            "cuda:0" if torch.has_cuda else "mps" if torch.has_mps else "cpu"
        )

    def set_parameters(self, parameters: NDArrays) -> None:
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict)

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        # this function receives the weights from the server.
        # so the first thing we must load is, load the server weights to the client's local model
        self.set_parameters(parameters=parameters)

        # once the model is loaded, we are next supposed to get the config variables
        optim = instantiate(config=self.optim_conf)
        scaler = amp.grad_scaler.GradScaler()
        scheduler = get_scheduler(optimizer=optim)

        loss = train(
            model=self.model,
            optim=optim,
            dataloader=self.train_loader,
            scheduler=scheduler,
            scalar=scaler,
            device=self.device,
        )
        return (
            self.get_parameters(config=config),
            len(self.train_loader),
            {"loss": loss},
        )

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        self.set_parameters(parameters=parameters)
        loss = test(model=self.model, dataloader=self.valid_loader, device=self.device)
        return float(loss), len(self.valid_loader), {}


def get_client_fn(
    trainloaders: List[DataLoader], validloaders: List[DataLoader], model_conf
):
    def client_fn(cid: str):
        return GptClient(
            train_loader=trainloaders[int(cid)],
            valid_loader=validloaders[int(cid)],
            model_conf=model_conf,
        )

    return client_fn
