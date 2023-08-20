import flwr as fl
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import call, instantiate
from omegaconf import DictConfig, OmegaConf

from fedGPT.client import get_client_fn
from fedGPT.server import get_evaluate_fn, get_on_fit_config
from fedGPT.dataset import get_dataloaders


@hydra.main(config_path="./conf/", config_name="base", version_base=None)
def main(cfg: DictConfig):
    # 1. Load the config
    print('Config:')
    print('-------')
    print(OmegaConf.to_yaml(cfg))
    
    # 2. Load the dataloaders
    train_loaders, valid_loaders, test_loader = get_dataloaders()

    # 3. Define the clients
    client_fn = get_client_fn(trainloaders=train_loaders, validloaders=valid_loaders, model_conf=cfg.model)

    # 4. Define strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=cfg.strategy.fraction_fit,
        min_fit_clients=cfg.strategy.min_fit_clients,
        min_evaluate_clients=cfg.strategy.min_evaluate_clients,
        min_available_clients=cfg.strategy.min_available_clients,
        on_fit_config_fn=get_on_fit_config(config=cfg.config_fit),
        evaluate_fn=get_evaluate_fn(model_cfg=cfg.model, test_loader=test_loader)
    )

    # 5. Start Simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy
    )

if __name__ == '__main__':
    main()
