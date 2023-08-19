import flwr as fl
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import call, instantiate
from omegaconf import DictConfig, OmegaConf

from fedGPT.client import get_client_fn
from fedGPT.server import get_evaluate_fn, get_on_fit_config


@hydra.main(config_path="./conf/", config_name="nase", version_base=None)
def main(cfg: DictConfig):
    print('Config:')
    print('-------\n')
    print(OmegaConf.to_yaml(cfg))


if __name__ == '__main__':
    main()
