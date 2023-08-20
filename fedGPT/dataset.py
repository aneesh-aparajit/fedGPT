import torch
import torch.nn as nn
import tiktoken
from torch.utils.data import Dataset, DataLoader


class IidDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
    
    def __len__(self) -> int:
        return
    
    def __getitem__(self, ix: int):
        pass


def get_dataloaders():
    pass
