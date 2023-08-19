import torch
from model import nanoGPT
from torch.cuda import amp
from torch.utils.data import DataLoader
from tqdm import tqdm


# ----------------------------- Training Function ---------------------------- #
def train(
    model: nanoGPT,
    optim: torch.optim,
    dataloader: DataLoader,
    scalar: amp.grad_scaler.GradScaler,
    device: str,
    scheduler: torch.optim.lr_scheduler = None,
) -> float:
    model.train()
    model.to(device=device)
    pbar = tqdm(enumerate(dataloader), desc="(train)", total=len(dataloader))

    dataset_size, running_loss = 0, 0
    for step, batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}
        batch_size = batch["input_ids"].shape

        optim.zero_grad()

        with amp.autocast_mode.autocast():
            _, loss = model.forward()

        scalar.scale(loss).backward()
        scalar.step(optimizer=optim)
        scalar.step()
        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size
        epoch_loss = running_loss / dataset_size

    return epoch_loss


# ----------------------------- Testing Function ----------------------------- #
def test(model: nanoGPT, dataloader: DataLoader, device: str):
    model.eval()
    model.to(device=device)
    pbar = tqdm(enumerate(dataloader), desc="(valid)", total=len(dataloader))

    dataset_size, running_loss = 0, 0
    for step, batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}
        batch_size = batch["input_ids"].shape

        with amp.autocast_mode.autocast():
            _, loss = model.forward()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size
        epoch_loss = running_loss / dataset_size

    return epoch_loss
