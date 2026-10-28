"""Train / evaluate routines shared by the centralized baseline and the federated clients."""
import torch
from monai import transforms as T
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric

import config as cfg

CLASSES = ["tc", "wt", "et"]  # tumour core, whole tumour, enhancing tumour


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loss():
    return DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, sigmoid=True)


def make_optimizer(model, lr):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)


def make_scaler(device):
    return torch.amp.GradScaler("cuda", enabled=device.type == "cuda")


def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler):
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        x, y = batch["image"].to(device), batch["label"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            loss = loss_fn(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_fn = make_loss()
    post = T.Compose([T.Activations(sigmoid=True), T.AsDiscrete(threshold=0.5)])
    dice = DiceMetric(include_background=True, reduction="mean_batch")
    total, n = 0.0, 0
    for batch in loader:
        x, y = batch["image"].to(device), batch["label"].to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = sliding_window_inference(x, cfg.ROI_SIZE, sw_batch_size=2,
                                              predictor=model, overlap=0.25)
        total += loss_fn(logits, y).item()
        n += 1
        dice(y_pred=[post(p) for p in decollate_batch(logits)], y=decollate_batch(y))
    per_class = torch.nan_to_num(dice.aggregate(), nan=0.0)
    dice.reset()
    out = {"loss": total / max(n, 1)}
    out.update({f"dice_{c}": float(v) for c, v in zip(CLASSES, per_class)})
    out["dice_mean"] = float(per_class.mean())
    return out
