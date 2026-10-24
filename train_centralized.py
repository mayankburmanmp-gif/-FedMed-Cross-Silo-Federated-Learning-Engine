"""WEEK 1 - Centralized baseline: train the 3D U-Net on the pooled data.

This is the accuracy ceiling the federated model (Week 2 audit) is compared against.
"""
import argparse
import json

import torch

import config as cfg
from data import get_datalist, make_loader, split_datalist
from engine import evaluate, get_device, make_loss, make_optimizer, make_scaler, train_one_epoch
from model import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=cfg.NUM_ROUNDS * cfg.LOCAL_EPOCHS,
                    help="default = rounds x local_epochs, i.e. the same compute as the federated run")
    ap.add_argument("--lr", type=float, default=cfg.LR)
    ap.add_argument("--val-every", type=int, default=1)
    args = ap.parse_args()

    torch.manual_seed(cfg.SEED)
    device = get_device()
    train_items, val_items = split_datalist(get_datalist())
    train_loader, val_loader = make_loader(train_items, True), make_loader(val_items, False)
    print(f"device={device}  train={len(train_items)}  val={len(val_items)}")

    model = build_model().to(device)
    loss_fn, opt, scaler = make_loss(), make_optimizer(model, args.lr), make_scaler(device)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    history, best = [], {"dice_mean": -1.0, "epoch": 0}
    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, opt, loss_fn, device, scaler)
        sched.step()
        rec = {"epoch": epoch, "train_loss": tr}
        if epoch % args.val_every == 0 or epoch == args.epochs:
            val = evaluate(model, val_loader, device)
            rec.update({f"val_{k}": v for k, v in val.items()})
            if val["dice_mean"] > best["dice_mean"]:
                best = {"dice_mean": val["dice_mean"], "epoch": epoch}
                torch.save(model.state_dict(), cfg.OUT_DIR / "baseline_best.pt")
        history.append(rec)
        print(json.dumps(rec))
        (cfg.OUT_DIR / "baseline_metrics.json").write_text(
            json.dumps({"best": best, "history": history}, indent=2))
    print(f"Best mean Dice {best['dice_mean']:.4f} at epoch {best['epoch']}")


if __name__ == "__main__":
    main()
