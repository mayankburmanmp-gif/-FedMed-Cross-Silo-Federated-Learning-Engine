"""Data loading, train/val split and per-hospital partitioning."""
import random

import torch
from monai import transforms as T
from monai.apps import DecathlonDataset
from monai.data import DataLoader, Dataset, load_decathlon_datalist
from torch.utils.data import Dataset as TorchDataset

import config as cfg

TASK = "Task01_BrainTumour"  # Medical Segmentation Decathlon BraTS (484 labelled cases)


def download_brats():
    """One-off download (~7 GB) of MSD Task01 BrainTumour."""
    DecathlonDataset(root_dir=str(cfg.DATA_DIR), task=TASK, section="training",
                     transform=(), download=True, cache_rate=0.0)


def get_datalist():
    if cfg.SYNTHETIC:
        return [{"id": i} for i in range(cfg.SYNTHETIC_SAMPLES)]
    json_path = cfg.DATA_DIR / TASK / "dataset.json"
    if not json_path.exists():
        raise FileNotFoundError(f"{json_path} not found. Run: python download_data.py")
    return load_decathlon_datalist(str(json_path), is_segmentation=True, data_list_key="training")


def split_datalist(datalist):
    """Deterministic global train/val split, identical on every process."""
    items = sorted(datalist, key=lambda d: str(d.get("image", d.get("id"))))
    random.Random(cfg.SEED).shuffle(items)
    n_val = max(cfg.NUM_NODES, int(len(items) * cfg.VAL_FRACTION))
    return items[n_val:], items[:n_val]


def partition(items, num_nodes, node_id):
    """IID round-robin shard of an already-shuffled list: hospital `node_id` of `num_nodes`."""
    return items[node_id::num_nodes]


def train_transforms():
    keys = ["image", "label"]
    return T.Compose([
        T.LoadImaged(keys=keys),
        T.EnsureChannelFirstd(keys="image"),
        T.EnsureTyped(keys=keys),
        T.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        T.Orientationd(keys=keys, axcodes="RAS"),
        T.RandSpatialCropd(keys=keys, roi_size=cfg.ROI_SIZE, random_size=False),
        T.RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
        T.RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
        T.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        T.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        T.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
    ])


def val_transforms():
    keys = ["image", "label"]
    return T.Compose([
        T.LoadImaged(keys=keys),
        T.EnsureChannelFirstd(keys="image"),
        T.EnsureTyped(keys=keys),
        T.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        T.Orientationd(keys=keys, axcodes="RAS"),
        T.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ])


class SyntheticBrats(TorchDataset):
    """Random 4-channel volumes with a nested spherical 'tumour' (ET inside TC inside WT)."""

    SIZE = 64

    def __init__(self, items, train):
        self.items, self.train = items, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        s = self.SIZE
        g = torch.Generator().manual_seed(int(self.items[i]["id"]))
        img = torch.randn(4, s, s, s, generator=g)
        grid = torch.stack(torch.meshgrid(*[torch.arange(s)] * 3, indexing="ij")).float()
        centre = torch.randint(16, s - 16, (3, 1, 1, 1), generator=g).float()
        radius = float(torch.randint(7, 13, (1,), generator=g))
        dist = ((grid - centre) ** 2).sum(0).sqrt()
        wt, tc, et = (dist < radius).float(), (dist < radius * 0.66).float(), (dist < radius * 0.33).float()
        img = img + 2.0 * wt.unsqueeze(0)
        label = torch.stack([tc, wt, et])
        if self.train:
            r = cfg.ROI_SIZE[0]
            o = torch.randint(0, s - r + 1, (3,)).tolist()
            sl = (slice(None),) + tuple(slice(k, k + r) for k in o)
            img, label = img[sl], label[sl]
        return {"image": img, "label": label}


def make_loader(items, train):
    if cfg.SYNTHETIC:
        ds = SyntheticBrats(items, train)
    else:
        ds = Dataset(items, transform=train_transforms() if train else val_transforms())
    return DataLoader(ds, batch_size=cfg.BATCH_SIZE if train else 1, shuffle=train,
                      num_workers=cfg.NUM_WORKERS, persistent_workers=cfg.NUM_WORKERS > 0)
