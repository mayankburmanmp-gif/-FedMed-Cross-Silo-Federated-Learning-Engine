"""3D U-Net + helpers to move weights between PyTorch and Flower (NumPy lists)."""
from collections import OrderedDict

import numpy as np
import torch
from monai.networks.layers import Norm
from monai.networks.nets import UNet

import config as cfg


def build_model():
    # InstanceNorm keeps no running statistics, so state_dict == trainable weights only.
    # That makes FedAvg well-defined (no BatchNorm buffer mismatch across hospitals).
    return UNet(spatial_dims=3, in_channels=cfg.IN_CHANNELS, out_channels=cfg.OUT_CHANNELS,
                channels=cfg.UNET_CHANNELS, strides=cfg.UNET_STRIDES,
                num_res_units=2, norm=Norm.INSTANCE)


def get_weights(model):
    return [v.detach().cpu().numpy() for v in model.state_dict().values()]


def set_weights(model, weights):
    keys = model.state_dict().keys()
    state = OrderedDict((k, torch.as_tensor(np.asarray(w))) for k, w in zip(keys, weights))
    model.load_state_dict(state, strict=True)
