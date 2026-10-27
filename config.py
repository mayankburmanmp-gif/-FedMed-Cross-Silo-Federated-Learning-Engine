"""Central configuration for FedMed (Weeks 1-2). Override via environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = Path(os.getenv("FEDMED_DATA_DIR", ROOT / "data"))
OUT_DIR = Path(os.getenv("FEDMED_OUT_DIR", ROOT / "outputs"))
CERT_DIR = ROOT / "certs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True,exist_ok=True)

# FEDMED_SYNTHETIC=1 -> tiny random "tumour" volumes, for smoke-testing the pipeline
# on a laptop without downloading BraTS.
SYNTHETIC = os.getenv("FEDMED_SYNTHETIC", "0") == "1"
SYNTHETIC_SAMPLES = 24

SEED = 42
NUM_NODES = 3
VAL_FRACTION = 0.2
NUM_WORKERS = int(os.getenv("FEDMED_WORKERS", "2"))

# Model (3D U-Net). BraTS MSD Task01: 4 MRI modalities in, 3 nested regions out (TC, WT, ET).
IN_CHANNELS = 4
OUT_CHANNELS = 3
ROI_SIZE = (32, 32, 32) if SYNTHETIC else (96, 96, 96)
UNET_CHANNELS = (8, 16, 32, 64) if SYNTHETIC else (16, 32, 64, 128, 256)
UNET_STRIDES = (2, 2, 2) if SYNTHETIC else (2, 2, 2, 2)

BATCH_SIZE = 2
LR = 1e-3

# Federated settings
SERVER_ADDRESS = "127.0.0.1:8080"
STATUS_BASE_PORT = 8081          # hospital node i exposes a status endpoint on 8081 + i
NUM_ROUNDS = 10
LOCAL_EPOCHS = 2
