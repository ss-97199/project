# config.py
import os

DATA_ROOT = os.environ.get("CBIS_DDSM_DATA_ROOT", "./data/cbis_ddsm")
NUM_CLASSES = 2
IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 4

MODEL_NAME = os.environ.get("BC_MODEL_NAME", "tf_efficientnet_b3")
CHECKPOINT_PATH = os.environ.get("BC_CHECKPOINT_PATH", "./runs/best_model.pth")

CLASS_NAMES = ["benign", "malignant"]

USE_TTA = True
TTA_RUNS = 4
MC_DROPOUT = False
MC_RUNS = 8