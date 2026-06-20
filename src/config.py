"""Central configuration: filesystem paths and pipeline hyperparameters.

All paths default to a layout relative to the repository root and can be
overridden with environment variables (useful on HPC or cloud nodes), e.g.

    export DLPFC_DATA_ROOT=/scratch/$USER/dlpfc

Hyperparameters reproduce the values used for the dissertation experiments.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("DLPFC_DATA_ROOT", REPO_ROOT / "data"))

RAW_IMAGES_DIR = Path(os.environ.get("DLPFC_RAW_IMAGES", DATA_ROOT / "raw" / "images"))
RAW_LABELS_DIR = Path(os.environ.get("DLPFC_RAW_LABELS", DATA_ROOT / "raw" / "labels"))
PREPROC_DIR = Path(os.environ.get("DLPFC_PREPROC", DATA_ROOT / "preprocessed"))
MODELS_DIR = Path(os.environ.get("DLPFC_MODELS", REPO_ROOT / "models"))
RESULTS_DIR = Path(os.environ.get("DLPFC_RESULTS", REPO_ROOT / "results" / "figures"))

# --------------------------------------------------------------------------- #
# Hyperparameters (as used for the reported N=10 LOSO experiments)
# --------------------------------------------------------------------------- #
SEED = 42

TARGET_SPACING = (1.0, 1.0, 1.0)        # mm, isotropic resampling target
TARGET_SHAPE = (128, 128, 128)          # voxels, after centre crop/pad
PATCH_SIZE = (96, 96, 96)               # voxels, training/inference patch
STRIDE = 48                             # voxels, sliding-window step (50% overlap)

ENCODER_FILTERS = (16, 32, 64)          # per encoder level
BOTTLENECK_FILTERS = 128

BATCH_SIZE = 1
FOREGROUND_RATIO = 0.6                  # P(sampling a foreground-containing patch)
LEARNING_RATE = 1e-4
EPOCHS = 80
STEPS_PER_EPOCH = 200
VALIDATION_STEPS = 50
PROB_THRESHOLD = 0.5                    # sigmoid -> binary mask


def set_global_seeds(seed: int = SEED) -> None:
    """Seed Python, NumPy and (if available) TensorFlow for reproducibility.

    Note: full bitwise determinism on GPU additionally requires
    ``TF_DETERMINISTIC_OPS=1`` and is not guaranteed across hardware.
    """
    import random
    random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
