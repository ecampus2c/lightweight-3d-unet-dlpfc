"""Loading of preprocessed volumes and the class-balanced patch generator."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config


def load_preprocessed(preproc_dir=config.PREPROC_DIR):
    """Load every ``.npz`` subject into memory.

    Returns a list of dicts with keys ``filename``, ``image`` (float32,
    128x128x128) and ``label`` (uint8, 128x128x128), sorted by filename.
    """
    preproc_dir = Path(preproc_dir)
    files = sorted(preproc_dir.glob("*.npz"))
    data = []
    for f in files:
        npz = np.load(f)
        data.append({"filename": str(f), "image": npz["image"], "label": npz["label"]})
    return data


def ram_balanced_generator(
    loaded_data,
    batch_size: int = config.BATCH_SIZE,
    patch_size=config.PATCH_SIZE,
    fg_ratio: float = config.FOREGROUND_RATIO,
):
    """Yield class-balanced augmented patches from in-memory volumes.

    With probability ``fg_ratio`` a patch is required to contain foreground;
    augmentation is random axis flips and a random 90-degree in-plane rotation.
    Designed to be passed directly to ``tf.keras.Model.fit``.
    """
    while True:
        np.random.shuffle(loaded_data)
        for patient_data in loaded_data:
            vol = patient_data["image"]
            mask = patient_data["label"]

            batch_x, batch_y = [], []
            while len(batch_x) < batch_size:
                require_foreground = np.random.rand() < fg_ratio
                valid_patch_found = False

                for _ in range(50):
                    z = np.random.randint(0, vol.shape[0] - patch_size[0])
                    y = np.random.randint(0, vol.shape[1] - patch_size[1])
                    x = np.random.randint(0, vol.shape[2] - patch_size[2])

                    p_mask = mask[z:z + patch_size[0], y:y + patch_size[1], x:x + patch_size[2]]

                    if (require_foreground and np.sum(p_mask) > 0) or (not require_foreground):
                        p_vol = vol[z:z + patch_size[0], y:y + patch_size[1], x:x + patch_size[2]]

                        if np.random.rand() < 0.5:
                            p_vol, p_mask = p_vol[::-1, :, :], p_mask[::-1, :, :]
                        if np.random.rand() < 0.5:
                            p_vol, p_mask = p_vol[:, ::-1, :], p_mask[:, ::-1, :]
                        if np.random.rand() < 0.5:
                            p_vol, p_mask = p_vol[:, :, ::-1], p_mask[:, :, ::-1]

                        k = np.random.randint(0, 4)
                        p_vol = np.rot90(p_vol, k, axes=(1, 2))
                        p_mask = np.rot90(p_mask, k, axes=(1, 2))

                        batch_x.append(p_vol)
                        batch_y.append(p_mask)
                        valid_patch_found = True
                        break

                if not valid_patch_found:
                    batch_x.append(vol[z:z + patch_size[0], y:y + patch_size[1], x:x + patch_size[2]])
                    batch_y.append(p_mask)

            yield np.array(batch_x)[..., np.newaxis], np.array(batch_y)[..., np.newaxis]
