"""Full-volume segmentation by sliding-window patch aggregation.

The volume is dynamically zero-padded so its extent is at least the patch size
and the remainder is an exact multiple of the stride; overlapping patch
predictions are reassembled and the result is cropped back to the original
dimensions. This removes the boundary artefact ("1-patch bug") discussed in
the dissertation.
"""
from __future__ import annotations

import numpy as np
from patchify import patchify, unpatchify

from . import config


def sliding_window_inference(model, full_volume, patch_size=config.PATCH_SIZE,
                             stride: int = config.STRIDE):
    """Return a probability map (D, H, W, 1) for ``full_volume`` (D, H, W)."""
    # 1. Dynamic padding so size >= patch_size and remainder is a stride multiple
    pad_width = []
    for i in range(3):
        dim_size = full_volume.shape[i]
        p_size = patch_size[i]
        if dim_size < p_size:
            pad_amount = p_size - dim_size
        else:
            remainder = (dim_size - p_size) % stride
            pad_amount = 0 if remainder == 0 else stride - remainder
        pad_width.append((0, pad_amount))

    padded = np.pad(full_volume, pad_width, mode="constant", constant_values=0)

    # 2. Patch extraction and batched prediction
    patches = patchify(padded, patch_size, step=stride)
    patches_reshaped = patches.reshape(-1, *patch_size, 1)
    preds = model.predict(patches_reshaped, batch_size=8, verbose=0)

    # 3. Squeeze channel, reshape to the patch grid, reassemble
    preds_squeezed = preds.squeeze(axis=-1)
    preds_grid = preds_squeezed.reshape(patches.shape)
    prob_padded = unpatchify(preds_grid, padded.shape)

    # 4. Crop back to original dimensions and restore channel axis
    slices = tuple(slice(0, orig_dim) for orig_dim in full_volume.shape)
    return prob_padded[slices][..., np.newaxis]
