"""Volume preprocessing: isotropic resampling, centre crop/pad, normalisation.

Pipeline per subject:
    1. Resample to 1 mm isotropic (B-spline for the image, nearest-neighbour
       for the binary mask), preserving direction and origin.
    2. Centre crop or zero-pad to a fixed 128x128x128 grid.
    3. Z-score normalise intensities over non-zero (brain) voxels only.
    4. Save image, label and original affine to a compressed .npz.

``center_crop_or_pad`` and ``normalize_zscore`` are pure NumPy and unit-tested;
``resample_to_isotropic`` and the file-level driver require SimpleITK/nibabel.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config


def center_crop_or_pad(volume: np.ndarray, target_shape=(128, 128, 128)) -> np.ndarray:
    """Centre the volume in a fixed-size grid by symmetric crop and/or zero-pad."""
    out = np.zeros(target_shape, dtype=volume.dtype)
    src_shape = np.array(volume.shape)
    tgt_shape = np.array(target_shape)

    starts = np.maximum((src_shape - tgt_shape) // 2, 0)        # crop offset in source
    dst_starts = np.maximum((tgt_shape - src_shape) // 2, 0)    # pad offset in target
    copy_extent = np.minimum(tgt_shape, src_shape)

    zs, ys, xs = starts
    zd, yd, xd = dst_starts
    cz, cy, cx = copy_extent
    out[zd:zd + cz, yd:yd + cy, xd:xd + cx] = volume[zs:zs + cz, ys:ys + cy, xs:xs + cx]
    return out


def normalize_zscore(volume: np.ndarray) -> np.ndarray:
    """Z-score normalise over non-zero voxels; background stays zero."""
    brain_mask = volume > 0
    out = np.zeros_like(volume, dtype=np.float32)
    if np.any(brain_mask):
        mean_val = volume[brain_mask].mean()
        std_val = volume[brain_mask].std() + 1e-8
        out[brain_mask] = (volume[brain_mask] - mean_val) / std_val
    else:
        out = volume.astype(np.float32)
    return out


def resample_to_isotropic(image_path, is_mask: bool = False) -> np.ndarray:
    """Resample a NIfTI file to 1 mm isotropic spacing (SimpleITK)."""
    import SimpleITK as sitk

    sitk_img = sitk.ReadImage(str(image_path))
    orig_spacing = sitk_img.GetSpacing()
    orig_size = sitk_img.GetSize()
    target_spacing = config.TARGET_SPACING

    new_size = [
        int(np.round(orig_size[i] * (orig_spacing[i] / target_spacing[i])))
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetOutputDirection(sitk_img.GetDirection())
    resampler.SetOutputOrigin(sitk_img.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(
        sitk.sitkNearestNeighbor if is_mask else sitk.sitkBSpline
    )
    return sitk.GetArrayFromImage(resampler.Execute(sitk_img))


def preprocess_image_file(image_path):
    """Preprocess one raw T1 NIfTI for inference.

    Applies the same steps as the training pipeline (resample to 1 mm isotropic,
    centre crop/pad to 128^3, z-score over brain voxels) and returns the
    model-ready volume together with the original NIfTI affine.

    Returns
    -------
    (np.ndarray, np.ndarray)
        The (128, 128, 128) float32 volume and the 4x4 affine of the input.
    """
    import nibabel as nib

    affine = nib.load(str(image_path)).affine
    img_iso = resample_to_isotropic(image_path, is_mask=False)
    img_padded = center_crop_or_pad(img_iso, config.TARGET_SHAPE)
    return normalize_zscore(img_padded), affine


def _label_for_image(img_name: str, label_files):
    """Match an image to its mask by exact stem (caseN -> caseN_seg).

    Uses exact leading-token comparison rather than substring containment, so
    ``case1`` is not mismatched to ``case10_seg`` (a bug in the original loop).
    """
    img_stem = img_name.split(".")[0]
    for lbl in label_files:
        lbl_stem = lbl.split(".")[0]
        if lbl_stem in (img_stem, f"{img_stem}_seg"):
            return lbl
    for lbl in label_files:                      # fall back: exact token before '_'
        if lbl.split(".")[0].split("_")[0] == img_stem:
            return lbl
    return None


def preprocess_dataset(
    images_dir=config.RAW_IMAGES_DIR,
    labels_dir=config.RAW_LABELS_DIR,
    out_dir=config.PREPROC_DIR,
) -> int:
    """Preprocess every image/mask pair into ``out_dir`` as ``<stem>.npz``.

    Returns the number of subjects written.
    """
    import nibabel as nib
    from tqdm import tqdm

    images_dir, labels_dir, out_dir = Path(images_dir), Path(labels_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = (".nii", ".nii.gz")
    image_files = sorted(f.name for f in images_dir.iterdir() if f.name.endswith(exts))
    label_files = sorted(f.name for f in labels_dir.iterdir() if f.name.endswith(exts))

    written = 0
    for img_name in tqdm(image_files, desc="Preprocessing"):
        lbl_name = _label_for_image(img_name, label_files)
        if lbl_name is None:
            print(f"WARNING: no matching mask for {img_name}; skipping.")
            continue

        img_path = images_dir / img_name
        lbl_path = labels_dir / lbl_name
        affine = nib.load(str(img_path)).affine

        img_iso = resample_to_isotropic(img_path, is_mask=False)
        lbl_iso = resample_to_isotropic(lbl_path, is_mask=True)

        img_padded = center_crop_or_pad(img_iso, config.TARGET_SHAPE)
        lbl_padded = center_crop_or_pad(lbl_iso, config.TARGET_SHAPE)
        img_norm = normalize_zscore(img_padded)

        np.savez_compressed(
            out_dir / f"{img_name.split('.')[0]}.npz",
            image=img_norm.astype(np.float32),
            label=lbl_padded.astype(np.uint8),
            affine=affine,
        )
        written += 1

    print(f"Preprocessing complete: {written} subjects -> {out_dir}")
    return written


if __name__ == "__main__":
    preprocess_dataset()
