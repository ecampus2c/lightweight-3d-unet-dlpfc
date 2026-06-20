"""Evaluation metrics and LOSO batch evaluation.

Metrics:
    dice_score        Volumetric Dice Similarity Coefficient (overlap).
    centroid_error    Euclidean distance between mask centres of mass (mm).

Both ``dice_score`` and ``centroid_error`` are pure NumPy and unit-tested.
``evaluate_loso`` additionally requires TensorFlow to run inference.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config


def dice_score(true_mask: np.ndarray, pred_mask: np.ndarray, smooth: float = 1e-6) -> float:
    """Volumetric Dice = 2|A n B| / (|A| + |B|) on binary masks."""
    intersection = np.sum(pred_mask * true_mask)
    union = np.sum(pred_mask) + np.sum(true_mask)
    return float((2.0 * intersection) / (union + smooth))


def center_of_mass(mask: np.ndarray):
    """Mean voxel coordinate of the foreground, or None if the mask is empty."""
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return None
    return coords.mean(axis=0)


def centroid_error(true_mask: np.ndarray, pred_mask: np.ndarray,
                   spacing=config.TARGET_SPACING) -> float:
    """Euclidean distance between true and predicted centres of mass (mm).

    Returns NaN if either mask is empty. With 1 mm isotropic preprocessing the
    voxel distance equals the physical distance in millimetres.
    """
    cm_t = center_of_mass(true_mask)
    cm_p = center_of_mass(pred_mask)
    if cm_t is None or cm_p is None:
        return float("nan")
    diff = (cm_t - cm_p) * np.asarray(spacing, dtype=float)
    return float(np.linalg.norm(diff))


def evaluate_loso(preproc_dir=config.PREPROC_DIR, models_dir=config.MODELS_DIR):
    """Run inference with each fold's held-out model and tabulate metrics.

    Returns a list of dicts: {subject, dice, centroid_error_mm}.
    """
    from .inference import sliding_window_inference
    from .model import build_lightweight_unet

    preproc_dir, models_dir = Path(preproc_dir), Path(models_dir)
    results = []
    for npz_path in sorted(preproc_dir.glob("*.npz")):
        subject = npz_path.stem
        model_path = models_dir / f"best_model_{subject}.h5"
        if not model_path.exists():
            print(f"WARNING: model not found for {subject}; skipping.")
            continue

        npz = np.load(npz_path)
        volume, true_mask = npz["image"], npz["label"]

        model = build_lightweight_unet()
        model.load_weights(str(model_path))
        prob = sliding_window_inference(model, volume)
        pred_mask = (prob > config.PROB_THRESHOLD).astype(np.uint8)
        if pred_mask.ndim == 4:
            pred_mask = pred_mask[..., 0]

        results.append({
            "subject": subject,
            "dice": dice_score(true_mask, pred_mask),
            "centroid_error_mm": centroid_error(true_mask, pred_mask),
        })

        import tensorflow as tf
        tf.keras.backend.clear_session()

    return results


def summarize(results) -> str:
    """Format a per-subject metrics table with cohort mean +/- SD."""
    if not results:
        return "No results."
    lines = [f"{'subject':<12}{'dice':>8}{'centroid_mm':>14}", "-" * 34]
    for r in results:
        lines.append(f"{r['subject']:<12}{r['dice']:>8.4f}{r['centroid_error_mm']:>14.2f}")
    dices = np.array([r["dice"] for r in results])
    errs = np.array([r["centroid_error_mm"] for r in results])
    lines.append("-" * 34)
    lines.append(f"{'mean+/-SD':<12}{dices.mean():>8.4f}{np.nanmean(errs):>14.2f}")
    lines.append(f"{'':<12}{dices.std():>8.4f}{np.nanstd(errs):>14.2f}")
    return "\n".join(lines)


if __name__ == "__main__":
    config.set_global_seeds()
    print(summarize(evaluate_loso()))
