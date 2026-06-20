"""Lightweight 3D U-Net pipeline for DLPFC segmentation in T1-weighted MRI.

Modules:
    config         Paths and hyperparameters (single source of truth).
    preprocessing  Isotropic resampling, crop/pad, intensity normalisation.
    model          Lightweight 3D U-Net and the composite BCE+Dice loss.
    data           Preprocessed-volume loading and the patch generator.
    inference      Sliding-window full-volume inference.
    evaluate       Dice coefficient and centroid-error metrics.
    train          Leave-One-Subject-Out (LOSO) training driver.
"""
__version__ = "1.0.0"
