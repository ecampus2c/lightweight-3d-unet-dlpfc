# Methods

This document describes the methodology implemented in `src/`. It is consistent
with the dissertation (Chapters 2–4) and with the original Colab export retained
in `notebooks/full_pipeline_colab.py`.

## 1. Problem formulation

DLPFC localization is treated as binary semantic segmentation of a T1-weighted
volume followed by reduction of the predicted mask to a single physical target
coordinate (its centre of mass) for TMS neuronavigation. The design constraint
is a small training cohort (N = 10), addressed by limiting model capacity rather
than by transfer learning or large-scale pretraining.

## 2. Preprocessing (`src/preprocessing.py`)

Per subject:

1. **Isotropic resampling** to 1.0 × 1.0 × 1.0 mm. The image uses B-spline
   interpolation; the binary mask uses nearest-neighbour interpolation to
   preserve label values. Output spacing, direction and origin are set
   explicitly so geometry is preserved (`SimpleITK.ResampleImageFilter`).
2. **Centre crop / zero-pad** to a fixed 128 × 128 × 128 grid, centring the
   content symmetrically.
3. **Intensity normalization** by z-score computed over non-zero (brain)
   voxels only; background voxels remain zero.
4. The normalized image (`float32`), mask (`uint8`) and the original NIfTI
   affine (`float64`, 4×4) are saved to a compressed `.npz`.

## 3. Model architecture (`src/model.py`)

A lightweight 3D U-Net (`build_lightweight_unet`):

- Encoder: three levels with two 3×3×3 Conv3D (ReLU) blocks each and 2×2×2
  max pooling; channel widths 16 → 32 → 64.
- Bottleneck: two Conv3D blocks, 128 channels.
- Decoder: three levels of 2×2×2 transposed convolution, skip concatenation,
  and two Conv3D blocks; widths 64 → 32 → 16.
- Output: 1×1×1 Conv3D with sigmoid activation.

Approximately 1.4 million trainable parameters. The deliberately small capacity
acts as a structural regularizer for the small-data setting.

## 4. Loss and optimization

Composite loss `bce_dice_loss = BCE + (1 − soft_Dice)` (`src/model.py`), with
the soft Dice coefficient computed with a 1e-6 smoothing term. Optimizer: Adam,
learning rate 1e-4. The soft Dice term is differentiable and stabilises
training under class imbalance, where the foreground (DLPFC) is a small fraction
of the volume.

## 5. Patch sampling and augmentation (`src/data.py`)

Training operates on 96 × 96 × 96 patches drawn by a class-balanced generator.
With probability 0.6 a patch is required to contain foreground voxels (up to 50
rejection-sampling attempts); otherwise a random patch is accepted.
Augmentation applies independent random flips along each axis and a random
90° in-plane rotation. Volumes are held in RAM for throughput.

## 6. Training procedure (`src/train.py`)

Leave-one-subject-out (LOSO) cross-validation over the N subjects. For each
fold a fresh model is trained from random initialisation on the N−1 training
subjects:

- steps/epoch = 200, epochs = 80, batch size = 1;
- callbacks: `ModelCheckpoint` (best `val_dice_coef`), `ReduceLROnPlateau`
  (factor 0.5, patience 8), `EarlyStopping` (patience 20, restore best weights);
- weights saved to `models/best_model_<subject>.h5`.

Random seeds for Python/NumPy/TensorFlow are set via `config.set_global_seeds`.
A methodological caveat regarding the validation split is documented in
[VALIDATION_REPORT.md](VALIDATION_REPORT.md).

## 7. Inference (`src/inference.py`)

Full volumes are segmented by overlapping sliding-window aggregation
(`sliding_window_inference`): the volume is zero-padded so its extent is at
least the patch size and the remainder is an exact multiple of the stride
(96, stride 48 = 50% overlap); patch probabilities are reassembled and cropped
back to the original dimensions. The probability map is thresholded at 0.5 to a
binary mask. Dynamic padding guarantees full-volume coverage and removes the
boundary artefact ("1-patch bug") described in the dissertation.

## 8. Evaluation metrics (`src/evaluate.py`)

- **Dice similarity coefficient (DSC):** volumetric overlap,
  `2|A∩B| / (|A|+|B|)`, on binary masks.
- **Centroid error:** Euclidean distance between the centres of mass of the
  predicted and ground-truth masks. With 1 mm isotropic preprocessing the voxel
  distance equals the physical distance in millimetres; this is the clinically
  relevant TMS targeting error.

`evaluate_loso` runs each fold's held-out model over its subject and tabulates
both metrics; `summarize` reports per-subject values and the cohort mean ± SD.

## 9. Reproducibility notes

- All hyperparameters are in `src/config.py`.
- Stochastic components (weight initialisation, patch sampling, augmentation)
  are seeded; exact bitwise reproducibility on GPU additionally requires
  `TF_DETERMINISTIC_OPS=1` and identical hardware/library versions.
- The released `models/*.h5` weights reproduce the reported metrics without
  retraining; see [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
