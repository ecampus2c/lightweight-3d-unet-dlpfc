# Reproducibility

Instructions for a researcher reproducing this work for the first time. Two
paths are supported: (A) reproduce the reported metrics from the released
weights, and (B) retrain from scratch.

## 1. Environment

- Python 3.10. A CUDA-capable GPU is recommended for training and for full-volume
  inference; CPU is sufficient to run the unit tests and the metric functions.
- The released `.h5` weights were trained with TensorFlow 2.x on Google Colab.
  The pinned `tensorflow==2.15.0` is a tested, NumPy<2-compatible choice; if you
  know the exact training version, set it in `requirements.txt`/`environment.yml`.

```bash
conda env create -f environment.yml
conda activate dlpfc-seg
# or:  pip install -r requirements.txt   (in a fresh Python 3.10 venv)
```

## 2. Data preparation

Place raw volumes under `data/raw/images/caseN.nii` and masks under
`data/raw/labels/caseN_seg.nii` (see [DATASET.md](DATASET.md)). Then:

```bash
python -m src.preprocessing      # writes data/preprocessed/caseN.npz
```

The repository already includes the preprocessed `.npz` files used for the
reported results (when distributed; they are git-ignored), so this step can be
skipped for path A.

## 3A. Reproduce reported metrics (no retraining)

With `models/best_model_caseN.h5` and `data/preprocessed/caseN.npz` present:

```bash
python -m src.evaluate
```

This loads each fold's held-out model, runs sliding-window inference on its
subject, and prints per-subject DSC and centroid error with the cohort
mean ± SD. Expected (N = 10): mean DSC ≈ 0.84 (≈ 0.96 on the 8 canonical-anatomy
subjects); mean centroid error ≈ 5.6 mm (≈ 4.3 mm canonical).

## 3B. Retrain from scratch

```bash
python -m src.train --epochs 80 --seed 42
```

Performs LOSO training and writes fresh weights to `models/`. Runtime is
dominated by N × 80 epochs of 3D convolutions; budget several hours on a single
modern GPU. Hyperparameters are in `src/config.py`.

## 4. Tests

```bash
pip install pytest
pytest -q
```

Unit tests cover preprocessing (crop/pad, normalization, mask matching), the
metrics (DSC, centroid error) and dataset integrity. Model-architecture tests
run only if TensorFlow is installed (otherwise skipped).

## 5. Determinism

`src/config.set_global_seeds` seeds Python, NumPy and TensorFlow. Exact bitwise
reproducibility of GPU training additionally requires `TF_DETERMINISTIC_OPS=1`
and identical hardware and library versions; without these, retrained metrics
may vary by small margins from the released weights.

## 6. Known limitations

- **Small cohort (N = 10).** Results are reported under leave-one-subject-out
  cross-validation; generalisation beyond this cohort is not established.
- **Validation split.** Checkpoint selection and early stopping monitor the
  held-out subject (see [VALIDATION_REPORT.md](VALIDATION_REPORT.md)); this is
  an optimistic-bias risk preserved here for faithful reproduction.
- **Exact training versions.** The original Colab TensorFlow/CUDA versions are
  not fully pinned; retraining environments may differ slightly.
- **Hardware.** Full-volume sliding-window inference is memory-intensive; reduce
  batch size in `src/inference.py` if GPU memory is limited.
