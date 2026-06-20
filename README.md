# Lightweight 3D U-Net for DLPFC Segmentation in T1-Weighted MRI

Automated segmentation and spatial localization of the dorsolateral prefrontal
cortex (DLPFC) from structural brain MRI, intended to support target definition
for repetitive transcranial magnetic stimulation (rTMS) neuronavigation.

This repository accompanies the doctoral dissertation of K. A. Apana
(St. Petersburg Electrotechnical University "LETI"). It contains the
preprocessing, model, training, inference and evaluation code, the trained
leave-one-subject-out (LOSO) models, and the figures reported in the thesis.

## Research objectives

1. Segment the DLPFC from T1-weighted MRI under a small-data regime (N = 10)
   without transfer learning or large-scale pretraining.
2. Constrain model capacity (a lightweight 3D U-Net, ~1.4M parameters) as a
   structural regularizer to mitigate overfitting on limited data.
3. Recover a single physical target coordinate (mask centre of mass) suitable
   for TMS neuronavigation, and quantify spatial targeting error.

## Methodology summary

T1 volumes are resampled to 1 mm isotropic spacing, centre cropped/padded to a
128×128×128 grid, and intensity-normalised (z-score over brain voxels). A
lightweight 3D U-Net is trained on class-balanced 96×96×96 patches with a
combined binary cross-entropy + soft-Dice loss. Models are evaluated by
leave-one-subject-out cross-validation; full volumes are segmented by
overlapping sliding-window inference and assessed with the Dice similarity
coefficient (DSC) and centroid (centre-of-mass) error. See
[METHODS.md](METHODS.md) for details.

## Repository structure

```
.
├── src/                      Pipeline package (importable as `src`)
│   ├── config.py             Paths and hyperparameters (single source of truth)
│   ├── preprocessing.py      Resampling, crop/pad, normalization
│   ├── model.py              3D U-Net + BCE/Dice loss
│   ├── data.py               .npz loading + class-balanced patch generator
│   ├── inference.py          Sliding-window full-volume inference
│   ├── evaluate.py           DSC + centroid-error metrics, LOSO evaluation
│   └── train.py              LOSO training driver (CLI)
├── tests/                    Pure-NumPy unit tests (+ optional TF model tests)
├── app/                      Gradio web demo (upload MRI -> segmentation)
├── notebooks/
│   └── full_pipeline_colab.py  Original Google Colab export (reference record)
├── data/
│   ├── raw/{images,labels}/  T1 volumes and expert masks (.nii) [not in Git]
│   └── preprocessed/         128³ .npz volumes [not in Git]
├── models/                   LOSO model weights, best_model_caseN.h5 [not in Git]
├── results/figures/          Per-subject overlays and analysis figures (.png)
├── requirements.txt, environment.yml
└── *.md                      Documentation (see below)
```

Large binary artifacts (`*.nii`, `*.npz`, `*.h5`) are excluded from version
control; see [DATASET.md](DATASET.md) and [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Dataset

Ten T1-weighted MRI subjects with expert DLPFC masks delineated from anatomical
landmarks (middle frontal gyrus, inferior frontal sulcus, precentral sulcus).
Provenance, acquisition and ethics are documented in [DATASET.md](DATASET.md).

## Installation

```bash
conda env create -f environment.yml
conda activate dlpfc-seg
# or, with pip in a Python 3.10 environment:
pip install -r requirements.txt
```

## Usage

```bash
# 1. Preprocess raw NIfTI volumes -> data/preprocessed/*.npz
python -m src.preprocessing

# 2. Leave-one-subject-out training -> models/best_model_<id>.h5
python -m src.train --epochs 80

# 3. Evaluate all folds (Dice + centroid error)
python -m src.evaluate
```

Paths and hyperparameters are centralised in `src/config.py` and can be
overridden with environment variables (e.g. `DLPFC_DATA_ROOT`).

### Web demo

An interactive [Gradio](https://www.gradio.app) app accepts a T1 volume and
returns the predicted DLPFC segmentation:

```bash
pip install -r app/requirements.txt
git lfs pull            # fetch the model weights (models/*.h5)
python app/app.py       # then open the printed local URL
```

It produces an overlay, the target centroid, and a downloadable mask. See
[app/README.md](app/README.md) for configuration, a temporary public link, and
Hugging Face Spaces deployment. Research/educational use only — not a medical
device.

## Experimental workflow

`preprocess → LOSO train (N folds) → sliding-window inference → metrics`.
The full workflow is described in [PIPELINE_DIAGRAM.md](PIPELINE_DIAGRAM.md) and
reproduction is covered in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Results overview

Leave-one-subject-out cross-validation (N = 10), volumetric DSC per subject:

| Subset | Mean DSC | Centroid error |
|--------|----------|----------------|
| All subjects (N = 10) | 0.839 ± 0.248 | 5.64 ± 3.21 mm |
| Canonical anatomy (N = 8) | 0.961 ± 0.007 | 4.31 ± 1.18 mm |

Two subjects (cases 2 and 4) show reduced overlap attributable to annotation
convention rather than model failure; this is analysed in the dissertation and
summarised in [VALIDATION_REPORT.md](VALIDATION_REPORT.md). Per-subject overlay
figures are in `results/figures/`.

## Documentation

- [METHODS.md](METHODS.md) — methodology, architecture, metrics
- [DATASET.md](DATASET.md) — data origin, structure, ethics
- [REPRODUCIBILITY.md](REPRODUCIBILITY.md) — environment and reproduction steps
- [PIPELINE_DIAGRAM.md](PIPELINE_DIAGRAM.md) — end-to-end workflow
- [PROJECT_AUDIT.md](PROJECT_AUDIT.md) — repository audit and issue tracking
- [VALIDATION_REPORT.md](VALIDATION_REPORT.md) — verification of code vs. results
- [STRUCTURAL_CHANGES.md](STRUCTURAL_CHANGES.md) — reorganization record
- [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md) — pre-publication checks

## Citation

If you use this code, please cite the dissertation:

```bibtex
@phdthesis{apana2026dlpfc,
  author = {Apana, Kenneth Ayinbuno},
  title  = {Development of artificial intelligence methods for precise spatial
            localization in volumetric neuroimaging data},
  school = {St. Petersburg Electrotechnical University (LETI)},
  year   = {2026}
}
```

## License

Released under the [MIT License](LICENSE), covering the code, documentation and
figures in this repository. Any separately distributed dataset or model weights
remain subject to the data-use and ethics terms described in
[DATASET.md](DATASET.md).
