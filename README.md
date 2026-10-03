---
title: DLPFC Segmentation
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app/app.py
python_version: "3.10"
pinned: false
license: mit
---

# Lightweight 3D U-Net for DLPFC Segmentation in T1-Weighted MRI

[![🤗 Live Demo](https://img.shields.io/badge/%F0%9F%A4%97_Live-Demo-ffd21e?labelColor=4f46e5)](https://huggingface.co/spaces/k12apana/dlpfc-segmentation)
[![Tests](https://github.com/ecampus2c/lightweight-3d-unet-dlpfc/actions/workflows/tests.yml/badge.svg)](https://github.com/ecampus2c/lightweight-3d-unet-dlpfc/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**🤗 Live demo —** upload an MRI and get an instant segmentation:
**<https://huggingface.co/spaces/k12apana/dlpfc-segmentation>**

Automated segmentation and spatial localization of the dorsolateral prefrontal
cortex (DLPFC) from structural brain MRI, intended to support target definition
for repetitive transcranial magnetic stimulation (rTMS) neuronavigation.

This repository accompanies the doctoral research of Kenneth Apana, a PhD
student in Artificial Intelligence and Machine Learning at St. Petersburg
Electrotechnical University "LETI" (dissertation in preparation). It contains
the preprocessing, model, training, inference and evaluation code (TensorFlow/Keras
and PyTorch), the trained leave-one-subject-out (LOSO) models, and the figures
reported in the thesis and the publications listed [below](#publications).

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
│   ├── train.py              LOSO training driver (CLI)
│   ├── torch_model.py        PyTorch port of the 3D U-Net + BCE/Dice loss
│   ├── torch_train.py        LOSO training driver, PyTorch (CLI)
│   └── convert_weights.py    Keras .h5 -> PyTorch .pt weight conversion
├── tests/                    Pure-NumPy unit tests (+ optional TF / PyTorch model tests)
├── app/                      Gradio web demo (upload MRI -> segmentation)
├── notebooks/
│   └── full_pipeline_colab.py  Original Google Colab export (reference record)
├── data/
│   ├── raw/{images,labels}/  T1 volumes and expert masks (.nii) [not in Git]
│   └── preprocessed/         128³ .npz volumes [not in Git]
├── models/                   LOSO model weights, best_model_caseN.h5 [Git LFS]
│   └── pytorch/              The same weights converted to PyTorch, best_model_caseN.pt [Git LFS]
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

### PyTorch

The model is also available in PyTorch, with the trained LOSO weights converted
from Keras (`models/pytorch/best_model_caseN.pt`). Architecture, loss, sampling,
augmentation and training schedule are the same as the Keras pipeline.

```bash
pip install -r requirements.txt -r requirements-torch.txt
git lfs pull                                  # fetch the weights

# Re-train with PyTorch -> models/pytorch/best_model_<id>.pt
python -m src.torch_train --epochs 80

# Or convert Keras weights yourself (reads .h5 with h5py; TensorFlow not needed)
python -m src.convert_weights
```

Inference reuses the existing sliding-window code through a small adapter:

```python
from src.inference import sliding_window_inference
from src.torch_model import TorchPredictor, load_model

model = TorchPredictor(load_model("models/pytorch/best_model_case1.pt"))
prob = sliding_window_inference(model, volume)   # (D, H, W, 1) probabilities
```

**Equivalence with Keras.** All ten converted models were checked against the
original `.h5` models on the same T1 patch (96³): maximum probability
difference 8.3 × 10⁻⁶; thresholded masks identical in nine folds and differing
by a single borderline voxel in the tenth (mask Dice 1.0000).
`tests/test_torch_model.py` repeats the check on a randomly initialised model
when both frameworks are installed.

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

## Publications

1. Apana A., Shichkina Yu. *Patch-Based Lightweight 3D CNNs for Anatomical Brain
   Segmentation Under Severe Data Scarcity: Automated DLPFC Segmentation in
   Structural MRI.* Preprints.org, 2026.
   DOI: [10.20944/preprints202603.0020.v1](https://doi.org/10.20944/preprints202603.0020.v1)
2. Apana K.A., Shichkina Yu.A. *A Method for Mitigating Boundary Truncation
   Artifacts in Sliding-Window 3D Inference for Lightweight Brain Segmentation.*
   Proc. 7th Int. Conf. on Neural Networks and Neurotechnologies (NeuroNT'2026),
   St. Petersburg, 2026, pp. 243–248.
3. Апана К.А., Шичкина Ю.А. Сегментация анатомических структур мозга с помощью
   патчевых легковесных 3D сверточных нейронных сетей в условиях экстремальной
   нехватки данных: автоматическая сегментация DLPFC в структурной МРТ //
   Современная наука: актуальные проблемы теории и практики. Серия: Естественные
   и технические науки. 2026. № 4. С. 55–63.
4. Апана К.А., Шичкина Ю.А. Эффективная обработка малых данных в трёхмерной
   объёмной сегментации нейроизображений: ограничение параметров как структурный
   регуляризатор // Современная наука: актуальные проблемы теории и практики.
   Серия: Естественные и технические науки. 2026. № 5-2. С. 42–47.
5. Апана К.А., Шичкина Ю.А. Алгоритмическое устранение артефактов граничного
   усечения при выводе методом скользящего окна в трёхмерном случае для
   сегментации головного мозга // Перспективы науки. 2026. № 6 (201). С. 106–109.
6. Ayinbuno A. *A Novel Approach for Personalized DLPFC Localization in
   Neuroimaging and Brain Stimulation Using Mask R-CNN.* IEEE SCM 2025,
   pp. 456–462. DOI: [10.1109/SCM66446.2025.11060233](https://doi.org/10.1109/SCM66446.2025.11060233)
   (earlier 2D approach to the same localization task)

## Citation

If you use this code, please cite the preprint (publication 1 above); the
dissertation is in preparation:

```bibtex
@misc{apana2026patch,
  author    = {Apana, Kenneth Ayinbuno and Shichkina, Yulia Aleksandrovna},
  title     = {Patch-Based Lightweight {3D} {CNNs} for Anatomical Brain Segmentation
               Under Severe Data Scarcity: Automated {DLPFC} Segmentation in
               Structural {MRI}},
  publisher = {Preprints.org},
  year      = {2026},
  doi       = {10.20944/preprints202603.0020.v1}
}

@phdthesis{apana_dlpfc_dissertation,
  author = {Apana, Kenneth Ayinbuno},
  title  = {Development of artificial intelligence methods for precise spatial
            localization in volumetric neuroimaging data},
  school = {St. Petersburg Electrotechnical University (LETI)},
  note   = {In preparation}
}
```

## License

Released under the [MIT License](LICENSE), covering the code, documentation and
figures in this repository. Any separately distributed dataset or model weights
remain subject to the data-use and ethics terms described in
[DATASET.md](DATASET.md).
