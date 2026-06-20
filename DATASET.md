# Dataset

This document describes the dataset as it exists in the repository. Items that
cannot be determined from the repository contents and must be supplied by the
author before public release are marked **[Author to confirm]**.

## Origin and acquisition

- Cohort: 10 subjects, T1-weighted structural MRI.
- Source, scanner, field strength and acquisition parameters: **[Author to
  confirm]** — record the data source (public dataset name and version, or
  acquiring institution) and the acquisition protocol.
- Ethics / governance: **[Author to confirm]** — institutional review board
  approval or data-use agreement reference, and consent/anonymization status.
  Raw MRI is patient-derived and is therefore excluded from version control
  (see "Usage notes").

## Annotation methodology

Each subject has a binary DLPFC mask. Masks were delineated by expert reference
to anatomical landmarks — the middle frontal gyrus, inferior frontal sulcus and
precentral sulcus — consistent with the convention discussed in the dissertation
(Chapter 4). Number of raters and any inter-rater procedure: **[Author to
confirm]**.

## Directory layout and naming

```
data/
├── raw/
│   ├── images/   caseN.nii        (N = 1..10)   T1 volume
│   └── labels/   caseN_seg.nii    (N = 1..10)   binary DLPFC mask
└── preprocessed/ caseN.npz        (N = 1..10)   model-ready tensors
```

Naming convention: an image `caseN.nii` pairs with the mask `caseN_seg.nii` and
the preprocessed file `caseN.npz`. Pairing in code is by exact case identifier
(see `src/preprocessing._label_for_image`).

## Data structure

Raw NIfTI (`.nii`): native voxel grids and spacing vary by subject; geometry is
carried in the NIfTI header/affine and preserved during resampling.

Preprocessed (`.npz`), verified across all 10 subjects:

| Key      | dtype    | shape           | notes |
|----------|----------|-----------------|-------|
| `image`  | float32  | (128, 128, 128) | z-scored over brain voxels |
| `label`  | uint8    | (128, 128, 128) | binary, values in {0, 1} |
| `affine` | float64  | (4, 4)          | original NIfTI affine |

## Class definitions

Single foreground class:

- `1` — DLPFC (target region of interest)
- `0` — background (all other tissue and air)

## Preprocessing

See [METHODS.md](METHODS.md) §2: 1 mm isotropic resampling, centre crop/pad to
128³, z-score normalization over brain voxels. Reproduce with
`python -m src.preprocessing`.

## Usage notes

- Raw `.nii`, preprocessed `.npz` and model `.h5` files are **not** committed to
  Git (`.gitignore`). For distribution, use Git LFS or an external data
  repository and record the access procedure here before release.
- The preprocessed `.npz` files are sufficient to reproduce training, inference
  and all reported metrics without access to the original NIfTI volumes.
- If the cohort cannot be shared for ethical reasons, release the trained
  `models/*.h5` and the preprocessing code so methods remain reproducible on
  equivalent data.
