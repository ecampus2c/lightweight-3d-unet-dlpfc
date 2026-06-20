# Structural Changes

Record of the repository reorganization performed during preparation for
release. The goal was a layout that separates source, data, models, results and
documentation, and that makes the pipeline runnable as a package — not maximal
restructuring.

## Original structure

```
Dissertation Folder/
├── full_pipeline.py              # 2151-line Colab export (all stages)
├── dataset/{images,labels}/*.nii # raw T1 volumes and masks
├── preproc/*.npz                 # preprocessed volumes
├── models/*.h5                   # LOSO model weights
├── visual_case*.png              # per-subject result figures (root)
├── slide9_real_targeting_comparison.png, volume_outlier_analysis.png
└── *.docx                        # dissertation documents (later moved out by author)
```

## New structure

```
Dissertation Folder/
├── src/                          # importable pipeline package
├── tests/                        # unit tests
├── notebooks/full_pipeline_colab.py   # original export, preserved
├── data/raw/{images,labels}/     # was dataset/
├── data/preprocessed/            # was preproc/
├── models/                       # unchanged
├── results/figures/              # was loose *.png at root
├── requirements.txt, environment.yml, .gitignore, conftest.py
└── README.md, METHODS.md, DATASET.md, REPRODUCIBILITY.md,
    PIPELINE_DIAGRAM.md, PROJECT_AUDIT.md, VALIDATION_REPORT.md,
    RELEASE_CHECKLIST.md, STRUCTURAL_CHANGES.md
```

## Files moved

| From | To | Rationale |
|------|----|-----------|
| `full_pipeline.py` | `notebooks/full_pipeline_colab.py` | Preserve the original record; the runnable implementation now lives in `src/` |
| `dataset/images/*.nii` | `data/raw/images/*.nii` | Group raw inputs under a single `data/` root |
| `dataset/labels/*.nii` | `data/raw/labels/*.nii` | As above |
| `preproc/*.npz` | `data/preprocessed/*.npz` | Co-locate derived data with raw under `data/` |
| `visual_case*.png` | `results/figures/` | Separate generated outputs from source |
| `slide9_real_targeting_comparison.png` | `results/figures/` | As above |
| `volume_outlier_analysis.png` | `results/figures/` | As above |

## Files renamed

- `full_pipeline.py` → `full_pipeline_colab.py` (clarifies it is the original
  Colab export, distinct from the `src/` package).

## Files created

- `src/`: `__init__.py`, `config.py`, `preprocessing.py`, `model.py`, `data.py`,
  `inference.py`, `evaluate.py`, `train.py`.
- `tests/`: `__init__.py`, `test_preprocessing.py`, `test_data.py`,
  `test_evaluate.py`, `test_model.py`; `conftest.py` at root.
- Manifests: `requirements.txt`, `environment.yml`, `.gitignore`.
- Documentation: the `.md` files listed above.

## Removed

- The empty `dataset/` and `preproc/` directories (after their contents moved).
- The dissertation `.docx` files are no longer in this repository; the author
  relocated them to a separate "Conference Paper" working folder. This
  repository is therefore code/data/results only, with the thesis maintained
  separately.

## Expected benefits

- The pipeline runs outside Colab as a documented package (`python -m src.*`).
- Clear separation of source, data, models, results and documentation aids
  navigation and review.
- Centralised configuration and a dependency manifest improve reproducibility.
- The original implementation remains available verbatim for provenance.
