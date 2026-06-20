# Release Checklist

Status of publication-readiness checks. ☑ done · ☐ requires author action.

## Code

- ☑ Pipeline runs outside Colab (no `google.colab`, no `!pip` magics)
- ☑ Paths centralised and configurable (`src/config.py`, env vars)
- ☑ Image/mask pairing bug fixed and tested
- ☑ Random seeds set for Python/NumPy/TensorFlow
- ☑ Modules compile; pure-NumPy logic unit-tested (`pytest`: 15 passed, 1 skipped)
- ☐ End-to-end run on a TensorFlow/GPU environment to confirm released weights
  reproduce the reported DSC and centroid error (`python -m src.evaluate`)

## Dependencies

- ☑ `requirements.txt` and `environment.yml` generated from actual imports
- ☐ Confirm the exact TensorFlow version used to train the released `.h5`
  weights and pin it (currently `tensorflow==2.15.0`, flagged for review)

## Data and models

- ☑ `.gitignore` excludes `*.nii`, `*.npz`, `*.h5` (size + patient data)
- ☐ Decide distribution for large artifacts: Git LFS, or an external data
  repository (Zenodo/OSF) with a DOI referenced in `DATASET.md`
- ☐ Confirm dataset provenance, acquisition protocol and ethics/IRB status in
  `DATASET.md` (fields marked **[Author to confirm]**)
- ☐ Confirm the cohort may be shared publicly; if not, release weights + code only

## Documentation

- ☑ `README.md` (overview, structure, install, usage, results, citation)
- ☑ `METHODS.md`, `DATASET.md`, `REPRODUCIBILITY.md`, `PIPELINE_DIAGRAM.md`
- ☑ `PROJECT_AUDIT.md`, `VALIDATION_REPORT.md`, `STRUCTURAL_CHANGES.md`
- ☐ Fill citation block in `README.md` with final thesis/paper details and DOI

## Scientific integrity (address for any journal version)

- ☐ Replace the test-subject validation split in LOSO with a held-out training
  subject and re-report metrics (VALIDATION_REPORT §5.1)
- ☐ Recompute the volume-correlation figure from measured mask volumes rather
  than simulated values (VALIDATION_REPORT §5.2)

## Repository hygiene

- ☑ Consistent naming (`caseN` across raw, preprocessed, models)
- ☑ Clear directory structure (`src/`, `tests/`, `data/`, `models/`, `results/`)
- ☑ Add a `LICENSE` file — MIT (`LICENSE`)
- ☐ Optional: add CI (e.g. GitHub Actions) running `pytest` on push
- ☐ Verify no credentials, tokens or private paths remain (Colab paths removed)
