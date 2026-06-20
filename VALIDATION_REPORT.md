# Validation Report

Verification of internal consistency between the source code, the released
artifacts and the reported results, and of the correctness of the refactored
pipeline.

## 1. Validation environment

Validation was performed with NumPy 2.x and matplotlib available; TensorFlow,
SimpleITK and nibabel were **not** installed in the validation environment.
Consequently:

- Pure-NumPy logic (preprocessing utilities, metrics, data loading, the patch
  generator) was executed and unit-tested.
- TensorFlow-dependent code (`model.py`, `train.py`, `inference.py`) was
  syntax-checked and ported verbatim from the original, but **not executed**
  here. It must be exercised on a TensorFlow/GPU environment before release
  (see Remaining concerns).

## 2. Validation steps and outcomes

| Step | Method | Result |
|------|--------|--------|
| Module syntax | `python -m py_compile src/*.py` | All modules compile |
| Importability | Import `config`, `preprocessing`, `data`, `evaluate` | OK |
| Unit tests | `pytest -q` | 15 passed, 1 skipped (TF tests) |
| Dataset integrity | `test_loaded_dataset_integrity` over all 10 `.npz` | All shape (128³), dtypes (float32/uint8), labels ∈ {0,1} |
| Metric correctness | DSC and centroid-error tests (identity, disjoint, half-overlap, known shift, empty) | All pass |
| Mask-matching fix | `test_label_matching_no_substring_collision` | `case1`→`case1_seg`, `case10`→`case10_seg` |
| Crop/pad + z-score | dedicated tests (centring, zero-mean/unit-SD, background) | All pass |

## 3. Consistency of code, artifacts and documentation

- **Per-subject DSC.** The hardcoded array in the original analysis cell
  `[0.9530, 0.9507, 0.2544, 0.9692, 0.4476, 0.9633, 0.9667, 0.9526, 0.9642,
  0.9655]` (order: case1, case10, case2…case9) yields mean 0.839 over N = 10 and
  0.961 over the 8 canonical-anatomy subjects (excluding cases 2 and 4). These
  match the values reported in the dissertation and in `README.md`.
- **Artifacts.** `models/` contains 10 `best_model_caseN.h5` weights and
  `data/preprocessed/` contains 10 `caseN.npz`, consistent with N = 10 LOSO.
- **Metrics implementation.** `evaluate.dice_score` reproduces the volumetric
  Dice used in the original evaluation cell; `centroid_error` implements the
  centre-of-mass targeting error reported in the dissertation.

## 4. Issues found and corrections

- **Image/mask mispairing (corrected).** Substring matching paired `case1` with
  `case10_seg`. Replaced with exact case-id matching and covered by a test.
- **Non-reproducible runs (corrected).** Seeds are now set centrally.
- **Environment coupling (corrected).** Colab/Drive code and `!pip` magics
  removed from the runnable package.

## 5. Remaining concerns

1. **LOSO validation leakage (scientific, high).** In `train.py` the held-out
   subject is also passed as `validation_data`, and both `ModelCheckpoint`
   (best `val_dice_coef`) and `EarlyStopping` (`restore_best_weights`) select
   the model that best fits that subject. This lets model selection observe the
   test subject and is expected to bias reported per-fold DSC optimistically.
   The behaviour is preserved to reproduce the dissertation; for a journal
   version, reserve a held-out *training* subject (or k-fold inner split) for
   validation and re-report metrics.
2. **Simulated data in a results figure (scientific, high).** The volume
   correlation figure generates ground-truth volumes with
   `np.random.randint(3000, 4200)` rather than measuring them from the masks.
   Recompute volumes from the actual masks before presenting that figure as an
   empirical result.
3. **TensorFlow execution not validated here.** Re-run `python -m src.evaluate`
   on a TensorFlow environment to confirm the released weights reproduce the
   tabulated DSC and centroid error end-to-end.
4. **Exact training versions.** The original Colab TensorFlow/CUDA versions are
   not fully pinned; confirm to guarantee weight-loading and numerical parity.
