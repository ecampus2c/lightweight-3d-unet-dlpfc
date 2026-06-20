# Pipeline

End-to-end workflow, mapped to the modules in `src/`.

```
            data/raw/images/caseN.nii   data/raw/labels/caseN_seg.nii
                          │                         │
                          ▼                         ▼
        ┌─────────────────────────────────────────────────────┐
        │ src/preprocessing.py                                 │
        │  • resample to 1 mm isotropic (B-spline / NN)        │
        │  • centre crop/pad to 128³                           │
        │  • z-score normalize (brain voxels)                  │
        └─────────────────────────────────────────────────────┘
                          │
                          ▼
              data/preprocessed/caseN.npz   (image, label, affine)
                          │
        ┌─────────────────┴───────────────────┐
        ▼                                      ▼
┌──────────────────────────┐      ┌──────────────────────────────┐
│ src/train.py (LOSO)      │      │ src/data.py                  │
│  for each held-out k:    │◀─────│  load_preprocessed()         │
│   build model            │      │  ram_balanced_generator()    │
│   fit on N−1 subjects    │      │   96³ patches, fg_ratio 0.6, │
│   (src/model.py:         │      │   flips + rot90 augmentation │
│    U-Net + BCE/Dice)     │      └──────────────────────────────┘
│   checkpoint best        │
└──────────────────────────┘
                          │
                          ▼
                 models/best_model_caseN.h5
                          │
                          ▼
        ┌─────────────────────────────────────────────────────┐
        │ src/inference.py                                     │
        │  sliding_window_inference (96³, stride 48,           │
        │  dynamic padding) → probability map → threshold 0.5  │
        └─────────────────────────────────────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────────────────────────┐
        │ src/evaluate.py                                      │
        │  dice_score  +  centroid_error (mm)                  │
        │  evaluate_loso() → summarize()                       │
        └─────────────────────────────────────────────────────┘
                          │
                          ▼
        per-subject DSC / centroid error  ·  results/figures/*.png
```

## Stage summary

| Stage | Module | Input | Output |
|-------|--------|-------|--------|
| Preprocess | `preprocessing.py` | `data/raw/*` | `data/preprocessed/*.npz` |
| Train (LOSO) | `train.py`, `model.py`, `data.py` | `*.npz` | `models/*.h5` |
| Infer | `inference.py` | volume + model | probability map / mask |
| Evaluate | `evaluate.py` | mask + ground truth | DSC, centroid error |

The original single-file Colab implementation that these modules were derived
from is preserved at `notebooks/full_pipeline_colab.py`.
