# Project Audit

Audit of the repository prior to public release. Severity reflects impact on
reproducibility and scientific correctness. "Fixed" changes are implemented in
`src/` and the repository layout; "Documented" issues are preserved to keep the
released artifacts faithful to the dissertation and are flagged for the author.

Original code referenced below is `notebooks/full_pipeline_colab.py` (the
preserved Colab export) unless stated otherwise.

| # | Issue | Severity | Files affected | Recommended fix | Status |
|---|-------|----------|----------------|-----------------|--------|
| 1 | Colab-only runtime: `from google.colab import drive`, `drive.mount` | High | original pipeline | Remove Colab coupling; load from local/config paths | Fixed (`src/`) |
| 2 | `!pip install` shell magics embedded in `.py` (not valid Python) | High | original pipeline | Move to `requirements.txt` / `environment.yml` | Fixed |
| 3 | Hardcoded paths (`/content/drive/MyDrive/DLPFC_Segmentation`) | High | original pipeline | Centralise in `src/config.py`; env-var overrides | Fixed |
| 4 | Conflicting dependency pins across 4 install cells (`numpy<2.0` vs `numpy==1.26.0`; `simpleitk` vs `==2.3.1`) | Medium | original pipeline | Single, consistent manifest | Fixed |
| 5 | Mask matching by substring (`"case1" in "case10_seg.nii"`) mispairs image/mask | High | preprocessing loop | Exact case-id matching (`_label_for_image`) + test | Fixed |
| 6 | No random seeds set (weights, patch sampling, augmentation) | High | training, generator | `config.set_global_seeds`; seeded `train_loso` | Fixed |
| 7 | LOSO checkpoint/early-stopping monitor the **held-out test** subject | High | training loop | Use a held-out *training* subject for validation | Documented (see VALIDATION_REPORT.md) |
| 8 | Correlation figure uses **simulated** GT volumes (`np.random.randint(3000,4200)`) | High | `generate_correlation_plot` cell | Compute real per-mask volumes | Documented |
| 9 | Monolithic 2151-line script; repeated imports, duplicate function defs | Medium | original pipeline | Modular `src/` package; original kept as reference | Fixed |
| 10 | Loose figures and documents at repository root; no package layout | Low | repo root | Reorganise into `src/data/results/...` | Fixed (see STRUCTURAL_CHANGES.md) |
| 11 | No dependency manifest | High | repo | `requirements.txt`, `environment.yml` | Fixed |
| 12 | No README or methods/dataset/reproducibility documentation | High | repo | Documentation set added | Fixed |
| 13 | No automated tests | Medium | repo | `tests/` (preprocessing, metrics, data, model) | Fixed |
| 14 | No `.gitignore`; ~916 MB of binaries would exceed GitHub limits | High | repo | `.gitignore` large artifacts; Git LFS / external host | Fixed (LFS pending) |
| 15 | TensorFlow training version not pinned; legacy `.h5` weights | Medium | `models/`, manifests | Pin TF; confirm exact training version | Partial (author to confirm) |
| 16 | No license | Medium | repo | Add `LICENSE` defining code/model/figure reuse | Open |
| 17 | Dataset provenance, acquisition and ethics undocumented | High | `data/` | `DATASET.md` with provenance/IRB fields | Partial (author to confirm) |
| 18 | `np.random.randint(0, dim-patch)` fails if a volume equals patch size | Low | `data.py` generator | Guard for `dim == patch`; not triggered at 128³ | Open (low) |

## Summary

- High-severity reproducibility blockers (1–6, 11–12, 14) are resolved: the
  pipeline now runs outside Colab from a clean, documented, dependency-pinned
  package, with the data-pairing bug fixed and stochastic steps seeded.
- Two high-severity *scientific* findings (7, 8) are intentionally **not**
  silently changed, because doing so would alter the reported results. They are
  documented with recommended corrections for any future journal version.
- Remaining open items (15–17) require author input (exact TF version, license,
  dataset provenance/ethics) and are tracked in
  [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md).
