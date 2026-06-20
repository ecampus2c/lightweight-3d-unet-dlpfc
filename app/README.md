# Web demo

A Gradio interface (`app/app.py`) for interactive DLPFC segmentation: upload a
T1-weighted MRI (`.nii`/`.nii.gz`) and receive an overlay, the target centroid,
and a downloadable predicted mask.

> **Research and educational use only.** Prototype trained on a 10-subject
> cohort; not a medical device; not for clinical use. The downloadable mask is
> in the preprocessed 1 mm isotropic 128³ space, not the original scanner space.

## Run locally

```bash
pip install -r app/requirements.txt
git lfs pull                 # fetch the model weights (models/*.h5)
python app/app.py
```

Open the printed local URL (default http://127.0.0.1:7860). First inference is
slower while the model loads; subsequent runs reuse the cached model. By default
a single LOSO model is used (fast); tick "Ensemble all models" for a more robust
average across all available weights.

Configuration (environment variables):

- `DLPFC_MODEL` — path to a specific `.h5` weight file (overrides the default).
- `DLPFC_MODELS` — directory of weights (defaults to `models/`).

### Temporary public link

To share a short-lived public URL without deploying, change the last line of
`app/app.py` to `build_demo().launch(share=True)` and re-run.

## Deploy to Hugging Face Spaces

1. Create a new **Gradio** Space at https://huggingface.co/new-space.
2. Push this repository to the Space (the model weights are tracked with Git LFS;
   ensure LFS objects are uploaded). 
3. Add a front-matter header to the Space's `README.md` so it runs this app:

   ```yaml
   ---
   title: DLPFC Segmentation
   sdk: gradio
   app_file: app/app.py
   ---
   ```

4. Set the Space's `requirements.txt` to `app/requirements.txt` (or copy its
   contents). CPU hardware is sufficient; a GPU Space reduces inference latency.

Note: hosting an inference service trained on patient data publicly carries
ethical and governance responsibilities. Keep the research-use disclaimer
visible and confirm that sharing the model weights is permitted.
