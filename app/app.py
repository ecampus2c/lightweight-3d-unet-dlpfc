"""Web demo: upload a T1 MRI volume and obtain a DLPFC segmentation.

Runs the published pipeline (preprocess -> lightweight 3D U-Net -> sliding-window
inference) behind a Gradio interface. Launch locally with:

    python app/app.py

then open the printed URL. TensorFlow, SimpleITK and nibabel must be installed
(see app/requirements.txt) and at least one weight file must exist in models/.

Research/educational use only. This is a prototype trained on a 10-subject
cohort; it is not a medical device and must not be used for clinical decisions.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

# Make the repository root importable when run as `python app/app.py`.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src import config                                   # noqa: E402
from src.preprocessing import preprocess_image_file      # noqa: E402
from src.evaluate import center_of_mass                  # noqa: E402

_MODEL_CACHE: dict = {}


def _model_paths():
    env = os.environ.get("DLPFC_MODEL")
    if env:
        return [Path(env)]
    return sorted(config.MODELS_DIR.glob("best_model_*.h5"))


def _load_models(ensemble: bool):
    """Load and cache model(s); returns a list of (name, model). TF imported lazily."""
    key = "ensemble" if ensemble else "single"
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    from src.model import build_lightweight_unet

    paths = _model_paths()
    if not paths:
        raise FileNotFoundError(
            f"No model weights in {config.MODELS_DIR}. Fetch the Git LFS weights "
            "(`git lfs pull`) or train with `python -m src.train`."
        )
    if not ensemble:
        paths = paths[:1]

    models = []
    for p in paths:
        net = build_lightweight_unet()
        net.load_weights(str(p))
        models.append((p.stem, net))
    _MODEL_CACHE[key] = models
    return models


def _infer(models, volume):
    from src.inference import sliding_window_inference
    probs = [sliding_window_inference(net, volume)[..., 0] for _, net in models]
    return np.mean(probs, axis=0)


def _overlay_rgb(volume, mask):
    """Render the axial slice with the most predicted foreground as an RGB image."""
    fg_per_slice = mask.sum(axis=(1, 2))
    best = int(np.argmax(fg_per_slice)) if fg_per_slice.max() > 0 else volume.shape[0] // 2
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(np.rot90(volume[best]), cmap="gray")
    m = np.rot90(mask[best])
    overlay = np.zeros((*m.shape, 4))
    overlay[m > 0] = [1, 0, 0, 0.45]
    ax.imshow(overlay)
    ax.set_title(f"Predicted DLPFC - axial slice {best}")
    ax.axis("off")
    fig.tight_layout()
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    plt.close(fig)
    return rgba[..., :3].copy()


def _save_mask(mask):
    import nibabel as nib
    out = Path(tempfile.gettempdir()) / "dlpfc_predicted_mask.nii.gz"
    # Saved in the preprocessed 1 mm isotropic 128^3 space (identity affine).
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), np.eye(4)), str(out))
    return str(out)


def segment(nii_path, ensemble):
    if not nii_path:
        return None, "Please upload a .nii or .nii.gz T1 volume.", None
    try:
        volume, _affine = preprocess_image_file(nii_path)
        models = _load_models(bool(ensemble))
        prob = _infer(models, volume)
        mask = (prob > config.PROB_THRESHOLD).astype(np.uint8)

        overlay = _overlay_rgb(volume, mask)
        cm = center_of_mass(mask)
        used = ", ".join(name for name, _ in models)
        if cm is None:
            info = ("No DLPFC predicted (empty mask). Try enabling the ensemble "
                    "option, or check the input volume orientation/contrast.")
        else:
            info = (
                "Target centroid (voxel index in the 1 mm isotropic 128^3 grid): "
                f"z={cm[0]:.1f}, y={cm[1]:.1f}, x={cm[2]:.1f}\n"
                f"Predicted volume: {int(mask.sum()):,} mm^3\n"
                f"Model(s): {used}"
            )
        return overlay, info, _save_mask(mask)
    except Exception as exc:  # surface a readable message in the UI
        return None, f"Error: {exc}", None


DESCRIPTION = """
# DLPFC Segmentation from T1 MRI

Upload a T1-weighted brain MRI (`.nii` or `.nii.gz`). The volume is resampled to
1 mm isotropic, cropped/padded to 128x128x128 and z-score normalised, then a
lightweight 3D U-Net predicts the dorsolateral prefrontal cortex (DLPFC). You
get an overlay, the target centroid, and the mask as a downloadable NIfTI.

**Research and educational use only.** This is a prototype trained on a
10-subject cohort. It is not a medical device and must not be used for diagnosis
or treatment planning. The downloadable mask is in the preprocessed 1 mm
isotropic space, not the original scanner space.
"""


def build_demo():
    with gr.Blocks(title="DLPFC Segmentation") as demo:
        gr.Markdown(DESCRIPTION)
        with gr.Row():
            with gr.Column():
                inp = gr.File(label="T1 MRI (.nii / .nii.gz)",
                              file_types=[".nii", ".gz"], type="filepath")
                ens = gr.Checkbox(label="Ensemble all models (slower, more robust)",
                                  value=False)
                btn = gr.Button("Segment", variant="primary")
            with gr.Column():
                out_img = gr.Image(label="Predicted segmentation (overlay)")
                out_txt = gr.Textbox(label="Target details", lines=4)
                out_file = gr.File(label="Download predicted mask (.nii.gz)")
        btn.click(segment, inputs=[inp, ens], outputs=[out_img, out_txt, out_file])
    return demo


if __name__ == "__main__":
    build_demo().launch()
