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
        return None, "⚠️ Please **upload** a `.nii` or `.nii.gz` T1 volume first.", None
    try:
        volume, _affine = preprocess_image_file(nii_path)
        models = _load_models(bool(ensemble))
        prob = _infer(models, volume)
        mask = (prob > config.PROB_THRESHOLD).astype(np.uint8)

        overlay = _overlay_rgb(volume, mask)
        cm = center_of_mass(mask)
        used = ", ".join(name for name, _ in models)
        if cm is None:
            info = ("### ⚠️ No DLPFC predicted\n"
                    "The model returned an empty mask. Try the **Ensemble** option, "
                    "or check the input volume's orientation and contrast.")
            return overlay, info, None
        info = (
            "### ✅ Segmentation complete\n"
            f"**Target centroid** &nbsp;(voxel, 1 mm grid)&nbsp; "
            f"`z {cm[0]:.1f}` · `y {cm[1]:.1f}` · `x {cm[2]:.1f}`\n\n"
            f"**Predicted volume** &nbsp; {int(mask.sum()):,} mm³\n\n"
            f"**Model(s)** &nbsp; {used}"
        )
        return overlay, info, _save_mask(mask)
    except Exception as exc:  # surface a readable message in the UI
        return None, f"### ❌ Error\n```\n{exc}\n```", None


THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
)

CSS = """
.gradio-container {max-width: 1024px !important; margin: auto !important;}
#hero {text-align:center; padding: 30px 22px; border-radius: 20px; margin-bottom: 4px;
  background: linear-gradient(135deg,#4f46e5 0%,#7c3aed 55%,#9333ea 100%);
  box-shadow: 0 12px 34px rgba(99,102,241,.30);}
#hero h1 {color:#fff; font-size: 2.15rem; font-weight: 800; margin: 0; letter-spacing:-.01em;}
#hero p {color:#e0e7ff; margin: 10px 0 0; font-size: 1.04rem;}
#disclaimer {font-size:.84rem; line-height:1.45; color:#92400e; background:#fffbeb;
  border:1px solid #fcd34d; border-radius:12px; padding:10px 14px; margin:10px 0;}
#go {font-weight:700;}
#steps {font-size:.86rem; color:var(--body-text-color-subdued); margin-top:8px;}
#foot {text-align:center; color:var(--body-text-color-subdued); font-size:.84rem;
  margin-top:14px; padding-top:10px; border-top:1px solid var(--border-color-primary);}
#foot a {color:var(--primary-500); text-decoration:none; font-weight:600;}
"""

HERO = """
<div id="hero">
  <h1>🧠 DLPFC Segmentation</h1>
  <p>Upload a T1-weighted brain MRI and localize the dorsolateral prefrontal cortex for rTMS targeting</p>
</div>
"""

DISCLAIMER = (
    "<div id='disclaimer'>⚠️ <b>Research and educational use only.</b> A prototype trained on a "
    "10-subject cohort — not a medical device, and not for diagnosis or treatment planning. "
    "The downloadable mask is in the preprocessed 1&nbsp;mm isotropic space, not the original "
    "scanner space.</div>"
)

FOOTER = (
    "<div id='foot'>Lightweight 3D U-Net &nbsp;·&nbsp; leave-one-subject-out validated (N=10) "
    "&nbsp;·&nbsp; <a href='https://github.com/ecampus2c/lightweight-3d-unet-dlpfc' "
    "target='_blank'>source on GitHub</a></div>"
)


def build_demo():
    with gr.Blocks(title="DLPFC Segmentation") as demo:
        gr.HTML(HERO)
        gr.HTML(DISCLAIMER)
        with gr.Row():
            with gr.Column(scale=5):
                with gr.Group():
                    inp = gr.File(label="T1 MRI  (.nii / .nii.gz)",
                                  file_types=[".nii", ".gz"], type="filepath")
                    ens = gr.Checkbox(label="Ensemble all models  (slower, more robust)",
                                      value=False)
                    btn = gr.Button("🔬  Segment", variant="primary", size="lg", elem_id="go")
                gr.HTML("<div id='steps'><b>Pipeline:</b> resample to 1&nbsp;mm isotropic → "
                        "128³ crop/pad → z-score → 3D U-Net → sliding-window inference → "
                        "threshold at 0.5.</div>")
            with gr.Column(scale=6):
                out_img = gr.Image(label="Predicted DLPFC (overlay)", height=360)
                out_md = gr.Markdown("Upload a volume and press **Segment** — "
                                     "results will appear here.")
                out_file = gr.File(label="Predicted mask (.nii.gz)")
        gr.HTML(FOOTER)
        btn.click(segment, inputs=[inp, ens], outputs=[out_img, out_md, out_file])
    return demo


if __name__ == "__main__":
    build_demo().launch(theme=THEME, css=CSS)
