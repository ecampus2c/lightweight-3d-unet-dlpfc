"""Leave-One-Subject-Out (LOSO) training driver, PyTorch version.

Mirrors ``src/train.py``: same data, sampling, augmentation, loss, optimiser and
schedule, with the Keras callbacks written out explicitly:

* checkpoint on best validation Dice  -> ``models/pytorch/best_model_<id>.pt``
* ReduceLROnPlateau(val_loss, factor 0.5, patience 8)
* EarlyStopping(val_dice, patience 20, restore best weights)

As in the Keras driver, the held-out subject drives checkpointing and early
stopping (see VALIDATION_REPORT.md, "LOSO validation leakage"); this is kept
for faithful reproduction.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path

import numpy as np
import torch

from . import config
from .data import load_preprocessed, ram_balanced_generator
from .torch_model import BCEDiceLoss, build_lightweight_unet, dice_coef

LR_FACTOR = 0.5
LR_PATIENCE = 8
EARLY_STOP_PATIENCE = 20


def _to_tensors(batch, device):
    """Channels-last NumPy batch from ``ram_balanced_generator`` -> channels-first tensors."""
    x, y = batch
    x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).permute(0, 4, 1, 2, 3)
    y = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32)).permute(0, 4, 1, 2, 3)
    return x.to(device), y.to(device)


def run_epoch(model, batches, loss_fn, device, optimizer=None):
    """Train (if ``optimizer``) or evaluate over ``batches``; returns mean loss and mean Dice."""
    training = optimizer is not None
    model.train(training)
    losses, dices = [], []
    with torch.set_grad_enabled(training):
        for batch in batches:
            x, y = _to_tensors(batch, device)
            logits = model(x)
            loss = loss_fn(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            losses.append(loss.item())
            dices.append(dice_coef(y, torch.sigmoid(logits)).item())
    return float(np.mean(losses)), float(np.mean(dices))


def train_loso(preproc_dir=config.PREPROC_DIR, models_dir=Path(config.MODELS_DIR) / "pytorch",
               epochs=config.EPOCHS, steps_per_epoch=config.STEPS_PER_EPOCH,
               validation_steps=config.VALIDATION_STEPS, seed=config.SEED, device=None):
    config.set_global_seeds(seed)
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    all_data = load_preprocessed(preproc_dir)
    if not all_data:
        raise FileNotFoundError(
            f"No preprocessed .npz files in {preproc_dir}. Run preprocessing first."
        )
    print(f"Loaded {len(all_data)} subjects for LOSO training on {device}.")

    for fold in range(len(all_data)):
        test_patient = all_data[fold]
        train_patients = [p for i, p in enumerate(all_data) if i != fold]
        subject = os.path.basename(test_patient["filename"]).split(".")[0]
        print(f"========== FOLD {fold + 1}/{len(all_data)} | held-out: {subject} ==========")

        model = build_lightweight_unet().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE,
            threshold=1e-4, threshold_mode="abs")
        loss_fn = BCEDiceLoss()

        train_gen = ram_balanced_generator(train_patients, fg_ratio=config.FOREGROUND_RATIO)
        val_gen = ram_balanced_generator([test_patient], fg_ratio=0.2)

        save_path = models_dir / f"best_model_{subject}.pt"
        best_dice, since_best, history = -1.0, 0, []
        for epoch in range(1, epochs + 1):
            loss, dice = run_epoch(model, itertools.islice(train_gen, steps_per_epoch),
                                   loss_fn, device, optimizer)
            val_loss, val_dice = run_epoch(model, itertools.islice(val_gen, validation_steps),
                                           loss_fn, device)
            scheduler.step(val_loss)
            history.append({"epoch": epoch, "loss": loss, "dice_coef": dice,
                            "val_loss": val_loss, "val_dice_coef": val_dice,
                            "lr": optimizer.param_groups[0]["lr"]})
            print(f"Epoch {epoch}/{epochs} - loss: {loss:.4f} - dice_coef: {dice:.4f} "
                  f"- val_loss: {val_loss:.4f} - val_dice_coef: {val_dice:.4f}")

            if val_dice > best_dice:
                print(f"val_dice_coef improved from {best_dice:.5f} to {val_dice:.5f}, saving to {save_path}")
                best_dice, since_best = val_dice, 0
                torch.save(model.state_dict(), save_path)
            else:
                since_best += 1
                if since_best >= EARLY_STOP_PATIENCE:
                    print(f"Early stopping: no val_dice_coef improvement for {EARLY_STOP_PATIENCE} epochs.")
                    break

        with open(models_dir / f"history_{subject}.json", "w") as f:
            json.dump(history, f, indent=1)
        print(f"Fold {fold + 1} complete. Best model -> {save_path} (val Dice {best_dice:.4f})")


def main():
    parser = argparse.ArgumentParser(description="LOSO training for DLPFC 3D U-Net (PyTorch).")
    parser.add_argument("--preproc-dir", default=config.PREPROC_DIR)
    parser.add_argument("--models-dir", default=Path(config.MODELS_DIR) / "pytorch")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--steps-per-epoch", type=int, default=config.STEPS_PER_EPOCH)
    parser.add_argument("--validation-steps", type=int, default=config.VALIDATION_STEPS)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--device", default=None, help="cuda or cpu (default: auto)")
    args = parser.parse_args()
    train_loso(args.preproc_dir, args.models_dir, args.epochs, args.steps_per_epoch,
               args.validation_steps, args.seed, args.device)


if __name__ == "__main__":
    main()
