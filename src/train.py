"""Leave-One-Subject-Out (LOSO) training driver.

For each of the N subjects, one subject is held out and a fresh model is
trained on the remaining N-1, then checkpointed to ``models/best_model_<id>.h5``.

Reproducibility note: the held-out subject is also used as the Keras
``validation_data`` that drives checkpointing and early stopping. This mirrors
the original experiments but is an optimistic-bias risk (see VALIDATION_REPORT.md,
"LOSO validation leakage"); it is preserved here for faithful reproduction.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import tensorflow as tf

from . import config
from .data import load_preprocessed, ram_balanced_generator
from .model import bce_dice_loss, build_lightweight_unet, dice_coef


def train_loso(preproc_dir=config.PREPROC_DIR, models_dir=config.MODELS_DIR,
               epochs=config.EPOCHS, steps_per_epoch=config.STEPS_PER_EPOCH,
               validation_steps=config.VALIDATION_STEPS, seed=config.SEED):
    config.set_global_seeds(seed)
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    all_data = load_preprocessed(preproc_dir)
    if not all_data:
        raise FileNotFoundError(
            f"No preprocessed .npz files in {preproc_dir}. Run preprocessing first."
        )
    print(f"Loaded {len(all_data)} subjects for LOSO training.")

    for fold in range(len(all_data)):
        test_patient = all_data[fold]
        train_patients = [p for i, p in enumerate(all_data) if i != fold]
        subject = os.path.basename(test_patient["filename"]).split(".")[0]
        print(f"========== FOLD {fold + 1}/{len(all_data)} | held-out: {subject} ==========")

        model = build_lightweight_unet()
        model.compile(optimizer=tf.keras.optimizers.Adam(config.LEARNING_RATE),
                      loss=bce_dice_loss, metrics=[dice_coef])

        train_gen = ram_balanced_generator(train_patients, fg_ratio=config.FOREGROUND_RATIO)
        val_gen = ram_balanced_generator([test_patient], fg_ratio=0.2)

        save_path = str(models_dir / f"best_model_{subject}.h5")
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(save_path, monitor="val_dice_coef",
                                               mode="max", save_best_only=True, verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                                 patience=8, verbose=1),
            tf.keras.callbacks.EarlyStopping(monitor="val_dice_coef", mode="max",
                                             patience=20, restore_best_weights=True, verbose=1),
        ]

        model.fit(train_gen, steps_per_epoch=steps_per_epoch, epochs=epochs,
                  validation_data=val_gen, validation_steps=validation_steps,
                  callbacks=callbacks, verbose=1)
        print(f"Fold {fold + 1} complete. Best model -> {save_path}")
        tf.keras.backend.clear_session()


def main():
    parser = argparse.ArgumentParser(description="LOSO training for DLPFC 3D U-Net.")
    parser.add_argument("--preproc-dir", default=config.PREPROC_DIR)
    parser.add_argument("--models-dir", default=config.MODELS_DIR)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--steps-per-epoch", type=int, default=config.STEPS_PER_EPOCH)
    parser.add_argument("--validation-steps", type=int, default=config.VALIDATION_STEPS)
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()
    train_loso(args.preproc_dir, args.models_dir, args.epochs,
               args.steps_per_epoch, args.validation_steps, args.seed)


if __name__ == "__main__":
    main()
