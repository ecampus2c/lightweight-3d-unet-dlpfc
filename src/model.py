"""Lightweight 3D U-Net architecture and the composite BCE+Dice loss.

The architecture is intentionally small (three encoder levels, base width 16)
to limit the parameter count under the small-data regime described in the
dissertation. Kept identical to the original implementation.
"""
from __future__ import annotations

import tensorflow as tf

from . import config


def build_lightweight_unet(input_shape=(96, 96, 96, 1)) -> tf.keras.Model:
    """3D U-Net with encoder widths (16, 32, 64) and a 128-filter bottleneck."""
    f1, f2, f3 = config.ENCODER_FILTERS
    fb = config.BOTTLENECK_FILTERS

    def conv_block(x, filters):
        x = tf.keras.layers.Conv3D(filters, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.Conv3D(filters, 3, padding="same", activation="relu")(x)
        return x

    def encoder_block(x, filters):
        c = conv_block(x, filters)
        p = tf.keras.layers.MaxPooling3D((2, 2, 2))(c)
        return c, p

    def decoder_block(x, skip, filters):
        x = tf.keras.layers.Conv3DTranspose(filters, 2, strides=2, padding="same")(x)
        x = tf.keras.layers.Concatenate()([x, skip])
        x = conv_block(x, filters)
        return x

    inputs = tf.keras.layers.Input(shape=input_shape)

    c1, p1 = encoder_block(inputs, f1)
    c2, p2 = encoder_block(p1, f2)
    c3, p3 = encoder_block(p2, f3)

    b = conv_block(p3, fb)

    d3 = decoder_block(b, c3, f3)
    d2 = decoder_block(d3, c2, f2)
    d1 = decoder_block(d2, c1, f1)

    outputs = tf.keras.layers.Conv3D(1, 1, activation="sigmoid")(d1)
    return tf.keras.Model(inputs, outputs, name="Lightweight_3D_UNet")


def dice_coef(y_true, y_pred, smooth: float = 1e-6):
    """Soft Dice coefficient. y_true is cast to float32 to match y_pred."""
    y_true = tf.cast(y_true, tf.float32)
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth
    )


def bce_dice_loss(y_true, y_pred):
    """Composite loss: binary cross-entropy plus (1 - soft Dice)."""
    y_true = tf.cast(y_true, tf.float32)
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    return bce + (1.0 - dice_coef(y_true, y_pred))
